from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(os.path.dirname(_HERE), "src")]

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit(
        "train_pathb.py needs torch.\n"
        f"  import failed: {exc}\n"
        "Everything else in this repo runs on numpy/scipy alone.")

from data_a import ModelADataset
from lens_models import deflection
from raytrace import area_downsample, image_plane_grid, load_psf
from sources import SersicSource
from theta_e_init import initial_guess

PARAMS = ("theta_E", "gamma", "e1", "e2", "g1", "g2",
          "amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy", "background")
LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "cx", "cy")
SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


##########
# network
##########

class PathBNet(nn.Module):

    def __init__(self, n_c: int = 24, width: int = 32, n_par: int = 14,
                 in_ch: int = 5, n_ring: int = 4):
        super().__init__()
        c = width
        self.n_c = n_c
        self.n_ring = n_ring
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, c, 5, stride=2, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU(),
            nn.Conv2d(4 * c, 4 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU(),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        feat = 4 * c + n_ring

        #head 1: the 14 physical parameters
        self.par_head = nn.Sequential(nn.Linear(feat, 128), nn.SiLU(),
                                      nn.Linear(128, n_par))

        # head 2: the source-plane correction map
        # A decoder rather than one dense layer: the correction is a spatial
        # field, so transposed convolutions are the better prior.
        # base = n_c // 4 makes the two stride-2 layers land on exactly n_c, so
        # nothing is resized and the degrees of freedom stay n_c^2.
        self.base = max(2, n_c // 4)
        self.dec_fc = nn.Linear(feat, 64 * self.base * self.base)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # small random init, not zeros: zero-init can leave the head at its
        # neutral output for the whole run
        for m in (self.par_head[-1], self.dec[-1]):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # the ring scalars are constant over the plane, so [:, 1:, 0, 0] reads
        # them back exactly; concatenating them re-injects them past the trunk
        ring = x[:, 1:1 + self.n_ring, 0, 0]
        h = torch.cat([self.pool(self.trunk(x)), ring], dim=1)
        z = self.par_head(h)
        c = self.dec(self.dec_fc(h).view(-1, 64, self.base, self.base))
        if c.shape[-1] != self.n_c:
            c = F.interpolate(c, size=(self.n_c, self.n_c), mode="bilinear",
                              align_corners=False)
        return z, c[:, 0]


def to_params(z, ring):
    s = lambda i: z[:, i]
    tE, m1 = ring[:, 0], ring[:, 1]
    # The source offset is scaled by the ring dipole, not set equal to it.
    # Setting the magnitude directly needs atan2, which is singular at the
    # origin where a zero-initialised head starts; test_pathb.py check 5 goes
    # NaN in that form. Scaling has no singularity and still lets beta move by
    # up to a factor ~2.8 either way.
    b_scale = 2.0 * m1.clamp_min(1e-3)
    return {
        "theta_E": tE * torch.exp(0.3 * torch.tanh(s(0))),
        "gamma": 2.0 + 0.5 * torch.tanh(s(1)),
        "e1": 0.6 * torch.tanh(s(2)),
        "e2": 0.6 * torch.tanh(s(3)),
        "g1": 0.3 * torch.tanh(s(4)),
        "g2": 0.3 * torch.tanh(s(5)),
        "amp": F.softplus(s(6) + 1.0),
        "R_sersic": 0.05 + 2.0 * torch.sigmoid(s(7) - 1.0),
        "n_sersic": 0.3 + 5.0 * torch.sigmoid(s(8) - 1.0),
        "se1": 0.6 * torch.tanh(s(9)),
        "se2": 0.6 * torch.tanh(s(10)),
        "sx": b_scale * torch.tanh(s(11)),
        "sy": b_scale * torch.tanh(s(12)),
        # bounded: left free as 0.1*z this drifts to a median of -0.98
        # against Path A's +0.018, i.e. a negative sky pedestal offsetting
        # an over-bright source
        "background": torch.tanh(s(13)),
        "cx": torch.zeros_like(s(0)),
        "cy": torch.zeros_like(s(0)),
    }


####################
# forward model
####################

def sample_correction(C, bx, by, half_extent):
    n = C.shape[-1]
    edge = half_extent * (1.0 + 1.0 / (n - 1))
    grid = torch.stack([bx / edge, by / edge], dim=-1)        # (B,H,W,2)
    return F.grid_sample(C.unsqueeze(1), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)[:, 0]


def ray_coverage(bx, by, n_c, half_extent, smooth: int = 3):
    B = bx.shape[0]
    scale = 2.0 * half_extent / (n_c - 1)
    with torch.no_grad():
        fj = (bx.detach() + half_extent) / scale
        fi = (by.detach() + half_extent) / scale
        inside = ((fj >= -0.5) & (fj <= n_c - 0.5) &
                  (fi >= -0.5) & (fi <= n_c - 0.5))
        j = torch.round(fj).long().clamp(0, n_c - 1)
        i = torch.round(fi).long().clamp(0, n_c - 1)
        flat = (i * n_c + j).reshape(B, -1)
        cov = torch.zeros(B, n_c * n_c, device=bx.device, dtype=bx.dtype)
        cov.scatter_add_(1, flat, inside.reshape(B, -1).to(bx.dtype))
        cov = cov.view(B, 1, n_c, n_c)

        # The histogram is a Monte-Carlo estimate of mu. At supersample = 1
        # there are only ~1-2 rays per source pixel, so the raw count is
        # mostly shot noise. A 3x3 box average keeps the large-scale
        # structure and drops the aliasing. It adds no information;
        # --supersample 2 is the fix for genuinely low coverage, and the
        # startup banner reports the ray count.
        if smooth and smooth > 1:
            pad = smooth // 2
            cov = F.avg_pool2d(F.pad(cov, (pad,) * 4, mode="replicate"),
                               smooth, stride=1)
    return cov[:, 0]


def reg_weights(cov, mode: str, power: float = 0.5, clip: float = 5.0):
    if mode == "uniform":
        return torch.ones_like(cov)
    B = cov.shape[0]
    flat = cov.view(B, -1)
    pos = torch.where(flat > 0, flat, torch.full_like(flat, float("nan")))
    med = torch.nanmedian(pos, dim=1, keepdim=True).values
    med = torch.nan_to_num(med, nan=1.0).clamp_min(1e-6)
    w = torch.pow(med / flat.clamp_min(1e-3 * med), power)
    return w.clamp(1.0 / clip, clip).view_as(cov)


def render_batch(p, C, grid, psf, n_pix, supersample, half_extent,
                 corr_scale: float, source_mode: str = "b3"):
    X, Y = grid
    B = p["theta_E"].shape[0]
    b = lambda t: t.reshape(B, 1, 1)
    lens = {k: b(p[k]) for k in LENS_KEYS}

    ax, ay = deflection(X[None], Y[None], lens)
    bx, by = X[None] - ax, Y[None] - ay # ray shooting

    if source_mode == "b2":
        C_eff = b(p["amp"]) * F.softplus(C)
        sky = sample_correction(C_eff, bx, by, half_extent)
    else:
        src = {k: b(p[k]) for k in SRC_KEYS}
        sky = SersicSource(src).at(bx, by)
        C_eff = corr_scale * b(p["amp"]) * torch.tanh(C)
        sky = sky + sample_correction(C_eff, bx, by, half_extent)

    sky = sky.unsqueeze(1)
    pad = psf.shape[-1] // 2
    sky = F.conv2d(F.pad(sky, (pad, pad, pad, pad), mode="replicate"),
                   psf[None, None])
    pred = area_downsample(sky, supersample)[:, 0] + b(p["background"])
    return pred, C_eff, (bx, by)


def curvature(C):
    lap = (C[:, :-2, 1:-1] + C[:, 2:, 1:-1] +
           C[:, 1:-1, :-2] + C[:, 1:-1, 2:] - 4.0 * C[:, 1:-1, 1:-1])
    return (lap ** 2).mean()


RING_FEATURES = ("theta_E", "m1_abs", "m2_abs", "ring_completeness")


def ring_summary(img, pixel_scale):
    q = initial_guess(img, pixel_scale)
    return [q["theta_E"], float(np.hypot(q["sx"], q["sy"])),
            abs(q["m2_over_theta_E"]), q["ring_completeness"]]


def pack_split(ds_root, split, classes, n, target, pixel_scale):
    ds = ModelADataset(ds_root, split=split, classes=classes, limit=n)
    imgs, sigs, feat = [], [], []
    for i in range(len(ds)):
        im = ds.image(i) if target == "image" else ds.image_nss(i)
        imgs.append(im)
        sigs.append(ds.sigma(i, im))
        feat.append(ring_summary(im, pixel_scale))
    return (torch.as_tensor(np.array(imgs), dtype=torch.float32),
            torch.as_tensor(np.array(sigs), dtype=torch.float32),
            torch.as_tensor(np.array(feat), dtype=torch.float32))


def build_input(X, S, R, stretch: bool = True):
    x = X / S[:, None, None]
    if stretch:
        x = torch.asinh(x)
    ones = torch.ones_like(x)
    ch = [x] + [R[:, k, None, None] * ones for k in range(R.shape[1])]
    return torch.stack(ch, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--classes", nargs="+", default=["axion", "cdm", "wdm"])
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--target", default="image", choices=["image", "image_nss"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--psf-path", default="results/psf_empirical.npy")
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--n-c", type=int, default=32, help="correction map side")
    ap.add_argument("--half-extent", type=float, default=0.8,
                    help="source-plane half-width in arcsec. Together with --n-c "
                         "this sets the super-resolution factor; see the note "
                         "printed at startup.")
    ap.add_argument("--max-gain", type=float, default=3.1,
                    help="measured median tangential stretch from "
                         "magnification_extract.py. The source pixel should not "
                         "be finer than the detector pixel divided by this.")
    ap.add_argument("--corr-scale", type=float, default=0.30,
                    help="max correction as a fraction of the Sersic amplitude "
                         "(b3 only). This is a SAFETY BOUND, not an operating "
                         "point -- if the reported saturation is high, the "
                         "penalty is too weak, not the bound too tight.")
    ap.add_argument("--source-mode", default="b3", choices=["b3", "b2"],
                    help="b3 = Sersic + bounded correction; "
                         "b2 = free-form source only, no Sersic")
    ap.add_argument("--reg-mode", default="mu", choices=["mu", "uniform", "none"])
    ap.add_argument("--reg-power", type=float, default=0.5)
    ap.add_argument("--lambda-corr", type=float, default=3.0,
                    help="correction penalty, as a FRACTION OF chi^2 (the code "
                         "multiplies by chi2.detach(), so this is scale-free). "
                         "At full saturation the b3 penalty is "
                         "lambda_corr * corr_scale^2 = 3 * 0.09 = 27% of chi^2.")
    ap.add_argument("--lambda-curv", type=float, default=0.0,
                    help="Laplacian smoothness penalty on the source map, also "
                         "as a fraction of chi^2. Optional in b3; ESSENTIAL in "
                         "b2, where 3.0 is a sane starting value.")
    ap.add_argument("--sigma-floor", type=float, default=0.02,
                    help="fractional systematic error floor: "
                         "sigma_eff^2 = sigma_bg^2 + (f * model)^2. "
                         "THE SINGLE MOST IMPORTANT FLAG IN THIS FILE -- see the "
                         "note in the header. 0 reproduces the broken v1 loss.")
    ap.add_argument("--augment", type=int, default=1,
                    help="random 90-degree rotations and flips of the training "
                         "images. Free and exactly valid here, because the loss "
                         "is self-supervised: a rotated image is a legitimate "
                         "member of the data distribution and needs no label "
                         "transformation.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--stretch", type=int, default=1,
                    help="arcsinh-stretch the input image channel. The raw "
                         "S/N peak varies 260x between images, which a shared "
                         "filter bank cannot absorb; arcsinh cuts that to 1.9x.")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="epochs with the correction switched OFF, so the "
                         "parametric part converges first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/pathb.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    psf_np = load_psf(a.psf_path) if os.path.exists(a.psf_path) \
        else load_psf("results/psf_empirical.npy")
    psf = torch.as_tensor(psf_np, dtype=torch.float32, device=dev)

    print("loading data ...", flush=True)
    Xtr, Str, Rtr = pack_split(a.root, "train", a.classes, a.n_train,
                               a.target, a.pixel_scale)
    Xva, Sva, Rva = pack_split(a.root, "val", a.classes, a.n_val,
                               a.target, a.pixel_scale)
    n_pix = Xtr.shape[-1]
    gx, gy = image_plane_grid(n_pix, a.pixel_scale, a.supersample)
    grid = (torch.as_tensor(gx, dtype=torch.float32, device=dev),
            torch.as_tensor(gy, dtype=torch.float32, device=dev))
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask = torch.as_tensor(np.hypot(yy - c, xx - c) <= a.fit_radius_px, device=dev)

    net = PathBNet(n_c=a.n_c, in_ch=1 + len(RING_FEATURES)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr,
                           weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    n_par = sum(q.numel() for q in net.parameters())

    src_scale = 2.0 * a.half_extent / (a.n_c - 1)
    gain = a.pixel_scale / src_scale
    print(f"device {dev}   train {len(Xtr)}   val {len(Xva)}   weights {n_par:,}")
    print(f"correction {a.n_c}^2 = {a.n_c**2} px at {src_scale:.4f} arcsec/px "
          f"({gain:.2f}x finer than the detector)")
    print(f"unknowns per image: 14 + {a.n_c**2} = {14+a.n_c**2} "
          f"vs ~4300 informative data pixels "
          f"({4300/(14+a.n_c**2):.1f}:1 overdetermined)")
    # The super-resolution factor is bounded by the lens stretch factor, which
    # magnification_extract.py measures as the median tangential
    # 1/|1-kappa-|gamma||. A finer source grid than that is interpolation.
    print(f"super-resolution factor {gain:.2f}x against a measured median "
          f"tangential stretch of {a.max_gain:.2f}x")
    if gain > a.max_gain:
        print("  *** the requested source grid is finer than the lens can")
        print("  *** support. Reduce --n-c or raise --half-extent, otherwise the")
        print("  *** extra detail is not constrained by the data.")

    # Rays per source pixel. About a third of image-plane rays land inside a
    # 0.8 arcsec source map, and that fraction decides whether the
    # magnification weights carry signal.
    rays = (n_pix * a.supersample) ** 2
    per_px = 0.34 * rays / (a.n_c ** 2)
    print(f"rays per source pixel ~ {per_px:.1f} "
          f"({rays:,} rays, ~34% land inside the map, {a.n_c**2} source pixels)")
    if per_px < 4 and a.reg_mode == "mu":
        print("  *** Below ~4 rays/pixel the coverage histogram is mostly shot")
        print("  *** noise. ray_coverage() 3x3-smooths it, which helps, but")
        print("  *** --supersample 2 is the real fix (4x the rays, ~4x the cost).")
    print(f"source mode {a.source_mode}   reg {a.reg_mode}   "
          f"lambda_corr {a.lambda_corr} x chi2   lambda_curv {a.lambda_curv} x chi2")
    print(f"sigma floor {a.sigma_floor:.3f} (fractional systematic)   "
          f"augment {bool(a.augment)}   weight decay {a.weight_decay}   "
          f"warmup {a.warmup_epochs} epochs\n")
    if a.sigma_floor <= 0:
        print("  *** sigma-floor = 0 reproduces the earlier loss, where one")
        print("  *** image in a batch of 16 took 54% of the gradient.\n")

    n_skip = [0]
    hist = []

    def augment_batch(X):
        k = int(torch.randint(0, 4, (1,)))
        if k:
            X = torch.rot90(X, k, dims=(-2, -1))
        if int(torch.randint(0, 2, (1,))):
            X = torch.flip(X, dims=(-1,))
        return X.contiguous()

    def step(X, S, R, train, use_corr):
        X, S, R = X.to(dev), S.to(dev), R.to(dev)
        if train and a.augment:
            X = augment_batch(X) # ring features are rotation-invariant
        inp = build_input(X, S, R, stretch=bool(a.stretch))
        z, Craw = net(inp)
        p = to_params(z, R)# ring vector: tE, |m1|, |m2|, comp
        if not use_corr:
            Craw = torch.zeros_like(Craw)
        pred, C_eff, (bx, by) = render_batch(
            p, Craw, grid, psf, n_pix, a.supersample, a.half_extent,
            a.corr_scale, a.source_mode)

        # the error model
        # sigma_bg alone treats background noise as the only uncertainty.
        # calibrate_psf.py puts the PSF-wing error at ~1-3%, and that scales
        # with model flux. Without a fractional floor the brightest image in
        # a batch of 16 took 54% of the gradient. With it the per-image
        # chi^2-vs-SNR trend flattens from 1411 -> 8455 to 329 -> 229 and the
        # p90/p10 spread drops from 95x to 8.7x.
        sig = S[:, None, None]
        if a.sigma_floor > 0:
            sig = torch.sqrt(sig ** 2 +
                             (a.sigma_floor * pred.detach().clamp_min(0.0)) ** 2)
        r = ((pred - X) / sig)[:, mask]
        chi2 = (r ** 2).mean()

        reg = torch.zeros((), device=dev)
        curv = torch.zeros((), device=dev)
        sat = torch.zeros((), device=dev)
        if use_corr:
            amp = p["amp"].reshape(-1, 1, 1).clamp_min(1e-6)
            rel = C_eff / amp
            if a.reg_mode != "none":
                cov = ray_coverage(bx, by, a.n_c, a.half_extent)
                w = reg_weights(cov, a.reg_mode, a.reg_power)
                reg = (w * rel ** 2).mean()
            if a.lambda_curv > 0:
                curv = curvature(rel)
            if a.source_mode == "b3":
                # fraction of the map within 1% of the tanh bound. Large means
                # the penalty is too weak and the correction has stopped
                # receiving gradient; at 75% saturated, corr(|C|,log mu) = -0.03.
                sat = (rel.abs() > 0.99 * a.corr_scale).to(rel.dtype).mean()

        # lambda is a fraction of chi^2, multiplied by a detached chi^2, so it
        # is scale-free. An absolute lambda = 1.0 against chi^2 ~ 1e5 is six
        # orders of magnitude too small and makes the mu and uniform runs come
        # out identical (Pearson +0.99 on every parameter).
        scale = chi2.detach().clamp_min(1e-12)
        loss = chi2 + scale * (a.lambda_corr * reg + a.lambda_curv * curv)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad = [n for n, q in net.named_parameters()
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad or not torch.isfinite(loss):
                n_skip[0] += 1
                if n_skip[0] <= 5:
                    print(f"   [skip] non-finite loss/grad; loss={float(loss)}; "
                          f"first bad: {bad[:3]}", flush=True)
                opt.zero_grad(set_to_none=True)
                return float("nan"), float("nan")
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        return (float(chi2.detach()), float(reg.detach()), float(sat.detach()))

    best = float("inf")
    for ep in range(a.epochs):
        use_corr = ep >= a.warmup_epochs
        net.train()
        perm = torch.randperm(len(Xtr))
        tc, tr_, ts, nb = 0.0, 0.0, 0.0, 0
        for k in range(0, len(Xtr), a.batch_size):
            j = perm[k:k + a.batch_size]
            ch, rg, st = step(Xtr[j], Str[j], Rtr[j], True, use_corr)
            if np.isfinite(ch):
                tc += ch; tr_ += rg; ts += st; nb += 1
        net.eval()
        vc, vn = 0.0, 0
        for k in range(0, len(Xva), a.batch_size):
            sl = slice(k, k + a.batch_size)
            with torch.enable_grad():          # deflection needs grad-capable coords
                ch, _, _ = step(Xva[sl], Sva[sl], Rva[sl], False, use_corr)
            if np.isfinite(ch):
                vc += ch; vn += 1
        sched.step()
        row = {"epoch": ep + 1, "train_chi2": tc / max(nb, 1),
               "train_reg": tr_ / max(nb, 1), "val_chi2": vc / max(vn, 1),
               "saturation": ts / max(nb, 1),
               "correction_on": bool(use_corr), "lr": sched.get_last_lr()[0]}
        hist.append(row)
        tag = "  (correction ON)" if use_corr and ep == a.warmup_epochs else ""
        # train/val gap and saturation are printed every epoch: both are
        # needed to spot a failed run before it finishes
        print(f"epoch {ep+1:3d}  train {row['train_chi2']:10.2f}"
              f"  val {row['val_chi2']:10.2f}"
              f"  gap {row['val_chi2']/max(row['train_chi2'],1e-9):5.2f}x"
              f"  reg {row['train_reg']:8.4f}"
              f"  sat {100*row['saturation']:5.1f}%{tag}", flush=True)

        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        blob = {"model": net.state_dict(), "args": vars(a),
                "history": hist, "n_skipped": n_skip[0]}
        torch.save(blob, a.out)
        if row["val_chi2"] < best:            # keep the best-val checkpoint too
            best = row["val_chi2"]
            torch.save(blob, a.out.replace(".pt", "_best.pt"))

    print(f"\nwrote {a.out}   (skipped batches: {n_skip[0]})")
    print(f"best val chi2 {best:.2f} -> {a.out.replace('.pt', '_best.pt')}")
    fin = hist[-1]
    if a.source_mode == "b3" and fin["saturation"] > 0.25:
        print(f"\n  *** {100*fin['saturation']:.0f}% of the correction map is at the")
        print("  *** tanh bound. A saturated tanh has ~zero gradient, so the")
        print("  *** correction has stopped learning and the penalty can no")
        print("  *** longer shape it. Raise --lambda-corr (try 10) or lengthen")
        print("  *** --warmup-epochs so the parametric part converges first.")
    if fin["val_chi2"] > 2.0 * fin["train_chi2"]:
        print(f"\n  *** val/train = {fin['val_chi2']/fin['train_chi2']:.1f}x: overfitting.")
        print("  *** Raise --weight-decay, keep --augment 1, or use fewer epochs.")
    print("\nnow run:  python scripts/eval_pathb.py --ckpt "
          + a.out.replace(".pt", "_best.pt") + " --root .")


if __name__ == "__main__":
    main()