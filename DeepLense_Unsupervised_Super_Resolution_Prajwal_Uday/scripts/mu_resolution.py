from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(os.path.dirname(_HERE), "src")]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(f"mu_resolution.py needs torch (import failed: {exc})")

from backproject import backproject, ray_shoot_batch, sample_source
from data_a import ModelADataset
from raytrace import area_downsample, convolve, image_plane_grid, load_psf
from sisr_net import SourceSISR
from sources import SersicSource
from train_b4 import LENS_KEYS, SRC_KEYS, build_inputs

NAVY, AMBER, GOOD, BAD, GREY = "#1E2761", "#D98C1F", "#2E9E5B", "#C0392B", "#8891A5"


#############################################################
# a reusable "run the trained network on this image" wrapper
#############################################################

class B4Runner:

    def __init__(self, ckpt, device="cpu"):
        try:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(ckpt, map_location="cpu")
        self.a = ck["args"]
        self.dev = device
        self.res = self.a["pixel_scale"]
        self.H = self.a["half_extent"]
        self.n_in = int(round(2 * self.H / self.res)) + 1
        self.n_out = self.n_in * self.a["mag"]
        self.net = SourceSISR(in_ch=2 + (self.a["base"] == "sersic"),
                              width=self.a["width"], depth=self.a["depth"],
                              mag=self.a["mag"], n_up=1).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.psf_np = load_psf(self.a["psf_path"])
        self.psf = torch.as_tensor(self.psf_np, dtype=torch.float32, device=device)

    def grids(self, n_pix):
        gx, gy = image_plane_grid(n_pix, self.res, self.a["supersample"])
        return (torch.as_tensor(gx, dtype=torch.float32, device=self.dev),
                torch.as_tensor(gy, dtype=torch.float32, device=self.dev))

    @torch.no_grad()
    def reconstruct(self, img, lens_d, src_d, sigma, GX, GY):
        X = torch.as_tensor(img[None], dtype=torch.float32, device=self.dev)
        S = torch.as_tensor([sigma], dtype=torch.float32, device=self.dev)
        lens = {k: torch.tensor([lens_d[k]], dtype=torch.float32, device=self.dev)
                for k in LENS_KEYS}
        src = {k: torch.tensor([[[src_d[k]]]], dtype=torch.float32, device=self.dev)
               for k in SRC_KEYS}
        bx, by = ray_shoot_batch(GX, GY, lens)
        bp, cov = backproject(X, bx, by, self.n_in, self.H)
        base_in = base_out = None
        if self.a["base"] == "sersic":
            ai = torch.linspace(-self.H, self.H, self.n_in, device=self.dev)
            YY, XX = torch.meshgrid(ai, ai, indexing="ij")
            base_in = SersicSource(src).at(XX[None], YY[None])
            ao = torch.linspace(-self.H, self.H, self.n_out, device=self.dev)
            YO, XO = torch.meshgrid(ao, ao, indexing="ij")
            base_out = SersicSource(src).at(XO[None], YO[None])
        y = self.net(build_inputs(X, S, bp, cov, base_in))
        amp = torch.tensor([[[src_d["amp"]]]], dtype=torch.float32,
                           device=self.dev).clamp_min(1e-6)
        S_map = amp * y
        if base_out is not None:
            S_map = base_out + amp * (y - float(np.log(2.0)))
        return S_map[0].cpu().numpy(), cov[0].cpu().numpy()

    def render(self, S_map, lens_d, background, n_pix, GX, GY):
        lens = {k: torch.tensor([lens_d[k]], dtype=torch.float32, device=self.dev)
                for k in LENS_KEYS}
        with torch.no_grad():
            bx, by = ray_shoot_batch(GX, GY, lens)
            Sm = torch.as_tensor(S_map[None], dtype=torch.float32, device=self.dev)
            sky = sample_source(Sm, bx, by, self.H).unsqueeze(1)
            pad = self.psf.shape[-1] // 2
            blur = F.conv2d(F.pad(sky, (pad,) * 4, mode="replicate"),
                            self.psf[None, None])
            out = area_downsample(blur, self.a["supersample"])[0, 0].cpu().numpy()
        return out + background


def source_axes(n, H):
    ax = np.linspace(-H, H, n)
    return np.meshgrid(ax, ax, indexing="xy")[0], np.meshgrid(ax, ax, indexing="ij")[0]


###############
# EXPERIMENT A
###############

def experiment_a(run, ds, lens_rows, idx, GX, GY, n_pix, bins):
    Xs, Ys = source_axes(run.n_out, run.H)
    acc = {k: [] for k in range(len(bins) - 1)}
    for i in idx:
        img = ds.image(i)
        sig = ds.sigma(i, img)
        r = lens_rows[i]
        S_map, cov = run.reconstruct(img, r, r, sig, GX, GY)
        # true source resampled onto the source grid, PSF-matched
        tru = ds.unlensed(i)
        c = (n_pix - 1) / 2.0
        ti = np.clip(((Ys / run.res) + c).round().astype(int), 0, n_pix - 1)
        tj = np.clip(((Xs / run.res) + c).round().astype(int), 0, n_pix - 1)
        T = tru[ti, tj]
        P = convolve(np.clip(S_map, 0, None), run.psf_np)
        # optimal global amplitude, so the test measures shape, not calibration
        s = (P * T).sum() / max((P * P).sum(), 1e-30)
        err = (s * P - T) ** 2
        m = run.a["mag"]
        cov_up = np.kron(cov, np.ones((m, m)))[:run.n_out, :run.n_out]
        for b in range(len(bins) - 1):
            sel = (cov_up >= bins[b]) & (cov_up < bins[b + 1])
            if sel.sum() >= 20:
                # normalise inside the bin. Using the whole-map mean turns
                # this into a brightness measurement: high-mu regions sit
                # near the bright source centre, so their absolute error is
                # larger however well they are reconstructed.
                acc[b].append(err[sel].mean() / max((T[sel] ** 2).mean(), 1e-30))
    return {b: (len(v), float(np.median(v)) if v else np.nan) for b, v in acc.items()}


###############
# EXPERIMENT B
###############

def experiment_b(run, ds, lens_rows, idx, GX, GY, n_pix, rng,
                 clump_fwhm=0.08, clump_frac=0.15, n_pos=24, ap_px=2.0):
    Xs, Ys = source_axes(run.n_out, run.H)
    sc = 2 * run.H / (run.n_out - 1)
    sig_c = clump_fwhm / 2.3548200450309493
    rows = []
    for i in idx:
        img = ds.image(i)
        noise = ds.sigma(i, img)
        r = lens_rows[i]
        src_d = {k: r[k] for k in SRC_KEYS}

        # clean (noiseless) parametric system for this image
        base = SersicSource({k: np.float64(v) for k, v in src_d.items()}).at(Xs, Ys)
        peak = float(base.max())
        A = clump_frac * peak

        # Sample clump positions over the whole box, not just the centre. mu
        # is highest at the centre, so restricting the sampling would drop the
        # low-mu control points the test needs.
        cand = np.argwhere(np.hypot(Xs, Ys) < 0.92 * run.H)
        if len(cand) < n_pos:
            continue
        pick = cand[rng.choice(len(cand), n_pos, replace=False)]

        # One noise realisation, shared by the with- and without-clump renders.
        # Independent noise would make the difference clump + sqrt(2)*noise,
        # which buries the signal: Spearman(mu, contrast) = -0.04 on pure noise.
        eps = rng.normal(0.0, noise, (n_pix, n_pix))

        img0 = run.render(base, r, r["background"], n_pix, GX, GY) + eps
        S0, cov = run.reconstruct(img0, r, src_d, noise, GX, GY)
        m = run.a["mag"]
        cov_up = np.kron(cov, np.ones((m, m)))[:run.n_out, :run.n_out]

        for (pi, pj) in pick:
            x0, y0 = Xs[pi, pj], Ys[pi, pj]
            clump = A * np.exp(-((Xs - x0) ** 2 + (Ys - y0) ** 2) / (2 * sig_c ** 2))
            img1 = run.render(base + clump, r, r["background"], n_pix, GX, GY) + eps
            S1, _ = run.reconstruct(img1, r, src_d, noise, GX, GY)

            d = S1 - S0
            ap = (np.hypot(Xs - x0, Ys - y0) <= ap_px * sc)
            inj_ap = clump[ap].mean()
            rec_ap = d[ap].mean()
            rows.append(dict(index=int(i), mu=float(cov_up[pi, pj]),
                             injected=float(inj_ap), recovered=float(rec_ap),
                             contrast=float(rec_ap / max(inj_ap, 1e-30)),
                             snr=float(ds.truth(i)["snr_max"]),
                             r_src=float(np.hypot(x0 - src_d["sx"],
                                                  y0 - src_d["sy"]))))
    return rows


def figures(a_res, bins, b_rows, outdir, run, clump_fwhm):
    # figure A
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    xs, ys, ns = [], [], []
    for b in range(len(bins) - 1):
        n, v = a_res[b]
        if n >= 5 and np.isfinite(v):
            xs.append(0.5 * (bins[b] + bins[b + 1])); ys.append(v); ns.append(n)
    ax[0].plot(xs, ys, "-o", c=NAVY, ms=6)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"local magnification $\mu$  (= ray coverage per source pixel)")
    ax[0].set_ylabel("normalised squared error")
    ax[0].set_title("A. Reconstruction error vs magnification", fontsize=11, color=NAVY)
    ax[0].grid(alpha=.25, which="both")
    for x, y, n in zip(xs, ys, ns):
        ax[0].annotate(f"{n}", (x, y), fontsize=7, color=GREY,
                       textcoords="offset points", xytext=(0, 7), ha="center")

    #figure B
    if b_rows:
        mu = np.array([r["mu"] for r in b_rows], float)
        ct = np.array([r["contrast"] for r in b_rows], float)
        ax[1].scatter(mu, ct, s=10, c=NAVY, alpha=.35, edgecolors="none")
        edges = np.logspace(np.log10(max(mu.min(), 0.3)), np.log10(mu.max()), 9)
        cx, cy, lo, hi = [], [], [], []
        for k in range(len(edges) - 1):
            s = (mu >= edges[k]) & (mu < edges[k + 1])
            if s.sum() >= 8:
                cx.append(np.sqrt(edges[k] * edges[k + 1]))
                cy.append(np.median(ct[s]))
                lo.append(np.percentile(ct[s], 25)); hi.append(np.percentile(ct[s], 75))
        ax[1].plot(cx, cy, "-o", c=AMBER, ms=7, lw=2.2, label="median")
        ax[1].fill_between(cx, lo, hi, color=AMBER, alpha=.20, label="25-75%")
        ax[1].axhline(0, c="k", lw=1.0)
        ax[1].axhline(1, c=GOOD, ls="--", lw=1.3, label="perfect recovery")
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r"local magnification $\mu$ at the clump")
        ax[1].set_ylabel("recovered / injected contrast")
        ax[1].set_title(f"B. Recovery of a {clump_fwhm:.2f}\" clump "
                        f"({clump_fwhm/run.res:.2f} detector px)",
                        fontsize=11, color=NAVY)
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.25, which="both")
    fig.suptitle("Does the magnification predict where super-resolution works?",
                 fontsize=12, color=NAVY)
    fig.tight_layout()
    out = f"{outdir}/figMU1_resolution_vs_mu.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/b4_sersic_best.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--n-pos", type=int, default=24,
                    help="clump positions per image (experiment B)")
    ap.add_argument("--clump-fwhm", type=float, default=0.08,
                    help="arcsec; 0.08 is 0.76 detector pixels")
    ap.add_argument("--clump-frac", type=float, default=0.15,
                    help="clump peak as a fraction of the Sersic peak")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--out", default="results/mu_resolution.json")
    a = ap.parse_args()

    run = B4Runner(a.ckpt)
    lens_rows = {r["index"]: r for r in json.load(open(run.a["lens_fits"]))["rows"]}
    ds = ModelADataset(a.root, split="val", classes=run.a["classes"],
                       limit=a.n_images)
    ds.assert_rows_match(lens_rows, run.a["lens_fits"])
    idx = [i for i in range(len(ds)) if i in lens_rows][:a.n_images]
    n_pix = ds.image(0).shape[-1]
    GX, GY = run.grids(n_pix)
    rng = np.random.default_rng(a.seed)

    print("=" * 78)
    print("MAGNIFICATION AND SUPER-RESOLUTION")
    print("=" * 78)
    print(f"  checkpoint {a.ckpt}   base={run.a['base']}")
    print(f"  source grid {run.n_out}^2 at {2*run.H/(run.n_out-1):.4f} arcsec/px"
          f"   detector {run.res:.4f}   ratio {run.res/(2*run.H/(run.n_out-1)):.2f}x")
    print(f"  coverage grid {run.n_in}^2 at {2*run.H/(run.n_in-1):.4f} arcsec/px,")
    print(f"  = {2*run.H/(run.n_in-1)/run.res:.3f} detector pixels, so one ray per")
    print("  source pixel is mu ~ 1 and coverage can be read as mu.\n")

    bins = np.array([0.5, 1, 2, 4, 8, 16, 32, 1e9])
    print("A. RECONSTRUCTION ERROR STRATIFIED BY LOCAL MAGNIFICATION\n")
    a_res = experiment_a(run, ds, lens_rows, idx, GX, GY, n_pix, bins)
    print(f"   {'mu range':>14}{'images':>9}{'normalised sq. error':>24}")
    for b in range(len(bins) - 1):
        n, v = a_res[b]
        hi = "inf" if bins[b + 1] > 1e8 else f"{bins[b+1]:g}"
        print(f"   {f'{bins[b]:g} - {hi}':>14}{n:>9}{v:>24.5f}")
    ok = [(0.5 * (bins[b] + bins[b + 1]), a_res[b][1]) for b in range(len(bins) - 1)
          if a_res[b][0] >= 5 and np.isfinite(a_res[b][1])]
    if len(ok) >= 3:
        x = np.log10([p[0] for p in ok]); y = np.log10([p[1] for p in ok])
        sl = float(np.polyfit(x, y, 1)[0])
        print(f"\n   power-law slope d(log error)/d(log mu) = {sl:+.2f}")
        print("   Negative means the reconstruction improves where the lens")
        print("   magnifies. mu is high near the bright source centre, so part")
        print("   of this trend is confounded; experiment B removes that.")

    print(f"\nB. SUB-PIXEL CLUMP INJECTION  "
          f"(FWHM {a.clump_fwhm}\" = {a.clump_fwhm/run.res:.2f} detector px)\n")
    b_rows = experiment_b(run, ds, lens_rows, idx, GX, GY, n_pix, rng,
                          a.clump_fwhm, a.clump_frac, a.n_pos)
    if b_rows:
        mu = np.array([r["mu"] for r in b_rows]); ct = np.array([r["contrast"] for r in b_rows])
        print(f"   {len(b_rows)} injections over {len(idx)} images")
        print(f"   {'mu range':>14}{'n':>7}{'median contrast':>18}{'p25':>9}{'p75':>9}")
        edges = np.array([0.5, 1, 2, 4, 8, 16, 1e9])
        for k in range(len(edges) - 1):
            s = (mu >= edges[k]) & (mu < edges[k + 1])
            if s.sum() < 5:
                continue
            hi = "inf" if edges[k + 1] > 1e8 else f"{edges[k+1]:g}"
            print(f"   {f'{edges[k]:g} - {hi}':>14}{int(s.sum()):>7}"
                  f"{np.median(ct[s]):>18.4f}{np.percentile(ct[s],25):>9.4f}"
                  f"{np.percentile(ct[s],75):>9.4f}")
        from scipy.stats import spearmanr
        rho = spearmanr(mu, ct).statistic
        print(f"\n   Spearman(mu, recovered contrast) = {rho:+.3f}")
        print("   Positive and significant means the lens is what makes the")
        print("   sub-pixel structure recoverable. Contrast near zero everywhere")
        print("   means no recovery at all; flat and non-zero means it comes")
        print("   from the learned prior rather than from the lensing.")

    os.makedirs(a.outdir, exist_ok=True)
    json.dump({"ckpt": a.ckpt, "bins": bins.tolist(),
               "experiment_a": {str(k): v for k, v in a_res.items()},
               "clump_fwhm": a.clump_fwhm, "clump_frac": a.clump_frac,
               "experiment_b": b_rows}, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    figures(a_res, bins, b_rows, a.outdir, run, a.clump_fwhm)


if __name__ == "__main__":
    main()