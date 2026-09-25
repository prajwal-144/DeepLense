from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(os.path.dirname(_HERE), "src")]

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit(f"eval_b5_mu.py needs torch (import failed: {exc})")

import metrics as M
from backproject import backproject, ray_shoot_batch, sample_source
from data_a import ModelADataset
from eval_b4 import to_detector
from raytrace import area_downsample, convolve, image_plane_grid, load_psf
from sisr_net import SourceSISR
from sources import SersicSource
from train_b4 import LENS_KEYS, SRC_KEYS, build_inputs
from train_b5_mu import apply_gate, mu_gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/b5_gate_best.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-save", type=int, default=6)
    ap.add_argument("--out", default="results/b5_gate_metrics.json")
    ap.add_argument("--out-npz", default="results/b5_gate_examples.npz")
    a = ap.parse_args()

    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(a.ckpt, map_location="cpu")
    ta = ck["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res, H = ta["pixel_scale"], ta["half_extent"]

    lens_rows = {r["index"]: r for r in json.load(open(ta["lens_fits"]))["rows"]}
    ds = ModelADataset(a.root, split="val", classes=ta["classes"],
                       limit=(a.n or ta["n_val"]))
    ds.assert_rows_match(lens_rows, ta["lens_fits"])
    idx = [i for i in range(len(ds)) if i in lens_rows]
    n_pix = ds.image(0).shape[-1]
    n_in = int(round(2 * H / res)) + 1
    n_out = n_in * ta["mag"]

    net = SourceSISR(in_ch=2 + (ta["base"] == "sersic"), width=ta["width"],
                     depth=ta["depth"], mag=ta["mag"], n_up=1).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    psf_np = load_psf(ta["psf_path"])
    psf = torch.as_tensor(psf_np, dtype=torch.float32, device=dev)
    gx, gy = image_plane_grid(n_pix, res, ta["supersample"])
    GX = torch.as_tensor(gx, dtype=torch.float32, device=dev)
    GY = torch.as_tensor(gy, dtype=torch.float32, device=dev)
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask_np = np.hypot(yy - c, xx - c) <= ta["fit_radius_px"]
    Xl, Yl = image_plane_grid(n_pix, res, 1)

    print("=" * 78)
    print(f"B5 (mu-gated) EVALUATION   {a.ckpt}")
    print("=" * 78)
    h = ck.get("history", [])
    if h:
        print(f"  {len(h)} epochs   final train {h[-1]['train_chi2']:.1f}"
              f"   val {h[-1]['val_chi2']:.1f}"
              f"   gap {h[-1]['val_chi2']/max(h[-1]['train_chi2'],1e-9):.2f}x")
    print(f"  base {ta['base']}   mu_gate {ta['mu_gate']} (tau {ta['gate_tau']})"
          f"   curv_mu {ta['curv_mu']}   lambda_curv {ta['lambda_curv']}"
          f"   lambda_l2 {ta['lambda_l2']}")
    print(f"  source {n_out}^2 at {2*H/(n_out-1):.4f} arcsec/px"
          f"   ({res/(2*H/(n_out-1)):.2f}x finer than the detector)")
    print(f"  scoring {len(idx)} val images\n")

    st_b5, st_par, rows, examples = [], [], [], {}
    t_all, saved = 0.0, 0
    gate_frac = []
    for k0 in range(0, len(idx), a.batch_size):
        sub = idx[k0:k0 + a.batch_size]
        imgs = np.array([ds.image(i) for i in sub])
        sigs = np.array([ds.sigma(i, imgs[j]) for j, i in enumerate(sub)])
        L = np.array([[lens_rows[i][k] for k in LENS_KEYS] for i in sub])
        P = np.array([[lens_rows[i][k] for k in SRC_KEYS] for i in sub])
        BG = np.array([lens_rows[i]["background"] for i in sub])
        X = torch.as_tensor(imgs, dtype=torch.float32, device=dev)
        S = torch.as_tensor(sigs, dtype=torch.float32, device=dev)
        Lt = torch.as_tensor(L, dtype=torch.float32, device=dev)
        Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)

        t0 = time.time()
        with torch.no_grad():
            lens = {kk: Lt[:, i] for i, kk in enumerate(LENS_KEYS)}
            bx, by = ray_shoot_batch(GX, GY, lens)
            bp, cov = backproject(X, bx, by, n_in, H)
            g = mu_gate(cov, ta["mu_gate"], ta["gate_tau"])
            if g is not None:
                gate_frac.append(float(g.mean()))
            src = {kk: Pt[:, i].reshape(-1, 1, 1) for i, kk in enumerate(SRC_KEYS)}
            base_in = base_out = None
            if ta["base"] == "sersic":
                ai = torch.linspace(-H, H, n_in, device=dev)
                YY, XX = torch.meshgrid(ai, ai, indexing="ij")
                base_in = SersicSource(src).at(XX[None], YY[None])
                ao = torch.linspace(-H, H, n_out, device=dev)
                YO, XO = torch.meshgrid(ao, ao, indexing="ij")
                base_out = SersicSource(src).at(XO[None], YO[None])
            y = net(build_inputs(X, S, bp, cov, base_in))
            amp = Pt[:, SRC_KEYS.index("amp")].reshape(-1, 1, 1).clamp_min(1e-6)
            S_map = amp * y
            if base_out is not None:
                S_map = base_out + amp * (y - float(np.log(2.0)))
            S_map = apply_gate(S_map, g, ta["mag"])
            sky = sample_source(S_map, bx, by, H).unsqueeze(1)
            pad = psf.shape[-1] // 2
            sky = F.conv2d(F.pad(sky, (pad,) * 4, mode="replicate"), psf[None, None])
            pred = area_downsample(sky, ta["supersample"])[:, 0] + \
                torch.as_tensor(BG, dtype=torch.float32, device=dev).reshape(-1, 1, 1)
            det = to_detector(S_map, H, n_pix, res).cpu().numpy()
        t_all += time.time() - t0

        pr = pred.cpu().numpy()
        for j, i in enumerate(sub):
            tru = ds.unlensed(i)
            st_b5.append(M.source_truth(convolve(det[j], psf_np), tru, res))
            par = SersicSource({kk: lens_rows[i][kk] for kk in SRC_KEYS}).at(Xl, Yl)
            st_par.append(M.source_truth(convolve(par, psf_np), tru, res))
            r = (pr[j] - imgs[j]) / max(sigs[j], 1e-12)
            rows.append({"index": i,
                         "chi2_per_dof": float((r[mask_np] ** 2).sum()
                                               / (mask_np.sum() - 14)),
                         "snr_max": ds.truth(i)["snr_max"],
                         "src_flux_in_box": float(
                             np.clip(det[j], 0, None).sum()
                             / max(np.clip(tru, 0, None).sum(), 1e-30))})
            if saved < a.n_save:
                examples[f"obs_{saved}"] = imgs[j].astype(np.float32)
                examples[f"pred_{saved}"] = pr[j].astype(np.float32)
                examples[f"bp_{saved}"] = bp[j].cpu().numpy().astype(np.float32)
                examples[f"cov_{saved}"] = cov[j].cpu().numpy().astype(np.float32)
                examples[f"src_b4_{saved}"] = det[j].astype(np.float32)
                examples[f"src_par_{saved}"] = par.astype(np.float32)
                examples[f"src_true_{saved}"] = tru.astype(np.float32)
                examples[f"src_b4_native_{saved}"] = \
                    S_map[j].cpu().numpy().astype(np.float32)
                if g is not None:
                    examples[f"gate_{saved}"] = g[j].cpu().numpy().astype(np.float32)
                saved += 1

    gm = lambda L, k: float(np.nanmedian([x[k] for x in L]))
    print("1. SOURCE PLANE vs the npz `unlensed` array\n")
    print(f"   {'method':>34}{'corr':>9}{'size_ratio':>12}{'nmse':>9}"
          f"{'peak_ratio':>12}{'centroid px':>13}")
    for lbl, L in (("Sersic (7 params)", st_par), ("B5 (mu-gated)", st_b5)):
        print(f"   {lbl:>34}{gm(L,'corr'):9.4f}{gm(L,'size_ratio'):12.4f}"
              f"{gm(L,'nmse'):9.4f}{gm(L,'peak_ratio'):12.4f}"
              f"{gm(L,'centroid_err_px'):13.4f}")
    print("\n   For context on the same 800 images:")
    print("     B4 free       corr 0.9731  size 1.130  nmse 0.0527  peak 0.748")
    print("     B4 +sersic    corr 0.9794  size 1.043  nmse 0.0408  peak 0.927")

    fb = np.array([r["src_flux_in_box"] for r in rows], float)
    ch = np.array([r["chi2_per_dof"] for r in rows], float)
    print("\n2. WELL-POSEDNESS\n")
    print(f"   flux in box: p10 {np.percentile(fb,10):.3f}  median {np.median(fb):.3f}"
          f"  p90 {np.percentile(fb,90):.3f}   (ceiling ~0.89 at H=1.2)")
    print(f"   chi2/dof (plain sigma): median {np.median(ch):.0f}"
          f"   p10 {np.percentile(ch,10):.0f}   p90 {np.percentile(ch,90):.0f}")
    if gate_frac:
        print(f"\n   mean gate value {np.mean(gate_frac):.3f}"
              f"  (fraction of the source box where the full detail passes)")
        print("   Near 1 means the gate is doing nothing (raise --mu-gate);")
        print("   near 0 means it is off entirely.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": ta, "b4": {k: gm(st_b5, k) for k in st_b5[0]},
               "parametric": {k: gm(st_par, k) for k in st_par[0]},
               "rows": rows}, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out_npz, **examples)
    print(f"\n  wrote {a.out}\n  wrote {a.out_npz}")
    print(f"  figures:  python scripts/make_figures_b4.py --metrics {a.out} "
          f"--npz {a.out_npz}")


if __name__ == "__main__":
    main()