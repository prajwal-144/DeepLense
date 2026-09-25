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
    raise SystemExit(f"eval_b4.py needs torch (import failed: {exc})")

import metrics as M
from backproject import backproject, ray_shoot_batch, sample_source
from data_a import ModelADataset
from raytrace import area_downsample, convolve, image_plane_grid, load_psf
from sisr_net import SourceSISR
from sources import SersicSource
from train_b4 import LENS_KEYS, SRC_KEYS, build_inputs


def to_detector(S_map, half_extent, n_pix, res):
    ax = (np.arange(n_pix) - (n_pix - 1) / 2.0) * res
    XX, YY = np.meshgrid(ax, ax, indexing="xy")[0], np.meshgrid(ax, ax, indexing="ij")[0]
    bx = torch.as_tensor(XX, dtype=S_map.dtype, device=S_map.device)[None]
    by = torch.as_tensor(YY, dtype=S_map.dtype, device=S_map.device)[None]
    out = []
    for i in range(S_map.shape[0]):
        out.append(sample_source(S_map[i:i + 1], bx, by, half_extent)[0])
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/b4_best.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-save", type=int, default=6)
    ap.add_argument("--out", default="results/b4_metrics.json")
    ap.add_argument("--out-npz", default="results/b4_examples.npz")
    a = ap.parse_args()

    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(a.ckpt, map_location="cpu")
    ta = ck["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = ta["pixel_scale"]
    H = ta["half_extent"]

    lens_rows = {r["index"]: r for r in json.load(open(ta["lens_fits"]))["rows"]}
    ds = ModelADataset(a.root, split="val", classes=ta["classes"], limit=(a.n or ta["n_val"]))
    ds.assert_rows_match(lens_rows, ta["lens_fits"])
    idx = [i for i in range(len(ds)) if i in lens_rows]
    n_pix = ds.image(0).shape[-1]

    n_in = int(round(2 * H / res)) + 1
    n_out = n_in * ta["mag"]
    net = SourceSISR(in_ch=2 + (ta["base"] == "sersic"), width=ta["width"], depth=ta["depth"], mag=ta["mag"], n_up=1).to(dev)
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
    print(f"B4 EVALUATION   {a.ckpt}")
    print("=" * 78)
    h = ck.get("history", [])
    if h:
        print(f"{len(h)} epochs   final train {h[-1]['train_chi2']:.1f}"
              f"val {h[-1]['val_chi2']:.1f}"
              f"gap {h[-1]['val_chi2']/max(h[-1]['train_chi2'],1e-9):.2f}x")
    print(f"lens frozen from {ta['lens_fits']}   base {ta['base']}")
    print(f"source {n_out}^2 at {2*H/(n_out-1):.4f} arcsec/px "
          f"({res/(2*H/(n_out-1)):.2f}x finer than the detector)")
    print(f"scoring {len(idx)} val images\n")

    st_b4, st_par, rows, examples = [], [], [], {}
    t_all, saved = 0.0, 0
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
            st_b4.append(M.source_truth(convolve(det[j], psf_np), tru, res))
            par = SersicSource({kk: lens_rows[i][kk] for kk in SRC_KEYS}).at(Xl, Yl)
            st_par.append(M.source_truth(convolve(par, psf_np), tru, res))
            r = (pr[j] - imgs[j]) / max(sigs[j], 1e-12)
            rows.append({"index": i,
                         "chi2_per_dof": float((r[mask_np] ** 2).sum() / (mask_np.sum() - 14)),
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
                examples[f"src_b4_native_{saved}"] = S_map[j].cpu().numpy().astype(np.float32)
                saved += 1
        if (k0 + a.batch_size) % 160 == 0:
            print(f"  {min(k0+a.batch_size, len(idx))}/{len(idx)}", flush=True)

    gm = lambda L, k: float(np.nanmedian([x[k] for x in L]))
    print("\n1. SOURCE PLANE vs the npz `unlensed` array, same images, same PSF\n")
    print(f"   {'method':>34}{'corr':>9}{'size_ratio':>12}{'nmse':>9}"
          f"{'peak_ratio':>12}{'centroid px':>13}")
    ref = [("Sersic (7 params, the prior is exact here)", st_par),
           ("B4: back-projection + SISR", st_b4)]
    for lbl, L in ref:
        print(f"   {lbl:>34}{gm(L,'corr'):9.4f}{gm(L,'size_ratio'):12.4f}"
              f"{gm(L,'nmse'):9.4f}{gm(L,'peak_ratio'):12.4f}"
              f"{gm(L,'centroid_err_px'):13.4f}")
    print("\nFor context, measured earlier on the same dataset:")
    print(" Option 2 linear inversion, 64^2 free px ... corr 0.958  nmse 0.082")
    print(" B3 decoder (Sersic + 32^2 correction) ... corr 0.959  nmse 0.080")
    print(" original grid pipeline ... size_ratio 10.6-12.8")

    print("\n2. WELL-POSEDNESS OF THE FREE-FORM SOURCE\n")
    fb = np.array([r["src_flux_in_box"] for r in rows], float)
    print(f" recovered flux / true flux:  p10 {np.percentile(fb,10):.3f}"
          f" median {np.median(fb):.3f}   p90 {np.percentile(fb,90):.3f}")
    print(" A boxed source cannot hold flux outside its box; at "
          f"half-extent {H} arcsec")
    print(" the true source has ~89% of its flux inside, so the ceiling is")
    print(" ~0.9, not 1.0. Below that the network is losing flux; above it,")
    print(" adding flux that is not there.")
    ch = np.array([r["chi2_per_dof"] for r in rows], float)
    print(f"\n chi2/dof (plain sigma, comparable with evaluate.py): "
          f"median {np.median(ch):.0f}   p10 {np.percentile(ch,10):.0f}"
          f" p90 {np.percentile(ch,90):.0f}")
    print(f" inference {1000*t_all/max(len(rows),1):.1f} ms/image on {dev}")

    print("\n3. STRATIFIED BY snr_max\n")
    snr = np.array([r["snr_max"] for r in rows], float)
    for key, L in (("nmse", st_b4), ("size_ratio", st_b4)):
        v = np.array([x[key] for x in L], float)
        line = "   ".join(
            f"{lo:g}-{hi if hi < 1e8 else 'inf'}: {np.median(v[(snr>=lo)&(snr<hi)]):.4f}"
            for lo, hi in ((0, 6), (6, 15), (15, 1e9)))
        print(f"   {key:12s} {line}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": ta,
               "b4": {k: gm(st_b4, k) for k in st_b4[0]},
               "parametric": {k: gm(st_par, k) for k in st_par[0]},
               "rows": rows}, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out_npz, **examples)
    print(f"\n  wrote {a.out}\n  wrote {a.out_npz}")
    print("  figures:  python scripts/make_figures_b4.py")


if __name__ == "__main__":
    main()