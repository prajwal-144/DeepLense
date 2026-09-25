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
except Exception as exc:
    raise SystemExit(f"eval_pathb.py needs torch (import failed: {exc})")

import metrics as M
from data_a import ModelADataset
from raytrace import convolve, image_plane_grid, load_psf
from sources import SersicSource
from theta_e_init import initial_guess
from train_pathb import (RING_FEATURES, PathBNet, build_input, ray_coverage,
                         render_batch, ring_summary, sample_correction,
                         to_params)

SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")
FIT_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "amp", "R_sersic",
            "n_sersic", "se1", "se2", "sx", "sy", "background")


def source_maps(p, C_eff, X, Y, half_extent, i, source_mode="b3"):
    Xt = torch.as_tensor(X, dtype=C_eff.dtype, device=C_eff.device)[None]
    Yt = torch.as_tensor(Y, dtype=C_eff.dtype, device=C_eff.device)[None]
    corr = sample_correction(C_eff[i:i + 1], Xt, Yt,
                             half_extent)[0].detach().cpu().numpy()
    if source_mode == "b2":
        return np.zeros_like(corr), corr
    sp = {k: p[k][i].item() for k in SRC_KEYS}
    base = SersicSource(sp).at(X, Y)
    return base, base + corr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/pathb.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0, help="0 = the training --n-val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fits", default="results/fits_img.json",
                    help="per-image fits, for the head-to-head")
    ap.add_argument("--n-save", type=int, default=6)
    ap.add_argument("--out", default="results/fits_pathb.json")
    ap.add_argument("--out-npz", default="results/pathb_examples.npz")
    a = ap.parse_args()

    try: # weights_only= only exists on torch>=2
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(a.ckpt, map_location="cpu")
    ta = ck["args"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = ta["pixel_scale"]
    n_show = a.n or ta["n_val"]

    net = PathBNet(n_c=ta["n_c"], in_ch=1 + len(RING_FEATURES)).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    psf_path = ta["psf_path"] if os.path.exists(ta["psf_path"]) \
        else "results/psf_empirical.npy"
    psf_np = load_psf(psf_path)
    psf = torch.as_tensor(psf_np, dtype=torch.float32, device=dev)

    ds = ModelADataset(a.root, split=a.split, classes=ta["classes"], limit=n_show)
    n_pix = ds.image(0).shape[-1]
    gx, gy = image_plane_grid(n_pix, res, ta["supersample"])
    grid = (torch.as_tensor(gx, dtype=torch.float32, device=dev),
            torch.as_tensor(gy, dtype=torch.float32, device=dev))
    Xl, Yl = image_plane_grid(n_pix, res, 1)
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask_np = np.hypot(yy - c, xx - c) <= ta["fit_radius_px"]
    mask = torch.as_tensor(mask_np, device=dev)

    print("=" * 78)
    print(f"PATH B EVALUATION   {a.ckpt}")
    print("=" * 78)
    h = ck.get("history", [])
    if h:
        print(f" trained {len(h)} epochs   final train chi2 {h[-1]['train_chi2']:.1f}"
              f" val chi2 {h[-1]['val_chi2']:.1f}"
              f" batches skipped {ck.get('n_skipped', 0)}")
    smode = ta.get("source_mode", "b3")
    print(f"  source mode {smode}   correction {ta['n_c']}^2"
          f"   corr_scale {ta['corr_scale']}"
          f"   reg {ta['reg_mode']}   lambda {ta['lambda_corr']}"
          f"   sigma_floor {ta.get('sigma_floor', 0.0)}")
    if h:
        print(f"  final train/val gap {h[-1]['val_chi2']/max(h[-1]['train_chi2'],1e-9):.2f}x"
              f"   final tanh saturation {100*h[-1].get('saturation', float('nan')):.1f}%")
    print(f"  scoring {len(ds)} {a.split} images\n")

    # forward pass
    rows, st_base, st_corr, examples = [], [], [], {}
    t_all, k = 0.0, 0
    for k0 in range(0, len(ds), a.batch_size):
        idx = list(range(k0, min(k0 + a.batch_size, len(ds))))
        imgs = np.array([ds.image(i) for i in idx])
        sigs = np.array([ds.sigma(i, imgs[j]) for j, i in enumerate(idx)])
        ring = np.array([ring_summary(imgs[j], res) for j in range(len(idx))])
        Xb = torch.as_tensor(imgs, dtype=torch.float32, device=dev)
        Sb = torch.as_tensor(sigs, dtype=torch.float32, device=dev)
        Rb = torch.as_tensor(ring, dtype=torch.float32, device=dev)

        t0 = time.time()
        inp = build_input(Xb, Sb, Rb, stretch=bool(ta.get("stretch", 1)))
        z, Craw = net(inp)
        p = to_params(z, Rb)
        pred, C_eff, (bx, by) = render_batch(
            p, Craw, grid, psf, n_pix, ta["supersample"], ta["half_extent"],
            ta["corr_scale"], smode)
        t_all += time.time() - t0

        cov = ray_coverage(bx, by, ta["n_c"], ta["half_extent"])
        r = ((pred - Xb) / Sb[:, None, None])[:, mask]
        chi2 = (r ** 2).sum(1) / (int(mask_np.sum()) - 14 - ta["n_c"] ** 2)

        for j, i in enumerate(idx):
            row = {kk: float(p[kk][j]) for kk in FIT_KEYS}
            row.update(ring_theta_E=float(ring[j, 0]),
                       ring_m1=float(ring[j, 1]), ring_m2=float(ring[j, 2]),
                       index=i, path=str(ds.paths[i]), sigma=float(sigs[j]),
                       target=ta["target"], chi2_per_dof=float(chi2[j]),
                       n_pixels=int(mask_np.sum()),
                       n_params=14 + ta["n_c"] ** 2, seconds=np.nan,
                       e=float(np.hypot(row["e1"], row["e2"])),
                       g=float(np.hypot(row["g1"], row["g2"])),
                       beta=float(np.hypot(row["sx"], row["sy"])))
            Ce = C_eff[j].detach().cpu().numpy()
            cv = cov[j].detach().cpu().numpy()
            row["corr_rms_frac"] = float(np.sqrt((Ce ** 2).mean()) /
                                         max(abs(row["amp"]), 1e-9))
            rel = np.abs(Ce) / max(abs(row["amp"]), 1e-9)
            row["corr_saturation"] = float((rel > 0.99 * ta["corr_scale"]).mean())
            m = cv > 0
            row["corr_vs_mu"] = float(np.corrcoef(
                np.abs(Ce[m]).ravel(), np.log10(cv[m]).ravel())[0, 1]) \
                if m.sum() > 10 else np.nan
            rows.append(row)

            b0, b1 = source_maps(p, C_eff, Xl, Yl, ta["half_extent"], j, smode)
            tru = ds.unlensed(i)
            st_base.append(M.source_truth(convolve(b0, psf_np), tru, res))
            st_corr.append(M.source_truth(convolve(b1, psf_np), tru, res))

            if k < a.n_save:
                examples[f"obs_{k}"] = imgs[j].astype(np.float32)
                examples[f"pred_{k}"] = pred[j].detach().cpu().numpy().astype(np.float32)
                examples[f"src_base_{k}"] = b0.astype(np.float32)
                examples[f"src_corr_{k}"] = b1.astype(np.float32)
                examples[f"src_true_{k}"] = tru.astype(np.float32)
                examples[f"corr_map_{k}"] = Ce.astype(np.float32)
                examples[f"cov_{k}"] = cv.astype(np.float32)
                k += 1
        if (k0 + a.batch_size) % 160 == 0:
            print(f"  {min(k0+a.batch_size, len(ds))}/{len(ds)}", flush=True)

    # 0. collapse check
    print("\n0. COLLAPSE CHECK: across-image spread of the predictions\n")
    print("   A ratio near zero means the network emits the same number for")
    print("   every image, in which case nothing below is usable.\n")
    print(f"   {'parameter':12s}{'pred std':>12}{'truth std':>12}{'ratio':>9}")
    TRUTH_OF = {"theta_E": "theta_E", "gamma": "host_slope",
                "R_sersic": "source_R_sersic", "n_sersic": "source_n_sersic",
                "e": "host_e", "g": "gamma_ext", "beta": "beta"}
    collapsed = []
    for fk, tk in TRUTH_OF.items():
        pv = np.array([r[fk] for r in rows], float)
        tv = np.array([ds.truth(r["index"])[tk] for r in rows], float)
        sp_, stv = float(np.std(pv)), float(np.std(tv))
        ratio = sp_ / max(stv, 1e-12)
        flag = "  <-- COLLAPSED" if ratio < 0.05 else ""
        if ratio < 0.05:
            collapsed.append(fk)
        print(f"   {fk:12s}{sp_:12.4f}{stv:12.4f}{ratio:9.3f}{flag}")
    if collapsed:
        print(f"\n   *** {len(collapsed)} parameter(s) collapsed: {collapsed}")
        print("   *** Do not report the recovery table. Retrain with a larger")
        print("   *** --lr or check tests/test_backend_parity.py first.")
    else:
        print("\n   No collapse: every parameter varies across images.")

    # 0b. control: ring statistic alone
    print("\n0b. NETWORK vs ITS OWN INPUTS\n")
    print(" Four ring statistics are fed in as input planes, so a network that")
    print(" only echoed them would still score well. Below: Spearman rho of the")
    print(" ring statistic alone against truth, next to the network's own.\n")
    from scipy.stats import spearmanr as _sp
    CTRL = [("theta_E", "ring_theta_E", "theta_E"),
            ("beta", "ring_m1", "beta"),
            ("e", "ring_m2", "host_e")]
    for fk, rk, tk in CTRL:
        t = np.array([ds.truth(r["index"])[tk] for r in rows], float)
        net_v = np.array([r[fk] for r in rows], float)
        ring_v = np.array([r[rk] for r in rows], float)
        m = np.isfinite(t) & np.isfinite(net_v) & np.isfinite(ring_v)
        rn = _sp(net_v[m], t[m]).statistic
        rr = _sp(ring_v[m], t[m]).statistic
        verdict = "network ADDS" if rn > rr + 0.03 else (
            "no gain (echoes input)" if rn > rr - 0.05 else "network LOSES info")
        print(f"   {fk:10s} ring-only rho {rr:+.3f}   network rho {rn:+.3f}"
              f"   -> {verdict}")
    print("\n   Path A (per-image fit) for reference: theta_E +0.956, "
          "beta +0.914, e +0.762")

    # A. correction contribution
    gm = lambda L, kk: float(np.nanmedian([x[kk] for x in L]))
    if smode == "b2":
        print("\nA. FREE-FORM SOURCE  (b2: no Sersic, the map is the source)\n")
        print(f"   {'source model':>26}{'corr':>9}{'size_ratio':>12}{'nmse':>9}"
              f"{'peak_ratio':>12}")
        print(f"   {'free-form (b2)':>26}{gm(st_corr,'corr'):9.4f}"
              f"{gm(st_corr,'size_ratio'):12.4f}{gm(st_corr,'nmse'):9.4f}"
              f"{gm(st_corr,'peak_ratio'):12.4f}")
        print("\n   Path A parametric fit, same images: corr 0.9865,")
        print("   size_ratio 1.0887, nmse 0.0271. Model_A's sources are Sersics,")
        print("   so a free-form model has no prior advantage on this data.")
    else:
        print("\nA. WHAT THE CORRECTION MAP ADDED\n")
        print(f"   {'source model':>26}{'corr':>9}{'size_ratio':>12}{'nmse':>9}"
              f"{'peak_ratio':>12}")
        for lbl, L in (("Sersic only", st_base), ("Sersic + correction", st_corr)):
            print(f"   {lbl:>26}{gm(L,'corr'):9.4f}{gm(L,'size_ratio'):12.4f}"
                  f"{gm(L,'nmse'):9.4f}{gm(L,'peak_ratio'):12.4f}")
        d = 100.0 * (gm(st_corr, "nmse") - gm(st_base, "nmse")) / max(gm(st_base, "nmse"), 1e-12)
        win = 100.0 * float(np.mean([b["nmse"] > c["nmse"]
                                     for b, c in zip(st_base, st_corr)]))
        print(f"\n   source nmse change from the correction: {d:+.1f}%"
              f"   (helps on {win:.0f}% of images)")
        print("   Negative means the correction improved the source.")

    cr = np.array([r["corr_rms_frac"] for r in rows], float)
    sat = np.array([r.get("corr_saturation", np.nan) for r in rows], float)
    print(f"   correction amplitude: rms {np.nanmedian(cr)*100:.2f}% of amp"
          f"   (b3 cap is {ta['corr_scale']*100:.0f}%)")
    if smode == "b3":
        print(f"   tanh saturation: {100*np.nanmedian(sat):.1f}% of the map sits "
              f"within 1% of the bound")
        if np.nanmedian(sat) > 0.25:
            print(" *** Saturated. A saturated tanh has ~zero gradient, so the")
            print(" *** correction has stopped learning and the mu weighting")
            print(" *** cannot shape it. Section C below does not apply.")
            print(" *** Raise --lambda-corr and retrain.")
        if np.nanmedian(cr) < 0.005:
            print(" *** The correction is near zero: the network has reduced to")
            print(" *** the parametric model. Lower --lambda-corr.")

    # C. where the correction went
    cv = np.array([r["corr_vs_mu"] for r in rows], float)
    print("\nC. WHERE THE CORRECTION WAS PLACED\n")
    print(f" corr(|correction|, log ray coverage): median {np.nanmedian(cv):+.4f}")
    print(" Ray coverage counts image sub-pixels landing in each source pixel,")
    print(" so it is the magnification. Positive means detail was added where")
    print(" the lens delivered resolution; near zero means it was spread")
    print(" uniformly.")

    # B. speed
    print("\nB. AMORTISATION\n")
    ms = 1000.0 * t_all / max(len(rows), 1)
    print(f"   network        {ms:9.2f} ms/image   (batched, {dev})")
    if os.path.exists(a.fits):
        pf = json.load(open(a.fits))["rows"]
        sec = np.array([r.get("seconds", np.nan) for r in pf], float)
        sec = sec[np.isfinite(sec)]
        if sec.size:
            print(f"   per-image fit  {1000*np.median(sec):9.2f} ms/image"
                  f"   (Levenberg-Marquardt, CPU)")
            print(f"   speedup        {np.median(sec)*1000/max(ms,1e-9):9.0f}x")
        print("\n The optimisation cost is paid once at training time; inference")
        print(" is a single forward pass.")

    # write
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cfg = {"root": a.root, "split": a.split, "classes": ta["classes"],
           "n": len(rows), "pixel_scale": res, "psf_mode": "empirical",
           "psf_path": psf_path, "supersample": ta["supersample"],
           "fit_radius_px": ta["fit_radius_px"], "target": ta["target"],
           "source": "pathb", "ckpt": a.ckpt}
    json.dump({"config": cfg, "rows": rows,
               "source_truth_base": {kk: gm(st_base, kk) for kk in st_base[0]},
               "source_truth_corrected": {kk: gm(st_corr, kk) for kk in st_corr[0]}},
              open(a.out, "w"), indent=1)
    np.savez_compressed(a.out_npz, **examples)
    print(f"\n  wrote {a.out}")
    print(f" wrote {a.out_npz}")
    print("\n  score with the same scorer as the per-image fit:")
    print(f" python scripts/evaluate.py --fits {a.out} --root {a.root}")


if __name__ == "__main__":
    main()