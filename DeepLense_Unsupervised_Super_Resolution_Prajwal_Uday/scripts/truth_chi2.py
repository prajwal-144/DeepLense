from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_HERE, _os.path.join(_os.path.dirname(_HERE), "src")]

import argparse
import json

import numpy as np

import metrics as M
from data_a import ModelADataset
from raytrace import (gaussian_psf, image_plane_grid, load_psf, moffat_psf,
                      render)
from sources import SersicSource

N_FIT_PARAMS = 14 # matches evaluate.py / fit_per_image.py


def true_source(t: dict, amp: float = 1.0) -> dict:
    return {"amp": amp,
            "R_sersic": t["source_R_sersic"],
            "n_sersic": t["source_n_sersic"],
            "se1": t["source_e1"], "se2": t["source_e2"],
            "sx": t["source_x"], "sy": t["source_y"]}


def true_lens(t: dict) -> dict:
    return {"theta_E": t["theta_E"], "gamma": t["host_slope"],
            "e1": t["host_e1"], "e2": t["host_e2"],
            "g1": t["gamma1_ext"], "g2": t["gamma2_ext"]}


def linear_amp_background(model: np.ndarray, data: np.ndarray, sigma: float):
    w = 1.0 / max(sigma, 1e-12) ** 2
    m = model.astype(np.float64).ravel()
    d = data.astype(np.float64).ravel()
    n = m.size
    Smm = w * float(m @ m)
    Sm = w * float(m.sum())
    Sdm = w * float(d @ m)
    Sd = w * float(d.sum())
    Snn = w * n
    det = Smm * Snn - Sm * Sm
    if abs(det) < 1e-30:
        return 1.0, 0.0
    a = (Sdm * Snn - Sm * Sd) / det
    b = (Smm * Sd - Sm * Sdm) / det
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", default=["axion"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--psf-mode", default="empirical",
                    choices=["gaussian", "moffat", "empirical", "none"])
    ap.add_argument("--psf-fwhm", type=float, default=0.20)
    ap.add_argument("--psf-beta", type=float, default=2.5)
    ap.add_argument("--psf-path", default="results/psf_empirical.npy")
    ap.add_argument("--amp-mode", default="both", choices=["flux", "fit", "both"])
    ap.add_argument("--substructure-probe", action="store_true",
                    help="also report rms(image - image_nss) in the disc. READ "
                         "THE WARNING it prints before quoting the number.")
    ap.add_argument("--out", default="results/truth_chi2.json")
    a = ap.parse_args()

    if a.psf_mode == "empirical" and a.supersample != 1:
        raise SystemExit("--psf-mode empirical requires --supersample 1")

    ds = ModelADataset(root=a.root, split=a.split, classes=a.classes, limit=a.n)
    print(ds.summary())
    n_used = min(a.n, len(ds))
    n_pix = ds.image(0).shape[-1]
    grid = image_plane_grid(n_pix, a.pixel_scale, a.supersample)
    flat_grid = image_plane_grid(n_pix, a.pixel_scale, 1)

    if a.psf_mode == "gaussian":
        psf = gaussian_psf(a.psf_fwhm, a.pixel_scale / a.supersample)
    elif a.psf_mode == "moffat":
        psf = moffat_psf(a.psf_fwhm, a.pixel_scale / a.supersample, beta=a.psf_beta)
    elif a.psf_mode == "empirical":
        psf = load_psf(a.psf_path)
    else:
        psf = None

    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    rad = np.hypot(yy - c, xx - c)
    disc = rad <= a.fit_radius_px
    annulus = rad > ds.bg_radius_px

    rows = []
    for i in range(n_used):
        t = ds.truth(i)
        img = ds.image(i)
        sig = ds.sigma(i, img)
        lens = true_lens(t)

        unit = render(SersicSource(true_source(t, 1.0)), lens, n_pix,
                      a.pixel_scale, psf=psf, supersample=a.supersample,
                      grid=grid, background=0.0)

        r = {"index": i, "path": str(ds.paths[i]), "sigma": sig,
             "snr_max": t["snr_max"], "host_e": t["host_e"],
             "n_pixels": int(disc.sum())}

        if a.amp_mode in ("flux", "both"):
            # amp from a truth array: convolution conserves flux, so the
            # amplitude is just the flux ratio between the stored unlensed
            # source and the same profile rendered at amp = 1.
            X, Y = flat_grid
            unit_src = SersicSource(true_source(t, 1.0)).at(X, Y)
            denom = float(np.sum(unit_src))
            amp = float(np.sum(ds.unlensed(i)) / denom) if denom > 0 else 1.0
            pred = amp * unit
            bg = float(np.median((img - pred)[annulus]))
            pred = pred + bg
            r["flux"] = {"amp": amp, "background": bg,
                         "chi2_per_dof": M.chi2_per_dof(pred, img, sig, disc,
                                                        N_FIT_PARAMS),
                         "chi2_sum": float((((pred - img) / sig) ** 2)[disc].sum())}

        if a.amp_mode in ("fit", "both"):
            amp, bg = linear_amp_background(unit[disc], img[disc], sig)
            pred = amp * unit + bg
            r["fit"] = {"amp": amp, "background": bg,
                        "chi2_per_dof": M.chi2_per_dof(pred, img, sig, disc, N_FIT_PARAMS),
                        "chi2_sum": float((((pred - img) / sig) ** 2)[disc].sum())}

        if a.substructure_probe:
            d = (img - ds.image_nss(i))[rad <= 30.0]
            r["nss_rms_r30"] = float(np.sqrt(np.mean(d * d)))

        rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_used}")

    # report
    print(f"\nCHI2 FLOOR: the exact physical model   n = {len(rows)}")
    print(f"  true source + true lens + {a.psf_mode} kernel, noiseless, "
          f"supersample {a.supersample}")
    print(f"  scored on the fit disc (r <= {a.fit_radius_px:.0f} px), "
          f"dof = n_pixels - {N_FIT_PARAMS}\n")

    modes = [m for m in ("flux", "fit") if m in rows[0]]
    print(f"   {'amplitude mode':<16} {'median':>11} {'p10':>11} {'p90':>11}")
    summary = {}
    for m in modes:
        v = np.array([x[m]["chi2_per_dof"] for x in rows], dtype=float)
        summary[m] = {"median": float(np.median(v)),
                      "p10": float(np.percentile(v, 10)),
                      "p90": float(np.percentile(v, 90))}
        print(f"   {m:<16} {np.median(v):>11.1f} {np.percentile(v, 10):>11.1f} "
              f"{np.percentile(v, 90):>11.1f}")

    snr = np.array([x["snr_max"] for x in rows], dtype=float)
    print(f"\n   stratified by snr_max (median chi2/dof):")
    print("   " + " ".join(f"{k:>12}" for k in ("bin", "n", *modes)))
    strat = []
    for lo, hi, lab in ((0, 6, "0-6"), (6, 15, "6-15"), (15, np.inf, ">15")):
        msk = (snr >= lo) & (snr < hi)
        if not msk.any():
            continue
        cells = [float(np.median([rows[j][m]["chi2_per_dof"]
                                  for j in np.where(msk)[0]])) for m in modes]
        strat.append({"bin": lab, "n": int(msk.sum()),
                      **{m: c for m, c in zip(modes, cells)}})
        print("   " + f"{lab:>12} {int(msk.sum()):>12d} " +
              " ".join(f"{c:>12.1f}" for c in cells))

    print("\n   FIT vs FLOOR")
    print("   The per-image fit reports chi2/dof ~ 3228 on the same convention.")
    for m in modes:
        ratio = 3228.0 / summary[m]["median"] if summary[m]["median"] > 0 else np.inf
        print(f"     vs the {m} floor ({summary[m]['median']:.0f}): "
              f"the fit is {ratio:.2f}x the floor")
    print("   A ratio near 1 puts the residual in the model family and the")
    print("   substructure rather than in the fit. Well above 1 leaves an")
    print("   excess the floor does not account for.")

    if a.substructure_probe:
        v = np.array([x["nss_rms_r30"] for x in rows], dtype=float)
        print(f"\n   rms(image - image_nss) inside r < 30 px: median {np.median(v):.3f}")
        print("   image_nss was rendered with a circular deflector, so this")
        print("   difference carries the whole arc displacement as well as the")
        print("   substructure. It is an upper bound on the substructure")
        print("   amplitude, not a measurement.")

    out = {"config": vars(a), "n": len(rows), "summary": summary,
           "stratified": strat, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   wrote {a.out}")


if __name__ == "__main__":
    main()