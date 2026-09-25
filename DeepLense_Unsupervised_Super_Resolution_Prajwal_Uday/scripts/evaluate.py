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
from scipy.stats import spearmanr

import metrics as M
from data_a import ModelADataset
from raytrace import (area_downsample, convolve, gaussian_psf, image_plane_grid,
                      load_psf, moffat_psf, render)
from sources import SersicSource

# fitted name -> truth name
PAIRS = [("theta_E", "theta_E", "arcsec"),
         ("beta", "beta", "arcsec"),
         ("R_sersic", "source_R_sersic", "arcsec"),
         ("n_sersic", "source_n_sersic", ""),
         ("gamma", "host_slope", ""),
         ("e", "host_e", ""),
         ("g", "gamma_ext", "")]


def build_psf(cfg, pixel_scale):
    mode = cfg.get("psf_mode", "gaussian")
    S = cfg.get("supersample", 2)
    if mode == "gaussian":
        return gaussian_psf(cfg.get("psf_fwhm", 0.20), pixel_scale / S)
    if mode == "moffat":
        return moffat_psf(cfg.get("psf_fwhm", 0.20), pixel_scale / S,
                          beta=cfg.get("psf_beta", 2.5))
    return load_psf(cfg.get("psf_path", "results/psf_empirical.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    blob = json.load(open(a.fits))
    cfg, rows = blob["config"], blob["rows"]
    res = cfg.get("pixel_scale", 0.10593)
    S = cfg.get("supersample", 2)
    ds = ModelADataset(a.root, split=cfg.get("split", "val"),
                       classes=cfg.get("classes", ["axion"]), limit=len(rows))
    ds.assert_rows_match(rows, a.fits)
    n_pix = ds.image(0).shape[-1]
    grid = image_plane_grid(n_pix, res, S)
    psf = build_psf(cfg, res)
    psf_lr = build_psf({**cfg, "supersample": 1}, res) if S != 1 else psf

    print("=" * 78)
    print(f"EVALUATION  {a.fits}")
    print(f"  target = {rows[0].get('target','image')}   psf = {cfg.get('psf_mode','gaussian')}"
          f"   supersample = {S}   n = {len(rows)}")
    print("=" * 78)

    # parameter recovery
    print("\n1. PARAMETER RECOVERY  (no truth entered the fit)\n")
    print(f"   {'parameter':12s}{'fit med':>10}{'true med':>10}{'med |err|':>11}"
          f"{'p90 |err|':>11}{'Spearman':>10}")
    recov = {}
    for fk, tk, unit in PAIRS:
        f = np.array([r[fk] for r in rows], float)
        t = np.array([ds.truth(r["index"])[tk] for r in rows], float)
        ok = np.isfinite(f) & np.isfinite(t)
        err = np.abs(f - t)[ok]
        rho = spearmanr(f[ok], t[ok]).statistic if ok.sum() > 3 else np.nan
        recov[fk] = dict(fit=float(np.median(f[ok])), true=float(np.median(t[ok])),
                         mae=float(np.median(err)), p90=float(np.percentile(err, 90)),
                         spearman=float(rho))
        print(f"   {fk:12s}{np.median(f[ok]):10.3f}{np.median(t[ok]):10.3f}"
              f"{np.median(err):11.3f}{np.percentile(err,90):11.3f}{rho:+10.3f}")
    print("\n Fit `image`, not `image_nss`. The _nss arrays were rendered with")
    print(" a circular lens and are inconsistent with `image` and `kappa`:")
    print(" fitting them gives |e| ~ 0.03 at Spearman ~ 0, against |e| ~ 0.20")
    print(" at Spearman ~ +0.78 from `image`, which matches the manifest.")

    # source-plane truth
    print("\n2. SOURCE PLANE vs the npz `unlensed` array\n")
    st = []
    for r in rows:
        sp = {k: r[k] for k in ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")}
        X, Y = image_plane_grid(n_pix, res, 1)
        s = SersicSource(sp).at(X, Y)
        s = convolve(s, psf_lr) # PSF-matched: the truth array is already post-PSF
        st.append(M.source_truth(s, ds.unlensed(r["index"]), res))
    for k in ("corr", "size_ratio", "centroid_err_px", "peak_ratio", "nmse"):
        v = np.array([x[k] for x in st], float)
        v = v[np.isfinite(v)]
        print(f"   {k:20s} median {np.median(v):9.4f}   p10 {np.percentile(v,10):8.4f}"
              f"   p90 {np.percentile(v,90):8.4f}")
    print("\n   size_ratio should be 1. The SIS pipeline sat at 10.6-12.8 across")
    print("   a 100x sweep of the TV weight (tv = 0, 0.03, 0.1, 0.3, 3.0).")
    print("   `unlensed` is post-PSF, so the model source is convolved with the")
    print("   PSF before the comparison.")

    # goodness of fit
    print("\n3. GOODNESS OF FIT\n")
    chi = np.array([r["chi2_per_dof"] for r in rows], float)
    print(f"   chi2/dof   median {np.median(chi):.1f}   p10 {np.percentile(chi,10):.1f}"
          f"   p90 {np.percentile(chi,90):.1f}")
    imgs = [ds.image_nss(r["index"]) if r.get("target") == "image_nss"
            else ds.image(r["index"]) for r in rows[:12]]
    sigs = [r["sigma"] for r in rows[:12]]
    nb = M.null_baselines(imgs, sigs)
    print(f"   null baselines (median chi2/dof): " +
          "   ".join(f"{k} {v:.0f}" for k, v in nb.items()))
    print("\n chi2/dof does not reach 1. calibrate_psf.py finds Moffat-like PSF")
    print(" wings: at r = 0.25 arcsec the true kernel is 0.139 of its peak")
    print(" against 0.005 for a 0.18 arcsec Gaussian. Model_A arcs reach")
    print(" peak/sigma_bg of 10^3 to 10^4, so a 1% kernel-shape error is a")
    print(" ~50 sigma per-pixel residual. The kernel shape sets the floor, not")
    print(" the parameters.")

    # stratified
    print("\n4. STRATIFIED BY snr_max\n")
    for r in rows:
        r["snr_max"] = ds.truth(r["index"])["snr_max"]
    for k in ("chi2_per_dof",):
        print(M.stratify(rows, k))
    for i, r in enumerate(rows):
        r["_size_ratio"] = st[i]["size_ratio"]
    print(M.stratify(rows, "_size_ratio"))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"recovery": recov,
                   "source_truth": {k: float(np.nanmedian([x[k] for x in st]))
                                    for k in st[0]},
                   "chi2_median": float(np.median(chi)),
                   "null_baselines": nb},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()