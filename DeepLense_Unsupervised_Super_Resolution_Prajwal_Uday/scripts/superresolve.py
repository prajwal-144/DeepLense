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

from data_a import ModelADataset
from lens_models import magnification
from raytrace import (area_downsample, convolve, gaussian_psf, image_plane_grid,
                      load_psf, moffat_psf, render)
from sources import SersicSource

SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")
LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2")


def build_psf(cfg, pixel_scale, supersample=1):
    mode = cfg.get("psf_mode", "empirical")
    if mode == "gaussian":
        return gaussian_psf(cfg.get("psf_fwhm", 0.20), pixel_scale / supersample)
    if mode == "moffat":
        return moffat_psf(cfg.get("psf_fwhm", 0.20), pixel_scale / supersample,
                          beta=cfg.get("psf_beta", 2.5))
    return load_psf(cfg.get("psf_path", "results/psf_empirical.npy"))


def source_params(row):
    return {k: row[k] for k in SRC_KEYS}


def lens_params(row):
    p = {k: row[k] for k in LENS_KEYS}
    p["cx"] = 0.0
    p["cy"] = 0.0
    return p


def truth_source_params(t):
    return {"amp": 1.0, "R_sersic": t["source_R_sersic"],
            "n_sersic": t["source_n_sersic"], "se1": t["source_e1"],
            "se2": t["source_e2"], "sx": t["source_x"], "sy": t["source_y"]}


def corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def nmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = (a * b).sum() / max((a * a).sum(), 1e-30)
    return float(((s * a - b) ** 2).sum() / max((b * b).sum(), 1e-30))


def rms_size(a, thresh_frac=0.05):
    a = np.clip(np.asarray(a, float), 0, None)
    pk = a.max()
    if pk <= 0:
        return np.nan
    f = np.where(a >= thresh_frac * pk, a, 0.0)
    tot = f.sum()
    if tot <= 0:
        return np.nan
    yy, xx = np.indices(a.shape)
    cy = (f * yy).sum() / tot
    cx = (f * xx).sum() / tot
    return float(np.sqrt((f * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", default="results/fits_img.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="super-resolution factors to render and score")
    ap.add_argument("--n-save", type=int, default=6,
                    help="how many example images to save arrays for (for figures)")
    ap.add_argument("--out-json", default="results/superres_metrics.json")
    ap.add_argument("--out-npz", default="results/superres_examples.npz")
    a = ap.parse_args()

    blob = json.load(open(a.fits))
    cfg, rows = blob["config"], blob["rows"]
    res = cfg.get("pixel_scale", 0.10593)
    ds = ModelADataset(a.root, split=cfg.get("split", "val"),
                       classes=cfg.get("classes", ["axion"]), limit=len(rows))
    ds.assert_rows_match(rows, a.fits)
    n_pix = ds.image(0).shape[-1]
    psf_lr = build_psf(cfg, res, 1)

    print("=" * 76)
    print("SUPER-RESOLUTION FROM THE FITTED PHYSICAL MODEL")
    print("=" * 76)
    print(f"  fits      : {a.fits}   n = {len(rows)}   target = {rows[0].get('target')}")
    print(f"  detector  : {n_pix} px at {res} arcsec/px")
    print(f"  factors   : {a.factors}")
    print("  the fitted source is analytic, so it can be sampled on any grid;")
    print("  the truth is rendered on the same grid from the manifest Sersic")
    print("  parameters (evaluation only).\n")

    per_factor = {int(f): {"corr": [], "nmse": [], "size_ratio": []} for f in a.factors}
    mu_med = []
    examples = {}

    for i, r in enumerate(rows):
        t = ds.truth(r["index"])
        sp_fit = source_params(r)
        sp_true = truth_source_params(t)

        for f in a.factors:
            N = n_pix * f
            sc = res / f
            X, Y = image_plane_grid(N, sc, 1)
            s_fit = SersicSource(sp_fit).at(X, Y)
            s_true = SersicSource(sp_true).at(X, Y)
            per_factor[f]["corr"].append(corr(s_fit, s_true))
            per_factor[f]["nmse"].append(nmse(s_fit, s_true))
            zf, zt = rms_size(s_fit), rms_size(s_true)
            per_factor[f]["size_ratio"].append(zf / zt if zt else np.nan)

        # magnification from the fitted lens; diagnostic only, not a flux factor
        Xl, Yl = image_plane_grid(n_pix, res, 1)
        mu = magnification(Xl, Yl, lens_params(r), mu_clip=50.0)
        c = (n_pix - 1) / 2.0
        yy, xx = np.indices((n_pix, n_pix))
        ring = np.abs(np.hypot(yy - c, xx - c) - r["theta_E"] / res) < 3.0
        mu_med.append(float(np.median(mu[ring])))

        if i < a.n_save:
            f_hi = max(a.factors)
            N = n_pix * f_hi
            sc = res / f_hi
            X, Y = image_plane_grid(N, sc, 1)
            examples[f"obs_{i}"] = ds.image(r["index"]).astype(np.float32)
            examples[f"unlensed_{i}"] = ds.unlensed(r["index"]).astype(np.float32)
            examples[f"src_fit_hi_{i}"] = SersicSource(sp_fit).at(X, Y).astype(np.float32)
            examples[f"src_true_hi_{i}"] = SersicSource(sp_true).at(X, Y).astype(np.float32)
            # model image at the detector scale, for the residual panel
            examples[f"model_{i}"] = render(
                SersicSource(sp_fit), lens_params(r), n_pix, res, psf=psf_lr,
                supersample=1, background=r.get("background", 0.0)).astype(np.float32)
            examples[f"mu_{i}"] = mu.astype(np.float32)
            examples[f"meta_{i}"] = np.array(
                [r["theta_E"], t["theta_E"], r["e"], t["host_e"],
                 t["snr_max"], float(f_hi)], dtype=np.float32)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n  SOURCE FIDELITY vs the analytic truth, on the same grid\n")
    print(f"   {'factor':>7}{'grid':>10}{'scale(as/px)':>14}"
          f"{'corr':>9}{'nmse':>9}{'size_ratio':>12}")
    summary = {}
    for f in a.factors:
        d = per_factor[f]
        cm = float(np.median(d["corr"]))
        nm = float(np.median(d["nmse"]))
        sm = float(np.nanmedian(d["size_ratio"]))
        summary[str(f)] = {"corr": cm, "nmse": nm, "size_ratio": sm}
        print(f"   {f:>7}{n_pix*f:>7} px{res/f:>14.4f}{cm:>9.4f}{nm:>9.4f}{sm:>12.4f}")

    print("\n   These should be flat across factors. Sampling an analytic source")
    print("   more finely adds no error of its own, so a degradation would point")
    print("   at the parameters rather than the sampling.\n")

    print(f"   median |mu| on the ring (fitted lens): {np.median(mu_med):.2f}")
    print("   The lens spreads a small patch of source over many detector")
    print("   pixels, so the sky has already oversampled it. Surface brightness")
    print("   is conserved; mu is never multiplied in.\n")

    # 1x check: the fitted source, PSF-convolved, against the stored unlensed
    c1, s1 = [], []
    X1, Y1 = image_plane_grid(n_pix, res, 1)
    for r in rows:
        s = convolve(SersicSource(source_params(r)).at(X1, Y1), psf_lr)
        u = ds.unlensed(r["index"])
        c1.append(corr(s, u))
        zs, zu = rms_size(s), rms_size(u)
        s1.append(zs / zu if zu else np.nan)
    print("   1x cross-check against the stored `unlensed` array:")
    print(f"     corr {np.median(c1):.4f}   size_ratio {np.nanmedian(s1):.4f}")
    print("     (independent of the analytic-truth comparison above)\n")

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump({"config": cfg, "fits": a.fits, "n": len(rows),
               "by_factor": summary,
               "mu_ring_median": float(np.median(mu_med)),
               "check_1x_vs_stored_unlensed": {
                   "corr": float(np.median(c1)),
                   "size_ratio": float(np.nanmedian(s1))}},
              open(a.out_json, "w"), indent=1)
    np.savez_compressed(a.out_npz, **examples)
    print(f"  wrote {a.out_json}")
    print(f"  wrote {a.out_npz}  ({a.n_save} examples for the figures)")


if __name__ == "__main__":
    main()