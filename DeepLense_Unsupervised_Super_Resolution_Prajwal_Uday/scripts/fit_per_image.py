from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(os.path.dirname(_HERE), "src")]

import numpy as np
from scipy.optimize import least_squares

from data_a import ModelADataset
from raytrace import gaussian_psf, image_plane_grid, load_psf, moffat_psf, render
from sources import SersicSource, sersic_defaults
from theta_e_init import initial_guess

# order is fixed: the optimiser works on a flat vector
PARAMS = ("theta_E", "gamma", "e1", "e2", "g1", "g2",
          "amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy", "background")

# (lo, hi) in the same order. Wide enough not to bind on Model_A, tight enough
# to keep the profile exponent and the ellipse well conditioned.
BOUNDS = {
    "theta_E":   (0.20, 4.00),
    "gamma":     (1.40, 2.60),
    "e1":        (-0.60, 0.60),
    "e2":        (-0.60, 0.60),
    "g1":        (-0.30, 0.30),
    "g2":        (-0.30, 0.30),
    "amp":       (1e-6, 1e6),
    "R_sersic":  (0.03, 3.00),
    "n_sersic":  (0.30, 6.00),
    "se1":       (-0.60, 0.60),
    "se2":       (-0.60, 0.60),
    "sx":        (-2.00, 2.00),
    "sy":        (-2.00, 2.00),
    "background": (-np.inf, np.inf),
}

# which parameters are free at each stage; earlier stages warm-start the later
STAGES = [
    ("geometry", ("theta_E", "amp", "sx", "sy", "R_sersic", "background")),
    ("+shape",   ("theta_E", "amp", "sx", "sy", "R_sersic", "background",
                  "n_sersic", "se1", "se2")),
    ("+shear",   ("theta_E", "amp", "sx", "sy", "R_sersic", "background",
                  "n_sersic", "se1", "se2", "g1", "g2")),
    ("all",      PARAMS),
]


def _split(v: np.ndarray):
    d = dict(zip(PARAMS, v))
    lens = {k: d[k] for k in ("theta_E", "gamma", "e1", "e2", "g1", "g2")}
    lens["cx"] = 0.0
    lens["cy"] = 0.0
    src = {"amp": d["amp"], "R_sersic": d["R_sersic"], "n_sersic": d["n_sersic"],
           "se1": d["se1"], "se2": d["se2"], "sx": d["sx"], "sy": d["sy"]}
    return lens, src, d["background"]


def initial_vector(img: np.ndarray, pixel_scale: float) -> np.ndarray:
    g = initial_guess(img, pixel_scale)
    s = sersic_defaults()
    amp0 = max(float(np.percentile(img, 99.5)), 1e-3)
    v = {
        "theta_E": float(np.clip(g["theta_E"], 0.4, 3.0)),
        "gamma": 2.0, "e1": 0.0, "e2": 0.0, "g1": 0.0, "g2": 0.0,
        "amp": amp0,
        "R_sersic": s["R_sersic"], "n_sersic": s["n_sersic"],
        "se1": 0.0, "se2": 0.0,
        "sx": float(np.clip(g["sx"], -1.0, 1.0)),
        "sy": float(np.clip(g["sy"], -1.0, 1.0)),
        "background": float(np.median(img)),
    }
    return np.array([v[k] for k in PARAMS], dtype=float)


def fit_one(img: np.ndarray, sigma: float, n_pix: int, pixel_scale: float,
            psf_fwhm: float = 0.18, supersample: int = 2,
            fit_radius_px: float = 45.0, grid=None, psf=None,
            verbose: bool = False) -> Dict:
    if grid is None:
        grid = image_plane_grid(n_pix, pixel_scale, supersample)
    if psf is None:
        psf = gaussian_psf(psf_fwhm, pixel_scale / supersample)

    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask = np.hypot(yy - c, xx - c) <= fit_radius_px
    data = img[mask]
    inv_sig = 1.0 / max(sigma, 1e-12)

    v0 = initial_vector(img, pixel_scale)
    lo = np.array([BOUNDS[k][0] for k in PARAMS])
    hi = np.array([BOUNDS[k][1] for k in PARAMS])
    v0 = np.clip(v0, lo + 1e-9, hi - 1e-9)

    n_eval = [0]

    def model(v):
        lens, src, bg = _split(v)
        n_eval[0] += 1
        return render(SersicSource(src), lens, n_pix, pixel_scale, psf=psf,
                      supersample=supersample, grid=grid, background=bg)

    t0 = time.time()
    v = v0.copy()
    for name, free in STAGES:
        idx = np.array([PARAMS.index(k) for k in free])

        def resid(sub, _idx=idx, _v=v):
            w = _v.copy()
            w[_idx] = sub
            return (model(w)[mask] - data) * inv_sig

        try:
            r = least_squares(resid, v[idx], bounds=(lo[idx], hi[idx]),
                              method="trf", xtol=1e-8, ftol=1e-8, gtol=1e-8,
                              max_nfev=60 * len(idx))
            v[idx] = r.x
        except Exception as exc: # keep the last good vector
            if verbose:
                print(f"    stage {name} failed: {exc}")
            break
        if verbose:
            chi = float((r.fun ** 2).sum() / max(data.size - len(idx), 1))
            print(f"    stage {name:9s} chi2/dof = {chi:8.3f}   nfev={r.nfev}")

    pred = model(v)
    chi2 = float((((pred[mask] - data) * inv_sig) ** 2).sum()
                 / max(data.size - len(PARAMS), 1))
    out = dict(zip(PARAMS, [float(x) for x in v]))
    out.update({"chi2_per_dof": chi2, "n_pixels": int(data.size),
                "n_params": len(PARAMS), "seconds": time.time() - t0,
                "n_model_evals": n_eval[0],
                "e": float(np.hypot(out["e1"], out["e2"])),
                "g": float(np.hypot(out["g1"], out["g2"])),
                "beta": float(np.hypot(out["sx"], out["sy"]))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="folder containing Model_A/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", default=["axion"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--psf-fwhm", type=float, default=0.20,
                    help="MEASURED, not assumed -- calibrate_psf.py finds a clean "
                         "minimum at 0.18-0.20 arcsec against the npz `unlensed` "
                         "array, versus the 0.10 every SIS run used.")
    ap.add_argument("--psf-mode", default="gaussian",
                    choices=["gaussian", "moffat", "empirical"],
                    help="`empirical` uses the stacked Wiener-deconvolved kernel from "
                         "calibrate_psf.py, which has Moffat-like wings the Gaussian "
                         "misses by a factor of 29 at r = 0.25 arcsec.")
    ap.add_argument("--psf-beta", type=float, default=2.5)
    ap.add_argument("--psf-path", default="results/psf_empirical.npy")
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--target", default="image", choices=["image", "image_nss"],
                    help="`image_nss` removes substructure. Fitting it validates the "
                         "forward model and the optimiser; fitting `image` leaves the "
                         "substructure in the residual, which is the DM signal. On the "
                         "axion class rms(image - image_nss) inside r<30 px is 4.33 "
                         "against a background sigma of 0.032, so a smooth model can "
                         "NEVER reach chi2/dof ~ 1 against `image`.")
    ap.add_argument("--out", default="results/fits_val_axion.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ds = ModelADataset(a.root, split=a.split, classes=a.classes, limit=a.n)
    print(ds.summary())
    n_pix = ds.image(0).shape[-1]
    grid = image_plane_grid(n_pix, a.pixel_scale, a.supersample)
    if a.psf_mode == "gaussian":
        psf = gaussian_psf(a.psf_fwhm, a.pixel_scale / a.supersample)
    elif a.psf_mode == "moffat":
        psf = moffat_psf(a.psf_fwhm, a.pixel_scale / a.supersample, beta=a.psf_beta)
    else:
        if a.supersample != 1:
            raise SystemExit("--psf-mode empirical requires --supersample 1 "
                             "(the stored kernel is on the detector grid)")
        psf = load_psf(a.psf_path)
    print("grid %d px @ %s arcsec/px, supersample %d, PSF %s FWHM %.2f\n"
          % (n_pix, a.pixel_scale, a.supersample, a.psf_mode, a.psf_fwhm))

    rows: List[Dict] = []
    t0 = time.time()
    for i in range(len(ds)):
        img = ds.image(i) if a.target == "image" else ds.image_nss(i)
        sig = ds.sigma(i, img)
        if a.verbose:
            print(f"[{i}] sigma={sig:.4f}")
        r = fit_one(img, sig, n_pix, a.pixel_scale, psf_fwhm=a.psf_fwhm,
                    supersample=a.supersample, fit_radius_px=a.fit_radius_px,
                    grid=grid, psf=psf, verbose=a.verbose)
        r["index"] = i
        r["path"] = str(ds.paths[i])
        r["sigma"] = sig
        r["target"] = a.target
        rows.append(r)
        if (i + 1) % 5 == 0 or i == len(ds) - 1:
            el = time.time() - t0
            print(f"  {i+1}/{len(ds)}   chi2/dof median "
                  f"{np.median([x['chi2_per_dof'] for x in rows]):.3f}   "
                  f"{el/(i+1):.1f} s/image   eta {el/(i+1)*(len(ds)-i-1)/60:.1f} min",
                  flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()