from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_HERE, _os.path.join(_os.path.dirname(_HERE), "src")]

import argparse
import json

import numpy as np

from data_a import ModelADataset
from raytrace import gaussian_psf, image_plane_grid, load_psf, render
from sources import SersicSource

MODELS = ("sis_true", "sis_refit", "sie", "epl", "sie_shear")


def true_source(t: dict) -> dict:
    return {"amp": 1.0,
            "R_sersic": t["source_R_sersic"],
            "n_sersic": t["source_n_sersic"],
            "se1": t["source_e1"], "se2": t["source_e2"],
            "sx": t["source_x"], "sy": t["source_y"]}


def true_lens(t: dict) -> dict:
    return {"theta_E": t["theta_E"], "gamma": t["host_slope"],
            "e1": t["host_e1"], "e2": t["host_e2"],
            "g1": t["gamma1_ext"], "g2": t["gamma2_ext"]}


def candidate_lens(name: str, t: dict, theta_E=None) -> dict:
    L = true_lens(t)
    tE = L["theta_E"] if theta_E is None else float(theta_E)
    if name in ("sis_true", "sis_refit"):
        return {"theta_E": tE, "gamma": 2.0,
                "e1": 0.0, "e2": 0.0, "g1": 0.0, "g2": 0.0}
    if name == "sie":
        return {"theta_E": tE, "gamma": 2.0,
                "e1": L["e1"], "e2": L["e2"], "g1": 0.0, "g2": 0.0}
    if name == "epl":
        return {"theta_E": tE, "gamma": L["gamma"],
                "e1": L["e1"], "e2": L["e2"], "g1": 0.0, "g2": 0.0}
    if name == "sie_shear":
        return {"theta_E": tE, "gamma": 2.0,
                "e1": L["e1"], "e2": L["e2"], "g1": L["g1"], "g2": L["g2"]}
    raise ValueError(f"unknown lens model {name!r}")


def weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    sw = w.sum()
    if sw <= 0:
        return np.nan
    ma = float((w * a).sum() / sw)
    mb = float((w * b).sum() / sw)
    da, db = a - ma, b - mb
    va = float((w * da * da).sum() / sw)
    vb = float((w * db * db).sum() / sw)
    if va <= 0 or vb <= 0:
        return np.nan
    return float((w * da * db).sum() / sw / np.sqrt(va * vb))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", default=["axion"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--supersample", type=int, default=3)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--psf-mode", default="gaussian",
                    choices=["gaussian", "empirical", "none"])
    ap.add_argument("--psf-fwhm", type=float, default=0.20)
    ap.add_argument("--psf-path", default="results/psf_empirical.npy")
    ap.add_argument("--refit-span", type=float, default=0.30,
                    help="sis_refit scans theta_E over true*(1 +/- span)")
    ap.add_argument("--refit-steps", type=int, default=41)
    ap.add_argument("--good", type=float, default=0.9,
                    help="threshold for the 'frac below' column")
    ap.add_argument("--out", default="results/lens_ladder.json")
    a = ap.parse_args()

    if a.psf_mode == "empirical" and a.supersample != 1:
        raise SystemExit("--psf-mode empirical requires --supersample 1 "
                         "(the stored kernel is on the detector grid)")

    ds = ModelADataset(root=a.root, split=a.split, classes=a.classes, limit=a.n)
    print(ds.summary())
    n_used = min(a.n, len(ds))
    n_pix = ds.image(0).shape[-1]
    grid = image_plane_grid(n_pix, a.pixel_scale, a.supersample)

    if a.psf_mode == "gaussian":
        psf = gaussian_psf(a.psf_fwhm, a.pixel_scale / a.supersample)
    elif a.psf_mode == "empirical":
        psf = load_psf(a.psf_path)
    else:
        psf = None

    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    disc = np.hypot(yy - c, xx - c) <= a.fit_radius_px

    def draw(lens, src):
        return render(SersicSource(src), lens, n_pix, a.pixel_scale, psf=psf,
                      supersample=a.supersample, grid=grid, background=0.0)

    rows = []
    for i in range(n_used):
        t = ds.truth(i)
        src = true_source(t)
        ref = draw(true_lens(t), src)
        w = np.clip(ref[disc], 0.0, None)
        r = {"index": i, "path": str(ds.paths[i]),
             "host_e": t["host_e"], "gamma_ext": t["gamma_ext"],
             "host_slope": t["host_slope"], "theta_E": t["theta_E"]}

        for name in MODELS:
            if name == "sis_refit":
                span = np.linspace(1.0 - a.refit_span, 1.0 + a.refit_span,
                                   a.refit_steps)
                best, best_tE = -np.inf, t["theta_E"]
                for f in span:
                    tE = float(t["theta_E"] * f)
                    cc = weighted_corr(draw(candidate_lens(name, t, tE), src)[disc],
                                       ref[disc], w)
                    if np.isfinite(cc) and cc > best:
                        best, best_tE = cc, tE
                r[name] = float(best)
                r["sis_refit_theta_E"] = best_tE
            else:
                r[name] = weighted_corr(draw(candidate_lens(name, t), src)[disc],
                                        ref[disc], w)
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_used}")

    # report
    print(f"\nLENS-MODEL LADDER   n = {len(rows)}   {a.split} / {','.join(a.classes)}")
    print(f"  true source, no noise, supersample {a.supersample}, "
          f"psf {a.psf_mode}, arc-weighted over r <= {a.fit_radius_px:.0f} px\n")
    print(f"   {'deflection model':<14} {'mean corr':>10} {'median':>9} "
          f"{'frac < ' + str(a.good):>11}")
    summary = {}
    for name in MODELS:
        v = np.array([x[name] for x in rows], dtype=float)
        v = v[np.isfinite(v)]
        summary[name] = {"mean": float(v.mean()), "median": float(np.median(v)),
                         "frac_below": float((v < a.good).mean()), "n": int(v.size)}
        print(f"   {name:<14} {v.mean():>10.3f} {np.median(v):>9.3f} "
              f"{(v < a.good).mean():>11.0%}")

    print("\nwhat each term buys (mean corr):")
    print(f" refitting theta_E   {summary['sis_refit']['mean'] - summary['sis_true']['mean']:+.3f}")
    print(f" adding ellipticity  {summary['sie']['mean'] - summary['sis_refit']['mean']:+.3f}")
    print(f" adding free slope   {summary['epl']['mean'] - summary['sie']['mean']:+.3f}")
    print(f" adding ext. shear   {summary['sie_shear']['mean'] - summary['sie']['mean']:+.3f}")

    # split by lens ellipticity
    e = np.array([x["host_e"] for x in rows], dtype=float)
    edges = [0.0, 0.1, 0.2, 0.3, 1.0]
    print(f"\n   split by lens |e|:")
    print("   " + " ".join(f"{k:>11}" for k in ("|e| bin", "n", *MODELS)))
    by_e = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e >= lo) & (e < hi)
        if not m.any():
            continue
        cells = [float(np.nanmean([rows[j][k] for j in np.where(m)[0]]))
                 for k in MODELS]
        by_e.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     **{k: c for k, c in zip(MODELS, cells)}})
        label = f"{lo:.1f}-{hi:.1f}"
        print("   " + f"{label:>11} {int(m.sum()):>11d} " +
              " ".join(f"{c:>11.3f}" for c in cells))

    out = {"config": vars(a), "n": len(rows), "summary": summary,
           "by_ellipticity": by_e, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   wrote {a.out}")


if __name__ == "__main__":
    main()