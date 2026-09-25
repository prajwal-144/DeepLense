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
from scipy.optimize import least_squares
from scipy.stats import spearmanr

import metrics as M
from data_a import ModelADataset
from fit_per_image import BOUNDS, PARAMS, STAGES, _split, initial_vector
from raytrace import convolve, image_plane_grid, load_psf, render
from sources import SersicSource

SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


def fit_from(img, sigma, v0, stages, n_pix, res, psf, grid, mask,
             xtol=1e-8, max_nfev_per=60):
    data = img[mask]
    inv = 1.0 / max(sigma, 1e-12)
    lo = np.array([BOUNDS[k][0] for k in PARAMS])
    hi = np.array([BOUNDS[k][1] for k in PARAMS])
    n_eval = [0]

    def model(v):
        L, S, bg = _split(v)
        n_eval[0] += 1
        return render(SersicSource(S), L, n_pix, res, psf=psf,
                      supersample=1, grid=grid, background=bg)

    v = np.clip(np.asarray(v0, float).copy(), lo + 1e-9, hi - 1e-9)
    t0 = time.time()
    for _name, free in stages:
        idx = np.array([PARAMS.index(k) for k in free])

        def resid(sub, _i=idx, _v=v):
            w = _v.copy()
            w[_i] = sub
            return (model(w)[mask] - data) * inv

        try:
            r = least_squares(resid, v[idx], bounds=(lo[idx], hi[idx]),
                              method="trf", xtol=xtol, ftol=xtol, gtol=xtol,
                              max_nfev=max_nfev_per * len(idx))
            v[idx] = r.x
        except Exception:
            break
    pred = model(v)
    chi2 = float((((pred[mask] - data) * inv) ** 2).sum()
                 / max(data.size - len(PARAMS), 1))
    out = dict(zip(PARAMS, [float(x) for x in v]))
    out.update(chi2_per_dof=chi2, n_model_evals=n_eval[0],
               seconds=time.time() - t0, n_pixels=int(data.size),
               n_params=len(PARAMS),
               e=float(np.hypot(out["e1"], out["e2"])),
               g=float(np.hypot(out["g1"], out["g2"])),
               beta=float(np.hypot(out["sx"], out["sy"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pathb", default="results/fits_pb3_mu.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--psf-path", default="results/psf_empirical.npy")
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--xtol", type=float, default=1e-8)
    ap.add_argument("--net-ms", type=float, default=12.0,
                    help="measured network inference time per image, for the "
                         "total-cost line")
    ap.add_argument("--out", default="results/fits_refined.json")
    a = ap.parse_args()

    blob = json.load(open(a.pathb))
    net = {r["index"]: r for r in blob["rows"]}
    idx = sorted(net)[:a.n]
    res = a.pixel_scale

    ds = ModelADataset(a.root, split=blob["config"]["split"],
                       classes=blob["config"]["classes"], limit=max(idx) + 1)
    ds.assert_rows_match(blob["rows"], a.pathb)
    n_pix = ds.image(0).shape[-1]
    grid = image_plane_grid(n_pix, res, 1)
    psf = load_psf(a.psf_path)
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask = np.hypot(yy - c, xx - c) <= a.fit_radius_px

    print("=" * 78)
    print("AMORTISED INITIALISATION + PER-IMAGE REFINEMENT")
    print("=" * 78)
    print(f"  network fits: {a.pathb}")
    print(f"  n = {len(idx)}   cold = 4-stage schedule from the ring geometry")
    print("                 warm = one stage, all 14 free, from the network\n")

    cold, warm, rows = [], [], []
    for k, i in enumerate(idx):
        img = ds.image(i)
        sig = ds.sigma(i, img)
        rc = fit_from(img, sig, initial_vector(img, res), STAGES,
                      n_pix, res, psf, grid, mask, a.xtol)
        v0 = np.array([net[i][p] for p in PARAMS], float)
        rw = fit_from(img, sig, v0, [("all", PARAMS)],
                      n_pix, res, psf, grid, mask, a.xtol)
        for r in (rc, rw):
            r["index"] = i
            r["sigma"] = float(sig)
            r["path"] = str(ds.paths[i])
            r["target"] = "image"
        cold.append(rc)
        warm.append(rw)
        rows.append(rw)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(idx)}", flush=True)

    T = {i: ds.truth(i) for i in idx}
    med = lambda L, key: float(np.median([r[key] for r in L]))

    def rho(L, fk, tk):
        f = np.array([r[fk] for r in L], float)
        t = np.array([T[r["index"]][tk] for r in L], float)
        m = np.isfinite(f) & np.isfinite(t)
        return float(spearmanr(f[m], t[m]).statistic)

    net_rows = [net[i] for i in idx]
    print("\n1. COST\n")
    print(f"   {'method':>26}{'model evals':>13}{'LM seconds':>12}"
          f"{'total ms/img':>14}{'chi2/dof':>11}")
    print(f"   {'network alone':>26}{'-':>13}{'-':>12}"
          f"{a.net_ms:>14.1f}{med(net_rows,'chi2_per_dof'):>11.0f}")
    print(f"   {'cold LM (4 stages)':>26}{med(cold,'n_model_evals'):>13.0f}"
          f"{med(cold,'seconds'):>12.3f}{1000*med(cold,'seconds'):>14.1f}"
          f"{med(cold,'chi2_per_dof'):>11.0f}")
    print(f"   {'network + warm LM':>26}{med(warm,'n_model_evals'):>13.0f}"
          f"{med(warm,'seconds'):>12.3f}"
          f"{a.net_ms + 1000*med(warm,'seconds'):>14.1f}"
          f"{med(warm,'chi2_per_dof'):>11.0f}")
    sp = med(cold, 'seconds') / max(med(warm, 'seconds'), 1e-9)
    ev = med(cold, 'n_model_evals') / max(med(warm, 'n_model_evals'), 1e-9)
    print(f"\n   speedup {sp:.2f}x wall clock, {ev:.2f}x model evaluations,"
          f"   chi2 ratio {med(warm,'chi2_per_dof')/max(med(cold,'chi2_per_dof'),1e-9):.3f}")
    print("   A chi2 ratio near 1 means the speedup did not cost accuracy.")

    print("\n2. PARAMETER RECOVERY\n")
    print(f"   {'parameter':12s}{'network':>10}{'cold LM':>10}{'warm LM':>10}"
          f"{'truth':>10}   |   {'net rho':>9}{'cold rho':>10}{'warm rho':>10}")
    for fk, tk in (("theta_E", "theta_E"), ("beta", "beta"),
                   ("e", "host_e"), ("g", "gamma_ext"),
                   ("R_sersic", "source_R_sersic"),
                   ("n_sersic", "source_n_sersic"), ("gamma", "host_slope")):
        tv = float(np.median([T[i][tk] for i in idx]))
        print(f"   {fk:12s}{med(net_rows,fk):>10.4f}{med(cold,fk):>10.4f}"
              f"{med(warm,fk):>10.4f}{tv:>10.4f}   |   "
              f"{rho(net_rows,fk,tk):>+9.3f}{rho(cold,fk,tk):>+10.3f}"
              f"{rho(warm,fk,tk):>+10.3f}")

    print("\n3. SOURCE PLANE vs the npz `unlensed` array\n")
    X, Y = image_plane_grid(n_pix, res, 1)
    print(f"   {'method':>26}{'corr':>9}{'size_ratio':>12}{'nmse':>9}{'peak_ratio':>12}")
    for lbl, L in (("network alone", net_rows), ("cold LM", cold),
                   ("network + warm LM", warm)):
        st = []
        for r in L:
            s = convolve(SersicSource({k: r[k] for k in SRC_KEYS}).at(X, Y), psf)
            st.append(M.source_truth(s, ds.unlensed(r["index"]), res))
        gm = lambda k: float(np.nanmedian([x[k] for x in st]))
        print(f"   {lbl:>26}{gm('corr'):9.4f}{gm('size_ratio'):12.4f}"
              f"{gm('nmse'):9.4f}{gm('peak_ratio'):12.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cfg = dict(blob["config"])
    cfg.update(source="pathb+refine", n=len(rows), supersample=1,
               psf_mode="empirical", psf_path=a.psf_path)
    json.dump({"config": cfg, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    print("  score it with:")
    print(f"     python scripts/evaluate.py --fits {a.out} --root {a.root}")


if __name__ == "__main__":
    main()