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
from lens_models import hessian_analytic, magnification
from raytrace import image_plane_grid, render
from sources import SersicSource

LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2")
SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


def fitted_lens(row):
    p = {k: row[k] for k in LENS_KEYS}
    p["cx"] = p["cy"] = 0.0
    return p


def true_lens(t):
    return dict(theta_E=t["theta_E"], gamma=t["host_slope"],
                e1=t["host_e1"], e2=t["host_e2"],
                g1=t["gamma1_ext"], g2=t["gamma2_ext"], cx=0.0, cy=0.0)


def true_source(t):
    return dict(amp=1.0, R_sersic=t["source_R_sersic"],
                n_sersic=t["source_n_sersic"], se1=t["source_e1"],
                se2=t["source_e2"], sx=t["source_x"], sy=t["source_y"])


def corr(a, b, m=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if m is not None:
        a, b = a[m], b[m]
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def total_magnification(lens, src, n_pix, pixel_scale, supersample=3):
    X, Y = image_plane_grid(n_pix, pixel_scale, supersample)
    S = SersicSource(src)
    lensed = render(S, lens, n_pix, pixel_scale, psf=None,
                    supersample=supersample, grid=(X, Y))
    unlensed = S.at(X, Y)
    unlensed = unlensed.reshape(n_pix, supersample, n_pix, supersample).mean((1, 3))
    f_l, f_u = float(lensed.sum()), float(unlensed.sum())
    return f_l / max(f_u, 1e-30)


def critical_radius(lens, n_phi=72, r_lo=0.3, r_hi=2.6, n_r=400):
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r = np.linspace(r_lo * lens["theta_E"], r_hi * lens["theta_E"], n_r)
    R, P = np.meshgrid(r, phi)
    k, g1, g2 = hessian_analytic(R * np.cos(P), R * np.sin(P), lens)
    lam_t = 1.0 - k - np.hypot(g1, g2)
    out = np.full(n_phi, np.nan)
    for i in range(n_phi):
        s = np.sign(lam_t[i])
        j = np.nonzero(np.diff(s) != 0)[0]
        if j.size:
            a, b = lam_t[i, j[0]], lam_t[i, j[0] + 1]
            out[i] = r[j[0]] + (r[j[0] + 1] - r[j[0]]) * (-a) / (b - a)
    return phi, out


def resolution_gain(lens, X, Y, mask):
    k, g1, g2 = hessian_analytic(X, Y, lens)
    g = np.hypot(g1, g2)
    lt = np.abs(1.0 - k - g)
    lr = np.abs(1.0 - k + g)
    return (float(np.median(1.0 / np.maximum(lt[mask], 1e-3))),
            float(np.median(1.0 / np.maximum(lr[mask], 1e-3))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", default="results/fits_img.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=0, help="0 = all rows in --fits")
    ap.add_argument("--mu-clip", type=float, default=50.0)
    ap.add_argument("--supersample", type=int, default=3)
    ap.add_argument("--n-save", type=int, default=6)
    ap.add_argument("--out", default="results/magnification.json")
    ap.add_argument("--out-npz", default="results/magnification_examples.npz")
    a = ap.parse_args()

    blob = json.load(open(a.fits))
    cfg, rows = blob["config"], blob["rows"]
    if a.n:
        rows = rows[:a.n]
    res = cfg.get("pixel_scale", 0.10593)
    ds = ModelADataset(a.root, split=cfg.get("split", "val"),
                       classes=cfg.get("classes", ["axion"]), limit=len(rows))
    ds.assert_rows_match(rows, a.fits)
    n_pix = ds.image(0).shape[-1]
    X, Y = image_plane_grid(n_pix, res, 1)
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    rpx = np.hypot(yy - c, xx - c)

    print("=" * 78)
    print("MAGNIFICATION EXTRACTION FROM THE FITTED LENS")
    print("=" * 78)
    print(f"  fits : {a.fits}   n = {len(rows)}")
    print("  mu is derived from the lens parameters, so this checks the fitted")
    print("  lens; it is not an independent measurement of mu. Surface")
    print("  brightness is conserved throughout; mu never multiplies an")
    print("  intensity.\n")

    recs, examples = [], {}
    for i, r in enumerate(rows):
        t = ds.truth(r["index"])
        lf, lt_ = fitted_lens(r), true_lens(t)

        #1. magnification map, over the arc annulus
        tE_px = t["theta_E"] / res
        ring = (rpx > 0.45 * tE_px) & (rpx < 1.9 * tE_px)
        mu_f = magnification(X, Y, lf, mu_clip=a.mu_clip)
        mu_t = magnification(X, Y, lt_, mu_clip=a.mu_clip)
        rec = {"index": r["index"],
               "mu_map_corr": corr(np.log10(mu_f), np.log10(mu_t), ring),
               "mu_ring_fit": float(np.median(mu_f[ring])),
               "mu_ring_true": float(np.median(mu_t[ring]))}

        # 2. total magnification
        rec["mu_tot_fit"] = total_magnification(
            lf, {k: r[k] for k in SRC_KEYS}, n_pix, res, a.supersample)
        rec["mu_tot_true"] = total_magnification(
            lt_, true_source(t), n_pix, res, a.supersample)

        # 3. critical curve
        phi, rc_f = critical_radius(lf)
        _, rc_t = critical_radius(lt_)
        ok = np.isfinite(rc_f) & np.isfinite(rc_t)
        if ok.sum() > 10:
            rec["rcrit_mean_fit"] = float(np.mean(rc_f[ok]))
            rec["rcrit_mean_true"] = float(np.mean(rc_t[ok]))
            rec["rcrit_rms_err_arcsec"] = float(np.sqrt(np.mean((rc_f[ok] - rc_t[ok]) ** 2)))
            rec["rcrit_swing_fit"] = float(np.ptp(rc_f[ok]))
            rec["rcrit_swing_true"] = float(np.ptp(rc_t[ok]))

        #4. resolution gain
        gt_f, gr_f = resolution_gain(lf, X, Y, ring)
        gt_t, gr_t = resolution_gain(lt_, X, Y, ring)
        rec.update(gain_tangential_fit=gt_f, gain_radial_fit=gr_f,
                   gain_tangential_true=gt_t, gain_radial_true=gr_t,
                   snr_max=t["snr_max"])
        recs.append(rec)

        if i < a.n_save:
            examples[f"obs_{i}"] = ds.image(r["index"]).astype(np.float32)
            examples[f"mu_fit_{i}"] = mu_f.astype(np.float32)
            examples[f"mu_true_{i}"] = mu_t.astype(np.float32)
            examples[f"rcrit_{i}"] = np.column_stack([phi, rc_f, rc_t]).astype(np.float32)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    g = lambda k: np.array([x.get(k, np.nan) for x in recs], float)

    def line(lbl, f, t, unit=""):
        f, t = g(f), g(t)
        m = np.isfinite(f) & np.isfinite(t)
        from scipy.stats import spearmanr
        rho = spearmanr(f[m], t[m]).statistic if m.sum() > 3 else np.nan
        rel = np.abs(f[m] - t[m]) / np.maximum(np.abs(t[m]), 1e-9)
        print(f"   {lbl:28s}{np.median(f[m]):>10.3f}{np.median(t[m]):>10.3f}"
              f"{np.median(rel)*100:>11.1f}%{rho:>+10.3f}   {unit}")

    print("\n1-2. MAGNIFICATION, fitted lens vs true lens\n")
    print(f"   {'quantity':28s}{'fitted':>10}{'true':>10}{'med |rel err|':>12}{'Spearman':>10}")
    line("mu on the arc annulus", "mu_ring_fit", "mu_ring_true")
    line("TOTAL magnification mu_tot", "mu_tot_fit", "mu_tot_true")
    mm = g("mu_map_corr")
    print(f"\n   per-pixel corr(log mu_fit, log mu_true) over the arc:"
          f"  median {np.nanmedian(mm):.4f}   p10 {np.nanpercentile(mm,10):.4f}")

    print("\n3. CRITICAL CURVE (the Einstein ring)\n")
    print(f" {'quantity':28s}{'fitted':>10}{'true':>10}{'med |rel err|':>12}{'Spearman':>10}")
    line("mean radius", "rcrit_mean_fit", "rcrit_mean_true", "arcsec")
    line("azimuthal swing (p-p)", "rcrit_swing_fit", "rcrit_swing_true", "arcsec")
    rr = g("rcrit_rms_err_arcsec")
    print(f"\n rms radial error of the recovered critical curve:"
          f"median {np.nanmedian(rr):.4f} arcsec = {np.nanmedian(rr)/res:.2f} px")
    print(" A circular lens has swing 0, so a recovered swing means e1, e2, g1")
    print(" and g2 were recovered jointly.")

    print("\n4. RESOLUTION GAIN\n")
    print(f"   {'quantity':28s}{'fitted':>10}{'true':>10}{'med |rel err|':>12}{'Spearman':>10}")
    line("tangential stretch 1/|lam_t|", "gain_tangential_fit", "gain_tangential_true")
    line("radial stretch 1/|lam_r|", "gain_radial_fit", "gain_radial_true")
    gt = np.nanmedian(g("gain_tangential_fit"))
    print(f"\n The lens spreads a source patch over ~{gt:.1f}x more detector")
    print(" pixels tangentially, so the sky has already oversampled the source")
    print(f" by that factor. A source grid up to ~{gt:.1f}x finer than the")
    print(" detector is supported by the data.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    summary = {k: float(np.nanmedian(g(k))) for k in
               ("mu_ring_fit", "mu_ring_true", "mu_tot_fit", "mu_tot_true",
                "mu_map_corr", "rcrit_rms_err_arcsec", "rcrit_swing_fit",
                "rcrit_swing_true", "gain_tangential_fit", "gain_radial_fit")}
    json.dump({"config": vars(a), "summary": summary, "rows": recs},
              open(a.out, "w"), indent=1)
    np.savez_compressed(a.out_npz, **examples)
    print(f"\n  wrote {a.out}")
    print(f"  wrote {a.out_npz}")


if __name__ == "__main__":
    main()