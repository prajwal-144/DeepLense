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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, AMBER, GOOD, BAD, GREY = "#1E2761", "#D98C1F", "#2E9E5B", "#C0392B", "#8891A5"


def _stretch(a, p=99.5):
    a = np.asarray(a, float)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, p)
    return np.sqrt(np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1))


def figM1(ex, out, n_show=3):
    have = [i for i in range(n_show) if f"mu_fit_{i}" in ex]
    if not have:
        print("  (no mu maps saved; skipping figM1)")
        return
    cols = ["observation", r"$\log_{10}\ \mu$   fitted lens",
            r"$\log_{10}\ \mu$   TRUE lens", "difference"]
    fig, ax = plt.subplots(len(have), 4, figsize=(14, 3.4 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        mf, mt = ex[f"mu_fit_{i}"], ex[f"mu_true_{i}"]
        lf, lt = np.log10(np.maximum(mf, 1e-3)), np.log10(np.maximum(mt, 1e-3))
        vmax = max(np.percentile(lf, 99.5), np.percentile(lt, 99.5))
        ax[k, 0].imshow(_stretch(ex[f"obs_{i}"]), origin="lower", cmap="magma")
        for c, L in ((1, lf), (2, lt)):
            im = ax[k, c].imshow(L, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            # the critical curve: mu diverges there, so it is the ridge of log mu
            ax[k, c].contour(L, levels=[0.9 * vmax], colors="w", linewidths=0.9)
            plt.colorbar(im, ax=ax[k, c], fraction=0.046)
        dm = ax[k, 3].imshow(lf - lt, origin="lower", cmap="coolwarm",
                             vmin=-0.5, vmax=0.5)
        plt.colorbar(dm, ax=ax[k, 3], fraction=0.046)
        for c in range(4):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            if k == 0:
                ax[k, c].set_title(cols[c], fontsize=10, color=NAVY)
    fig.suptitle("Magnification field from the FITTED lens vs the TRUE lens\n"
                 r"$\mu = 1/[(1-\kappa)^2 - |\gamma|^2]$; the white contour is the "
                 "critical curve. Surface brightness is conserved -- "
                 r"$\mu$ never multiplies an intensity.",
                 fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figM2(ex, out, n_show=4):
    have = [i for i in range(n_show) if f"rcrit_{i}" in ex]
    if not have:
        print("  (no critical curves saved; skipping figM2)")
        return
    fig, ax = plt.subplots(1, len(have), figsize=(3.8 * len(have), 3.6),
                           subplot_kw=dict(polar=True))
    if len(have) == 1:
        ax = [ax]
    for k, i in enumerate(have):
        T = ex[f"rcrit_{i}"]
        phi, rf, rt = T[:, 0], T[:, 1], T[:, 2]
        cl = lambda p, r: (np.append(p, p[0]), np.append(r, r[0]))
        ax[k].plot(*cl(phi, rt), c=NAVY, lw=2.2, label="true lens")
        ax[k].plot(*cl(phi, rf), c=AMBER, lw=1.8, ls="--", label="fitted lens")
        ax[k].set_title(f"image {i}", fontsize=10, color=NAVY, pad=14)
        ax[k].tick_params(labelsize=7)
        if k == 0:
            ax[k].legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.25, -0.18))
    fig.suptitle("The tangential critical curve, per azimuth\n"
                 "a circular lens would be a perfect circle -- the departure from "
                 "one is what tests $e_1,e_2,g_1,g_2$ jointly",
                 fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def _scatter(ax, f, t, label, unit=""):
    m = np.isfinite(f) & np.isfinite(t)
    f, t = f[m], t[m]
    ax.scatter(t, f, s=9, c=NAVY, alpha=.45, edgecolors="none")
    lo, hi = np.nanpercentile(np.r_[f, t], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], c=BAD, lw=1.4, ls="--", label="perfect")
    from scipy.stats import spearmanr
    rho = spearmanr(f, t).statistic
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true {label} {unit}"); ax.set_ylabel(f"fitted {label} {unit}")
    ax.set_title(rf"$\rho$ = {rho:+.3f}   (n = {len(f)})", fontsize=10, color=NAVY)
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    return rho


def figM3(rows, out):
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    _scatter(ax[0], g("mu_tot_fit"), g("mu_tot_true"), r"total magnification $\mu_{tot}$")
    _scatter(ax[1], g("rcrit_swing_fit"), g("rcrit_swing_true"),
             "critical-curve swing", "[arcsec]")
    ax[0].text(.03, .95, "converts observed flux\ninto INTRINSIC luminosity",
               transform=ax[0].transAxes, va="top", fontsize=8, color=GREY)
    ax[1].text(.03, .95, "zero for a circular lens,\nso this is the shape test",
               transform=ax[1].transAxes, va="top", fontsize=8, color=GREY)
    fig.suptitle("Magnification recovered from the fitted lens, against truth",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figM4(rows, out):
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    gt, gr = g("gain_tangential_fit"), g("gain_radial_fit")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(gt[np.isfinite(gt)], bins=40, color=NAVY, alpha=.85,
               label="tangential  $1/|1-\\kappa-|\\gamma||$")
    ax[0].hist(gr[np.isfinite(gr)], bins=40, color=AMBER, alpha=.65,
               label="radial  $1/|1-\\kappa+|\\gamma||$")
    ax[0].axvline(1.0, c=BAD, ls="--", lw=1.4, label="no gain")
    ax[0].set_xlabel("stretch factor over the arc region")
    ax[0].set_ylabel("images")
    ax[0].set_title("How much the lens stretches the source", fontsize=10, color=NAVY)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

    m = np.isfinite(gt) & np.isfinite(g("gain_tangential_true"))
    ax[1].scatter(g("gain_tangential_true")[m], gt[m], s=9, c=NAVY, alpha=.45,
                  edgecolors="none")
    lo, hi = np.nanpercentile(gt[m], [1, 99])
    ax[1].plot([lo, hi], [lo, hi], c=BAD, ls="--", lw=1.4)
    ax[1].set_xlabel("true tangential stretch"); ax[1].set_ylabel("fitted")
    ax[1].set_title("recovered from the image alone", fontsize=10, color=NAVY)
    ax[1].grid(alpha=.25)

    med = float(np.nanmedian(gt))
    fig.suptitle("Why super-resolution is possible at all\n"
                 f"the lens spreads a source patch over ~{med:.1f}x more detector "
                 "pixels tangentially, so the sky has already oversampled the "
                 "source by that factor",
                 fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mag", default="results/magnification.json")
    ap.add_argument("--npz", default="results/magnification_examples.npz")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    rows = json.load(open(a.mag))["rows"]
    ex = dict(np.load(a.npz))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"figures from {a.mag}   n = {len(rows)}\n")

    figM1(ex, f"{a.outdir}/figM1_mu_maps.png")
    figM2(ex, f"{a.outdir}/figM2_critical.png")
    figM3(rows, f"{a.outdir}/figM3_scatter.png")
    figM4(rows, f"{a.outdir}/figM4_resolution.png")
    print("\ndone.")


if __name__ == "__main__":
    main()