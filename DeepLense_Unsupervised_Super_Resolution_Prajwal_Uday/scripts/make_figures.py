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
from scipy.stats import spearmanr

from data_a import ModelADataset

NAVY, AMBER, GOOD, BAD, GREY = "#1E2761", "#D98C1F", "#2E9E5B", "#C0392B", "#8891A5"

PAIRS = [("theta_E", "theta_E", r"$\theta_E$  [arcsec]"),
         ("beta", "beta", r"$\beta$  [arcsec]"),
         ("R_sersic", "source_R_sersic", r"$R_{\rm Sersic}$  [arcsec]"),
         ("n_sersic", "source_n_sersic", r"$n_{\rm Sersic}$"),
         ("e", "host_e", r"lens $|e|$"),
         ("g", "gamma_ext", r"external shear $|\gamma|$"),
         ("gamma", "host_slope", r"slope $\gamma$")]


def _stretch(a, p=99.5):
    a = np.asarray(a, float)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, p)
    a = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.sqrt(a)


def fig1_superres(ex, out, n_show=3):
    rows = []
    for i in range(n_show):
        if f"obs_{i}" not in ex:
            break
        rows.append(i)
    fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
    if len(rows) == 1:
        ax = ax[None, :]
    for k, i in enumerate(rows):
        meta = ex[f"meta_{i}"]
        f_hi = int(meta[5])
        panels = [
            (ex[f"obs_{i}"], f"observation\n{ex[f'obs_{i}'].shape[0]}$^2$ px @ 0.106\"/px"),
            (ex[f"src_fit_hi_{i}"], f"RECONSTRUCTED source\n{f_hi}$\\times$ finer, "
                                    f"{ex[f'src_fit_hi_{i}'].shape[0]}$^2$ px"),
            (ex[f"src_true_hi_{i}"], f"TRUE source\nsame {f_hi}$\\times$ grid"),
            (ex[f"unlensed_{i}"], "stored `unlensed`\n(truth at 1$\\times$)"),
        ]
        for c, (im, ttl) in enumerate(panels):
            # zoom the source panels: the source occupies a small central region
            if c in (1, 2):
                n = im.shape[0]
                h = n // 6
                im = im[n // 2 - h:n // 2 + h, n // 2 - h:n // 2 + h]
            elif c == 3:
                n = im.shape[0]
                h = max(8, n // 6)
                im = im[n // 2 - h:n // 2 + h, n // 2 - h:n // 2 + h]
            ax[k, c].imshow(_stretch(im), origin="lower", cmap="magma")
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            if k == 0:
                ax[k, c].set_title(ttl, fontsize=10, color=NAVY)
        ax[k, 0].set_ylabel(f"$\\theta_E$={meta[0]:.2f}\"\nSNR={meta[4]:.0f}",
                            fontsize=9, color=NAVY)
    fig.suptitle("Super-resolution by inverting a calibrated physical model\n"
                 "the source is fitted as a continuous function, then sampled "
                 "on a finer grid than the detector",
                 fontsize=12, color=NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig2_recovery(rows, ds, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for k, (fk, tk, lbl) in enumerate(PAIRS):
        f = np.array([r[fk] for r in rows], float)
        t = np.array([ds.truth(r["index"])[tk] for r in rows], float)
        m = np.isfinite(f) & np.isfinite(t)
        rho = spearmanr(f[m], t[m]).statistic
        ax = axes[k]
        ax.scatter(t[m], f[m], s=11, alpha=.55, c=NAVY, edgecolors="none")
        lim = [min(t[m].min(), f[m].min()), max(t[m].max(), f[m].max())]
        pad = 0.05 * (lim[1] - lim[0] + 1e-9)
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, "--", c=BAD, lw=1.4, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true " + lbl, fontsize=9)
        ax.set_ylabel("recovered", fontsize=9)
        col = GOOD if rho > 0.7 else (AMBER if rho > 0.4 else GREY)
        ax.set_title(f"{lbl}    $\\rho$ = {rho:+.3f}", fontsize=10, color=col)
        ax.grid(alpha=.25)
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.72,
                  "Unsupervised.\nNothing on the x axis\nwas available to the fit.\n\n"
                  f"n = {len(rows)}\nall three DM classes\ntarget = `image`",
                  fontsize=12, color=NAVY, va="top")
    fig.suptitle("Parameter recovery from a single image, no labels",
                 fontsize=13, color=NAVY)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig3_sizeratio(rows, ds, out, old_metrics=None):
    from raytrace import convolve, image_plane_grid, load_psf
    from sources import SersicSource
    import metrics as M
    n_pix = ds.image(0).shape[-1]
    X, Y = image_plane_grid(n_pix, 0.10593, 1)
    psf = load_psf("results/psf_empirical.npy")
    new = []
    for r in rows:
        sp = {k: r[k] for k in ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")}
        s = convolve(SersicSource(sp).at(X, Y), psf)
        new.append(M.source_truth(s, ds.unlensed(r["index"]))["size_ratio"])
    new = np.array([v for v in new if np.isfinite(v)])

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if old_metrics:
        old = np.array([x.get("src_truth_size_ratio", np.nan)
                        for x in json.load(open(old_metrics))["per_image"]], float)
        old = old[np.isfinite(old)]
        ax.hist(old, bins=np.logspace(np.log10(0.5), np.log10(40), 45),
                color=BAD, alpha=.65, label=f"old SIS pipeline  (median {np.median(old):.1f})")
    ax.hist(new, bins=np.logspace(np.log10(0.5), np.log10(40), 45),
            color=GOOD, alpha=.75, label=f"this work  (median {np.median(new):.2f})")
    ax.axvline(1.0, ls="--", c="k", lw=1.6)
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "correct = 1", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("reconstructed source size / true source size")
    ax.set_ylabel("images")
    ax.set_title("The failure that is fixed", fontsize=12, color=NAVY)
    ax.legend(fontsize=9); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def fig4_residuals(ex, out, n_show=4):
    rows = [i for i in range(n_show) if f"obs_{i}" in ex]
    fig, ax = plt.subplots(len(rows), 3, figsize=(9.5, 3.2 * len(rows)))
    if len(rows) == 1:
        ax = ax[None, :]
    for k, i in enumerate(rows):
        d = ex[f"obs_{i}"]; m = ex[f"model_{i}"]; r = d - m
        for c, (im, ttl, cm) in enumerate([(d, "data", "magma"),
                                           (m, "model", "magma"),
                                           (r, "residual", "coolwarm")]):
            if c == 2:
                v = np.percentile(np.abs(im), 99)
                ax[k, c].imshow(im, origin="lower", cmap=cm, vmin=-v, vmax=v)
            else:
                ax[k, c].imshow(_stretch(im), origin="lower", cmap=cm)
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            if k == 0:
                ax[k, c].set_title(ttl, fontsize=11, color=NAVY)
    fig.suptitle("Data, fitted model, and residual\n"
                 "the residual carries the dark-matter substructure the smooth "
                 "model cannot represent", fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig5_magnification(ex, out, n_show=3):
    rows = [i for i in range(n_show) if f"mu_{i}" in ex]
    fig, ax = plt.subplots(len(rows), 2, figsize=(8, 3.4 * len(rows)))
    if len(rows) == 1:
        ax = ax[None, :]
    for k, i in enumerate(rows):
        ax[k, 0].imshow(_stretch(ex[f"obs_{i}"]), origin="lower", cmap="magma")
        im = ax[k, 1].imshow(np.log10(np.clip(ex[f"mu_{i}"], 1, None)),
                             origin="lower", cmap="viridis")
        for c in (0, 1):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
        if k == 0:
            ax[k, 0].set_title("observation", fontsize=11, color=NAVY)
            ax[k, 1].set_title(r"$\log_{10}|\mu|$ from the fitted lens",
                               fontsize=11, color=NAVY)
        plt.colorbar(im, ax=ax[k, 1], fraction=0.046)
    fig.suptitle("Where the resolution comes from\n"
                 "the lens spreads a small patch of source over many detector "
                 "pixels; surface brightness is conserved", fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--fits", default="results/fits_img.json")
    ap.add_argument("--npz", default="results/superres_examples.npz")
    ap.add_argument("--old-metrics", default="eval_man_tv03/metrics.json",
                    help="the old SIS run, for the before/after histogram")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    blob = json.load(open(a.fits))
    rows, cfg = blob["rows"], blob["config"]
    ds = ModelADataset(a.root, split=cfg.get("split", "val"),
                       classes=cfg.get("classes", ["axion"]), limit=len(rows))
    ex = dict(np.load(a.npz))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"figures from {a.fits}  (n = {len(rows)})\n")

    fig1_superres(ex, f"{a.outdir}/fig1_superres.png")
    fig2_recovery(rows, ds, f"{a.outdir}/fig2_recovery.png")
    old = a.old_metrics if os.path.exists(a.old_metrics) else None
    if old is None:
        print(f"  (no {a.old_metrics}; fig3 will show only the new distribution)")
    fig3_sizeratio(rows, ds, f"{a.outdir}/fig3_sizeratio.png", old)
    fig4_residuals(ex, f"{a.outdir}/fig4_residuals.png")
    fig5_magnification(ex, f"{a.outdir}/fig5_magnification.png")
    print("\ndone.")


if __name__ == "__main__":
    main()