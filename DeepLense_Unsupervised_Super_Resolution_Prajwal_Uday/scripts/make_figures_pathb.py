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

NAVY, AMBER, GOOD, BAD, GREY = "#1E2761", "#D98C1F", "#2E9E5B", "#C0392B", "#8891A5"

PAIRS = [("theta_E", "theta_E", "arcsec"), ("beta", "beta", "arcsec"),
         ("R_sersic", "source_R_sersic", "arcsec"),
         ("n_sersic", "source_n_sersic", ""), ("gamma", "host_slope", ""),
         ("e", "host_e", ""), ("g", "gamma_ext", "")]


def _stretch(a, p=99.5):
    a = np.asarray(a, float)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, p)
    return np.sqrt(np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1))


def figB1(ex, out, n_show=3):
    have = [i for i in range(n_show) if f"obs_{i}" in ex]
    if not have:
        print("  (no examples saved; skipping figB1)")
        return
    cols = ["observation", "network model", "residual / $\\sigma$",
            "source:\nSersic only", "source:\nSersic + correction", "TRUE source"]
    fig, ax = plt.subplots(len(have), 6, figsize=(19, 3.3 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        obs, pred = ex[f"obs_{i}"], ex[f"pred_{i}"]
        r = obs - pred
        s = max(np.std(r), 1e-12)
        panels = [obs, pred, None, ex[f"src_base_{i}"], ex[f"src_corr_{i}"],
                  ex[f"src_true_{i}"]]
        for c, im in enumerate(panels):
            if c == 2:
                h = ax[k, c].imshow(r / s, origin="lower", cmap="coolwarm",
                                    vmin=-4, vmax=4)
                plt.colorbar(h, ax=ax[k, c], fraction=0.046)
            else:
                ax[k, c].imshow(_stretch(im), origin="lower", cmap="magma")
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            if k == 0:
                ax[k, c].set_title(cols[c], fontsize=10, color=NAVY)
    fig.suptitle("Path B: one forward pass per image, no high-resolution target "
                 "anywhere\n"
                 "the only supervision is that the model, once lensed, blurred and "
                 "binned, must reproduce the observation",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figB2(ex, rows, out, n_show=3):
    have = [i for i in range(n_show) if f"corr_map_{i}" in ex]
    if not have:
        print("  (no correction maps saved; skipping figB2)")
        return
    fig, ax = plt.subplots(len(have), 3, figsize=(12, 3.4 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        C, cov = ex[f"corr_map_{i}"], ex[f"cov_{i}"]
        v = float(np.abs(C).max()) or 1.0
        h0 = ax[k, 0].imshow(C, origin="lower", cmap="coolwarm", vmin=-v, vmax=v)
        plt.colorbar(h0, ax=ax[k, 0], fraction=0.046)
        h1 = ax[k, 1].imshow(np.log10(np.maximum(cov, 1e-2)), origin="lower",
                             cmap="viridis")
        plt.colorbar(h1, ax=ax[k, 1], fraction=0.046)
        m = cov > 0
        ax[k, 2].scatter(np.log10(cov[m]), np.abs(C[m]), s=5, c=NAVY,
                         alpha=.35, edgecolors="none")
        rho = np.corrcoef(np.log10(cov[m]), np.abs(C[m]))[0, 1] if m.sum() > 10 else np.nan
        ax[k, 2].set_title(f"r = {rho:+.3f}", fontsize=9, color=NAVY)
        ax[k, 2].set_xlabel(r"$\log_{10}$ ray coverage ($=\mu$)")
        ax[k, 2].set_ylabel("|correction|")
        ax[k, 2].grid(alpha=.25)
        for c in (0, 1):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
        if k == 0:
            ax[k, 0].set_title("learned correction map", fontsize=10, color=NAVY)
            ax[k, 1].set_title(r"$\log_{10}$ ray coverage per source pixel"
                               "\n(this IS the magnification)", fontsize=10, color=NAVY)
    cv = np.array([r.get("corr_vs_mu", np.nan) for r in rows], float)
    fig.suptitle("Where the network added detail\n"
                 f"median corr(|correction|, log $\\mu$) = {np.nanmedian(cv):+.3f} "
                 "over all images -- positive means detail went where the lens "
                 "delivered resolution, not everywhere",
                 fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figB3(net_rows, fit_rows, truth, out):
    fitmap = {r["index"]: r for r in fit_rows} if fit_rows else {}
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.2))
    ax = ax.ravel()
    for j, (fk, tk, unit) in enumerate(PAIRS):
        t = np.array([truth[r["index"]][tk] for r in net_rows], float)
        n = np.array([r[fk] for r in net_rows], float)
        m = np.isfinite(t) & np.isfinite(n)
        lo, hi = np.nanpercentile(np.r_[t[m], n[m]], [1, 99])
        ax[j].scatter(t[m], n[m], s=9, c=NAVY, alpha=.45, edgecolors="none",
                      label=f"network  $\\rho$={spearmanr(t[m],n[m]).statistic:+.3f}")
        if fitmap:
            idx = [i for i, r in enumerate(net_rows) if r["index"] in fitmap]
            tf = np.array([truth[net_rows[i]["index"]][tk] for i in idx], float)
            ff = np.array([fitmap[net_rows[i]["index"]][fk] for i in idx], float)
            mm = np.isfinite(tf) & np.isfinite(ff)
            if mm.sum() > 3:
                ax[j].scatter(tf[mm], ff[mm], s=9, c=AMBER, alpha=.35,
                              edgecolors="none",
                              label=f"per-image fit  $\\rho$="
                                    f"{spearmanr(tf[mm],ff[mm]).statistic:+.3f}")
        ax[j].plot([lo, hi], [lo, hi], c=BAD, ls="--", lw=1.3)
        ax[j].set_xlim(lo, hi); ax[j].set_ylim(lo, hi)
        ax[j].set_xlabel(f"true {fk} {unit}"); ax[j].set_ylabel(f"predicted {fk}")
        ax[j].legend(fontsize=7, loc="upper left"); ax[j].grid(alpha=.25)
    ax[-1].axis("off")
    ax[-1].text(.02, .6,
                "Amortisation\n\n"
                "The network sees each image ONCE and emits every parameter in a\n"
                "single forward pass. The per-image fit runs Levenberg-Marquardt\n"
                "to convergence on that same image.\n\n"
                "If the two clouds overlap, the network has learned the inversion\n"
                "rather than memorising a population mean -- and it does so at a\n"
                "cost per image that makes a survey-sized run possible.",
                fontsize=9, color=NAVY, va="top", family="monospace")
    fig.suptitle("Parameter recovery: amortised network vs per-image optimiser",
                 fontsize=13, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figB4(hist, out):
    if not hist:
        print("  (no training history; skipping figB4)")
        return
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(ep, [h["train_chi2"] for h in hist], c=NAVY, lw=1.8, label="train")
    ax[0].plot(ep, [h["val_chi2"] for h in hist], c=AMBER, lw=1.8, label="val")
    on = [h["epoch"] for h in hist if h.get("correction_on")]
    if on:
        ax[0].axvline(on[0], c=GOOD, ls="--", lw=1.4, label="correction switched on")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(r"mean $\chi^2$ per pixel")
    ax[0].set_title("Fit to the data", fontsize=10, color=NAVY)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, which="both")

    ax[1].plot(ep, [h["train_reg"] for h in hist], c=BAD, lw=1.8)
    if on:
        ax[1].axvline(on[0], c=GOOD, ls="--", lw=1.4)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel(r"$\mu$-weighted correction penalty")
    ax[1].set_title("How hard the correction is pushing", fontsize=10, color=NAVY)
    ax[1].grid(alpha=.25)
    fig.suptitle("Path B training\n"
                 "the parametric part converges first; the correction is released "
                 "afterwards so it can only add what the Sersic missed",
                 fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pathb", default="results/fits_pathb.json")
    ap.add_argument("--npz", default="results/pathb_examples.npz")
    ap.add_argument("--fits", default="results/fits_img.json")
    ap.add_argument("--ckpt", default="results/pathb.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    blob = json.load(open(a.pathb))
    rows, cfg = blob["rows"], blob["config"]
    ex = dict(np.load(a.npz))
    fit_rows = json.load(open(a.fits))["rows"] if os.path.exists(a.fits) else []

    from data_a import ModelADataset
    ds = ModelADataset(a.root, split=cfg["split"], classes=cfg["classes"],
                       limit=max(r["index"] for r in rows) + 1)
    truth = {r["index"]: ds.truth(r["index"]) for r in rows}

    hist = []
    if os.path.exists(a.ckpt):
        try:
            import torch
            try:
                ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            except TypeError:              # torch < 2 has no weights_only=
                ck = torch.load(a.ckpt, map_location="cpu")
            hist = ck.get("history", [])
        except Exception as e:
            print(f"  (could not read history from {a.ckpt}: {e})")

    os.makedirs(a.outdir, exist_ok=True)
    print(f"figures from {a.pathb}   n = {len(rows)}\n")
    figB1(ex, f"{a.outdir}/figB1_reconstruction.png")
    figB2(ex, rows, f"{a.outdir}/figB2_correction.png")
    figB3(rows, fit_rows, truth, f"{a.outdir}/figB3_recovery.png")
    figB4(hist, f"{a.outdir}/figB4_training.png")
    print("\ndone.")


if __name__ == "__main__":
    main()