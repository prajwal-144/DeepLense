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

import metrics as M
from data_a import ModelADataset
from raytrace import convolve, image_plane_grid, load_psf
from sources import SersicSource

NAVY, AMBER, GOOD, BAD, GREY = "#1E2761", "#D98C1F", "#2E9E5B", "#C0392B", "#8891A5"
SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


def figR1(A, W, out):
    ea = np.array([r["n_model_evals"] for r in A], float)
    ew = np.array([r["n_model_evals"] for r in W], float)
    ta = np.array([r["seconds"] for r in A], float)
    tw = np.array([r["seconds"] for r in W], float)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    lo, hi = 50, max(ea.max(), ew.max()) * 1.1
    ax[0].scatter(ea, ew, s=8, c=NAVY, alpha=.35, edgecolors="none")
    ax[0].plot([lo, hi], [lo, hi], c=BAD, ls="--", lw=1.4, label="equal cost")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("cold start: model evaluations (4-stage schedule)")
    ax[0].set_ylabel("warm start: model evaluations (1 stage)")
    ax[0].set_title(f"median {np.median(ea):.0f} -> {np.median(ew):.0f}"
                    f"   ({np.median(ea)/np.median(ew):.2f}x fewer)",
                    fontsize=11, color=NAVY)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, which="both")
    ax[0].text(.04, .95, "below the line = cheaper", transform=ax[0].transAxes,
               va="top", fontsize=8, color=GREY)

    ratio = ea / np.maximum(ew, 1)
    ax[1].hist(ratio, bins=np.logspace(np.log10(0.3), np.log10(30), 45),
               color=NAVY, alpha=.85)
    ax[1].axvline(1.0, c=BAD, ls="--", lw=1.4, label="no saving")
    ax[1].axvline(np.median(ratio), c=AMBER, lw=2.0,
                  label=f"median {np.median(ratio):.2f}x")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("cold evaluations / warm evaluations")
    ax[1].set_ylabel("images")
    ax[1].set_title(f"cheaper on {100*(ratio>1).mean():.0f}% of images",
                    fontsize=11, color=NAVY)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)

    fig.suptitle("Cost of the per-image fit, with and without the network as "
                 "its starting point\n"
                 f"end-to-end wall clock including {12} ms of inference: "
                 f"{1000*np.median(ta):.0f} ms -> {12 + 1000*np.median(tw):.0f} ms",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figR2(A, W, out):
    ca = np.array([r["chi2_per_dof"] for r in A], float)
    cw = np.array([r["chi2_per_dof"] for r in W], float)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    lo, hi = min(ca.min(), cw.min()) * 0.7, max(ca.max(), cw.max()) * 1.3
    better = cw < ca
    ax[0].scatter(ca[better], cw[better], s=9, c=GOOD, alpha=.45,
                  edgecolors="none", label=f"warm better ({100*better.mean():.0f}%)")
    ax[0].scatter(ca[~better], cw[~better], s=9, c=BAD, alpha=.45,
                  edgecolors="none", label=f"cold better ({100*(~better).mean():.0f}%)")
    ax[0].plot([lo, hi], [lo, hi], c="k", ls="--", lw=1.2)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel(r"cold start  $\chi^2$/dof")
    ax[0].set_ylabel(r"warm start  $\chi^2$/dof")
    ax[0].set_title("below the line = the warm start found a better minimum",
                    fontsize=10, color=NAVY)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, which="both")

    r = cw / ca
    ax[1].hist(np.log10(r), bins=50, color=NAVY, alpha=.85)
    ax[1].axvline(0.0, c=BAD, ls="--", lw=1.4)
    ax[1].axvline(np.log10(np.median(r)), c=AMBER, lw=2.0,
                  label=f"median {np.median(r):.3f}")
    ax[1].set_xlabel(r"$\log_{10}(\chi^2_{\rm warm} / \chi^2_{\rm cold})$")
    ax[1].set_ylabel("images")
    ax[1].set_title("negative = warm is a better fit", fontsize=10, color=NAVY)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
    fig.suptitle("Refining from the network does not trade accuracy for speed",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figR3(N, A, W, truth, out):
    t = np.array([truth[r["index"]]["host_e"] for r in W], float)
    sets = [("network alone\n(amortised, 12 ms)", N, AMBER),
            ("cold LM\n(4 stages, 815 ms)", A, GREY),
            ("network + warm LM\n(1 stage, 472 ms)", W, NAVY)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    hi = float(np.percentile(t, 99.5)) * 1.15
    for k, (lbl, L, col) in enumerate(sets):
        f = np.array([r["e"] for r in L], float)
        ax[k].scatter(t, f, s=9, c=col, alpha=.4, edgecolors="none")
        ax[k].plot([0, hi], [0, hi], c=BAD, ls="--", lw=1.4)
        ax[k].set_xlim(0, hi); ax[k].set_ylim(0, hi)
        ax[k].set_xlabel(r"true lens $|e|$")
        if k == 0:
            ax[k].set_ylabel(r"recovered $|e|$")
        rho = spearmanr(f, t).statistic
        slope = float(np.median(f) / max(np.median(t), 1e-9))
        ax[k].set_title(f"{lbl}\n" + rf"$\rho$ = {rho:+.3f}   median ratio {slope:.2f}",
                        fontsize=10, color=NAVY)
        ax[k].grid(alpha=.25)
    ax[0].text(.04, .95,
               "shrunk ~4x:\na squared loss is a\nconditional-MEAN estimator,\n"
               "so it hedges on a spin-2\nquantity it is unsure of",
               transform=ax[0].transAxes, va="top", fontsize=8, color=BAD)
    ax[2].text(.04, .95, "shrinkage undone by\n~10 LM iterations",
               transform=ax[2].transAxes, va="top", fontsize=8, color=GOOD)
    fig.suptitle("Lens ellipticity: what the network gets wrong, and why it "
                 "still helps", fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figR4(sets, truth, ds, res, psf, n_pix, out):
    PAIRS = [("theta_E", "theta_E"), ("beta", "beta"),
             ("R_sersic", "source_R_sersic"), ("n_sersic", "source_n_sersic"),
             ("gamma", "host_slope"), ("e", "host_e"), ("g", "gamma_ext")]
    X, Y = image_plane_grid(n_pix, res, 1)
    labels = [s[0] for s in sets]
    cols = [AMBER, GREY, NAVY]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    w = 0.8 / len(sets)
    for j, (lbl, L, _) in enumerate(sets):
        vals = []
        for fk, tk in PAIRS:
            f = np.array([r[fk] for r in L], float)
            t = np.array([truth[r["index"]][tk] for r in L], float)
            m = np.isfinite(f) & np.isfinite(t)
            vals.append(spearmanr(f[m], t[m]).statistic)
        ax[0].bar(np.arange(len(PAIRS)) + j * w, vals, w, color=cols[j],
                  label=lbl, alpha=.9)
    ax[0].set_xticks(np.arange(len(PAIRS)) + 0.4 - w / 2)
    ax[0].set_xticklabels([p[0] for p in PAIRS], rotation=30, fontsize=9)
    ax[0].set_ylabel("Spearman vs truth"); ax[0].set_ylim(-0.1, 1.05)
    ax[0].axhline(0, c="k", lw=0.8)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, axis="y")
    ax[0].set_title("Parameter recovery", fontsize=11, color=NAVY)

    keys = ["corr", "size_ratio", "peak_ratio", "nmse x10"]
    for j, (lbl, L, _) in enumerate(sets):
        st = []
        for r in L:
            s = convolve(SersicSource({k: r[k] for k in SRC_KEYS}).at(X, Y), psf)
            st.append(M.source_truth(s, ds.unlensed(r["index"]), res))
        gm = lambda k: float(np.nanmedian([x[k] for x in st]))
        vals = [gm("corr"), gm("size_ratio"), gm("peak_ratio"), 10 * gm("nmse")]
        ax[1].bar(np.arange(4) + j * w, vals, w, color=cols[j], label=lbl, alpha=.9)
    ax[1].axhline(1.0, c=BAD, ls="--", lw=1.2, label="target (except nmse)")
    ax[1].set_xticks(np.arange(4) + 0.4 - w / 2)
    ax[1].set_xticklabels(keys, fontsize=9)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25, axis="y")
    ax[1].set_title("Source plane vs the true `unlensed` array (nmse x10)",
                    fontsize=11, color=NAVY)
    fig.suptitle("Amortised network, per-image optimiser, and the two combined",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refined", default="results/fits_refined_mu.json")
    ap.add_argument("--network", default="results/fits_pb3_mu.json")
    ap.add_argument("--cold", default="results/fits_img.json")
    ap.add_argument("--psf-path", default="results/psf_empirical.npy")
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    blob = json.load(open(a.refined))
    W = blob["rows"]
    idx = [r["index"] for r in W]
    A = [{**{k: v for k, v in r.items()}} for r in
         [dict(x) for x in json.load(open(a.cold))["rows"]] if r["index"] in set(idx)]
    A = sorted(A, key=lambda r: r["index"])
    nb = {r["index"]: r for r in json.load(open(a.network))["rows"]}
    N = [nb[i] for i in idx]
    W = sorted(W, key=lambda r: r["index"])

    res = blob["config"].get("pixel_scale", 0.10593)
    ds = ModelADataset(a.root, split=blob["config"]["split"],
                       classes=blob["config"]["classes"], limit=max(idx) + 1)
    ds.assert_rows_match(W, a.refined)
    truth = {i: ds.truth(i) for i in idx}
    n_pix = ds.image(0).shape[-1]
    psf = load_psf(a.psf_path)

    os.makedirs(a.outdir, exist_ok=True)
    print(f"figures from {a.refined}   n = {len(W)}\n")
    figR1(A, W, f"{a.outdir}/figR1_cost.png")
    figR2(A, W, f"{a.outdir}/figR2_chi2.png")
    figR3(N, A, W, truth, f"{a.outdir}/figR3_ellipticity.png")
    figR4([("network alone", N, AMBER), ("cold LM (Path A)", A, GREY),
           ("network + warm LM", W, NAVY)],
          truth, ds, res, psf, n_pix, f"{a.outdir}/figR4_summary.png")
    print("\ndone.")


if __name__ == "__main__":
    main()