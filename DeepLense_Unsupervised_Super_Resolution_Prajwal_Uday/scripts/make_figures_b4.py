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


def _st(a, p=99.5):
    a = np.asarray(a, float)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, p)
    return np.sqrt(np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1))


def _crop(a, H, res):
    n = a.shape[-1]
    c = (n - 1) // 2
    h = int(round(H / res))
    return a[c - h:c + h + 1, c - h:c + h + 1]


def figS1(ex, out, H=1.2, res=0.10593, n_show=3):
    have = [i for i in range(n_show) if f"src_b4_{i}" in ex]
    if not have:
        print("  (no examples; skipping figS1)"); return
    cols = ["observation", "back-projection\n(network INPUT)",
            r"$\log_{10}$ coverage" "\n(= magnification)",
            "B4 source", "Sersic source", "TRUE source"]
    fig, ax = plt.subplots(len(have), 6, figsize=(19, 3.3 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        panels = [ex[f"obs_{i}"], ex[f"bp_{i}"], None,
                  _crop(ex[f"src_b4_{i}"], H, res),
                  _crop(ex[f"src_par_{i}"], H, res),
                  _crop(ex[f"src_true_{i}"], H, res)]
        for c, im in enumerate(panels):
            if c == 2:
                h = ax[k, c].imshow(np.log10(np.maximum(ex[f"cov_{i}"], 1e-2)),
                                    origin="lower", cmap="viridis")
                plt.colorbar(h, ax=ax[k, c], fraction=0.046)
            else:
                ax[k, c].imshow(_st(im), origin="lower", cmap="magma")
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            if k == 0:
                ax[k, c].set_title(cols[c], fontsize=10, color=NAVY)
    fig.suptitle("B4: the image is ray-shot back into the source plane, a fully "
                 "convolutional SISR sharpens it,\nand the result is lensed "
                 "forward again and compared with the observation. "
                 "No high-resolution target anywhere.",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figS2(ex, out, n_show=3):
    have = [i for i in range(n_show) if f"pred_{i}" in ex]
    if not have:
        print("  (skipping figS2)"); return
    fig, ax = plt.subplots(len(have), 3, figsize=(11, 3.4 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        obs, pred = ex[f"obs_{i}"], ex[f"pred_{i}"]
        r = obs - pred
        s = max(np.std(r), 1e-12)
        ax[k, 0].imshow(_st(obs), origin="lower", cmap="magma")
        ax[k, 1].imshow(_st(pred), origin="lower", cmap="magma")
        h = ax[k, 2].imshow(r / s, origin="lower", cmap="coolwarm", vmin=-4, vmax=4)
        plt.colorbar(h, ax=ax[k, 2], fraction=0.046)
        for c in range(3):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
        if k == 0:
            for c, t in enumerate(["observation", "B4 forward model",
                                   r"residual / $\sigma$"]):
                ax[k, c].set_title(t, fontsize=10, color=NAVY)
    fig.suptitle("The self-supervision loop closing", fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figS3(blob, out):
    b4, par, rows = blob["b4"], blob["parametric"], blob["rows"]
    keys = ["corr", "size_ratio", "peak_ratio", "nmse x10", "centroid px"]
    v_b4 = [b4["corr"], b4["size_ratio"], b4["peak_ratio"],
            10 * b4["nmse"], b4["centroid_err_px"]]
    v_pa = [par["corr"], par["size_ratio"], par["peak_ratio"],
            10 * par["nmse"], par["centroid_err_px"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(keys))
    ax[0].bar(x - 0.2, v_pa, 0.4, color=GREY, label="Sersic (7 params)")
    ax[0].bar(x + 0.2, v_b4, 0.4, color=NAVY, label="B4 (free-form SISR)")
    ax[0].axhline(1.0, c=BAD, ls="--", lw=1.2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(keys, rotation=20, fontsize=9)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, axis="y")
    ax[0].set_title("Source-plane metrics (target = 1 except nmse, centroid)",
                    fontsize=11, color=NAVY)

    fb = np.array([r["src_flux_in_box"] for r in rows], float)
    ax[1].hist(np.clip(fb, 0, 3), bins=60, color=NAVY, alpha=.85)
    ax[1].axvline(1.0, c=BAD, ls="--", lw=1.4, label="all the flux")
    ax[1].axvline(0.89, c=GOOD, ls="--", lw=1.4,
                  label="ceiling for a 1.2 arcsec box")
    ax[1].axvline(np.median(fb), c=AMBER, lw=2.0,
                  label=f"median {np.median(fb):.2f}")
    ax[1].set_xlabel("recovered source flux / true source flux")
    ax[1].set_ylabel("images")
    ax[1].set_title("Is the free-form source losing or inventing flux?",
                    fontsize=11, color=NAVY)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
    fig.suptitle("B4 against the parametric fit on the same frozen lens",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def figS4(ex, blob, out, n_show=3):
    have = [i for i in range(n_show) if f"src_b4_native_{i}" in ex]
    if not have:
        print("  (skipping figS4)"); return
    H = blob["config"]["half_extent"]
    res = blob["config"]["pixel_scale"]
    n_out = ex[f"src_b4_native_{have[0]}"].shape[-1]
    sc = 2 * H / (n_out - 1)
    fig, ax = plt.subplots(len(have), 2, figsize=(8, 3.7 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        S = ex[f"src_b4_native_{i}"]
        T = ex[f"src_true_{i}"]
        n_t = T.shape[-1]
        c = (n_t - 1) / 2.0
        h = int(round(H / res))
        Tc = T[int(c) - h:int(c) + h + 1, int(c) - h:int(c) + h + 1]
        ax[k, 0].imshow(_st(S), origin="lower", cmap="magma")
        ax[k, 1].imshow(_st(Tc), origin="lower", cmap="magma")
        for c2 in range(2):
            ax[k, c2].set_xticks([]); ax[k, c2].set_yticks([])
        if k == 0:
            ax[k, 0].set_title(f"B4 source, native grid\n{n_out}$^2$ at "
                               f"{sc:.4f}\"/px", fontsize=10, color=NAVY)
            ax[k, 1].set_title(f"TRUE source, detector grid\n{Tc.shape[0]}$^2$ at "
                               f"{res:.4f}\"/px", fontsize=10, color=NAVY)
    fig.suptitle(f"Super-resolution at native scale: {res/sc:.2f}x finer than "
                 f"the detector,\nagainst a measured median tangential stretch "
                 f"of 3.07x", fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="results/b4_metrics.json")
    ap.add_argument("--npz", default="results/b4_examples.npz")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    blob = json.load(open(a.metrics))
    ex = dict(np.load(a.npz))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"figures from {a.metrics}   n = {len(blob['rows'])}\n")
    H = blob["config"]["half_extent"]; res = blob["config"]["pixel_scale"]
    figS1(ex, f"{a.outdir}/figS1_pipeline.png", H, res)
    figS2(ex, f"{a.outdir}/figS2_residual.png")
    figS3(blob, f"{a.outdir}/figS3_compare.png")
    figS4(ex, blob, f"{a.outdir}/figS4_native.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
