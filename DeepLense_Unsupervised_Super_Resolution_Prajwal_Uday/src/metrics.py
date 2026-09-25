from __future__ import annotations
from typing import Dict, Iterable, Optional
import numpy as np

__all__ = ["chi2_per_dof", "source_truth", "centroid_size", "parameter_errors",
           "stratify", "null_baselines"]

EPS = 1e-12

######################
# goodness of fit
######################

def chi2_per_dof(pred: np.ndarray, data: np.ndarray, sigma: float, mask: Optional[np.ndarray] = None, n_params: int = 0) -> float:
    r = (pred - data) / max(sigma, EPS)
    if mask is not None:
        r = r[mask]
    n = r.size
    return float((r ** 2).sum() / max(n - n_params, 1))


######################
# source-plane truth
######################

def centroid_size(a: np.ndarray, thresh_frac: float = 0.05):
    a = np.clip(np.squeeze(np.asarray(a, dtype=np.float64)), 0, None)
    peak = a.max()
    if peak <= EPS:
        return np.nan, np.nan, np.nan
    f = np.where(a >= thresh_frac * peak, a, 0.0)
    tot = f.sum()
    if tot <= EPS:
        return np.nan, np.nan, np.nan
    yy, xx = np.indices(a.shape)
    cy = float((f * yy).sum() / tot)
    cx = float((f * xx).sum() / tot)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return cy, cx, float(np.sqrt((f * r2).sum() / tot))


def source_truth(model_source_psf: np.ndarray, truth_unlensed: np.ndarray,
                 pixel_scale: float = 1.0) -> Dict[str, float]:
    p = np.clip(np.squeeze(np.asarray(model_source_psf, float)), 0, None)
    t = np.clip(np.squeeze(np.asarray(truth_unlensed, float)), 0, None)
    out: Dict[str, float] = {}

    pf, tf = p.ravel(), t.ravel()
    a = pf - pf.mean()
    b = tf - tf.mean()
    out["corr"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + EPS))

    s = (pf * tf).sum() / max((pf * pf).sum(), EPS)          # optimal amplitude
    out["nmse"] = float(((s * pf - tf) ** 2).sum() / max((tf * tf).sum(), EPS))

    cy_p, cx_p, sz_p = centroid_size(p)
    cy_t, cx_t, sz_t = centroid_size(t)
    out["pred_size_px"] = sz_p
    out["true_size_px"] = sz_t
    out["size_ratio"] = float(sz_p / sz_t) if sz_t and np.isfinite(sz_t) else np.nan
    out["centroid_err_px"] = float(np.hypot(cy_p - cy_t, cx_p - cx_t))
    out["centroid_err_arcsec"] = out["centroid_err_px"] * pixel_scale

    ps = p / max(p.sum(), EPS)
    ts = t / max(t.sum(), EPS)
    out["peak_ratio"] = float(ps.max() / max(ts.max(), EPS))
    return out


######################
# parameter recovery
######################

_CIRCULAR = {"e": "vector", "g": "vector"}


def parameter_errors(fit: Dict[str, float], truth: Dict[str, float], keys: Iterable[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        if k in fit and k in truth and np.isfinite(truth[k]):
            out[f"d_{k}"] = float(fit[k] - truth[k])
    for a, b, name in (("e1", "e2", "e"), ("g1", "g2", "g")):
        if a in fit and a in truth:
            out[f"{name}_fit"] = float(np.hypot(fit[a], fit[b]))
            out[f"{name}_true"] = float(np.hypot(truth[a], truth[b]))
            out[f"d_{name}_vec"] = float(np.hypot(fit[a] - truth[a], fit[b] - truth[b]))
    return out


######################
# reporting
######################

def stratify(rows, key: str, by: str = "snr_max",
             bins=((0, 6), (6, 15), (15, 1e9))) -> str:
    lines = [f"  {key} stratified by {by}"]
    v = np.array([r.get(key, np.nan) for r in rows], float)
    s = np.array([r.get(by, np.nan) for r in rows], float)
    for lo, hi in bins:
        m = np.isfinite(v) & np.isfinite(s) & (s >= lo) & (s < hi)
        lbl = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        if m.sum() == 0:
            lines.append(f"    {lbl:>10}  n=  0")
            continue
        lines.append(f"    {lbl:>10}  n={m.sum():3d}   median {np.median(v[m]):9.4f}"
                     f"   p25 {np.percentile(v[m],25):9.4f}"
                     f"   p75 {np.percentile(v[m],75):9.4f}")
    ok = np.isfinite(v)
    lines.append(f"    {'ALL':>10}  n={ok.sum():3d}   median {np.median(v[ok]):9.4f}")
    return "\n".join(lines)


def null_baselines(images, sigmas) -> Dict[str, float]:
    from scipy.ndimage import gaussian_filter
    out = {"constant": [], "blur2": [], "blur3": []}
    for img, sig in zip(images, sigmas):
        out["constant"].append(chi2_per_dof(np.full_like(img, img.mean()), img, sig))
        out["blur2"].append(chi2_per_dof(gaussian_filter(img, 2.0), img, sig))
        out["blur3"].append(chi2_per_dof(gaussian_filter(img, 3.0), img, sig))
    return {k: float(np.median(v)) for k, v in out.items()}
