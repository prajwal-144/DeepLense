from __future__ import annotations
from typing import Dict
import numpy as np

__all__ = ["polar_ridge", "fourier_rphi", "completeness", "initial_guess"]

EPS = 1e-12


def polar_ridge(img: np.ndarray, n_ang: int = 72, r_min: float = 1.5,
                r_max: float = None, dr: float = 0.5):
    a = np.clip(np.squeeze(np.asarray(img, dtype=np.float64)), 0, None)
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r_max = r_max or (min(h, w) / 2.0 - 1.5)

    phi = np.linspace(-np.pi, np.pi, n_ang, endpoint=False)
    radii = np.arange(r_min, r_max, dr)
    yy = cy + radii[None, :] * np.sin(phi[:, None])
    xx = cx + radii[None, :] * np.cos(phi[:, None])

    y0 = np.clip(np.floor(yy).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy, fx = yy - y0, xx - x0
    vals = (a[y0, x0] * (1 - fy) * (1 - fx) + a[y1, x0] * fy * (1 - fx) + a[y0, x1] * (1 - fy) * fx + a[y1, x1] * fy * fx)

    j = np.argmax(vals, axis=1)
    return phi, radii[j], vals[np.arange(n_ang), j]


def fourier_rphi(phi, r, weight):
    w = np.clip(weight, 0, None)
    if w.sum() <= EPS:
        return (np.nan,) * 4
    w = w / w.sum()
    A = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                  np.cos(2 * phi), np.sin(2 * phi)], axis=1)
    W = np.sqrt(w)[:, None]
    try:
        coef, *_ = np.linalg.lstsq(A * W, r * W[:, 0], rcond=None)
    except np.linalg.LinAlgError:
        return (np.nan,) * 4
    return (float(coef[0]), float(coef[1]), float(coef[2]),
            float(np.hypot(coef[3], coef[4])))


def completeness(intensity, frac: float = 0.35) -> float:
    peak = float(np.max(intensity))
    if peak <= EPS:
        return 0.0
    return float((intensity >= frac * peak).mean())


def initial_guess(img: np.ndarray, pixel_scale: float) -> Dict[str, float]:
    phi, r, ival = polar_ridge(img)
    r0, a1, b1, m2 = fourier_rphi(phi, r, ival)
    comp = completeness(ival)
    if not np.isfinite(r0) or r0 <= 0:
        r0, a1, b1, m2 = 12.5, 0.0, 0.0, 0.0 # Model_A median, last resort
    return {
        "theta_E": float(r0 * pixel_scale),
        "sx": float(a1 * pixel_scale),
        "sy": float(b1 * pixel_scale),
        "m2_over_theta_E": float(m2 / max(r0, EPS)),
        "ring_completeness": float(comp),
        "ring_peak": float(np.max(ival)),
    }