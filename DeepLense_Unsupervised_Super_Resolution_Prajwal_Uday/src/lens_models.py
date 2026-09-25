from __future__ import annotations
from typing import Dict, Tuple
from backend import get_backend

__all__ = ["ellipticity_to_phi_q", "deflection_epl", "deflection_shear",
           "deflection", "ray_shoot", "hessian_analytic", "magnification",
           "magnification_autograd", "LENS_PARAM_NAMES"]

N_HYP_TERMS = 50
R_MIN_ARCSEC = 1e-4 # ~1/1000 of a Model_A pixel; see deflection_epl
E_MAX = 0.9 # cap on |e| during optimisation; Model_A max is 0.393
EPS_E2 = 1e-24 # guards the 0/0 gradient of |e| at the origin
E_FLOOR = 1e-10 # floor on |e|; see ellipticity_to_phi_q

LENS_PARAM_NAMES = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "cx", "cy")


def ellipticity_to_phi_q(e1, e2, max_c: float = E_MAX, floor: float = E_FLOOR):
    xp = get_backend(e1)
    u = e1 * e1 + e2 * e2
    degenerate = u < floor * floor
    e1 = xp.where(degenerate, xp.ones_like(e1) * floor, e1)
    e2 = xp.where(degenerate, xp.zeros_like(e2), e2)
    phi = 0.5 * xp.atan2(e2, e1)
    # +EPS_E2 inside the sqrt keeps the gradient finite at e1 = e2 = 0, where d|e| is 0/0 and any zero-initialised ellipticity head would go NaN on
    # the first step. The shift in |e| is at most 1e-12, far below Model_A's smallest recorded |e| of 1e-3, and test_lens_models.py still matches
    # lenstronomy to 1e-12 with it in place.
    c = xp.clip(xp.sqrt(e1 * e1 + e2 * e2 + EPS_E2), None, max_c)
    q = (1.0 - c) / (1.0 + c)
    return phi, q


def _rotate(x, y, phi):
    xp = get_backend(x)
    c, s = xp.cos(phi), xp.sin(phi)
    return x * c + y * s, -x * s + y * c


def _n_terms_for(f, tol: float = 1e-12, cap: int = N_HYP_TERMS) -> int:
    import math
    try:
        fmax = float(f.max()) if hasattr(f, "max") else float(f)
    except Exception:
        return cap
    if not math.isfinite(fmax):
        raise FloatingPointError(
            f"_n_terms_for got a non-finite f = {fmax}. f = (1-q)/(1+q), so this "
            "means e1/e2 reaching ellipticity_to_phi_q were NaN or inf -- the "
            "lens parameters have already diverged upstream. In a training loop "
            "the usual cause is a NaN gradient poisoning the weights; check the "
            "first step where the loss stops being finite, not this function.")
    fmax = min(max(fmax, 1e-12), 0.999)
    if fmax < 1e-9:
        return 1
    return int(min(cap, max(2, math.ceil(math.log(tol) / math.log(fmax)))))


def _hyp2f1_series(wr, wi, t, n_terms: int = N_HYP_TERMS):
    xp = get_backend(wr)
    half_t = 0.5 * t
    sr = xp.ones_like(wr)
    si = xp.zeros_like(wr)
    tr = xp.ones_like(wr)
    ti = xp.zeros_like(wr)
    for n in range(n_terms):
        c = (half_t + n) / (2.0 - half_t + n)
        nr = c * (tr * wr - ti * wi)
        ni = c * (tr * wi + ti * wr)
        tr, ti = nr, ni
        sr = sr + tr
        si = si + ti
    return sr, si


def deflection_epl(x, y, theta_E, gamma, e1, e2, cx=0.0, cy=0.0, r_min: float = R_MIN_ARCSEC, n_terms: int = N_HYP_TERMS):
    xp = get_backend(x)
    x = x - cx
    y = y - cy

    phi, q = ellipticity_to_phi_q(e1, e2)
    b = theta_E * xp.sqrt(q)
    t = gamma - 1.0

    xr, yr = _rotate(x, y, phi)
    Zr = q * xr
    Zi = yr
    R2 = xp.clip(Zr * Zr + Zi * Zi, r_min * r_min, None)
    R = xp.sqrt(R2)

    f = (1.0 - q) / (1.0 + q)

    # Circular fast path. At q = 1 the hypergeometric argument is zero and 2F1 = 1 exactly, so the series can be skipped. Model_A's macro lens is
    # circular and fit_per_image.py holds e1 = e2 = 0 for its first three stages, so this is the common branch. The 1e-9 threshold on f is |e| < 5e-10.
    if _n_terms_for(f) <= 1:
        ror, roi = Zr, Zi
    else:
        inv = 1.0 / R2
        wr = -f * (Zr * Zr - Zi * Zi) * inv
        wi = -f * (2.0 * Zr * Zi) * inv
        nt = min(n_terms, _n_terms_for(f))
        sr, si = _hyp2f1_series(wr, wi, t, nt)
        ror = Zr * sr - Zi * si
        roi = Zr * si + Zi * sr

    pref = (2.0 / (1.0 + q)) * xp.power(b / R, t)
    return _rotate(pref * ror, pref * roi, -phi)


def deflection_shear(x, y, g1, g2):
    return g1 * x + g2 * y, g2 * x - g1 * y


def deflection(x, y, p: Dict, n_terms: int = N_HYP_TERMS):
    ax, ay = deflection_epl(x, y, p["theta_E"], p.get("gamma", 2.0),
                            p.get("e1", 0.0), p.get("e2", 0.0),
                            p.get("cx", 0.0), p.get("cy", 0.0), n_terms=n_terms)
    if "g1" in p or "g2" in p:
        sx, sy = deflection_shear(x, y, p.get("g1", 0.0), p.get("g2", 0.0))
        ax, ay = ax + sx, ay + sy
    return ax, ay


def ray_shoot(x, y, p: Dict, **kw):
    ax, ay = deflection(x, y, p, **kw)
    return x - ax, y - ay


###################################
# convergence, shear, magnification
###################################

def hessian_analytic(x, y, p: Dict, r_min: float = R_MIN_ARCSEC):
    xp = get_backend(x)
    x = x - p.get("cx", 0.0)
    y = y - p.get("cy", 0.0)
    theta_E, gamma = p["theta_E"], p.get("gamma", 2.0)
    e1, e2 = p.get("e1", 0.0), p.get("e2", 0.0)

    phi, q = ellipticity_to_phi_q(e1, e2)
    b = theta_E * xp.sqrt(q)
    t = gamma - 1.0

    xr, yr = _rotate(x, y, phi)
    R = xp.clip(xp.sqrt((q * xr) ** 2 + yr * yr), r_min, None)
    r = xp.clip(xp.sqrt(xr * xr + yr * yr), r_min, None)

    Zr, Zi = q * xr, yr
    R2 = xp.clip(Zr * Zr + Zi * Zi, r_min * r_min, None)
    f = (1.0 - q) / (1.0 + q)
    wr = -f * (Zr * Zr - Zi * Zi) / R2
    wi = -f * (2.0 * Zr * Zi) / R2
    sr, si = _hyp2f1_series(wr, wi, t)
    ror, roi = Zr * sr - Zi * si, Zr * si + Zi * sr
    pref = (2.0 / (1.0 + q)) * xp.power(b / R, t)
    ax_, ay_ = pref * ror, pref * roi                # major-axis frame

    cos, sin = xr / r, yr / r
    cos2, sin2 = cos * cos * 2.0 - 1.0, sin * cos * 2.0
    kappa = (2.0 - t) / 2.0 * xp.power(b / R, t)
    g1m = (1.0 - t) * (ax_ * cos - ay_ * sin) / r - kappa * cos2
    g2m = (1.0 - t) * (ay_ * cos + ax_ * sin) / r - kappa * sin2

    c2, s2 = xp.cos(2.0 * phi), xp.sin(2.0 * phi)
    g1 = g1m * c2 - g2m * s2
    g2 = g1m * s2 + g2m * c2

    if "g1" in p or "g2" in p:
        g1 = g1 + p.get("g1", 0.0)
        g2 = g2 + p.get("g2", 0.0)
    return kappa, g1, g2


def magnification(x, y, p: Dict, mu_clip: float = 50.0, signed: bool = False):
    xp = get_backend(x)
    kappa, g1, g2 = hessian_analytic(x, y, p)
    det = (1.0 - kappa) ** 2 - (g1 * g1 + g2 * g2)
    sign = xp.where(det >= 0, xp.ones_like(det), -xp.ones_like(det))
    safe = sign * xp.clip(xp.abs(det), 1.0 / mu_clip, None)
    mu = 1.0 / safe
    return mu if signed else xp.clip(xp.abs(mu), None, mu_clip)


def magnification_autograd(x, y, p: Dict, mu_clip: float = 50.0):
    import torch
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    ax, ay = deflection(x, y, p)
    axx, axy = torch.autograd.grad(ax.sum(), (x, y), create_graph=True)
    ayx, ayy = torch.autograd.grad(ay.sum(), (x, y), create_graph=True)
    kappa = 0.5 * (axx + ayy)
    g1 = 0.5 * (axx - ayy)
    g2 = 0.5 * (axy + ayx)
    det = (1.0 - kappa) ** 2 - (g1 * g1 + g2 * g2)
    sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    return (1.0 / (sign * det.abs().clamp_min(1.0 / mu_clip))).abs().clamp(max=mu_clip)
