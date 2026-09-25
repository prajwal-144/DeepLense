import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]
warnings.filterwarnings("ignore")
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
import lens_models as L

RES, N = 0.10593, 127
c = (N - 1) / 2.0
yy, xx = np.indices((N, N)).astype(float)
X, Y = (xx - c) * RES, (yy - c) * RES

CASES = [
    dict(theta_E=1.34, gamma=2.0, e1=0.0,   e2=0.0,   g1=0.0,   g2=0.0),   # SIS
    dict(theta_E=1.34, gamma=2.0, e1=0.22,  e2=0.0,   g1=0.0,   g2=0.0),   # SIE
    dict(theta_E=1.34, gamma=2.0, e1=-0.15, e2=0.17,  g1=0.0,   g2=0.0),
    dict(theta_E=0.90, gamma=1.90, e1=0.30, e2=-0.24, g1=0.0,   g2=0.0),   # EPL
    dict(theta_E=2.10, gamma=2.20, e1=-0.05,e2=-0.38, g1=0.0,   g2=0.0),
    dict(theta_E=1.54, gamma=1.93, e1=0.187,e2=-0.155,g1=-0.0053, g2=-0.0647),
    dict(theta_E=1.34, gamma=2.0, e1=0.22,  e2=0.11,  g1=0.06,  g2=-0.04),
]

def lenstronomy_ref(p):
    use_shear = abs(p["g1"]) + abs(p["g2"]) > 0
    ll = ["EPL", "SHEAR"] if use_shear else ["EPL"]
    kw = [dict(theta_E=p["theta_E"], gamma=p["gamma"], e1=p["e1"], e2=p["e2"],
               center_x=0.0, center_y=0.0)]
    if use_shear:
        kw.append(dict(gamma1=p["g1"], gamma2=p["g2"], ra_0=0, dec_0=0))
    lm = LensModel(ll)
    ax, ay = lm.alpha(X, Y, kw)
    fxx, fxy, fyx, fyy = lm.hessian(X, Y, kw)
    kap = 0.5 * (fxx + fyy)
    g1 = 0.5 * (fxx - fyy)
    g2 = fxy
    return ax, ay, kap, g1, g2

def main():
    r = np.hypot(xx - c, yy - c)
    mask = (r > 2) & (r < 55) # skip the singular centre and the corners
    ok = True
    print(f"{'case':52s}{'d|alpha|':>11}{'d kappa':>11}{'d gamma1':>11}{'d gamma2':>11}")
    for p in CASES:
        ax_r, ay_r, k_r, g1_r, g2_r = lenstronomy_ref(p)
        ax, ay = L.deflection(X, Y, p)
        k, g1, g2 = L.hessian_analytic(X, Y, p)
        d_a = np.abs(np.hypot(ax, ay) - np.hypot(ax_r, ay_r))[mask].max()
        # componentwise too, so a rotation error cannot hide behind the modulus
        d_ax = np.abs(ax - ax_r)[mask].max(); d_ay = np.abs(ay - ay_r)[mask].max()
        d_k = np.abs(k - k_r)[mask].max()
        d_g1 = np.abs(g1 - g1_r)[mask].max(); d_g2 = np.abs(g2 - g2_r)[mask].max()
        lbl = (f"tE={p['theta_E']:.2f} g={p['gamma']:.2f} "
               f"e=({p['e1']:+.2f},{p['e2']:+.2f}) sh=({p['g1']:+.3f},{p['g2']:+.3f})")
        print(f"{lbl:52s}{max(d_ax,d_ay):11.2e}{d_k:11.2e}{d_g1:11.2e}{d_g2:11.2e}")
        if max(d_ax, d_ay) > 1e-8 or max(d_k, d_g1, d_g2) > 1e-7:
            ok = False
    # hypergeometric truncation, measured not assumed
    from scipy.special import hyp2f1
    print("\nhypergeometric series truncation vs scipy.special.hyp2f1:")
    for q, t in [(0.436, 1.0), (0.436, 1.2), (0.64, 1.0), (0.9, 0.9)]:
        f = (1 - q) / (1 + q)
        th = np.linspace(0, 2 * np.pi, 97)
        w = -f * np.exp(2j * th)
        ref = hyp2f1(1, t / 2, 2 - t / 2, w)
        sr, si = L._hyp2f1_series(w.real, w.imag, np.array(t))
        err = np.abs((sr + 1j * si) - ref).max()
        print(f"   q={q:.3f} t={t:.2f}  |w|={f:.3f}   max |err| = {err:.2e}")
        if err > 1e-12:
            ok = False
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
