import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]
warnings.filterwarnings("ignore")
import numpy as np
import lens_models as L

def main():
    RES, N = 0.10593, 127
    c = (N - 1) / 2.0
    yy, xx = np.indices((N, N)).astype(float)
    X, Y = (xx - c) * RES, (yy - c) * RES
    r = np.hypot(xx - c, yy - c)
    m = (r > 4) & (r < 45)
    h = 1e-5
    ok = True
    print(f"{'case':46s}{'max |d kappa|':>15}{'max |d |gamma||':>17}")
    for p in [dict(theta_E=1.34, gamma=2.0, e1=0.0, e2=0.0, g1=0.0, g2=0.0),
              dict(theta_E=1.34, gamma=2.0, e1=0.22, e2=-0.11, g1=0.05, g2=0.03),
              dict(theta_E=0.90, gamma=1.90, e1=0.30, e2=-0.24, g1=-0.08, g2=0.02),
              dict(theta_E=2.10, gamma=2.20, e1=-0.05, e2=-0.38, g1=0.0, g2=0.0)]:
        axp, ayp = L.deflection(X + h, Y, p); axm, aym = L.deflection(X - h, Y, p)
        axx = (axp - axm) / (2 * h); ayx = (ayp - aym) / (2 * h)
        axp, ayp = L.deflection(X, Y + h, p); axm, aym = L.deflection(X, Y - h, p)
        axy = (axp - axm) / (2 * h); ayy = (ayp - aym) / (2 * h)
        k_n = 0.5 * (axx + ayy); g1_n = 0.5 * (axx - ayy); g2_n = 0.5 * (axy + ayx)
        k_a, g1_a, g2_a = L.hessian_analytic(X, Y, p)
        dk = np.abs(k_n - k_a)[m].max()
        dg = np.abs(np.hypot(g1_n, g2_n) - np.hypot(g1_a, g2_a))[m].max()
        lbl = f"tE={p['theta_E']:.2f} g={p['gamma']:.2f} e=({p['e1']:+.2f},{p['e2']:+.2f})"
        print(f"{lbl:46s}{dk:15.2e}{dg:17.2e}")
        if dk > 1e-4 or dg > 1e-4:
            ok = False
    # SIS closed form: lambda_r = 1, lambda_t = 1 - theta_E/r, mu = 1/(1-theta_E/r)
    p = dict(theta_E=1.34, gamma=2.0, e1=0.0, e2=0.0)
    mu = L.magnification(X, Y, p, mu_clip=1e6, signed=True)
    rad = np.hypot(X, Y)
    mu_ref = 1.0 / (1.0 - p["theta_E"] / np.maximum(rad, 1e-6))
    sel = (rad > 2.0 * p["theta_E"])
    d = np.abs(mu - mu_ref)[sel].max()
    print(f"\nSIS magnification vs 1/(1 - theta_E/r) outside 2 theta_E: max |diff| = {d:.2e}")
    if d > 1e-6:
        ok = False
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
