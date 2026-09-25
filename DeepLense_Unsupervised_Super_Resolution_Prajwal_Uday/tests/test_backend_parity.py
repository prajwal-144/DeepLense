import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]
warnings.filterwarnings("ignore")
import numpy as np
try:
    import torch
except Exception as e:
    raise SystemExit(f"torch not available: {e}")
import lens_models as L
from sources import SersicSource

def main():
    RES, N = 0.10593, 64
    c = (N - 1) / 2.0
    yy, xx = np.indices((N, N)).astype(float)
    Xn, Yn = (xx - c) * RES, (yy - c) * RES
    Xt = torch.as_tensor(Xn, dtype=torch.float64)
    Yt = torch.as_tensor(Yn, dtype=torch.float64)
    ok = True
    for p in [dict(theta_E=1.34, gamma=2.0, e1=0.0, e2=0.0, g1=0.0, g2=0.0),
              dict(theta_E=1.34, gamma=1.93, e1=0.22, e2=-0.11, g1=0.05, g2=0.03)]:
        pt = {k: torch.tensor(v, dtype=torch.float64) for k, v in p.items()}
        an = L.deflection(Xn, Yn, p)
        at = L.deflection(Xt, Yt, pt)
        d = max(float(np.abs(an[0] - at[0].numpy()).max()), float(np.abs(an[1] - at[1].numpy()).max()))
        print(f"deflection  tE={p['theta_E']} e=({p['e1']},{p['e2']})   max |np - torch| = {d:.2e}")
        if d > 1e-6:
            ok = False
    sp = dict(amp=1.0, R_sersic=0.45, n_sersic=1.3, se1=0.2, se2=-0.1, sx=0.1, sy=-0.2)
    spt = {k: torch.tensor(v, dtype=torch.float64) for k, v in sp.items()}
    d = float(np.abs(SersicSource(sp).at(Xn, Yn) - SersicSource(spt).at(Xt, Yt).numpy()).max())
    print(f"sersic                                   max |np - torch| = {d:.2e}")
    if d > 1e-6:
        ok = False
    # autograd magnification vs the analytic Hessian
    p = dict(theta_E=1.34, gamma=2.0, e1=0.22, e2=-0.11, g1=0.05, g2=0.03)
    pt = {k: torch.tensor(v, dtype=torch.float64) for k, v in p.items()}
    mu_ag = L.magnification_autograd(Xt.clone(), Yt.clone(), pt, mu_clip=1e6).detach().numpy()
    mu_an = L.magnification(Xn, Yn, p, mu_clip=1e6)
    r = np.hypot(xx - c, yy - c)
    m = (r > 4) & (r < 28)
    d = float(np.abs(mu_ag - mu_an)[m].max())
    print(f"magnification autograd vs analytic       max |diff| = {d:.2e}")
    if d > 1e-4:
        ok = False
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())