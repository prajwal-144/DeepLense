import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]
warnings.filterwarnings("ignore")
import numpy as np
from lenstronomy.LightModel.light_model import LightModel
from sources import SersicSource

def main():
    RES, N = 0.10593, 127
    c = (N - 1) / 2.0
    yy, xx = np.indices((N, N)).astype(float)
    X, Y = (xx - c) * RES, (yy - c) * RES
    lm = LightModel(["SERSIC_ELLIPSE"])
    ok = True
    print(f"{'case':52s}{'max |diff| / peak':>19}")
    for kw in [dict(amp=1.0, R_sersic=0.45, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0),
               dict(amp=2.3, R_sersic=0.21, n_sersic=0.5, e1=0.31, e2=-0.12, center_x=0.13, center_y=-0.31),
               dict(amp=0.7, R_sersic=1.10, n_sersic=2.9, e1=-0.24, e2=0.30, center_x=-0.4, center_y=0.2),
               dict(amp=1.0, R_sersic=0.70, n_sersic=1.5, e1=0.07, e2=0.34, center_x=0.006, center_y=-0.309)]:
        ref = lm.surface_brightness(X, Y, [kw])
        mine = SersicSource(dict(amp=kw["amp"], R_sersic=kw["R_sersic"],
                                 n_sersic=kw["n_sersic"], se1=kw["e1"], se2=kw["e2"],
                                 sx=kw["center_x"], sy=kw["center_y"])).at(X, Y)
        d = np.abs(mine - ref).max() / ref.max()
        print(f"{str({k: round(v,3) for k,v in kw.items() if k!='amp'}):52s}{d:19.2e}")
        if d > 1e-10:
            ok = False
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
