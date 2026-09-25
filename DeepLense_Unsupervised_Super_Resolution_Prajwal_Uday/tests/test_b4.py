from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")]

import numpy as np

try:
    import torch
except Exception as exc:
    raise SystemExit(f"test_b4.py needs torch (import failed: {exc})")

from backproject import backproject, ray_shoot_batch, sample_source
from raytrace import image_plane_grid
from sisr_net import SourceSISR
from sources import SersicSource

N_PIX, RES, H = 127, 0.10593, 1.2
N_IN = int(round(2 * H / RES)) + 1
MAG = 2
N_OUT = N_IN * MAG
LENS = dict(theta_E=1.35, gamma=2.05, e1=0.14, e2=-0.07, g1=0.03, g2=0.01)
SRC = dict(amp=3.0, R_sersic=0.45, n_sersic=1.4, se1=0.10, se2=0.06, sx=0.15, sy=-0.10)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")


gx, gy = image_plane_grid(N_PIX, RES, 1)
GX = torch.as_tensor(gx, dtype=torch.float64)
GY = torch.as_tensor(gy, dtype=torch.float64)
lens = {k: torch.tensor([v], dtype=torch.float64) for k, v in LENS.items()}
bx, by = ray_shoot_batch(GX, GY, lens)

print("\n1. BACK-PROJECTION ROUND TRIP")
srcp = {k: torch.tensor([[[v]]], dtype=torch.float64) for k, v in SRC.items()}
lensed = SersicSource(srcp).at(bx, by)                       # noiseless, no PSF
bp, cov = backproject(lensed, bx, by, N_IN, H)

ax = torch.linspace(-H, H, N_IN, dtype=torch.float64)
YY, XX = torch.meshgrid(ax, ax, indexing="ij")
truth = SersicSource(srcp).at(XX[None], YY[None])[0]

good = cov[0] >= 3
a = bp[0][good].numpy(); b = truth[good].numpy()
r = float(np.corrcoef(a, b)[0, 1])
rel = float(np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
check("back-projection correlates with the true source where coverage >= 3",
      r > 0.99, f"corr {r:.6f} over {int(good.sum())} px")
check("and matches its VALUE, not just its shape (surface brightness "
      "is conserved, so no mu factor may appear)", rel < 0.05,
      f"median relative error {100*rel:.2f}%")

print("\n2. COVERAGE = MAGNIFICATION")
scale = 2.0 * H / (N_IN - 1)
edge = H + scale / 2.0
n_in_box = int(((bx.abs() <= edge) & (by.abs() <= edge)).sum())
check("only in-box rays counted, each exactly once", abs(float(cov.sum()) - n_in_box) < 0.5 * N_IN,
      f"cov.sum() {float(cov.sum()):.0f}   in box {n_in_box}   total {bx.numel()}")
c = cov[0].numpy()
mid = N_IN // 2
inner = c[mid - 3:mid + 3, mid - 3:mid + 3].mean()
outer = np.r_[c[:3].ravel(), c[-3:].ravel()].mean()
check("coverage higher near the caustic than at the box edge", inner > outer,
      f"inner {inner:.1f} vs edge {outer:.1f}")

print("\n3. SAMPLER ALIGNMENT")
S = torch.zeros(1, N_OUT, N_OUT, dtype=torch.float64)
i0, j0 = 13, 29
S[0, i0, j0] = 1.0
sc = 2.0 * H / (N_OUT - 1)
xq = torch.tensor([[[-H + j0 * sc]]], dtype=torch.float64)
yq = torch.tensor([[[-H + i0 * sc]]], dtype=torch.float64)
v_on = float(sample_source(S, xq, yq, H)[0, 0, 0])
v_off = float(sample_source(S, xq + sc, yq, H)[0, 0, 0])
check("a delta reads back at its own pixel centre", abs(v_on - 1.0) < 1e-9,f"value {v_on:.9f}")
check("and is zero one pixel away", abs(v_off) < 1e-9, f"value {v_off:.2e}")

print("\n4. SISR SHAPE, POSITIVITY, GRADIENTS")
net = SourceSISR(in_ch=2, width=16, depth=2, mag=MAG, n_up=1)
x = torch.randn(2, 2, N_IN, N_IN)
y = net(x)
check(f"output is mag x input  ({N_IN} -> {N_OUT})", tuple(y.shape[-2:]) == (N_OUT, N_OUT),
      f"got {tuple(y.shape)}")
check("output strictly positive", bool((y > 0).all()), f"min {float(y.min()):.3e}")
y.square().mean().backward()
bad = [n for n, q in net.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
check("all gradients finite", not bad, f"non-finite in {bad[:3]}")


print("\n5. SPATIAL CORRESPONDENCE")
# B3 pools with AdaptiveAvgPool2d(1), so where the input is poked does not affect where the output changes. Here it must: poke one off-centre input
# pixel and the largest response has to land at the matching output pixel. GroupNorm normalises over (C,H,W), so a small global response is expected;
# only the location of the peak is asserted.
net.eval()
i0, j0 = N_IN // 4, (3 * N_IN) // 4
with torch.no_grad():
    x0 = torch.zeros(1, 2, N_IN, N_IN)
    y0 = net(x0)
    x1 = x0.clone()
    x1[0, 0, i0, j0] = 10.0
    d = (net(x1) - y0).abs()[0].numpy()
pi, pj = np.unravel_index(int(d.argmax()), d.shape)
exp_i, exp_j = i0 * MAG + (MAG - 1) / 2.0, j0 * MAG + (MAG - 1) / 2.0
err = float(np.hypot(pi - exp_i, pj - exp_j))
check("the peak output response lands on the poked pixel", err <= 2.0 * MAG,
      f"peak at ({pi},{pj}), expected ~({exp_i:.1f},{exp_j:.1f}), off by {err:.1f} px")
rad = np.hypot(*(np.indices((N_OUT, N_OUT)) - np.array([[[exp_i]], [[exp_j]]])))
near, far = d[rad <= 4].mean(), d[rad >= N_OUT // 2].mean()
check("and the response decays away from it", near > 3.0 * max(far, 1e-12),
      f"mean |change| near {near:.3e} vs far {far:.3e}  ratio {near/max(far,1e-12):.1f}x")

print("\n" + ("PASS: safe to run train_b4.py" if ok else "FAIL"))
sys.exit(0 if ok else 1)
