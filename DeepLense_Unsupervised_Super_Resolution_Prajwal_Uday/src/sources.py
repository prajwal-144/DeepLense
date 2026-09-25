from __future__ import annotations
from typing import Dict
from backend import get_backend

__all__ = ["SersicSource", "PixelSource", "SOURCE_PARAM_NAMES", "b_n",
           "sersic_defaults"]

SOURCE_PARAM_NAMES = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")

_SMOOTHING = 1e-4 # lenstronomy SersicUtil default
_N_MIN, _N_MAX = 0.3, 6.0 # Model_A range is 0.50-2.86; a cushion for optimisation
_R_MIN = 1e-3 # arcsec


def b_n(n):
    xp = get_backend(n)
    return xp.clip(1.9992 * n - 0.3271, 1e-5, None)


class SersicSource:

    def __init__(self, params: Dict):
        self.p = params

    def at(self, bx, by):
        xp = get_backend(bx)
        p = self.p
        Rs = xp.clip(p["R_sersic"], _R_MIN, None)
        n = xp.clip(p["n_sersic"], _N_MIN, _N_MAX)
        e1, e2 = p.get("se1", 0.0), p.get("se2", 0.0)
        dx = bx - p.get("sx", 0.0)
        dy = by - p.get("sy", 0.0)

        norm = xp.sqrt(xp.clip(xp.abs(1.0 - e1 * e1 - e2 * e2), 1e-6, None))
        xpr = ((1.0 - e1) * dx - e2 * dy) / norm
        ypr = (-e2 * dx + (1.0 + e1) * dy) / norm
        R = xp.clip(xp.sqrt(xpr * xpr + ypr * ypr), _SMOOTHING, None)

        return p["amp"] * xp.exp(-b_n(n) * (xp.power(R / Rs, 1.0 / n) - 1.0))


def sersic_defaults() -> Dict[str, float]:
    return {"amp": 1.0, "R_sersic": 0.45, "n_sersic": 1.0,
            "se1": 0.0, "se2": 0.0, "sx": 0.0, "sy": 0.0}


class PixelSource:

    def __init__(self, pixels, half_extent: float = 1.6):
        self.pix = pixels # (B, 1, n, n)
        self.half = float(half_extent)

    @classmethod
    def zeros(cls, batch: int, n_src: int = 48, half_extent: float = 1.6,
              device=None, dtype=None, init: float = 1e-3):
        import torch
        t = torch.full((batch, 1, n_src, n_src), init, device=device, dtype=dtype or torch.float32)
        return cls(t.requires_grad_(True), half_extent)

    def at(self, bx, by):
        import torch
        import torch.nn.functional as F
        n = self.pix.shape[-1]
        edge = self.half * (1.0 + 1.0 / (n - 1))
        grid = torch.stack([bx / edge, by / edge], dim=-1)     # (B, H, W, 2)
        return F.grid_sample(self.pix, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]

    def curvature(self):
        s = self.pix
        lap = (-4.0 * s[..., 1:-1, 1:-1]
               + s[..., :-2, 1:-1] + s[..., 2:, 1:-1]
               + s[..., 1:-1, :-2] + s[..., 1:-1, 2:])
        return (lap ** 2).sum(dim=(1, 2, 3))

    def nonneg_penalty(self):
        return (self.pix.clamp(max=0.0) ** 2).sum(dim=(1, 2, 3))

    def parameters(self):
        return [self.pix]