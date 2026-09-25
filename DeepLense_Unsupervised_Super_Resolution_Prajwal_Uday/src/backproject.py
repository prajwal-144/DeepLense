from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from lens_models import deflection

__all__ = ["source_grid", "ray_shoot_batch", "backproject", "sample_source"]

LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "cx", "cy")


def source_grid(n: int, half_extent: float):
    ax = np.linspace(-half_extent, half_extent, n)
    return np.meshgrid(ax, ax, indexing="xy")[0], np.meshgrid(ax, ax, indexing="ij")[0]


def ray_shoot_batch(X, Y, lens: dict):
    B = lens["theta_E"].shape[0]
    p = {k: lens[k].reshape(B, 1, 1) for k in LENS_KEYS if k in lens}
    p.setdefault("cx", torch.zeros(B, 1, 1, dtype=X.dtype, device=X.device))
    p.setdefault("cy", torch.zeros(B, 1, 1, dtype=X.dtype, device=X.device))
    ax, ay = deflection(X[None], Y[None], p)
    return X[None] - ax, Y[None] - ay


def backproject(img, bx, by, n_src: int, half_extent: float, smooth: int = 3):
    B = img.shape[0]
    scale = 2.0 * half_extent / (n_src - 1)
    with torch.no_grad():
        fj = (bx + half_extent) / scale
        fi = (by + half_extent) / scale
        inside = ((fj >= -0.5) & (fj <= n_src - 0.5) & (fi >= -0.5) & (fi <= n_src - 0.5))
        j = torch.round(fj).long().clamp(0, n_src - 1)
        i = torch.round(fi).long().clamp(0, n_src - 1)
        flat = (i * n_src + j).reshape(B, -1)
        w = inside.reshape(B, -1).to(img.dtype)

        num = torch.zeros(B, n_src * n_src, dtype=img.dtype, device=img.device)
        cov = torch.zeros_like(num)
        num.scatter_add_(1, flat, img.reshape(B, -1) * w)
        cov.scatter_add_(1, flat, w)

        bp = (num / cov.clamp_min(1.0)).view(B, 1, n_src, n_src)
        cv = cov.view(B, 1, n_src, n_src)
        if smooth and smooth > 1:
            pad = smooth // 2
            cv = F.avg_pool2d(F.pad(cv, (pad,) * 4, mode="replicate"),
                              smooth, stride=1)
    return bp[:, 0], cv[:, 0]


def sample_source(S, bx, by, half_extent: float):
    n = S.shape[-1]
    edge = half_extent * (1.0 + 1.0 / (n - 1))
    grid = torch.stack([bx / edge, by / edge], dim=-1)
    return F.grid_sample(S.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
