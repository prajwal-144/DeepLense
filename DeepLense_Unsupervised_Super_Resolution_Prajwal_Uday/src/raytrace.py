from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import numpy as np
from backend import get_backend
from lens_models import ray_shoot

__all__ = ["image_plane_grid", "gaussian_psf", "moffat_psf", "load_psf",
           "convolve", "area_downsample", "render", "SUPERSAMPLE"]

SUPERSAMPLE = 3


def image_plane_grid(n: int, pixel_scale: float, supersample: int = 1):
    S = int(supersample)
    c = (n - 1) / 2.0
    idx = np.arange(n, dtype=np.float64)
    if S > 1:
        off = (np.arange(S, dtype=np.float64) + 0.5) / S - 0.5
        pix = (idx[:, None] + off[None, :]).reshape(-1)
    else:
        pix = idx
    ang = (pix - c) * pixel_scale
    Y, X = np.meshgrid(ang, ang, indexing="ij")
    return X, Y


def gaussian_psf(fwhm_arcsec: float, pixel_scale: float, truncate: float = 4.0):
    if fwhm_arcsec is None or fwhm_arcsec <= 0:
        return None
    sigma = (fwhm_arcsec / 2.3548200450309493) / pixel_scale
    rad = max(1, int(math.ceil(truncate * sigma)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    return np.outer(k, k)


def moffat_psf(fwhm_arcsec: float, pixel_scale: float, beta: float = 2.5, truncate: float = 6.0):
    if fwhm_arcsec is None or fwhm_arcsec <= 0:
        return None
    alpha = (fwhm_arcsec / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))) / pixel_scale
    rad = max(1, int(math.ceil(truncate * alpha)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    xx, yy = np.meshgrid(t, t, indexing="xy")
    k = (1.0 + (xx * xx + yy * yy) / (alpha * alpha)) ** (-beta)
    return k / k.sum()


def load_psf(path: str, crop: int = 25):
    k = np.load(path)
    n = k.shape[-1]
    c = n // 2
    h = crop // 2
    k = k[c - h:c + h + 1, c - h:c + h + 1]
    return k / k.sum()


def convolve(img, kernel):
    if kernel is None:
        return img
    xp = get_backend(img)
    if xp.name == "torch":
        import torch
        import torch.nn.functional as F
        k = kernel if torch.is_tensor(kernel) else torch.as_tensor(kernel)
        k = k.to(img.dtype).to(img.device)[None, None]
        pad = k.shape[-1] // 2
        return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="replicate"), k)
    from scipy.signal import fftconvolve
    pad = kernel.shape[-1] // 2
    padded = np.pad(np.asarray(img), pad, mode="edge")
    return fftconvolve(padded, kernel, mode="valid")


def area_downsample(img, factor: int):
    if factor == 1:
        return img
    S = int(factor)
    xp = get_backend(img)
    if xp.name == "torch":
        b, c, h, w = img.shape
        assert h % S == 0 and w % S == 0
        return img.reshape(b, c, h // S, S, w // S, S).mean(dim=(3, 5))
    a = np.asarray(img)
    h, w = a.shape[-2:]
    assert h % S == 0 and w % S == 0, f"{h}x{w} not divisible by {S}"
    return a.reshape(*a.shape[:-2], h // S, S, w // S, S).mean(axis=(-3, -1))


def render(source, lens_params: Dict, n_pix: int, pixel_scale: float,
           psf=None, supersample: int = SUPERSAMPLE, grid=None,
           background: float = 0.0, return_sky: bool = False):
    if grid is None:
        X, Y = image_plane_grid(n_pix, pixel_scale, supersample)
    else:
        X, Y = grid
    bx, by = ray_shoot(X, Y, lens_params)
    sky = source.at(bx, by)
    blurred = convolve(sky, psf)
    pred = area_downsample(blurred, supersample) + background
    if return_sky:
        return pred, sky
    return pred