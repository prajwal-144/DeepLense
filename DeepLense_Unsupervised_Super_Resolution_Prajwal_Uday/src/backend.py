from __future__ import annotations
import numpy as np

__all__ = ["get_backend", "NumpyBackend", "TorchBackend", "HAS_TORCH"]

try:
    import torch as _torch
    HAS_TORCH = True
except Exception:
    _torch = None
    HAS_TORCH = False


class NumpyBackend:
    name = "numpy"
    sqrt = staticmethod(np.sqrt)
    exp = staticmethod(np.exp)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    atan2 = staticmethod(np.arctan2)
    abs = staticmethod(np.abs)
    where = staticmethod(np.where)
    ones_like = staticmethod(np.ones_like)
    zeros_like = staticmethod(np.zeros_like)

    @staticmethod
    def power(x, p):
        return np.power(x, p)

    @staticmethod
    def clip(x, lo=None, hi=None):
        return np.clip(x, lo, hi)

    @staticmethod
    def ndim(x):
        return np.ndim(x)


class TorchBackend:
    name = "torch"

    @staticmethod
    def sqrt(x):
        return _torch.sqrt(x)

    @staticmethod
    def exp(x):
        return _torch.exp(x)

    @staticmethod
    def cos(x):
        return _torch.cos(x)

    @staticmethod
    def sin(x):
        return _torch.sin(x)

    @staticmethod
    def atan2(a, b):
        return _torch.atan2(a, b)

    @staticmethod
    def abs(x):
        return _torch.abs(x)

    @staticmethod
    def where(c, a, b):
        return _torch.where(c, a, b)

    @staticmethod
    def ones_like(x):
        return _torch.ones_like(x)

    @staticmethod
    def zeros_like(x):
        return _torch.zeros_like(x)

    @staticmethod
    def power(x, p):
        return _torch.pow(x, p)

    @staticmethod
    def clip(x, lo=None, hi=None):
        return _torch.clamp(x, min=lo, max=hi)

    @staticmethod
    def ndim(x):
        return x.dim()


def get_backend(x):
    if HAS_TORCH and isinstance(x, _torch.Tensor):
        return TorchBackend
    return NumpyBackend
