from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["SourceSISR"]


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class _Residual(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), _gn(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), _gn(c),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.body(x))


class SourceSISR(nn.Module):

    def __init__(self, in_ch: int = 2, width: int = 64, depth: int = 6,
                 mag: int = 2, n_up: int = 1):
        super().__init__()
        c = width
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c, 3, padding=1), _gn(c), nn.SiLU())
        self.body = nn.Sequential(*[_Residual(c) for _ in range(depth)])
        self.merge = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _gn(c))
        up = []
        for _ in range(n_up):
            up += [nn.Conv2d(c, c * mag * mag, 3, padding=1),
                   nn.PixelShuffle(mag), _gn(c), nn.SiLU()]
        self.up = nn.Sequential(*up)
        self.head = nn.Conv2d(c, 1, 3, padding=1)

        # small random init on the head, not zeros: a zero-initialised head can sit at its neutral output for the whole run
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.stem(x)
        h = h + self.merge(self.body(h))          # long skip as in SRResNet
        return torch.nn.functional.softplus(self.head(self.up(h)))[:, 0]