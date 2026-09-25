from __future__ import annotations
import csv
import glob
import itertools
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

__all__ = ["ModelADataset", "SCALAR_KEYS"]

SCALAR_KEYS = ("theta_E", "gamma1_ext", "gamma2_ext", "host_e1", "host_e2", "host_slope", "source_x", "source_y",
               "source_R_sersic", "source_n_sersic", "source_e1", "source_e2", "snr_max", "z_lens", "z_source", "num_subhalos")


def _scalar(z, key, default=np.nan):
    try:
        return float(np.asarray(z[key]))
    except Exception:
        return default


class ModelADataset:

    def __init__(self, root: str = ".", split: str = "val",
                 classes: Optional[List[str]] = None, band: int = 0,
                 limit: Optional[int] = None,
                 max_host_e: Optional[float] = None,
                 min_host_e: Optional[float] = None,
                 min_snr: Optional[float] = None,
                 max_snr: Optional[float] = None,
                 bg_radius_px: float = 50.0,
                 manifest: Optional[str] = None,
                 model: str = "Model_A"):
        self.root = Path(root)
        self.band = int(band)
        self.bg_radius_px = float(bg_radius_px)
        classes = classes or ["axion", "cdm", "wdm"]

        if manifest:
            paths = []
            with open(manifest, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("model") != model or r.get("split") != split:
                        continue
                    if r.get("class") not in classes:
                        continue
                    paths.append(self.root / r["path"])
            paths.sort()
        else:
            per_class = [sorted(glob.glob(str(self.root / model / split / c / "*.npz")))
                         for c in classes]
            paths = [Path(p) for p in
                     itertools.chain.from_iterable(itertools.zip_longest(*per_class))
                     if p is not None]
        if not paths:
            raise FileNotFoundError(
                f"no .npz under {self.root/model/split} for classes={classes}")

        # read scalars once, then filter
        keep, meta = [], []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                m = {k: _scalar(z, k) for k in SCALAR_KEYS}
            m["host_e"] = float(np.hypot(m["host_e1"], m["host_e2"]))
            m["gamma_ext"] = float(np.hypot(m["gamma1_ext"], m["gamma2_ext"]))
            m["beta"] = float(np.hypot(m["source_x"], m["source_y"]))
            if max_host_e is not None and not (m["host_e"] <= max_host_e):
                continue
            if min_host_e is not None and not (m["host_e"] >= min_host_e):
                continue
            if min_snr is not None and not (m["snr_max"] >= min_snr):
                continue
            if max_snr is not None and not (m["snr_max"] <= max_snr):
                continue
            keep.append(p)
            meta.append(m)
            if limit and len(keep) >= limit:
                break
        self.paths, self.meta = keep, meta
        if not self.paths:
            raise ValueError("every image was filtered out; loosen the cuts")

        # Realised per-class counts. The cuts above can reject unevenly, so
        # the classes going in are not necessarily the classes coming out.
        self.classes = list(classes)
        self.class_counts = Counter(p.parent.name for p in self.paths)

    def __len__(self):
        return len(self.paths)

    # index guard
    def assert_rows_match(self, rows, what: str = "rows") -> int:
        def tail(p) -> tuple:
            parts = [q for q in str(p).replace("\\", "/").split("/") if q]
            return tuple(parts[-2:])

        checked, outside, bad = 0, 0, []
        for r in (rows.values() if isinstance(rows, dict) else rows):
            i, p = r.get("index"), r.get("path")
            if i is None or p is None:
                continue
            if not (0 <= int(i) < len(self.paths)):
                outside += 1
                continue
            checked += 1
            have = self.paths[int(i)]
            if tail(p) != tail(have):
                bad.append((i, str(p), str(have)))
                if len(bad) >= 5:
                    break
        if bad:
            lines = "\n".join(f"    index {i}: json says {w}, dataset has {h}"
                              for i, w, h in bad)
            raise RuntimeError(
                f"{what}: index/path mismatch on {len(bad)}+ of {checked} rows.\n"
                f"{lines}\n"
                "  The JSON was written under a different dataset ordering. Re-run\n"
                "  the step that produced it rather than re-scoring it, or the\n"
                "  numbers will be silently wrong. See ModelADataset's docstring.")
        if checked == 0:
            raise RuntimeError(
                f"{what}: none of the {outside} rows carries an index inside this\n"
                f"  dataset (len={len(self.paths)}), so nothing was verified and\n"
                "  nothing may be scored against it. Either the JSON was written for\n"
                "  a different split/class list, or this dataset was built with\n"
                "  filters that removed every image the JSON refers to.")
        return checked

    # pixel data
    def _band(self, a: np.ndarray) -> np.ndarray:
        a = np.squeeze(np.asarray(a))
        if a.ndim == 3:
            a = a[min(self.band, a.shape[0] - 1)]
        return a.astype(np.float64)

    def image(self, i: int) -> np.ndarray:
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["image"])

    def image_nss(self, i: int) -> np.ndarray:
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["image_nss"])

    def sigma(self, i: int, img: Optional[np.ndarray] = None) -> float:
        a = self.image(i) if img is None else img
        n = a.shape[-1]
        c = (n - 1) / 2.0
        yy, xx = np.indices(a.shape[-2:])
        bg = np.hypot(yy - c, xx - c) > self.bg_radius_px
        return float(a[bg].std())

    # truth: EVALUATION ONLY
    def truth(self, i: int) -> Dict[str, float]:
        return dict(self.meta[i])

    def unlensed(self, i: int) -> np.ndarray:
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["unlensed"])

    def kappa_sub(self, i: int) -> np.ndarray:
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["kappa_sub"])

    def summary(self) -> str:
        m = self.meta
        g = lambda k: np.array([x[k] for x in m], dtype=float)
        by_class = " ".join(f"{c}={self.class_counts.get(c, 0)}"
                            for c in self.classes)
        return (f"{len(m)} images   [{by_class}]   "
                f"theta_E {np.median(g('theta_E')):.2f}\"   "
                f"|e| {np.median(g('host_e')):.3f}   "
                f"snr_max {np.median(g('snr_max')):.1f}   "
                f"beta {np.median(g('beta')):.3f}\"")