"""Feature loading and label-free standardization (protocol 3.3, 3.4)."""
from __future__ import annotations

import numpy as np

from .data_sciq import load_items
from .extract import load_cached

REL_SD_FLOOR = 1e-6  # a feature whose SD is below 1e-6 x max(|mean|, 1) counts as constant


class Standardizer:
    """Per-dimension mean / SD from the unlabeled train pool (all options of all train questions).

    Constant features (e.g. every optical pixel when beta -> 0) are mapped to exactly 0 instead of
    dividing float rounding noise by a tiny SD.
    """

    def fit(self, X: np.ndarray) -> "Standardizer":
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        self.mean_ = flat.mean(0)
        sd = flat.std(0)
        self.constant_ = sd <= REL_SD_FLOOR * np.maximum(np.abs(self.mean_), 1.0)
        self.sd_ = np.where(self.constant_, np.inf, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.sd_).astype(np.float32)


def load_sciq_features(model_tag: str, layer: str, pooling: str, splits=("train", "dev")) -> dict:
    """Return {split: (X of shape (n, 4, d), y of shape (n,), ids)} with raw (unstandardized) features."""
    out = {}
    for s in splits:
        items = load_items(s)
        X = load_cached(model_tag, layer, pooling, s)
        if X.shape[0] != len(items):
            raise RuntimeError(f"cache/{model_tag}/{layer}_{pooling}/{s}.npy has {X.shape[0]} rows, split has {len(items)}")
        y = np.array([it["label"] for it in items], dtype=np.int64)
        out[s] = (X, y, [it["id"] for it in items])
    return out
