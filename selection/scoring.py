"""Minimal scoring helpers for layer selection."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def label_correlation(scores: np.ndarray, y: Sequence[int]) -> np.ndarray:
    """Compute absolute correlation between PC scores and labels (binary or multi-class)."""
    y_arr = np.asarray(y)
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(np.unique(y_arr)) <= 2:
        yv = y_arr.astype(float)
        corrs = []
        for j in range(scores.shape[1]):
            s = scores[:, j]
            if s.std() < 1e-8 or yv.std() < 1e-8:
                corrs.append(0.0)
            else:
                corrs.append(abs(np.corrcoef(s, yv)[0, 1]))
        return np.array(corrs, dtype=np.float32)

    K = int(np.max(y_arr)) + 1
    corrs = []
    for j in range(scores.shape[1]):
        s = scores[:, j]
        vals = []
        for k in range(K):
            yk = (y_arr == k).astype(float)
            if s.std() < 1e-8 or yk.std() < 1e-8:
                vals.append(0.0)
            else:
                vals.append(abs(np.corrcoef(s, yk)[0, 1]))
        corrs.append(max(vals))
    return np.array(corrs, dtype=np.float32)
