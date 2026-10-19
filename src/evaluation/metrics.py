"""Discrimination and agreement metrics."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def auc(y: Sequence[int], s: Sequence[float], mask: Optional[Sequence[bool]] = None) -> float:
    """ROC AUC (ties count 1/2). NaN if the evaluated subset has a single class."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), s))


def kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa between two binary raters (unweighted)."""
    return float(cohen_kappa_score(np.asarray(a, int), np.asarray(b, int)))
