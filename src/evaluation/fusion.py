"""Score orientation and late fusion of component scores.

All functions operate on per-case scores (one value per case and component);
no model weights or patient data are involved.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from .metrics import auc


def minmax(x: Sequence[float]) -> np.ndarray:
    """Min-max scale to [0, 1] over the whole cohort (constant input -> zeros)."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def orient(scores: Dict[str, Sequence[float]], y: Sequence[int]
           ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Orient each component so that higher = higher risk.

    A component whose AUC against `y` is below 0.5 is replaced by `1 - s`.
    This is the orientation rule used in the article (one sign per component,
    fixed against the primary evaluator over the whole cohort). Applying it to
    already-oriented scores is a no-op.
    """
    out, signs = {}, {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        if auc(y, s) < 0.5:
            out[name], signs[name] = 1.0 - s, -1
        else:
            out[name], signs[name] = s, +1
    return out, signs


def fuse_rank(matrix: np.ndarray) -> np.ndarray:
    """`rank_all` fusion: mean of per-component normalized ranks.

    Ranks use average ranking for ties. The mean is computed from the exact
    rank sums so that two cases with identical rank sums stay exactly tied
    (normalizing each column with floating point before averaging can split
    such ties by one ulp and shift the AUC by one tied pair).
    Output is in (0, 1]; any positive affine rescaling gives the same AUC.
    """
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    rank_sum = np.zeros(n)
    for j in range(k):
        rank_sum += rankdata(m[:, j])          # multiples of 0.5, exact in float
    return rank_sum / (k * n)


def fuse_mean(matrix: np.ndarray, scale: Sequence[bool] = None) -> np.ndarray:
    """`mean_all` fusion: mean of the oriented component scores.

    Columns flagged in `scale` are min-max scaled over the cohort first. In the
    article the label-free components (unbounded descriptors) are scaled and the
    supervised components, already probabilities in [0, 1], are averaged as-is.
    `scale=None` scales every column.
    """
    m = np.asarray(matrix, dtype=float)
    if scale is None:
        scale = [True] * m.shape[1]
    cols = [minmax(m[:, j]) if scale[j] else m[:, j] for j in range(m.shape[1])]
    return np.column_stack(cols).mean(axis=1)


def stack(scores: Dict[str, Sequence[float]], names: List[str]) -> np.ndarray:
    """Columns of `scores` in the order given by `names`."""
    return np.column_stack([np.asarray(scores[n], dtype=float) for n in names])


def percentile_vs_reference(reference: Sequence[float], x: float, orientation: int) -> float:
    """Mid-rank percentile of `x` within a fixed reference distribution.

    Used to score a single new case against a persisted reference cohort
    instead of against the batch it is processed with. orientation: +1 / -1.
    """
    ref = np.asarray(reference, dtype=float)
    if ref.size == 0:
        return 0.5
    frac = (np.sum(ref < x) + 0.5 * np.sum(ref == x)) / ref.size
    return float(1.0 - frac) if orientation < 0 else float(frac)
