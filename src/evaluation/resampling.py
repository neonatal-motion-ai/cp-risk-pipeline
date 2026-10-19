"""Bootstrap confidence intervals and permutation tests for AUC."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .metrics import auc


def bootstrap_auc(y: Sequence[int], s: Sequence[float], n_boot: int = 5000,
                  seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI of the AUC (cases resampled with replacement).

    Resamples containing a single class are skipped (and counted in `n_valid`).
    """
    y, s = np.asarray(y, int), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auc(y[idx], s[idx]))
    vals = np.asarray(vals)
    return {"auc": auc(y, s),
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_valid": int(vals.size)}


def paired_bootstrap_delta(y_a: Sequence[int], s_a: Sequence[float],
                           y_b: Sequence[int], s_b: Sequence[float],
                           n_boot: int = 5000, seed: int = 42) -> Dict[str, float]:
    """Paired bootstrap of AUC_b - AUC_a when both refer to the same cases."""
    y_a, s_a = np.asarray(y_a, int), np.asarray(s_a, float)
    y_b, s_b = np.asarray(y_b, int), np.asarray(s_b, float)
    rng = np.random.default_rng(seed)
    n = len(y_a)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_a[idx])) < 2 or len(np.unique(y_b[idx])) < 2:
            continue
        d.append(auc(y_b[idx], s_b[idx]) - auc(y_a[idx], s_a[idx]))
    d = np.asarray(d)
    return {"delta": auc(y_b, s_b) - auc(y_a, s_a),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def _null_summary(null: np.ndarray, observed: float) -> Dict[str, float]:
    return {"observed": float(observed), "n_perm": int(null.size),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)),
            "null_p97_5": float(np.percentile(null, 97.5)),
            "null_p99": float(np.percentile(null, 99)),
            "null_max": float(null.max()),
            # one-sided p with the +1 correction (never exactly 0) ...
            "p_value": float((np.sum(null >= observed) + 1) / (null.size + 1)),
            # ... and the plain proportion used in some of the original analyses
            "p_value_raw": float(np.mean(null >= observed))}


def label_permutations(y: Sequence[int], n_perm: int, seed: int):
    """The n_perm permuted label vectors, generated in order with
    `np.random.default_rng(seed)` (copy + shuffle), as in the original analyses."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        y_s = np.array(y, dtype=int, copy=True)
        rng.shuffle(y_s)
        out.append(y_s)
    return out


def permutation_test_fixed_score(y: Sequence[int], s: Sequence[float],
                                 n_perm: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Label-permutation test on a FIXED score.

    Exact for scores that never saw labels (label-free components). For a score
    produced by models trained on the labels it is anti-conservative, because it
    does not refit the models (see `permutation_null_with_retraining`).
    """
    y, s = np.asarray(y, int), np.asarray(s, float)
    null = np.array([auc(y_p, s) for y_p in label_permutations(y, n_perm, seed)])
    return _null_summary(null, auc(y, s))


def permutation_null_with_retraining(X: np.ndarray, y: Sequence[int],
                                     oof_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     n_perm: int = 1000, seed: int = 0,
                                     observed: Optional[float] = None) -> Dict[str, float]:
    """Null distribution of the out-of-fold AUC when the model is REFIT on permuted labels.

    `oof_fn(X, y_perm)` must return out-of-fold scores for every case (for
    example `cv.oof_predictions` with a fixed fold seed). This is the
    conservative null for supervised components: it captures how high an AUC
    a model can reach by fitting noise at this sample size.
    """
    X, y = np.asarray(X, float), np.asarray(y, int)
    null = []
    for y_perm in label_permutations(y, n_perm, seed):   # all permutations drawn up front
        null.append(auc(y_perm, oof_fn(X, y_perm)))
    null = np.asarray(null)
    if observed is None:
        observed = auc(y, oof_fn(X, y))
    return _null_summary(null, observed)
