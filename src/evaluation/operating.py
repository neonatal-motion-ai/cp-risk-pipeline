"""Threshold-dependent metrics: F1-optimal operating point and a binarized
"computational rater" compared with human evaluators by Cohen's kappa."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_curve


def f1_optimal_operating_point(y: Sequence[int], s: Sequence[float]) -> Dict[str, float]:
    """Threshold that maximizes F1 along the precision-recall curve; case is
    positive when s >= threshold. Returns accuracy, sensitivity, specificity, F1."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
    t = float(thr[int(np.argmax(f1))]) if f1.size else 0.5
    pred = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": t, "accuracy": (tp + tn) / len(y),
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp)}


def prevalence_matched_rater(s: Sequence[float], prevalence: float) -> np.ndarray:
    """Binarize a score so that the fraction of positives matches `prevalence`
    (threshold = quantile 1 - prevalence; s >= threshold is positive)."""
    s = np.asarray(s, float)
    return (s >= np.quantile(s, 1 - prevalence)).astype(int)


def bootstrap_kappa(a: Sequence[int], b: Sequence[int], n_boot: int = 5000,
                    seed: int = 42, weights=None) -> Dict[str, float]:
    """Cohen's kappa with a percentile bootstrap 95 % CI (resamples where either
    rater is constant are skipped). `weights`: None, "linear" or "quadratic"."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(a[idx])) < 2 or len(np.unique(b[idx])) < 2:
            continue
        vals.append(cohen_kappa_score(a[idx], b[idx], weights=weights))
    vals = np.asarray(vals)
    return {"kappa": float(cohen_kappa_score(a, b, weights=weights)),
            "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}


def parametric_z(auc_value: float, null_sd: float = 0.14, null_mean: float = 0.5) -> Dict[str, float]:
    """Distance of an AUC from a normal approximation of the permutation null:
    z = (AUC - mean) / sd and one-sided p = 1 - Phi(z)."""
    z = (auc_value - null_mean) / null_sd
    return {"z": float(z), "p_one_sided": float(norm.sf(z))}
