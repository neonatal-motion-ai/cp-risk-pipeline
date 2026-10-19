"""Stratified K-fold out-of-fold (OOF) predictions."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_folds(y_strat: Sequence[int], n_splits: int = 5, seed: int = 42):
    """Folds used in the article: StratifiedKFold(5, shuffle=True, random_state=42),
    stratified on the primary evaluator's binary label."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_strat = np.asarray(y_strat, int)
    return list(skf.split(np.zeros(len(y_strat)), y_strat))


def oof_predictions(X: np.ndarray, y_target: Sequence[float], factory: Callable[[], object],
                    y_strat: Optional[Sequence[int]] = None, n_splits: int = 5,
                    seed: int = 42, method: str = "auto") -> np.ndarray:
    """Out-of-fold scores: every case is scored by a model that never saw it.

    factory: returns a fresh unfitted sklearn-style estimator.
    y_target: training target (binary, or continuous soft labels in [0, 1]).
    method: "proba" (predict_proba[:, 1]), "predict" (clipped to [0, 1]) or "auto".
    """
    X = np.asarray(X, float)
    y_target = np.asarray(y_target, float)
    y_strat = np.asarray(y_target if y_strat is None else y_strat).round().astype(int)
    oof = np.zeros(len(y_target))
    for tr, va in make_folds(y_strat, n_splits, seed):
        model = factory()
        use_proba = method == "proba" or (method == "auto" and hasattr(model, "predict_proba"))
        if use_proba:
            model.fit(X[tr], y_target[tr].astype(int))
            oof[va] = model.predict_proba(X[va])[:, 1]
        else:
            model.fit(X[tr], y_target[tr])
            oof[va] = np.clip(model.predict(X[va]), 0.0, 1.0)
    return oof
