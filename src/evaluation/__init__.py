"""Evaluation and statistics: AUC, orientation, fusion, CV folds, bootstrap, permutations."""
from .fusion import fuse_mean, fuse_rank, minmax, orient, percentile_vs_reference, stack
from .metrics import auc, kappa
from .resampling import (bootstrap_auc, paired_bootstrap_delta,
                         permutation_null_with_retraining, permutation_test_fixed_score)
from .cv import make_folds, oof_predictions
from .operating import (bootstrap_kappa, f1_optimal_operating_point, parametric_z,
                        prevalence_matched_rater)

__all__ = [
    "f1_optimal_operating_point", "prevalence_matched_rater", "bootstrap_kappa", "parametric_z",
    "auc", "kappa", "minmax", "orient", "fuse_rank", "fuse_mean", "stack",
    "percentile_vs_reference", "bootstrap_auc", "paired_bootstrap_delta",
    "permutation_test_fixed_score", "permutation_null_with_retraining",
    "make_folds", "oof_predictions",
]
