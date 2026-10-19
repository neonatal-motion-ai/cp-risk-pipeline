import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (bootstrap_kappa, f1_optimal_operating_point, parametric_z,  # noqa: E402
                            prevalence_matched_rater)
from src.evaluation import (auc, bootstrap_auc, fuse_mean, fuse_rank, kappa, make_folds,  # noqa: E402
                            minmax, oof_predictions, orient, paired_bootstrap_delta,
                            percentile_vs_reference, permutation_null_with_retraining,
                            permutation_test_fixed_score)


def test_auc_basic_and_ties():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], mask=[True, True, False, False]) == 1.0


def test_orient_flips_inverted_components_and_is_idempotent():
    y = np.array([0, 0, 1, 1])
    s = {"a": np.array([0.9, 0.8, 0.2, 0.1]), "b": np.array([0.1, 0.2, 0.8, 0.9])}
    o, signs = orient(s, y)
    assert signs == {"a": -1, "b": 1}
    assert auc(y, o["a"]) == 1.0
    o2, signs2 = orient(o, y)
    assert signs2 == {"a": 1, "b": 1}
    np.testing.assert_allclose(o2["a"], o["a"])


def test_fuse_rank_keeps_exact_ties():
    # Two cases end with identical rank sums; the fused values must be equal.
    m = np.array([[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    e = fuse_rank(m)
    assert e[0] == e[1]
    # it is a positive affine transform of the mean normalized rank
    ref = np.column_stack([(rankdata(m[:, j]) - 1) / 3 for j in range(2)]).mean(1)
    np.testing.assert_allclose(np.argsort(np.argsort(e)), np.argsort(np.argsort(ref)))


def test_fuse_mean_scaling():
    m = np.array([[0.0, 0.2], [5.0, 0.4], [10.0, 0.6]])
    np.testing.assert_allclose(fuse_mean(m), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(fuse_mean(m, scale=[True, False]), [0.1, 0.45, 0.8])
    np.testing.assert_allclose(minmax([3, 3, 3]), [0, 0, 0])


def test_percentile_vs_reference():
    ref = [1, 2, 3, 4]
    assert percentile_vs_reference(ref, 2.5, +1) == 0.5
    assert percentile_vs_reference(ref, 3, +1) == pytest.approx(0.625)
    assert percentile_vs_reference(ref, 3, -1) == pytest.approx(0.375)


def test_bootstrap_is_deterministic_and_brackets_point():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(25), np.ones(11)].astype(int)
    s = y * 0.8 + rng.normal(size=36)
    a = bootstrap_auc(y, s, n_boot=300, seed=42)
    b = bootstrap_auc(y, s, n_boot=300, seed=42)
    assert a == b
    assert a["ci_low"] <= a["auc"] <= a["ci_high"]
    d = paired_bootstrap_delta(y, s, y, s, n_boot=200)
    assert d["delta"] == 0.0 and d["ci_low"] == 0.0 and d["ci_high"] == 0.0


def test_permutation_fixed_score():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(25), np.ones(9)].astype(int)
    strong = permutation_test_fixed_score(y, y + rng.normal(scale=0.3, size=34), n_perm=500)
    noise = permutation_test_fixed_score(y, rng.normal(size=34), n_perm=500)
    assert strong["p_value"] < 0.01
    assert noise["p_value"] > 0.01
    assert 1 / 501 <= strong["p_value"] <= 1
    assert abs(noise["null_mean"] - 0.5) < 0.05


def test_folds_and_oof():
    y = np.r_[np.zeros(25), np.ones(11)].astype(int)
    folds = make_folds(y, 5, 42)
    assert len(folds) == 5
    va = np.concatenate([v for _, v in folds])
    assert sorted(va.tolist()) == list(range(36))
    X = np.random.default_rng(3).normal(size=(36, 4)) + y[:, None]
    oof = oof_predictions(X, y, lambda: RandomForestClassifier(n_estimators=20, random_state=0))
    assert oof.shape == (36,) and np.all((oof >= 0) & (oof <= 1))


def test_permutation_with_retraining_runs():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(25), np.ones(9)].astype(int)
    X = rng.normal(size=(34, 3))
    fn = lambda X_, y_: oof_predictions(X_, y_, lambda: RandomForestClassifier(
        n_estimators=10, random_state=0))
    res = permutation_null_with_retraining(X, y, fn, n_perm=15, seed=0)
    assert res["n_perm"] == 15 and 0 <= res["null_mean"] <= 1


def test_kappa():
    assert kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


def test_operating_point_and_rater():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.6, 0.9])
    op = f1_optimal_operating_point(y, s)
    assert op["tp"] + op["fn"] == 2 and op["tn"] + op["fp"] == 4
    assert 0 <= op["f1"] <= 1
    r = prevalence_matched_rater(s, 2 / 6)
    assert r.sum() == 2 and r[-1] == 1
    k = bootstrap_kappa(y, y, n_boot=50)
    assert k["kappa"] == 1.0
    z = parametric_z(0.684, 0.14)
    assert abs(z["z"] - 1.314) < 1e-3 and abs(z["p_one_sided"] - 0.094) < 1e-3
