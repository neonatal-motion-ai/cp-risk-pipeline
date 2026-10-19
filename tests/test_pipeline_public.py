import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import generate  # noqa: E402
from scripts.reproduce_metrics import evaluate, load_scores  # noqa: E402
from src.interfaces import (COMPONENTS, FEATURE_DIM, SEQ_DIM, PoseSequence,  # noqa: E402
                            RiskComponent)
from src.metrics import compute_quality  # noqa: E402

SYNTH = ROOT / "data" / "synthetic_example" / "scores_synthetic.csv"


def test_pose_quality():
    still = np.tile(np.linspace(0.1, 0.9, 32), (50, 1))
    q = compute_quality(still)
    assert q["sjr"] == 0.0 and q["tss"] == 1.0 and q["frames"] == 50
    jump = still.copy()
    jump[25:, 0] += 0.2                     # one keypoint jumps once
    q = compute_quality(jump)
    assert q["sjr"] == pytest.approx(100.0 / (49 * 16))
    assert q["tss"] < 1.0


def test_contracts():
    assert SEQ_DIM == 42 and FEATURE_DIM == 60 and len(COMPONENTS) == 7
    PoseSequence(np.zeros((10, 32)))
    with pytest.raises(ValueError):
        PoseSequence(np.zeros((10, 30)))
    with pytest.raises(TypeError):
        RiskComponent()                     # abstract: no implementation shipped


def test_synthetic_generator_is_deterministic_and_matches_schema():
    a, b = generate(seed=7), generate(seed=7)
    assert a.equals(b)
    cols = ["case_id"] + [f"score_{k}" for k in range(1, 8)] + \
           ["label_primary", "label_secondary", "label_consensus", "label_majority"]
    assert list(a.columns) == cols
    assert a["case_id"].str.startswith("synth_").all()


def test_reproduce_metrics_on_shipped_synthetic_file():
    df = load_scores(SYNTH)
    res = evaluate(df, n_boot=200, n_perm=100)
    assert res["n"] == len(df)
    assert set(res["references"]) == {"primary", "secondary", "consensus", "majority"}
    op = res["operating_point"]["rank_all (7) vs primary"]
    assert op["tp"] + op["fn"] == int(df["label_primary"].sum())
    assert -1 <= res["agreement"]["rater mean_all (7)"]["vs_primary"]["kappa"] <= 1
    for name in ("rank_all (7)", "mean_all (7)", "rank label-free (4)", "mean label-free (4)"):
        assert 0.0 <= res["ensembles"][name]["primary"]["auc"] <= 1.0


def test_reproduce_metrics_cli_end_to_end(tmp_path):
    out = tmp_path / "res.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "reproduce_metrics.py"),
                        "--scores", str(SYNTH), "--n-boot", "100", "--n-perm", "50",
                        "--json", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "rank_all (7)" in r.stdout and out.exists()


def test_compare_reference_standards_cli():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_reference_standards.py"),
                        "--scores-a", str(SYNTH), "--label-a", "label_primary",
                        "--scores-b", str(SYNTH), "--label-b", "label_primary",
                        "--n-boot", "100"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Delta (B - A) = +0.0000" in r.stdout


def test_hyperparameters_yaml_is_valid():
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "hyperparameters.yaml").read_text(encoding="utf-8"))
    assert cfg["temporal_encoder"]["d_model"] == 128
    assert cfg["evaluation"]["cross_validation"]["random_state"] == 42
    assert len(cfg["components"]) == 7


def test_spanish_column_names_are_accepted(tmp_path):
    df = load_scores(SYNTH)
    es = df.rename(columns={"case_id": "case_id_anonimo", "label_primary": "label_primario",
                            "label_consensus": "label_consenso",
                            **{f"score_{k}": f"score_componente_{k}" for k in range(1, 8)}})
    es = es.drop(columns=["label_secondary", "label_majority"])
    path = tmp_path / "es.csv"
    es.to_csv(path, index=False)
    a = evaluate(load_scores(path), n_boot=50, n_perm=20)
    b = evaluate(df, n_boot=50, n_perm=20)
    assert a["ensembles"]["rank_all (7)"]["primary"]["auc"] == b["ensembles"]["rank_all (7)"]["primary"]["auc"]
    assert "consensus" in a["references"]
