#!/usr/bin/env python3
"""Recompute the article's evaluation tables from a per-case score file.

Input CSV schema (one row per case):
    case_id, score_1, ..., score_7, label_primary
    [, label_secondary][, label_consensus][, label_majority]

- score_1..score_4 : label-free components (mse_ft, entropy_score_ft,
                     sampen_vel_mean_ft, corr_sym_mean_ft)
- score_5..score_7 : supervised components (out-of-fold scores)
- label_primary    : binary GMA label of the primary evaluator (1 = CP risk)
- label_secondary  : binary label of the secondary evaluator (empty if not rated)
- label_consensus  : label where both evaluators agree, empty otherwise
- label_majority   : majority of three evaluators (empty if not available)

Equivalent Spanish headers are accepted: case_id_anonimo, score_componente_1..7,
label_primario, label_secundario, label_consenso, label_mayoria.

Outputs: component and ensemble AUCs against every available reference with
bootstrap CIs, fixed-score permutation tests, the parametric distance of each
AUC from the null, the F1-optimal operating point, inter-rater kappa and the
kappa of the binarized ensemble ("computational rater") against each evaluator.

Usage:
    python scripts/reproduce_metrics.py --scores data/synthetic_example/scores_synthetic.csv
    python scripts/reproduce_metrics.py --scores <restricted.csv> --json out.json

The per-case score files of the article are NOT public; see README
("Data availability").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (auc, bootstrap_auc, bootstrap_kappa,  # noqa: E402
                            f1_optimal_operating_point, fuse_mean, fuse_rank, orient,
                            parametric_z, permutation_test_fixed_score,
                            prevalence_matched_rater, stack)

SCORE_COLS = [f"score_{k}" for k in range(1, 8)]
LABEL_FREE = SCORE_COLS[:4]
SUPERVISED = SCORE_COLS[4:]
OPTIONAL_LABELS = ("label_secondary", "label_consensus", "label_majority")
# Equivalent Spanish column names (schema as specified in the project brief)
SPANISH_COLUMNS = {"case_id_anonimo": "case_id", "label_primario": "label_primary",
                   "label_secundario": "label_secondary", "label_consenso": "label_consensus",
                   "label_mayoria": "label_majority",
                   **{f"score_componente_{k}": f"score_{k}" for k in range(1, 8)}}


def load_scores(path):
    df = pd.read_csv(path).rename(columns=SPANISH_COLUMNS)
    missing = [c for c in ["case_id"] + SCORE_COLS + ["label_primary"] if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] missing columns: {missing}")
    for c in OPTIONAL_LABELS:
        if c not in df.columns:
            df[c] = np.nan
    if df["case_id"].duplicated().any():
        raise SystemExit("[ERROR] duplicated case_id")
    return df


def ensembles(scores):
    """Fused scores computed over the whole cohort (before any subsetting)."""
    # mean_all: label-free descriptors min-max scaled, supervised probabilities as-is
    scale_all = [c in LABEL_FREE for c in SCORE_COLS]
    return {
        "rank_all (7)": fuse_rank(stack(scores, SCORE_COLS)),
        "mean_all (7)": fuse_mean(stack(scores, SCORE_COLS), scale=scale_all),
        "rank label-free (4)": fuse_rank(stack(scores, LABEL_FREE)),
        "mean label-free (4)": fuse_mean(stack(scores, LABEL_FREE)),
        "rank supervised (3)": fuse_rank(stack(scores, SUPERVISED)),
    }


def _reference(df, col):
    m = df[col].notna().to_numpy()
    return np.where(m, df[col].fillna(0), 0).astype(int), m


def evaluate(df, n_boot=5000, n_perm=1000, seed_boot=42, seed_perm=0, null_sd=0.14):
    y = df["label_primary"].to_numpy(int)
    scores, signs = orient({c: df[c].to_numpy(float) for c in SCORE_COLS}, y)
    fused = ensembles(scores)

    refs = {"primary": (y, np.ones(len(y), bool))}
    for col in OPTIONAL_LABELS:
        if df[col].notna().any():
            refs[col.replace("label_", "")] = _reference(df, col)

    out = {"n": int(len(y)), "positives_primary": int(y.sum()),
           "orientation": signs, "references": {}, "components": {}, "ensembles": {},
           "permutation": {}, "parametric_null": {}, "operating_point": {},
           "agreement": {}}
    for r, (yy, m) in refs.items():
        out["references"][r] = {"n": int(m.sum()), "positives": int(yy[m].sum())}
    for c in SCORE_COLS:
        out["components"][c] = {r: auc(yy, scores[c], m) for r, (yy, m) in refs.items()}
    for name, s in fused.items():
        out["ensembles"][name] = {}
        for r, (yy, m) in refs.items():
            out["ensembles"][name][r] = bootstrap_auc(yy[m], s[m], n_boot=n_boot, seed=seed_boot)
            out["parametric_null"].setdefault(name, {})[r] = parametric_z(
                out["ensembles"][name][r]["auc"], null_sd=null_sd)
    for name in ("rank_all (7)", "rank label-free (4)", "mean label-free (4)"):
        out["permutation"][name] = permutation_test_fixed_score(y, fused[name], n_perm=n_perm,
                                                                seed=seed_perm)
    out["operating_point"]["rank_all (7) vs primary"] = f1_optimal_operating_point(
        y, fused["rank_all (7)"])

    if "secondary" in refs:
        y2, m = refs["secondary"]
        out["agreement"]["n_both"] = int(m.sum())
        out["agreement"]["n_disagree"] = int((y[m] != y2[m]).sum())
        out["agreement"]["primary_vs_secondary"] = bootstrap_kappa(y[m], y2[m], n_boot, seed_boot)
        # computational rater: ensemble binarized at the evaluators' mean prevalence
        prev = (y[m].mean() + y2[m].mean()) / 2
        for name in ("mean_all (7)", "rank_all (7)"):
            rater = prevalence_matched_rater(fused[name][m], prev)
            out["agreement"][f"rater {name}"] = {
                "positives": int(rater.sum()),
                "vs_primary": bootstrap_kappa(rater, y[m], n_boot, seed_boot),
                "vs_secondary": bootstrap_kappa(rater, y2[m], n_boot, seed_boot)}
    return out


def print_report(res, title):
    w = 78
    print("=" * w)
    print(title)
    print("=" * w)
    refs = list(res["references"])
    print("n = %d | positives (primary) = %d" % (res["n"], res["positives_primary"]))
    for r, v in res["references"].items():
        print("  reference %-10s n = %2d  positives = %2d" % (r, v["n"], v["positives"]))
    print("\nComponent AUC (oriented; sign fixed against the primary evaluator)")
    print("  %-10s %5s " % ("component", "sign") + "".join("%11s" % r for r in refs))
    for c, v in res["components"].items():
        print("  %-10s %+5d " % (c, res["orientation"][c]) +
              "".join("%11.4f" % v[r] for r in refs))
    n_valid = res["ensembles"]["rank_all (7)"]["primary"]["n_valid"]
    print("\nEnsemble AUC [95%% bootstrap CI, %d valid resamples]" % n_valid)
    for name, v in res["ensembles"].items():
        print("  %s" % name)
        for r in refs:
            z = res["parametric_null"][name][r]
            print("      %-10s %.4f [%.3f, %.3f]   z=%.2f p(normal)=%.3f"
                  % (r, v[r]["auc"], v[r]["ci_low"], v[r]["ci_high"], z["z"], z["p_one_sided"]))
    print("\nPermutation test on the fixed fused score vs primary (one-sided)")
    for name, p in res["permutation"].items():
        print("  %-22s AUC=%.4f  n_perm=%d  null mean=%.3f sd=%.3f p95=%.3f p99=%.3f  p=%.4f"
              % (name, p["observed"], p["n_perm"], p["null_mean"], p["null_sd"],
                 p["null_p95"], p["null_p99"], p["p_value"]))
    print("\nF1-optimal operating point")
    for name, o in res["operating_point"].items():
        print("  %s: thr=%.4f acc=%.2f sens=%.2f (%d/%d) spec=%.2f (%d/%d) F1=%.2f"
              % (name, o["threshold"], o["accuracy"], o["sensitivity"], o["tp"], o["tp"] + o["fn"],
                 o["specificity"], o["tn"], o["tn"] + o["fp"], o["f1"]))
    a = res["agreement"]
    if a:
        k = a["primary_vs_secondary"]
        print("\nAgreement (Cohen's kappa, binary) on %d doubly-rated cases (%d disagreements)"
              % (a["n_both"], a["n_disagree"]))
        print("  primary vs secondary          %.3f [%.3f, %.3f]" % (k["kappa"], k["ci_low"], k["ci_high"]))
        for key, v in a.items():
            if key.startswith("rater"):
                for r in ("vs_primary", "vs_secondary"):
                    print("  %-22s %-12s %.3f [%.3f, %.3f]  (rater positives=%d)"
                          % (key, r, v[r]["kappa"], v[r]["ci_low"], v[r]["ci_high"], v["positives"]))
    print("=" * w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="per-case score CSV")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed-boot", type=int, default=42)
    ap.add_argument("--seed-perm", type=int, default=0)
    ap.add_argument("--null-sd", type=float, default=0.14,
                    help="sd of the AUC null used for the parametric z (article: 0.14)")
    ap.add_argument("--json", help="optional path to write all results as JSON")
    args = ap.parse_args(argv)

    df = load_scores(args.scores)
    res = evaluate(df, n_boot=args.n_boot, n_perm=args.n_perm, seed_boot=args.seed_boot,
                   seed_perm=args.seed_perm, null_sd=args.null_sd)
    print_report(res, "Evaluation of %s" % Path(args.scores).name)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
