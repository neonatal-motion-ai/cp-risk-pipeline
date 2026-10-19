#!/usr/bin/env python3
"""Compare the rank_all AUC of two runs evaluated against two reference standards
on the same cases (paired bootstrap of the difference).

Article use: run trained/evaluated against the primary evaluator (A) versus the
run trained/evaluated against the three-evaluator majority (B), both n = 34.

Usage:
    python scripts/compare_reference_standards.py \
        --scores-a <run_primary.csv> --label-a label_primary \
        --scores-b <run_majority.csv> --label-b label_majority
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_metrics import SCORE_COLS, load_scores  # noqa: E402
from src.evaluation import (bootstrap_auc, fuse_rank, orient,  # noqa: E402
                            paired_bootstrap_delta, stack)


def fused_rank(df):
    y = df["label_primary"].to_numpy(int)          # orientation fixed against the primary evaluator
    scores, _ = orient({c: df[c].to_numpy(float) for c in SCORE_COLS}, y)
    return fuse_rank(stack(scores, SCORE_COLS))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-a", required=True)
    ap.add_argument("--label-a", default="label_primary")
    ap.add_argument("--scores-b", required=True)
    ap.add_argument("--label-b", default="label_majority")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    a, b = load_scores(args.scores_a), load_scores(args.scores_b)
    a, b = a.sort_values("case_id").reset_index(drop=True), b.sort_values("case_id").reset_index(drop=True)
    if not a["case_id"].equals(b["case_id"]):
        raise SystemExit("[ERROR] the two files must contain the same cases")
    if a[args.label_a].isna().any() or b[args.label_b].isna().any():
        raise SystemExit("[ERROR] the selected labels must be available for every case")
    ya, yb = a[args.label_a].to_numpy(int), b[args.label_b].to_numpy(int)
    sa, sb = fused_rank(a), fused_rank(b)
    ra = bootstrap_auc(ya, sa, args.n_boot, args.seed)
    rb = bootstrap_auc(yb, sb, args.n_boot, args.seed)
    d = paired_bootstrap_delta(ya, sa, yb, sb, args.n_boot, args.seed)
    print("n = %d" % len(ya))
    print("A  %-16s positives=%2d  AUC=%.4f [%.3f, %.3f]" % (args.label_a, ya.sum(), ra["auc"], ra["ci_low"], ra["ci_high"]))
    print("B  %-16s positives=%2d  AUC=%.4f [%.3f, %.3f]" % (args.label_b, yb.sum(), rb["auc"], rb["ci_low"], rb["ci_high"]))
    print("Delta (B - A) = %+.4f  paired bootstrap 95%% CI [%.3f, %.3f]  p(two-sided) = %.3f"
          % (d["delta"], d["ci_low"], d["ci_high"], d["p_two_sided"]))
    return {"a": ra, "b": rb, "delta": d}


if __name__ == "__main__":
    main()
