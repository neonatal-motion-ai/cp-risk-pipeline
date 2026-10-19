#!/usr/bin/env python3
"""Generate a SYNTHETIC per-case score file with the same schema as the real one.

Everything is drawn from parametric distributions (no real data is read,
sampled, perturbed or averaged). The cases are fictitious.

Model: each case has a latent risk z ~ N(0, 1). The primary label is
1[z + noise > threshold] with a prevalence of ~30 %; the secondary evaluator
and a third (adjudicating) evaluator re-label with independent noise (moderate
agreement); the majority label is the vote of the three. Each of the 7 component
scores is a noisy monotone function of z with a component-specific signal level;
some components are generated with inverted direction to exercise orientation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (signal weight, direction) per component; small weights ~ chance-level components
COMPONENTS = [(0.2, +1), (0.6, +1), (0.8, -1), (0.3, +1), (0.9, +1), (0.1, +1), (0.1, -1)]


def generate(n=36, prevalence=0.30, rater_noise=0.9, seed=2026):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    thr = np.quantile(z, 1 - prevalence)
    y1 = (z + rng.normal(scale=0.4, size=n) > thr).astype(int)
    y2 = (z + rng.normal(scale=rater_noise, size=n) > thr).astype(int)
    y3 = (z + rng.normal(scale=rater_noise, size=n) > thr).astype(int)   # adjudicator
    rated2 = rng.random(n) > 0.06                   # a few cases without second rating
    cols = {"case_id": [f"synth_{i:03d}" for i in range(1, n + 1)]}
    for k, (wgt, sign) in enumerate(COMPONENTS, 1):
        raw = sign * wgt * z + rng.normal(size=n)
        cols[f"score_{k}"] = np.round(1.0 / (1.0 + np.exp(-raw)), 6)   # squash to (0, 1)
    cols["label_primary"] = y1
    cols["label_secondary"] = np.where(rated2, y2, np.nan)
    cols["label_consensus"] = np.where(rated2 & (y1 == y2), y1, np.nan)
    cols["label_majority"] = np.where(rated2, ((y1 + y2 + y3) >= 2).astype(int), np.nan)
    df = pd.DataFrame(cols)
    for c in ("label_secondary", "label_consensus", "label_majority"):
        df[c] = df[c].astype("Int64")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent /
                                         "data" / "synthetic_example" / "scores_synthetic.csv"))
    args = ap.parse_args()
    df = generate(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print("[OK] synthetic scores -> %s  (n=%d, positives=%d)"
          % (args.out, len(df), int(df["label_primary"].sum())))


if __name__ == "__main__":
    main()
