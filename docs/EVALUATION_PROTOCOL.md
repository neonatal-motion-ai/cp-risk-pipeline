# Evaluation protocol

How every performance figure of the article was computed, and how to recompute
it with this repository. The statistics are implemented in `src/evaluation/`
and driven by `scripts/reproduce_metrics.py` and
`scripts/compare_reference_standards.py`.

## 1. Reference standards

| Reference | Cases | At risk | Definition |
|---|---|---|---|
| Primary evaluator | 36 | 11 | Binary GMA label of the primary evaluator (METHODS 1). Main reference. |
| Primary evaluator, doubly-rated cohort | 34 | 9 | Same label restricted to the 34 cases also rated by the secondary evaluator. |
| Secondary evaluator | 34 (of 36 assessed) | 18 | Independent binary label on the common cohort. |
| Consensus | 21 | 7 | Cases where both evaluators agree. Excludes the hardest cases, so it is an upper bound, not a population AUC. |
| Majority of three | 34 | 14 | The 13 disputed cases re-assessed blindly by a third evaluator (8 resolved as the primary evaluator, 5 as the secondary). |

## 2. Cross-validation and out-of-fold scores

* `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified on
  the primary evaluator's binary label (`src/evaluation/cv.py::make_folds`).
* Supervised components (5-7) are trained on 4 folds and score the 5th; every
  case receives exactly one out-of-fold (OOF) score. Threshold-dependent
  metrics are computed only after all OOF scores exist.
* Label-free components (1-4) need no training and are rank-identical in every
  run (they were recomputed per run; small numerical differences, e.g. in the
  entropy score, do not change any rank or AUC).

## 3. Orientation and fusion

* Orientation: a component is flipped (1 - s) when its AUC against the primary
  evaluator over the evaluated cohort is below 0.5 (`fusion.orient`).
* `rank_all` (reported): mean of the per-component ranks, average ranks for ties
  (`fusion.fuse_rank`). Cases with identical rank sums are exact ties and count
  1/2 in the AUC (see note A in section 10).
* `mean_all`: mean of the oriented scores, label-free components min-max scaled
  over the cohort, supervised probabilities as-is (`fusion.fuse_mean`).
* Fusion is always computed over the whole cohort first; subsets (e.g. the
  consensus cases) are selected afterwards.

## 4. AUC and confidence intervals

* AUC = area under the ROC curve (scikit-learn `roc_auc_score`; ties count 1/2).
* 95 % CI: percentile bootstrap, 5 000 resamples of cases with replacement,
  NumPy `default_rng(42)`; resamples with a single class are skipped
  (`resampling.bootstrap_auc`). The original analyses drew several intervals
  from one shared random stream, so recomputed interval bounds and bootstrap
  p-values can differ slightly (typically in the second or third decimal);
  point estimates are exact.
* Difference between two reference standards on the same cases: paired
  bootstrap of the AUC difference (same resampled indices for both;
  `resampling.paired_bootstrap_delta`), two-sided p = 2 min(P(d <= 0), P(d >= 0)).

## 5. Permutation tests and distance from the null

| Null | How | Applies to | Values reported |
|---|---|---|---|
| Fixed score | Permute the labels and recompute the AUC of a fixed score. Permutations drawn in order with `numpy.random.default_rng(seed)` (copy + shuffle; `resampling.label_permutations`). One-sided p reported both as (#{null >= observed} + 1) / (N + 1) (`p_value`) and as the plain proportion #{null >= observed} / N (`p_value_raw`, used in the original analysis); null sd with ddof = 1. | Label-free scores (exact test). For scores of trained models it is a lower bound of p (anti-conservative). | 10 000 permutations, `default_rng(42)`: sd 0.114, p97.5 0.724, p99 0.762 for the 4 label-free scores (n=34, 9 at risk). In the original analysis one random stream was shared, first by the 7-component score and then by the label-free score, so the label-free null is reproduced only up to Monte Carlo error (e.g. p99 0.758-0.764). |
| With retraining | All N permutations are generated first with `default_rng(seed)` (copy + shuffle, same as above); then, for each, refit the supervised model inside the same 5-fold CV and record the OOF AUC. In the article this was done for the supervised kinematic model (random-forest classifier and regressor on the 129 kinematic features); the label-free components need no refit. | Supervised components: how high an AUC can reach by fitting noise at this n. | 1 000 permutations per model (seed 0, n=34) -> pooled 2 000: mean 0.495, sd 0.140, p95 0.722, p97.5 0.773, p99 0.805. |

* **Parametric distance from the null** (the "standard deviations above the
  null" of the article): z = (AUC - 0.5) / 0.14 and one-sided p = 1 - Phi(z)
  (`operating.parametric_z`; printed by `reproduce_metrics.py`, option
  `--null-sd`). Examples: AUC 0.684 -> z = 1.31, p = 0.094; consensus 0.806 ->
  z = 2.19, p = 0.014.
* Derived thresholds: significance AUC = 0.730 (parametric p95,
  0.50 + 1.645 x 0.14; slightly conservative versus the empirical 0.722) and
  chance ceiling AUC = 0.80 (empirical p99 of the pooled retraining null).

`reproduce_metrics.py` runs the fixed-score test (`--n-perm`, default 1 000;
use `--n-perm 10000 --seed-perm 42` for the setting above). The retraining null
requires the kinematic features, which are not public;
`resampling.permutation_null_with_retraining` implements it for any feature
matrix and is exercised on synthetic data in the tests.

## 6. Operating point and agreement

* **F1-optimal operating point** (`operating.f1_optimal_operating_point`): the
  threshold maximizing F1 along the precision-recall curve of the `rank_all`
  score against the primary evaluator; a case is positive when
  score >= threshold. Accuracy, sensitivity, specificity and F1 are reported at
  that threshold.
* **Inter-rater agreement:** Cohen's kappa between the two evaluators on the 34
  common cases, with a 5 000-resample bootstrap CI. Multiclass kappa (the 7 GMA
  categories) and quadratic-weighted kappa (ordinal severity) use the same
  function with `weights=None` / `"quadratic"` on the category codes; GMA
  categories are not part of the score files.
* **Computational rater:** the OOF `mean_all` score of the soft-target run
  (METHODS 5, n = 34) binarized at the quantile 1 - p, where p is the mean of the
  two evaluators' prevalences (14 positives); Cohen's kappa against each
  evaluator with bootstrap CIs. It learned from both evaluators, so it measures
  internal consistency with the combined jury, not independent agreement.

## 7. Causal-leak verification of the self-supervised encoder

The encoder (METHODS section 3) is trained to predict x(t+1) at every position
with bidirectional attention and no causal mask, so the target frame is present
in its input. This was checked by intervention on the trained model:

* **Sensitivity profile.** Perturb one input frame at a time and measure the
  change of output t, on 100-frame windows (the training regime), averaged over
  several positions and five videos. Output t depends on its own input frame
  t about 2 000 times more than on any other frame; its sensitivity to the
  target frame t+1 is only 1.4-1.5 times that to the equally adjacent control
  frame t-1, within the spread between neighbouring offsets.
* **Result.** The model does not exploit the future frame; the validation loss
  is not inflated by leakage. Instead, it behaves as a noisy per-frame identity
  (output t is closest to input t, not to input t+1) and does not beat the
  persistence baseline x(t+1) = x(t). Consequently `mse_ft` measures how hard a
  posture is to auto-encode rather than how unpredictable the movement is, and
  the mean encoding is close to the average pose.
* **Window length.** The feature vector of the MLP classifier (METHODS 4.3) is
  computed on a single window of up to 1 000 frames, 10 times longer than the
  training windows; the encoder error grows 5-20x between 100 and 1 000 frames.
  This is a mechanistic explanation for the near-chance performance of that
  component.
* A retraining with a causal mask was not performed: it is not warranted when
  the trained model does not use the future frame.

## 8. Subject-grouped cross-validation

Three infants contribute several videos (8 of the 36; identifiable only from
file names, so the analysis is partial). The CV was repeated with
`StratifiedGroupKFold` grouped by infant (the supervised components of the
soft-target configuration were refit for this experiment under both schemes):

| Score (reference) | StratifiedKFold | StratifiedGroupKFold |
|---|---|---|
| Label-free consensus signal, `mean label-free (4)` (consensus, n = 21) | 0.806 | 0.806 |
| Complete ensemble, `mean_all (7)`, soft-target configuration (consensus, n = 21) | 0.878 | 0.888 |
| Complete ensemble, `mean_all (7)`, soft-target configuration (primary, n = 34) | 0.818 | 0.818 |
| Random-forest classifier alone (primary, n = 36) | 0.513 | 0.447 |
| Random-forest regressor alone (primary, n = 36) | 0.687 | 0.607 |

Label-free scores are unaffected by construction.

## 9. Known sources of optimism (disclosed)

* The orientation sign of each component is chosen on the whole evaluated
  cohort (in-sample, one bit per component) **against the primary evaluator**.
  This also applies to the label-free descriptors, so their AUCs against the
  primary evaluator (e.g. 0.793) and against the consensus subset (e.g. 0.806)
  are optimistic by construction. The most conservative full-cohort estimate of
  the label-free motor signal is its AUC against the **majority-of-three
  reference: 0.602** (95 % CI about 0.40-0.80; n = 34, 14 at risk). Against the
  secondary evaluator the same signal is at chance (0.526).
* Early stopping of the two neural supervised components monitors the
  held-out fold.
* The deployed single-case reference is in-sample for the final supervised
  models (ARCHITECTURE, "Two scoring modes").
* With n = 34-36 the null standard deviation of the AUC is 0.11-0.14; point
  estimates must be read with their intervals.

## 10. Reported figures and how to recompute them

Score files are restricted (README, "Data availability"); each run is one file
with the schema of `scripts/reproduce_metrics.py`. Runs: **n=36** (primary
labels), **n=34 primary** (primary labels), **n=34 soft** (soft target),
**n=34 majority** (majority-of-three target).

| Figure in the article | Run | Output | Recomputed |
|---|---|---|---|
| Component AUCs vs primary, n = 36: 0.545 / 0.626 / 0.651 / 0.567 / 0.513 / 0.524 / 0.509 | n=36 | component table, primary | 0.5455 / 0.6255 / 0.6509 / 0.5673 / 0.5127 / 0.5236 / 0.5091 |
| Rank ensemble vs primary, n = 36: **0.684** | n=36 | `rank_all (7)`, primary | **0.6855** (= 377/550). See note A. |
| Mean ensemble vs primary, n = 36: 0.680 | n=36 | `mean_all (7)`, primary | 0.6800 |
| Operating point: accuracy 0.75, sensitivity 0.73 (8/11), specificity 0.76 (19/25), F1 0.64 | n=36 | F1-optimal operating point | 0.75, 0.73 (8/11), 0.76 (19/25), 0.64 |
| Ensemble 1.31 SD above the null, p ~ 0.094 | n=36 | `z`, `p(normal)` | z = 1.31 / p = 0.094 for 0.684 (1.32 / 0.093 for 0.6855) |
| Consensus signal, n = 21: **0.806**, 2.19 SD, p ~ 0.014, lower bound ~ 0.587 | n=36 or n=34 | `mean label-free (4)`, consensus | **0.8061**, z = 2.19, p = 0.014, CI [0.589, 0.971]. This figure uses the **mean** fusion of the 4 label-free scores; the adjudication analysis uses their **rank** fusion (0.755 on the same 21 cases). |
| Full ensemble vs consensus: 0.816 | n=36 | `mean_all (7)`, consensus | 0.8163 |
| Retrained, same components and folds, n = 34 (9/34): **0.796** | n=34 primary | `rank_all (7)`, primary | **0.7956** |
| Label-free half alone, n = 34: 0.793 | n=34 primary | `rank label-free (4)`, primary | 0.7933 |
| Label-free motor signal against the other references (adjudication analysis): secondary 0.526, prior consensus (n = 21) 0.755, **majority of three 0.602 [0.400, 0.802]** | n=34 primary | `rank label-free (4)`, secondary / consensus / majority | 0.5260 / 0.7551 / **0.6018 [0.404, 0.793]** |
| Majority-of-three reference: 0.668; difference -0.13 (95 % CI -0.35 to +0.09; p = 0.25) | n=34 primary vs n=34 majority | `compare_reference_standards.py` | 0.6679; -0.128 (CI -0.342 to +0.095; p = 0.26) |
| Clinician kappa 0.256 (CI ~ -0.018 to 0.54); 13 disagreements | any n=34 | agreement | 0.256 (CI ~ -0.01 to 0.53); 13 |
| Computational rater kappa 0.423 vs primary (0.12-0.71) and 0.418 vs secondary (0.10-0.71) | n=34 soft | `rater mean_all (7)` | 0.423 (0.10-0.71) and 0.418 (0.11-0.70) |
| Multiclass kappa 0.432, quadratic-weighted kappa 0.328 | - | needs GMA categories (not in score files) | not recomputed here |
| Permutation tests: 1 000 (per model) and 2 000 (pooled) with retraining | - | needs kinematic features | section 5 |
| Learning curve of the MLP: validation AUC 0.527 (primary) and 0.442 (secondary) | - | needs retraining | not recomputed here |
| Uncertainty signals vs disputed cases (0.546 / 0.311 / 0.597 / 0.670) | - | needs per-case model outputs not in the score files | not recomputed here |
| Pose fine-tuning (4.45 % -> 2.28 %) and pose quality (SJR 5.0 %, TSS 0.914) | - | need annotations / keypoint trajectories | metrics defined in METHODS 2 and `src/metrics` |

**Note A (ties in `rank_all`, n = 36).** Two cases of the n = 36 run have
identical rank sums. Treated as an exact tie (this code, and the n = 34 figure
0.796) the AUC is 0.6855. The original n = 36 analysis normalized each rank
column to [0, 1] in floating point before averaging, which separated that pair
by rounding error and gave 188/275 = 0.6836 (reported as 0.684). The
difference is exactly half of one positive-negative pair (0.5/275 = 0.0018).
The tie-consistent value, 0.685, is consistent with every other figure in the
table and is proposed as a correction at the proof stage.

## 11. Notes for readers of the article

* The public repository contains the evaluation and statistics code,
  configuration, method documentation and interface definitions; the
  orchestration scripts, container definition and REST API described in the
  article are part of the proprietary pipeline (PROPRIETARY_COMPONENTS).
* "Retrained the complete pipeline" in the permutation analysis refers to
  refitting the supervised model on permuted labels (section 5); the
  self-supervised encoder and the pose model were not retrained per permutation.
* The stand-alone MLP described in the article's classifier section (batch
  normalization, learning rate 1e-3) differs from the MLP that produced the
  ensemble scores and the learning curve (METHODS 5).
* The article reports a corpus of 38 videos; the encoder was pre-trained on 39
  coordinate sequences (METHODS 1).
