# Methods

This document is the substitute for the proprietary code: it specifies every
step, formula and hyperparameter needed to reimplement the pipeline described
in the article. Numeric values are also collected in
[`config/hyperparameters.yaml`](../config/hyperparameters.yaml). Section numbers
are referenced from the code and the other documents.

## 1. Data

* Video recordings of infants' spontaneous movements (General Movements
  Assessment, GMA). Video data cannot be shared (see README).
* **Corpus.** The article describes a corpus of 38 videos. Pose extraction and
  the self-supervised pre-training of section 3 used **39 coordinate
  sequences** (the number recorded in the encoder's training metadata); the
  labelled analyses use the 36 / 34 sequences below.
* **Labels.** Each evaluator assigned one GMA category per sequence (writhing
  period: W_normal, W_poor_repertoire, W_cramped_synchronized, W_chaotic;
  fidgety period: F_normal, F_abnormal, F_absent). Binary label:
  **W_normal and F_normal -> 0 (typical / low risk); every other category -> 1
  (CP risk)**.
  * Primary evaluator: 36 sequences (11 at risk / 25 low risk).
  * Secondary evaluator (independent): 36 sequences assessed; 34 of them
    overlap with the primary evaluator's 36 (the doubly-rated cohort).
  * Doubly-rated cohort: 34 cases; the primary evaluator rates 9 of them at risk,
    the secondary evaluator 18.
  * Consensus subset: 21 of 34 cases where both evaluators agree (7 at risk);
    13 disputed cases.
  * Adjudication: the 13 disputed cases were rated by a third evaluator
    (8 resolved in favour of the primary evaluator, 5 in favour of the
    secondary one), giving a majority-of-three label (14 of 34 at risk).
* Some infants contribute more than one video (3 infants, 8 of the 36 videos);
  see EVALUATION_PROTOCOL section 8 for the grouped-CV check.

## 2. Pose estimation (layer 1, proprietary implementation)

* **Model.** EfficientHourglass-B4 (Block1to7 variant) from the third-party
  *Markerless* framework (DeepInMotion, Apache-2.0; see README for the link),
  ported to Keras 3 with identical layer names, warm-started from the authors'
  MPII 384x384 weights.
* **Inference.** Each frame: BGR -> RGB, bilinear resize to 384x384, pixel
  values / 255 and ImageNet mean/std normalization. The network outputs 16
  heatmaps of 96x96. Each heatmap is decoded by a *local soft-argmax*: find the
  arg-max pixel; take the square window of half-size round(2 sigma) pixels
  around it, with sigma = W/32 (= 3 px for W = 96); weight each pixel of the
  window by softmax(beta * value) with beta = 9; the keypoint is the weighted
  mean pixel position (px, py), mapped to [0, 1] as ((px + 0.5) / W,
  (py + 0.5) / H). Output per video: an array (T, 32).
* **Keypoints (order of the 32 columns):** head_top, upper_neck,
  right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow,
  left_wrist, right_hip, right_knee, right_ankle, left_hip, left_knee,
  left_ankle, thorax, pelvis.
* **Fine-tuning for neonates.** 137 keyframes from the corpus with all 16
  keypoints corrected manually (2 219 keypoints, pre-annotation with the base
  model + manual correction in CVAT). Preprocessing: crop around the infant's
  region of interest with 15 % padding, resize to 384x384, Gaussian target
  heatmaps (sigma = 2 px at 96x96). Split **by patient**: keyframes of 30
  videos for training and of 8 disjoint videos for validation (110 / 27
  frames).
  * Initialization: backbone loaded from the MPII weights by layer name
    (mismatching layers skipped); the fine-tuning heatmap head
    (`stage1_confs_tune`) is **warm-started by copying the MPII head weights**
    (`stage1_confs`). A from-scratch head (VarianceScaling) was tried first and
    collapsed to near-zero heatmaps, so the reported model uses the warm start.
  * Frozen layers: stem and first block (`stem_*`, `block1*`); the rest is
    trained.
  * Loss / optimizer: MSE on the 96x96x16 Gaussian heatmaps; Adam, initial
    learning rate 1e-4; batch size 4; up to 50 epochs.
  * Training augmentation (validation: none): horizontal flip with p = 0.5
    (x -> 1 - x and left/right keypoints swapped), then rotation by an angle
    drawn uniformly in [-25 deg, +25 deg] about the image centre (applied when
    |angle| > 0.5 deg; bilinear, constant black border; keypoints rotated with
    the same affine matrix and clipped to [0, 1]).
  * Callbacks: EarlyStopping on validation loss (patience 12, best weights
    restored), ReduceLROnPlateau (factor 0.5, patience 5, min 1e-7),
    checkpoint of the best validation loss. Training batches shuffled with a
    generator seeded at 42.
  * Run of the reported model: all 50 epochs (learning rate halved at epochs
    10, 15, 20, 30, 35, 40, 45, 50), best epoch 50, best validation loss
    6.4e-4.
* **Validation.** 8 held-out videos, 432 keypoints. Metric: localization error
  as a percentage of the image diagonal. Median error 4.45 % (base) -> 2.28 %
  (fine-tuned); paired one-sided Wilcoxon signed-rank test, Cliff's delta and
  bootstrap 95 % CIs.
* **Pose quality metrics** (public implementation in `src/metrics`):
  * Spatial Jump Rate: SJR = 100 * mean over frames t and keypoints k of
    1[ ||p_k(t+1) - p_k(t)|| > 0.05 ] (normalized units).
  * Temporal Stability Score: TSS = 1 / (1 + std of all displacements
    ||p_k(t+1) - p_k(t)||).

## 3. Temporal layer: self-supervised Transformer encoder (layer 2, proprietary)

* **Input.** (W, 42) windows: the 32 coordinates followed by the 10 joint
  angles of section 4.1.
* **Architecture.**
  1. Dense(128) input projection (no activation).
  2. Add sinusoidal positional encoding (standard sin/cos, base 10000,
     d_model = 128, table built for up to 2 000 positions and sliced to the
     window length), then Dropout(0.1).
  3. 4 identical post-norm encoder layers: MultiHeadAttention(8 heads,
     key_dim 16, dropout 0.1) over the whole window **without a causal mask**
     -> Dropout(0.1) -> residual + LayerNorm(eps 1e-6) -> Dense(512, ReLU) ->
     Dense(128) -> Dropout(0.1) -> residual + LayerNorm(eps 1e-6).
  4. Dense(42) output projection at every time step.
  About 0.8 M trainable parameters.
* **Objective.** Windows of 100 frames starting at 0, 50, 100, ... while
  start < T - 100, over each of the 39 sequences (sequences with T <= 100 are
  skipped; 3 073 windows). Sequences are processed in lexicographic order of
  their file names and the windows are concatenated in that order, which
  determines which windows fall in the final 20 % used for validation. Target = the same window shifted by one frame
  (`y[t] = x[t+1]`); MSE loss over every position.
* **Training.** 50 epochs, batch size 32, Adam (learning rate 1e-4), Keras
  `validation_split = 0.2` (last 20 % of the windows), early stopping on
  validation loss (patience 10, best weights restored), ReduceLROnPlateau
  (factor 0.5, patience 5, min 1e-6). Best validation loss 0.0115.
* Because attention is bidirectional, the target frame is present in the input.
  A causal-leak verification showed that the trained model does not exploit
  it; see EVALUATION_PROTOCOL section 7 for the result and its implications.

## 4. Features

Conventions for every formula in this section: standard deviations are
population standard deviations (`numpy.std`, ddof = 0), including the
tolerance r of SampEn; arrays are float32 as produced by the coordinate loader.

### 4.1 Joint angles (10)

For a triplet (A, V, B) of keypoints, the angle at the vertex V is
`arccos( clip( (A-V).(B-V) / (|A-V| |B-V|), -1, 1 ) ) / pi`, in [0, 1]
When |A-V| |B-V| <= 1e-8 (overlapping points) the cosine is set to 0, so the
normalized angle is arccos(0)/pi = **0.5**. Triplets (A, V, B):

| Angle | A | V (vertex) | B |
|---|---|---|---|
| right_elbow | right_shoulder | right_elbow | right_wrist |
| left_elbow | left_shoulder | left_elbow | left_wrist |
| right_knee | right_hip | right_knee | right_ankle |
| left_knee | left_hip | left_knee | left_ankle |
| right_hip | thorax | right_hip | right_knee |
| left_hip | thorax | left_hip | left_knee |
| right_shoulder | upper_neck | right_shoulder | right_elbow |
| left_shoulder | upper_neck | left_shoulder | left_elbow |
| head_neck | head_top | upper_neck | thorax |
| trunk | upper_neck | thorax | pelvis |

Keypoint names refer to positions in the 32-column array (index `i` -> columns
`2i, 2i+1`) in the order of section 2. Wherever a keypoint column is missing in
a coordinate file, its x and y columns are filled with zeros.

### 4.2 Sample entropy and symmetry (label-free descriptors)

Computed on the raw (T, 32) coordinates, taking every 5th frame. The reported
values were produced with the built-in implementation described below (the
code switches to `antropy.sample_entropy` only when that package is installed,
which was not the case). Joint angles
(section 4.1) are computed on the subsampled sequence; angular velocities are
their first differences.

* SampEn(x; m = 2, r = 0.2 * std(x)) with Chebyshev distance: `-ln(A / B)`.
  B counts pairs (i < j) among the N - m templates of length m, A counts pairs
  among the N - m - 1 templates of length m + 1 (template i = x[i : i + L],
  i = 0 .. N - L - 1), each pair counted when the maximum absolute difference is
  <= r. (This differs slightly from the Richman-Moorman definition, which uses
  N - m templates for both lengths.) SampEn = 0 if N < m + 2, std(x) < 1e-10, or
  A or B is 0.
* If the (raw, non-subsampled) sequence has fewer than 20 frames, all entropy
  descriptors and `entropy_score` are set to 0.
* `sampen_mean` = mean SampEn over the 10 angle series.
* `sampen_vel_mean` = mean SampEn over the 10 angular-velocity series.
* `corr_sym_mean` = mean Pearson correlation between the right and left
  elbow, knee, hip and shoulder angle series (NaN -> 0).
* `entropy_score` = 1 / (1 + sampen_mean).

### 4.3 Feature vector for the MLP classifier (60-D)

1. Build the (T, 42) sequence (coordinates + angles). If T > 1 000 keep the
   **last** 1 000 frames; otherwise use the whole sequence unchanged (no
   padding). Run the encoder once on that single window (up to 10x longer than
   the training windows).
2. `mean_encoding` = mean over time of the encoder output -> 42 values.
3. 18 statistics of the full (T, 42) input sequence:
   mean, std, min and max of all values (4); mean and std of |x(t+1) - x(t)|
   over all values (2); v(t) = ||x(t+1) - x(t)|| (Euclidean over the 42
   columns): mean and std of v (2); a(t) = v(t+1) - v(t): mean |a| and std a
   (2); for i = 0..7, mean |x[:, 2i:2i+2] - x[:, 2(i+8):2(i+8)+2]| (8).
   (Edge case: if T < 2 the implementation returns the 4 global values followed
   by 18 zeros, i.e. a 22-value block and a 64-value vector.)
4. Feature vector = [mean_encoding (42), statistics (18)] -> (60,).

### 4.4 Kinematic features (129)

On the (T, 32) coordinates smoothed with a centred moving average of 5 frames
(edge padding; no smoothing when T < 5; sequences with T < 4 are rejected). For each of the 16 keypoints: speed series
s(t) = ||p(t+1) - p(t)||; features mean, std, max of s (3); mean |diff(s)| and
std diff(s) (2); range of motion max - min of x and of y (2) -> 112. Global:
kinetic energy e(t) = 0.5 * sum_k s_k(t)^2: mean, std, max (3); speed of the
centroid of the 16 keypoints: mean, std (2); jerkiness = mean over keypoints of
mean |second difference of s_k| (1); Pearson correlation of the left and right
speed series for shoulders, elbows, wrists, hips, knees and ankles (6, 0 if
either series has std <= 1e-8) and their mean over the valid pairs only (1; 0 if
none); mean over arm joints (shoulders, elbows, wrists) and leg joints (hips,
knees, ankles) of the per-joint speed mean (2) and speed std (2) -> 17.
Total 129.

**Column order** of the 129-feature matrix fed to the random forest (it
matters for an exact reproduction with a fixed `random_state`; the matrix is
cast to float32 before fitting): for each
keypoint in the order of section 2 — `vel_<k>_mean, vel_<k>_std, vel_<k>_max,
acc_<k>_mean, acc_<k>_std, rom_<k>_x, rom_<k>_y` — then
`total_kinetic_energy_mean, _std, _max, center_vel_mean, center_vel_std,
jerkiness_mean, symmetry_vel_shoulder, _elbow, _wrist, _hip, _knee, _ankle,
symmetry_vel_mean, arms_vel_mean, legs_vel_mean, arms_vel_std, legs_vel_std`.

### 4.5 Label-free baseline (Classifier A) and atypicality profile

* **Anomaly score.** The reconstruction MSE of each case (component 1) is
  min-max scaled over the analysed cohort: score = (MSE - min) / (max - min)
  (0 if all equal).
* **Binary decision.** Either the top quartile (MSE >= 75th percentile of the
  cohort) or z-score of the MSE > 1.5 (sample std); for evaluation against
  labels the threshold maximizing F1 on the precision-recall curve is used.
* **Descriptive levels** of the anomaly score: low < 0.25, moderate-low < 0.50,
  moderate-high < 0.75, high >= 0.75.
* **Atypicality profile (deployment).** Each descriptor (MSE, MAE, maximum,
  std and 95th percentile of the per-frame error, `sampen_mean`,
  `sampen_vel_mean`, `corr_sym_mean`, `entropy_score`) is converted to a
  mid-rank percentile against the corpus of 39 sequences. The global
  atypicality level uses the MSE percentile: low < 0.50, moderate < 0.80,
  high >= 0.80. This describes statistical atypicality relative to a mixed
  corpus, not CP risk.

## 5. Risk components (layer 3, proprietary implementations)

| # | Component | Type | Definition |
|---|---|---|---|
| 1 | `mse_ft` | label-free | Reconstruction error of the encoder on the (T, 42) sequence: windows starting at i = 0, 50, 100, ... while i < T - 100, each of 100 frames; MSE between the output and the input shifted by one frame (target x[i+1 : i+101]), averaged over windows. If T <= 100 a single window x[0 : T-1] with target x[1 : T] is used. |
| 2 | `entropy_score_ft` | label-free | `entropy_score` (4.2). |
| 3 | `sampen_vel_mean_ft` | label-free | `sampen_vel_mean` (4.2). |
| 4 | `corr_sym_mean_ft` | label-free | `corr_sym_mean` (4.2). |
| 5 | `kinematic_full` | supervised | Random forest on the 129 kinematic features (4.4). |
| 6 | `classb` | supervised | MLP on the 60-D vector (4.3): Dense 128 ReLU, Dropout 0.3, Dense 64 ReLU, Dropout 0.3, Dense 32 ReLU, Dense 1 sigmoid. |
| 7 | `transformer_sup_ft` | supervised | The pre-trained encoder (section 3) on the **first** 1 000 frames of the (T, 42) sequence (zero-padded at the end if T < 1 000; note that component 6 uses the *last* 1 000), only the last encoder layer trainable, pooling = concat(global average, global max, last time step), head Dropout 0.4 -> Dense 64 ReLU -> Dropout 0.4 -> Dense 1 sigmoid. |

Components 1-4 never see labels. Components 5-7 are trained inside the
cross-validation folds (EVALUATION_PROTOCOL section 2). The reported runs are:

* **Run n=36** (primary evaluator, binary labels): `kinematic_full` =
  RandomForestClassifier(300 trees, min_samples_leaf 2, class_weight balanced,
  random_state 42), probability of class 1; `classb` and `transformer_sup_ft`
  with balanced class weights.
* **Runs n=34** (doubly-rated cohort): `kinematic_full` =
  RandomForestRegressor(300 trees, min_samples_leaf 2, random_state 42),
  prediction clipped to [0, 1]; no class weights. Three training targets:
  * primary evaluator's binary label (the 0.796 figure);
  * soft target 1 (both at risk), 0 (both low risk), 0.5 (disagreement) — the
    "computational rater" of the agreement analysis and, refit on all 34 cases,
    the **deployed model**;
  * majority-of-three binary label (adjudication analysis).

**Final refit of the deployed model.** After the cross-validation, the three
supervised components are refit once on all 34 cases with the soft target and
**without any validation data or early stopping**: random-forest regressor
(300 trees, min_samples_leaf 2, random_state 42); `classb` MLP for exactly 100
epochs (batch 8, Adam 1e-4); `transformer_sup_ft` for exactly 50 epochs (batch
4, Adam 1e-4). These refit models produce the deployed reference values
(section 6.5); they are not used for any reported AUC.

**Random seeds.** Seeded: the cross-validation folds (42), the random forests
(42), the pose-training batch shuffling (42), bootstrap (42) and permutations
(0 or 42, see EVALUATION_PROTOCOL). Not seeded: TensorFlow/Keras weight
initialization, dropout and batch order of the self-supervised encoder, the
MLP and the supervised encoder head. The neural components are therefore not
bit-reproducible; re-running them gives OOF scores that differ by run-to-run
variance (the label-free components 1-4 and the random forest do not).

Random forests use the scikit-learn 1.4 defaults for every other parameter
(classifier: `max_features="sqrt"`; regressor: `max_features=1.0`;
`bootstrap=True`).

Training of `classb` (MLP without batch normalization, layers as in the table
above): Adam 1e-4, binary cross-entropy, 100 epochs, batch 8,
early stopping on the validation loss (patience 15, best weights restored).
A second MLP variant exists in the codebase as the stand-alone "Classifier B"
module (Dense 128 + BatchNorm + Dropout 0.3, Dense 64 + BatchNorm + Dropout
0.3, Dense 32 + Dropout 0.2, Dense 1 sigmoid; Adam 1e-3, validation split 0.2);
it is **not** the one that produced the reported ensemble scores. The
learning-curve figure was produced with the MLP above (no batch normalization,
Adam 1e-4, batch 8, 100 epochs, balanced class weights, no early stopping,
StratifiedKFold(5, shuffle, 42)) with these inputs, which differ from the
ensemble runs: features = the **42-D mean encoding only** (no 18 statistics);
encoder input = the **first** 1 000 frames of the (T, 42) sequence, zero-padded
at the end; encoder = an earlier self-supervised checkpoint ("v007", same
architecture, weights loaded into the same model); coordinates = the script's default
coordinate set produced with the base (pre-fine-tuning) pose model. The
validation AUC per fold is recorded at every epoch; the reported value is the
mean over the 5 folds of the final-epoch validation AUC.
Training of `transformer_sup_ft`: Adam 1e-4, binary cross-entropy, 50 epochs,
batch 4, early stopping (patience 8, best weights restored), ReduceLROnPlateau
(factor 0.5, patience 4, min 1e-6). In both cases the validation data used for
early stopping is the held-out fold (see EVALUATION_PROTOCOL section 9).

**Channel order of the supervised components.** Components 5-7 were trained
and evaluated with the 32 coordinate columns in an alternative keypoint order:
head_top, upper_neck, right_shoulder, right_elbow, right_wrist, thorax,
left_shoulder, left_elbow, left_wrist, pelvis, right_hip, right_knee,
right_ankle, left_hip, left_knee, left_ankle. All index-based definitions
(joint-angle triplets, named keypoints of 4.4, pairs of 4.3) and the encoder
input were applied to that array as if it were in the canonical order of
section 2. Components 1-4 use the canonical order. To reproduce the reported
numbers this mismatch must be replicated; it is a known limitation that
degrades components 6 and 7 (both near chance).

## 6. Orientation and fusion

1. **Orientation.** For each component, if its AUC against the primary
   evaluator over the evaluated cohort is < 0.5, the score s is replaced by
   1 - s (so that higher = higher risk). Deployed signs:
   `config/hyperparameters.yaml` (`ensemble_orientation_deployed`).
2. **Scaling.** Label-free components are min-max scaled over the cohort; the
   supervised components are already probabilities in [0, 1].
3. **Fusion (`rank_all`, reported).** Rank each oriented component over the
   cohort (average ranks for ties) and average the ranks of the 7 components.
   The fused value of a case is (sum of its ranks) / (7 n). Equal rank sums are
   exact ties.
4. **Alternative (`mean_all`).** Mean of the 7 oriented scores after step 2.
5. **Single-case scoring (deployment).** For a new case, each component's raw
   score is converted to a mid-rank percentile within the persisted reference
   values of the 34 training cases, (count(ref < x) + 0.5 count(ref = x)) / 34,
   inverted when the deployed sign is -1; the risk is the mean of the 7
   percentiles. Display categories: < 0.40 low, < 0.55 moderate-low,
   < 0.70 moderate-high, >= 0.70 high.

## 7. Software environment

Python 3.10, TensorFlow 2.18 / Keras 3.12, NumPy 1.26, scikit-learn 1.4, SciPy
1.11, pandas 2.1. Pose fine-tuning ran on an NVIDIA Blackwell GPU. The public
evaluation code needs only NumPy, SciPy, pandas and scikit-learn
(`requirements.txt`).
