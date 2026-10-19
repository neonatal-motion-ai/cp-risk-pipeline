# Architecture

The pipeline turns a video of an infant's spontaneous movements into a cerebral
palsy (CP) risk score in [0, 1]. It has three layers plus a late-fusion step.

```
 Video (RGB, ~30 fps)
   |
   v
 [1] Pose estimation  (PoseEstimator)                         PROPRIETARY
     EfficientHourglass-B4 fine-tuned on neonatal frames
     -> 16 keypoints per frame, normalized to [0, 1]
   |
   |  (T, 32)   x0, y0, ..., x15, y15            pose quality: SJR, TSS  (public: src/metrics)
   v
 [+] Joint angles (10, divided by pi)                          documented (METHODS 4.1)
   |
   |  (T, 42)
   v
 [2] Temporal layer  (TemporalEncoder)                         PROPRIETARY
     Self-supervised Transformer encoder, 4 x 8 heads, d_model 128
   |
   +--> reconstruction error (MSE), windows of 100 / stride 50 ----> mse_ft
   +--> (T, 42) -> mean over time (42) + 18 statistics = (60,) --+
   |                                                              |
   v                                                              v
 [3] Risk components  (RiskComponent x 7)                      PROPRIETARY implementations,
     label-free:  mse_ft, entropy_score_ft,                    formulas in METHODS 4-5
                  sampen_vel_mean_ft, corr_sym_mean_ft
     supervised:  kinematic_full (random forest, 129 features)
                  classb (MLP on the 60-D vector)
                  transformer_sup_ft (encoder + classification head)
   |
   |  7 scores per case
   v
 [4] Orientation + late fusion (rank_all)                     PUBLIC: src/evaluation/fusion.py
     risk = mean of the per-component normalized ranks
   |
   v
 Risk score in [0, 1]  ->  evaluation (AUC, CV, bootstrap,      PUBLIC: src/evaluation,
                           permutations)                         scripts/reproduce_metrics.py
```

## Data contracts

| From -> to | Object | Shape / type | Defined in |
|---|---|---|---|
| Video -> pose | coordinates normalized by image width/height | `(T, 32)` float, column `2i` = x, `2i+1` = y of keypoint `i` | `src/interfaces/contracts.py` (`PoseSequence`, `BODY_PARTS`) |
| Pose -> temporal | coordinates + 10 joint angles | `(T, 42)` | `SEQ_DIM` |
| Temporal -> features | mean encoding + statistics | `(60,)` | `FEATURE_DIM`, `FeatureExtractor` |
| Components -> fusion | one raw score per component and case | 7 floats | `COMPONENTS`, `RiskComponent` |
| Fusion -> output | risk score + per-component percentiles | `RiskResult` | `src/evaluation/fusion.py` |
| Evaluation input | per-case score file | CSV: `case_id, score_1..score_7, label_primary, label_secondary, label_consensus, label_majority` | `scripts/reproduce_metrics.py` |

Keypoint order (index 0-15): head_top, upper_neck, right_shoulder, right_elbow,
right_wrist, left_shoulder, left_elbow, left_wrist, right_hip, right_knee,
right_ankle, left_hip, left_knee, left_ankle, thorax, pelvis.

## Two scoring modes

* **Cohort evaluation (article).** Components are computed for every case with
  5-fold out-of-fold (OOF) predictions for the supervised ones; the fused score
  is a rank over the evaluated cohort. All reported AUCs come from this mode.
* **Single-case scoring (deployment).** A new case is scored against a fixed
  reference distribution of the 34 doubly-rated training cases: each component
  becomes a mid-rank percentile (`percentile_vs_reference`) and the risk is the
  mean of the 7 oriented percentiles. The reference is in-sample for the final
  supervised models and is therefore optimistic for those cases; expected
  performance on new cases is the OOF performance.

## What is in this repository

* `src/interfaces/` - contracts and abstract classes of the proprietary layers.
* `src/metrics/` - pose quality metrics (SJR, TSS).
* `src/evaluation/` - AUC, orientation, fusion, CV folds, bootstrap, permutation tests.
* `scripts/reproduce_metrics.py` - recomputes the article's tables from a score file.
* `docs/` - this file, METHODS, EVALUATION_PROTOCOL, PROPRIETARY_COMPONENTS.

The implementations of layers 1-3, the trained weights and all patient data are
not included (see `PROPRIETARY_COMPONENTS.md`).
