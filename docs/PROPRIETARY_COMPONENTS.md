# Components that are not published

The complete pipeline is the property of the **Comité de Rehabilitación de
Antioquia** (Colombia). This repository publishes only what is needed to
understand the method and to recompute the reported metrics. The items below
are excluded; each one is described in `METHODS.md` precisely enough to be
reimplemented, and the interfaces in `src/interfaces/` show where an own
implementation plugs in.

| Component | Interface (inputs -> outputs) | Described in | Why it is not published |
|---|---|---|---|
| Pose estimation (fine-tuned EfficientHourglass-B4 port, video reader, heatmap decoder) | `PoseEstimator.process_video(path) -> PoseSequence (T, 32)` | METHODS 2 | Proprietary implementation; the base network is third-party code (see below). |
| Fine-tuning scripts and annotation tooling for the pose model | - | METHODS 2 | Proprietary; operate on patient video frames. |
| Self-supervised Transformer encoder (implementation and training) | `TemporalEncoder.encode((W, 42)) -> (W, 42)` | METHODS 3 | Proprietary implementation (architecture fully specified in METHODS). |
| Feature extraction (joint angles, sample entropy, kinematics, 60-D vector) | `FeatureExtractor.extract((T, 32)) -> (60,)` | METHODS 4 | Proprietary implementation; formulas published. |
| The 7 risk components and their training code | `RiskComponent.score((T, 32)) -> float`, `RiskComponent.fit(...)` | METHODS 5 | Proprietary implementation; architectures and hyperparameters published. |
| Deployment: REST API, job manager, web interface, containers | - | ARCHITECTURE ("Two scoring modes") | Institutional software of the Comité. |
| **Trained weights** (pose model, encoder, classifiers, random forests) | - | - | Trained on patient data; assets of the Comité. Never published. |
| Reference distributions of the deployed model (per-case values) | - | METHODS 6.5 | Derived from patient data. |
| **Per-case score files and GMA labels** used to compute the article's figures | CSV schema in `scripts/reproduce_metrics.py` | EVALUATION_PROTOCOL 10 | Clinical information of minors. Available from the Comité upon reasonable request, subject to ethics approval. |
| **Videos, frames, keypoint trajectories, annotation sheets** | - | - | Patient privacy and ethics restrictions. Not shareable. |

## Third-party code

The pose network is based on **EfficientHourglass** from the *Markerless*
framework by DeepInMotion (https://github.com/DeepInMotion/Markerless),
licensed under Apache-2.0. It is not redistributed here. To obtain it, clone
the original repository and follow its instructions; its pretrained MPII
weights are distributed by the original authors. The Apache-2.0 license allows
use, modification and redistribution provided the license and notices are
kept and modifications are stated.

## How to request access

Qualified researchers can address requests for the proprietary components, or
for the restricted per-case score files, to the Comité de Rehabilitación de
Antioquia (the corresponding author of the article can forward them). Requests
are evaluated by the institution; access to clinical data additionally requires
ethics approval and a data-use agreement.
