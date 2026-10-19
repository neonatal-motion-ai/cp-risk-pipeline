# cp-risk-pipeline

Reproducibility repository for the article

> **A Neonate-Specific Self-Supervised Pose-Transformer Pipeline for Early
> Cerebral Palsy Risk Screening from General Movement Videos.**
> Gabriel Mauricio Ramírez Villegas, Sergio Armando Ruiz Contreras, Carlos
> Quintero, Ana María Cardona. *Sci* (MDPI), 2026 (accepted). DOI: pending.

The pipeline estimates an infant's pose from a General Movements video, encodes
the pose sequence with a self-supervised Transformer, scores it with seven
components (four label-free, three supervised) and fuses them into a cerebral
palsy (CP) risk score:

```
Video -> Pose (16 keypoints) -> Temporal encoder -> 7 risk components -> rank fusion -> Risk [0, 1]
```

This repository is a **minimal public subset**. It contains the structure and
data contracts of the pipeline, the full method documentation, and the
evaluation and statistics code that recomputes the reported metrics. The core
pipeline is proprietary to the **Comité de Rehabilitación de Antioquia** and is
not included.

## What is included

| Path | Content |
|---|---|
| `docs/ARCHITECTURE.md` | Pipeline diagram and data contracts between layers |
| `docs/METHODS.md` | Preprocessing, feature formulas, architectures, exact hyperparameters, folds and seeds |
| `docs/EVALUATION_PROTOCOL.md` | AUC, cross-validation, bootstrap, permutation tests, adjudication, causal-leak verification, and how each reported figure is recomputed |
| `docs/PROPRIETARY_COMPONENTS.md` | What is not published, its interface, and why |
| `config/hyperparameters.yaml` | Values used in the article |
| `src/interfaces/` | Abstract classes of the proprietary components (signatures only) |
| `src/metrics/` | Pose quality metrics (Spatial Jump Rate, Temporal Stability Score) |
| `src/evaluation/` | AUC, orientation, rank/mean fusion, stratified folds, bootstrap CIs, permutation tests |
| `scripts/reproduce_metrics.py` | Recomputes the article's tables from a per-case score file |
| `scripts/compare_reference_standards.py` | Paired comparison of two reference standards (primary vs majority of three) |
| `scripts/generate_synthetic_data.py` | Generates a synthetic score file with the same schema |
| `data/synthetic_example/` | Synthetic example (not derived from patients) |
| `tests/` | Tests on synthetic data |

## What is not included

Trained weights, patient videos, frames, keypoint trajectories, GMA labels,
per-case scores, the pose / encoder / feature / classifier implementations, and
the deployment software. See `docs/PROPRIETARY_COMPONENTS.md`.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/generate_synthetic_data.py          # writes data/synthetic_example/scores_synthetic.csv
python scripts/reproduce_metrics.py --scores data/synthetic_example/scores_synthetic.csv
python -m pytest -q
```

Score file schema (one row per case):

```
case_id, score_1, ..., score_7, label_primary, label_secondary, label_consensus, label_majority
```

Equivalent Spanish headers are also accepted (`case_id_anonimo`,
`score_componente_1..7`, `label_primario`, `label_secundario`,
`label_consenso`, `label_mayoria`). `score_1..score_4` are the label-free components and `score_5..score_7` the
out-of-fold scores of the supervised components (order in
`config/hyperparameters.yaml`). Labels are binary (1 = CP risk); the secondary,
consensus and majority labels are optional and empty where not available.

With the restricted score files of the article, the same command recomputes
the figures listed in `docs/EVALUATION_PROTOCOL.md` (section 10).

## Data availability

Per-case model scores and clinical labels are available from the Comité de
Rehabilitación de Antioquia upon reasonable request and subject to ethics
approval; requests are addressed to the institution and evaluated under the
applicable ethical and legal conditions. Video data cannot be shared due to patient privacy and ethical
restrictions. The synthetic example in `data/synthetic_example/` is generated
from parametric distributions and contains no patient information.

## Access to proprietary components

The core pipeline components (pose estimation, temporal encoder, feature
extraction, risk components) and the trained models are proprietary to the
Comité de Rehabilitación de Antioquia. Qualified researchers may address
access requests to the Comité de Rehabilitación de Antioquia (the corresponding
author of the article can forward them); requests are evaluated by the
institution and, where clinical data are involved, are subject to ethics
approval and a data-use agreement. Each component is documented in
`docs/METHODS.md` so that it can be reimplemented independently.

## Third-party software

The pose network builds on EfficientHourglass from the *Markerless* framework
by DeepInMotion (https://github.com/DeepInMotion/Markerless, Apache-2.0). It is
not redistributed here; please cite and obtain it from the original repository.

## Citation

See `CITATION.cff`. The DOI will be added when the article is published.

## License

Code and documentation in this repository: Apache License 2.0 (`LICENSE`,
`NOTICE`). The license does not grant any right to the proprietary components
or data described above.
