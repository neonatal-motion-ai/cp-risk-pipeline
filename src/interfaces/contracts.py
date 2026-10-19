"""Data contracts between pipeline layers.

    Video --(PoseEstimator)--> (T, 32) --(+10 joint angles)--> (T, 42)
          --(TemporalEncoder)--> (T, 42) --(FeatureExtractor)--> (60,)
          --(RiskComponent x7)--> 7 scores --(fusion, src/evaluation)--> risk in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# Keypoint order of the (T, 32) array: column 2*i is x, 2*i+1 is y of keypoint i.
# This is the order the self-supervised temporal encoder was trained with.
BODY_PARTS = [
    "head_top", "upper_neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "thorax", "pelvis",
]
NUM_KEYPOINTS = 16
COORD_DIM = 2 * NUM_KEYPOINTS          # 32
NUM_JOINT_ANGLES = 10
SEQ_DIM = COORD_DIM + NUM_JOINT_ANGLES  # 42
FEATURE_DIM = 60                        # 42 (mean over time) + 18 statistics

COMPONENTS = [
    "mse_ft", "entropy_score_ft", "sampen_vel_mean_ft", "corr_sym_mean_ft",   # label-free
    "kinematic_full", "classb", "transformer_sup_ft",                          # supervised
]
LABEL_FREE_COMPONENTS = COMPONENTS[:4]


@dataclass
class PoseSequence:
    """Output of the pose layer: coordinates normalized to [0, 1], shape (T, 32)."""
    coords: np.ndarray
    fps: float = 30.0

    def __post_init__(self):
        if self.coords.ndim != 2 or self.coords.shape[1] != COORD_DIM:
            raise ValueError(f"coords must be (T, {COORD_DIM}), got {self.coords.shape}")


@dataclass
class RiskResult:
    """Output of the risk layer."""
    risk: float                                   # fused score in [0, 1]
    components: Dict[str, float] = field(default_factory=dict)  # per-component percentile
