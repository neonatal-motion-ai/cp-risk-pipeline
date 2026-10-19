"""Interfaces of the proprietary pipeline components.

The implementations (and the trained weights) belong to the Comite de
Rehabilitacion de Antioquia and are NOT included in this repository. Each class
documents its inputs, outputs and the method described in docs/METHODS.md, so
that a third party can plug in its own implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .contracts import PoseSequence


class PoseEstimator(ABC):
    """Layer 1. Video -> 16 keypoints per frame.

    Reference method (METHODS.md section 2): EfficientHourglass-B4 (third-party,
    Apache-2.0) fine-tuned on neonatal frames; 384x384 RGB input with ImageNet
    normalization; 96x96x16 heatmaps decoded with local soft-argmax.
    """

    @abstractmethod
    def process_video(self, video_path: str) -> PoseSequence:
        """Return coordinates normalized to [0, 1], shape (T, 32)."""


class TemporalEncoder(ABC):
    """Layer 2. Self-supervised Transformer encoder over pose sequences.

    Input (B, W, 42) = 32 coordinates + 10 joint angles; output (B, W, 42).
    Reference: 4 encoder layers, 8 heads, d_model 128, dff 512, dropout 0.1,
    sinusoidal positional encoding, no causal mask; trained to predict the
    sequence shifted by one frame (METHODS.md section 3).
    """

    @abstractmethod
    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """(W, 42) -> (W, 42)."""


class FeatureExtractor(ABC):
    """(T, 32) coordinates -> (60,) feature vector.

    Reference: mean over time of the encoder output (42) + 18 movement
    statistics of the input sequence (METHODS.md section 4.3).
    """

    @abstractmethod
    def extract(self, coords: np.ndarray) -> np.ndarray:
        """(T, 32) -> (60,)."""


class RiskComponent(ABC):
    """One of the 7 ensemble components. Produces one raw score per case.

    Label-free: mse_ft, entropy_score_ft, sampen_vel_mean_ft, corr_sym_mean_ft.
    Supervised: kinematic_full, classb, transformer_sup_ft (METHODS.md section 5).
    The raw scores are oriented and fused with src/evaluation/fusion.py.
    """

    name: str = "component"
    label_free: bool = True

    @abstractmethod
    def score(self, coords: np.ndarray) -> float:
        """(T, 32) coordinates of one case -> raw score (orientation applied later)."""

    def fit(self, coords_list, y) -> "RiskComponent":
        """Supervised components only: fit on training cases. Label-free: no-op."""
        return self
