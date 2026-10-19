"""Pose-estimation quality metrics computed from a keypoint trajectory.

Input: coordinates normalized to [0, 1] (by image width/height), shape (T, 32)
with columns x0, y0, x1, y1, ... for the 16 keypoints, or shape (T, 16, 2).

- Spatial Jump Rate (SJR): percentage of (frame, keypoint) displacements between
  consecutive frames larger than `threshold` (default 0.05 of the normalized
  frame). Lower is better; < 5 % is considered excellent.
- Temporal Stability Score (TSS): 1 / (1 + std(displacements)). Higher is
  better; > 0.90 is considered excellent.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _as_tk2(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    if c.ndim == 2 and c.shape[1] % 2 == 0:
        return c.reshape(c.shape[0], c.shape[1] // 2, 2)
    if c.ndim == 3 and c.shape[2] == 2:
        return c
    raise ValueError(f"expected (T, 2K) or (T, K, 2), got {c.shape}")


def displacements(coords: np.ndarray) -> np.ndarray:
    """Euclidean displacement of each keypoint between consecutive frames, (T-1, K)."""
    c = _as_tk2(coords)
    return np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=2))


def spatial_jump_rate(coords: np.ndarray, threshold: float = 0.05) -> float:
    d = displacements(coords)
    return float((d > threshold).mean() * 100.0) if d.size else 0.0


def temporal_stability_score(coords: np.ndarray) -> float:
    d = displacements(coords)
    return float(1.0 / (1.0 + d.std())) if d.size else 1.0


def compute_quality(coords: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    return {"sjr": spatial_jump_rate(coords, threshold),
            "tss": temporal_stability_score(coords),
            "frames": int(_as_tk2(coords).shape[0])}
