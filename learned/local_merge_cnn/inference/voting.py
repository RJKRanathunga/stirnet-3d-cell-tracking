"""Deterministic vector voting helpers in canonical ROI coordinates."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def vector_vote_accumulator(
    vectors_normalized: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    smooth_sigma_vox: float = 1.0,
) -> np.ndarray:
    """Scatter each foreground voxel's vote to its predicted center location."""
    vectors = np.asarray(vectors_normalized, dtype=np.float32)
    foreground = np.asarray(foreground_mask, dtype=bool)
    if vectors.shape != (3, *foreground.shape):
        raise ValueError("vectors must have shape [3,Z,Y,X] aligned to foreground")
    coords = np.argwhere(foreground)
    accumulator = np.zeros(foreground.shape, dtype=np.float32)
    if coords.size == 0:
        return accumulator
    span = np.maximum(np.asarray(foreground.shape, dtype=np.float32) - 1.0, 1.0)
    z, y, x = coords.T
    delta = vectors[:, z, y, x].T * span[None, :]
    votes = np.rint(coords.astype(np.float32) + delta).astype(int)
    votes = np.clip(votes, 0, np.asarray(foreground.shape) - 1)
    np.add.at(accumulator, tuple(votes.T), 1.0)
    if smooth_sigma_vox > 0:
        accumulator = ndimage.gaussian_filter(accumulator, smooth_sigma_vox)
    if accumulator.max(initial=0) > 0:
        accumulator /= float(accumulator.max())
    return accumulator
