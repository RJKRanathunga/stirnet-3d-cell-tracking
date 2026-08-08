"""Robust intensity normalization shared by external datasets."""

from __future__ import annotations

import numpy as np


def robust_intensity_bounds(
    image: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> tuple[float, float]:
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("invalid percentile range")
    values = np.asarray(image)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match image shape")
        values = values[mask]
    else:
        values = values.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [low_percentile, high_percentile])
    low = float(low)
    high = float(high)
    if not low < high:
        high = low + max(1.0, abs(low) * 1e-6)
    return low, high


def normalize_intensity(image: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = map(float, bounds)
    if not low < high:
        raise ValueError("bounds must satisfy low < high")
    normalized = (np.asarray(image, dtype=np.float32) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
