"""Robust intensity normalization shared by external datasets."""

from __future__ import annotations

import numpy as np


def robust_intensity_bounds(
    image: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
    max_samples: int = 1_000_000,
) -> tuple[float, float]:
    """Estimate robust bounds without requiring a full-volume copy."""

    values = np.asarray(image)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match image shape")
        flat = values[mask]
    else:
        flat = values.reshape(-1)

    if flat.size == 0:
        raise ValueError("cannot normalize an empty valid region")
    if flat.size > max_samples:
        stride = max(1, flat.size // max_samples)
        flat = flat[::stride][:max_samples]
    flat = np.asarray(flat, dtype=np.float32)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("image contains no finite values")

    low, high = np.percentile(finite, [low_percentile, high_percentile])
    low = float(low)
    high = float(high)
    if not low < high:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if not low < high:
        # Constant image. Preserve a deterministic finite mapping to zero.
        high = low + 1.0
    return low, high


def normalize_intensity(
    image: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    """Clip to robust bounds and map to float32 [0,1]."""

    low, high = (float(v) for v in bounds)
    if not low < high:
        raise ValueError("bounds must satisfy low < high")
    normalized = (np.asarray(image, dtype=np.float32) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
