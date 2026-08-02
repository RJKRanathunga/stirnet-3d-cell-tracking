import numpy as np


def robust_normalize(
        volume: np.ndarray,
        low_percentile: float = 1.0,
        high_percentile: float = 99.5,
) -> np.ndarray:
    """Robust percentile normalization to [0, 1]."""

    volume = volume.astype(np.float32)

    p_low = np.percentile(volume, low_percentile)
    p_high = np.percentile(volume, high_percentile)

    normalized = (volume - p_low) / (p_high - p_low)

    return np.clip(normalized, 0.0, 1.0)


def robust_normalize_with_percentiles(
        volume: np.ndarray,
        low_percentile: float = 1.0,
        high_percentile: float = 99.5,
) -> tuple[np.ndarray, float, float]:
    """Normalize once while returning the percentile values already computed."""

    values = volume.astype(np.float32)
    p_low = float(np.percentile(values, low_percentile))
    p_high = float(np.percentile(values, high_percentile))
    normalized = np.clip((values - p_low) / (p_high - p_low), 0.0, 1.0)
    return normalized, p_low, p_high


def normalization_percentiles(
        volume: np.ndarray,
        low_percentile: float = 1.0,
        high_percentile: float = 99.5,
) -> tuple[float, float]:
    """Return the exact percentile values used by :func:`robust_normalize`."""

    values = volume.astype(np.float32)
    return (
        float(np.percentile(values, low_percentile)),
        float(np.percentile(values, high_percentile)),
    )
