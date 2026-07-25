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