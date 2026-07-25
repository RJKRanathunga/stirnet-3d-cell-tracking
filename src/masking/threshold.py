from skimage.filters import threshold_otsu
import numpy as np


def compute_otsu_threshold(
        volume: np.ndarray,
) -> float:
    """Compute the global Otsu threshold."""

    return threshold_otsu(volume)


def apply_threshold(
        volume: np.ndarray,
        threshold: float,
) -> np.ndarray:
    """Threshold a volume into a binary mask."""

    return volume > threshold