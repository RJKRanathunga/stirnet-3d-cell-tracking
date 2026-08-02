from .threshold import (
    compute_otsu_threshold,
    apply_threshold,
)
from .connected_components import (
    label_connected_components,
)
import numpy as np


def create_binary_mask(
        volume: np.ndarray,
) -> np.ndarray:

    threshold = compute_otsu_threshold(volume)

    binary_mask = apply_threshold(
        volume,
        threshold,
    )

    return binary_mask