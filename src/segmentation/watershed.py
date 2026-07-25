from skimage.segmentation import watershed
import numpy as np


def watershed_segmentation(
        distance: np.ndarray,
        markers: np.ndarray,
        binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Perform marker-controlled watershed segmentation.
    """

    return watershed(
        -distance,
        markers=markers,
        mask=binary_mask,
    )