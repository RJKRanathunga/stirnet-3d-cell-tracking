import numpy as np

from skimage.filters import threshold_otsu


def otsu_threshold(
        volume: np.ndarray
) -> float:
    """
    Calculate the Otsu threshold for a 3D volume.

    Parameters
    ----------
    volume:
        Preprocessed 3D volume.

    Returns
    -------
    float
        Otsu threshold.
    """

    threshold = threshold_otsu(
        volume
    )

    return threshold


def create_binary_mask(
        volume: np.ndarray
) -> np.ndarray:
    """
    Create a binary segmentation mask using Otsu thresholding.

    Parameters
    ----------
    volume:
        Preprocessed 3D volume.

    Returns
    -------
    np.ndarray
        Boolean 3D binary mask.
    """

    threshold = otsu_threshold(
        volume
    )

    binary_mask = (
            volume > threshold
    )

    return binary_mask