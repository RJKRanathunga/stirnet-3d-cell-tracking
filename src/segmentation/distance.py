import numpy as np
from scipy import ndimage


def compute_distance_transform(
        binary_mask: np.ndarray,
        voxel_size: tuple[float, float, float],
) -> np.ndarray:
    """
    Compute the Euclidean distance transform using physical voxel spacing.
    """

    return ndimage.distance_transform_edt(
        binary_mask,
        sampling=voxel_size,
    )


def smooth_distance_transform(
        distance: np.ndarray,
        sigma_physical: float = 0.8,
) -> np.ndarray:
    """
    Smooth the distance transform before marker detection.
    """

    sigma_zyx = (
        sigma_physical / 1.625,
        sigma_physical / 0.40625,
        sigma_physical / 0.40625,
    )

    return ndimage.gaussian_filter(
        distance,
        sigma=sigma_zyx,
    )