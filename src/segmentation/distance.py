import numpy as np
from scipy import ndimage

from .config import DEFAULT_SEGMENTATION_CONFIG


def validate_voxel_size(
    voxel_size: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return a validated three-axis physical voxel size."""

    values = np.asarray(voxel_size, dtype=float)
    if (
        values.shape != (3,)
        or np.any(values <= 0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("voxel_size must contain three finite positive values")
    return values


def physical_sigma_voxels(
    sigma_um: float,
    voxel_size: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Convert an isotropic physical Gaussian width to per-axis voxel units."""

    if not np.isfinite(sigma_um) or sigma_um < 0:
        raise ValueError("sigma_um must be finite and non-negative")
    return float(sigma_um) / validate_voxel_size(voxel_size)


def physical_distance(
    point_a_zyx: tuple[int, int, int] | np.ndarray,
    point_b_zyx: tuple[int, int, int] | np.ndarray,
    voxel_size: tuple[float, float, float] | np.ndarray,
) -> float:
    """Measure the Euclidean distance between voxel coordinates in physical space."""

    delta = np.asarray(point_a_zyx, dtype=float) - np.asarray(
        point_b_zyx, dtype=float
    )
    if delta.shape != (3,):
        raise ValueError("points must contain z, y, and x coordinates")
    return float(np.linalg.norm(delta * validate_voxel_size(voxel_size)))


def compute_distance_transform(
    binary_mask: np.ndarray,
    voxel_size: tuple[float, float, float],
) -> np.ndarray:
    """
    Compute the Euclidean distance transform using physical voxel spacing.
    """

    mask = np.asarray(binary_mask, dtype=bool)
    return ndimage.distance_transform_edt(
        mask,
        sampling=validate_voxel_size(voxel_size),
    )


def smooth_distance_transform(
    distance: np.ndarray,
    sigma_physical: float = 0.8,
    voxel_size: tuple[float, float, float] = (
        DEFAULT_SEGMENTATION_CONFIG.voxel_size_zyx_um
    ),
) -> np.ndarray:
    """
    Smooth the distance transform before marker detection.
    """

    return ndimage.gaussian_filter(
        distance,
        sigma=physical_sigma_voxels(sigma_physical, voxel_size),
    )
