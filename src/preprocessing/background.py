import numpy as np
from scipy.ndimage import gaussian_filter


def background_correction(
        volume: np.ndarray,
        voxel_size: tuple[float, float, float],
        sigma_um: float = 4.0,
) -> np.ndarray:
    """Subtract a smooth background estimate."""

    sigma = sigma_um / np.asarray(voxel_size)

    background = gaussian_filter(volume, sigma=sigma)

    corrected = volume - background

    corrected = np.clip(corrected, 0.0, None)

    if corrected.max() > 0:
        corrected /= corrected.max()

    return corrected