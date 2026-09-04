import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_denoise(
        volume: np.ndarray,
        voxel_size: tuple[float, float, float],
        sigma_um: float = 0.8,
) -> np.ndarray:
    """Gaussian denoising using a physical sigma."""

    sigma = sigma_um / np.asarray(voxel_size)

    return gaussian_filter(volume, sigma=sigma)