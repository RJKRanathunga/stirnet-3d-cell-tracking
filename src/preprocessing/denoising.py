import numpy as np

from scipy.ndimage import gaussian_filter


def denoise_volume(
        volume: np.ndarray,
        sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 3D volume.

    Parameters
    ----------
    volume:
        Input 3D volume.

    sigma:
        Gaussian smoothing strength.

    Returns
    -------
    np.ndarray
        Denoised 3D volume.
    """

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than 0."
        )

    denoised = gaussian_filter(
        volume,
        sigma=sigma
    )

    return denoised