import numpy as np

from scipy.ndimage import gaussian_filter


def estimate_background(
        volume: np.ndarray,
        sigma: float = 20.0
) -> np.ndarray:
    """
    Estimate the slowly varying background of a 3D volume.

    Parameters
    ----------
    volume:
        Input 3D volume.

    sigma:
        Gaussian scale used for background estimation.

    Returns
    -------
    np.ndarray
        Estimated background volume.
    """

    background = gaussian_filter(
        volume,
        sigma=sigma
    )

    return background


def correct_background(
        volume: np.ndarray,
        sigma: float = 20.0
) -> np.ndarray:
    """
    Subtract estimated background from the volume.

    Negative values are clipped to zero.

    Parameters
    ----------
    volume:
        Input 3D volume.

    sigma:
        Gaussian scale used for background estimation.

    Returns
    -------
    np.ndarray
        Background-corrected volume.
    """

    background = estimate_background(
        volume,
        sigma=sigma
    )

    corrected = volume - background

    corrected = np.clip(
        corrected,
        0.0,
        None
    )

    return corrected