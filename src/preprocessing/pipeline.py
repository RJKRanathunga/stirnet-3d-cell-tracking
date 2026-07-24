import numpy as np

from .normalization import normalize_volume
from .denoising import denoise_volume
from .background import correct_background


def preprocess_volume(
        volume: np.ndarray,
        normalization_low: float = 1.0,
        normalization_high: float = 99.0,
        denoise_sigma: float = 1.0,
        background_sigma: float = 20.0
) -> dict[str, np.ndarray]:
    """
    Run the complete preprocessing pipeline.

    Pipeline:

        Raw
         ↓
        Normalization
         ↓
        Denoising
         ↓
        Background correction

    Parameters
    ----------
    volume:
        Raw 3D volume.

    normalization_low:
        Lower percentile for normalization.

    normalization_high:
        Upper percentile for normalization.

    denoise_sigma:
        Gaussian sigma for denoising.

    background_sigma:
        Gaussian sigma for background estimation.

    Returns
    -------
    dict
        Dictionary containing intermediate results.

        Keys:
            normalized
            denoised
            corrected
    """

    normalized = normalize_volume(
        volume,
        percentile_low=normalization_low,
        percentile_high=normalization_high
    )

    denoised = denoise_volume(
        normalized,
        sigma=denoise_sigma
    )

    corrected = correct_background(
        denoised,
        sigma=background_sigma
    )

    return {
        "normalized": normalized,
        "denoised": denoised,
        "corrected": corrected
    }