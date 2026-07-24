import numpy as np


def normalize_volume(
        volume: np.ndarray,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0
) -> np.ndarray:
    """
    Normalize a 3D volume using percentile-based intensity scaling.

    Intensities below percentile_low are mapped to 0.
    Intensities above percentile_high are mapped to 1.

    Parameters
    ----------
    volume:
        Input 3D image.

    percentile_low:
        Lower percentile used for normalization.

    percentile_high:
        Upper percentile used for normalization.

    Returns
    -------
    np.ndarray
        Normalized volume with values in [0, 1].
    """

    volume = volume.astype(np.float32)

    low = np.percentile(
        volume,
        percentile_low
    )

    high = np.percentile(
        volume,
        percentile_high
    )

    if high <= low:
        raise ValueError(
            "Invalid intensity range. "
            "Upper percentile must be greater than lower percentile."
        )

    normalized = (
                         volume - low
                 ) / (
                         high - low
                 )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized