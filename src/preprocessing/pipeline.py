from .normalize import robust_normalize
from .denoise import gaussian_denoise
from .background import background_correction


VOXEL_SIZE = (
    1.625,
    0.40625,
    0.40625,
)


def preprocess_volume(volume):
    """Complete preprocessing pipeline."""

    normalized = robust_normalize(volume)

    denoised = gaussian_denoise(
        normalized,
        voxel_size=VOXEL_SIZE,
    )

    corrected = background_correction(
        denoised,
        voxel_size=VOXEL_SIZE,
    )

    return corrected