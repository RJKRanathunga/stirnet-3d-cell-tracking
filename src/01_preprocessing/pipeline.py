from .normalize import robust_normalize, robust_normalize_with_percentiles
from .denoise import gaussian_denoise
from .background import background_correction, background_correction_with_estimate

from src.diagnostics import StageTrace


VOXEL_SIZE = (
    1.625,
    0.40625,
    0.40625,
)


def preprocess_volume(volume, *, return_diagnostics: bool = False):
    """Complete preprocessing pipeline."""

    if return_diagnostics:
        normalized, low, high = robust_normalize_with_percentiles(volume)
    else:
        normalized = robust_normalize(volume)

    denoised = gaussian_denoise(
        normalized,
        voxel_size=VOXEL_SIZE,
    )

    if return_diagnostics:
        corrected, background = background_correction_with_estimate(
            denoised,
            voxel_size=VOXEL_SIZE,
        )
    else:
        corrected = background_correction(
            denoised,
            voxel_size=VOXEL_SIZE,
        )

    if not return_diagnostics:
        return corrected

    trace = StageTrace(
        stage_name="01_preprocessing",
        inputs={"volume": volume},
        outputs={"corrected": corrected},
        intermediates={
            "normalized": normalized,
            "denoised": denoised,
            "background": background,
        },
        metrics={"low_percentile_value": low, "high_percentile_value": high},
    )
    return corrected, trace
