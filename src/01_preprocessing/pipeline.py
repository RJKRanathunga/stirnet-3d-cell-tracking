from .normalize import robust_normalize, robust_normalize_with_percentiles
from .denoise import gaussian_denoise
from .background import background_correction, background_correction_with_estimate

from src.diagnostics import StageTrace
from .config import DEFAULT_PREPROCESSING_CONFIG, PreprocessingConfig


VOXEL_SIZE = DEFAULT_PREPROCESSING_CONFIG.voxel_size_zyx_um


def preprocess_volume(
    volume,
    *,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    return_diagnostics: bool = False,
):
    """Complete preprocessing pipeline."""

    if return_diagnostics:
        normalized, low, high = robust_normalize_with_percentiles(
            volume, config.low_percentile, config.high_percentile
        )
    else:
        normalized = robust_normalize(
            volume, config.low_percentile, config.high_percentile
        )

    denoised = gaussian_denoise(
        normalized,
        voxel_size=config.voxel_size_zyx_um,
        sigma_um=config.denoise_sigma_um,
    )

    if return_diagnostics:
        corrected, background = background_correction_with_estimate(
            denoised,
            voxel_size=config.voxel_size_zyx_um,
            sigma_um=config.background_sigma_um,
        )
    else:
        corrected = background_correction(
            denoised,
            voxel_size=config.voxel_size_zyx_um,
            sigma_um=config.background_sigma_um,
        )

    if not return_diagnostics:
        return corrected

    trace = StageTrace(
        stage_name="01_preprocessing",
        inputs={"volume": volume, "config": config},
        outputs={"corrected": corrected},
        intermediates={
            "raw": volume,
            "normalized": normalized,
            "denoised": denoised,
            "background": background,
        },
        metrics={"low_percentile_value": low, "high_percentile_value": high},
    )
    return corrected, trace
