"""Configuration for the canonical preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Physical and intensity parameters used by :func:`preprocess_volume`."""

    low_percentile: float = 1.0
    high_percentile: float = 99.5
    denoise_sigma_um: float = 0.8
    background_sigma_um: float = 4.0
    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    def __post_init__(self) -> None:
        if not 0 <= self.low_percentile < self.high_percentile <= 100:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        if self.denoise_sigma_um < 0 or self.background_sigma_um < 0:
            raise ValueError("Gaussian sigmas cannot be negative")
        if len(self.voxel_size_zyx_um) != 3 or any(
            value <= 0 for value in self.voxel_size_zyx_um
        ):
            raise ValueError("voxel_size_zyx_um must contain three positive values")


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
