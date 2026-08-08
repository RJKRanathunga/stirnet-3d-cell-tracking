"""Configuration for external 3-D instance-segmentation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from os import environ
from pathlib import Path

# User's current C. elegans checkout location. Kept in one place and always
# overridable from the CLI or an environment variable.
_WINDOWS_C_ELEGANS_ROOT = Path(
    r"D:\Projects\Kaggle\cell-tracking\data\external\c_elegans_nuclei"
)

# Published C. elegans voxel spacing is (x, y, z)=(0.116, 0.116, 0.122) um.
# NumPy volumes are represented everywhere in this package as (z, y, x).
C_ELEGANS_SPACING_ZYX_UM = (0.122, 0.116, 0.116)

# BlastoSPIM acquisition resolution is 0.208 um in XY and 2.0 um in Z.
BLASTOSPIM_SPACING_ZYX_UM = (2.0, 0.208, 0.208)

# Biohub target grid used by the learned correction model.
TARGET_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
CANONICAL_CROP_SHAPE_ZYX = (16, 64, 64)


def _default_external_root(env_name: str, *relative_candidates: str) -> Path:
    env_value = environ.get(env_name)
    if env_value:
        return Path(env_value).expanduser()
    for relative in relative_candidates:
        candidate = Path("data") / "external" / relative
        if candidate.exists():
            return candidate
    return Path("data") / "external" / relative_candidates[0]


def default_c_elegans_root() -> Path:
    """Return the configured C. elegans dataset directory."""

    env_value = environ.get("C_ELEGANS_NUCLEI_DIR")
    if env_value:
        return Path(env_value).expanduser()
    if _WINDOWS_C_ELEGANS_ROOT.exists():
        return _WINDOWS_C_ELEGANS_ROOT
    return Path("data") / "external" / "c_elegans_nuclei"


def default_nis3d_root() -> Path:
    """Return the configured NIS3D dataset directory."""

    return _default_external_root("NIS3D_DIR", "nis3d", "NIS3D")


def default_blastospim_root() -> Path:
    """Return the configured BlastoSPIM dataset directory."""

    return _default_external_root("BLASTOSPIM_DIR", "blastospim", "BlastoSPIM")


@dataclass(frozen=True)
class SampleBuildConfig:
    """Physical and numerical settings for canonical CNN sample generation."""

    target_spacing_zyx_um: tuple[float, float, float] = TARGET_SPACING_ZYX_UM
    crop_shape_zyx: tuple[int, int, int] = CANONICAL_CROP_SHAPE_ZYX

    # Channel construction.
    edt_clip_um: float = 8.0
    marker_sigma_um: float = 1.1

    # Target construction. Keep vector_max_distance_um synchronized with
    # model.VectorCNNConfig.vector_max_distance_um (currently 16 um).
    vector_max_distance_um: float = 16.0
    center_sigma_um: float = 1.0
    center_interior_fraction: float = 0.70
    boundary_radius_um: float = 0.75

    # Synthetic Stage-2 failure generation.
    bridge_radius_um: float = 0.45
    closing_radius_um: float = 0.0
    require_single_input_component: bool = True

    # Pair discovery and crop validity.
    adjacency_max_distance_um: float = 2.5
    border_margin_voxels: int = 1
    min_instance_voxels_after_resampling: int = 4

    # Resampling / intensity normalization.
    anti_alias_image: bool = True
    image_percentile_low: float = 1.0
    image_percentile_high: float = 99.8

    def __post_init__(self) -> None:
        if len(self.target_spacing_zyx_um) != 3 or any(
            not isfinite(float(v)) or float(v) <= 0 for v in self.target_spacing_zyx_um
        ):
            raise ValueError("target_spacing_zyx_um must contain three positive values")
        if len(self.crop_shape_zyx) != 3 or any(int(v) <= 0 for v in self.crop_shape_zyx):
            raise ValueError("crop_shape_zyx must contain three positive values")
        positive = (
            "edt_clip_um",
            "marker_sigma_um",
            "vector_max_distance_um",
            "center_sigma_um",
            "boundary_radius_um",
            "adjacency_max_distance_um",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative = ("bridge_radius_um", "closing_radius_um")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 < self.center_interior_fraction <= 1.0:
            raise ValueError("center_interior_fraction must be in (0, 1]")
        if not 0.0 <= self.image_percentile_low < self.image_percentile_high <= 100.0:
            raise ValueError("image percentiles must satisfy 0 <= low < high <= 100")
        if self.border_margin_voxels < 0:
            raise ValueError("border_margin_voxels cannot be negative")
        if self.min_instance_voxels_after_resampling <= 0:
            raise ValueError("min_instance_voxels_after_resampling must be positive")


DEFAULT_SAMPLE_BUILD_CONFIG = SampleBuildConfig()
