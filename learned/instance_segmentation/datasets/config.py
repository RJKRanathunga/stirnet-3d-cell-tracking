"""Configuration for object-centric external 3-D instance-segmentation samples."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from os import environ
from pathlib import Path

_WINDOWS_C_ELEGANS_ROOT = Path(
    r"D:\Projects\Kaggle\cell-tracking\data\external\c_elegans_nuclei"
)

C_ELEGANS_SPACING_ZYX_UM = (0.122, 0.116, 0.116)
BLASTOSPIM_SPACING_ZYX_UM = (2.0, 0.208, 0.208)

# The output tensor keeps the existing Biohub-shaped sampling grid.  In the new
# architecture these numbers define CANONICAL ROI UNITS, not an assertion that
# every normalized source cell still has its original biological micrometre size.
CANONICAL_SPACING_ZYX = (1.625, 0.40625, 0.40625)
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
    env_value = environ.get("C_ELEGANS_NUCLEI_DIR")
    if env_value:
        return Path(env_value).expanduser()
    if _WINDOWS_C_ELEGANS_ROOT.exists():
        return _WINDOWS_C_ELEGANS_ROOT
    return Path("data") / "external" / "c_elegans_nuclei"


def default_nis3d_root() -> Path:
    return _default_external_root("NIS3D_DIR", "nis3d", "NIS3D")


def default_blastospim_root() -> Path:
    return _default_external_root("BLASTOSPIM_DIR", "blastospim", "BlastoSPIM")


@dataclass(frozen=True)
class SampleBuildConfig:
    """Settings for canonical object-centric CNN sample generation.

    ``component_occupancy`` is the central design parameter.  A selected single,
    pair, or larger group is isotropically scaled in physical source space so
    that its tight union bounding box occupies at most this fraction of the
    usable canonical span along its limiting axis.  The same scalar is applied
    to Z, Y, and X physical coordinates, preserving source morphology.
    """

    crop_shape_zyx: tuple[int, int, int] = CANONICAL_CROP_SHAPE_ZYX
    canonical_spacing_zyx: tuple[float, float, float] = CANONICAL_SPACING_ZYX

    # Object-centric normalization.
    component_occupancy: float = 0.78
    border_margin_voxels: int = 1
    min_normalization_scale: float = 0.05
    max_normalization_scale: float = 8.0

    # Canonical-coordinate input channels.
    edt_clip_canonical: float = 8.0
    marker_sigma_canonical: float = 1.1

    # Canonical-coordinate supervision.
    center_sigma_canonical: float = 1.0
    center_interior_fraction: float = 0.70
    boundary_radius_canonical: float = 0.75

    # Synthetic Stage-2 failure generation on the canonical ROI.
    bridge_radius_canonical: float = 0.45
    closing_radius_canonical: float = 0.0
    require_single_input_component: bool = True

    # Native candidate discovery.  This is intentionally permissive; pair
    # geometry is normalized only after selection.
    adjacency_max_distance_um: float = 2.5

    # Resampling quality.
    anti_alias_image: bool = True
    min_instance_voxels_after_resampling: int = 12
    min_instance_bbox_zyx_vox: tuple[int, int, int] = (2, 3, 3)
    max_group_aspect_ratio: float = 12.0

    # Intensity normalization.
    image_percentile_low: float = 1.0
    image_percentile_high: float = 99.8

    def __post_init__(self) -> None:
        if len(self.crop_shape_zyx) != 3 or any(int(v) <= 0 for v in self.crop_shape_zyx):
            raise ValueError("crop_shape_zyx must contain three positive values")
        if len(self.canonical_spacing_zyx) != 3 or any(
            not isfinite(float(v)) or float(v) <= 0 for v in self.canonical_spacing_zyx
        ):
            raise ValueError("canonical_spacing_zyx must contain three positive values")
        if not 0.0 < float(self.component_occupancy) <= 1.0:
            raise ValueError("component_occupancy must be in (0, 1]")
        if self.border_margin_voxels < 0:
            raise ValueError("border_margin_voxels cannot be negative")
        if not 0 < self.min_normalization_scale <= self.max_normalization_scale:
            raise ValueError("normalization scale bounds are invalid")
        positive = (
            "edt_clip_canonical",
            "marker_sigma_canonical",
            "center_sigma_canonical",
            "boundary_radius_canonical",
            "adjacency_max_distance_um",
            "max_group_aspect_ratio",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative = ("bridge_radius_canonical", "closing_radius_canonical")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 < self.center_interior_fraction <= 1.0:
            raise ValueError("center_interior_fraction must be in (0, 1]")
        if self.min_instance_voxels_after_resampling <= 0:
            raise ValueError("min_instance_voxels_after_resampling must be positive")
        if len(self.min_instance_bbox_zyx_vox) != 3 or any(
            int(v) <= 0 for v in self.min_instance_bbox_zyx_vox
        ):
            raise ValueError("min_instance_bbox_zyx_vox must contain three positive values")
        if not 0.0 <= self.image_percentile_low < self.image_percentile_high <= 100.0:
            raise ValueError("image percentiles must satisfy 0 <= low < high <= 100")

    # Read-only migration aliases for notebooks/debugging code written against
    # the previous fixed-physical-FOV configuration.  They intentionally return
    # canonical ROI values; no biological-size normalization is implied.
    @property
    def target_spacing_zyx_um(self) -> tuple[float, float, float]:
        return self.canonical_spacing_zyx

    @property
    def edt_clip_um(self) -> float:
        return self.edt_clip_canonical

    @property
    def marker_sigma_um(self) -> float:
        return self.marker_sigma_canonical

    @property
    def center_sigma_um(self) -> float:
        return self.center_sigma_canonical

    @property
    def boundary_radius_um(self) -> float:
        return self.boundary_radius_canonical

    @property
    def bridge_radius_um(self) -> float:
        return self.bridge_radius_canonical

    @property
    def closing_radius_um(self) -> float:
        return self.closing_radius_canonical

    @property
    def usable_span_canonical(self) -> tuple[float, float, float]:
        """Center-to-center canonical span available after hard border margins."""
        result = []
        for shape, spacing in zip(self.crop_shape_zyx, self.canonical_spacing_zyx):
            usable_intervals = int(shape) - 1 - 2 * int(self.border_margin_voxels)
            if usable_intervals <= 0:
                raise ValueError("border margin leaves no usable canonical span")
            result.append(float(usable_intervals) * float(spacing))
        return tuple(result)  # type: ignore[return-value]


DEFAULT_SAMPLE_BUILD_CONFIG = SampleBuildConfig()

# Compatibility aliases for external code that imported the old names.  The
# semantics have changed: these are canonical ROI units, not biological target
# spacing that forces every dataset into Biohub absolute scale.
TARGET_SPACING_ZYX_UM = CANONICAL_SPACING_ZYX
