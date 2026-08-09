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

# The learned model now operates on a normalized cubic lattice.  These are
# canonical coordinate units, not biological micrometres.  One canonical voxel
# has the same size along Z, Y, and X.
CANONICAL_SPACING_ZYX = (1.0, 1.0, 1.0)
CANONICAL_CROP_SHAPE_ZYX = (64, 64, 64)


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
    """Settings for cubic object-centric CNN sample generation.

    A selected single/pair/group is interpreted in true source physical space
    using the dataset's native voxel spacing.  One scalar ``normalization_scale``
    (canonical voxels per source micrometre) then maps the whole group into a
    fixed cubic ``64 x 64 x 64`` canonical lattice.  No axis-specific stretching
    is permitted.
    """

    crop_shape_zyx: tuple[int, int, int] = CANONICAL_CROP_SHAPE_ZYX
    canonical_spacing_zyx: tuple[float, float, float] = CANONICAL_SPACING_ZYX

    # Object-centric normalization.
    component_occupancy: float = 0.78
    border_margin_voxels: int = 1
    min_normalization_scale: float = 0.05  # canonical voxels / source um
    max_normalization_scale: float = 32.0  # canonical voxels / source um

    # Canonical-voxel input channels.
    edt_clip_vox: float = 16.0
    marker_sigma_vox: float = 2.0

    # Canonical-voxel supervision.
    center_sigma_vox: float = 2.0
    center_interior_fraction: float = 0.70
    boundary_radius_vox: float = 1.5

    # Synthetic Stage-2 failure generation in cubic canonical voxels.
    bridge_radius_vox: float = 1.5
    closing_radius_vox: float = 0.0
    require_single_input_component: bool = True

    # Training-only source quality gates.  Boundary-truncated GT nuclei do not
    # provide complete center/shape supervision and are rejected by default.
    reject_native_boundary_instances: bool = True
    native_boundary_margin_voxels: int = 0

    # Native candidate discovery.  Pair discovery still happens in true source
    # physical space before object normalization.
    adjacency_max_distance_um: float = 2.5

    # Resampling quality / sample validity.
    anti_alias_image: bool = True
    min_instance_voxels_after_resampling: int = 24
    min_instance_bbox_zyx_vox: tuple[int, int, int] = (3, 3, 3)
    max_group_aspect_ratio: float = 12.0

    # Intensity normalization.
    image_percentile_low: float = 1.0
    image_percentile_high: float = 99.8

    def __post_init__(self) -> None:
        if len(self.crop_shape_zyx) != 3 or any(int(v) <= 0 for v in self.crop_shape_zyx):
            raise ValueError("crop_shape_zyx must contain three positive values")
        if tuple(int(v) for v in self.crop_shape_zyx) != CANONICAL_CROP_SHAPE_ZYX:
            # Custom shapes remain useful for tests, but production defaults are 64^3.
            if any(int(v) < 16 for v in self.crop_shape_zyx):
                raise ValueError("custom canonical shapes must be at least 16 voxels per axis")
        if len(self.canonical_spacing_zyx) != 3 or any(
            not isfinite(float(v)) or float(v) <= 0 for v in self.canonical_spacing_zyx
        ):
            raise ValueError("canonical_spacing_zyx must contain three positive values")
        if not all(abs(float(v) - float(self.canonical_spacing_zyx[0])) < 1e-9 for v in self.canonical_spacing_zyx):
            raise ValueError("canonical voxels must be cubic: spacing must be isotropic")
        if not 0.0 < float(self.component_occupancy) <= 1.0:
            raise ValueError("component_occupancy must be in (0, 1]")
        if self.border_margin_voxels < 0:
            raise ValueError("border_margin_voxels cannot be negative")
        if self.native_boundary_margin_voxels < 0:
            raise ValueError("native_boundary_margin_voxels cannot be negative")
        if not 0 < self.min_normalization_scale <= self.max_normalization_scale:
            raise ValueError("normalization scale bounds are invalid")
        positive = (
            "edt_clip_vox",
            "marker_sigma_vox",
            "center_sigma_vox",
            "boundary_radius_vox",
            "adjacency_max_distance_um",
            "max_group_aspect_ratio",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative = ("bridge_radius_vox", "closing_radius_vox")
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

    @property
    def usable_span_vox(self) -> tuple[float, float, float]:
        """Center-to-center canonical voxel span after hard border margins."""
        result = []
        for shape in self.crop_shape_zyx:
            usable_intervals = int(shape) - 1 - 2 * int(self.border_margin_voxels)
            if usable_intervals <= 0:
                raise ValueError("border margin leaves no usable canonical span")
            result.append(float(usable_intervals))
        return tuple(result)  # type: ignore[return-value]

    # Compatibility aliases retained for older notebooks.  Their values now
    # refer to unit cubic canonical voxels, never biological micrometres.
    @property
    def target_spacing_zyx_um(self) -> tuple[float, float, float]:
        return self.canonical_spacing_zyx

    @property
    def edt_clip_canonical(self) -> float:
        return self.edt_clip_vox

    @property
    def marker_sigma_canonical(self) -> float:
        return self.marker_sigma_vox

    @property
    def center_sigma_canonical(self) -> float:
        return self.center_sigma_vox

    @property
    def boundary_radius_canonical(self) -> float:
        return self.boundary_radius_vox

    @property
    def bridge_radius_canonical(self) -> float:
        return self.bridge_radius_vox

    @property
    def closing_radius_canonical(self) -> float:
        return self.closing_radius_vox

    @property
    def usable_span_canonical(self) -> tuple[float, float, float]:
        spacing = float(self.canonical_spacing_zyx[0])
        return tuple(v * spacing for v in self.usable_span_vox)  # type: ignore[return-value]


DEFAULT_SAMPLE_BUILD_CONFIG = SampleBuildConfig()

# Deprecated compatibility names.  Canonical voxels are now cubic.
TARGET_SPACING_ZYX_UM = CANONICAL_SPACING_ZYX
