"""Configuration for the small-cell size and temporal-variation investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_VARIATION_FEATURES = (
    "analysis_volume_voxels",
    "equivalent_radius",
    "extent",
    "axis_major",
    "axis_middle",
    "axis_minor",
    "elongation",
    "flatness",
    "anisotropy",
    "solidity",
    "compactness",
    "bbox_depth",
    "bbox_height",
    "bbox_width",
    "intensity_mean",
    "intensity_median",
    "intensity_std",
    "intensity_iqr",
    "intensity_cv",
    "intensity_sum",
)


@dataclass(frozen=True)
class SmallCellStatisticsConfig:
    """Parameters defining one reproducible, read-only statistics run."""

    scenes_root: Path | None = None
    voxel_size_zyx_um: tuple[float, float, float] = DEFAULT_VOXEL_SIZE_ZYX_UM
    target_link_max_distance_um: float = 15.0
    control_minimum_observations: int = 5
    control_require_contiguous: bool = True
    control_exclude_boundary: bool = True
    control_exclude_virtual: bool = True
    control_exclude_event_tracks: bool = True
    matched_controls_per_target: int = 10
    variation_features: tuple[str, ...] = DEFAULT_VARIATION_FEATURES
    validate_mask_volumes: bool = True
    create_plots: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(value <= 0 for value in self.voxel_size_zyx_um):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if self.target_link_max_distance_um <= 0:
            raise ValueError("target_link_max_distance_um must be positive")
        if self.control_minimum_observations < 2:
            raise ValueError("control_minimum_observations must be at least two")
        if self.matched_controls_per_target < 0:
            raise ValueError("matched_controls_per_target cannot be negative")
        if not self.variation_features:
            raise ValueError("variation_features cannot be empty")

    def resolved_scenes_root(self, paths) -> Path:
        return Path(self.scenes_root) if self.scenes_root is not None else paths.tracking_scenes / "small_cells"

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        if data["scenes_root"] is not None:
            data["scenes_root"] = str(data["scenes_root"])
        data["voxel_size_zyx_um"] = list(self.voxel_size_zyx_um)
        data["variation_features"] = list(self.variation_features)
        return data
