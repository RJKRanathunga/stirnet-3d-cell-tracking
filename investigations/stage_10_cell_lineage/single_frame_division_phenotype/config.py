"""Configuration for the single-frame division phenotype investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class PhenotypeInvestigationConfig:
    """Parameters defining one reproducible, read-only investigation run."""

    scenes_root: Path | None = None
    voxel_size_zyx_um: tuple[float, float, float] = DEFAULT_VOXEL_SIZE_ZYX_UM
    boundary_margin_um: float = 4.0
    local_population_radius_um: float = 25.0
    background_shell_inner_um: float = 0.8
    background_shell_outer_um: float = 3.0
    intensity_core_fraction: float = 0.66
    intensity_middle_fraction: float = 0.33
    internal_peak_minimum_distance_um: float = 0.8
    internal_peak_threshold_percentile: float = 90.0
    geometric_peak_threshold_fraction: float = 0.50
    minimum_size_model_controls: int = 20
    size_model_degree: int = 1
    gallery_control_count: int = 12
    gallery_local_control_count: int = 6
    gallery_half_size_um: tuple[float, float, float] = (8.0, 10.0, 10.0)
    random_seed: int = 17
    create_plots: bool = True
    create_galleries: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(v <= 0 for v in self.voxel_size_zyx_um):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if self.boundary_margin_um < 0:
            raise ValueError("boundary_margin_um cannot be negative")
        if self.local_population_radius_um <= 0:
            raise ValueError("local_population_radius_um must be positive")
        if self.background_shell_inner_um < 0:
            raise ValueError("background_shell_inner_um cannot be negative")
        if self.background_shell_outer_um <= self.background_shell_inner_um:
            raise ValueError("background_shell_outer_um must exceed inner shell radius")
        if not 0 < self.intensity_middle_fraction < self.intensity_core_fraction < 1:
            raise ValueError("radial fractions must satisfy 0 < middle < core < 1")
        if self.internal_peak_minimum_distance_um <= 0:
            raise ValueError("internal_peak_minimum_distance_um must be positive")
        if not 0 < self.internal_peak_threshold_percentile < 100:
            raise ValueError("internal_peak_threshold_percentile must be between 0 and 100")
        if not 0 < self.geometric_peak_threshold_fraction <= 1:
            raise ValueError("geometric_peak_threshold_fraction must be in (0, 1]")
        if self.minimum_size_model_controls < 3:
            raise ValueError("minimum_size_model_controls must be at least three")
        if self.size_model_degree not in {1, 2}:
            raise ValueError("size_model_degree must be one or two")
        if self.gallery_control_count < 0 or self.gallery_local_control_count < 0:
            raise ValueError("gallery control counts cannot be negative")
        if self.gallery_local_control_count > self.gallery_control_count:
            raise ValueError("gallery_local_control_count cannot exceed total gallery controls")
        if len(self.gallery_half_size_um) != 3 or any(v <= 0 for v in self.gallery_half_size_um):
            raise ValueError("gallery_half_size_um must contain three positive values")

    def resolved_scenes_root(self, paths) -> Path:
        return Path(self.scenes_root) if self.scenes_root is not None else paths.tracking_scenes / "divisions"

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        if data["scenes_root"] is not None:
            data["scenes_root"] = str(data["scenes_root"])
        data["voxel_size_zyx_um"] = list(self.voxel_size_zyx_um)
        data["gallery_half_size_um"] = list(self.gallery_half_size_um)
        return data
