"""Configuration for the Stage 10 division-characteristics investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class InvestigationConfig:
    """Parameters defining one reproducible analysis run."""

    scenes_root: Path | None = None
    voxel_size_zyx_um: tuple[float, float, float] = DEFAULT_VOXEL_SIZE_ZYX_UM
    core_erosion_um: float = 0.8
    fixed_radius_um: float = 2.5
    background_shell_inner_um: float = 0.8
    background_shell_outer_um: float = 3.0
    baseline_exclude_last_parent_frames: int = 1
    minimum_baseline_frames: int = 2
    create_plots: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(
            value <= 0 for value in self.voxel_size_zyx_um
        ):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if self.core_erosion_um < 0:
            raise ValueError("core_erosion_um cannot be negative")
        if self.fixed_radius_um <= 0:
            raise ValueError("fixed_radius_um must be positive")
        if self.background_shell_inner_um < 0:
            raise ValueError("background_shell_inner_um cannot be negative")
        if self.background_shell_outer_um <= self.background_shell_inner_um:
            raise ValueError(
                "background_shell_outer_um must exceed background_shell_inner_um"
            )
        if self.baseline_exclude_last_parent_frames < 0:
            raise ValueError("baseline_exclude_last_parent_frames cannot be negative")
        if self.minimum_baseline_frames < 1:
            raise ValueError("minimum_baseline_frames must be at least one")

    def resolved_scenes_root(self, paths) -> Path:
        return (
            Path(self.scenes_root)
            if self.scenes_root is not None
            else paths.tracking_scenes / "divisions"
        )

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        if data["scenes_root"] is not None:
            data["scenes_root"] = str(data["scenes_root"])
        return data
