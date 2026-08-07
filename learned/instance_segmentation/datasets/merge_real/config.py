"""Configuration for mining and reviewing real merged-cell cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MergeRealConfig:
    """Research configuration for real merge-case mining.

    Thresholds are deliberately permissive. The miner is a high-recall candidate
    generator; the three manual review passes provide the ground truth.
    """

    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    reference_history_frames: int = 3
    motion_history_frames: int = 4

    # Candidate association.
    prediction_component_radius_um: float = 6.0
    source3_require_continuing_next_frame: bool = True

    # Combined-volume evidence. Keep broad during mining; rank near 1.0 higher.
    volume_sum_ratio_min: float = 0.50
    volume_sum_ratio_max: float = 1.60
    strong_volume_sum_ratio_min: float = 0.70
    strong_volume_sum_ratio_max: float = 1.30

    # Source 4: component anomaly recall source.
    anomaly_frame_volume_ratio: float = 1.45
    anomaly_track_volume_ratio: float = 1.40
    anomaly_min_edt_peaks: int = 2
    anomaly_edt_peak_relative_height: float = 0.50
    anomaly_edt_peak_separation_um: float = 2.0

    # Structural exclusions.
    division_guard_frames: int = 1
    exclude_boundary_components: bool = True

    # Review/extraction.
    review_context_frames: int = 2
    crop_margin_um: float = 6.0

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(v <= 0 for v in self.voxel_size_zyx_um):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if self.reference_history_frames < 1 or self.motion_history_frames < 1:
            raise ValueError("history lengths must be positive")
        if not 0 < self.volume_sum_ratio_min < self.volume_sum_ratio_max:
            raise ValueError("invalid volume_sum_ratio range")
        if not 0 < self.strong_volume_sum_ratio_min < self.strong_volume_sum_ratio_max:
            raise ValueError("invalid strong volume_sum_ratio range")
        if self.prediction_component_radius_um <= 0 or self.crop_margin_um <= 0:
            raise ValueError("physical radii must be positive")
        if self.division_guard_frames < 0:
            raise ValueError("division_guard_frames cannot be negative")


@dataclass(frozen=True)
class MergeRealPaths:
    """Canonical local paths for this learned dataset subproject."""

    project_root: Path
    full_processed_root: Path
    output_root: Path

    @classmethod
    def discover(
        cls,
        project_root: str | Path | None = None,
        *,
        full_processed_root: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> "MergeRealPaths":
        root = _find_project_root(project_root)
        full = Path(full_processed_root).expanduser().resolve() if full_processed_root else root / "data" / "full_processed"
        output = Path(output_root).expanduser().resolve() if output_root else root / "data" / "learned" / "instance_segmentation" / "merge_real"
        return cls(root, full, output)

    @property
    def mining_dir(self) -> Path:
        return self.output_root / "mining"

    @property
    def reviews_dir(self) -> Path:
        return self.output_root / "reviews"

    @property
    def filter_reviews_csv(self) -> Path:
        return self.reviews_dir / "filter_reviews.csv"

    @property
    def centers_dir(self) -> Path:
        return self.reviews_dir / "centers"

    @property
    def partitions_dir(self) -> Path:
        return self.reviews_dir / "partitions"

    @property
    def cases_dir(self) -> Path:
        return self.output_root / "cases"


def _find_project_root(start: str | Path | None) -> Path:
    if start is not None:
        current = Path(start).expanduser().resolve()
        if current.is_file():
            current = current.parent
        candidates = (current, *current.parents)
    else:
        current = Path.cwd().resolve()
        candidates = (current, *current.parents)

    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the cell-detection repository root. Run from inside the "
        "repository or pass --project-root explicitly."
    )


DEFAULT_CONFIG = MergeRealConfig()

__all__ = ["DEFAULT_CONFIG", "MergeRealConfig", "MergeRealPaths"]
