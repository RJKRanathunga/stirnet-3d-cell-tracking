"""Canonical paths for the repository's existing storage layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the project without depending on a notebook working directory."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelinePaths:
    """Path construction that preserves the current on-disk layout."""

    project_root: Path

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "PipelinePaths":
        return cls(find_project_root(start))

    @property
    def data_root(self) -> Path:
        return self.project_root / "data" / "sample"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def training_root(self) -> Path:
        return self.data_root / "biohub_5samples_20timepoints" / "train"

    def sample_zarr(self, sample_id: str) -> Path:
        return (
            self.training_root
            / sample_id
            / f"{sample_id}.zarr"
        )

    def sample_zarr_array(self, sample_id: str) -> Path:
        return self.sample_zarr(sample_id) / "0"

    def processed_dataset(self, sample_id: str) -> Path:
        return self.processed_root / "stage_6_processed_dataset" / sample_id

    def processed_series(self, sample_id: str, name: str) -> Path:
        if name not in {"preprocessing", "masking", "segmentation", "cells"}:
            raise ValueError(f"Unknown processed series: {name}")
        return self.processed_dataset(sample_id) / name

    @property
    def stage7_tracking(self) -> Path:
        return self.processed_root / "stage_7_cell_tracking"

    @property
    def stage8_stitching(self) -> Path:
        return self.processed_root / "stage_8_track_stitching"

    @property
    def tracking_scenes(self) -> Path:
        return self.project_root / "data" / "tracking_scenes"


def get_sample_output_dir(
    sample_id: str,
    *,
    paths: PipelinePaths | None = None,
) -> Path:
    """Return the existing Stage 6 output directory for one sample."""

    resolved = paths or PipelinePaths.discover()
    return resolved.processed_dataset(sample_id)


def get_stage_dir(
    sample_id: str,
    stage: str,
    *,
    paths: PipelinePaths | None = None,
) -> Path:
    """Return a named stage directory below one sample's processed output."""

    return get_sample_output_dir(sample_id, paths=paths) / stage
