"""Canonical paths for the repository's existing storage layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT_ENV = "CELL_TRACKING_PROJECT_ROOT"
PROJECT_ROOT_MARKERS = ("src", "notebooks", "pyproject.toml")


def _normalize(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_project_root(path: Path) -> bool:
    return (
        (path / "src").is_dir()
        and (path / "notebooks").is_dir()
        and (path / "pyproject.toml").is_file()
    )


def _invalid_root(path: Path | str, source: str) -> RuntimeError:
    markers = ", ".join(PROJECT_ROOT_MARKERS)
    return RuntimeError(
        f"Invalid project root from {source}: {path}. "
        f"Expected repository markers: {markers}."
    )


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate and validate the project root without depending on the CWD."""

    if start is not None:
        current = _normalize(start)
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            if _is_project_root(candidate):
                return candidate
        raise _invalid_root(current, "explicit input")

    environment_root = os.environ.get(PROJECT_ROOT_ENV)
    if environment_root is not None:
        if not environment_root.strip():
            raise _invalid_root("<empty>", f"{PROJECT_ROOT_ENV} environment override")
        candidate = _normalize(environment_root)
        if _is_project_root(candidate):
            return candidate
        raise _invalid_root(
            candidate,
            f"{PROJECT_ROOT_ENV} environment override",
        )

    module_root = Path(__file__).resolve().parents[2]
    if _is_project_root(module_root):
        return module_root
    raise _invalid_root(module_root, "installed module location")


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
    def stage9_comparisons(self) -> Path:
        return self.project_root / "data" / "comparisons" / "stage_09"

    @property
    def stage10_lineage(self) -> Path:
        return self.processed_root / "stage_10_cell_lineage"

    @property
    def stage11_reconciliation(self) -> Path:
        return self.processed_root / "stage_11_track_reconciliation"

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
