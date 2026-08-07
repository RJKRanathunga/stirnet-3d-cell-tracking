"""Read the per-sample artifacts written by data/full_processed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MergeRealPaths


STAGE_DIRS = {
    6: "stage_6_processed_dataset",
    7: "stage_7_cell_tracking",
    8: "stage_8_track_stitching",
    10: "stage_10_cell_lineage",
    11: "stage_11_track_reconciliation",
}


@dataclass(frozen=True)
class SampleArtifacts:
    paths: MergeRealPaths
    sample_id: str

    @property
    def sample_root(self) -> Path:
        return self.paths.full_processed_root / "train" / self.sample_id

    def stage_root(self, stage: int) -> Path:
        try:
            name = STAGE_DIRS[int(stage)]
        except KeyError as exc:
            raise ValueError(f"Unsupported stage {stage}") from exc
        return self.sample_root / name

    @property
    def stage6_root(self) -> Path:
        return self.stage_root(6)

    @property
    def stage11_root(self) -> Path:
        return self.stage_root(11)

    @property
    def source_metadata_path(self) -> Path:
        return self.sample_root / "source.json"

    @property
    def source_metadata(self) -> dict[str, Any]:
        return _read_json(self.source_metadata_path)

    @property
    def frame_count(self) -> int:
        value = self.source_metadata.get("frame_count")
        if value is not None:
            return int(value)
        return len(tuple((self.stage6_root / "cells").glob("t*.csv")))

    @property
    def spatial_shape_zyx(self) -> tuple[int, int, int]:
        shape = self.source_metadata.get("raw_shape_tzyx")
        if isinstance(shape, list) and len(shape) == 4:
            return tuple(int(v) for v in shape[1:])
        return tuple(int(v) for v in self.labels(0, mmap=True).shape)

    def validate(self) -> None:
        if not self.sample_root.is_dir():
            raise FileNotFoundError(self.sample_root)
        for stage in (6, 11):
            root = self.stage_root(stage)
            if not root.is_dir():
                raise FileNotFoundError(root)
        required_stage11 = (
            "tracks.csv",
            "endpoint_classifications.csv",
            "continuation_candidates.csv",
            "continuation_decisions.csv",
            "track_id_remap.csv",
            "unresolved_endings.csv",
            "division_events.csv",
            "segmentation_events.csv",
        )
        missing = [name for name in required_stage11 if not (self.stage11_root / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.sample_id}: Stage 11 is missing required files: {missing}"
            )
        if self.frame_count < 1:
            raise ValueError(f"{self.sample_id}: no Stage 6 frames found")

    def cell_file(self, frame: int) -> Path:
        return self.stage6_root / "cells" / f"t{int(frame):03d}.csv"

    def labels_file(self, frame: int) -> Path:
        return self.stage6_root / "segmentation" / f"t{int(frame):03d}.npy"

    def preprocessed_file(self, frame: int) -> Path:
        return self.stage6_root / "preprocessing" / f"t{int(frame):03d}.npy"

    def mask_file(self, frame: int) -> Path:
        return self.stage6_root / "masking" / f"t{int(frame):03d}.npy"

    @lru_cache(maxsize=12)
    def cells(self, frame: int) -> pd.DataFrame:
        return pd.read_csv(self.cell_file(frame))

    @lru_cache(maxsize=8)
    def _labels_mmap(self, frame: int):
        return np.load(self.labels_file(frame), mmap_mode="r", allow_pickle=False)

    def labels(self, frame: int, *, mmap: bool = False):
        if mmap:
            return self._labels_mmap(int(frame))
        return np.load(self.labels_file(frame), allow_pickle=False)

    def preprocessed(self, frame: int, *, mmap: bool = False):
        return np.load(
            self.preprocessed_file(frame),
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )

    def binary_mask(self, frame: int, *, mmap: bool = False):
        return np.load(
            self.mask_file(frame),
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )

    @lru_cache(maxsize=None)
    def stage11_table(self, filename: str) -> pd.DataFrame:
        path = self.stage11_root / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @property
    def tracks(self) -> pd.DataFrame:
        return self.stage11_table("tracks.csv")

    @property
    def endpoint_classifications(self) -> pd.DataFrame:
        return self.stage11_table("endpoint_classifications.csv")

    @property
    def continuation_candidates(self) -> pd.DataFrame:
        return self.stage11_table("continuation_candidates.csv")

    @property
    def continuation_decisions(self) -> pd.DataFrame:
        return self.stage11_table("continuation_decisions.csv")

    @property
    def track_id_remap(self) -> pd.DataFrame:
        return self.stage11_table("track_id_remap.csv")

    @property
    def unresolved_endings(self) -> pd.DataFrame:
        return self.stage11_table("unresolved_endings.csv")

    @property
    def division_events(self) -> pd.DataFrame:
        return self.stage11_table("division_events.csv")

    @property
    def segmentation_events(self) -> pd.DataFrame:
        return self.stage11_table("segmentation_events.csv")

    def raw_frame(self, frame: int) -> np.ndarray | None:
        """Load raw data through the repository's existing Zarr API when available."""
        zarr_path = self.source_metadata.get("zarr_path")
        if not zarr_path:
            return None
        try:
            from src.io import load_timepoint  # type: ignore

            return np.asarray(load_timepoint(Path(zarr_path), int(frame)))
        except Exception:
            return None


class FullProcessedRepository:
    def __init__(self, paths: MergeRealPaths) -> None:
        self.paths = paths

    @property
    def train_root(self) -> Path:
        return self.paths.full_processed_root / "train"

    def sample_ids(self, requested: tuple[str, ...] = ()) -> tuple[str, ...]:
        if not self.train_root.is_dir():
            raise FileNotFoundError(
                f"Full processed train directory does not exist: {self.train_root}"
            )
        discovered = tuple(sorted(p.name for p in self.train_root.iterdir() if p.is_dir()))
        if not requested:
            return discovered
        wanted = tuple(str(v) for v in requested)
        missing = sorted(set(wanted) - set(discovered))
        if missing:
            raise FileNotFoundError(f"Requested samples are not in full_processed: {missing}")
        return wanted

    def sample(self, sample_id: str) -> SampleArtifacts:
        sample = SampleArtifacts(self.paths, str(sample_id))
        sample.validate()
        return sample


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["FullProcessedRepository", "SampleArtifacts", "STAGE_DIRS"]
