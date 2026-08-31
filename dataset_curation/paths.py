from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


VALID_SPLITS = ("train", "test")


def validate_split(split: str) -> str:
    value = str(split).strip().lower()
    if value not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {VALID_SPLITS}, got {split!r}"
        )
    return value


@dataclass(frozen=True)
class BioHubVolumePaths:
    """Canonical external-drive paths for one BioHub volume."""

    data_root: Path
    split: str
    volume_id: str

    def __post_init__(self) -> None:
        validate_split(self.split)

    @property
    def source_split_root(self) -> Path:
        return self.data_root / "source" / self.split

    @property
    def source_root(self) -> Path:
        return self.source_split_root / self.volume_id

    @property
    def zarr(self) -> Path:
        return self.source_root / f"{self.volume_id}.zarr"

    @property
    def zarr_array(self) -> Path:
        return self.zarr / "0"

    @property
    def ground_truth_root(self) -> Path:
        return self.source_root / "ground_truth"

    @property
    def ground_truth_nodes(self) -> Path:
        return self.ground_truth_root / "ground_truth_nodes.csv"

    @property
    def ground_truth_edges(self) -> Path:
        return self.ground_truth_root / "ground_truth_edges.csv"

    @property
    def preprocessed_split_root(self) -> Path:
        return self.data_root / "preprocessed" / self.split

    @property
    def preprocessed_root(self) -> Path:
        return self.preprocessed_split_root / self.volume_id

    def inference_run(self, run_id: str = "current") -> Path:
        return self.preprocessed_root / str(run_id)

    def inference_manifest(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "curation_manifest.json"

    def movies(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "movies"

    def raw(self, run_id: str = "current") -> Path:
        return self.movies(run_id) / "raw.npy"

    def preprocessed(self, run_id: str = "current") -> Path:
        return self.movies(run_id) / "preprocessed.npy"

    def binary_mask(self, run_id: str = "current") -> Path:
        return self.movies(run_id) / "binary_mask.npy"

    def source_instances(self, run_id: str = "current") -> Path:
        return self.movies(run_id) / "source_instances.npy"

    def supervoxels(self, run_id: str = "current") -> Path:
        return self.movies(run_id) / "supervoxels.npy"

    def final_instances(self, run_id: str = "current") -> Path:
        return self.movies(run_id) / "final_instances.npy"

    def cells_csv(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "cells_all.csv"

    def spatial_success(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "_SPATIAL_SUCCESS.json"

    def spatial_summary(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "spatial_summary.json"

    def trackastra_root(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "trackastra"

    def track_graph(self, run_id: str = "current") -> Path:
        return self.trackastra_root(run_id) / "track_graph.pkl"

    def tracked_masks(self, run_id: str = "current") -> Path:
        return self.trackastra_root(run_id) / "tracked_masks.npy"

    def napari_tracks(self, run_id: str = "current") -> Path:
        return self.trackastra_root(run_id) / "napari_tracks.npy"

    def napari_graph(self, run_id: str = "current") -> Path:
        return self.trackastra_root(run_id) / "napari_graph.json"

    def tracks_csv(self, run_id: str = "current") -> Path:
        return self.trackastra_root(run_id) / "tracks.csv"

    def trackastra_summary(self, run_id: str = "current") -> Path:
        return self.trackastra_root(run_id) / "summary.json"

    @property
    def annotations_split_root(self) -> Path:
        return self.data_root / "annotations" / self.split

    @property
    def annotations_root(self) -> Path:
        return self.annotations_split_root / self.volume_id

    def annotation_set(self, name: str = "main") -> Path:
        return self.annotations_root / str(name)

    def annotation_manifest(self, name: str = "main") -> Path:
        return self.annotation_set(name) / "manifest.json"

    def instance_annotations(self, name: str = "main") -> Path:
        return self.annotation_set(name) / "instances"

    def track_annotations(self, name: str = "main") -> Path:
        return self.annotation_set(name) / "tracks"

    def point_annotations(self, name: str = "main") -> Path:
        return self.annotation_set(name) / "points"

    def suspect_scores(self, run_id: str = "current") -> Path:
        return self.inference_run(run_id) / "suspects"

    def ensure_output_roots(self) -> None:
        self.preprocessed_split_root.mkdir(parents=True, exist_ok=True)
        self.annotations_split_root.mkdir(parents=True, exist_ok=True)

    def has_ground_truth_files(self) -> bool:
        return (
            self.ground_truth_nodes.is_file()
            and self.ground_truth_edges.is_file()
        )

    def has_any_preprocessed_data(
        self,
        run_id: str = "current",
    ) -> bool:
        root = self.inference_run(run_id)
        if not root.is_dir():
            return False
        try:
            return next(root.iterdir(), None) is not None
        except OSError:
            return True

    @staticmethod
    def _movie_has_frames(
        path: Path,
        frame_count: int | None,
    ) -> bool:
        if not path.is_file():
            return False
        if frame_count is None:
            return True
        try:
            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
            return (
                array.ndim == 4
                and int(array.shape[0]) == int(frame_count)
            )
        except Exception:
            return False

    def spatial_complete(
        self,
        run_id: str = "current",
        *,
        frame_count: int | None = None,
    ) -> bool:
        # supervoxels.npy is part of the contract because fresh inference must
        # be immediately usable by the merged-cell annotator.
        required = (
            self.raw(run_id),
            self.preprocessed(run_id),
            self.binary_mask(run_id),
            self.source_instances(run_id),
            self.supervoxels(run_id),
            self.final_instances(run_id),
            self.cells_csv(run_id),
            self.spatial_success(run_id),
        )
        if not all(path.is_file() for path in required):
            return False

        return (
            self._movie_has_frames(
                self.final_instances(run_id),
                frame_count,
            )
            and self._movie_has_frames(
                self.supervoxels(run_id),
                frame_count,
            )
        )

    def tracking_complete(
        self,
        run_id: str = "current",
    ) -> bool:
        required = (
            self.track_graph(run_id),
            self.tracked_masks(run_id),
            self.napari_tracks(run_id),
            self.napari_graph(run_id),
            self.tracks_csv(run_id),
            self.trackastra_summary(run_id),
        )
        return all(path.is_file() for path in required)

    def inference_complete(
        self,
        run_id: str = "current",
        *,
        frame_count: int | None = None,
    ) -> bool:
        return (
            self.spatial_complete(
                run_id,
                frame_count=frame_count,
            )
            and self.tracking_complete(run_id)
        )
