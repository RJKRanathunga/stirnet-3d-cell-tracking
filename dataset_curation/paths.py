from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


VALID_SPLITS = ("train", "test")
PERSISTED_LABEL_DTYPE = np.dtype(np.uint16)


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
        """The one canonical production inference root for this volume."""
        return self.preprocessed_split_root / self.volume_id

    @property
    def inference_manifest(self) -> Path:
        return self.preprocessed_root / "curation_manifest.json"

    @property
    def skip_marker(self) -> Path:
        return self.preprocessed_root / "_SKIPPED.json"

    @property
    def movies(self) -> Path:
        return self.preprocessed_root / "movies"

    @property
    def supervoxels(self) -> Path:
        return self.movies / "supervoxels.npy"

    @property
    def final_instances(self) -> Path:
        return self.movies / "final_instances.npy"

    @property
    def cells_dir(self) -> Path:
        return self.preprocessed_root / "cells"

    @property
    def cells_csv(self) -> Path:
        return self.preprocessed_root / "cells_all.csv"

    @property
    def spatial_success(self) -> Path:
        return self.preprocessed_root / "_SPATIAL_SUCCESS.json"

    @property
    def spatial_summary(self) -> Path:
        return self.preprocessed_root / "spatial_summary.json"

    @property
    def trackastra_root(self) -> Path:
        return self.preprocessed_root / "trackastra"

    @property
    def track_graph(self) -> Path:
        return self.trackastra_root / "track_graph.pkl"

    @property
    def napari_tracks(self) -> Path:
        return self.trackastra_root / "napari_tracks.npy"

    @property
    def napari_graph(self) -> Path:
        return self.trackastra_root / "napari_graph.json"

    @property
    def tracks_csv(self) -> Path:
        return self.trackastra_root / "tracks.csv"

    @property
    def trackastra_summary(self) -> Path:
        return self.trackastra_root / "summary.json"

    @property
    def suspect_scores(self) -> Path:
        return self.preprocessed_root / "suspects"

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

    def ensure_output_roots(self) -> None:
        self.preprocessed_split_root.mkdir(parents=True, exist_ok=True)
        self.annotations_split_root.mkdir(parents=True, exist_ok=True)

    def has_ground_truth_files(self) -> bool:
        return (
            self.ground_truth_nodes.is_file()
            and self.ground_truth_edges.is_file()
        )

    def has_any_preprocessed_data(self) -> bool:
        root = self.preprocessed_root
        if not root.is_dir():
            return False
        try:
            return next(root.iterdir(), None) is not None
        except OSError:
            return True

    def inference_skipped(self) -> bool:
        return self.skip_marker.is_file()

    def read_skip_record(self) -> dict:
        if not self.skip_marker.is_file():
            raise FileNotFoundError(self.skip_marker)
        payload = json.loads(
            self.skip_marker.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected JSON object in {self.skip_marker}"
            )
        return payload

    def inference_id(self) -> str | None:
        if not self.inference_manifest.is_file():
            return None
        try:
            payload = json.loads(
                self.inference_manifest.read_text(encoding="utf-8")
            )
            value = str(payload.get("inference_id", "")).strip()
            return value or None
        except Exception:
            return None

    @staticmethod
    def _label_movie_is_valid(
        path: Path,
        frame_count: int | None,
    ) -> bool:
        if not path.is_file():
            return False
        try:
            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if array.ndim != 4:
                return False
            if array.dtype != PERSISTED_LABEL_DTYPE:
                return False
            if (
                frame_count is not None
                and int(array.shape[0]) != int(frame_count)
            ):
                return False
            return True
        except Exception:
            return False

    def spatial_complete(
        self,
        *,
        frame_count: int | None = None,
    ) -> bool:
        """
        Compact persistent spatial contract.

        Raw imagery remains in source Zarr. Preprocessed intensities, binary
        masks, source instances and the 5-channel STIR-Net input are ephemeral.
        Only uint16 atomic supervoxels + final spatial instances are persisted.
        """
        required = (
            self.supervoxels,
            self.final_instances,
            self.cells_csv,
            self.spatial_success,
        )
        if not all(path.is_file() for path in required):
            return False

        return (
            self._label_movie_is_valid(
                self.final_instances,
                frame_count,
            )
            and self._label_movie_is_valid(
                self.supervoxels,
                frame_count,
            )
        )

    def tracking_complete(self) -> bool:
        required = (
            self.track_graph,
            self.napari_tracks,
            self.napari_graph,
            self.tracks_csv,
            self.trackastra_summary,
        )
        return all(path.is_file() for path in required)

    def inference_complete(
        self,
        *,
        frame_count: int | None = None,
    ) -> bool:
        return (
            not self.inference_skipped()
            and self.spatial_complete(frame_count=frame_count)
            and self.tracking_complete()
            and self.inference_id() is not None
        )
