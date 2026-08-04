"""Repository-aware loading for division-scene analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import load_timepoint, open_sample


@dataclass(frozen=True)
class FrameArtifacts:
    sample_id: str
    frame: int
    raw: np.ndarray
    preprocessed: np.ndarray
    binary_mask: np.ndarray
    instance_labels: np.ndarray
    cells: pd.DataFrame


class RepositoryData:
    """Small read-only cache over raw and Stage 6/8 artifacts."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self._frames: dict[tuple[str, int], FrameArtifacts] = {}
        self._sample_shapes: dict[str, tuple[int, ...]] = {}
        self._tracks: pd.DataFrame | None = None

    def sample_shape(self, sample_id: str) -> tuple[int, ...]:
        if sample_id not in self._sample_shapes:
            self._sample_shapes[sample_id] = tuple(
                int(value) for value in open_sample(self.paths.sample_zarr(sample_id)).shape
            )
        return self._sample_shapes[sample_id]

    def load_frame(self, sample_id: str, frame: int) -> FrameArtifacts:
        key = (str(sample_id), int(frame))
        if key in self._frames:
            return self._frames[key]

        shape = self.sample_shape(sample_id)
        if not 0 <= int(frame) < shape[0]:
            raise IndexError(
                f"Frame {frame} is outside sample {sample_id} range 0..{shape[0] - 1}"
            )

        stem = f"t{int(frame):03d}"
        raw = np.asarray(load_timepoint(self.paths.sample_zarr(sample_id), int(frame)))
        preprocessed = np.load(
            self.paths.processed_series(sample_id, "preprocessing") / f"{stem}.npy",
            allow_pickle=False,
        )
        binary_mask = np.load(
            self.paths.processed_series(sample_id, "masking") / f"{stem}.npy",
            allow_pickle=False,
        ).astype(bool, copy=False)
        labels = np.load(
            self.paths.processed_series(sample_id, "segmentation") / f"{stem}.npy",
            allow_pickle=False,
        )
        cells = pd.read_csv(
            self.paths.processed_series(sample_id, "cells") / f"{stem}.csv"
        )

        shapes = {
            "raw": raw.shape,
            "preprocessed": preprocessed.shape,
            "binary_mask": binary_mask.shape,
            "instance_labels": labels.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(
                f"Sample {sample_id} frame {frame} artifact shapes differ: {shapes}"
            )

        result = FrameArtifacts(
            sample_id=str(sample_id),
            frame=int(frame),
            raw=np.asarray(raw),
            preprocessed=np.asarray(preprocessed),
            binary_mask=np.asarray(binary_mask, dtype=bool),
            instance_labels=np.asarray(labels),
            cells=cells,
        )
        self._frames[key] = result
        return result

    def load_tracks(self) -> pd.DataFrame:
        """Load Stage 8 tracks when present, otherwise Stage 7 tracks."""
        if self._tracks is not None:
            return self._tracks

        candidates = (
            self.paths.stage8_stitching / "tracks.csv",
            self.paths.stage7_tracking / "tracks.csv",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            self._tracks = pd.DataFrame()
            return self._tracks

        tracks = pd.read_csv(path)
        required = {"frame", "cell_id", "track_id"}
        missing = required.difference(tracks.columns)
        if missing:
            raise KeyError(f"{path} is missing track columns: {sorted(missing)}")
        tracks = tracks.copy()
        tracks["frame"] = tracks["frame"].astype(int)
        tracks["cell_id"] = tracks["cell_id"].astype(int)
        tracks["track_id"] = tracks["track_id"].astype(int)
        self._tracks = tracks
        return tracks

    def lookup_track_id(self, frame: int, cell_id: int) -> float:
        tracks = self.load_tracks()
        if tracks.empty:
            return float("nan")
        matches = tracks[
            (tracks["frame"] == int(frame))
            & (tracks["cell_id"] == int(cell_id))
        ]
        if "is_virtual_merge" in matches.columns:
            ordinary = matches[~matches["is_virtual_merge"].fillna(False).astype(bool)]
            if not ordinary.empty:
                matches = ordinary
        ids = matches["track_id"].dropna().astype(int).unique()
        return float(ids[0]) if len(ids) == 1 else float("nan")


def find_cell_row(cells: pd.DataFrame, cell_id: int) -> pd.Series | None:
    if "cell_id" not in cells.columns:
        return None
    matches = cells[cells["cell_id"].astype(int) == int(cell_id)]
    if len(matches) != 1:
        return None
    return matches.iloc[0]
