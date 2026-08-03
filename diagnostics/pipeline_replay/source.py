"""Validated scene metadata and authoritative production-source loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io import PipelinePaths, load_csv, load_json, load_npy, load_optional_csv
from src.io.arrays import load_timepoint, open_sample

from .models import ProductionFrame, TrackingContext, frame_artifact_path


def _triplet(metadata: dict[str, Any], name: str, cast: type) -> tuple:
    values = metadata.get(name)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"scene.json field {name!r} must contain three values")
    return tuple(cast(value) for value in values)


@dataclass(frozen=True)
class PipelineReplaySource:
    scene_path: Path
    sample_id: str
    frames: tuple[int, ...]
    crop_origin_zyx: tuple[int, int, int]
    crop_stop_zyx: tuple[int, int, int]
    crop_shape_zyx: tuple[int, int, int]
    voxel_size_zyx_um: tuple[float, float, float]
    selected_cells: dict[int, tuple[int, ...]]
    paths: PipelinePaths
    source_shape_tzyx: tuple[int, int, int, int]

    @classmethod
    def from_scene(
        cls, scene_path: str | Path, paths: PipelinePaths
    ) -> "PipelineReplaySource":
        root = Path(scene_path).expanduser().resolve()
        metadata = load_json(root / "scene.json")
        sample_id = str(metadata.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("scene.json is missing a non-empty sample_id")
        origin = _triplet(metadata, "crop_origin_zyx", int)
        stop = _triplet(metadata, "crop_stop_zyx", int)
        shape = _triplet(metadata, "crop_shape_zyx", int)
        if any(start < 0 or end <= start for start, end in zip(origin, stop)):
            raise ValueError("scene crop bounds are invalid")
        if tuple(end - start for start, end in zip(origin, stop)) != shape:
            raise ValueError("scene crop origin/stop do not agree with crop_shape_zyx")
        voxel_size = _triplet(metadata, "voxel_size_zyx", float)
        if any(value <= 0 for value in voxel_size):
            raise ValueError("voxel_size_zyx must contain positive values")

        frame_values = metadata.get("frame_numbers")
        if not isinstance(frame_values, list) or not frame_values:
            files = metadata.get("files", {})
            frame_file = files.get("frames", "frames.npy") if isinstance(files, dict) else "frames.npy"
            frame_values = load_npy(root / str(frame_file), expected_ndim=1).tolist()
        frames = tuple(int(value) for value in frame_values)
        if len(frames) != len(set(frames)):
            raise ValueError("scene frame numbers must be unique")

        selected_raw = metadata.get("selected_cells", {})
        if not isinstance(selected_raw, dict):
            raise ValueError("scene.json field 'selected_cells' must be an object")
        selected = {
            int(frame): tuple(int(cell_id) for cell_id in cell_ids)
            for frame, cell_ids in selected_raw.items()
        }
        if any(frame not in frames for frame in selected):
            raise ValueError("selected_cells contains a frame absent from frame_numbers")

        source_array = open_sample(paths.sample_zarr(sample_id))
        if source_array.ndim != 4:
            raise ValueError(f"source Zarr must be TZYX, found shape {source_array.shape}")
        if any(frame < 0 or frame >= source_array.shape[0] for frame in frames):
            raise IndexError("a scene frame is outside the source Zarr time axis")
        if any(end > limit for end, limit in zip(stop, source_array.shape[1:])):
            raise ValueError("scene crop extends outside the source Zarr spatial shape")

        return cls(
            root, sample_id, frames, origin, stop, shape, voxel_size, selected,
            paths, tuple(int(value) for value in source_array.shape),
        )

    @property
    def crop_slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.crop_origin_zyx, self.crop_stop_zyx))  # type: ignore[return-value]

    def global_to_local(self, coordinate_zyx) -> tuple[float, float, float]:
        values = tuple(float(value) for value in coordinate_zyx)
        if len(values) != 3:
            raise ValueError("coordinate must contain Z, Y, X")
        return tuple(value - origin for value, origin in zip(values, self.crop_origin_zyx))

    def local_to_global(self, coordinate_zyx) -> tuple[float, float, float]:
        values = tuple(float(value) for value in coordinate_zyx)
        if len(values) != 3:
            raise ValueError("coordinate must contain Z, Y, X")
        return tuple(value + origin for value, origin in zip(values, self.crop_origin_zyx))

    def load_frame(self, frame: int) -> ProductionFrame:
        """Load raw Zarr data and complete saved Stage 6 artifacts.

        Saved scene images are deliberately never consulted here.
        """

        frame = int(frame)
        if frame not in self.frames:
            raise ValueError(f"frame {frame} is not present in the selected scene")
        root = self.paths.processed_dataset(self.sample_id)
        raw = load_timepoint(self.paths.sample_zarr(self.sample_id), frame)
        preprocessed = load_npy(frame_artifact_path(root / "preprocessing", frame, "npy"), expected_ndim=3)
        binary_mask = load_npy(frame_artifact_path(root / "masking", frame, "npy"), expected_ndim=3)
        labels = load_npy(frame_artifact_path(root / "segmentation", frame, "npy"), expected_ndim=3)
        cells = load_csv(frame_artifact_path(root / "cells", frame, "csv"), required_columns=("cell_id", "centroid_z", "centroid_y", "centroid_x"))
        expected = tuple(raw.shape)
        if any(tuple(array.shape) != expected for array in (preprocessed, binary_mask, labels)):
            raise ValueError("Stage 6 arrays and source Zarr frame have different shapes")
        return ProductionFrame(frame, raw, preprocessed, binary_mask, labels, cells)

    def load_tracking_context(self, frame: int, cell_id: int) -> TrackingContext:
        """Load Stage 7/8 tables as read-only context, filtered where possible."""

        stage7_names = (
            "tracks", "association_events", "association_candidates",
            "boundary_predictions", "missing_predictions", "track_states",
        )
        stage8_names = (
            "tracks", "detections", "merge_onset_candidates", "merge_onsets",
            "segmentation_events", "merge_center_trajectories", "merge_split_links",
            "merge_track_repairs", "merge_trace_failures",
        )
        def load_all(root: Path, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
            return {name: load_optional_csv(root / f"{name}.csv") for name in names}
        stage7 = load_all(self.paths.stage7_tracking, stage7_names)
        selected_tracks: set[int] = set()
        tracks = stage7["tracks"]
        if not tracks.empty and {"frame", "cell_id", "track_id"}.issubset(tracks.columns):
            selected_tracks = set(
                tracks.loc[
                    (tracks["frame"] == frame) & (tracks["cell_id"] == cell_id),
                    "track_id",
                ].astype(int)
            )
        for name, table in stage7.items():
            if table.empty:
                continue
            frame_mask = pd.Series(True, index=table.index)
            frame_columns = [column for column in ("frame", "from_frame", "to_frame") if column in table.columns]
            if frame_columns:
                frame_mask = pd.concat(
                    [table[column].between(frame - 1, frame + 1) for column in frame_columns],
                    axis=1,
                ).any(axis=1)
            track_mask = pd.Series(True, index=table.index)
            if selected_tracks and "track_id" in table.columns:
                track_mask = table["track_id"].isin(selected_tracks)
            stage7[name] = table[frame_mask & track_mask].reset_index(drop=True)

        stage8 = load_all(self.paths.stage8_stitching, stage8_names)
        for name, table in stage8.items():
            if table.empty:
                continue
            track_masks = []
            if selected_tracks:
                for column in ("track_id", "source_track_id", "parent_track_id"):
                    if column in table.columns:
                        track_masks.append(table[column].isin(selected_tracks))
            if track_masks:
                keep = pd.concat(track_masks, axis=1).any(axis=1)
            elif "source_merged_cell_id" in table.columns:
                keep = table["source_merged_cell_id"] == cell_id
            else:
                cell_masks = [
                    table[column] == cell_id
                    for column in ("cell_id", "cell")
                    if column in table.columns
                ]
                if cell_masks:
                    keep = pd.concat(cell_masks, axis=1).any(axis=1)
                    if "frame" in table.columns:
                        keep &= table["frame"] == frame
                elif "frame" in table.columns:
                    keep = table["frame"].between(frame - 1, frame + 1)
                else:
                    keep = pd.Series(True, index=table.index)
            stage8[name] = table[keep].reset_index(drop=True)
        metadata: dict[str, object] = {}
        for root, key in ((self.paths.stage7_tracking, "stage7"), (self.paths.stage8_stitching, "stage8")):
            path = root / "metadata.json"
            if path.is_file():
                metadata[key] = load_json(path)
        return TrackingContext(
            stage7,
            stage8,
            metadata,
        )
