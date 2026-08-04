"""Repository-aware, read-only loading for the phenotype investigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import load_timepoint, open_sample


def _read_optional_csv(path: Path) -> pd.DataFrame:
    """Return an empty table when an optional CSV is missing or has no content."""
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()



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
    """Cache raw, Stage 6, Stage 8, and optional Stage 10 artifacts."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self._frames: dict[tuple[str, int], FrameArtifacts] = {}
        self._sample_shapes: dict[str, tuple[int, ...]] = {}
        self._tracks: pd.DataFrame | None = None
        self._track_summary: pd.DataFrame | None = None
        self._protected_tracks: pd.DataFrame | None = None
        self._segmentation_events: pd.DataFrame | None = None

    def sample_shape(self, sample_id: str) -> tuple[int, ...]:
        if sample_id not in self._sample_shapes:
            self._sample_shapes[sample_id] = tuple(
                int(v) for v in open_sample(self.paths.sample_zarr(sample_id)).shape
            )
        return self._sample_shapes[sample_id]

    def load_frame(self, sample_id: str, frame: int) -> FrameArtifacts:
        key = (str(sample_id), int(frame))
        if key in self._frames:
            return self._frames[key]
        shape = self.sample_shape(sample_id)
        if not 0 <= int(frame) < shape[0]:
            raise IndexError(f"Frame {frame} is outside sample {sample_id} range 0..{shape[0] - 1}")
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
        cells = pd.read_csv(self.paths.processed_series(sample_id, "cells") / f"{stem}.csv")
        shapes = {raw.shape, preprocessed.shape, binary_mask.shape, labels.shape}
        if len(shapes) != 1:
            raise ValueError(f"Sample {sample_id} frame {frame} artifact shapes differ")
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
        tracks = pd.read_csv(path).copy()
        if "cell_id" not in tracks.columns and "cell" in tracks.columns:
            tracks["cell_id"] = tracks["cell"]
        required = {"frame", "cell_id", "track_id"}
        missing = required.difference(tracks.columns)
        if missing:
            raise KeyError(f"{path} is missing track columns: {sorted(missing)}")
        for name in required:
            tracks[name] = pd.to_numeric(tracks[name], errors="coerce")
        tracks = tracks.dropna(subset=list(required))
        tracks[["frame", "cell_id", "track_id"]] = tracks[["frame", "cell_id", "track_id"]].astype(int)
        if "is_virtual_merge" not in tracks.columns:
            tracks["is_virtual_merge"] = False
        tracks["is_virtual_merge"] = tracks["is_virtual_merge"].fillna(False).astype(bool)
        self._tracks = tracks
        return tracks

    def track_summary(self) -> pd.DataFrame:
        if self._track_summary is not None:
            return self._track_summary
        tracks = self.load_tracks()
        if tracks.empty:
            self._track_summary = pd.DataFrame(columns=["track_id", "first_frame", "last_frame", "observation_count"])
        else:
            real = tracks.loc[~tracks["is_virtual_merge"]].copy()
            self._track_summary = (
                real.groupby("track_id", as_index=False)
                .agg(first_frame=("frame", "min"), last_frame=("frame", "max"), observation_count=("frame", "count"))
            )
        return self._track_summary

    def track_metadata(self, frame: int, cell_id: int) -> dict[str, object]:
        tracks = self.load_tracks()
        if tracks.empty:
            return {
                "track_id": np.nan,
                "is_virtual_merge": False,
                "track_starts_here": False,
                "track_ends_here": False,
                "track_observation_count": np.nan,
            }
        matches = tracks.loc[(tracks["frame"] == int(frame)) & (tracks["cell_id"] == int(cell_id))]
        if matches.empty:
            return {
                "track_id": np.nan,
                "is_virtual_merge": False,
                "track_starts_here": False,
                "track_ends_here": False,
                "track_observation_count": np.nan,
            }
        real = matches.loc[~matches["is_virtual_merge"]]
        row = (real if not real.empty else matches).iloc[0]
        track_id = int(row["track_id"])
        summary = self.track_summary()
        selected = summary.loc[summary["track_id"] == track_id]
        if selected.empty:
            first = last = int(frame)
            count = len(matches)
        else:
            item = selected.iloc[0]
            first, last, count = int(item["first_frame"]), int(item["last_frame"]), int(item["observation_count"])
        return {
            "track_id": track_id,
            "is_virtual_merge": bool(row.get("is_virtual_merge", False)),
            "track_starts_here": int(frame) == first,
            "track_ends_here": int(frame) == last,
            "track_observation_count": count,
        }

    def protected_tracks(self) -> pd.DataFrame:
        if self._protected_tracks is None:
            path = self.paths.stage10_lineage / "protected_tracks.csv"
            self._protected_tracks = _read_optional_csv(path)
        return self._protected_tracks

    def known_lineage_role(self, track_id: object) -> str:
        if pd.isna(track_id):
            return ""
        table = self.protected_tracks()
        if table.empty or "track_id" not in table.columns:
            return ""
        rows = table.loc[pd.to_numeric(table["track_id"], errors="coerce") == int(track_id)]
        if rows.empty:
            return ""
        if "role" in rows.columns:
            return "|".join(sorted(set(rows["role"].dropna().astype(str))))
        return "protected"

    def segmentation_events(self) -> pd.DataFrame:
        if self._segmentation_events is None:
            path = self.paths.stage8_stitching / "segmentation_events.csv"
            self._segmentation_events = _read_optional_csv(path)
        return self._segmentation_events

    def overlaps_segmentation_event(self, frame: int, cell_id: int, track_id: object) -> bool:
        events = self.segmentation_events()
        if events.empty:
            return False
        frame_columns = [name for name in events.columns if "frame" in name.lower()]
        cell_columns = [name for name in events.columns if "cell" in name.lower()]
        track_columns = [name for name in events.columns if "track" in name.lower()]
        frame_match = pd.Series(False, index=events.index)
        for name in frame_columns:
            frame_match |= pd.to_numeric(events[name], errors="coerce").eq(int(frame))
        identity_match = pd.Series(False, index=events.index)
        for name in cell_columns:
            identity_match |= pd.to_numeric(events[name], errors="coerce").eq(int(cell_id))
        if not pd.isna(track_id):
            for name in track_columns:
                identity_match |= pd.to_numeric(events[name], errors="coerce").eq(int(track_id))
        return bool((frame_match & identity_match).any())


def saved_cell_row(cells: pd.DataFrame, cell_id: int) -> pd.Series | None:
    if "cell_id" not in cells.columns:
        return None
    rows = cells.loc[pd.to_numeric(cells["cell_id"], errors="coerce") == int(cell_id)]
    return rows.iloc[0] if len(rows) == 1 else None
