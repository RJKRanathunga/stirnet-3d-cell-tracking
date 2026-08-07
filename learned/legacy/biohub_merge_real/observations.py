"""Normalized Stage 11 observation/index helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MergeRealConfig
from .repository_io import SampleArtifacts


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
        if isinstance(value, str)
        else bool(value)
    )


@dataclass
class ObservationIndex:
    sample: SampleArtifacts
    config: MergeRealConfig

    def __post_init__(self) -> None:
        tracks = self.sample.tracks.copy()
        required = {"track_id", "frame", "cell_id", "z", "y", "x", "volume"}
        missing = sorted(required - set(tracks.columns))
        if missing:
            raise ValueError(f"{self.sample.sample_id}: tracks.csv missing {missing}")
        for column in ("track_id", "frame", "cell_id"):
            tracks[column] = pd.to_numeric(tracks[column], errors="raise").astype(int)
        for column in ("z", "y", "x", "volume"):
            tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
        if "is_virtual_merge" in tracks:
            virtual = _as_bool_series(tracks["is_virtual_merge"])
            tracks = tracks.loc[~virtual].copy()
        self.tracks = tracks.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
        self._by_track = {
            int(track_id): group.sort_values("frame", kind="stable").reset_index(drop=True)
            for track_id, group in self.tracks.groupby("track_id", sort=False)
        }
        self._frame_track = {
            (int(row.frame), int(row.track_id)): row
            for row in self.tracks.itertuples(index=False)
        }
        self._frame_cell_tracks: dict[tuple[int, int], list[int]] = {}
        for row in self.tracks.itertuples(index=False):
            self._frame_cell_tracks.setdefault((int(row.frame), int(row.cell_id)), []).append(int(row.track_id))

        self._original_to_canonical = {track_id: track_id for track_id in self._by_track}
        remap = self.sample.track_id_remap
        if not remap.empty and {"original_segment_track_id", "canonical_track_id"}.issubset(remap.columns):
            for row in remap.to_dict("records"):
                canonical = _safe_int(row.get("canonical_track_id"))
                if canonical is None:
                    continue
                for column in ("original_segment_track_id", "predecessor_track_id"):
                    original = _safe_int(row.get(column))
                    if original is not None:
                        self._original_to_canonical[original] = canonical

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_track))

    def track(self, track_id: int) -> pd.DataFrame:
        return self._by_track.get(int(track_id), pd.DataFrame(columns=self.tracks.columns))

    def observation(self, track_id: int, frame: int):
        return self._frame_track.get((int(frame), int(track_id)))

    def track_ids_for_cell(self, frame: int, cell_id: int) -> tuple[int, ...]:
        return tuple(sorted(set(self._frame_cell_tracks.get((int(frame), int(cell_id)), []))))

    def tracks_in_frame(self, frame: int) -> pd.DataFrame:
        return self.tracks.loc[self.tracks["frame"] == int(frame)].copy()

    def present(self, track_id: int, frame: int) -> bool:
        return (int(frame), int(track_id)) in self._frame_track

    def first_frame(self, track_id: int) -> int | None:
        group = self.track(track_id)
        return int(group["frame"].min()) if not group.empty else None

    def last_frame(self, track_id: int) -> int | None:
        group = self.track(track_id)
        return int(group["frame"].max()) if not group.empty else None

    def history_before(self, track_id: int, target_frame: int, count: int | None = None) -> pd.DataFrame:
        group = self.track(track_id)
        history = group.loc[group["frame"] < int(target_frame)]
        return history.tail(int(count or self.config.motion_history_frames)).copy()

    def reference_volume(self, track_id: int, before_frame: int) -> float:
        history = self.history_before(
            track_id,
            before_frame,
            self.config.reference_history_frames,
        )
        if history.empty:
            return float("nan")
        values = pd.to_numeric(history["volume"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        return float(np.median(values)) if values.size else float("nan")

    def predict_zyx(self, track_id: int, target_frame: int) -> np.ndarray | None:
        history = self.history_before(track_id, target_frame, self.config.motion_history_frames)
        if history.empty:
            return None
        spacing = np.asarray(self.config.voxel_size_zyx_um, dtype=float)
        frames = history["frame"].to_numpy(dtype=float)
        positions = history[["z", "y", "x"]].to_numpy(dtype=float) * spacing[None, :]
        last_frame = int(frames[-1])
        last_pos = positions[-1]
        if len(history) < 2:
            predicted_um = last_pos
        else:
            dt = np.diff(frames)
            valid = dt > 0
            if valid.any():
                velocities = np.diff(positions, axis=0)[valid] / dt[valid, None]
                velocity = np.median(velocities, axis=0)
            else:
                velocity = np.zeros(3, dtype=float)
            predicted_um = last_pos + velocity * (int(target_frame) - last_frame)
        return predicted_um / spacing


    def canonical_track_id(self, track_id: int) -> int:
        return int(self._original_to_canonical.get(int(track_id), int(track_id)))

    def stage11_unresolved_support(self, track_id: int, source_end_frame: int) -> bool:
        table = self.sample.unresolved_endings
        if table.empty or not {"source_track_id", "source_end_frame"}.issubset(table.columns):
            return False
        for row in table.to_dict("records"):
            original = _safe_int(row.get("source_track_id"))
            frame = _safe_int(row.get("source_end_frame"))
            if original is None or frame is None:
                continue
            if self.canonical_track_id(original) == int(track_id) and frame == int(source_end_frame):
                return True
        return False

    def stage11_remap_support(self, track_id: int, frame: int) -> bool:
        table = self.sample.track_id_remap
        if table.empty or "canonical_track_id" not in table.columns:
            return False
        for row in table.to_dict("records"):
            canonical = _safe_int(row.get("canonical_track_id"))
            if canonical != int(track_id):
                continue
            from_frame = _safe_int(row.get("from_frame"))
            to_frame = _safe_int(row.get("to_frame"))
            if from_frame is None or to_frame is None:
                return True
            if min(from_frame, to_frame) - 1 <= int(frame) <= max(from_frame, to_frame) + 1:
                return True
        return False

    def endpoint_is_boundary(self, track_id: int, frame: int) -> bool:
        table = self.sample.endpoint_classifications
        if table.empty or "track_id" not in table.columns:
            return False
        for row in table.to_dict("records"):
            original = _safe_int(row.get("track_id"))
            if original is None or self.canonical_track_id(original) != int(track_id):
                continue
            last_real = _safe_int(row.get("last_real_frame"))
            if last_real != int(frame):
                continue
            reason = str(row.get("source_exclusion_reason", ""))
            boundary = row.get("last_is_boundary", False)
            if isinstance(boundary, str):
                boundary = boundary.strip().lower() in {"1", "true", "yes", "y"}
            return bool(boundary) or reason == "excluded_boundary_exit"
        return False

    def division_contaminated(self, track_ids: tuple[int, ...], frame: int) -> bool:
        events = self.sample.division_events
        if events.empty:
            return False
        ids = set(int(v) for v in track_ids)
        guard = int(self.config.division_guard_frames)
        for row in events.to_dict("records"):
            decision = str(row.get("decision", "")).strip().lower()
            if decision not in {"confirmed", "probable"}:
                continue
            involved = {
                _safe_int(row.get("parent_track_id")),
                _safe_int(row.get("child_track_a")),
                _safe_int(row.get("child_track_b")),
            }
            involved.discard(None)
            if not ids.intersection(involved):
                continue
            relevant_frames = [
                _safe_int(row.get("parent_end_frame")),
                _safe_int(row.get("child_birth_frame")),
            ]
            if any(v is not None and abs(int(frame) - v) <= guard for v in relevant_frames):
                return True
        return False


def _safe_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        number = float(value)
        if not np.isfinite(number):
            return None
        return int(number)
    except (TypeError, ValueError):
        return None


__all__ = ["ObservationIndex"]
