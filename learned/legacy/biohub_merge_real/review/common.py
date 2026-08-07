"""Shared manifest and Napari review helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MergeRealConfig, MergeRealPaths
from ..extraction import candidate_crop_slices, crop_origin
from ..mining import load_candidates
from ..models import parse_track_ids
from ..observations import ObservationIndex
from ..repository_io import FullProcessedRepository


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def upsert_review(path: Path, record: dict, key: str = "candidate_id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = read_csv_or_empty(path)
    if not frame.empty and key in frame.columns:
        frame = frame.loc[frame[key].astype(str) != str(record[key])]
    frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def review_candidates(paths: MergeRealPaths, *, confirmed_only: bool = False) -> pd.DataFrame:
    candidates = load_candidates(paths)
    if not confirmed_only:
        return candidates.reset_index(drop=True)
    reviews = read_csv_or_empty(paths.filter_reviews_csv)
    if reviews.empty:
        return candidates.iloc[0:0].copy()
    allowed = reviews.loc[
        reviews["classification"].isin(
            ["confirmed_2_cell_merge", "confirmed_3plus_cell_merge"]
        ),
        ["candidate_id", "expected_cell_count"],
    ]
    result = candidates.merge(allowed, on="candidate_id", how="inner")
    return result.reset_index(drop=True)


def temporal_case_data(
    paths: MergeRealPaths,
    config: MergeRealConfig,
    candidate: pd.Series,
) -> dict[str, object]:
    repo = FullProcessedRepository(paths)
    sample = repo.sample(str(candidate.sample_id))
    event_frame = int(candidate.frame)
    cell_id = int(candidate.cell_id)
    slices = candidate_crop_slices(sample, event_frame, cell_id, config)
    start = max(0, event_frame - config.review_context_frames)
    end = min(sample.frame_count - 1, event_frame + config.review_context_frames)
    frames = list(range(start, end + 1))

    preprocessed = np.stack(
        [np.asarray(sample.preprocessed(frame, mmap=True)[slices]) for frame in frames],
        axis=0,
    )
    labels = np.stack(
        [np.asarray(sample.labels(frame, mmap=True)[slices]) for frame in frames],
        axis=0,
    )
    candidate_mask = np.zeros_like(labels, dtype=np.uint8)
    candidate_mask[frames.index(event_frame)] = (labels[frames.index(event_frame)] == cell_id).astype(np.uint8)

    raw_frames = []
    raw_available = True
    for frame in frames:
        raw = sample.raw_frame(frame)
        if raw is None:
            raw_available = False
            break
        raw_frames.append(np.asarray(raw[slices]))
    raw = np.stack(raw_frames, axis=0) if raw_available else None

    origin = np.asarray(crop_origin(slices), dtype=float)
    track_points: dict[int, np.ndarray] = {}
    predicted_points: dict[int, np.ndarray] = {}
    involved = parse_track_ids(candidate.get("involved_track_ids", ""))
    observation_index = ObservationIndex(sample, config)
    tracks = sample.tracks
    if "is_virtual_merge" in tracks.columns:
        virtual = tracks["is_virtual_merge"].fillna(False).map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
            if isinstance(value, str) else bool(value)
        )
        tracks = tracks.loc[~virtual]
    for track_id in involved:
        group = tracks.loc[
            (pd.to_numeric(tracks["track_id"], errors="coerce") == track_id)
            & (pd.to_numeric(tracks["frame"], errors="coerce").between(start, end))
        ]
        points = []
        for row in group.itertuples(index=False):
            local = np.asarray([row.z, row.y, row.x], dtype=float) - origin
            shape = np.asarray(preprocessed.shape[1:])
            if np.all(local >= 0) and np.all(local < shape):
                points.append([frames.index(int(row.frame)), *local.tolist()])
        if points:
            track_points[track_id] = np.asarray(points, dtype=float)
        predicted = observation_index.predict_zyx(track_id, event_frame)
        if predicted is not None:
            local_predicted = np.asarray(predicted, dtype=float) - origin
            shape = np.asarray(preprocessed.shape[1:])
            if np.all(local_predicted >= 0) and np.all(local_predicted < shape):
                predicted_points[track_id] = np.asarray([frames.index(event_frame), *local_predicted.tolist()], dtype=float)

    return {
        "sample": sample,
        "frames": frames,
        "event_index": frames.index(event_frame),
        "slices": slices,
        "preprocessed": preprocessed,
        "raw": raw,
        "labels": labels,
        "candidate_mask": candidate_mask,
        "track_points": track_points,
        "predicted_points": predicted_points,
    }


def candidate_summary(candidate: pd.Series) -> str:
    ratio = candidate.get("volume_sum_ratio", math.nan)
    ratio_text = f"{float(ratio):.3f}" if pd.notna(ratio) and math.isfinite(float(ratio)) else "n/a"
    return (
        f"{candidate.candidate_id}\n"
        f"sample={candidate.sample_id} frame={int(candidate.frame)} cell={int(candidate.cell_id)}\n"
        f"tier={candidate.tier} sources={candidate.sources}\n"
        f"tracks={candidate.get('involved_track_ids', '')}\n"
        f"volume_sum_ratio={ratio_text} EDT_peaks={int(candidate.get('edt_peak_count', 0))}"
    )


__all__ = [
    "candidate_summary",
    "read_csv_or_empty",
    "review_candidates",
    "temporal_case_data",
    "upsert_review",
]
