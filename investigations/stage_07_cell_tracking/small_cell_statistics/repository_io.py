"""Read-only loading of Stage 6 cells, labels, tracks, and event protections."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_tracks_path(paths, hints: Iterable[str | Path]) -> Path:
    for raw in hints:
        if raw is None or not str(raw).strip():
            continue
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
    candidates = (
        paths.stage11_reconciliation / "tracks.csv",
        paths.stage8_stitching / "tracks.csv",
        paths.stage7_tracking / "tracks.csv",
    )
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    if selected is None:
        raise FileNotFoundError(
            "No tracks.csv was found in the scene metadata or canonical Stage 11/8/7 output directories"
        )
    return selected


def _timepoint_paths(paths, sample_id: str, kind: str, suffix: str) -> list[Path]:
    directory = paths.processed_series(sample_id, kind)
    return sorted(directory.glob(f"t*.{suffix}"))


def load_sample_cells(paths, sample_id: str, *, validate_mask_volumes: bool) -> pd.DataFrame:
    cell_paths = _timepoint_paths(paths, sample_id, "cells", "csv")
    if not cell_paths:
        raise FileNotFoundError(f"No Stage 6 cell tables found for sample {sample_id}")
    label_paths = {
        int(path.stem[1:]): path
        for path in _timepoint_paths(paths, sample_id, "segmentation", "npy")
    }
    frames: list[pd.DataFrame] = []
    for cell_path in cell_paths:
        frame = int(cell_path.stem[1:])
        table = pd.read_csv(cell_path).copy()
        if "cell_id" not in table.columns:
            raise KeyError(f"{cell_path} is missing cell_id")
        table["frame"] = frame
        table["sample_id"] = str(sample_id)
        table["cell_id"] = pd.to_numeric(table["cell_id"], errors="raise").astype(int)
        if validate_mask_volumes:
            label_path = label_paths.get(frame)
            if label_path is None:
                raise FileNotFoundError(f"Missing segmentation labels for sample {sample_id} frame {frame}")
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
            counts = np.bincount(np.asarray(labels).ravel().astype(np.int64, copy=False))
            table["mask_volume_voxels"] = [
                int(counts[cell_id]) if 0 <= int(cell_id) < len(counts) else 0
                for cell_id in table["cell_id"]
            ]
            if "volume_voxels" in table.columns:
                table["stored_minus_mask_volume_voxels"] = (
                    pd.to_numeric(table["volume_voxels"], errors="coerce")
                    - table["mask_volume_voxels"]
                )
        frames.append(table)
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "mask_volume_voxels" in result.columns:
        result["analysis_volume_voxels"] = pd.to_numeric(
            result["mask_volume_voxels"], errors="coerce"
        )
    elif "volume_voxels" in result.columns:
        result["analysis_volume_voxels"] = pd.to_numeric(
            result["volume_voxels"], errors="coerce"
        )
    else:
        raise KeyError("Stage 6 cells contain neither volume_voxels nor a recomputable mask volume")
    return result.sort_values(["frame", "cell_id"], kind="mergesort").reset_index(drop=True)


def load_tracks(path: Path, sample_id: str) -> pd.DataFrame:
    tracks = pd.read_csv(path).copy()
    required = {"track_id", "frame"}
    missing = required.difference(tracks.columns)
    if missing:
        raise KeyError(f"{path} is missing track columns: {sorted(missing)}")
    if "cell_id" not in tracks.columns:
        if "cell" not in tracks.columns:
            raise KeyError(f"{path} contains neither cell_id nor cell")
        tracks["cell_id"] = tracks["cell"]
    for column in ("track_id", "frame", "cell_id"):
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    tracks = tracks.dropna(subset=["track_id", "frame", "cell_id"])
    tracks[["track_id", "frame", "cell_id"]] = tracks[["track_id", "frame", "cell_id"]].astype(int)
    if "is_virtual_merge" not in tracks.columns:
        tracks["is_virtual_merge"] = False
    tracks["is_virtual_merge"] = tracks["is_virtual_merge"].fillna(False).astype(bool)
    tracks["sample_id"] = str(sample_id)
    tracks["tracks_source"] = str(path)
    return tracks.sort_values(["track_id", "frame", "cell_id"], kind="mergesort").reset_index(drop=True)


def enrich_tracks_with_cells(tracks: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column for column in cells.columns
        if column not in {"sample_id", "frame", "cell_id"}
    ]
    right = cells[["sample_id", "frame", "cell_id", *feature_columns]].copy()
    if right.duplicated(["sample_id", "frame", "cell_id"]).any():
        raise ValueError("Stage 6 cell table contains duplicate sample/frame/cell_id keys")
    result = tracks.merge(
        right,
        on=["sample_id", "frame", "cell_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_cell"),
    )
    missing = result["analysis_volume_voxels"].isna() if "analysis_volume_voxels" in result else pd.Series(True, index=result.index)
    if missing.any() and "cell" in tracks.columns:
        # Fallback for repositories where tracks.cell_id was historically a positional index.
        by_position = cells.sort_values(["sample_id", "frame", "cell_id"]).copy()
        by_position["cell"] = by_position.groupby(["sample_id", "frame"]).cumcount()
        fallback = tracks.loc[missing, ["sample_id", "frame", "cell", "track_id", "cell_id"]].merge(
            by_position,
            on=["sample_id", "frame", "cell"],
            how="left",
            suffixes=("_track", ""),
        )
        for column in feature_columns:
            if column in fallback.columns:
                result.loc[missing, column] = fallback[column].to_numpy()
    return result


def _numeric_track_ids(frame: pd.DataFrame) -> set[int]:
    ids: set[int] = set()
    for column in frame.columns:
        if column == "track_id" or column.endswith("_track_id") or "track" in column.lower():
            values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(int)
            ids.update(values.tolist())
    return ids


def load_event_track_ids(paths, tracks_path: Path) -> set[int]:
    candidate_files = [
        tracks_path.parent / "segmentation_events.csv",
        tracks_path.parent / "division_events.csv",
        tracks_path.parent / "protected_tracks.csv",
        paths.stage8_stitching / "segmentation_events.csv",
        paths.stage10_lineage / "division_events.csv",
        paths.stage10_lineage / "protected_tracks.csv",
        paths.stage11_reconciliation / "segmentation_events.csv",
        paths.stage11_reconciliation / "division_events.csv",
        paths.stage11_reconciliation / "protected_tracks.csv",
    ]
    ids: set[int] = set()
    seen: set[Path] = set()
    for path in candidate_files:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        table = read_optional_csv(path)
        if not table.empty:
            ids.update(_numeric_track_ids(table))
    return ids
