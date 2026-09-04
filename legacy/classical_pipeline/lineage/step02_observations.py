"""Canonical observation preparation and endpoint helpers for Stage 10."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd

from .step01_config import CellLineageConfig, TRACK_SUMMARY_COLUMNS


REQUIRED_TRACK_COLUMNS = (
    "track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume",
)

OPTIONAL_DETECTION_FEATURES = (
    "volume_voxels", "intensity_sum", "intensity_mean", "intensity_std",
    "equivalent_radius", "axis_major", "axis_middle", "axis_minor",
    "elongation", "flatness", "anisotropy", "solidity", "compactness",
    "bbox_depth", "bbox_height", "bbox_width", "z_min", "y_min", "x_min",
    "z_max", "y_max", "x_max", "touches_boundary", "boundary_faces",
    "distance_to_boundary_um",
)

STAGE8_PROVENANCE_COLUMNS = (
    "is_virtual_merge", "merge_event_id", "merge_role", "source_track_id",
    "source_merged_cell_id", "observed_merged_volume",
)


def as_bool(value: object) -> bool:
    """Interpret common persisted boolean representations deterministically."""

    if value is None or value is pd.NA:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _validate_integral_column(frame: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"Track column {column!r} must contain finite integers")


def prepare_observations(
    tracks: pd.DataFrame,
    time_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Validate corrected Stage 8 rows and attach Stage 6 detection features."""

    if not isinstance(tracks, pd.DataFrame):
        raise TypeError("tracks must be a pandas DataFrame")
    missing = [column for column in REQUIRED_TRACK_COLUMNS if column not in tracks]
    if missing:
        raise ValueError(f"Corrected Stage 8 tracks are missing required columns: {missing}")

    enriched = tracks.copy()
    if enriched.empty:
        for column in OPTIONAL_DETECTION_FEATURES:
            if column not in enriched:
                enriched[column] = pd.Series(dtype=float)
        if "is_virtual_merge" not in enriched:
            enriched["is_virtual_merge"] = pd.Series(dtype=bool)
        for column in STAGE8_PROVENANCE_COLUMNS[1:]:
            if column not in enriched:
                enriched[column] = pd.Series(dtype=object)
        return enriched.reset_index(drop=True)

    for column in ("track_id", "frame", "cell", "cell_id"):
        _validate_integral_column(enriched, column)
    for column in ("z", "y", "x", "volume"):
        values = pd.to_numeric(enriched[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Track column {column!r} must contain finite numeric values")
    if np.any(pd.to_numeric(enriched["volume"]).to_numpy(dtype=float) <= 0):
        raise ValueError("Track column 'volume' must contain positive values")

    enriched["track_id"] = pd.to_numeric(enriched["track_id"]).astype(int)
    enriched["frame"] = pd.to_numeric(enriched["frame"]).astype(int)
    enriched["cell"] = pd.to_numeric(enriched["cell"]).astype(int)
    enriched["cell_id"] = pd.to_numeric(enriched["cell_id"]).astype(int)
    if (enriched["cell_id"] <= 0).any():
        raise ValueError("Track column 'cell_id' must contain positive instance-label IDs")

    duplicate = enriched.duplicated(["track_id", "frame"], keep=False)
    if duplicate.any():
        examples = enriched.loc[duplicate, ["track_id", "frame"]].drop_duplicates()
        raise ValueError(
            "Corrected Stage 8 tracks contain duplicate (track_id, frame) observations: "
            f"{examples.head(10).to_dict('records')}"
        )

    attached: dict[str, list[object]] = {
        column: [] for column in OPTIONAL_DETECTION_FEATURES if column not in enriched
    }
    for row in enriched.itertuples(index=False):
        frame = int(row.frame)
        cell = int(row.cell)
        if not 0 <= frame < len(time_frames):
            raise IndexError(
                f"Track observation frame {frame} is outside 0..{len(time_frames) - 1}"
            )
        detections = time_frames[frame]
        if not isinstance(detections, pd.DataFrame):
            raise TypeError(f"time_frames[{frame}] must be a pandas DataFrame")
        if not 0 <= cell < len(detections):
            raise IndexError(
                f"Track observation cell index {cell} is invalid for frame {frame} "
                f"with {len(detections)} detections"
            )
        detection = detections.iloc[cell]
        for column in attached:
            attached[column].append(detection[column] if column in detection.index else np.nan)

    for column, values in attached.items():
        enriched[column] = values

    if "is_virtual_merge" not in enriched:
        enriched["is_virtual_merge"] = False
    else:
        enriched["is_virtual_merge"] = enriched["is_virtual_merge"].map(as_bool)
    neutral_defaults: dict[str, object] = {
        "merge_event_id": pd.NA,
        "merge_role": pd.NA,
        "source_track_id": pd.NA,
        "source_merged_cell_id": pd.NA,
        "observed_merged_volume": np.nan,
    }
    for column, default in neutral_defaults.items():
        if column not in enriched:
            enriched[column] = default

    return enriched.sort_values(["track_id", "frame"], kind="mergesort").reset_index(drop=True)


def build_track_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Derive deterministic starts and endings from corrected Stage 8 tracks."""

    if observations.empty:
        return pd.DataFrame(columns=TRACK_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for track_id, group in observations.groupby("track_id", sort=True):
        ordered = group.sort_values("frame", kind="mergesort")
        first_index = int(ordered.index[0])
        last_index = int(ordered.index[-1])
        rows.append({
            "track_id": int(track_id),
            "first_frame": int(ordered.iloc[0]["frame"]),
            "last_frame": int(ordered.iloc[-1]["frame"]),
            "observation_count": int(len(ordered)),
            "first_observation_index": first_index,
            "last_observation_index": last_index,
            "first_is_virtual": as_bool(ordered.iloc[0]["is_virtual_merge"]),
            "last_is_virtual": as_bool(ordered.iloc[-1]["is_virtual_merge"]),
        })
    return pd.DataFrame(rows, columns=TRACK_SUMMARY_COLUMNS)


def physical_distance(
    first_zyx: np.ndarray | Sequence[float],
    second_zyx: np.ndarray | Sequence[float],
    voxel_size_zyx_um: Sequence[float],
) -> float:
    """Return physical Euclidean distance between voxel-order ZYX points."""

    delta = np.asarray(first_zyx, dtype=float) - np.asarray(second_zyx, dtype=float)
    return float(np.linalg.norm(delta * np.asarray(voxel_size_zyx_um, dtype=float)))


def observation_is_boundary(
    row: pd.Series,
    spatial_shape_zyx: Sequence[int] | None,
    config: CellLineageConfig,
) -> bool:
    """Classify an endpoint using metadata first and physical fallbacks second."""

    if "touches_boundary" in row.index and pd.notna(row["touches_boundary"]):
        return as_bool(row["touches_boundary"])
    if "distance_to_boundary_um" in row.index:
        try:
            distance = float(row["distance_to_boundary_um"])
        except (TypeError, ValueError):
            distance = math.nan
        if math.isfinite(distance):
            return distance <= config.boundary_margin_um
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    shape = None if spatial_shape_zyx is None else np.asarray(spatial_shape_zyx, dtype=float)
    bbox_columns = ("z_min", "y_min", "x_min", "z_max", "y_max", "x_max")
    if shape is not None and all(column in row.index and pd.notna(row[column]) for column in bbox_columns):
        lower = row[["z_min", "y_min", "x_min"]].to_numpy(dtype=float) * spacing
        upper = (shape - row[["z_max", "y_max", "x_max"]].to_numpy(dtype=float)) * spacing
        return float(np.min(np.concatenate([lower, upper]))) <= config.boundary_margin_um
    if shape is not None:
        point = row[["z", "y", "x"]].to_numpy(dtype=float)
        lower = point * spacing
        upper = (shape - 1 - point) * spacing
        return float(np.min(np.concatenate([lower, upper]))) <= config.boundary_margin_um
    return False


def add_endpoint_classification(
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    spatial_shape_zyx: Sequence[int] | None,
    config: CellLineageConfig,
) -> pd.DataFrame:
    """Attach boundary flags to summary rows without mutating observations."""

    result = summary.copy()
    if result.empty:
        result["first_is_boundary"] = pd.Series(dtype=bool)
        result["last_is_boundary"] = pd.Series(dtype=bool)
        return result
    first_flags = []
    last_flags = []
    for row in result.itertuples(index=False):
        first = observations.loc[int(row.first_observation_index)]
        last = observations.loc[int(row.last_observation_index)]
        first_flags.append(observation_is_boundary(first, spatial_shape_zyx, config))
        last_flags.append(observation_is_boundary(last, spatial_shape_zyx, config))
    result["first_is_boundary"] = first_flags
    result["last_is_boundary"] = last_flags
    return result
