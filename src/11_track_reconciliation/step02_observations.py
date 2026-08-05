"""Input validation, feature enrichment, and endpoint summaries for Stage 11."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd

from .step01_config import ENDPOINT_CLASSIFICATION_COLUMNS, TrackReconciliationConfig


REQUIRED_TRACK_COLUMNS = (
    "track_id", "frame", "cell", "cell_id", "z", "y", "x", "volume",
)

OPTIONAL_DETECTION_FEATURES = (
    "volume_voxels", "intensity_sum", "intensity_mean", "intensity_std",
    "intensity_median", "intensity_iqr", "intensity_cv", "equivalent_radius",
    "axis_major", "axis_middle",
    "axis_minor", "elongation", "flatness", "anisotropy", "solidity",
    "compactness", "bbox_depth", "bbox_height", "bbox_width", "z_min",
    "y_min", "x_min", "z_max", "y_max", "x_max", "touches_boundary",
    "boundary_faces", "distance_to_boundary_um",
)

STAGE8_PROVENANCE_DEFAULTS: dict[str, object] = {
    "is_virtual_merge": False,
    "merge_event_id": pd.NA,
    "merge_role": pd.NA,
    "source_track_id": pd.NA,
    "source_merged_cell_id": pd.NA,
    "observed_merged_volume": np.nan,
}


def as_bool(value: object) -> bool:
    """Interpret persisted booleans without treating missing values as true."""

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
    """Copy, validate, and enrich corrected Stage 8 observations."""

    if not isinstance(tracks, pd.DataFrame):
        raise TypeError("tracks must be a pandas DataFrame")
    if not isinstance(time_frames, Sequence):
        raise TypeError("time_frames must be a sequence of pandas DataFrames")
    missing = [column for column in REQUIRED_TRACK_COLUMNS if column not in tracks]
    if missing:
        raise ValueError(f"Corrected Stage 8 tracks are missing required columns: {missing}")
    enriched = tracks.copy(deep=True)
    for index, detections in enumerate(time_frames):
        if not isinstance(detections, pd.DataFrame):
            raise TypeError(f"time_frames[{index}] must be a pandas DataFrame")

    if enriched.empty:
        for column in OPTIONAL_DETECTION_FEATURES:
            if column not in enriched:
                enriched[column] = pd.Series(dtype=object)
        for column, default in STAGE8_PROVENANCE_DEFAULTS.items():
            if column not in enriched:
                dtype = bool if column == "is_virtual_merge" else object
                enriched[column] = pd.Series(dtype=dtype)
        return enriched.reset_index(drop=True)

    for column in ("track_id", "frame", "cell", "cell_id"):
        _validate_integral_column(enriched, column)
    for column in ("z", "y", "x", "volume"):
        values = pd.to_numeric(enriched[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Track column {column!r} must contain finite numeric values")
    if (pd.to_numeric(enriched["volume"]).to_numpy(dtype=float) <= 0).any():
        raise ValueError("Track column 'volume' must contain positive values")

    for column in ("track_id", "frame", "cell", "cell_id"):
        enriched[column] = pd.to_numeric(enriched[column]).astype(int)
    if (enriched["track_id"] < 0).any():
        raise ValueError("Track IDs must be nonnegative")
    if (enriched["frame"] < 0).any():
        raise ValueError("Frame numbers must be nonnegative")
    if (enriched["cell"] < 0).any():
        raise ValueError("Cell indices must be nonnegative")
    if (enriched["cell_id"] <= 0).any():
        raise ValueError("Track column 'cell_id' must contain positive instance-label IDs")
    duplicate = enriched.duplicated(["track_id", "frame"], keep=False)
    if duplicate.any():
        examples = enriched.loc[duplicate, ["track_id", "frame"]].drop_duplicates()
        raise ValueError(
            "Corrected Stage 8 tracks contain duplicate (track_id, frame) observations: "
            f"{examples.head(10).to_dict('records')}"
        )

    attached = {
        column: [] for column in OPTIONAL_DETECTION_FEATURES if column not in enriched
    }
    for row in enriched.itertuples(index=False):
        frame_number = int(row.frame)
        cell_index = int(row.cell)
        if not 0 <= frame_number < len(time_frames):
            raise IndexError(
                f"Track observation frame {frame_number} is outside "
                f"0..{len(time_frames) - 1}"
            )
        detections = time_frames[frame_number]
        if not 0 <= cell_index < len(detections):
            raise IndexError(
                f"Track observation cell index {cell_index} is invalid for frame "
                f"{frame_number} with {len(detections)} detections"
            )
        detection = detections.iloc[cell_index]
        for column in attached:
            attached[column].append(
                detection[column] if column in detection.index else np.nan
            )
    for column, values in attached.items():
        enriched[column] = values
    for column, default in STAGE8_PROVENANCE_DEFAULTS.items():
        if column not in enriched:
            enriched[column] = default
    enriched["is_virtual_merge"] = enriched["is_virtual_merge"].map(as_bool)
    return enriched.sort_values(
        ["track_id", "frame", "cell"], kind="mergesort"
    ).reset_index(drop=True)


def observation_is_boundary(
    row: pd.Series,
    spatial_shape_zyx: Sequence[int] | None,
    config: TrackReconciliationConfig,
) -> bool:
    """Use explicit metadata first, then physical field-of-view fallbacks."""

    if "touches_boundary" in row.index and pd.notna(row["touches_boundary"]):
        return as_bool(row["touches_boundary"])
    if "distance_to_boundary_um" in row.index:
        try:
            distance = float(row["distance_to_boundary_um"])
        except (TypeError, ValueError):
            distance = math.nan
        if math.isfinite(distance):
            return distance <= config.boundary_margin_um
    if spatial_shape_zyx is None:
        return False
    shape = np.asarray(spatial_shape_zyx, dtype=float)
    if shape.shape != (3,) or not np.all(np.isfinite(shape)) or np.any(shape <= 0):
        raise ValueError("spatial_shape_zyx must contain three positive values")
    spacing = np.asarray(config.voxel_size_zyx_um, dtype=float)
    bbox = ("z_min", "y_min", "x_min", "z_max", "y_max", "x_max")
    if all(column in row.index and pd.notna(row[column]) for column in bbox):
        lower = row[["z_min", "y_min", "x_min"]].to_numpy(dtype=float) * spacing
        upper = (
            shape - row[["z_max", "y_max", "x_max"]].to_numpy(dtype=float)
        ) * spacing
        return float(np.min(np.concatenate([lower, upper]))) <= config.boundary_margin_um
    point = row[["z", "y", "x"]].to_numpy(dtype=float)
    lower = point * spacing
    upper = (shape - 1.0 - point) * spacing
    return float(np.min(np.concatenate([lower, upper]))) <= config.boundary_margin_um


def physical_position(row: pd.Series, config: TrackReconciliationConfig) -> np.ndarray:
    return row[["z", "y", "x"]].to_numpy(dtype=float) * np.asarray(
        config.voxel_size_zyx_um, dtype=float
    )


def build_endpoint_summary(
    observations: pd.DataFrame,
    spatial_shape_zyx: Sequence[int] | None,
    config: TrackReconciliationConfig,
) -> pd.DataFrame:
    """Summarize original Stage 8 segments using real endpoint observations."""

    if observations.empty:
        return pd.DataFrame(columns=ENDPOINT_CLASSIFICATION_COLUMNS)
    rows: list[dict[str, object]] = []
    for track_id, group in observations.groupby("track_id", sort=True):
        ordered = group.sort_values(["frame", "cell"], kind="mergesort")
        real = ordered.loc[~ordered["is_virtual_merge"].map(as_bool)]
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        if real.empty:
            first_real_frame = last_real_frame = pd.NA
            first_real_index = last_real_index = pd.NA
            first_boundary = last_boundary = False
        else:
            first_real = real.iloc[0]
            last_real = real.iloc[-1]
            first_real_frame = int(first_real["frame"])
            last_real_frame = int(last_real["frame"])
            first_real_index = int(real.index[0])
            last_real_index = int(real.index[-1])
            first_boundary = observation_is_boundary(
                first_real, spatial_shape_zyx, config
            )
            last_boundary = observation_is_boundary(
                last_real, spatial_shape_zyx, config
            )
        rows.append({
            "track_id": int(track_id),
            "first_frame": int(first["frame"]),
            "last_frame": int(last["frame"]),
            "observation_count": int(len(ordered)),
            "real_observation_count": int(len(real)),
            "first_real_frame": first_real_frame,
            "last_real_frame": last_real_frame,
            "first_observation_index": int(ordered.index[0]),
            "last_observation_index": int(ordered.index[-1]),
            "first_real_observation_index": first_real_index,
            "last_real_observation_index": last_real_index,
            "first_is_boundary": bool(first_boundary),
            "last_is_boundary": bool(last_boundary),
            "first_is_virtual": as_bool(first["is_virtual_merge"]),
            "last_is_virtual": as_bool(last["is_virtual_merge"]),
            "source_eligible": False,
            "target_eligible": False,
            "source_exclusion_reason": "",
            "target_exclusion_reason": "",
        })
    return pd.DataFrame(rows, columns=ENDPOINT_CLASSIFICATION_COLUMNS)


def real_track_observations(observations: pd.DataFrame, track_id: int) -> pd.DataFrame:
    group = observations.loc[observations["track_id"] == int(track_id)]
    return group.loc[~group["is_virtual_merge"].map(as_bool)].sort_values(
        "frame", kind="mergesort"
    )
