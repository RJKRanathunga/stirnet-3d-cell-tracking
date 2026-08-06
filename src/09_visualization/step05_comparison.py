"""Before/after snapshots for Stage 9 endpoint diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from src.io import load_csv, load_json, save_csv, save_json

from .step02_endpoints import EndpointTrackGroups


_TRACK_COLUMNS = ("track_id", "frame", "z", "y", "x", "cell_id")
_ENDPOINT_COLUMNS = ("frame", "cell_id", "track_id")
_SNAPSHOT_FILES = {
    "ended_tracks": "ended_tracks.csv",
    "ended_endpoints": "ended_endpoints.csv",
    "new_tracks": "new_tracks.csv",
    "new_endpoints": "new_endpoints.csv",
}


@dataclass(frozen=True)
class Stage9Snapshot:
    ended_tracks: pd.DataFrame
    ended_endpoints: pd.DataFrame
    new_tracks: pd.DataFrame
    new_endpoints: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Stage9Comparison:
    previous: Stage9Snapshot
    new: Stage9Snapshot

    removed_ended_tracks: pd.DataFrame
    added_ended_tracks: pd.DataFrame
    removed_new_tracks: pd.DataFrame
    added_new_tracks: pd.DataFrame

    removed_ended_endpoints: pd.DataFrame
    added_ended_endpoints: pd.DataFrame
    removed_new_endpoints: pd.DataFrame
    added_new_endpoints: pd.DataFrame

    summary: dict[str, int]


def _sorted_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy().reset_index(drop=True)
    columns = [name for name in ("track_id", "frame", "cell_id") if name in frame]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True)


def _sorted_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy().reset_index(drop=True)
    columns = [name for name in ("frame", "cell_id", "track_id") if name in frame]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True)


def _non_boundary_endpoints(frame: pd.DataFrame, event_name: str) -> pd.DataFrame:
    if "is_boundary_endpoint" not in frame.columns:
        raise ValueError(
            f"{event_name} endpoint table is missing 'is_boundary_endpoint'"
        )
    return _sorted_endpoints(frame.loc[~frame["is_boundary_endpoint"].astype(bool)])


def prepare_stage9_snapshot(
    endpoint_groups: EndpointTrackGroups,
    *,
    sample_id: str,
    boundary_margin_um: float,
    voxel_size_zyx: Sequence[float],
    spatial_shape_zyx: Sequence[int],
    stage8_metadata: Mapping[str, Any] | None = None,
) -> Stage9Snapshot:
    """Prepare the four persisted Stage 9 diagnostic tables and metadata."""

    ended_tracks = _sorted_tracks(endpoint_groups.ended_failure_tracks)
    new_tracks = _sorted_tracks(endpoint_groups.new_failure_tracks)
    ended_endpoints = _non_boundary_endpoints(
        endpoint_groups.ended_track_endpoints, "Ended"
    )
    new_endpoints = _non_boundary_endpoints(
        endpoint_groups.new_track_endpoints, "New"
    )

    metadata: dict[str, Any] = {
        "sample_id": str(sample_id),
        "boundary_margin_um": float(boundary_margin_um),
        "voxel_size_zyx": [float(value) for value in voxel_size_zyx],
        "spatial_shape_zyx": [int(value) for value in spatial_shape_zyx],
        "source_stage": "stage_08_track_stitching",
        "stage8_metadata": dict(stage8_metadata or {}),
        "ended_track_count": int(ended_tracks["track_id"].nunique()),
        "new_track_count": int(new_tracks["track_id"].nunique()),
        "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return Stage9Snapshot(
        ended_tracks=ended_tracks,
        ended_endpoints=ended_endpoints,
        new_tracks=new_tracks,
        new_endpoints=new_endpoints,
        metadata=metadata,
    )


def _is_non_empty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def save_stage9_snapshot(
    snapshot: Stage9Snapshot,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a snapshot using a directory swap so stale files cannot survive."""

    target = Path(directory)
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"Snapshot path is not a directory: {target}")
    if _is_non_empty_directory(target) and not overwrite:
        raise FileExistsError(
            f"Snapshot directory already exists and is non-empty: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.writing-", dir=target.parent)
    )
    backup: Path | None = None
    try:
        for attribute, filename in _SNAPSHOT_FILES.items():
            save_csv(getattr(snapshot, attribute), temporary / filename)
        save_json(snapshot.metadata, temporary / "metadata.json")

        if target.exists():
            if _is_non_empty_directory(target) and not overwrite:
                raise FileExistsError(
                    f"Snapshot directory already exists and is non-empty: {target}"
                )
            backup = target.parent / f".{target.name}.replaced-{uuid4().hex}"
            target.replace(backup)
        try:
            temporary.replace(target)
        except Exception:
            if backup is not None and not target.exists():
                backup.replace(target)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and backup.exists():
            if not target.exists():
                backup.replace(target)
            else:
                shutil.rmtree(backup)
    return target


def load_stage9_snapshot(directory: str | Path) -> Stage9Snapshot:
    """Load one Stage 9 comparison snapshot from disk."""

    root = Path(directory)
    return Stage9Snapshot(
        ended_tracks=load_csv(
            root / _SNAPSHOT_FILES["ended_tracks"], required_columns=_TRACK_COLUMNS
        ),
        ended_endpoints=load_csv(
            root / _SNAPSHOT_FILES["ended_endpoints"],
            required_columns=_ENDPOINT_COLUMNS,
        ),
        new_tracks=load_csv(
            root / _SNAPSHOT_FILES["new_tracks"], required_columns=_TRACK_COLUMNS
        ),
        new_endpoints=load_csv(
            root / _SNAPSHOT_FILES["new_endpoints"],
            required_columns=_ENDPOINT_COLUMNS,
        ),
        metadata=load_json(root / "metadata.json"),
    )


def _metadata_value(snapshot: Stage9Snapshot, label: str, key: str) -> Any:
    if key not in snapshot.metadata:
        raise ValueError(f"{label} snapshot metadata is missing '{key}'")
    return snapshot.metadata[key]


def _validate_compatible_metadata(
    previous: Stage9Snapshot, new: Stage9Snapshot
) -> None:
    scalar_fields = (
        ("sample_id", "sample ID"),
        ("boundary_margin_um", "boundary margin"),
    )
    for key, description in scalar_fields:
        previous_value = _metadata_value(previous, "Previous", key)
        new_value = _metadata_value(new, "New", key)
        if previous_value != new_value:
            raise ValueError(
                f"Stage 9 snapshot {description} values do not match: "
                f"previous={previous_value!r}, new={new_value!r}"
            )

    sequence_fields = (
        ("voxel_size_zyx", "voxel sizes"),
        ("spatial_shape_zyx", "spatial shapes"),
    )
    for key, description in sequence_fields:
        previous_value = _metadata_value(previous, "Previous", key)
        new_value = _metadata_value(new, "New", key)
        try:
            matches = tuple(previous_value) == tuple(new_value)
        except TypeError as exc:
            raise ValueError(
                f"Stage 9 snapshot metadata '{key}' must be a sequence"
            ) from exc
        if not matches:
            raise ValueError(
                f"Stage 9 snapshot {description} do not match: "
                f"previous={previous_value!r}, new={new_value!r}"
            )


def _event_keys(
    endpoints: pd.DataFrame,
    *,
    snapshot_label: str,
    event_label: str,
) -> set[tuple[Any, Any]]:
    missing = [name for name in _ENDPOINT_COLUMNS if name not in endpoints.columns]
    if missing:
        raise ValueError(
            f"{snapshot_label} {event_label} endpoint table is missing columns {missing}"
        )

    cell_ids = pd.to_numeric(endpoints["cell_id"], errors="coerce")
    if cell_ids.isna().any() or (cell_ids < 0).any():
        raise ValueError(
            f"{snapshot_label} {event_label} endpoint cell_id values "
            "must be non-negative"
        )
    if endpoints[["frame", "cell_id"]].isna().any(axis=None):
        raise ValueError(
            f"{snapshot_label} {event_label} endpoint keys cannot contain null values"
        )
    duplicate_mask = endpoints.duplicated(["frame", "cell_id"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = list(
            endpoints.loc[duplicate_mask, ["frame", "cell_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        raise ValueError(
            f"{snapshot_label} {event_label} endpoint (frame, cell_id) keys "
            f"must be unique; duplicates={duplicate_keys}"
        )
    return set(endpoints[["frame", "cell_id"]].itertuples(index=False, name=None))


def _select_endpoint_rows(
    endpoints: pd.DataFrame, keys: set[tuple[Any, Any]]
) -> pd.DataFrame:
    if not keys:
        return endpoints.iloc[0:0].copy().reset_index(drop=True)
    mask = [
        key in keys
        for key in endpoints[["frame", "cell_id"]].itertuples(
            index=False, name=None
        )
    ]
    return _sorted_endpoints(endpoints.loc[mask])


def _select_event_tracks(
    tracks: pd.DataFrame, endpoints: pd.DataFrame
) -> pd.DataFrame:
    if endpoints.empty:
        return tracks.iloc[0:0].copy().reset_index(drop=True)
    track_ids = endpoints["track_id"].drop_duplicates()
    return _sorted_tracks(tracks.loc[tracks["track_id"].isin(track_ids)])


def _unique_tracks(frame: pd.DataFrame) -> int:
    return int(frame["track_id"].nunique()) if "track_id" in frame else 0


def compare_stage9_snapshots(
    previous: Stage9Snapshot, new: Stage9Snapshot
) -> Stage9Comparison:
    """Compare Stage 9 events by stable ``(frame, cell_id)`` endpoint identity."""

    _validate_compatible_metadata(previous, new)
    previous_ended_keys = _event_keys(
        previous.ended_endpoints,
        snapshot_label="Previous",
        event_label="ended",
    )
    new_ended_keys = _event_keys(
        new.ended_endpoints,
        snapshot_label="New",
        event_label="ended",
    )
    previous_new_keys = _event_keys(
        previous.new_endpoints,
        snapshot_label="Previous",
        event_label="new",
    )
    new_new_keys = _event_keys(
        new.new_endpoints,
        snapshot_label="New",
        event_label="new",
    )

    removed_ended_endpoints = _select_endpoint_rows(
        previous.ended_endpoints, previous_ended_keys - new_ended_keys
    )
    added_ended_endpoints = _select_endpoint_rows(
        new.ended_endpoints, new_ended_keys - previous_ended_keys
    )
    removed_new_endpoints = _select_endpoint_rows(
        previous.new_endpoints, previous_new_keys - new_new_keys
    )
    added_new_endpoints = _select_endpoint_rows(
        new.new_endpoints, new_new_keys - previous_new_keys
    )

    removed_ended_tracks = _select_event_tracks(
        previous.ended_tracks, removed_ended_endpoints
    )
    added_ended_tracks = _select_event_tracks(new.ended_tracks, added_ended_endpoints)
    removed_new_tracks = _select_event_tracks(previous.new_tracks, removed_new_endpoints)
    added_new_tracks = _select_event_tracks(new.new_tracks, added_new_endpoints)

    summary = {
        "previous_ended_tracks": _unique_tracks(previous.ended_tracks),
        "new_ended_tracks": _unique_tracks(new.ended_tracks),
        "removed_ended_tracks": _unique_tracks(removed_ended_tracks),
        "added_ended_tracks": _unique_tracks(added_ended_tracks),
        "previous_new_tracks": _unique_tracks(previous.new_tracks),
        "new_new_tracks": _unique_tracks(new.new_tracks),
        "removed_new_tracks": _unique_tracks(removed_new_tracks),
        "added_new_tracks": _unique_tracks(added_new_tracks),
    }
    return Stage9Comparison(
        previous=previous,
        new=new,
        removed_ended_tracks=removed_ended_tracks,
        added_ended_tracks=added_ended_tracks,
        removed_new_tracks=removed_new_tracks,
        added_new_tracks=added_new_tracks,
        removed_ended_endpoints=removed_ended_endpoints,
        added_ended_endpoints=added_ended_endpoints,
        removed_new_endpoints=removed_new_endpoints,
        added_new_endpoints=added_new_endpoints,
        summary=summary,
    )
