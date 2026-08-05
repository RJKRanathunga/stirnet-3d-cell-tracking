"""Endpoint-specific division, lineage, boundary, and merge protections."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .step01_config import TrackReconciliationConfig


MERGE_TRACK_COLUMNS = (
    "track_a", "track_b", "merged_track", "parent_track_a", "parent_track_b",
    "split_track_a", "split_track_b", "source_track_id", "target_track_id",
)


def _integer(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number) or number != math.floor(number):
            return None
        return int(number)
    except (TypeError, ValueError):
        return None


def _event_track_ids(event: pd.Series | dict[str, object]) -> set[int]:
    return {
        value for column in MERGE_TRACK_COLUMNS
        if column in event and (value := _integer(event[column])) is not None
    }


def _merge_interval(event: pd.Series | dict[str, object]) -> tuple[int, int] | None:
    starts = (
        event.get("merged_interval_start"), event.get("merged_frame"),
        event.get("frame"),
    )
    start = next((value for raw in starts if (value := _integer(raw)) is not None), None)
    if start is None:
        return None
    ends = (
        event.get("merged_interval_end"), event.get("split_frame"), start,
    )
    end = next((value for raw in ends if (value := _integer(raw)) is not None), start)
    return min(start, end), max(start, end)


def transition_overlaps_merge(
    source_track_id: int,
    target_track_id: int,
    source_end_frame: int,
    target_start_frame: int,
    segmentation_events: pd.DataFrame | None,
) -> bool:
    """Return true only for a merge event relevant to this exact transition."""

    if segmentation_events is None or segmentation_events.empty:
        return False
    involved = {int(source_track_id), int(target_track_id)}
    for event in segmentation_events.to_dict("records"):
        if not (involved & _event_track_ids(event)):
            continue
        interval = _merge_interval(event)
        if interval is None:
            continue
        merge_start, merge_end = interval
        split = _integer(event.get("split_frame"))
        protected_start = merge_start - 1
        protected_end = max(merge_end + 1, split if split is not None else merge_end)
        if source_end_frame <= protected_end and target_start_frame >= protected_start:
            if target_start_frame >= merge_start and source_end_frame <= protected_end:
                return True
    return False


def _division_endpoint_sets(
    division_events: pd.DataFrame | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    confirmed_sources: set[tuple[int, int]] = set()
    confirmed_targets: set[tuple[int, int]] = set()
    probable_sources: set[tuple[int, int]] = set()
    probable_targets: set[tuple[int, int]] = set()
    if division_events is None or division_events.empty:
        return confirmed_sources, confirmed_targets, probable_sources, probable_targets
    required = {
        "parent_track_id", "parent_end_frame", "child_birth_frame",
        "child_track_a", "child_track_b", "decision",
    }
    missing = sorted(required - set(division_events.columns))
    if missing:
        raise ValueError(f"division_events is missing required columns: {missing}")
    for event in division_events.itertuples(index=False):
        decision = str(event.decision).strip().lower()
        if decision not in {"confirmed", "probable"}:
            continue
        source_set = confirmed_sources if decision == "confirmed" else probable_sources
        target_set = confirmed_targets if decision == "confirmed" else probable_targets
        source_set.add((int(event.parent_track_id), int(event.parent_end_frame)))
        for child in (int(event.child_track_a), int(event.child_track_b)):
            target_set.add((child, int(event.child_birth_frame)))
    return confirmed_sources, confirmed_targets, probable_sources, probable_targets


def _merge_endpoint_sets(
    segmentation_events: pd.DataFrame | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    source_keys: set[tuple[int, int]] = set()
    target_keys: set[tuple[int, int]] = set()
    if segmentation_events is None or segmentation_events.empty:
        return source_keys, target_keys
    for event in segmentation_events.to_dict("records"):
        interval = _merge_interval(event)
        if interval is None:
            continue
        start, end = interval
        split = _integer(event.get("split_frame"))
        ids = _event_track_ids(event)
        for track_id in ids:
            source_keys.update((track_id, frame) for frame in {start - 1, start, end})
            target_keys.update((track_id, frame) for frame in {start, end, end + 1})
            if split is not None:
                source_keys.add((track_id, split - 1))
                target_keys.add((track_id, split))
    return source_keys, target_keys


def classify_endpoints(
    summary: pd.DataFrame,
    *,
    sequence_first_frame: int,
    sequence_last_frame: int,
    config: TrackReconciliationConfig,
    segmentation_events: pd.DataFrame | None = None,
    division_events: pd.DataFrame | None = None,
    protected_tracks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply structural protections to each segment start and ending."""

    result = summary.copy(deep=True)
    confirmed_sources, confirmed_targets, probable_sources, probable_targets = (
        _division_endpoint_sets(division_events)
    )
    merge_sources, merge_targets = _merge_endpoint_sets(segmentation_events)

    if protected_tracks is not None and not protected_tracks.empty:
        required = {"track_id", "role", "protected_reason"}
        missing = sorted(required - set(protected_tracks.columns))
        if missing:
            raise ValueError(f"protected_tracks is missing required columns: {missing}")
        confirmed_ids = {track for track, _ in confirmed_sources} | {
            track for track, _ in confirmed_targets
        }
        advertised = set(pd.to_numeric(protected_tracks["track_id"]).astype(int))
        if not confirmed_ids.issubset(advertised):
            missing_ids = sorted(confirmed_ids - advertised)
            raise ValueError(
                "protected_tracks is inconsistent with confirmed division events; "
                f"missing track IDs {missing_ids}"
            )

    for index, row in result.iterrows():
        track_id = int(row["track_id"])
        real_count = int(row["real_observation_count"])
        first_real = _integer(row["first_real_frame"])
        last_real = _integer(row["last_real_frame"])

        if last_real is None:
            source_reason = "excluded_virtual_endpoint"
        elif bool(row["last_is_virtual"]) or int(row["last_frame"]) != last_real:
            source_reason = "excluded_virtual_endpoint"
        elif last_real >= sequence_last_frame:
            source_reason = "excluded_last_frame"
        elif bool(row["last_is_boundary"]):
            source_reason = "excluded_boundary_exit"
        elif (track_id, last_real) in confirmed_sources:
            source_reason = "excluded_confirmed_division"
        elif (track_id, last_real) in probable_sources:
            source_reason = "excluded_probable_division"
        elif (track_id, last_real) in merge_sources:
            source_reason = "excluded_merge_event"
        elif real_count < config.minimum_source_observations:
            source_reason = "excluded_insufficient_history"
        else:
            source_reason = "eligible"

        if first_real is None:
            target_reason = "excluded_virtual_endpoint"
        elif bool(row["first_is_virtual"]) or int(row["first_frame"]) != first_real:
            target_reason = "excluded_virtual_endpoint"
        elif first_real <= sequence_first_frame:
            target_reason = "excluded_sequence_start"
        elif bool(row["first_is_boundary"]):
            target_reason = "excluded_boundary_entry_target"
        elif (track_id, first_real) in confirmed_targets:
            target_reason = "excluded_confirmed_division"
        elif (track_id, first_real) in probable_targets:
            target_reason = "excluded_probable_division"
        elif (track_id, first_real) in merge_targets:
            target_reason = "excluded_merge_event"
        else:
            target_reason = "eligible"

        result.at[index, "source_eligible"] = source_reason == "eligible"
        result.at[index, "target_eligible"] = target_reason == "eligible"
        result.at[index, "source_exclusion_reason"] = source_reason
        result.at[index, "target_exclusion_reason"] = target_reason
    return result


def lineage_edge_conflict(
    source_track_id: int,
    target_track_id: int,
    lineage_edges: pd.DataFrame | None,
) -> bool:
    if lineage_edges is None or lineage_edges.empty:
        return False
    required = {"parent_track_id", "child_track_id"}
    missing = sorted(required - set(lineage_edges.columns))
    if missing:
        raise ValueError(f"lineage_edges is missing required columns: {missing}")
    source = int(source_track_id)
    target = int(target_track_id)
    for edge in lineage_edges.itertuples(index=False):
        pair = (int(edge.parent_track_id), int(edge.child_track_id))
        if pair in {(source, target), (target, source)}:
            return True
    return False


def protected_transition_tracks(
    start_frame: int,
    end_frame: int,
    division_events: pd.DataFrame | None,
    segmentation_events: pd.DataFrame | None,
) -> set[int]:
    """Track IDs unsuitable as stable anchors across a particular interval."""

    result: set[int] = set()
    if division_events is not None and not division_events.empty:
        for event in division_events.itertuples(index=False):
            parent_end = int(event.parent_end_frame)
            birth = int(event.child_birth_frame)
            if start_frame <= birth and parent_end <= end_frame:
                result.update({
                    int(event.parent_track_id), int(event.child_track_a),
                    int(event.child_track_b),
                })
    if segmentation_events is not None and not segmentation_events.empty:
        for event in segmentation_events.to_dict("records"):
            interval = _merge_interval(event)
            if interval is None:
                continue
            merge_start, merge_end = interval
            if merge_start <= end_frame and merge_end >= start_frame:
                result.update(_event_track_ids(event))
    return result
