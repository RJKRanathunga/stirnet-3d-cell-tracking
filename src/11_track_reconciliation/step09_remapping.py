"""Canonical identity remapping and final event/lineage canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math

import numpy as np
import pandas as pd

from .step01_config import TRACK_ID_REMAP_COLUMNS
from .step03_protections import MERGE_TRACK_COLUMNS


@dataclass(frozen=True)
class RemappingResult:
    tracks: pd.DataFrame
    segmentation_events: pd.DataFrame
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame
    track_id_remap: pd.DataFrame
    canonical_by_original: dict[int, int]
    lineage_ids_canonicalized: int


def _empty_stage10(column_name: str) -> pd.DataFrame:
    schemas = import_module("src.10_cell_lineage.step01_config")
    return pd.DataFrame(columns=getattr(schemas, column_name))


def _canonical(track_id: int, predecessor: dict[int, int]) -> int:
    current = int(track_id)
    seen: set[int] = set()
    while current in predecessor:
        if current in seen:
            raise ValueError("Stage 11 track remapping contains a cycle")
        seen.add(current)
        current = int(predecessor[current])
    return current


def _track_columns(frame: pd.DataFrame, explicit: tuple[str, ...]) -> list[str]:
    return [
        column for column in frame.columns
        if column in explicit or column.endswith("_track_id")
    ]


def _canonicalize_table(
    frame: pd.DataFrame | None,
    canonical_by_original: dict[int, int],
    *,
    explicit_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, int]:
    if frame is None:
        return pd.DataFrame(), 0
    result = frame.copy(deep=True)
    changed = 0
    for column in _track_columns(result, explicit_columns):
        values = []
        for value in result[column]:
            try:
                if pd.isna(value):
                    values.append(pd.NA)
                    continue
                original = int(float(value))
            except (TypeError, ValueError):
                values.append(value)
                continue
            canonical = canonical_by_original.get(original, original)
            changed += int(canonical != original)
            values.append(canonical)
        result[column] = values
    if not result.empty:
        result = result.drop_duplicates(ignore_index=True)
    return result, changed


def _validate_event_self_references(
    segmentation_events: pd.DataFrame,
    division_events: pd.DataFrame,
    lineage_edges: pd.DataFrame,
) -> None:
    for first, second in (
        ("track_a", "track_b"),
        ("parent_track_a", "parent_track_b"),
        ("split_track_a", "split_track_b"),
    ):
        if first in segmentation_events and second in segmentation_events:
            both = segmentation_events[[first, second]].notna().all(axis=1)
            equal = segmentation_events[first].astype(str) == segmentation_events[second].astype(str)
            if (both & equal).any():
                raise ValueError(f"Canonicalization collapsed distinct merge roles {first}/{second}")
    if not division_events.empty:
        for child in ("child_track_a", "child_track_b"):
            if child in division_events:
                both = division_events[["parent_track_id", child]].notna().all(axis=1)
                equal = (
                    division_events["parent_track_id"].astype(str)
                    == division_events[child].astype(str)
                )
                if (both & equal).any():
                    raise ValueError("Canonicalization made a division self-referential")
    if not lineage_edges.empty and (
        lineage_edges["parent_track_id"].astype(int)
        == lineage_edges["child_track_id"].astype(int)
    ).any():
        raise ValueError("Canonicalization made a lineage edge self-referential")


def _rebuild_track_lineage(
    tracks: pd.DataFrame,
    lineage_edges: pd.DataFrame,
    sequence_first_frame: int | None,
) -> pd.DataFrame:
    schemas = import_module("src.10_cell_lineage.step01_config")
    columns = schemas.TRACK_LINEAGE_COLUMNS
    if tracks.empty:
        return pd.DataFrame(columns=columns)
    track_ids = sorted(tracks["track_id"].astype(int).unique().tolist())
    existing = set(track_ids)
    first_frames = tracks.groupby("track_id", sort=True)["frame"].min().astype(int).to_dict()
    parent_by_child: dict[int, tuple[int, object]] = {}
    if not lineage_edges.empty:
        for edge in lineage_edges.sort_values(
            ["event_frame", "parent_track_id", "child_track_id"], kind="mergesort"
        ).itertuples(index=False):
            parent = int(edge.parent_track_id)
            child = int(edge.child_track_id)
            if parent not in existing or child not in existing:
                raise ValueError("Every confirmed lineage ID must exist in final tracks")
            if child in parent_by_child and parent_by_child[child][0] != parent:
                raise ValueError(f"Final track {child} has more than one lineage parent")
            parent_by_child[child] = (parent, edge.division_event_id)

    resolved: dict[int, tuple[int, int]] = {}
    visiting: set[int] = set()
    def root_and_generation(track_id: int) -> tuple[int, int]:
        if track_id in resolved:
            return resolved[track_id]
        if track_id in visiting:
            raise ValueError("Final lineage graph contains a cycle")
        visiting.add(track_id)
        if track_id not in parent_by_child:
            value = (track_id, 0)
        else:
            parent = parent_by_child[track_id][0]
            root, generation = root_and_generation(parent)
            value = (root, generation + 1)
        visiting.remove(track_id)
        resolved[track_id] = value
        return value

    rows = []
    for track_id in track_ids:
        root, generation = root_and_generation(track_id)
        if track_id in parent_by_child:
            parent, event_id = parent_by_child[track_id]
            status = "division_child"
        else:
            parent, event_id = math.nan, math.nan
            status = (
                "root" if sequence_first_frame is not None
                and first_frames[track_id] == sequence_first_frame else "unlinked"
            )
        rows.append({
            "track_id": track_id,
            "parent_track_id": parent,
            "division_event_id": event_id,
            "root_track_id": root,
            "generation": generation,
            "lineage_status": status,
        })
    return pd.DataFrame(rows, columns=columns)


def apply_remapping(
    tracks: pd.DataFrame,
    endpoints: pd.DataFrame,
    applied_decisions: pd.DataFrame,
    *,
    segmentation_events: pd.DataFrame | None,
    division_events: pd.DataFrame | None,
    lineage_edges: pd.DataFrame | None,
    track_lineage: pd.DataFrame | None,
    protected_tracks: pd.DataFrame | None,
) -> RemappingResult:
    """Rewrite accepted target segments and canonicalize downstream identity tables."""

    original_ids = sorted(tracks["track_id"].astype(int).unique().tolist())
    predecessor: dict[int, int] = {}
    remap_records: list[dict[str, object]] = []
    endpoint_by_track = endpoints.set_index("track_id", drop=False) if not endpoints.empty else None
    ordered_decisions = applied_decisions.sort_values(
        ["target_start_frame", "source_end_frame", "source_track_id", "target_track_id"],
        kind="mergesort",
    ) if not applied_decisions.empty else applied_decisions
    for decision in ordered_decisions.itertuples(index=False):
        source = int(decision.source_track_id)
        target = int(decision.target_track_id)
        if target in predecessor:
            raise ValueError(f"Target segment {target} has more than one predecessor")
        source_canonical = _canonical(source, predecessor)
        if source_canonical == target or _canonical(source_canonical, predecessor) == target:
            raise ValueError("Stage 11 track remapping would create a cycle")
        predecessor[target] = source_canonical
        canonical = _canonical(target, predecessor)
        to_frame = (
            int(endpoint_by_track.loc[target, "last_frame"])
            if endpoint_by_track is not None and target in endpoint_by_track.index
            else int(decision.target_start_frame)
        )
        remap_records.append({
            "decision_id": str(decision.decision_id),
            "original_segment_track_id": target,
            "predecessor_track_id": source,
            "canonical_track_id": canonical,
            "from_frame": int(decision.target_start_frame),
            "to_frame": to_frame,
            "gap_frames": int(decision.gap_frames),
            "decision": str(decision.decision),
            "policy_phase": str(decision.policy_phase),
            "forced": bool(decision.forced),
            "continuation_score": float(decision.continuation_score),
            "assignment_cost": float(decision.assignment_cost),
        })
    canonical_by_original = {
        track_id: _canonical(track_id, predecessor) for track_id in original_ids
    }
    final_tracks = tracks.copy(deep=True)
    final_tracks["track_id"] = final_tracks["track_id"].astype(int).map(canonical_by_original).astype(int)
    final_tracks = final_tracks.sort_values(
        ["track_id", "frame", "cell"], kind="mergesort"
    ).reset_index(drop=True)

    segmentation, segmentation_changed = _canonicalize_table(
        segmentation_events, canonical_by_original, explicit_columns=MERGE_TRACK_COLUMNS
    )
    final_ids = set(final_tracks["track_id"].astype(int))
    for column in _track_columns(segmentation, MERGE_TRACK_COLUMNS):
        segmentation[column] = [
            value if pd.isna(value) or int(value) in final_ids else pd.NA
            for value in segmentation[column]
        ]
    if division_events is None:
        divisions = _empty_stage10("DIVISION_EVENT_COLUMNS")
        division_changed = 0
    else:
        divisions, division_changed = _canonicalize_table(
            division_events, canonical_by_original,
            explicit_columns=("parent_track_id", "child_track_a", "child_track_b"),
        )
    if lineage_edges is None:
        edges = _empty_stage10("LINEAGE_EDGE_COLUMNS")
        edge_changed = 0
    else:
        edges, edge_changed = _canonicalize_table(
            lineage_edges, canonical_by_original,
            explicit_columns=("parent_track_id", "child_track_id"),
        )
    if protected_tracks is None:
        protected = _empty_stage10("PROTECTED_TRACK_COLUMNS")
        protected_changed = 0
    else:
        protected, protected_changed = _canonicalize_table(
            protected_tracks, canonical_by_original,
            explicit_columns=("track_id",),
        )
    _validate_event_self_references(segmentation, divisions, edges)
    sequence_first = int(final_tracks["frame"].min()) if not final_tracks.empty else None
    rebuilt_lineage = _rebuild_track_lineage(final_tracks, edges, sequence_first)
    remap = pd.DataFrame(remap_records, columns=TRACK_ID_REMAP_COLUMNS)
    return RemappingResult(
        tracks=final_tracks,
        segmentation_events=segmentation,
        division_events=divisions,
        lineage_edges=edges,
        track_lineage=rebuilt_lineage,
        protected_tracks=protected,
        track_id_remap=remap,
        canonical_by_original=canonical_by_original,
        lineage_ids_canonicalized=(
            segmentation_changed + division_changed + edge_changed + protected_changed
        ),
    )
