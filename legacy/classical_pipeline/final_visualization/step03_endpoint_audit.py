"""Residual endpoint, temporal-gap, and Stage 11 repair auditing."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .step01_config import (
    DIAGNOSTIC_EVENT_COLUMNS,
    TRACK_SUMMARY_COLUMNS,
    FinalVisualizationConfig,
)


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


def _number(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _join_ids(values: Iterable[object]) -> str:
    ids = sorted({value for raw in values if (value := _integer(raw)) is not None})
    return ";".join(str(value) for value in ids)


def build_original_to_canonical(
    track_ids: Iterable[object],
    endpoint_classifications: pd.DataFrame | None,
    track_id_remap: pd.DataFrame | None,
) -> dict[int, int]:
    mapping = {int(value): int(value) for value in track_ids}
    if endpoint_classifications is not None and not endpoint_classifications.empty:
        for value in endpoint_classifications.get("track_id", pd.Series(dtype=float)):
            integer = _integer(value)
            if integer is not None:
                mapping.setdefault(integer, integer)
    if track_id_remap is not None and not track_id_remap.empty:
        required = {"original_segment_track_id", "canonical_track_id"}
        missing = sorted(required - set(track_id_remap.columns))
        if missing:
            raise ValueError(f"track_id_remap is missing required columns: {missing}")
        for row in track_id_remap.itertuples(index=False):
            original = _integer(row.original_segment_track_id)
            canonical = _integer(row.canonical_track_id)
            if original is not None and canonical is not None:
                mapping[original] = canonical
            predecessor = _integer(getattr(row, "predecessor_track_id", None))
            if predecessor is not None and canonical is not None:
                mapping[predecessor] = canonical
    return mapping


def _division_endpoint_sets(
    division_events: pd.DataFrame | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    confirmed_parents: set[tuple[int, int]] = set()
    confirmed_children: set[tuple[int, int]] = set()
    probable_parents: set[tuple[int, int]] = set()
    probable_children: set[tuple[int, int]] = set()
    if division_events is None or division_events.empty:
        return confirmed_parents, confirmed_children, probable_parents, probable_children
    required = {
        "parent_track_id", "parent_end_frame", "child_birth_frame",
        "child_track_a", "child_track_b", "decision",
    }
    missing = sorted(required - set(division_events.columns))
    if missing:
        raise ValueError(f"division_events is missing required columns: {missing}")
    for row in division_events.itertuples(index=False):
        decision = str(row.decision).strip().lower()
        if decision not in {"confirmed", "probable"}:
            continue
        parent_set = confirmed_parents if decision == "confirmed" else probable_parents
        child_set = confirmed_children if decision == "confirmed" else probable_children
        parent_set.add((int(row.parent_track_id), int(row.parent_end_frame)))
        for child in (int(row.child_track_a), int(row.child_track_b)):
            child_set.add((child, int(row.child_birth_frame)))
    return confirmed_parents, confirmed_children, probable_parents, probable_children


def _event_track_ids(event: dict[str, object]) -> set[int]:
    return {
        value for column in MERGE_TRACK_COLUMNS
        if column in event and (value := _integer(event[column])) is not None
    }


def _merge_interval(event: dict[str, object]) -> tuple[int, int] | None:
    start = next((value for raw in (
        event.get("merged_interval_start"), event.get("merged_frame"), event.get("frame")
    ) if (value := _integer(raw)) is not None), None)
    if start is None:
        return None
    end = next((value for raw in (
        event.get("merged_interval_end"), event.get("split_frame"), start
    ) if (value := _integer(raw)) is not None), start)
    return min(start, end), max(start, end)


def _merge_endpoint_sets(
    segmentation_events: pd.DataFrame | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]:
    starts: set[tuple[int, int]] = set()
    ends: set[tuple[int, int]] = set()
    tracks: set[int] = set()
    if segmentation_events is None or segmentation_events.empty:
        return starts, ends, tracks
    for event in segmentation_events.to_dict("records"):
        interval = _merge_interval(event)
        if interval is None:
            continue
        start, end = interval
        split = _integer(event.get("split_frame"))
        ids = _event_track_ids(event)
        tracks.update(ids)
        for track_id in ids:
            starts.update((track_id, frame) for frame in {start, end, end + 1})
            ends.update((track_id, frame) for frame in {start - 1, start, end})
            if split is not None:
                starts.add((track_id, split))
                ends.add((track_id, split - 1))
    return starts, ends, tracks


def _build_cell_lookup(cells: pd.DataFrame) -> dict[tuple[int, int], dict[str, object]]:
    if cells.empty or not {"frame", "cell_id"}.issubset(cells.columns):
        return {}
    result: dict[tuple[int, int], dict[str, object]] = {}
    for row in cells.drop_duplicates(["frame", "cell_id"], keep="first").to_dict("records"):
        frame = _integer(row.get("frame"))
        cell_id = _integer(row.get("cell_id"))
        if frame is not None and cell_id is not None:
            result[(frame, cell_id)] = row
    return result


def _boundary_distance_um(
    track_row: pd.Series,
    cell_lookup: dict[tuple[int, int], dict[str, object]],
    spatial_shape_zyx: tuple[int, int, int] | None,
    voxel_size_zyx_um: tuple[float, float, float],
) -> float:
    if spatial_shape_zyx is None:
        return math.nan
    shape = np.asarray(spatial_shape_zyx, dtype=float)
    voxel = np.asarray(voxel_size_zyx_um, dtype=float)
    coordinates = np.asarray([track_row.z, track_row.y, track_row.x], dtype=float)
    centroid_distance = float(np.minimum(coordinates * voxel, (shape - 1 - coordinates) * voxel).min())
    frame = _integer(track_row.get("frame"))
    cell_id = _integer(track_row.get("cell_id"))
    cell = cell_lookup.get((frame, cell_id)) if frame is not None and cell_id is not None else None
    bbox_columns = ("z_min", "y_min", "x_min", "z_max", "y_max", "x_max")
    if cell is None or not all(column in cell for column in bbox_columns):
        return centroid_distance
    bbox_min = np.asarray([cell["z_min"], cell["y_min"], cell["x_min"]], dtype=float)
    bbox_max = np.asarray([cell["z_max"], cell["y_max"], cell["x_max"]], dtype=float)
    if not np.isfinite(np.concatenate([bbox_min, bbox_max])).all():
        return centroid_distance
    return float(np.minimum(bbox_min * voxel, (shape - bbox_max) * voxel).min())


def _endpoint_rows_by_canonical(
    endpoint_classifications: pd.DataFrame | None,
    mapping: dict[int, int],
) -> dict[int, pd.DataFrame]:
    if endpoint_classifications is None or endpoint_classifications.empty:
        return {}
    if "track_id" not in endpoint_classifications.columns:
        raise ValueError("endpoint_classifications is missing track_id")
    frame = endpoint_classifications.copy(deep=True)
    frame["canonical_track_id"] = [
        mapping.get(int(value), int(value)) for value in frame["track_id"]
    ]
    return {
        int(track_id): group.copy()
        for track_id, group in frame.groupby("canonical_track_id", sort=False)
    }


def _boundary_info(
    rows: pd.DataFrame | None,
    endpoint: str,
) -> tuple[bool, float, str]:
    if rows is None or rows.empty:
        return False, math.nan, ""
    if endpoint == "start":
        frame_column = "first_real_frame" if "first_real_frame" in rows else "first_frame"
        selected = rows.loc[pd.to_numeric(rows[frame_column], errors="coerce").idxmin()]
        boundary_column = "first_is_boundary"
        reason_column = "target_exclusion_reason"
    else:
        frame_column = "last_real_frame" if "last_real_frame" in rows else "last_frame"
        selected = rows.loc[pd.to_numeric(rows[frame_column], errors="coerce").idxmax()]
        boundary_column = "last_is_boundary"
        reason_column = "source_exclusion_reason"
    boundary = bool(selected.get(boundary_column, False))
    distance = _number(selected.get(f"{endpoint}_boundary_distance_um", math.nan))
    reason = str(selected.get(reason_column, ""))
    return boundary, distance, reason


def _classify_start(
    track_id: int,
    frame: int,
    *,
    sequence_first_frame: int,
    boundary: bool,
    stage11_reason: str,
    confirmed_children: set[tuple[int, int]],
    probable_children: set[tuple[int, int]],
    merge_starts: set[tuple[int, int]],
) -> tuple[str, str]:
    key = (track_id, frame)
    if frame <= sequence_first_frame:
        return "sequence_start", "track is present at the first sequence frame"
    if stage11_reason == "excluded_virtual_endpoint":
        return "virtual_endpoint", "Stage 11 marked the starting endpoint as virtual"
    if boundary or stage11_reason == "excluded_boundary_entry_target":
        return "boundary_entry", "track begins at the acquisition boundary"
    if key in confirmed_children or stage11_reason == "excluded_confirmed_division":
        return "confirmed_division_child", "track is a confirmed division child"
    if key in probable_children or stage11_reason == "excluded_probable_division":
        return "probable_division_child", "track is a probable division child"
    if key in merge_starts or stage11_reason == "excluded_merge_event":
        return "merge_related_start", "track start is protected by a merge transition"
    return "unexplained_interior_start", "track begins inside the volume without a protected event"


def _classify_end(
    track_id: int,
    frame: int,
    *,
    sequence_last_frame: int,
    boundary: bool,
    stage11_reason: str,
    confirmed_parents: set[tuple[int, int]],
    probable_parents: set[tuple[int, int]],
    merge_ends: set[tuple[int, int]],
) -> tuple[str, str]:
    key = (track_id, frame)
    if frame >= sequence_last_frame:
        return "sequence_end", "track is present at the final sequence frame"
    if stage11_reason == "excluded_virtual_endpoint":
        return "virtual_endpoint", "Stage 11 marked the ending endpoint as virtual"
    if boundary or stage11_reason == "excluded_boundary_exit":
        return "boundary_exit", "track ends at the acquisition boundary"
    if key in confirmed_parents or stage11_reason == "excluded_confirmed_division":
        return "confirmed_division_parent", "track is a confirmed division parent"
    if key in probable_parents or stage11_reason == "excluded_probable_division":
        return "probable_division_parent", "track is a probable division parent"
    if key in merge_ends or stage11_reason == "excluded_merge_event":
        return "merge_related_end", "track ending is protected by a merge transition"
    return "unexplained_interior_end", "track ends inside the volume without a protected event"


def audit_final_tracks(
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    spatial_shape_zyx: tuple[int, int, int] | None,
    sequence_first_frame: int,
    sequence_last_frame: int,
    endpoint_classifications: pd.DataFrame | None,
    continuation_decisions: pd.DataFrame | None,
    track_id_remap: pd.DataFrame | None,
    unresolved_endings: pd.DataFrame | None,
    division_events: pd.DataFrame | None,
    segmentation_events: pd.DataFrame | None,
    config: FinalVisualizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """Build one final-track summary and a normalized diagnostic event table."""

    final_ids = sorted(pd.to_numeric(tracks["track_id"], errors="raise").astype(int).unique())
    mapping = build_original_to_canonical(final_ids, endpoint_classifications, track_id_remap)
    endpoint_by_canonical = _endpoint_rows_by_canonical(endpoint_classifications, mapping)
    confirmed_parents, confirmed_children, probable_parents, probable_children = (
        _division_endpoint_sets(division_events)
    )
    merge_starts, merge_ends, merge_track_ids = _merge_endpoint_sets(segmentation_events)

    remap = track_id_remap if track_id_remap is not None else pd.DataFrame()
    decisions = continuation_decisions if continuation_decisions is not None else pd.DataFrame()
    unresolved = unresolved_endings if unresolved_endings is not None else pd.DataFrame()
    if not remap.empty:
        remap = remap.copy(deep=True)
        remap["canonical_track_id"] = pd.to_numeric(remap["canonical_track_id"], errors="coerce")
    if not unresolved.empty and "source_track_id" in unresolved:
        unresolved = unresolved.copy(deep=True)
        unresolved["canonical_track_id"] = [
            mapping.get(int(value), int(value)) for value in unresolved["source_track_id"]
        ]

    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    event_number = 0
    cell_lookup = _build_cell_lookup(cells)

    for track_id, group in tracks.groupby("track_id", sort=True):
        track_id = int(track_id)
        group = group.sort_values("frame", kind="stable")
        first = group.iloc[0]
        last = group.iloc[-1]
        frames = np.sort(group["frame"].astype(int).unique())
        differences = np.diff(frames)
        gap_sizes = differences[differences > 1] - 1
        endpoint_rows = endpoint_by_canonical.get(track_id)
        stage11_start_boundary, _, start_stage11_reason = _boundary_info(endpoint_rows, "start")
        stage11_end_boundary, _, end_stage11_reason = _boundary_info(endpoint_rows, "end")
        start_boundary_distance = _boundary_distance_um(
            first, cell_lookup, spatial_shape_zyx, config.voxel_size_zyx_um
        )
        end_boundary_distance = _boundary_distance_um(
            last, cell_lookup, spatial_shape_zyx, config.voxel_size_zyx_um
        )
        start_boundary = (
            start_boundary_distance <= config.boundary_margin_um
            if math.isfinite(start_boundary_distance) else stage11_start_boundary
        )
        end_boundary = (
            end_boundary_distance <= config.boundary_margin_um
            if math.isfinite(end_boundary_distance) else stage11_end_boundary
        )

        start_classification, start_reason = _classify_start(
            track_id, int(first.frame), sequence_first_frame=sequence_first_frame,
            boundary=start_boundary, stage11_reason=start_stage11_reason,
            confirmed_children=confirmed_children, probable_children=probable_children,
            merge_starts=merge_starts,
        )
        end_classification, end_reason = _classify_end(
            track_id, int(last.frame), sequence_last_frame=sequence_last_frame,
            boundary=end_boundary, stage11_reason=end_stage11_reason,
            confirmed_parents=confirmed_parents, probable_parents=probable_parents,
            merge_ends=merge_ends,
        )
        suspicious_start = start_classification == "unexplained_interior_start"
        suspicious_end = end_classification == "unexplained_interior_end"
        short_lived = len(group) <= config.short_track_max_observations
        has_gap = len(gap_sizes) > 0

        track_remap = remap[remap["canonical_track_id"] == track_id] if not remap.empty else remap
        repair_count = int(len(track_remap))
        forced_count = int(track_remap.get("forced", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if repair_count else 0
        scores = pd.to_numeric(track_remap.get("continuation_score", pd.Series(dtype=float)), errors="coerce")
        weakest_score = float(scores.min()) if scores.notna().any() else math.nan
        original_ids = {track_id}
        if endpoint_rows is not None:
            original_ids.update(pd.to_numeric(endpoint_rows["track_id"], errors="coerce").dropna().astype(int))
        if repair_count:
            original_ids.update(pd.to_numeric(track_remap["original_segment_track_id"], errors="coerce").dropna().astype(int))
        unresolved_rows = unresolved.iloc[0:0]
        unresolved_track = False
        if not unresolved.empty:
            source_eligible = unresolved.get(
                "source_eligible", pd.Series(True, index=unresolved.index)
            ).fillna(False).astype(bool)
            unresolved_rows = unresolved.loc[
                (unresolved["canonical_track_id"] == track_id)
                & (pd.to_numeric(unresolved.get("source_end_frame"), errors="coerce") == int(last.frame))
                & source_eligible
            ]
            unresolved_track = not unresolved_rows.empty

        failure_score = 0.0
        reasons: list[str] = []
        if suspicious_start:
            failure_score += config.suspicious_endpoint_weight
            reasons.append("unexplained interior birth")
        if suspicious_end:
            failure_score += config.suspicious_endpoint_weight
            reasons.append("unexplained interior termination")
        if unresolved_track:
            failure_score += config.unresolved_ending_weight
            reasons.append("Stage 11 unresolved ending")
        if short_lived:
            failure_score += config.short_track_weight
            reasons.append(f"short track ({len(group)} observations)")
        if has_gap:
            failure_score += config.temporal_gap_weight
            reasons.append(f"{int(gap_sizes.sum())} missing frame(s) inside final track")
        if forced_count:
            failure_score += config.forced_repair_weight
            reasons.append(f"{forced_count} forced Stage 11 repair(s)")
        if math.isfinite(weakest_score) and weakest_score < config.low_confidence_repair_score:
            failure_score += config.low_confidence_repair_weight
            reasons.append(f"weakest repair score {weakest_score:.3f}")
        failure_score = min(1.0, failure_score)
        division_related = any(
            classification.startswith(("confirmed_division", "probable_division"))
            for classification in (start_classification, end_classification)
        )
        merge_related = track_id in merge_track_ids or any(
            classification.startswith("merge_related")
            for classification in (start_classification, end_classification)
        )

        summaries.append({
            "track_id": track_id,
            "first_frame": int(first.frame),
            "last_frame": int(last.frame),
            "observation_count": int(len(group)),
            "duration_frames": int(last.frame - first.frame + 1),
            "frame_gap_count": int(len(gap_sizes)),
            "missing_frame_count": int(gap_sizes.sum()) if len(gap_sizes) else 0,
            "start_cell_id": _integer(first.get("cell_id")),
            "end_cell_id": _integer(last.get("cell_id")),
            "start_z": float(first.z), "start_y": float(first.y), "start_x": float(first.x),
            "end_z": float(last.z), "end_y": float(last.y), "end_x": float(last.x),
            "start_boundary_distance_um": start_boundary_distance,
            "end_boundary_distance_um": end_boundary_distance,
            "start_classification": start_classification,
            "end_classification": end_classification,
            "start_reason": start_reason,
            "end_reason": end_reason,
            "suspicious_start": suspicious_start,
            "suspicious_end": suspicious_end,
            "short_lived": short_lived,
            "has_temporal_gap": has_gap,
            "stage11_modified": repair_count > 0,
            "repair_count": repair_count,
            "forced_repair_count": forced_count,
            "weakest_repair_score": weakest_score,
            "original_segment_ids": _join_ids(original_ids),
            "unresolved_ending": unresolved_track,
            "division_related": division_related,
            "merge_related": merge_related,
            "failure_score": failure_score,
            "failure_reasons": "; ".join(reasons),
        })

        for endpoint, row, classification, reason, is_failure in (
            ("track_start", first, start_classification, start_reason, suspicious_start),
            ("track_end", last, end_classification, end_reason, suspicious_end),
        ):
            events.append({
                "event_id": f"endpoint_{event_number:06d}", "track_id": track_id,
                "event_type": endpoint, "frame": int(row.frame),
                "z": float(row.z), "y": float(row.y), "x": float(row.x),
                "classification": classification, "is_failure": is_failure,
                "severity": "high" if is_failure else "info",
                "score": config.suspicious_endpoint_weight if is_failure else 0.0,
                "reason": reason, "related_track_ids": str(track_id),
                "stage11_modified": repair_count > 0, "forced": False,
                "decision_id": "",
            })
            event_number += 1

        if unresolved_track:
            unresolved_row = unresolved_rows.iloc[0]
            events.append({
                "event_id": f"unresolved_{event_number:06d}", "track_id": track_id,
                "event_type": "stage11_unresolved", "frame": int(last.frame),
                "z": float(last.z), "y": float(last.y), "x": float(last.x),
                "classification": "unresolved_eligible_ending",
                "is_failure": True, "severity": "high",
                "score": config.unresolved_ending_weight,
                "reason": str(unresolved_row.get("reason", "Stage 11 did not resolve this ending")),
                "related_track_ids": _join_ids((
                    track_id, unresolved_row.get("best_target_track_id")
                )),
                "stage11_modified": repair_count > 0, "forced": False,
                "decision_id": "",
            })
            event_number += 1

        for previous_frame, next_frame in zip(frames[:-1], frames[1:]):
            if next_frame - previous_frame <= 1:
                continue
            next_row = group[group["frame"] == next_frame].iloc[0]
            events.append({
                "event_id": f"gap_{event_number:06d}", "track_id": track_id,
                "event_type": "temporal_gap", "frame": int(next_frame),
                "z": float(next_row.z), "y": float(next_row.y), "x": float(next_row.x),
                "classification": "missing_observations",
                "is_failure": True, "severity": "warning",
                "score": config.temporal_gap_weight,
                "reason": f"track resumes after {int(next_frame - previous_frame - 1)} missing frame(s)",
                "related_track_ids": str(track_id), "stage11_modified": repair_count > 0,
                "forced": False, "decision_id": "",
            })
            event_number += 1

        if repair_count:
            for repair in track_remap.to_dict("records"):
                forced = bool(repair.get("forced", False))
                score = _number(repair.get("continuation_score"))
                weak = math.isfinite(score) and score < config.low_confidence_repair_score
                source_frame = _integer(repair.get("from_frame"))
                location_rows = group[group["frame"] >= source_frame] if source_frame is not None else group.iloc[0:0]
                location = location_rows.iloc[0] if not location_rows.empty else first
                events.append({
                    "event_id": f"repair_{event_number:06d}", "track_id": track_id,
                    "event_type": "stage11_repair", "frame": int(location.frame),
                    "z": float(location.z), "y": float(location.y), "x": float(location.x),
                    "classification": "forced_repair" if forced else "accepted_repair",
                    "is_failure": False, "severity": "warning" if (forced or weak) else "info",
                    "score": score if math.isfinite(score) else 0.0,
                    "reason": str(repair.get("decision", "Stage 11 continuation")),
                    "related_track_ids": _join_ids((
                        repair.get("predecessor_track_id"), repair.get("original_segment_track_id"),
                        repair.get("canonical_track_id"),
                    )),
                    "stage11_modified": True, "forced": forced,
                    "decision_id": str(repair.get("decision_id", "")),
                })
                event_number += 1

    summary = pd.DataFrame(summaries, columns=TRACK_SUMMARY_COLUMNS)
    diagnostic_events = pd.DataFrame(events, columns=DIAGNOSTIC_EVENT_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(
            ["failure_score", "track_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
    return summary, diagnostic_events, mapping
