"""Production orchestration, conflict selection, and lineage outputs for Stage 10."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.diagnostics import DecisionRecord, Provenance, StageTrace

from .step01_config import (
    CONFIRMED,
    DIVISION_CANDIDATE_COLUMNS,
    DIVISION_EVENT_COLUMNS,
    LINEAGE_EDGE_COLUMNS,
    PROBABLE,
    PROTECTED_TRACK_COLUMNS,
    REJECTED,
    REJECTED_CONFLICT,
    TRACK_LINEAGE_COLUMNS,
    CellLineageConfig,
)
from .step02_observations import (
    add_endpoint_classification,
    build_track_summary,
    prepare_observations,
)
from .step03_candidates import generate_candidates
from .step04_evidence import RawEvidenceExtractor, extract_candidate_intensity_evidence
from .step05_scoring import score_candidate


@dataclass(frozen=True)
class CellLineageResult:
    division_candidates: pd.DataFrame
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame
    metadata: dict[str, object]


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _transition_overlaps_merge(
    record: dict[str, object],
    segmentation_events: pd.DataFrame | None,
) -> bool:
    if segmentation_events is None or segmentation_events.empty:
        return False
    involved = {
        int(record["parent_track_id"]),
        int(record["child_track_a"]),
        int(record["child_track_b"]),
    }
    parent_frame = int(record["parent_end_frame"])
    birth_frame = int(record["child_birth_frame"])
    id_columns = (
        "track_a", "track_b", "merged_track", "parent_track_a", "parent_track_b",
        "split_track_a", "split_track_b",
    )
    for event in segmentation_events.to_dict("records"):
        start = event.get("merged_interval_start", event.get("merged_frame", math.nan))
        end = event.get("merged_interval_end", start)
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue
        if end < parent_frame or start > birth_frame:
            continue
        event_ids: set[int] = set()
        for column in id_columns:
            value = event.get(column)
            try:
                if pd.notna(value):
                    event_ids.add(int(value))
            except (TypeError, ValueError):
                continue
        if involved & event_ids:
            return True
    return False


def _cheap_rejection(record: dict[str, object], config: CellLineageConfig) -> str:
    checks = (
        ("combined_volume_relative_error", config.maximum_combined_volume_relative_error,
         "combined_volume_mismatch"),
        ("weighted_centroid_error_um", config.maximum_weighted_centroid_error_um,
         "weighted_centroid_mismatch"),
        ("birth_separation_um", config.maximum_birth_separation_um,
         "birth_separation_too_large"),
    )
    for name, limit, reason in checks:
        try:
            value = float(record[name])
        except (TypeError, ValueError, KeyError):
            return reason
        if not math.isfinite(value) or value > limit:
            return reason
    return ""


def _intensity_defaults() -> dict[str, object]:
    return {
        "parent_final_integrated_intensity_ratio": math.nan,
        "parent_final_mask_mean_ratio": math.nan,
        "parent_final_core_intensity_ratio": math.nan,
        "parent_final_background_corrected_ratio": math.nan,
        "parent_final_core_frame_ratio_change": math.nan,
        "child_birth_intensity_ratio": math.nan,
        "child_delayed_max_intensity_ratio": math.nan,
        "child_delayed_intensity_slope": math.nan,
        "child_intensity_peak_relative_frame": math.nan,
        "child_delayed_paired_frame_count": 0,
    }


def _resolve_conflicts(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    selectable = result.loc[
        result["preliminary_decision"].isin([CONFIRMED, PROBABLE])
    ].copy()
    if selectable.empty:
        return result
    selectable["_decision_rank"] = selectable["preliminary_decision"].map(
        {CONFIRMED: 0, PROBABLE: 1}
    )
    selectable = selectable.sort_values(
        [
            "_decision_rank", "division_margin", "division_score",
            "combined_volume_score", "child_birth_frame", "parent_track_id",
            "child_track_a", "child_track_b",
        ],
        ascending=[True, False, False, False, True, True, True, True],
        kind="mergesort",
    )
    claimed_parents: set[int] = set()
    claimed_children: set[int] = set()
    selected_indices: list[int] = []
    for index, row in selectable.iterrows():
        parent = int(row["parent_track_id"])
        children = {int(row["child_track_a"]), int(row["child_track_b"])}
        if parent in claimed_parents or children & claimed_children:
            result.loc[index, "decision"] = REJECTED_CONFLICT
            result.loc[index, "rejection_reason"] = REJECTED_CONFLICT
            continue
        selected_indices.append(int(index))
        claimed_parents.add(parent)
        claimed_children.update(children)

    ordered = result.loc[selected_indices].sort_values(
        ["child_birth_frame", "parent_track_id", "child_track_a", "child_track_b"],
        kind="mergesort",
    )
    for event_id, index in enumerate(ordered.index):
        result.loc[index, "division_event_id"] = int(event_id)
    return result


def _build_division_events(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates.loc[candidates["decision"].isin([CONFIRMED, PROBABLE])]
    if selected.empty:
        return _empty(DIVISION_EVENT_COLUMNS)
    events = selected.assign(confidence=selected["division_score"])[
        list(DIVISION_EVENT_COLUMNS)
    ].sort_values("division_event_id", kind="mergesort").reset_index(drop=True)
    return events


def _build_lineage_edges(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for event in events.loc[events["decision"] == CONFIRMED].itertuples(index=False):
        for child in (int(event.child_track_a), int(event.child_track_b)):
            records.append({
                "division_event_id": int(event.division_event_id),
                "event_frame": int(event.child_birth_frame),
                "parent_track_id": int(event.parent_track_id),
                "child_track_id": child,
                "relation": "division",
                "confidence": float(event.confidence),
            })
    return pd.DataFrame(records, columns=LINEAGE_EDGE_COLUMNS)


def _assert_acyclic(parent_by_child: dict[int, int]) -> None:
    for start in parent_by_child:
        seen: set[int] = set()
        current = start
        while current in parent_by_child:
            if current in seen:
                raise ValueError("Confirmed lineage graph contains a cycle")
            seen.add(current)
            current = parent_by_child[current]


def _build_track_lineage(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    sequence_first_frame: int | None,
) -> pd.DataFrame:
    if summary.empty:
        return _empty(TRACK_LINEAGE_COLUMNS)
    track_ids = sorted(summary["track_id"].astype(int).unique().tolist())
    first_frames = summary.set_index("track_id")["first_frame"].astype(int).to_dict()
    rows: dict[int, dict[str, object]] = {
        track_id: {
            "track_id": track_id,
            "parent_track_id": math.nan,
            "division_event_id": math.nan,
            "root_track_id": track_id,
            "generation": 0,
            "lineage_status": (
                "root" if first_frames[track_id] == sequence_first_frame else "unlinked"
            ),
        }
        for track_id in track_ids
    }
    confirmed = events.loc[events["decision"] == CONFIRMED].sort_values(
        ["child_birth_frame", "parent_track_id", "child_track_a", "child_track_b"],
        kind="mergesort",
    )
    parent_by_child: dict[int, int] = {}
    divided_parents: set[int] = set()
    for event in confirmed.itertuples(index=False):
        parent = int(event.parent_track_id)
        if parent in divided_parents:
            raise ValueError(f"Track {parent} has more than one confirmed division")
        divided_parents.add(parent)
        if int(event.child_birth_frame) <= int(event.parent_end_frame):
            raise ValueError("A confirmed child must begin after the parent ends")
        for child in (int(event.child_track_a), int(event.child_track_b)):
            if child in parent_by_child:
                raise ValueError(f"Track {child} has more than one biological parent")
            parent_by_child[child] = parent
            rows[child].update({
                "parent_track_id": parent,
                "division_event_id": int(event.division_event_id),
                "root_track_id": int(rows[parent]["root_track_id"]),
                "generation": int(rows[parent]["generation"]) + 1,
                "lineage_status": "division_child",
            })
    _assert_acyclic(parent_by_child)
    lineage = pd.DataFrame([rows[track_id] for track_id in track_ids], columns=TRACK_LINEAGE_COLUMNS)
    if (lineage["generation"].astype(int) < 0).any():
        raise ValueError("Lineage generation values must be nonnegative")
    return lineage


def _build_protected_tracks(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for event in events.loc[events["decision"] == CONFIRMED].itertuples(index=False):
        for role, track_id in (
            ("parent", event.parent_track_id),
            ("child_a", event.child_track_a),
            ("child_b", event.child_track_b),
        ):
            records.append({
                "division_event_id": int(event.division_event_id),
                "track_id": int(track_id),
                "role": role,
                "protected_reason": "confirmed_division_lineage",
            })
    return pd.DataFrame(records, columns=PROTECTED_TRACK_COLUMNS)


def run_cell_lineage(
    tracks: pd.DataFrame,
    time_frames: list[pd.DataFrame],
    raw_volume,
    segmentation_files,
    *,
    segmentation_events: pd.DataFrame | None = None,
    sample_id: str = "44b6_0113de3b",
    config: CellLineageConfig | None = None,
    return_diagnostics: bool = False,
):
    """Detect conservative one-ended-parent to two-new-child divisions."""

    resolved_config = config or CellLineageConfig()
    if not isinstance(time_frames, list):
        time_frames = list(time_frames)
    segmentation_paths = tuple(Path(path) for path in segmentation_files)
    if len(segmentation_paths) != len(time_frames):
        raise ValueError(
            "segmentation_files must contain one path per Stage 6 cell table: "
            f"{len(segmentation_paths)} vs {len(time_frames)}"
        )
    if segmentation_events is not None and not isinstance(segmentation_events, pd.DataFrame):
        raise TypeError("segmentation_events must be a pandas DataFrame or None")

    observations = prepare_observations(tracks, time_frames)
    summary = build_track_summary(observations)
    extractor = RawEvidenceExtractor(raw_volume, segmentation_paths, resolved_config)
    spatial_shape = None
    if not observations.empty and segmentation_paths:
        spatial_shape = extractor.load_segmentation(int(observations["frame"].min())).shape
    summary = add_endpoint_classification(
        observations, summary, spatial_shape, resolved_config
    )
    generated = generate_candidates(
        observations, summary, str(sample_id), resolved_config
    )

    observation_lookup = observations.set_index(["track_id", "frame"], drop=False)
    scored_records: list[dict[str, object]] = []
    virtual_overlap_count = 0
    for source in generated.records:
        record = {**source, **_intensity_defaults()}
        record.update({
            "birth_masks_available": False,
            "child_a_volume_voxels": math.nan,
            "child_b_volume_voxels": math.nan,
            "tiny_fragment_child": "",
            "_raw_evidence_attempted": False,
        })
        if _transition_overlaps_merge(record, segmentation_events):
            record["transition_overlaps_virtual_merge"] = True
            record["_hard_rejection_reason"] = "virtual_observation_overlap"
            virtual_overlap_count += 1
        cheap_reason = _cheap_rejection(record, resolved_config)
        if cheap_reason and not record.get("_hard_rejection_reason"):
            record["_hard_rejection_reason"] = cheap_reason

        parent_key = (int(record["parent_track_id"]), int(record["parent_end_frame"]))
        child_a_key = (int(record["child_track_a"]), int(record["child_birth_frame"]))
        child_b_key = (int(record["child_track_b"]), int(record["child_birth_frame"]))
        parent_endpoint = observation_lookup.loc[parent_key]
        child_a_start = observation_lookup.loc[child_a_key]
        child_b_start = observation_lookup.loc[child_b_key]

        if not record.get("_hard_rejection_reason"):
            try:
                count_a = extractor.cell_voxel_count(
                    int(record["child_birth_frame"]), int(child_a_start["cell_id"])
                )
                count_b = extractor.cell_voxel_count(
                    int(record["child_birth_frame"]), int(child_b_start["cell_id"])
                )
                record["birth_masks_available"] = True
                record["child_a_volume_voxels"] = count_a
                record["child_b_volume_voxels"] = count_b
            except ValueError as error:
                if "Instance mask is missing" not in str(error):
                    raise
                record["_hard_rejection_reason"] = "missing_birth_mask"
                extractor.warnings.append(f"{record['candidate_id']}: {error}")

        if record["birth_masks_available"]:
            failed = []
            if (
                int(record["child_a_volume_voxels"]) < resolved_config.hard_minimum_child_voxels
                or float(record["child_a_volume_fraction"])
                < resolved_config.hard_minimum_child_volume_fraction
            ):
                failed.append("child_a")
            if (
                int(record["child_b_volume_voxels"]) < resolved_config.hard_minimum_child_voxels
                or float(record["child_b_volume_fraction"])
                < resolved_config.hard_minimum_child_volume_fraction
            ):
                failed.append("child_b")
            if failed:
                record["tiny_fragment_child"] = ";".join(failed)
                record["_hard_rejection_reason"] = "tiny_child_fragment"

        if not record.get("_hard_rejection_reason"):
            record["_raw_evidence_attempted"] = True
            record = extract_candidate_intensity_evidence(
                record, observations, extractor, resolved_config
            )
        record = score_candidate(
            record, parent_endpoint, child_a_start, child_b_start, resolved_config
        )
        scored_records.append(record)

    candidates = (
        pd.DataFrame(scored_records).reindex(columns=DIVISION_CANDIDATE_COLUMNS)
        if scored_records else _empty(DIVISION_CANDIDATE_COLUMNS)
    )
    candidates = _resolve_conflicts(candidates)
    if not candidates.empty:
        for column in ("tiny_fragment_child", "rejection_reason", "secondary_reasons"):
            candidates[column] = candidates[column].replace("", pd.NA)
    events = _build_division_events(candidates)
    edges = _build_lineage_edges(events)
    sequence_first = int(observations["frame"].min()) if not observations.empty else None
    sequence_last = int(observations["frame"].max()) if not observations.empty else None
    track_lineage = _build_track_lineage(summary, events, sequence_first)
    protected = _build_protected_tracks(events)

    confirmed_count = int((events["decision"] == CONFIRMED).sum()) if not events.empty else 0
    probable_count = int((events["decision"] == PROBABLE).sum()) if not events.empty else 0
    rejected_count = int(candidates["decision"].isin([REJECTED, REJECTED_CONFLICT]).sum())
    conflict_count = int((candidates["decision"] == REJECTED_CONFLICT).sum())
    tiny_count = int((candidates["rejection_reason"] == "tiny_child_fragment").sum())
    attempted_indices = [
        index for index, record in enumerate(scored_records)
        if bool(record.get("_raw_evidence_attempted", False))
    ]
    missing_intensity = 0
    for index in attempted_indices:
        row = candidates.iloc[index]
        if pd.isna(row["parent_intensity_score"]) or pd.isna(row["child_brightening_score"]):
            missing_intensity += 1
    metadata: dict[str, object] = {
        "schema_version": 1,
        "stage_name": "10_cell_lineage",
        "sample_id": str(sample_id),
        "topology_signature": "one_track_ends_then_two_tracks_start_next_frame",
        "configuration": resolved_config.as_dict(),
        "input_track_row_count": int(len(tracks)),
        "input_track_count": int(tracks["track_id"].nunique()) if "track_id" in tracks else 0,
        "sequence_first_frame": sequence_first,
        "sequence_last_frame": sequence_last,
        "eligible_parent_count": generated.eligible_parent_count,
        "topology_parent_count": generated.topology_parent_count,
        "candidate_pair_count": int(len(candidates)),
        "confirmed_event_count": confirmed_count,
        "probable_event_count": probable_count,
        "rejected_candidate_count": rejected_count,
        "conflict_rejection_count": conflict_count,
        "lineage_edge_count": int(len(edges)),
        "protected_track_count": int(len(protected)),
        "future_truncated_candidate_count": int(
            candidates["future_window_truncated"].fillna(False).astype(bool).sum()
        ),
        "missing_optional_intensity_count": missing_intensity,
        "tiny_fragment_rejection_count": tiny_count,
        "boundary_rejection_count": generated.boundary_rejection_count,
        "virtual_observation_rejection_count": (
            generated.virtual_observation_rejection_count + virtual_overlap_count
        ),
        "track_ids_rewritten": False,
        "raw_evidence_enabled": True,
        "intensity_is_hard_gate": False,
    }
    result = CellLineageResult(
        division_candidates=candidates,
        division_events=events,
        lineage_edges=edges,
        track_lineage=track_lineage,
        protected_tracks=protected,
        metadata=metadata,
    )
    if not return_diagnostics:
        return result

    decisions = [
        DecisionRecord(
            decision_type="division",
            outcome=str(row.decision),
            frame=int(row.child_birth_frame),
            subject_id=str(row.candidate_id),
            reason=(str(row.rejection_reason) if pd.notna(row.rejection_reason) else None),
            metrics={
                "division_score": float(row.division_score),
                "division_margin": float(row.division_margin),
                "spatial_score": float(row.spatial_score),
                "combined_volume_score": float(row.combined_volume_score),
                "persistence_score": float(row.persistence_score),
                "divergence_score": float(row.divergence_score),
                "best_continuation_score": float(row.best_continuation_score),
            },
            provenance=Provenance(
                source_type="division_candidate",
                source_stage="10_cell_lineage",
                source_frame=int(row.child_birth_frame),
                source_track_ids=(
                    int(row.parent_track_id), int(row.child_track_a), int(row.child_track_b)
                ),
                source_candidate_id=str(row.candidate_id),
            ),
        )
        for row in candidates.itertuples(index=False)
    ]
    trace = StageTrace(
        stage_name="10_cell_lineage",
        inputs={
            "track_row_count": int(len(tracks)),
            "time_frame_count": int(len(time_frames)),
            "segmentation_file_count": int(len(segmentation_paths)),
            "sample_id": str(sample_id),
        },
        outputs={
            "division_events": events,
            "lineage_edges": edges,
            "track_lineage": track_lineage,
            "protected_tracks": protected,
        },
        intermediates={
            "track_summary": summary,
            "division_candidates": candidates,
        },
        metrics={
            "candidate_pair_count": int(len(candidates)),
            "confirmed_event_count": confirmed_count,
            "probable_event_count": probable_count,
            "rejected_candidate_count": rejected_count,
            "conflict_rejection_count": conflict_count,
            "lineage_edge_count": int(len(edges)),
        },
        decisions=decisions,
        warnings=list(dict.fromkeys(extractor.warnings)),
    )
    return result, trace
