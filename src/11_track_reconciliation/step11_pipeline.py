"""Production orchestration for final Stage 11 track reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from src.diagnostics import DecisionRecord, Provenance, StageTrace

from .step01_config import TrackReconciliationConfig
from .step02_observations import build_endpoint_summary, prepare_observations
from .step03_protections import classify_endpoints
from .step06_candidates import generate_candidates
from .step07_scoring import score_candidates
from .step08_assignment import resolve_assignments
from .step09_remapping import apply_remapping
from .step10_validation import validate_reconciliation


@dataclass(frozen=True)
class TrackReconciliationResult:
    tracks: pd.DataFrame
    segmentation_events: pd.DataFrame
    division_events: pd.DataFrame
    lineage_edges: pd.DataFrame
    track_lineage: pd.DataFrame
    protected_tracks: pd.DataFrame
    endpoint_classifications: pd.DataFrame
    continuation_candidates: pd.DataFrame
    continuation_decisions: pd.DataFrame
    track_id_remap: pd.DataFrame
    unresolved_endings: pd.DataFrame
    validation_results: pd.DataFrame
    metadata: dict[str, object]
    summary: dict[str, object]


def _validate_optional_table(name: str, value: pd.DataFrame | None) -> None:
    if value is not None and not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame or None")


def run_track_reconciliation(
    tracks: pd.DataFrame,
    time_frames: list[pd.DataFrame],
    *,
    segmentation_events: pd.DataFrame | None = None,
    division_events: pd.DataFrame | None = None,
    lineage_edges: pd.DataFrame | None = None,
    track_lineage: pd.DataFrame | None = None,
    protected_tracks: pd.DataFrame | None = None,
    global_motion: pd.DataFrame | None = None,
    association_events: pd.DataFrame | None = None,
    association_candidates: pd.DataFrame | None = None,
    spatial_shape_zyx: tuple[int, int, int] | None = None,
    sample_id: str = "44b6_0113de3b",
    config: TrackReconciliationConfig | None = None,
    return_diagnostics: bool = False,
):
    """Repair unexplained interior ended-segment to new-segment breaks."""

    resolved_config = config or TrackReconciliationConfig()
    if not isinstance(time_frames, list):
        time_frames = list(time_frames)
    for name, value in (
        ("segmentation_events", segmentation_events),
        ("division_events", division_events),
        ("lineage_edges", lineage_edges),
        ("track_lineage", track_lineage),
        ("protected_tracks", protected_tracks),
        ("global_motion", global_motion),
        ("association_events", association_events),
        ("association_candidates", association_candidates),
    ):
        _validate_optional_table(name, value)
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")

    prepared = prepare_observations(tracks, time_frames)
    sequence_first = 0
    sequence_last = len(time_frames) - 1
    endpoints = build_endpoint_summary(prepared, spatial_shape_zyx, resolved_config)
    endpoints = classify_endpoints(
        endpoints,
        sequence_first_frame=sequence_first,
        sequence_last_frame=sequence_last,
        config=resolved_config,
        segmentation_events=segmentation_events,
        division_events=division_events,
        protected_tracks=protected_tracks,
    )
    candidate_records = generate_candidates(
        prepared,
        endpoints,
        sample_id=sample_id,
        sequence_last_frame=sequence_last,
        segmentation_events=segmentation_events,
        division_events=division_events,
        lineage_edges=lineage_edges,
        global_motion=global_motion,
        association_candidates=association_candidates,
        spatial_shape_zyx=spatial_shape_zyx,
        config=resolved_config,
    )
    candidates = score_candidates(candidate_records, resolved_config)
    assignments = resolve_assignments(candidates, endpoints, resolved_config)
    remapped = apply_remapping(
        tracks,
        endpoints,
        assignments.applied_decisions,
        segmentation_events=segmentation_events,
        division_events=division_events,
        lineage_edges=lineage_edges,
        track_lineage=track_lineage,
        protected_tracks=protected_tracks,
    )
    validation = validate_reconciliation(
        tracks,
        remapped.tracks,
        endpoints,
        candidates,
        assignments.applied_decisions,
        remapped.track_id_remap,
        remapped.division_events,
        remapped.lineage_edges,
        remapped.track_lineage,
        remapped.segmentation_events,
        remapped.canonical_by_original,
        resolved_config,
    )

    eligible_sources = endpoints["source_eligible"].astype(bool) if not endpoints.empty else pd.Series(dtype=bool)
    eligible_targets = endpoints["target_eligible"].astype(bool) if not endpoints.empty else pd.Series(dtype=bool)
    eligible_unresolved = assignments.unresolved_endings.loc[
        assignments.unresolved_endings["source_eligible"].astype(bool)
    ] if not assignments.unresolved_endings.empty else assignments.unresolved_endings
    source_reasons = endpoints["source_exclusion_reason"] if not endpoints.empty else pd.Series(dtype=str)
    target_reasons = endpoints["target_exclusion_reason"] if not endpoints.empty else pd.Series(dtype=str)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "stage_name": "11_track_reconciliation",
        "sample_id": str(sample_id),
        "policy": resolved_config.policy,
        "configuration": resolved_config.as_dict(),
        "input_track_row_count": int(len(tracks)),
        "input_track_count": int(tracks["track_id"].nunique()),
        "final_track_row_count": int(len(remapped.tracks)),
        "final_track_count": int(remapped.tracks["track_id"].nunique()),
        "eligible_source_count": int(eligible_sources.sum()),
        "eligible_target_count": int(eligible_targets.sum()),
        "candidate_edge_count": int(candidates["admissible"].astype(bool).sum()) if not candidates.empty else 0,
        "candidate_component_count": assignments.component_count,
        "conservative_assignment_count": assignments.conservative_count,
        "forced_assignment_count": assignments.forced_count,
        "total_assignment_count": int(len(assignments.applied_decisions)),
        "unresolved_ending_count": int(len(eligible_unresolved)),
        "no_candidate_count": int((eligible_unresolved["reason"] == "unresolved_no_candidate").sum()) if not eligible_unresolved.empty else 0,
        "boundary_exclusion_count": int(source_reasons.isin(["excluded_boundary_exit"]).sum() + target_reasons.isin(["excluded_boundary_entry_target"]).sum()),
        "virtual_exclusion_count": int(source_reasons.isin(["excluded_virtual_endpoint"]).sum() + target_reasons.isin(["excluded_virtual_endpoint"]).sum()),
        "merge_exclusion_count": int(source_reasons.isin(["excluded_merge_event"]).sum() + target_reasons.isin(["excluded_merge_event"]).sum()),
        "confirmed_division_exclusion_count": int(source_reasons.isin(["excluded_confirmed_division"]).sum() + target_reasons.isin(["excluded_confirmed_division"]).sum()),
        "probable_division_exclusion_count": int(source_reasons.isin(["excluded_probable_division"]).sum() + target_reasons.isin(["excluded_probable_division"]).sum()),
        "track_ids_rewritten": bool(len(remapped.track_id_remap)),
        "lineage_ids_canonicalized": int(remapped.lineage_ids_canonicalized),
        "validation_passed": bool(validation["passed"].all()) if not validation.empty else True,
    }
    summary: dict[str, object] = {
        "policy": resolved_config.policy,
        "input_tracks": metadata["input_track_count"],
        "final_tracks": metadata["final_track_count"],
        "candidate_edges": metadata["candidate_edge_count"],
        "conservative_assignments": assignments.conservative_count,
        "forced_assignments": assignments.forced_count,
        "unresolved_eligible_endings": len(eligible_unresolved),
    }
    result = TrackReconciliationResult(
        tracks=remapped.tracks,
        segmentation_events=remapped.segmentation_events,
        division_events=remapped.division_events,
        lineage_edges=remapped.lineage_edges,
        track_lineage=remapped.track_lineage,
        protected_tracks=remapped.protected_tracks,
        endpoint_classifications=endpoints,
        continuation_candidates=candidates,
        continuation_decisions=assignments.decisions,
        track_id_remap=remapped.track_id_remap,
        unresolved_endings=assignments.unresolved_endings,
        validation_results=validation,
        metadata=metadata,
        summary=summary,
    )
    if not return_diagnostics:
        return result

    decisions: list[DecisionRecord] = []
    for row in candidates.itertuples(index=False):
        decisions.append(DecisionRecord(
            decision_type="continuation_candidate",
            outcome="admissible" if bool(row.admissible) else "excluded",
            frame=int(row.target_start_frame),
            subject_id=str(row.candidate_id),
            reason=str(row.admissibility_reason),
            metrics={
                "continuation_score": float(row.continuation_score),
                "assignment_cost": float(row.assignment_cost),
                "direct_endpoint_distance_um": float(row.direct_endpoint_distance_um),
                "gap_frames": int(row.gap_frames),
            },
            provenance=Provenance(
                source_type="continuation_candidate",
                source_stage="11_track_reconciliation",
                source_frame=int(row.source_end_frame),
                source_track_ids=(int(row.source_track_id), int(row.target_track_id)),
                source_candidate_id=str(row.candidate_id),
            ),
        ))
    for row in assignments.decisions.itertuples(index=False):
        decisions.append(DecisionRecord(
            decision_type="reconciliation_decision",
            outcome=str(row.decision),
            frame=int(row.source_end_frame) if pd.notna(row.source_end_frame) else None,
            subject_id=str(row.decision_id),
            reason=str(row.reason),
            metrics={
                "continuation_score": (
                    float(row.continuation_score)
                    if pd.notna(row.continuation_score) else math.nan
                ),
                "forced": bool(row.forced),
                "candidate_count": int(row.candidate_count),
            },
            provenance=Provenance(
                source_type="track_endpoint",
                source_stage="11_track_reconciliation",
                source_frame=(
                    int(row.source_end_frame) if pd.notna(row.source_end_frame) else None
                ),
                source_track_ids=tuple(
                    int(value) for value in (row.source_track_id, row.target_track_id)
                    if pd.notna(value)
                ),
                source_candidate_id=str(row.decision_id),
            ),
        ))
    trace = StageTrace(
        stage_name="11_track_reconciliation",
        inputs={
            "track_row_count": int(len(tracks)),
            "track_count": int(tracks["track_id"].nunique()),
            "time_frame_count": int(len(time_frames)),
            "segmentation_event_count": int(len(segmentation_events)) if segmentation_events is not None else 0,
            "division_event_count": int(len(division_events)) if division_events is not None else 0,
            "sample_id": str(sample_id),
            "policy": resolved_config.policy,
        },
        outputs={
            "final_tracks": remapped.tracks,
            "final_division_events": remapped.division_events,
            "final_lineage_edges": remapped.lineage_edges,
            "final_track_lineage": remapped.track_lineage,
        },
        intermediates={
            "prepared_observations": prepared,
            "endpoint_classifications": endpoints,
            "continuation_candidates": candidates,
            "conservative_assignment": assignments.conservative_assignment,
            "forced_assignment": assignments.forced_assignment,
            "track_id_remap": remapped.track_id_remap,
            "unresolved_endings": assignments.unresolved_endings,
            "validation_results": validation,
        },
        metrics=summary.copy(),
        decisions=decisions,
    )
    return result, trace
