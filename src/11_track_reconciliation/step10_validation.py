"""Post-remap structural and provenance validation for Stage 11."""

from __future__ import annotations

import math

import pandas as pd

from .step01_config import VALIDATION_RESULT_COLUMNS, TrackReconciliationConfig
from .step03_protections import MERGE_TRACK_COLUMNS


def validate_reconciliation(
    input_tracks: pd.DataFrame,
    final_tracks: pd.DataFrame,
    endpoints: pd.DataFrame,
    candidates: pd.DataFrame,
    applied_decisions: pd.DataFrame,
    track_id_remap: pd.DataFrame,
    division_events: pd.DataFrame,
    lineage_edges: pd.DataFrame,
    track_lineage: pd.DataFrame,
    segmentation_events: pd.DataFrame,
    canonical_by_original: dict[int, int],
    config: TrackReconciliationConfig,
) -> pd.DataFrame:
    """Validate all destructive identity changes and raise on corruption."""

    records: list[dict[str, object]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, details: str, *, structural: bool = True) -> None:
        records.append({
            "check_name": name,
            "passed": bool(passed),
            "severity": "structural" if structural else "informational",
            "details": details,
        })
        if structural and not passed:
            failures.append(f"{name}: {details}")

    duplicate = final_tracks.duplicated(["track_id", "frame"], keep=False)
    record("no_duplicate_track_frames", not duplicate.any(), f"duplicates={int(duplicate.sum())}")
    record(
        "row_count_preserved", len(input_tracks) == len(final_tracks),
        f"input={len(input_tracks)}, final={len(final_tracks)}",
    )
    expected = input_tracks.copy(deep=True)
    expected["track_id"] = expected["track_id"].astype(int).map(canonical_by_original).astype(int)
    sort_columns = [column for column in ("track_id", "frame", "cell") if column in expected]
    expected = expected.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    actual = final_tracks.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=False)
        rows_identical = True
        row_details = "every input row appears once with only canonical track_id changed"
    except AssertionError as error:
        rows_identical = False
        row_details = str(error).splitlines()[0]
    record("original_rows_preserved", rows_identical, row_details)

    accepted_targets_unique = (
        applied_decisions["target_track_id"].nunique() == len(applied_decisions)
        if not applied_decisions.empty else True
    )
    accepted_sources_unique = (
        applied_decisions["source_track_id"].nunique() == len(applied_decisions)
        if not applied_decisions.empty else True
    )
    record("accepted_target_has_one_predecessor", accepted_targets_unique, "one-to-one targets")
    record("accepted_source_has_one_successor", accepted_sources_unique, "one-to-one sources")
    increasing = True
    for _, group in final_tracks.sort_values(["track_id", "frame"], kind="mergesort").groupby(
        "track_id", sort=True
    ):
        frames = group["frame"].astype(int).to_numpy()
        if len(frames) > 1 and not (frames[1:] > frames[:-1]).all():
            increasing = False
            break
    record("strictly_increasing_track_frames", increasing, "frame gaps are allowed")

    remap_cycle = False
    parent = {
        int(row.original_segment_track_id): int(row.predecessor_track_id)
        for row in track_id_remap.itertuples(index=False)
    }
    for start in parent:
        seen: set[int] = set()
        current = start
        while current in parent:
            if current in seen:
                remap_cycle = True
                break
            seen.add(current)
            current = parent[current]
    record("no_remap_cycle", not remap_cycle, f"remap_count={len(parent)}")
    overlap = False
    for row in applied_decisions.itertuples(index=False):
        if int(row.target_start_frame) <= int(row.source_end_frame):
            overlap = True
            break
    record("accepted_segments_do_not_overlap", not overlap, "target starts after source ends")

    confirmed = (
        division_events.loc[division_events["decision"].astype(str).str.lower() == "confirmed"]
        if not division_events.empty and "decision" in division_events else division_events.iloc[0:0]
    )
    division_distinct = True
    child_starts_intact = True
    final_first = final_tracks.groupby("track_id")["frame"].min().astype(int).to_dict()
    for event in confirmed.itertuples(index=False):
        parent_id = int(event.parent_track_id)
        children = (int(event.child_track_a), int(event.child_track_b))
        if parent_id in children or children[0] == children[1]:
            division_distinct = False
        for child in children:
            if final_first.get(child) != int(event.child_birth_frame):
                child_starts_intact = False
    record("confirmed_division_tracks_remain_distinct", division_distinct, "parent and children differ")
    record("confirmed_child_starts_remain_intact", child_starts_intact, "birth frames preserved")
    self_edge = (
        (lineage_edges["parent_track_id"].astype(int) == lineage_edges["child_track_id"].astype(int)).any()
        if not lineage_edges.empty else False
    )
    record("no_self_referential_lineage_edge", not self_edge, f"edge_count={len(lineage_edges)}")
    final_ids = set(final_tracks["track_id"].astype(int))
    lineage_ids: set[int] = set()
    if not lineage_edges.empty:
        lineage_ids.update(lineage_edges["parent_track_id"].astype(int))
        lineage_ids.update(lineage_edges["child_track_id"].astype(int))
    record(
        "lineage_ids_exist_in_tracks", lineage_ids.issubset(final_ids),
        f"missing={sorted(lineage_ids - final_ids)}",
    )
    segmentation_valid = True
    for column in [
        name for name in segmentation_events.columns
        if name in MERGE_TRACK_COLUMNS or name.endswith("_track_id")
    ]:
        values = pd.to_numeric(segmentation_events[column], errors="coerce").dropna().astype(int)
        if not set(values).issubset(final_ids):
            segmentation_valid = False
            break
    record("segmentation_track_references_resolve", segmentation_valid, "references resolve or are null")
    track_lineage_valid = (
        set(track_lineage["track_id"].astype(int)) == final_ids
        if not track_lineage.empty or final_ids else not final_ids
    )
    record("track_lineage_matches_final_tracks", track_lineage_valid, "one lineage row per final track")

    root_generation_valid = True
    if not track_lineage.empty:
        indexed = track_lineage.set_index("track_id")
        for row in track_lineage.itertuples(index=False):
            track_id = int(row.track_id)
            root = int(row.root_track_id)
            generation = int(row.generation)
            if root not in final_ids or generation < 0:
                root_generation_valid = False
                break
            if pd.notna(row.parent_track_id):
                parent_id = int(row.parent_track_id)
                if parent_id not in indexed.index:
                    root_generation_valid = False
                    break
                parent_row = indexed.loc[parent_id]
                if int(parent_row["root_track_id"]) != root or int(parent_row["generation"]) + 1 != generation:
                    root_generation_valid = False
                    break
            elif root != track_id or generation != 0:
                root_generation_valid = False
                break
    record("lineage_roots_and_generations_consistent", root_generation_valid, "acyclic generations")

    protected_endpoints = False
    endpoint_index = endpoints.set_index("track_id", drop=False) if not endpoints.empty else None
    for decision in applied_decisions.itertuples(index=False):
        source = endpoint_index.loc[int(decision.source_track_id)]
        target = endpoint_index.loc[int(decision.target_track_id)]
        if (
            not bool(source["source_eligible"])
            or not bool(target["target_eligible"])
            or bool(source["last_is_boundary"])
            or bool(target["first_is_boundary"])
            or bool(source["last_is_virtual"])
            or bool(target["first_is_virtual"])
        ):
            protected_endpoints = True
            break
    record("no_boundary_or_virtual_endpoint_reconciled", not protected_endpoints, "endpoint protections honored")

    outside_gate = False
    for decision in applied_decisions.itertuples(index=False):
        matches = candidates.loc[
            (candidates["source_track_id"].astype(int) == int(decision.source_track_id))
            & (candidates["target_track_id"].astype(int) == int(decision.target_track_id))
            & candidates["admissible"].astype(bool)
        ]
        if matches.empty:
            outside_gate = True
            break
        candidate = matches.iloc[0]
        if (
            int(candidate["gap_frames"]) < 1
            or int(candidate["gap_frames"]) > config.maximum_gap_frames
            or float(candidate["direct_endpoint_distance_um"])
            > float(candidate["hard_search_radius_um"]) + 1e-9
        ):
            outside_gate = True
            break
    record("accepted_edges_respect_hard_gates", not outside_gate, "candidate edge provenance verified")
    eliminated_targets = set(track_id_remap["original_segment_track_id"].astype(int))
    idempotent = eliminated_targets.isdisjoint(final_ids)
    record(
        "idempotent_remap_structure", idempotent,
        "accepted target segment IDs are no longer independent starts",
    )
    record(
        "score_range", (
            candidates["continuation_score"].dropna().between(0.0, 1.0).all()
            if not candidates.empty else True
        ), "continuation scores are normalized", structural=False,
    )
    if failures:
        raise ValueError("Stage 11 validation failed: " + "; ".join(failures))
    return pd.DataFrame(records, columns=VALIDATION_RESULT_COLUMNS)
