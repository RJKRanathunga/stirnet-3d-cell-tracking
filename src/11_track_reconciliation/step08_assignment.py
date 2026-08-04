"""Connected-component global one-to-one assignment for Stage 11."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .step01_config import (
    CONTINUATION_DECISION_COLUMNS,
    UNRESOLVED_ENDING_COLUMNS,
    TrackReconciliationConfig,
)


@dataclass(frozen=True)
class AssignmentResult:
    decisions: pd.DataFrame
    applied_decisions: pd.DataFrame
    unresolved_endings: pd.DataFrame
    component_count: int
    conservative_count: int
    forced_count: int
    conservative_assignment: pd.DataFrame
    forced_assignment: pd.DataFrame


def _component_maps(candidates: pd.DataFrame) -> tuple[dict[int, int], int]:
    admissible = candidates.loc[candidates["admissible"].astype(bool)]
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    edge_index: dict[tuple[tuple[str, int], tuple[str, int]], list[int]] = {}
    for index, row in admissible.iterrows():
        source = ("source", int(row["source_track_id"]))
        target = ("target", int(row["target_track_id"]))
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        edge_index.setdefault((source, target), []).append(int(index))
    mapping: dict[int, int] = {}
    seen: set[tuple[str, int]] = set()
    component_id = 0
    for start in sorted(adjacency, key=lambda node: (node[1], node[0])):
        if start in seen:
            continue
        stack = [start]
        nodes: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.add(node)
            stack.extend(sorted(adjacency[node] - seen, reverse=True))
        for (source, target), indices in edge_index.items():
            if source in nodes and target in nodes:
                for index in indices:
                    mapping[index] = component_id
        component_id += 1
    return mapping, component_id


def _solve(
    candidates: pd.DataFrame,
    allowed_indices: set[int],
    unmatched_cost: float,
) -> set[int]:
    """Solve each allowed bipartite component with source-specific dummies."""

    if not allowed_indices:
        return set()
    allowed = candidates.loc[sorted(allowed_indices)]
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for _, row in allowed.iterrows():
        source = ("source", int(row["source_track_id"]))
        target = ("target", int(row["target_track_id"]))
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    seen: set[tuple[str, int]] = set()
    selected: set[int] = set()
    for start in sorted(adjacency, key=lambda node: (node[1], node[0])):
        if start in seen:
            continue
        stack = [start]
        nodes: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.add(node)
            stack.extend(sorted(adjacency[node] - seen, reverse=True))
        sources = sorted(node[1] for node in nodes if node[0] == "source")
        targets = sorted(node[1] for node in nodes if node[0] == "target")
        source_lookup = {track_id: index for index, track_id in enumerate(sources)}
        target_lookup = {track_id: index for index, track_id in enumerate(targets)}
        matrix = np.full(
            (len(sources), len(targets) + len(sources)),
            1e9,
            dtype=float,
        )
        edge_by_pair: dict[tuple[int, int], int] = {}
        component_edges = allowed.loc[
            allowed["source_track_id"].astype(int).isin(sources)
            & allowed["target_track_id"].astype(int).isin(targets)
        ]
        for index, row in component_edges.iterrows():
            source_index = source_lookup[int(row["source_track_id"])]
            target_index = target_lookup[int(row["target_track_id"])]
            cost = float(row["assignment_cost"])
            pair = (source_index, target_index)
            previous = edge_by_pair.get(pair)
            if previous is None or cost < matrix[pair] or (
                cost == matrix[pair]
                and str(row["candidate_id"]) < str(candidates.loc[previous, "candidate_id"])
            ):
                matrix[pair] = cost
                edge_by_pair[pair] = int(index)
        for source_index in range(len(sources)):
            matrix[source_index, len(targets) + source_index] = unmatched_cost
        rows, columns = linear_sum_assignment(matrix)
        for row_index, column_index in zip(rows, columns):
            if column_index < len(targets) and (row_index, column_index) in edge_by_pair:
                selected.add(edge_by_pair[(row_index, column_index)])
    return selected


def _strong_support(row: pd.Series, config: TrackReconciliationConfig) -> tuple[bool, str]:
    anchor = (
        int(row["anchor_count"]) >= config.minimum_anchor_count_for_strong_support
        and math.isfinite(float(row["anchor_prediction_error_um"]))
        and float(row["anchor_prediction_error_um"]) <= config.anchor_support_max_error_um
    )
    backward_available = bool(row["backward_prediction_available"])
    bidirectional = (
        backward_available
        and math.isfinite(float(row["bidirectional_disagreement_um"]))
        and float(row["bidirectional_disagreement_um"])
        <= config.bidirectional_support_max_disagreement_um
        and float(row["forward_error_um"]) <= config.position_score_scale_um
        and float(row["backward_error_um"]) <= config.backward_score_scale_um
    )
    truly_unique = (
        int(row["source_candidate_count"]) == 1
        and int(row["target_predecessor_count"]) == 1
    )
    forward_unique = (
        truly_unique
        and float(row["forward_error_um"])
        <= config.highly_convincing_forward_error_um
    )
    if anchor:
        return True, "accepted_anchor_supported"
    if bidirectional:
        return True, "accepted_bidirectional_motion"
    if forward_unique:
        return True, "accepted_unique_candidate"
    return False, "accepted_high_confidence"


def _conservative_qualifies(row: pd.Series, config: TrackReconciliationConfig) -> tuple[bool, str]:
    if float(row["continuation_score"]) < config.conservative_minimum_score:
        return False, "below_conservative_score"
    truly_unique = (
        int(row["source_candidate_count"]) == 1
        and int(row["target_predecessor_count"]) == 1
    )
    margins = (
        float(row["source_score_margin"]) >= config.conservative_minimum_source_margin
        and float(row["target_score_margin"]) >= config.conservative_minimum_target_margin
    )
    if not (truly_unique or margins):
        return False, "insufficient_assignment_margin"
    supported, label = _strong_support(row, config)
    if not supported:
        return False, "insufficient_strong_support"
    return True, label


def _decision_record(
    row: pd.Series,
    *,
    component_id: int,
    decision: str,
    phase: str,
    forced: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "decision_id": "",
        "component_id": component_id,
        "source_track_id": int(row["source_track_id"]),
        "target_track_id": int(row["target_track_id"]),
        "source_end_frame": int(row["source_end_frame"]),
        "target_start_frame": int(row["target_start_frame"]),
        "gap_frames": int(row["gap_frames"]),
        "decision": decision,
        "policy_phase": phase,
        "forced": bool(forced),
        "continuation_score": float(row["continuation_score"]),
        "assignment_cost": float(row["assignment_cost"]),
        "source_rank": int(row["source_rank"]),
        "target_rank": int(row["target_rank"]),
        "source_score_margin": float(row["source_score_margin"]),
        "target_score_margin": float(row["target_score_margin"]),
        "candidate_count": int(row["source_candidate_count"]),
        "reason": reason,
    }


def resolve_assignments(
    candidates: pd.DataFrame,
    endpoints: pd.DataFrame,
    config: TrackReconciliationConfig,
) -> AssignmentResult:
    """Apply conservative resolution, then optional hard-gated submission fallback."""

    admissible_indices = set(
        candidates.index[candidates["admissible"].astype(bool)].astype(int).tolist()
    ) if not candidates.empty else set()
    component_by_index, component_count = _component_maps(candidates)
    conservative_allowed: set[int] = set()
    conservative_labels: dict[int, str] = {}
    for index in sorted(admissible_indices):
        qualifies, label = _conservative_qualifies(candidates.loc[index], config)
        if qualifies:
            conservative_allowed.add(index)
            conservative_labels[index] = label
    conservative_selected = _solve(
        candidates, conservative_allowed, config.conservative_unmatched_cost
    )
    selected_records: list[dict[str, object]] = []
    applied_indices: set[int] = set()
    if config.policy == "diagnostic":
        proposed = _solve(candidates, admissible_indices, config.forced_unmatched_cost)
        for index in sorted(proposed):
            row = candidates.loc[index]
            selected_records.append(_decision_record(
                row,
                component_id=component_by_index[index],
                decision="diagnostic_proposal",
                phase="diagnostic",
                forced=False,
                reason="globally_best_admissible_diagnostic_edge",
            ))
        conservative_selected = set()
        forced_selected: set[int] = set()
    else:
        for index in sorted(conservative_selected):
            row = candidates.loc[index]
            selected_records.append(_decision_record(
                row,
                component_id=component_by_index[index],
                decision=conservative_labels[index],
                phase="conservative",
                forced=False,
                reason="evidence_and_global_assignment_passed",
            ))
        applied_indices.update(conservative_selected)
        forced_selected = set()
        if config.policy == "submission":
            used_sources = {
                int(candidates.loc[index, "source_track_id"])
                for index in conservative_selected
            }
            used_targets = {
                int(candidates.loc[index, "target_track_id"])
                for index in conservative_selected
            }
            fallback_allowed = {
                index for index in admissible_indices
                if int(candidates.loc[index, "source_track_id"]) not in used_sources
                and int(candidates.loc[index, "target_track_id"]) not in used_targets
            }
            forced_selected = _solve(
                candidates, fallback_allowed, config.forced_unmatched_cost
            )
            for index in sorted(forced_selected):
                row = candidates.loc[index]
                selected_records.append(_decision_record(
                    row,
                    component_id=component_by_index[index],
                    decision="forced_best_candidate",
                    phase="forced_submission",
                    forced=True,
                    reason="globally_best_remaining_admissible_edge",
                ))
            applied_indices.update(forced_selected)

    selected_sources = {int(record["source_track_id"]) for record in selected_records}
    applied_targets = {
        int(candidates.loc[index, "target_track_id"]) for index in applied_indices
    }
    decision_records = list(selected_records)
    unresolved_records: list[dict[str, object]] = []
    for endpoint in endpoints.sort_values(
        ["last_real_frame", "track_id"], na_position="last", kind="mergesort"
    ).itertuples(index=False):
        source_track_id = int(endpoint.track_id)
        if source_track_id in selected_sources:
            continue
        eligible = bool(endpoint.source_eligible)
        source_candidates = candidates.loc[
            candidates["source_track_id"].astype("Int64") == source_track_id
        ] if not candidates.empty else candidates
        admissible = source_candidates.loc[source_candidates["admissible"].astype(bool)]
        best = (
            admissible.sort_values(
                ["continuation_score", "target_start_frame", "target_track_id"],
                ascending=[False, True, True], kind="mergesort",
            ).iloc[0]
            if not admissible.empty else None
        )
        if not eligible:
            reason = str(endpoint.source_exclusion_reason)
        elif admissible.empty:
            reason = "unresolved_no_candidate"
        elif config.policy == "submission" and set(
            admissible["target_track_id"].astype(int)
        ).issubset(applied_targets):
            reason = "unresolved_no_unused_candidate"
        elif config.policy == "submission":
            reason = "unresolved_global_conflict"
        elif config.policy == "diagnostic":
            reason = "unresolved_global_conflict"
        else:
            reason = "unresolved_conservative_threshold"
        source_end_frame = (
            int(endpoint.last_real_frame) if pd.notna(endpoint.last_real_frame) else pd.NA
        )
        unresolved_records.append({
            "source_track_id": source_track_id,
            "source_end_frame": source_end_frame,
            "source_eligible": eligible,
            "candidate_count": int(len(admissible)),
            "best_target_track_id": (
                int(best["target_track_id"]) if best is not None else pd.NA
            ),
            "best_continuation_score": (
                float(best["continuation_score"]) if best is not None else math.nan
            ),
            "reason": reason,
        })
        decision_records.append({
            "decision_id": "",
            "component_id": (
                component_by_index.get(int(best.name), pd.NA) if best is not None else pd.NA
            ),
            "source_track_id": source_track_id,
            "target_track_id": (
                int(best["target_track_id"]) if best is not None else pd.NA
            ),
            "source_end_frame": source_end_frame,
            "target_start_frame": (
                int(best["target_start_frame"]) if best is not None else pd.NA
            ),
            "gap_frames": int(best["gap_frames"]) if best is not None else pd.NA,
            "decision": reason,
            "policy_phase": config.policy,
            "forced": False,
            "continuation_score": (
                float(best["continuation_score"]) if best is not None else math.nan
            ),
            "assignment_cost": (
                float(best["assignment_cost"]) if best is not None else math.nan
            ),
            "source_rank": int(best["source_rank"]) if best is not None else pd.NA,
            "target_rank": int(best["target_rank"]) if best is not None else pd.NA,
            "source_score_margin": (
                float(best["source_score_margin"]) if best is not None else math.nan
            ),
            "target_score_margin": (
                float(best["target_score_margin"]) if best is not None else math.nan
            ),
            "candidate_count": int(len(admissible)),
            "reason": reason,
        })
    decisions = pd.DataFrame(decision_records, columns=CONTINUATION_DECISION_COLUMNS)
    if not decisions.empty:
        decisions = decisions.sort_values(
            ["source_end_frame", "source_track_id", "target_start_frame", "target_track_id"],
            na_position="last", kind="mergesort",
        ).reset_index(drop=True)
        decisions["decision_id"] = [f"reconciliation-{index:06d}" for index in range(len(decisions))]
    applied = decisions.loc[
        decisions["decision"].astype(str).str.startswith("accepted_")
        | (decisions["decision"] == "forced_best_candidate")
    ].copy()
    unresolved = pd.DataFrame(unresolved_records, columns=UNRESOLVED_ENDING_COLUMNS)
    return AssignmentResult(
        decisions=decisions,
        applied_decisions=applied.reset_index(drop=True),
        unresolved_endings=unresolved,
        component_count=component_count,
        conservative_count=len(conservative_selected),
        forced_count=len(forced_selected),
        conservative_assignment=decisions.loc[
            decisions["policy_phase"] == "conservative"
        ].reset_index(drop=True),
        forced_assignment=decisions.loc[
            decisions["policy_phase"] == "forced_submission"
        ].reset_index(drop=True),
    )
