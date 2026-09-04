"""End-to-end post-provisional multi-frame Stage 7 graph tracker."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..geometry import FACE_NAMES
from .ambiguity import ambiguity_seeds
from .components import build_ambiguity_components
from .config import FourDGraphConfig
from .diagnostics import (
    empty_assignment_changes,
    empty_boundary_events,
    empty_component_summary,
    empty_solver_diagnostics,
    empty_temporal_edges,
    empty_window_summary,
    table,
)
from .factors import build_pair_factors
from .iterative_solver import solve_component_iteratively
from .milp_model import solve_component_milp
from .observations import build_observations
from .relation_history import build_relation_histories
from .schemas import (
    ASSIGNMENT_CHANGE_COLUMNS,
    BOUNDARY_EVENT_COLUMNS,
    COMPONENT_SUMMARY_COLUMNS,
    SOLVER_DIAGNOSTIC_COLUMNS,
    TEMPORAL_EDGE_COLUMNS,
    WINDOW_SUMMARY_COLUMNS,
)
from .spatial_relations import build_spatial_relations
from .temporal_candidates import build_temporal_candidates
from .track_extraction import extract_tracks
from .types import (
    AmbiguityComponent,
    ComponentSolution,
    FourDGraphResult,
    ObservationStore,
    PairFactor,
    TemporalEdge,
    TransitionEvidence,
)
from .validation import validate_optimized_solution
from .windowing import build_windows


def _solver_row(
    solution: ComponentSolution,
    *,
    component_id: int,
    window_id: int,
    node_count: int,
    edge_count: int,
) -> dict[str, object]:
    return {
        "component_id": component_id,
        "window_id": window_id,
        "solver_type": solution.solver_type,
        "status": solution.status,
        "success": solution.success,
        "message": solution.message,
        "node_count": node_count,
        "edge_count": edge_count,
        "pair_factor_count": solution.pair_factor_count,
        "variable_count": solution.variable_count,
        "constraint_count": solution.constraint_count,
        "objective": solution.objective,
        "mip_gap": solution.mip_gap,
        "iterations": solution.iterations,
        "runtime_seconds": solution.runtime_seconds,
    }


def _solve(
    *,
    component: AmbiguityComponent,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    factors: list[PairFactor],
    selected: set[int],
    config: FourDGraphConfig,
) -> ComponentSolution:
    oversized = (
        len(component.node_indices) > config.maximum_exact_component_nodes
        or len(component.edge_indices) > config.maximum_exact_component_edges
        or len(factors) > config.maximum_exact_pair_factors
    )
    if oversized:
        return solve_component_iteratively(
            component=component,
            edges=edges,
            observations=observations,
            pair_factors=factors,
            provisional_selected_edges=selected,
            config=config,
        )
    return solve_component_milp(
        component=component,
        edges=edges,
        observations=observations,
        pair_factors=factors,
        provisional_selected_edges=selected,
        config=config,
    )


def _event_cost(
    node: int,
    event_type: str,
    face: str,
    observations: ObservationStore,
    config: FourDGraphConfig,
) -> tuple[float, float, float]:
    if not face or face not in FACE_NAMES:
        return math.nan, math.nan, math.nan
    face_index = FACE_NAMES.index(face)
    distance = float(observations.distances_to_faces_um[node, face_index])
    cost = config.boundary_event_base_cost + distance / max(config.boundary_margin_um, 1e-9)
    return cost, distance, math.nan


def _boundary_table(
    *,
    start_events: dict[int, tuple[str, str, int, int, str]],
    end_events: dict[int, tuple[str, str, int, int, str]],
    observations: ObservationStore,
    node_to_track: np.ndarray,
    config: FourDGraphConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for events in (start_events, end_events):
        for node, (event_type, face, component, window, solver) in sorted(events.items()):
            if event_type not in {"boundary_entry", "boundary_exit"}:
                continue
            cost, distance, support = _event_cost(node, event_type, face, observations, config)
            rows.append({
                "node_index": node,
                "frame": int(observations.frames[node]),
                "detection_index": int(observations.detection_indices[node]),
                "optimized_track_id": int(node_to_track[node]),
                "event_type": event_type,
                "boundary_face": face,
                "cost": cost,
                "distance_to_face_um": distance,
                "directional_support": support,
                "component_id": component,
                "window_id": window,
                "solver_type": solver,
            })
    return table(rows, BOUNDARY_EVENT_COLUMNS) if rows else empty_boundary_events()


def _edge_table(
    *,
    edges: list[TemporalEdge],
    observations: ObservationStore,
    selected: set[int],
    component_by_edge: dict[int, int],
    window_by_edge: dict[int, int],
    solver_by_edge: dict[int, str],
    factor_contributions: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for edge in edges:
        contributions = factor_contributions.get(edge.edge_index, {})
        trajectory = contributions.get("trajectory", 0.0)
        spatial = contributions.get("spatial", 0.0)
        persistent = contributions.get("persistent", 0.0)
        optimized = edge.edge_index in selected
        rows.append({
            "edge_index": edge.edge_index,
            "source_node": edge.source,
            "target_node": edge.target,
            "source_frame": int(observations.frames[edge.source]),
            "target_frame": int(observations.frames[edge.target]),
            "source_detection_index": int(observations.detection_indices[edge.source]),
            "target_detection_index": int(observations.detection_indices[edge.target]),
            "frame_gap": edge.frame_gap,
            "provisional_selected": edge.provisional_selected,
            "optimized_selected": optimized,
            "graph_expanded": edge.graph_expanded,
            "hard_safety_valid": edge.hard_safety_valid,
            "base_stage7_cost": edge.base_stage7_cost,
            "unary_cost": edge.unary_cost,
            "motion_cost": edge.motion_cost + trajectory,
            "graph_cost": edge.graph_cost + spatial,
            "persistent_relation_cost": edge.persistent_relation_cost + persistent,
            "boundary_cost": edge.boundary_cost,
            "total_effective_cost": edge.total_cost + trajectory + spatial + persistent,
            "displacement_um": edge.displacement_um,
            "global_motion_residual_um": edge.global_motion_residual_um,
            "relative_motion_residual_um": edge.relative_motion_residual_um,
            "volume_log_error": edge.volume_log_error,
            "shape_error": edge.shape_error,
            "intensity_error": edge.intensity_error,
            "provisional_probability": edge.provisional_probability,
            "provisional_margin": edge.provisional_margin,
            "component_id": component_by_edge.get(edge.edge_index, -1),
            "window_id": window_by_edge.get(edge.edge_index, -1),
            "solver_type": solver_by_edge.get(edge.edge_index, "fixed_provisional"),
            "changed_from_provisional": bool(optimized != edge.provisional_selected),
        })
    return table(rows, TEMPORAL_EDGE_COLUMNS) if rows else empty_temporal_edges()


def _change_table(
    *,
    edges: list[TemporalEdge],
    selected: set[int],
    observations: ObservationStore,
    node_to_track: np.ndarray,
    component_by_edge: dict[int, int],
    window_by_edge: dict[int, int],
    solver_by_edge: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for edge in edges:
        optimized = edge.edge_index in selected
        if optimized == edge.provisional_selected:
            continue
        rows.append({
            "change_type": "continuation_added" if optimized else "provisional_edge_rejected",
            "source_node": edge.source,
            "target_node": edge.target,
            "source_frame": int(observations.frames[edge.source]),
            "target_frame": int(observations.frames[edge.target]),
            "source_detection_index": int(observations.detection_indices[edge.source]),
            "target_detection_index": int(observations.detection_indices[edge.target]),
            "provisional_track_id": int(observations.provisional_track_ids[edge.source]),
            "optimized_track_id": int(node_to_track[edge.source]),
            "component_id": component_by_edge.get(edge.edge_index, -1),
            "window_id": window_by_edge.get(edge.edge_index, -1),
            "solver_type": solver_by_edge.get(edge.edge_index, "fixed_provisional"),
        })
    return table(rows, ASSIGNMENT_CHANGE_COLUMNS) if rows else empty_assignment_changes()


def _debug_artifacts(
    *,
    spatial_graphs,
    factors: list[PairFactor],
    histories,
) -> dict[str, dict[str, np.ndarray]]:
    spatial_frames: list[int] = []
    spatial_sources: list[int] = []
    spatial_targets: list[int] = []
    spatial_vectors: list[np.ndarray] = []
    spatial_distances: list[float] = []
    for frame, graph in sorted(spatial_graphs.items()):
        for index, (source, target) in enumerate(zip(graph.edge_sources, graph.edge_targets)):
            spatial_frames.append(frame)
            spatial_sources.append(int(source))
            spatial_targets.append(int(target))
            spatial_vectors.append(graph.relative_vectors_zyx_um[index])
            spatial_distances.append(float(graph.distances_um[index]))
    unique_factors = sorted({
        (factor.first_edge, factor.second_edge, factor.factor_type, factor.cost)
        for factor in factors
    })
    relation_a: list[int] = []
    relation_b: list[int] = []
    relation_frame: list[int] = []
    relation_vectors: list[np.ndarray] = []
    relation_distances: list[float] = []
    relation_volumes: list[float] = []
    relation_confidence: list[float] = []
    for history in histories.values():
        for index, frame in enumerate(history.observed_frames):
            relation_a.append(history.provisional_track_a)
            relation_b.append(history.provisional_track_b)
            relation_frame.append(int(frame))
            relation_vectors.append(history.relative_vectors_zyx_um[index])
            relation_distances.append(float(history.distances_um[index]))
            relation_volumes.append(float(history.relative_log_volume_ratios[index]))
            relation_confidence.append(float(history.confidence))
    return {
        "graph4d_spatial_edges.npz": {
            "frame": np.asarray(spatial_frames, dtype=np.int32),
            "source_node": np.asarray(spatial_sources, dtype=np.int32),
            "target_node": np.asarray(spatial_targets, dtype=np.int32),
            "relative_vector_zyx_um": np.asarray(spatial_vectors, dtype=float).reshape(-1, 3),
            "distance_um": np.asarray(spatial_distances, dtype=float),
        },
        "graph4d_pair_factors.npz": {
            "first_edge": np.asarray([value[0] for value in unique_factors], dtype=np.int32),
            "second_edge": np.asarray([value[1] for value in unique_factors], dtype=np.int32),
            "factor_type": np.asarray([value[2] for value in unique_factors], dtype="U32"),
            "cost": np.asarray([value[3] for value in unique_factors], dtype=float),
        },
        "graph4d_relation_histories.npz": {
            "provisional_track_a": np.asarray(relation_a, dtype=np.int64),
            "provisional_track_b": np.asarray(relation_b, dtype=np.int64),
            "frame": np.asarray(relation_frame, dtype=np.int32),
            "relative_vector_zyx_um": np.asarray(relation_vectors, dtype=float).reshape(-1, 3),
            "distance_um": np.asarray(relation_distances, dtype=float),
            "relative_log_volume_ratio": np.asarray(relation_volumes, dtype=float),
            "confidence": np.asarray(relation_confidence, dtype=float),
        },
    }


def run_four_d_graph_tracking(
    *,
    time_frames: list[pd.DataFrame],
    provisional_tracks: pd.DataFrame,
    transition_evidence: tuple[TransitionEvidence, ...],
    spatial_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
    config: FourDGraphConfig | None = None,
) -> FourDGraphResult:
    """Optimize the complete provisional sequence using bounded 4D components."""

    config = config or FourDGraphConfig()
    phase_started = perf_counter()
    runtimes: dict[str, float] = {}
    observations = build_observations(
        time_frames=time_frames,
        provisional_tracks=provisional_tracks,
        spatial_shape_zyx=spatial_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
        config=config,
    )
    spatial_graphs = build_spatial_relations(observations, config)
    runtimes["graph_construction"] = perf_counter() - phase_started

    started = perf_counter()
    edges = build_temporal_candidates(
        observations=observations,
        transition_evidence=transition_evidence,
        spatial_graphs=spatial_graphs,
        config=config,
    )
    histories = build_relation_histories(
        observations=observations, spatial_graphs=spatial_graphs, config=config
    )
    seeds = ambiguity_seeds(observations=observations, edges=edges, config=config)
    components = build_ambiguity_components(
        seeds=seeds,
        observations=observations,
        edges=edges,
        spatial_graphs=spatial_graphs,
        config=config,
    )
    runtimes["candidate_generation"] = perf_counter() - started

    selected = {edge.edge_index for edge in edges if edge.provisional_selected}
    provisional_selected = set(selected)
    start_events: dict[int, tuple[str, str, int, int, str]] = {}
    end_events: dict[int, tuple[str, str, int, int, str]] = {}
    component_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    component_by_edge: dict[int, int] = {}
    window_by_edge: dict[int, int] = {}
    solver_by_edge: dict[int, str] = {}
    factor_contributions: dict[int, dict[str, float]] = {}
    debug_factors: list[PairFactor] = []
    exact_count = 0
    fallback_count = 0
    failures = 0
    total_windows = 0
    factor_runtime = 0.0
    exact_runtime = 0.0
    fallback_runtime = 0.0

    for component in components:
        component_original = set(selected) & set(int(value) for value in component.edge_indices)
        windows = build_windows(
            component, observations=observations, edges=edges, config=config
        )
        total_windows += len(windows)
        component_success = True
        component_solver_types: list[str] = []
        component_message = ""
        component_factors: list[PairFactor] = []
        component_selected_after: set[int] = set(component_original)
        component_start_events: dict[int, tuple[str, str, int, int, str]] = {}
        component_end_events: dict[int, tuple[str, str, int, int, str]] = {}

        for window in windows:
            window_component = AmbiguityComponent(
                component_id=component.component_id,
                node_indices=window.node_indices,
                edge_indices=window.edge_indices,
                seed_nodes=np.intersect1d(component.seed_nodes, window.node_indices),
                minimum_frame=window.start_frame,
                maximum_frame=window.end_frame,
            )
            factor_started = perf_counter()
            factors = build_pair_factors(
                component=window_component,
                edges=edges,
                observations=observations,
                spatial_graphs=spatial_graphs,
                histories=histories,
                config=config,
            )
            factor_runtime += perf_counter() - factor_started
            component_factors.extend(factors)
            debug_factors.extend(factors)
            solution = _solve(
                component=window_component,
                edges=edges,
                observations=observations,
                factors=factors,
                selected=selected,
                config=config,
            )
            component_solver_types.append(solution.solver_type)
            if solution.solver_type == "exact_milp":
                exact_runtime += solution.runtime_seconds
            else:
                fallback_runtime += solution.runtime_seconds
            solver_rows.append(_solver_row(
                solution,
                component_id=component.component_id,
                window_id=window.window_id,
                node_count=len(window.node_indices),
                edge_count=len(window.edge_indices),
            ))
            commit_nodes = {
                int(node) for node in window.node_indices
                if window.commit_start_frame <= int(observations.frames[node]) <= window.commit_end_frame
            }
            conflicts = 0
            if solution.success:
                incident = {
                    int(index) for index in window.edge_indices
                    if edges[int(index)].source in commit_nodes or edges[int(index)].target in commit_nodes
                }
                component_selected_after.difference_update(incident)
                component_selected_after.update(
                    index for index in solution.selected_edge_indices
                    if edges[index].source in commit_nodes or edges[index].target in commit_nodes
                )
                for node, (event_type, face) in solution.start_events.items():
                    if node in commit_nodes:
                        component_start_events[node] = (
                            event_type, face, component.component_id,
                            window.window_id, solution.solver_type,
                        )
                for node, (event_type, face) in solution.end_events.items():
                    if node in commit_nodes:
                        component_end_events[node] = (
                            event_type, face, component.component_id,
                            window.window_id, solution.solver_type,
                        )
            else:
                component_success = False
                component_message = solution.message
                failures += 1
            window_rows.append({
                "window_id": window.window_id,
                "component_id": component.component_id,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "commit_start_frame": window.commit_start_frame,
                "commit_end_frame": window.commit_end_frame,
                "node_count": len(window.node_indices),
                "edge_count": len(window.edge_indices),
                "solver_type": solution.solver_type,
                "status": solution.status,
                "objective": solution.objective,
                "runtime_seconds": solution.runtime_seconds,
                "conflicts": conflicts,
            })

        # Reject inconsistent overlapping commits and preserve only this
        # component's provisional assignment, leaving successful components intact.
        incoming: dict[int, int] = {}
        outgoing: dict[int, int] = {}
        for index in component_selected_after:
            edge = edges[index]
            incoming[edge.target] = incoming.get(edge.target, 0) + 1
            outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
        if any(value > 1 for value in incoming.values()) or any(
            value > 1 for value in outgoing.values()
        ):
            component_success = False
            component_message = "overlapping window conflict"
            failures += 1
        component_indices = set(int(value) for value in component.edge_indices)
        if component_success:
            selected.difference_update(component_indices)
            selected.update(component_selected_after)
            start_events.update(component_start_events)
            end_events.update(component_end_events)
        else:
            selected.difference_update(component_indices)
            selected.update(component_original)

        solver_type = (
            "iterative_fallback" if "iterative_fallback" in component_solver_types
            else "exact_milp"
        )
        if solver_type == "exact_milp":
            exact_count += 1
        else:
            fallback_count += 1
        for index in component_indices:
            component_by_edge[index] = component.component_id
            solver_by_edge[index] = solver_type if component_success else "provisional_failure_fallback"
        for window in windows:
            for index in window.edge_indices:
                window_by_edge[int(index)] = window.window_id
        for factor in component_factors:
            if factor.first_edge in selected and factor.second_edge in selected:
                if factor.factor_type == "trajectory":
                    category = "trajectory"
                elif "persistent" in factor.factor_type:
                    category = "persistent"
                else:
                    category = "spatial"
                for index in (factor.first_edge, factor.second_edge):
                    values = factor_contributions.setdefault(index, {})
                    values[category] = values.get(category, 0.0) + factor.cost / 2.0
        component_rows.append({
            "component_id": component.component_id,
            "minimum_frame": component.minimum_frame,
            "maximum_frame": component.maximum_frame,
            "node_count": len(component.node_indices),
            "temporal_edge_count": len(component.edge_indices),
            "ambiguity_seed_count": len(component.seed_nodes),
            "window_count": len(windows),
            "solver_type": solver_type,
            "status": "optimized" if component_success else "provisional_fallback",
            "selected_edge_count": len(selected & component_indices),
            "changed_edge_count": len(
                (selected & component_indices) ^ component_original
            ),
            "fallback_used": solver_type != "exact_milp" or not component_success,
            "failure_message": component_message,
        })

    runtimes["factor_construction"] = factor_runtime
    runtimes["exact_optimization"] = exact_runtime
    runtimes["fallback_optimization"] = fallback_runtime
    extraction_started = perf_counter()
    extracted = extract_tracks(
        observations=observations,
        edges=edges,
        selected_edge_indices=selected,
        provisional_tracks=provisional_tracks,
    )
    boundary_events = _boundary_table(
        start_events=start_events,
        end_events=end_events,
        observations=observations,
        node_to_track=extracted.node_to_track_id,
        config=config,
    )
    runtimes["track_extraction"] = perf_counter() - extraction_started

    validation_started = perf_counter()
    validation = validate_optimized_solution(
        observations=observations,
        edges=edges,
        selected_edge_indices=selected,
        optimized_tracks=extracted.optimized_tracks,
        node_to_track_id=extracted.node_to_track_id,
        boundary_events=boundary_events,
        track_id_map=extracted.track_id_map,
        config=config,
    )
    runtimes["validation"] = perf_counter() - validation_started

    temporal_table = _edge_table(
        edges=edges,
        observations=observations,
        selected=selected,
        component_by_edge=component_by_edge,
        window_by_edge=window_by_edge,
        solver_by_edge=solver_by_edge,
        factor_contributions=factor_contributions,
    )
    changes = _change_table(
        edges=edges,
        selected=selected,
        observations=observations,
        node_to_track=extracted.node_to_track_id,
        component_by_edge=component_by_edge,
        window_by_edge=window_by_edge,
        solver_by_edge=solver_by_edge,
    )
    window_summary = (
        table(window_rows, WINDOW_SUMMARY_COLUMNS) if window_rows else empty_window_summary()
    )
    component_summary = (
        table(component_rows, COMPONENT_SUMMARY_COLUMNS) if component_rows else empty_component_summary()
    )
    solver_diagnostics = (
        table(solver_rows, SOLVER_DIAGNOSTIC_COLUMNS) if solver_rows else empty_solver_diagnostics()
    )
    selected_gap_count = sum(edges[index].frame_gap > 1 for index in selected)
    selected_expanded = sum(edges[index].graph_expanded for index in selected)
    selected_incoming_nodes = {edges[index].target for index in selected}
    selected_outgoing_nodes = {edges[index].source for index in selected}
    sequence_first_frame = int(observations.frames.min()) if observations.node_count else 0
    sequence_last_frame = int(observations.frames.max()) if observations.node_count else 0
    optimized_birth_count = 0
    optimized_death_count = 0
    for node in range(observations.node_count):
        if node not in selected_incoming_nodes and int(observations.frames[node]) > sequence_first_frame:
            event_type = start_events.get(node, ("birth", "", -1, -1, ""))[0]
            optimized_birth_count += int(event_type == "birth")
        if node not in selected_outgoing_nodes and int(observations.frames[node]) < sequence_last_frame:
            event_type = end_events.get(node, ("death", "", -1, -1, ""))[0]
            optimized_death_count += int(event_type == "death")
    id_remaps = int((
        extracted.track_id_map["provisional_track_id"]
        != extracted.track_id_map["optimized_track_id"]
    ).sum()) if not extracted.track_id_map.empty else 0
    observation_array_bytes = sum(
        array.nbytes for array in (
            observations.frames,
            observations.detection_indices,
            observations.cell_ids,
            observations.positions_zyx_um,
            observations.centroids_zyx_voxel,
            observations.volumes,
            observations.boundary_flags,
            observations.distances_to_faces_um,
            observations.provisional_track_ids,
            observations.provisional_confidences,
            observations.small_cell_reliability,
        )
    )
    spatial_array_bytes = sum(
        array.nbytes
        for graph in spatial_graphs.values()
        for array in (
            graph.node_indices,
            graph.edge_sources,
            graph.edge_targets,
            graph.relative_vectors_zyx_um,
            graph.distances_um,
            graph.unit_directions,
            graph.relative_log_volumes,
            graph.boundary_observability,
            graph.visibility_masks,
            graph.adjacency_indptr,
            graph.adjacency_edge_indices,
        )
    )
    metadata: dict[str, Any] = {
        "algorithm": "windowed_4d",
        "configuration": config,
        "provisional_track_count": int(provisional_tracks["track_id"].nunique()),
        "optimized_track_count": int(extracted.optimized_tracks["track_id"].nunique()),
        "observation_node_count": observations.node_count,
        "spatial_edge_count": int(sum(graph.edge_count for graph in spatial_graphs.values())),
        "temporal_candidate_count": len(edges),
        "graph_expanded_candidate_count": sum(edge.graph_expanded for edge in edges),
        "relation_history_count": len(histories),
        "component_count": len(components),
        "exact_milp_component_count": exact_count,
        "iterative_fallback_component_count": fallback_count,
        "total_window_count": total_windows,
        "selected_adjacent_edges": len(selected) - selected_gap_count,
        "selected_gap_edges": selected_gap_count,
        "selected_graph_expanded_edges": selected_expanded,
        "optimized_births": optimized_birth_count,
        "optimized_deaths": optimized_death_count,
        "optimized_boundary_entries": int((boundary_events["event_type"] == "boundary_entry").sum()) if not boundary_events.empty else 0,
        "optimized_boundary_exits": int((boundary_events["event_type"] == "boundary_exit").sum()) if not boundary_events.empty else 0,
        "assignment_changes": len(changes),
        "provisional_to_optimized_id_remap_count": id_remaps,
        "solver_failures": failures,
        "validation_status": "valid" if validation["valid"] else "invalid",
        "runtime_seconds_by_phase": runtimes,
        "estimated_compact_array_bytes": int(
            observation_array_bytes + spatial_array_bytes
        ),
        "global_motion_provenance": "provisional_stage7_prior",
    }
    summary = {
        "candidate_recall": float(
            sum(edge.provisional_selected for edge in edges) / max(len(provisional_selected), 1)
        ),
        "changed_edge_count": len(changes),
        "selected_gap_edges": selected_gap_count,
        "graph_expanded_selected_edges": selected_expanded,
        "boundary_entries": metadata["optimized_boundary_entries"],
        "boundary_exits": metadata["optimized_boundary_exits"],
        "track_fragmentation": int(extracted.optimized_tracks["track_id"].nunique()),
        "solver_fallback_rate": float(fallback_count / max(len(components), 1)),
        "runtime_seconds_by_phase": runtimes,
    }
    debug_artifacts = (
        _debug_artifacts(
            spatial_graphs=spatial_graphs,
            factors=debug_factors,
            histories=histories,
        )
        if config.save_detailed_debug_artifacts
        else {}
    )
    return FourDGraphResult(
        provisional_tracks=provisional_tracks.copy(),
        optimized_tracks=extracted.optimized_tracks,
        temporal_edges=temporal_table,
        assignment_changes=changes,
        boundary_events=boundary_events,
        window_summary=window_summary,
        component_summary=component_summary,
        solver_diagnostics=solver_diagnostics,
        provisional_to_optimized_track_map=extracted.track_id_map,
        metadata=metadata,
        summary=summary,
        debug_artifacts=debug_artifacts,
        selected_edge_indices=tuple(sorted(selected)),
        node_to_track_id=extracted.node_to_track_id,
    )
