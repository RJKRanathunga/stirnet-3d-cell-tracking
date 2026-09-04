"""Typed compact representations for multi-frame graph tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TransitionEvidence:
    """In-memory Stage 7 evidence retained before track-state mutation."""

    from_frame: int
    to_frame: int
    source_track_ids: np.ndarray
    source_frames: np.ndarray
    target_detection_indices: np.ndarray
    target_cell_ids: np.ndarray
    pair_cost_matrix: np.ndarray
    safety_invalid: np.ndarray
    distance_matrix: np.ndarray
    association_probabilities: np.ndarray
    track_candidate_probabilities: np.ndarray
    detection_candidate_probabilities: np.ndarray
    track_probability_margins: np.ndarray
    detection_probability_margins: np.ndarray
    miss_costs: np.ndarray
    birth_costs: np.ndarray
    global_only_positions_zyx_um: np.ndarray
    relative_motion_predicted_positions_zyx_um: np.ndarray
    selected_rows: np.ndarray
    selected_columns: np.ndarray
    missed_rows: np.ndarray
    birth_columns: np.ndarray
    global_shift_zyx_um: np.ndarray
    global_shift_confidence: float


@dataclass(frozen=True, slots=True)
class ObservationStore:
    """Columnar observation-node storage; intrinsic features are stored once."""

    table: pd.DataFrame
    frames: np.ndarray
    detection_indices: np.ndarray
    cell_ids: np.ndarray
    positions_zyx_um: np.ndarray
    centroids_zyx_voxel: np.ndarray
    volumes: np.ndarray
    boundary_flags: np.ndarray
    boundary_faces: tuple[tuple[str, ...], ...]
    distances_to_faces_um: np.ndarray
    provisional_track_ids: np.ndarray
    provisional_confidences: np.ndarray
    small_cell_reliability: np.ndarray
    node_by_frame_detection: dict[tuple[int, int], int]
    nodes_by_frame: dict[int, np.ndarray]

    @property
    def node_count(self) -> int:
        return int(len(self.frames))


@dataclass(frozen=True, slots=True)
class SpatialFrameGraph:
    frame: int
    node_indices: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    relative_vectors_zyx_um: np.ndarray
    distances_um: np.ndarray
    unit_directions: np.ndarray
    relative_log_volumes: np.ndarray
    boundary_observability: np.ndarray
    visibility_masks: np.ndarray
    adjacency_indptr: np.ndarray
    adjacency_edge_indices: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(len(self.edge_sources))


@dataclass(frozen=True, slots=True)
class TemporalEdge:
    edge_index: int
    source: int
    target: int
    frame_gap: int
    provisional_selected: bool
    graph_expanded: bool
    hard_safety_valid: bool
    base_stage7_cost: float
    unary_cost: float
    motion_cost: float
    graph_cost: float
    persistent_relation_cost: float
    boundary_cost: float
    displacement_um: float
    global_motion_residual_um: float
    relative_motion_residual_um: float
    volume_log_error: float
    shape_error: float
    intensity_error: float
    provisional_probability: float
    provisional_margin: float

    @property
    def total_cost(self) -> float:
        return float(
            self.unary_cost + self.motion_cost + self.graph_cost
            + self.persistent_relation_cost + self.boundary_cost
        )


@dataclass(frozen=True, slots=True)
class NeighborRelationHistory:
    provisional_track_a: int
    provisional_track_b: int
    observed_frames: np.ndarray
    relative_vectors_zyx_um: np.ndarray
    distances_um: np.ndarray
    relative_log_volume_ratios: np.ndarray
    visibility_masks: np.ndarray
    boundary_coverage: np.ndarray
    persistence_count: int
    vector_median_zyx_um: np.ndarray
    robust_vector_dispersion_um: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PairFactor:
    first_edge: int
    second_edge: int
    cost: float
    factor_type: str
    supporting_relation: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class AmbiguityComponent:
    component_id: int
    node_indices: np.ndarray
    edge_indices: np.ndarray
    seed_nodes: np.ndarray
    minimum_frame: int
    maximum_frame: int


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    window_id: int
    component_id: int
    start_frame: int
    end_frame: int
    commit_start_frame: int
    commit_end_frame: int
    node_indices: np.ndarray
    edge_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class ComponentSolution:
    success: bool
    selected_edge_indices: tuple[int, ...]
    start_events: dict[int, tuple[str, str]]
    end_events: dict[int, tuple[str, str]]
    objective: float
    solver_type: str
    status: str
    message: str
    variable_count: int
    constraint_count: int
    pair_factor_count: int
    mip_gap: float
    iterations: int
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class ExtractedTracks:
    optimized_tracks: pd.DataFrame
    node_to_track_id: np.ndarray
    track_id_map: pd.DataFrame
    selected_successor: dict[int, int]
    selected_predecessor: dict[int, int]


@dataclass(frozen=True, slots=True)
class FourDGraphResult:
    provisional_tracks: pd.DataFrame
    optimized_tracks: pd.DataFrame
    temporal_edges: pd.DataFrame
    assignment_changes: pd.DataFrame
    boundary_events: pd.DataFrame
    window_summary: pd.DataFrame
    component_summary: pd.DataFrame
    solver_diagnostics: pd.DataFrame
    provisional_to_optimized_track_map: pd.DataFrame
    metadata: dict[str, Any]
    summary: dict[str, Any]
    debug_artifacts: dict[str, dict[str, np.ndarray]] = field(repr=False)
    selected_edge_indices: tuple[int, ...] = field(repr=False)
    node_to_track_id: np.ndarray = field(repr=False)
