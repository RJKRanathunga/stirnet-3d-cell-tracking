"""Typed in-memory representations used by graph tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SpatialGraph:
    frame: int
    node_ids: np.ndarray
    positions_zyx_um: np.ndarray
    volumes: np.ndarray
    touches_boundary: np.ndarray
    boundary_faces: tuple[str, ...]
    edge_source_indices: np.ndarray
    edge_target_indices: np.ndarray
    edge_vectors_zyx_um: np.ndarray
    edge_distances_um: np.ndarray
    adjacency_indptr: np.ndarray
    adjacency_edge_indices: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.positions_zyx_um.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_source_indices.size)


@dataclass(frozen=True, slots=True)
class TemporalAnchor:
    track_id: int
    source_node_index: int
    target_node_index: int
    association_probability: float
    probability_margin: float
    position_error_um: float
    reliability: float


@dataclass(frozen=True, slots=True)
class VoteConsensus:
    position_zyx_um: np.ndarray
    dispersion_um: float
    inlier_mask: np.ndarray
    inlier_weight_fraction: float
    total_weight: float


@dataclass(frozen=True, slots=True)
class LocalTransform:
    transform_type: str
    linear_matrix: np.ndarray
    translation_zyx_um: np.ndarray
    condition_number: float
    residual_um: float

    def apply(self, point_zyx_um: np.ndarray) -> np.ndarray:
        point = np.asarray(point_zyx_um, dtype=float)
        return point @ self.linear_matrix.T + self.translation_zyx_um


@dataclass(frozen=True, slots=True)
class VolumePointClassification:
    inside: bool
    nearest_face: str
    signed_distance_to_volume_um: float
    outside_distance_um: float
    distances_to_faces_um: np.ndarray
    outside_axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GraphRefinementResult:
    """Graph refinement output returned to the Stage 7 orchestrator."""

    assignment: dict[str, Any]
    base_assignment: dict[str, Any]
    graph_assignment: dict[str, Any]
    candidate_evidence: pd.DataFrame
    anchor_votes: pd.DataFrame
    boundary_hypotheses: pd.DataFrame
    refinement_events: pd.DataFrame
    transition_summary: pd.DataFrame
    metadata: dict[str, Any]
