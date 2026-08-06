"""Configuration for the multi-frame Stage 7 continuation graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


Ablation = Literal[
    "unary",
    "trajectory",
    "spatial",
    "persistent",
    "expansion",
    "boundary",
    "complete",
]


@dataclass(frozen=True, slots=True)
class FourDGraphConfig:
    """Conservative, deterministic controls for the windowed 4D solver."""

    spatial_maximum_neighbors: int = 8
    spatial_maximum_radius_um: float = 25.0
    temporal_candidates_per_source: int = 4
    temporal_candidates_per_target: int = 4
    locally_competitive_cost_delta: float = 1.5
    maximum_gap_frames: int = 2
    adjacent_maximum_distance_um: float = 28.0
    gap_maximum_distance_um_per_frame: float = 18.0
    graph_expansion_radius_um: float = 6.0
    enable_candidate_expansion: bool = True

    relation_history_frames: int = 4
    minimum_persistent_relation_frames: int = 2
    relation_vector_scale_um: float = 4.0
    relation_distance_scale_um: float = 3.0
    relation_volume_log_scale: float = 0.45

    unary_weight: float = 1.0
    trajectory_weight: float = 0.65
    spatial_weight: float = 0.55
    persistent_relation_weight: float = 0.45
    gap_penalty_per_missing_frame: float = 1.25
    acceleration_scale_um: float = 4.0
    spatial_vector_scale_um: float = 4.0
    compatible_factor_reward: float = 0.35
    small_cell_volume_threshold: float = 250.0
    small_cell_feature_weight: float = 0.35

    generic_birth_cost: float = 3.25
    generic_death_cost: float = 3.25
    boundary_event_base_cost: float = 1.25
    boundary_margin_um: float = 6.0
    boundary_direction_weight: float = 1.0
    enable_boundary_events: bool = True

    ambiguous_probability_threshold: float = 0.45
    ambiguous_margin_threshold: float = 0.08
    ambiguous_cost_delta: float = 0.50
    graph_inconsistency_threshold: float = 6.0
    ambiguity_spatial_hops: int = 1
    ambiguity_temporal_hops: int = 2

    window_size: int = 7
    lookback_frames: int = 3
    lookahead_frames: int = 3
    maximum_exact_component_nodes: int = 80
    maximum_exact_component_edges: int = 400
    maximum_exact_pair_factors: int = 2000
    maximum_factor_candidates_per_node: int = 3
    maximum_variables: int = 12000
    maximum_constraints: int = 40000
    solver_time_limit_seconds: float = 30.0
    solver_relative_mip_gap: float = 0.0
    iterative_solver_iterations: int = 5
    deterministic_tie_break: float = 1.0e-8

    save_detailed_debug_artifacts: bool = False
    ablation: Ablation = "complete"

    def __post_init__(self) -> None:
        if self.spatial_maximum_neighbors < 1:
            raise ValueError("spatial_maximum_neighbors must be positive")
        if self.spatial_maximum_radius_um <= 0:
            raise ValueError("spatial_maximum_radius_um must be positive")
        if self.temporal_candidates_per_source < 1 or self.temporal_candidates_per_target < 1:
            raise ValueError("temporal candidate limits must be positive")
        if self.maximum_gap_frames not in {1, 2, 3}:
            raise ValueError("maximum_gap_frames must be 1, 2, or 3")
        if self.relation_history_frames < 2:
            raise ValueError("relation_history_frames must be at least two")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least three")
        if self.lookback_frames + self.lookahead_frames + 1 != self.window_size:
            raise ValueError("lookback_frames + lookahead_frames + 1 must equal window_size")
        if self.solver_time_limit_seconds <= 0:
            raise ValueError("solver_time_limit_seconds must be positive")
        if self.iterative_solver_iterations < 1:
            raise ValueError("iterative_solver_iterations must be positive")
        if self.maximum_factor_candidates_per_node < 1:
            raise ValueError("maximum_factor_candidates_per_node must be positive")
        if self.ablation not in {
            "unary", "trajectory", "spatial", "persistent", "expansion",
            "boundary", "complete",
        }:
            raise ValueError(f"Unsupported 4D ablation: {self.ablation}")

    def for_ablation(self, name: Ablation) -> "FourDGraphConfig":
        """Return a documented cumulative ablation without mutating this config."""

        ranks = {
            "unary": 0,
            "trajectory": 1,
            "spatial": 2,
            "persistent": 3,
            "expansion": 4,
            "boundary": 5,
            "complete": 6,
        }
        rank = ranks[name]
        return replace(
            self,
            ablation=name,
            trajectory_weight=self.trajectory_weight if rank >= 1 else 0.0,
            spatial_weight=self.spatial_weight if rank >= 2 else 0.0,
            persistent_relation_weight=(
                self.persistent_relation_weight if rank >= 3 else 0.0
            ),
            enable_candidate_expansion=self.enable_candidate_expansion and rank >= 4,
            enable_boundary_events=self.enable_boundary_events and rank >= 5,
        )
