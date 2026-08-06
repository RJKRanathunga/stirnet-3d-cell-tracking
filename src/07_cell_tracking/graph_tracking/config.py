"""Configuration for sparse graph-based Stage 7 tracking refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GraphMode = Literal["disabled", "shadow", "apply"]


@dataclass(frozen=True, slots=True)
class GraphTrackingConfig:
    """Runtime configuration for graph-based association refinement.

    The defaults are intentionally conservative. They are suitable for shadow-mode
    investigations and should be calibrated on curated failures before apply mode.
    """

    mode: GraphMode = "disabled"

    # Spatial graph construction.
    maximum_neighbors: int = 10
    maximum_radius_um: float = 25.0
    minimum_neighbors: int = 2
    prefer_mutual_neighbors: bool = False

    # Reliable temporal anchor selection.
    anchor_minimum_association_probability: float = 0.60
    anchor_minimum_probability_margin: float = 0.08
    anchor_maximum_position_error_um: float = 3.5
    anchor_allow_boundary: bool = False
    anchor_position_scale_um: float = 3.5
    anchor_margin_scale: float = 0.10

    # Voting and robust consensus.
    minimum_anchor_votes: int = 2
    strong_anchor_votes: int = 4
    vote_kernel_sigma_um: float = 3.0
    vote_inlier_radius_um: float = 4.0
    maximum_vote_dispersion_um: float = 4.0
    consensus_iterations: int = 3

    # Local deformation model.
    enable_similarity_transform: bool = True
    enable_affine_transform: bool = True
    minimum_similarity_anchors: int = 3
    minimum_affine_anchors: int = 4
    affine_maximum_condition_number: float = 1.0e4
    deformation_prediction_weight: float = 1.5

    # Neighbourhood evidence components.
    vote_score_weight: float = 0.45
    vector_score_weight: float = 0.25
    radial_score_weight: float = 0.10
    relative_volume_score_weight: float = 0.10
    consensus_score_weight: float = 0.10
    vector_scale_um: float = 4.0
    radial_scale_um: float = 3.0
    relative_volume_log_scale: float = 0.40

    # Cost integration.
    graph_pair_weight: float = 0.70
    neutral_graph_score: float = 0.50
    maximum_pair_cost_reduction: float = 2.0
    maximum_pair_cost_penalty: float = 2.0
    probability_floor: float = 1.0e-9
    invalid_cost: float = 1.0e6

    # Ambiguity selection and locking.
    ambiguous_probability_threshold: float = 0.45
    ambiguous_margin_threshold: float = 0.08
    ambiguous_cost_gap: float = 0.50
    lock_minimum_association_probability: float = 0.65
    lock_minimum_probability_margin: float = 0.10
    lock_maximum_position_error_um: float = 3.0

    # Boundary entry/exit evidence.
    enable_boundary_entry_exit: bool = True
    boundary_evidence_margin_um: float = 6.0
    outside_vote_minimum_distance_um: float = 0.50
    outside_vote_strong_distance_um: float = 3.0
    boundary_minimum_consensus: float = 0.65
    boundary_minimum_directional_agreement: float = 0.60
    boundary_exit_maximum_miss_cost_reduction: float = 1.50
    boundary_entry_maximum_birth_cost_reduction: float = 1.50
    inside_backward_vote_birth_penalty: float = 1.00
    boundary_coverage_floor: float = 0.15

    # Diagnostics.
    save_spatial_edges: bool = False
    save_all_anchor_votes: bool = True
    save_top_candidates_per_track: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "shadow", "apply"}:
            raise ValueError(f"Unsupported graph mode: {self.mode}")
        if self.maximum_neighbors < 1:
            raise ValueError("maximum_neighbors must be positive")
        if self.maximum_radius_um <= 0:
            raise ValueError("maximum_radius_um must be positive")
        if self.minimum_anchor_votes < 1:
            raise ValueError("minimum_anchor_votes must be positive")
        if self.vote_kernel_sigma_um <= 0 or self.vote_inlier_radius_um <= 0:
            raise ValueError("vote scales must be positive")
        if not 0.0 < self.neutral_graph_score < 1.0:
            raise ValueError("neutral_graph_score must lie strictly between 0 and 1")
