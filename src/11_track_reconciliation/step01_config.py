"""Configuration, decisions, and stable table schemas for Stage 11."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


POLICIES = ("diagnostic", "conservative", "submission")

ENDPOINT_CLASSIFICATION_COLUMNS = (
    "track_id", "first_frame", "last_frame", "observation_count",
    "real_observation_count", "first_real_frame", "last_real_frame",
    "first_observation_index", "last_observation_index",
    "first_real_observation_index", "last_real_observation_index",
    "first_is_boundary", "last_is_boundary", "first_is_virtual",
    "last_is_virtual", "source_eligible", "target_eligible",
    "source_exclusion_reason", "target_exclusion_reason",
)

CONTINUATION_CANDIDATE_COLUMNS = (
    "candidate_id", "sample_id", "source_track_id", "target_track_id",
    "source_end_frame", "target_start_frame", "gap_frames",
    "hard_search_radius_um", "direct_endpoint_distance_um", "admissible",
    "admissibility_reason",
    "forward_predicted_z_um", "forward_predicted_y_um",
    "forward_predicted_x_um", "forward_error_um", "forward_history_count",
    "forward_used_global_motion", "forward_used_relative_velocity",
    "forward_uncertainty_um",
    "backward_predicted_z_um", "backward_predicted_y_um",
    "backward_predicted_x_um", "backward_error_um", "backward_history_count",
    "backward_prediction_available", "bidirectional_disagreement_um",
    "anchor_count", "anchor_track_ids", "anchor_predicted_z_um",
    "anchor_predicted_y_um", "anchor_predicted_x_um",
    "anchor_prediction_error_um", "neighborhood_distance_error_um",
    "local_survival_ratio", "source_reference_volume",
    "target_reference_volume", "volume_log_error", "shape_error",
    "intensity_error", "target_real_observation_count",
    "candidate_quality_score", "stage7_candidate_available",
    "stage7_candidate_distance_um", "stage7_candidate_pair_cost",
    "stage7_candidate_probability", "stage7_candidate_rank",
    "forward_position_score", "backward_position_score",
    "bidirectional_score", "anchor_position_score", "neighborhood_score",
    "temporal_gap_score", "volume_score", "shape_score", "intensity_score",
    "stage7_alternative_score", "uniqueness_score", "continuation_score",
    "assignment_cost", "source_candidate_count", "target_predecessor_count",
    "source_rank", "target_rank", "source_runner_up_score",
    "target_runner_up_score", "source_score_margin", "target_score_margin",
    "mutual_best",
)

CONTINUATION_DECISION_COLUMNS = (
    "decision_id", "component_id", "source_track_id", "target_track_id",
    "source_end_frame", "target_start_frame", "gap_frames", "decision",
    "policy_phase", "forced", "continuation_score", "assignment_cost",
    "source_rank", "target_rank", "source_score_margin",
    "target_score_margin", "candidate_count", "reason",
)

TRACK_ID_REMAP_COLUMNS = (
    "decision_id", "original_segment_track_id", "predecessor_track_id",
    "canonical_track_id", "from_frame", "to_frame", "gap_frames",
    "decision", "policy_phase", "forced", "continuation_score",
    "assignment_cost",
)

UNRESOLVED_ENDING_COLUMNS = (
    "source_track_id", "source_end_frame", "source_eligible",
    "candidate_count", "best_target_track_id", "best_continuation_score",
    "reason",
)

VALIDATION_RESULT_COLUMNS = (
    "check_name", "passed", "severity", "details",
)


@dataclass(frozen=True)
class TrackReconciliationConfig:
    """Provisional, configurable Stage 11 policies and evidence weights.

    The defaults are engineering starting points for the supplied acquisition
    geometry, not calibrated biological probabilities or biological truths.
    """

    policy: str = "submission"
    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    maximum_gap_frames: int = 4
    minimum_source_observations: int = 2
    candidate_radius_base_um: float = 12.0
    candidate_radius_per_additional_gap_um: float = 4.0
    maximum_candidate_radius_um: float = 25.0
    source_motion_history: int = 4
    target_backward_history: int = 3
    feature_history: int = 3
    anchor_search_radius_um: float = 30.0
    maximum_anchor_count: int = 8
    minimum_anchor_count_for_strong_support: int = 2
    boundary_margin_um: float = 2.0
    conservative_minimum_score: float = 0.65
    conservative_minimum_source_margin: float = 0.08
    conservative_minimum_target_margin: float = 0.05
    position_score_scale_um: float = 5.0
    backward_score_scale_um: float = 6.0
    anchor_score_scale_um: float = 5.0
    neighborhood_score_scale_um: float = 4.0
    volume_log_score_scale: float = math.log(2.0)
    shape_score_scale: float = 0.30
    intensity_score_scale: float = 0.35
    gap_decay: float = 0.65
    forward_uncertainty_base_um: float = 1.5
    forward_uncertainty_per_gap_um: float = 1.25
    anchor_support_max_error_um: float = 5.0
    bidirectional_support_max_disagreement_um: float = 5.0
    highly_convincing_forward_error_um: float = 3.0
    probability_floor: float = 1e-9
    invalid_assignment_cost: float = 1e6
    conservative_unmatched_cost: float = -math.log(0.65)
    forced_unmatched_cost: float = 1e4
    forward_position_weight: float = 0.22
    backward_position_weight: float = 0.10
    bidirectional_weight: float = 0.05
    anchor_position_weight: float = 0.16
    neighborhood_weight: float = 0.14
    temporal_gap_weight: float = 0.08
    uniqueness_weight: float = 0.08
    volume_weight: float = 0.07
    shape_weight: float = 0.05
    candidate_quality_weight: float = 0.03
    intensity_weight: float = 0.02
    stage7_alternative_weight: float = 0.03

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        spacing = tuple(float(value) for value in self.voxel_size_zyx_um)
        if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("voxel_size_zyx_um must contain three positive finite values")
        positive_counts = {
            "maximum_gap_frames": self.maximum_gap_frames,
            "minimum_source_observations": self.minimum_source_observations,
            "source_motion_history": self.source_motion_history,
            "target_backward_history": self.target_backward_history,
            "feature_history": self.feature_history,
            "maximum_anchor_count": self.maximum_anchor_count,
            "minimum_anchor_count_for_strong_support": (
                self.minimum_anchor_count_for_strong_support
            ),
        }
        for name, value in positive_counts.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        positive_scales = {
            "candidate_radius_base_um": self.candidate_radius_base_um,
            "maximum_candidate_radius_um": self.maximum_candidate_radius_um,
            "anchor_search_radius_um": self.anchor_search_radius_um,
            "position_score_scale_um": self.position_score_scale_um,
            "backward_score_scale_um": self.backward_score_scale_um,
            "anchor_score_scale_um": self.anchor_score_scale_um,
            "neighborhood_score_scale_um": self.neighborhood_score_scale_um,
            "volume_log_score_scale": self.volume_log_score_scale,
            "shape_score_scale": self.shape_score_scale,
            "intensity_score_scale": self.intensity_score_scale,
            "forward_uncertainty_base_um": self.forward_uncertainty_base_um,
            "forward_uncertainty_per_gap_um": self.forward_uncertainty_per_gap_um,
            "anchor_support_max_error_um": self.anchor_support_max_error_um,
            "bidirectional_support_max_disagreement_um": (
                self.bidirectional_support_max_disagreement_um
            ),
            "highly_convincing_forward_error_um": (
                self.highly_convincing_forward_error_um
            ),
            "probability_floor": self.probability_floor,
            "invalid_assignment_cost": self.invalid_assignment_cost,
            "conservative_unmatched_cost": self.conservative_unmatched_cost,
            "forced_unmatched_cost": self.forced_unmatched_cost,
        }
        for name, value in positive_scales.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = {
            "candidate_radius_per_additional_gap_um": (
                self.candidate_radius_per_additional_gap_um
            ),
            "boundary_margin_um": self.boundary_margin_um,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.maximum_candidate_radius_um < self.candidate_radius_base_um:
            raise ValueError("maximum_candidate_radius_um cannot be below the base radius")
        thresholds = {
            "conservative_minimum_score": self.conservative_minimum_score,
            "conservative_minimum_source_margin": (
                self.conservative_minimum_source_margin
            ),
            "conservative_minimum_target_margin": (
                self.conservative_minimum_target_margin
            ),
            "gap_decay": self.gap_decay,
        }
        for name, value in thresholds.items():
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        weights = {
            name: value for name, value in vars(self).items() if name.endswith("_weight")
        }
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()):
            raise ValueError("score weights must be nonnegative and finite")
        if sum(float(value) for value in weights.values()) <= 0:
            raise ValueError("at least one score weight must be positive")
        if self.minimum_anchor_count_for_strong_support > self.maximum_anchor_count:
            raise ValueError("minimum strong anchor count cannot exceed maximum_anchor_count")

    def as_dict(self) -> dict[str, object]:
        """Return JSON-stable configuration metadata."""

        data = asdict(self)
        data["voxel_size_zyx_um"] = list(self.voxel_size_zyx_um)
        return data
