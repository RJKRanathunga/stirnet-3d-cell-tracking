"""Configuration, decisions, and stable table schemas for Stage 10."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


CONFIRMED = "confirmed"
PROBABLE = "probable"
REJECTED = "rejected"
REJECTED_CONFLICT = "rejected_conflict"

DIVISION_CANDIDATE_COLUMNS = (
    "candidate_id", "sample_id",
    "parent_track_id", "parent_end_frame", "parent_observation_count",
    "child_birth_frame", "child_track_a", "child_track_b",
    "parent_is_boundary", "parent_is_virtual",
    "child_a_is_boundary", "child_b_is_boundary",
    "child_a_is_virtual", "child_b_is_virtual",
    "transition_overlaps_virtual_merge",
    "predicted_parent_z", "predicted_parent_y", "predicted_parent_x",
    "prediction_history_count", "prediction_used_velocity",
    "parent_reference_volume", "parent_final_volume",
    "parent_recent_volume_count", "parent_final_volume_ratio",
    "parent_recent_volume_slope",
    "child_a_volume", "child_b_volume", "combined_child_volume",
    "combined_volume_ratio", "combined_volume_relative_error",
    "child_a_volume_fraction", "child_b_volume_fraction",
    "child_a_volume_voxels", "child_b_volume_voxels",
    "child_a_distance_um", "child_b_distance_um",
    "weighted_child_centroid_z", "weighted_child_centroid_y",
    "weighted_child_centroid_x", "weighted_centroid_error_um",
    "birth_separation_um", "birth_masks_available", "tiny_fragment_child",
    "child_a_observation_count_in_window",
    "child_b_observation_count_in_window", "minimum_child_observation_count",
    "both_children_persist", "future_window_truncated",
    "child_a_last_observed_frame_in_window",
    "child_b_last_observed_frame_in_window",
    "paired_future_frame_count", "last_available_separation_um",
    "maximum_separation_um", "minimum_separation_um", "separation_gain_um",
    "separation_slope_um_per_frame", "separation_increase_count",
    "separation_decrease_count",
    "parent_final_integrated_intensity_ratio",
    "parent_final_mask_mean_ratio", "parent_final_core_intensity_ratio",
    "parent_final_background_corrected_ratio",
    "parent_final_core_frame_ratio_change",
    "child_birth_intensity_ratio", "child_delayed_max_intensity_ratio",
    "child_delayed_intensity_slope", "child_intensity_peak_relative_frame",
    "child_delayed_paired_frame_count",
    "spatial_score", "combined_volume_score", "persistence_score",
    "divergence_score", "parent_intensity_score",
    "child_brightening_score", "child_a_preferred_quality_pass",
    "child_b_preferred_quality_pass", "artifact_penalty",
    "continuation_position_score_a", "continuation_volume_score_a",
    "continuation_shape_score_a", "continuation_score_a",
    "continuation_position_score_b", "continuation_volume_score_b",
    "continuation_shape_score_b", "continuation_score_b",
    "best_continuation_score", "division_score", "division_margin",
    "preliminary_decision", "decision", "rejection_reason",
    "secondary_reasons", "division_event_id",
)

DIVISION_EVENT_COLUMNS = (
    "division_event_id", "sample_id", "parent_track_id", "parent_end_frame",
    "child_birth_frame", "child_track_a", "child_track_b", "decision",
    "confidence", "division_score", "best_continuation_score",
    "division_margin", "combined_volume_ratio", "weighted_centroid_error_um",
    "birth_separation_um", "future_window_truncated",
)

LINEAGE_EDGE_COLUMNS = (
    "division_event_id", "event_frame", "parent_track_id", "child_track_id",
    "relation", "confidence",
)

TRACK_LINEAGE_COLUMNS = (
    "track_id", "parent_track_id", "division_event_id", "root_track_id",
    "generation", "lineage_status",
)

PROTECTED_TRACK_COLUMNS = (
    "division_event_id", "track_id", "role", "protected_reason",
)

TRACK_SUMMARY_COLUMNS = (
    "track_id", "first_frame", "last_frame", "observation_count",
    "first_observation_index", "last_observation_index", "first_is_virtual",
    "last_is_virtual",
)


@dataclass(frozen=True)
class CellLineageConfig:
    """Provisional, fully configurable Stage 10 thresholds and score weights."""

    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    division_frame_gap: int = 1
    minimum_parent_observations: int = 3
    parent_motion_history: int = 3
    parent_volume_history: int = 3
    parent_intensity_baseline_history: int = 3
    future_child_horizon: int = 3
    minimum_child_observations: int = 2
    boundary_margin_um: float = 4.0
    child_search_radius_um: float = 12.0
    maximum_weighted_centroid_error_um: float = 6.0
    maximum_birth_separation_um: float = 24.0
    maximum_combined_volume_relative_error: float = 0.35
    combined_volume_score_scale: float = 0.20
    hard_minimum_child_voxels: int = 20
    hard_minimum_child_volume_fraction: float = 0.05
    preferred_minimum_child_voxels: int = 50
    preferred_minimum_child_volume_fraction: float = 0.10
    core_erosion_um: float = 0.5
    background_shell_inner_um: float = 1.0
    background_shell_outer_um: float = 3.0
    confirmed_minimum_score: float = 0.70
    confirmed_minimum_margin: float = 0.10
    probable_minimum_score: float = 0.60
    probable_minimum_margin: float = 0.05
    spatial_weight: float = 0.30
    combined_volume_weight: float = 0.30
    persistence_weight: float = 0.20
    divergence_weight: float = 0.10
    parent_intensity_weight: float = 0.06
    child_brightening_weight: float = 0.04
    individual_child_distance_score_scale_um: float = 6.0
    weighted_centroid_score_scale_um: float = 4.0
    continuation_position_score_scale_um: float = 5.0
    continuation_volume_score_scale: float = 0.25
    continuation_shape_score_scale: float = 0.25
    divergence_negative_tolerance_um: float = 1.0
    divergence_gain_score_scale_um: float = 3.0
    divergence_slope_score_scale_um_per_frame: float = 1.0
    parent_intensity_log_ratio_scale: float = 0.25
    child_brightening_log_ratio_scale: float = 0.25
    preferred_fragment_penalty_per_child: float = 0.05

    def __post_init__(self) -> None:
        spacing = tuple(float(value) for value in self.voxel_size_zyx_um)
        if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("voxel_size_zyx_um must contain three positive finite values")
        positive_counts = {
            "minimum_parent_observations": self.minimum_parent_observations,
            "parent_motion_history": self.parent_motion_history,
            "parent_volume_history": self.parent_volume_history,
            "parent_intensity_baseline_history": self.parent_intensity_baseline_history,
            "future_child_horizon": self.future_child_horizon,
            "minimum_child_observations": self.minimum_child_observations,
            "hard_minimum_child_voxels": self.hard_minimum_child_voxels,
            "preferred_minimum_child_voxels": self.preferred_minimum_child_voxels,
        }
        for name, value in positive_counts.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.division_frame_gap != 1:
            raise ValueError("division_frame_gap must be 1 in the current implementation")
        positive_scales = {
            "boundary_margin_um": self.boundary_margin_um,
            "child_search_radius_um": self.child_search_radius_um,
            "maximum_weighted_centroid_error_um": self.maximum_weighted_centroid_error_um,
            "maximum_birth_separation_um": self.maximum_birth_separation_um,
            "maximum_combined_volume_relative_error": self.maximum_combined_volume_relative_error,
            "combined_volume_score_scale": self.combined_volume_score_scale,
            "core_erosion_um": self.core_erosion_um,
            "background_shell_outer_um": self.background_shell_outer_um,
            "individual_child_distance_score_scale_um": self.individual_child_distance_score_scale_um,
            "weighted_centroid_score_scale_um": self.weighted_centroid_score_scale_um,
            "continuation_position_score_scale_um": self.continuation_position_score_scale_um,
            "continuation_volume_score_scale": self.continuation_volume_score_scale,
            "continuation_shape_score_scale": self.continuation_shape_score_scale,
            "divergence_gain_score_scale_um": self.divergence_gain_score_scale_um,
            "divergence_slope_score_scale_um_per_frame": self.divergence_slope_score_scale_um_per_frame,
            "parent_intensity_log_ratio_scale": self.parent_intensity_log_ratio_scale,
            "child_brightening_log_ratio_scale": self.child_brightening_log_ratio_scale,
        }
        for name, value in positive_scales.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.background_shell_inner_um) or self.background_shell_inner_um < 0:
            raise ValueError("background_shell_inner_um must be nonnegative and finite")
        if self.background_shell_outer_um <= self.background_shell_inner_um:
            raise ValueError("background_shell_outer_um must exceed background_shell_inner_um")
        fractions = {
            "hard_minimum_child_volume_fraction": self.hard_minimum_child_volume_fraction,
            "preferred_minimum_child_volume_fraction": self.preferred_minimum_child_volume_fraction,
        }
        for name, value in fractions.items():
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.preferred_minimum_child_voxels < self.hard_minimum_child_voxels:
            raise ValueError("preferred child voxel limit cannot be weaker than the hard limit")
        if self.preferred_minimum_child_volume_fraction < self.hard_minimum_child_volume_fraction:
            raise ValueError("preferred child volume fraction cannot be weaker than the hard limit")
        weights = (
            self.spatial_weight, self.combined_volume_weight,
            self.persistence_weight, self.divergence_weight,
            self.parent_intensity_weight, self.child_brightening_weight,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in weights):
            raise ValueError("score weights must be nonnegative and finite")
        thresholds = (
            self.confirmed_minimum_score, self.confirmed_minimum_margin,
            self.probable_minimum_score, self.probable_minimum_margin,
        )
        if not all(math.isfinite(float(value)) for value in thresholds):
            raise ValueError("score and margin thresholds must be finite")
        if self.confirmed_minimum_score < self.probable_minimum_score:
            raise ValueError("confirmed score threshold must be at least probable")
        if self.confirmed_minimum_margin < self.probable_minimum_margin:
            raise ValueError("confirmed margin threshold must be at least probable")
        if not math.isfinite(self.divergence_negative_tolerance_um) or self.divergence_negative_tolerance_um < 0:
            raise ValueError("divergence_negative_tolerance_um must be nonnegative and finite")
        if not math.isfinite(self.preferred_fragment_penalty_per_child) or not 0 <= self.preferred_fragment_penalty_per_child <= 1:
            raise ValueError("preferred_fragment_penalty_per_child must be between zero and one")

    def as_dict(self) -> dict[str, object]:
        """Return JSON-stable configuration metadata."""

        data = asdict(self)
        data["voxel_size_zyx_um"] = list(self.voxel_size_zyx_um)
        return data
