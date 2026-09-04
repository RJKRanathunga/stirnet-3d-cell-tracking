"""Configuration and stable schemas for Stage 12 final visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


TRACK_SUMMARY_COLUMNS = (
    "track_id", "first_frame", "last_frame", "observation_count",
    "duration_frames", "frame_gap_count", "missing_frame_count",
    "start_cell_id", "end_cell_id", "start_z", "start_y", "start_x",
    "end_z", "end_y", "end_x", "start_boundary_distance_um",
    "end_boundary_distance_um", "start_classification", "end_classification",
    "start_reason", "end_reason", "suspicious_start", "suspicious_end",
    "short_lived", "has_temporal_gap", "stage11_modified", "repair_count",
    "forced_repair_count", "weakest_repair_score", "original_segment_ids",
    "unresolved_ending", "division_related", "merge_related",
    "failure_score", "failure_reasons",
)

DIAGNOSTIC_EVENT_COLUMNS = (
    "event_id", "track_id", "event_type", "frame", "z", "y", "x",
    "classification", "is_failure", "severity", "score", "reason",
    "related_track_ids", "stage11_modified", "forced", "decision_id",
)


@dataclass(frozen=True)
class FinalVisualizationConfig:
    """Display and diagnostic prioritization settings for Stage 12.

    Failure scores are visualization priorities, not calibrated probabilities.
    """

    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    boundary_margin_um: float = 4.0
    short_track_max_observations: int = 2
    low_confidence_repair_score: float = 0.65
    tail_length: int = 20
    show_boundary_tracks: bool = False
    show_modified_tracks: bool = False
    show_valid_event_points: bool = False
    scene_padding_zyx: tuple[int, int, int] = (2, 12, 12)
    suspicious_endpoint_weight: float = 0.55
    unresolved_ending_weight: float = 0.25
    short_track_weight: float = 0.15
    temporal_gap_weight: float = 0.15
    forced_repair_weight: float = 0.08
    low_confidence_repair_weight: float = 0.08

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.voxel_size_zyx_um
        ):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if len(self.scene_padding_zyx) != 3 or any(
            int(value) < 0 for value in self.scene_padding_zyx
        ):
            raise ValueError("scene_padding_zyx must contain three nonnegative integers")
        if self.boundary_margin_um < 0:
            raise ValueError("boundary_margin_um must be nonnegative")
        if self.short_track_max_observations < 1:
            raise ValueError("short_track_max_observations must be at least one")
        if not 0 <= self.low_confidence_repair_score <= 1:
            raise ValueError("low_confidence_repair_score must be in [0, 1]")
        if self.tail_length < 1:
            raise ValueError("tail_length must be at least one")
        for name in (
            "suspicious_endpoint_weight", "unresolved_ending_weight",
            "short_track_weight", "temporal_gap_weight", "forced_repair_weight",
            "low_confidence_repair_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
