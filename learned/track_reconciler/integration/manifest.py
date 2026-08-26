"""Exact default feature manifest for tracklet observations and reliability."""

from __future__ import annotations


# Each primitive is paired with a validity bit -> 48 structured channels.
TRACKLET_STRUCTURED_PRIMITIVES: tuple[str, ...] = (
    "relative_position_z_um",
    "relative_position_y_um",
    "relative_position_x_um",
    "global_shift_z_um",
    "global_shift_y_um",
    "global_shift_x_um",
    "velocity_z_um_per_frame",
    "velocity_y_um_per_frame",
    "velocity_x_um_per_frame",
    "relative_velocity_z_um_per_frame",
    "relative_velocity_y_um_per_frame",
    "relative_velocity_x_um_per_frame",
    "acceleration_z_um_per_frame2",
    "acceleration_y_um_per_frame2",
    "acceleration_x_um_per_frame2",
    "log_volume",
    "delta_log_volume",
    "axis_major",
    "axis_middle",
    "axis_minor",
    "elongation",
    "flatness",
    "anisotropy",
    "intensity_mean",
)
TRACKLET_STRUCTURED_DIM = 2 * len(TRACKLET_STRUCTURED_PRIMITIVES)
assert TRACKLET_STRUCTURED_DIM == 48


TRACKLET_RELIABILITY_FEATURES: tuple[str, ...] = (
    "log_observation_count",
    "temporal_span_frames",
    "association_probability_mean",
    "association_probability_min",
    "relative_velocity_sample_count",
    "relative_velocity_error_ema_um",
    "position_residual_ema_norm_um",
    "distance_to_boundary_um",
    "touches_boundary",
    "log_median_volume",
    "small_cell_indicator",
    "crop_valid_fraction",
)
assert len(TRACKLET_RELIABILITY_FEATURES) == 12
