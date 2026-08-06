"""Stable table schemas emitted by the 4D tracker."""

WINDOW_SUMMARY_COLUMNS = (
    "window_id", "component_id", "start_frame", "end_frame",
    "commit_start_frame", "commit_end_frame", "node_count", "edge_count",
    "solver_type", "status", "objective", "runtime_seconds", "conflicts",
)
COMPONENT_SUMMARY_COLUMNS = (
    "component_id", "minimum_frame", "maximum_frame", "node_count",
    "temporal_edge_count", "ambiguity_seed_count", "window_count",
    "solver_type", "status", "selected_edge_count", "changed_edge_count",
    "fallback_used", "failure_message",
)
TEMPORAL_EDGE_COLUMNS = (
    "edge_index", "source_node", "target_node", "source_frame", "target_frame",
    "source_detection_index", "target_detection_index", "frame_gap",
    "provisional_selected", "optimized_selected", "graph_expanded",
    "hard_safety_valid", "base_stage7_cost", "unary_cost", "motion_cost",
    "graph_cost", "persistent_relation_cost", "boundary_cost",
    "total_effective_cost", "displacement_um", "global_motion_residual_um",
    "relative_motion_residual_um", "volume_log_error", "shape_error",
    "intensity_error", "provisional_probability", "provisional_margin",
    "component_id", "window_id", "solver_type", "changed_from_provisional",
)
ASSIGNMENT_CHANGE_COLUMNS = (
    "change_type", "source_node", "target_node", "source_frame", "target_frame",
    "source_detection_index", "target_detection_index", "provisional_track_id",
    "optimized_track_id", "component_id", "window_id", "solver_type",
)
BOUNDARY_EVENT_COLUMNS = (
    "node_index", "frame", "detection_index", "optimized_track_id",
    "event_type", "boundary_face", "cost", "distance_to_face_um",
    "directional_support", "component_id", "window_id", "solver_type",
)
SOLVER_DIAGNOSTIC_COLUMNS = (
    "component_id", "window_id", "solver_type", "status", "success",
    "message", "node_count", "edge_count", "pair_factor_count",
    "variable_count", "constraint_count", "objective", "mip_gap",
    "iterations", "runtime_seconds",
)
TRACK_ID_MAP_COLUMNS = (
    "provisional_track_id", "optimized_track_id", "shared_observations",
    "provisional_observations", "optimized_observations",
)
