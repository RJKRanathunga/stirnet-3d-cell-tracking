"""Stable diagnostic table schemas for graph tracking artifacts."""

GRAPH_CANDIDATE_EVIDENCE_COLUMNS = (
    "from_frame", "to_frame", "source_track_id", "source_state_index",
    "candidate_detection_index", "anchor_count", "inlier_count",
    "anchor_track_ids", "vote_score", "vector_score", "radial_score",
    "relative_volume_score", "consensus_score", "deformation_score",
    "graph_score", "graph_confidence", "graph_cost_delta",
    "consensus_z_um", "consensus_y_um", "consensus_x_um",
    "consensus_dispersion_um", "deformation_transform_type",
    "deformation_condition_number", "deformation_fit_residual_um",
    "deformation_prediction_z_um", "deformation_prediction_y_um",
    "deformation_prediction_x_um", "source_boundary_coverage",
    "target_boundary_coverage", "base_selected", "graph_selected",
)

GRAPH_ANCHOR_VOTE_COLUMNS = (
    "from_frame", "to_frame", "source_track_id", "source_state_index",
    "candidate_detection_index", "anchor_track_id",
    "source_relative_z_um", "source_relative_y_um", "source_relative_x_um",
    "vote_z_um", "vote_y_um", "vote_x_um", "candidate_residual_z_um",
    "candidate_residual_y_um", "candidate_residual_x_um",
    "candidate_residual_norm_um", "relative_volume_log_error",
    "anchor_weight", "inlier",
)

GRAPH_BOUNDARY_HYPOTHESIS_COLUMNS = (
    "event_type", "from_frame", "to_frame", "track_id", "detection_index",
    "boundary_face", "predicted_z_um", "predicted_y_um", "predicted_x_um",
    "outside_distance_um", "anchor_count", "inlier_count", "consensus",
    "directional_agreement", "confidence", "supported", "decision",
)

GRAPH_REFINEMENT_EVENT_COLUMNS = (
    "from_frame", "to_frame", "entity_type", "entity_index", "track_id",
    "base_decision", "graph_decision", "changed", "base_target_index",
    "graph_target_index", "base_cost", "graph_cost",
)

GRAPH_TRANSITION_SUMMARY_COLUMNS = (
    "from_frame", "to_frame", "mode", "source_nodes", "target_nodes",
    "source_edges", "target_edges", "temporal_anchors",
    "ambiguous_tracks", "ambiguous_detections", "candidate_pairs_scored",
    "exit_hypotheses", "entry_hypotheses", "assignment_changes",
    "base_matches", "graph_matches", "base_misses", "graph_misses",
    "base_births", "graph_births",
)
