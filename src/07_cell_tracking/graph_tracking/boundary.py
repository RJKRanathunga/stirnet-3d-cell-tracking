"""Graph-supported boundary entry and exit hypotheses."""

from __future__ import annotations

import math

import numpy as np

from .config import GraphTrackingConfig
from .geometry import FACE_NORMALS, boundary_coverage_score, classify_point_against_volume, is_near_boundary
from .types import SpatialGraph
from .voting import SourceVoteBundle


def _weighted_directional_agreement(
    origins: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray,
    face: str,
) -> float:
    normal = FACE_NORMALS.get(face)
    if normal is None or len(origins) == 0:
        return 0.0
    outward = ((destinations - origins) @ normal) > 0.0
    return float(np.sum(weights * outward.astype(float)) / max(float(weights.sum()), 1.0e-12))


def exit_hypothesis(
    *,
    bundle: SourceVoteBundle,
    source_graph: SpatialGraph,
    source_node_index: int,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
    config: GraphTrackingConfig,
) -> dict[str, object] | None:
    if not config.enable_boundary_entry_exit:
        return None
    source_position = source_graph.positions_zyx_um[source_node_index]
    if not is_near_boundary(
        source_position,
        margin_um=config.boundary_evidence_margin_um,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
    ):
        return None
    classification = classify_point_against_volume(
        bundle.consensus.position_zyx_um,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
    )
    if classification.inside or classification.outside_distance_um < config.outside_vote_minimum_distance_um:
        return None
    inlier_weights = bundle.weights[bundle.consensus.inlier_mask]
    origins = np.repeat(source_position[None, :], int(bundle.consensus.inlier_mask.sum()), axis=0)
    destinations = bundle.vote_positions_zyx_um[bundle.consensus.inlier_mask]
    directional = _weighted_directional_agreement(origins, destinations, inlier_weights, classification.nearest_face)
    distance_score = min(1.0, classification.outside_distance_um / config.outside_vote_strong_distance_um)
    consensus_score = bundle.consensus.inlier_weight_fraction * math.exp(-bundle.consensus.dispersion_um / config.maximum_vote_dispersion_um)
    confidence = float(np.clip(distance_score * consensus_score * directional, 0.0, 1.0))
    supported = (
        confidence >= config.boundary_minimum_consensus
        and directional >= config.boundary_minimum_directional_agreement
    )
    return {
        "event_type": "exit",
        "predicted_position_zyx_um": bundle.consensus.position_zyx_um,
        "boundary_face": classification.nearest_face,
        "outside_distance_um": classification.outside_distance_um,
        "anchor_count": int(len(bundle.weights)),
        "inlier_count": int(bundle.consensus.inlier_mask.sum()),
        "consensus": float(consensus_score),
        "directional_agreement": directional,
        "confidence": confidence,
        "supported": bool(supported),
        "source_coverage": boundary_coverage_score(
            source_position,
            search_radius_um=config.maximum_radius_um,
            volume_shape_zyx=volume_shape_zyx,
            voxel_size_zyx_um=voxel_size_zyx_um,
        ),
    }


def entry_hypothesis(
    *,
    bundle: SourceVoteBundle,
    target_graph: SpatialGraph,
    target_node_index: int,
    volume_shape_zyx: np.ndarray,
    voxel_size_zyx_um: np.ndarray,
    config: GraphTrackingConfig,
) -> dict[str, object] | None:
    if not config.enable_boundary_entry_exit:
        return None
    target_position = target_graph.positions_zyx_um[target_node_index]
    if not is_near_boundary(
        target_position,
        margin_um=config.boundary_evidence_margin_um,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
    ):
        return None
    classification = classify_point_against_volume(
        bundle.consensus.position_zyx_um,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=voxel_size_zyx_um,
    )
    if classification.inside:
        return {
            "event_type": "inside_predecessor",
            "predicted_position_zyx_um": bundle.consensus.position_zyx_um,
            "boundary_face": classification.nearest_face,
            "outside_distance_um": 0.0,
            "anchor_count": int(len(bundle.weights)),
            "inlier_count": int(bundle.consensus.inlier_mask.sum()),
            "consensus": float(bundle.consensus.inlier_weight_fraction),
            "directional_agreement": 0.0,
            "confidence": float(bundle.consensus.inlier_weight_fraction * math.exp(-bundle.consensus.dispersion_um / config.maximum_vote_dispersion_um)),
            "supported": False,
        }
    if classification.outside_distance_um < config.outside_vote_minimum_distance_um:
        return None
    inlier_weights = bundle.weights[bundle.consensus.inlier_mask]
    origins = bundle.vote_positions_zyx_um[bundle.consensus.inlier_mask]
    destinations = np.repeat(target_position[None, :], int(bundle.consensus.inlier_mask.sum()), axis=0)
    # Entry should move opposite to the outward normal, so reverse the face normal test.
    normal = FACE_NORMALS[classification.nearest_face]
    inward = ((destinations - origins) @ (-normal)) > 0.0
    directional = float(np.sum(inlier_weights * inward.astype(float)) / max(float(inlier_weights.sum()), 1.0e-12))
    distance_score = min(1.0, classification.outside_distance_um / config.outside_vote_strong_distance_um)
    consensus_score = bundle.consensus.inlier_weight_fraction * math.exp(-bundle.consensus.dispersion_um / config.maximum_vote_dispersion_um)
    confidence = float(np.clip(distance_score * consensus_score * directional, 0.0, 1.0))
    supported = confidence >= config.boundary_minimum_consensus and directional >= config.boundary_minimum_directional_agreement
    return {
        "event_type": "entry",
        "predicted_position_zyx_um": bundle.consensus.position_zyx_um,
        "boundary_face": classification.nearest_face,
        "outside_distance_um": classification.outside_distance_um,
        "anchor_count": int(len(bundle.weights)),
        "inlier_count": int(bundle.consensus.inlier_mask.sum()),
        "consensus": float(consensus_score),
        "directional_agreement": directional,
        "confidence": confidence,
        "supported": bool(supported),
    }
