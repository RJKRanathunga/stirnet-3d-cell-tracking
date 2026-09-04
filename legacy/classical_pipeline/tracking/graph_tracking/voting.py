"""Vector-preserving Hough-style voting and candidate evidence extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import GraphTrackingConfig
from .deformation import fit_best_local_transform
from .spatial_graph import neighbor_indices
from .types import LocalTransform, SpatialGraph, TemporalAnchor, VoteConsensus


@dataclass(frozen=True, slots=True)
class SourceVoteBundle:
    source_node_index: int
    anchor_track_ids: np.ndarray
    anchor_source_indices: np.ndarray
    anchor_target_indices: np.ndarray
    source_relative_vectors_zyx_um: np.ndarray
    vote_positions_zyx_um: np.ndarray
    weights: np.ndarray
    consensus: VoteConsensus
    local_transform: LocalTransform | None
    deformation_prediction_zyx_um: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidateVoteEvidence:
    source_node_index: int
    candidate_node_index: int
    anchor_track_ids: np.ndarray
    source_relative_vectors_zyx_um: np.ndarray
    target_relative_vectors_zyx_um: np.ndarray
    residual_vectors_zyx_um: np.ndarray
    residual_norms_um: np.ndarray
    weights: np.ndarray
    inlier_mask: np.ndarray
    vote_score: float
    vector_score: float
    radial_score: float
    relative_volume_score: float
    consensus_score: float
    deformation_score: float


def weighted_geometric_median(points: np.ndarray, weights: np.ndarray, iterations: int = 32) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(points) == 1:
        return points[0].copy()
    current = np.average(points, axis=0, weights=np.maximum(weights, 1.0e-12))
    for _ in range(iterations):
        distances = np.linalg.norm(points - current, axis=1)
        if np.any(distances < 1.0e-9):
            return points[int(np.argmin(distances))].copy()
        adjusted = weights / np.maximum(distances, 1.0e-9)
        updated = np.sum(points * adjusted[:, None], axis=0) / float(adjusted.sum())
        if float(np.linalg.norm(updated - current)) < 1.0e-6:
            return updated
        current = updated
    return current


def robust_vote_consensus(votes: np.ndarray, weights: np.ndarray, config: GraphTrackingConfig) -> VoteConsensus:
    votes = np.asarray(votes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(votes) == 0:
        return VoteConsensus(np.full(3, np.nan), math.nan, np.zeros(0, dtype=bool), 0.0, 0.0)
    inliers = np.ones(len(votes), dtype=bool)
    center = weighted_geometric_median(votes, weights)
    for _ in range(config.consensus_iterations):
        residuals = np.linalg.norm(votes - center, axis=1)
        updated = residuals <= config.vote_inlier_radius_um
        if not np.any(updated):
            updated[int(np.argmin(residuals))] = True
        if np.array_equal(updated, inliers):
            break
        inliers = updated
        center = weighted_geometric_median(votes[inliers], weights[inliers])
    residuals = np.linalg.norm(votes[inliers] - center, axis=1)
    normalized = weights[inliers] / max(float(weights[inliers].sum()), 1.0e-12)
    dispersion = float(np.sqrt(np.sum(normalized * residuals**2)))
    fraction = float(weights[inliers].sum() / max(float(weights.sum()), 1.0e-12))
    return VoteConsensus(center, dispersion, inliers, fraction, float(weights.sum()))


def build_forward_vote_bundle(
    *,
    source_graph: SpatialGraph,
    target_graph: SpatialGraph,
    source_node_index: int,
    anchors_by_source_index: dict[int, TemporalAnchor],
    config: GraphTrackingConfig,
) -> SourceVoteBundle | None:
    source_position = source_graph.positions_zyx_um[source_node_index]
    records: list[tuple[int, int, int, np.ndarray, np.ndarray, float]] = []
    for neighbor_index in neighbor_indices(source_graph, source_node_index):
        anchor = anchors_by_source_index.get(int(neighbor_index))
        if anchor is None:
            continue
        source_relative = source_position - source_graph.positions_zyx_um[neighbor_index]
        target_anchor_position = target_graph.positions_zyx_um[anchor.target_node_index]
        vote = target_anchor_position + source_relative
        records.append((anchor.track_id, int(neighbor_index), anchor.target_node_index, source_relative, vote, anchor.reliability))
    if len(records) < config.minimum_anchor_votes:
        return None
    records.sort(key=lambda row: (-row[5], row[0]))
    track_ids = np.asarray([row[0] for row in records], dtype=np.int64)
    source_indices = np.asarray([row[1] for row in records], dtype=np.int32)
    target_indices = np.asarray([row[2] for row in records], dtype=np.int32)
    source_relative = np.asarray([row[3] for row in records], dtype=float)
    votes = np.asarray([row[4] for row in records], dtype=float)
    weights = np.asarray([row[5] for row in records], dtype=float)
    consensus = robust_vote_consensus(votes, weights, config)

    local_transform: LocalTransform | None = None
    deformation_prediction = consensus.position_zyx_um.copy()
    inlier_source_indices = source_indices[consensus.inlier_mask]
    inlier_target_indices = target_indices[consensus.inlier_mask]
    if len(inlier_source_indices) >= 1:
        local_transform = fit_best_local_transform(
            source_graph.positions_zyx_um[inlier_source_indices],
            target_graph.positions_zyx_um[inlier_target_indices],
            weights[consensus.inlier_mask],
            config,
        )
        deformation_prediction = local_transform.apply(source_position)

    return SourceVoteBundle(
        source_node_index=source_node_index,
        anchor_track_ids=track_ids,
        anchor_source_indices=source_indices,
        anchor_target_indices=target_indices,
        source_relative_vectors_zyx_um=source_relative,
        vote_positions_zyx_um=votes,
        weights=weights,
        consensus=consensus,
        local_transform=local_transform,
        deformation_prediction_zyx_um=deformation_prediction,
    )


def evaluate_candidate(
    *,
    bundle: SourceVoteBundle,
    source_graph: SpatialGraph,
    target_graph: SpatialGraph,
    candidate_node_index: int,
    config: GraphTrackingConfig,
) -> CandidateVoteEvidence:
    candidate_position = target_graph.positions_zyx_um[candidate_node_index]
    target_anchor_positions = target_graph.positions_zyx_um[bundle.anchor_target_indices]
    target_relative = candidate_position - target_anchor_positions
    residual_vectors = target_relative - bundle.source_relative_vectors_zyx_um
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    kernel = np.exp(-0.5 * (np.linalg.norm(candidate_position - bundle.vote_positions_zyx_um, axis=1) / config.vote_kernel_sigma_um) ** 2)
    vote_score = float(np.sum(bundle.weights * kernel) / max(float(bundle.weights.sum()), 1.0e-12))

    inlier_weights = bundle.weights * bundle.consensus.inlier_mask.astype(float)
    if float(inlier_weights.sum()) <= 0:
        inlier_weights = bundle.weights
    vector_error = float(np.sum(inlier_weights * residual_norms) / max(float(inlier_weights.sum()), 1.0e-12))
    vector_score = math.exp(-vector_error / config.vector_scale_um)

    source_radial = np.linalg.norm(bundle.source_relative_vectors_zyx_um, axis=1)
    target_radial = np.linalg.norm(target_relative, axis=1)
    radial_error = float(np.sum(inlier_weights * np.abs(target_radial - source_radial)) / max(float(inlier_weights.sum()), 1.0e-12))
    radial_score = math.exp(-radial_error / config.radial_scale_um)

    source_volume = max(float(source_graph.volumes[bundle.source_node_index]), 1.0e-9)
    candidate_volume = max(float(target_graph.volumes[candidate_node_index]), 1.0e-9)
    source_anchor_volume = np.maximum(source_graph.volumes[bundle.anchor_source_indices], 1.0e-9)
    target_anchor_volume = np.maximum(target_graph.volumes[bundle.anchor_target_indices], 1.0e-9)
    source_log_ratio = np.log(source_anchor_volume / source_volume)
    target_log_ratio = np.log(target_anchor_volume / candidate_volume)
    volume_error = float(np.sum(inlier_weights * np.abs(source_log_ratio - target_log_ratio)) / max(float(inlier_weights.sum()), 1.0e-12))
    relative_volume_score = math.exp(-volume_error / config.relative_volume_log_scale)

    consensus_residual = float(np.linalg.norm(candidate_position - bundle.consensus.position_zyx_um))
    dispersion_penalty = math.exp(-bundle.consensus.dispersion_um / max(config.maximum_vote_dispersion_um, 1.0e-9))
    consensus_score = math.exp(-consensus_residual / config.vote_kernel_sigma_um) * bundle.consensus.inlier_weight_fraction * dispersion_penalty

    deformation_residual = float(np.linalg.norm(candidate_position - bundle.deformation_prediction_zyx_um))
    deformation_score = math.exp(-deformation_residual / config.vote_kernel_sigma_um)

    return CandidateVoteEvidence(
        source_node_index=bundle.source_node_index,
        candidate_node_index=candidate_node_index,
        anchor_track_ids=bundle.anchor_track_ids,
        source_relative_vectors_zyx_um=bundle.source_relative_vectors_zyx_um,
        target_relative_vectors_zyx_um=target_relative,
        residual_vectors_zyx_um=residual_vectors,
        residual_norms_um=residual_norms,
        weights=bundle.weights,
        inlier_mask=bundle.consensus.inlier_mask,
        vote_score=vote_score,
        vector_score=vector_score,
        radial_score=radial_score,
        relative_volume_score=relative_volume_score,
        consensus_score=consensus_score,
        deformation_score=deformation_score,
    )


def build_backward_vote_bundle(
    *,
    source_graph: SpatialGraph,
    target_graph: SpatialGraph,
    target_node_index: int,
    anchors_by_target_index: dict[int, TemporalAnchor],
    config: GraphTrackingConfig,
) -> SourceVoteBundle | None:
    """Predict a target cell's previous position from matched target neighbours."""
    target_position = target_graph.positions_zyx_um[target_node_index]
    records: list[tuple[int, int, int, np.ndarray, np.ndarray, float]] = []
    for neighbor_index in neighbor_indices(target_graph, target_node_index):
        anchor = anchors_by_target_index.get(int(neighbor_index))
        if anchor is None:
            continue
        target_relative = target_position - target_graph.positions_zyx_um[neighbor_index]
        source_anchor_position = source_graph.positions_zyx_um[anchor.source_node_index]
        vote = source_anchor_position + target_relative
        records.append((anchor.track_id, anchor.source_node_index, int(neighbor_index), target_relative, vote, anchor.reliability))
    if len(records) < config.minimum_anchor_votes:
        return None
    records.sort(key=lambda row: (-row[5], row[0]))
    track_ids = np.asarray([row[0] for row in records], dtype=np.int64)
    source_indices = np.asarray([row[1] for row in records], dtype=np.int32)
    target_indices = np.asarray([row[2] for row in records], dtype=np.int32)
    relative = np.asarray([row[3] for row in records], dtype=float)
    votes = np.asarray([row[4] for row in records], dtype=float)
    weights = np.asarray([row[5] for row in records], dtype=float)
    consensus = robust_vote_consensus(votes, weights, config)
    return SourceVoteBundle(
        source_node_index=target_node_index,
        anchor_track_ids=track_ids,
        anchor_source_indices=source_indices,
        anchor_target_indices=target_indices,
        source_relative_vectors_zyx_um=relative,
        vote_positions_zyx_um=votes,
        weights=weights,
        consensus=consensus,
        local_transform=None,
        deformation_prediction_zyx_um=consensus.position_zyx_um.copy(),
    )
