"""Convert structured graph evidence into bounded assignment cost changes."""

from __future__ import annotations

import math

import numpy as np

from .config import GraphTrackingConfig
from .voting import CandidateVoteEvidence


def combined_graph_score(evidence: CandidateVoteEvidence, config: GraphTrackingConfig) -> float:
    weights = np.asarray(
        [
            config.vote_score_weight,
            config.vector_score_weight,
            config.radial_score_weight,
            config.relative_volume_score_weight,
            config.consensus_score_weight,
        ],
        dtype=float,
    )
    values = np.asarray(
        [
            evidence.vote_score,
            evidence.vector_score,
            evidence.radial_score,
            evidence.relative_volume_score,
            0.5 * evidence.consensus_score + 0.5 * evidence.deformation_score,
        ],
        dtype=float,
    )
    score = float(np.sum(weights * values) / max(float(weights.sum()), 1.0e-12))
    return float(np.clip(score, config.probability_floor, 1.0))


def graph_cost_delta(score: float, confidence: float, config: GraphTrackingConfig) -> float:
    score = float(np.clip(score, config.probability_floor, 1.0))
    neutral = float(config.neutral_graph_score)
    raw = -config.graph_pair_weight * confidence * math.log(score / neutral)
    return float(np.clip(raw, -config.maximum_pair_cost_reduction, config.maximum_pair_cost_penalty))


def graph_confidence(
    *,
    anchor_count: int,
    inlier_fraction: float,
    dispersion_um: float,
    source_coverage: float,
    target_coverage: float,
    config: GraphTrackingConfig,
) -> float:
    count_score = min(1.0, anchor_count / max(config.strong_anchor_votes, 1))
    dispersion_score = math.exp(-dispersion_um / max(config.maximum_vote_dispersion_um, 1.0e-9))
    coverage = math.sqrt(max(source_coverage, config.boundary_coverage_floor) * max(target_coverage, config.boundary_coverage_floor))
    return float(np.clip(count_score * inlier_fraction * dispersion_score * coverage, 0.0, 1.0))
