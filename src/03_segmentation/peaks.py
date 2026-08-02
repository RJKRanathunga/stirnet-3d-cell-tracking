"""Multi-scale EDT peaks, merge-tree evidence, and lobe collapsing."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.special import expit
from scipy.stats import beta as beta_distribution
from skimage.morphology import h_maxima

from .config import SegmentationConfig
from .distance import (
    compute_distance_transform,
    physical_distance,
    physical_sigma_voxels,
)


Position3D = tuple[int, int, int]


@dataclass(frozen=True)
class PeakCandidate:
    """One physical EDT maximum consolidated across smoothing settings."""

    peak_id: int
    position_zyx: Position3D
    raw_depth_um: float
    smoothed_depth_um: float
    scale_support: float
    h_support: float
    setting_support: float
    detection_count: int
    persistence_score: float


@dataclass(frozen=True)
class PairEvidence:
    """Probabilistic merge-tree evidence for a pair of peak candidates."""

    peak_id_a: int
    peak_id_b: int
    separation_um: float
    saddle_um: float
    branch_persistence: float
    branch_balance: float
    separation_support: float
    peak_support: float
    distinct_lobe_log_odds: float
    distinct_lobe_probability: float

    @property
    def same_lobe_probability(self) -> float:
        return 1.0 - self.distinct_lobe_probability


@dataclass(frozen=True)
class DistancePeakAnalysis:
    """EDT fields and peak candidates for one padded component crop."""

    raw_distance: np.ndarray
    merge_tree_distance: np.ndarray
    watershed_distance: np.ndarray
    peaks: tuple[PeakCandidate, ...]


@dataclass(frozen=True)
class LobeCollapseResult:
    """Candidates annotated by complete-link same-lobe consolidation."""

    effective_peaks: tuple[PeakCandidate, ...]
    lobe_id_by_peak: dict[int, int]


@dataclass(frozen=True)
class _PeakDetection:
    position_zyx: Position3D
    sigma_um: float
    h_um: float
    raw_depth_um: float
    smoothed_depth_um: float


_NEIGHBOR_OFFSETS_6: tuple[Position3D, ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def _representative_peak_position(
    peak_component: np.ndarray,
    smoothed_distance: np.ndarray,
) -> Position3D:
    coordinates = np.argwhere(peak_component)
    if coordinates.size == 0:
        raise ValueError("peak component is empty")
    values = smoothed_distance[tuple(coordinates.T)]
    return tuple(int(value) for value in coordinates[int(np.argmax(values))])


def _detection_sort_key(detection: _PeakDetection) -> tuple[float, ...]:
    return (
        -detection.raw_depth_um,
        -detection.smoothed_depth_um,
        float(detection.position_zyx[0]),
        float(detection.position_zyx[1]),
        float(detection.position_zyx[2]),
        detection.sigma_um,
        detection.h_um,
    )


def _cluster_peak_detections(
    detections: list[_PeakDetection],
    voxel_size_zyx: tuple[float, float, float],
    cluster_radius_um: float,
) -> list[list[_PeakDetection]]:
    """Greedily consolidate repeated detections of one physical maximum."""

    clusters: list[list[_PeakDetection]] = []
    for detection in sorted(detections, key=_detection_sort_key):
        best_index: int | None = None
        best_distance = np.inf
        for index, cluster in enumerate(clusters):
            representative = min(cluster, key=_detection_sort_key)
            distance = physical_distance(
                detection.position_zyx,
                representative.position_zyx,
                voxel_size_zyx,
            )
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None and best_distance <= cluster_radius_um:
            clusters[best_index].append(detection)
        else:
            clusters.append([detection])
    return clusters


def detect_persistent_distance_peaks(
    mask: np.ndarray,
    config: SegmentationConfig,
) -> DistancePeakAnalysis:
    """Generate liberal multi-scale peaks and three task-specific EDT fields."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("mask must be a non-empty 3-D binary array")

    voxel_size = config.voxel_size_zyx_um
    raw_distance = compute_distance_transform(mask, voxel_size).astype(np.float32)
    detections: list[_PeakDetection] = []

    for sigma_um in config.sigma_levels_um:
        smoothed = ndimage.gaussian_filter(
            raw_distance,
            sigma=physical_sigma_voxels(sigma_um, voxel_size),
            mode="nearest",
        )
        for h_um in config.h_levels_um:
            maxima = h_maxima(smoothed, h=h_um) & mask
            peak_labels, peak_count = ndimage.label(
                maxima,
                structure=ndimage.generate_binary_structure(3, 1),
            )
            for peak_label in range(1, int(peak_count) + 1):
                position = _representative_peak_position(
                    peak_labels == peak_label, smoothed
                )
                detections.append(
                    _PeakDetection(
                        position_zyx=position,
                        sigma_um=float(sigma_um),
                        h_um=float(h_um),
                        raw_depth_um=float(raw_distance[position]),
                        smoothed_depth_um=float(smoothed[position]),
                    )
                )

    if not detections:
        position = tuple(
            int(value)
            for value in np.unravel_index(
                int(np.argmax(raw_distance)), raw_distance.shape
            )
        )
        detections.append(
            _PeakDetection(
                position, 0.0, 0.0, float(raw_distance[position]), float(raw_distance[position])
            )
        )

    clusters = _cluster_peak_detections(
        detections, voxel_size, config.peak_cluster_radius_um
    )
    maximum_distance = max(float(raw_distance.max()), 1e-6)
    unranked: list[PeakCandidate] = []
    for cluster in clusters:
        representative = min(cluster, key=_detection_sort_key)
        unique_sigmas = {record.sigma_um for record in cluster}
        unique_h = {record.h_um for record in cluster}
        unique_settings = {(record.sigma_um, record.h_um) for record in cluster}
        scale_support = len(unique_sigmas) / len(config.sigma_levels_um)
        h_support = len(unique_h) / len(config.h_levels_um)
        setting_support = len(unique_settings) / (
            len(config.sigma_levels_um) * len(config.h_levels_um)
        )
        depth_score = representative.raw_depth_um / maximum_distance
        persistence = float(
            0.35 * scale_support
            + 0.25 * h_support
            + 0.25 * depth_score
            + 0.15 * np.sqrt(setting_support)
        )
        unranked.append(
            PeakCandidate(
                peak_id=0,
                position_zyx=representative.position_zyx,
                raw_depth_um=representative.raw_depth_um,
                smoothed_depth_um=representative.smoothed_depth_um,
                scale_support=float(scale_support),
                h_support=float(h_support),
                setting_support=float(setting_support),
                detection_count=len(cluster),
                persistence_score=persistence,
            )
        )

    ordered = sorted(
        unranked,
        key=lambda peak: (
            -peak.persistence_score,
            -peak.raw_depth_um,
            peak.position_zyx,
        ),
    )
    peaks = tuple(
        replace(peak, peak_id=index)
        for index, peak in enumerate(ordered, start=1)
    )

    merge_tree_distance = ndimage.gaussian_filter(
        raw_distance,
        sigma=physical_sigma_voxels(config.merge_tree_sigma_um, voxel_size),
        mode="nearest",
    ).astype(np.float32)
    watershed_distance = ndimage.gaussian_filter(
        raw_distance,
        sigma=physical_sigma_voxels(config.watershed_sigma_um, voxel_size),
        mode="nearest",
    ).astype(np.float32)
    return DistancePeakAnalysis(
        raw_distance, merge_tree_distance, watershed_distance, peaks
    )


def widest_path_saddle_level(
    distance_um: np.ndarray,
    mask: np.ndarray,
    start_zyx: Position3D,
    end_zyx: Position3D,
) -> float:
    """Return the highest attainable minimum EDT along a 6-connected path.

    This maximin value is the merge-tree saddle where the two peak branches
    first join as the EDT superlevel set is lowered.
    """

    distance = np.asarray(distance_um, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not mask[start_zyx] or not mask[end_zyx]:
        raise ValueError("both peak positions must lie inside the mask")

    best = np.full(distance.shape, -np.inf, dtype=np.float32)
    start_value = float(distance[start_zyx])
    best[start_zyx] = start_value
    queue: list[tuple[float, Position3D]] = [(-start_value, start_zyx)]
    shape = distance.shape

    while queue:
        negative_score, current = heapq.heappop(queue)
        current_score = -float(negative_score)
        if current == end_zyx:
            return current_score
        if current_score < float(best[current]) - 1e-8:
            continue
        z, y, x = current
        for dz, dy, dx in _NEIGHBOR_OFFSETS_6:
            neighbor = (z + dz, y + dy, x + dx)
            if not all(0 <= neighbor[axis] < shape[axis] for axis in range(3)):
                continue
            if not mask[neighbor]:
                continue
            candidate = min(current_score, float(distance[neighbor]))
            if candidate > float(best[neighbor]) + 1e-8:
                best[neighbor] = candidate
                heapq.heappush(queue, (-candidate, neighbor))
    raise RuntimeError("peak candidates are disconnected inside the component")


def branch_size_above_saddle(
    distance_um: np.ndarray,
    mask: np.ndarray,
    peak_zyx: Position3D,
    saddle_um: float,
) -> int:
    """Measure the peak-owned superlevel branch immediately above a saddle."""

    epsilon = max(
        np.finfo(np.float32).eps,
        1e-5 * max(float(distance_um[peak_zyx]), 1.0),
    )
    branch_mask = np.asarray(mask, dtype=bool) & (
        np.asarray(distance_um, dtype=float) > saddle_um + epsilon
    )
    if not branch_mask[peak_zyx]:
        return 1
    labels, _ = ndimage.label(
        branch_mask, structure=ndimage.generate_binary_structure(3, 1)
    )
    label_value = int(labels[peak_zyx])
    return int(np.count_nonzero(labels == label_value)) if label_value > 0 else 1


def _safe_beta_logpdf(
    value: float,
    parameters: tuple[float, float],
    epsilon: float,
) -> float:
    clipped = float(np.clip(value, epsilon, 1.0 - epsilon))
    return float(beta_distribution.logpdf(clipped, *parameters))


def pair_distinct_lobe_probability(
    transformed_features: dict[str, float],
    config: SegmentationConfig,
) -> tuple[float, float]:
    """Compute P(distinct lobes | pairwise merge-tree features)."""

    epsilon = config.probability_epsilon
    prior = config.pair_prior_distinct
    log_odds = float(np.log(prior + epsilon) - np.log(1.0 - prior + epsilon))
    for model in config.pair_feature_models:
        value = transformed_features[model.name]
        log_odds += model.weight * (
            _safe_beta_logpdf(value, model.distinct, epsilon)
            - _safe_beta_logpdf(value, model.same, epsilon)
        )
    log_odds /= max(config.pair_likelihood_temperature, epsilon)
    return float(expit(log_odds)), float(log_odds)


def build_peak_pair_evidence(
    peaks: tuple[PeakCandidate, ...],
    mask: np.ndarray,
    merge_tree_distance: np.ndarray,
    config: SegmentationConfig,
) -> tuple[PairEvidence, ...]:
    """Evaluate pairwise saddle and branch evidence for candidate peaks."""

    candidates = peaks[: config.max_candidate_peaks]
    total_voxels = max(int(np.count_nonzero(mask)), 1)
    records: list[PairEvidence] = []
    for first, second in combinations(candidates, 2):
        first_depth = float(merge_tree_distance[first.position_zyx])
        second_depth = float(merge_tree_distance[second.position_zyx])
        smaller_depth = max(min(first_depth, second_depth), 1e-6)
        saddle = widest_path_saddle_level(
            merge_tree_distance, mask, first.position_zyx, second.position_zyx
        )
        branch_persistence = float(
            np.clip(1.0 - saddle / smaller_depth, 0.0, 1.0)
        )
        first_branch = branch_size_above_saddle(
            merge_tree_distance, mask, first.position_zyx, saddle
        )
        second_branch = branch_size_above_saddle(
            merge_tree_distance, mask, second.position_zyx, saddle
        )
        smaller_branch_fraction = min(first_branch, second_branch) / total_voxels
        branch_balance = float(np.clip(smaller_branch_fraction / 0.08, 0.0, 1.0))
        separation = physical_distance(
            first.position_zyx, second.position_zyx, config.voxel_size_zyx_um
        )
        separation_ratio = separation / max(first_depth + second_depth, 1e-6)
        separation_support = float(separation_ratio / (separation_ratio + 0.65))
        peak_support = float(
            np.sqrt(
                max(first.persistence_score, 0.0)
                * max(second.persistence_score, 0.0)
            )
        )
        transformed = {
            "branch_persistence": branch_persistence,
            "branch_balance": branch_balance,
            "separation_support": separation_support,
            "peak_support": peak_support,
        }
        probability, log_odds = pair_distinct_lobe_probability(transformed, config)
        records.append(
            PairEvidence(
                first.peak_id,
                second.peak_id,
                separation,
                float(saddle),
                branch_persistence,
                branch_balance,
                separation_support,
                peak_support,
                log_odds,
                probability,
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                -record.distinct_lobe_probability,
                record.peak_id_a,
                record.peak_id_b,
            ),
        )
    )


def pair_probability_lookup(
    pair_evidence: tuple[PairEvidence, ...],
) -> dict[frozenset[int], float]:
    """Index distinct-lobe probabilities by unordered peak-ID pair."""

    return {
        frozenset((record.peak_id_a, record.peak_id_b)): record.distinct_lobe_probability
        for record in pair_evidence
    }


def collapse_same_lobe_peaks(
    peaks: tuple[PeakCandidate, ...],
    pair_evidence: tuple[PairEvidence, ...],
    config: SegmentationConfig,
) -> LobeCollapseResult:
    """Collapse likely within-cell maxima using complete-link grouping.

    Requiring compatibility with every group member avoids a transitive chain
    accidentally collapsing two genuine lobes through an intermediate peak.
    """

    candidates = peaks[: config.max_candidate_peaks]
    same_lookup = {
        frozenset((record.peak_id_a, record.peak_id_b)): record.same_lobe_probability
        for record in pair_evidence
    }
    ordered = sorted(
        candidates,
        key=lambda peak: (
            -(peak.persistence_score * peak.raw_depth_um),
            -peak.raw_depth_um,
            peak.position_zyx,
        ),
    )
    groups: list[list[PeakCandidate]] = []
    for peak in ordered:
        compatible_index: int | None = None
        best_minimum = -np.inf
        for index, group in enumerate(groups):
            minimum = min(
                same_lookup.get(frozenset((peak.peak_id, member.peak_id)), 0.0)
                for member in group
            )
            if (
                minimum >= config.same_lobe_collapse_probability
                and minimum > best_minimum
            ):
                compatible_index = index
                best_minimum = minimum
        if compatible_index is None:
            groups.append([peak])
        else:
            groups[compatible_index].append(peak)

    lobe_id_by_peak: dict[int, int] = {}
    representatives: list[PeakCandidate] = []
    for lobe_id, group in enumerate(groups, start=1):
        representative = min(
            group,
            key=lambda peak: (
                -(peak.persistence_score * peak.raw_depth_um),
                -peak.raw_depth_um,
                peak.position_zyx,
            ),
        )
        representatives.append(representative)
        for peak in group:
            lobe_id_by_peak[peak.peak_id] = lobe_id

    representatives.sort(
        key=lambda peak: (-peak.persistence_score, -peak.raw_depth_um, peak.position_zyx)
    )
    return LobeCollapseResult(tuple(representatives), lobe_id_by_peak)
