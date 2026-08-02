"""Probabilistic H1/H2/H3 watershed hypotheses for one component."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.special import expit, logsumexp
from skimage.segmentation import watershed

from .config import SegmentationConfig
from .distance import physical_distance
from .geometry import (
    CellShape,
    describe_cell_mask,
    pair_interface_statistics,
    region_surface_area_um2,
)
from .peaks import PairEvidence, PeakCandidate, pair_probability_lookup


@dataclass(frozen=True)
class HypothesisEvidence:
    """Soft spatial evidence contributing to one hypothesis posterior."""

    lobe_support: float
    coverage_support: float
    marker_quality: float
    neck_support: float
    child_shape: float
    shape_improvement: float
    child_volume: float
    fragment_safety: float

    def as_dict(self) -> dict[str, float]:
        """Return evidence values keyed like configuration weights."""

        return {
            "lobe_support": self.lobe_support,
            "coverage_support": self.coverage_support,
            "marker_quality": self.marker_quality,
            "neck_support": self.neck_support,
            "child_shape": self.child_shape,
            "shape_improvement": self.shape_improvement,
            "child_volume": self.child_volume,
            "fragment_safety": self.fragment_safety,
        }


@dataclass(frozen=True)
class SplitHypothesis:
    """A safety-checked watershed partition and its probabilistic score."""

    k: int
    labels: np.ndarray
    selected_peaks: tuple[PeakCandidate, ...]
    hard_valid: bool
    hard_reasons: tuple[str, ...]
    prior_probability: float
    log_likelihood: float
    log_posterior_unnormalized: float
    posterior_probability: float
    evidence: HypothesisEvidence
    minimum_child_fraction: float
    combination_prescore: float = 0.0


@dataclass(frozen=True)
class HypothesisEvaluation:
    """All evaluated hypotheses and the best normalized model for each k."""

    best_by_k: tuple[SplitHypothesis, ...]
    all_hypotheses: tuple[SplitHypothesis, ...]


@dataclass(frozen=True)
class HypothesisDecision:
    """Hierarchical H1/H2/H3 posterior decision."""

    chosen: SplitHypothesis
    decision_status: str
    h2_conditional_probability: float
    h2_odds_vs_h1: float
    h3_conditional_probability: float
    h3_odds_vs_h2: float

    @property
    def split_accepted(self) -> bool:
        return self.chosen.k > 1


def _geometric_mean_probability(
    values: list[float] | tuple[float, ...] | np.ndarray,
    epsilon: float,
) -> float:
    values_array = np.asarray(values, dtype=float)
    if values_array.size == 0:
        return 0.5
    values_array = np.clip(values_array, epsilon, 1.0)
    return float(np.exp(np.mean(np.log(values_array))))


def selected_pair_probabilities(
    selected_peaks: tuple[PeakCandidate, ...] | list[PeakCandidate],
    pair_lookup: dict[frozenset[int], float],
) -> list[float]:
    """Return distinct-lobe probabilities for every selected peak pair."""

    return [
        float(pair_lookup.get(frozenset((first.peak_id, second.peak_id)), 0.0))
        for first, second in combinations(selected_peaks, 2)
    ]


def single_cell_shape_probability(description: CellShape) -> float:
    """Return a broad probability that one region is a plausible single cell."""

    elongation_probability = float(expit((4.10 - description.elongation) / 0.80))
    flatness_probability = float(expit((3.10 - description.flatness) / 0.65))
    extent_probability = float(expit((description.extent - 0.075) / 0.035))
    return _geometric_mean_probability(
        [elongation_probability, flatness_probability, extent_probability], 1e-6
    )


def _child_shape_evidence(
    region_descriptions: dict[int, CellShape],
    merged_description: CellShape,
    epsilon: float,
) -> tuple[float, float]:
    absolute_child_shape = _geometric_mean_probability(
        [single_cell_shape_probability(value) for value in region_descriptions.values()],
        epsilon,
    )
    mean_child_elongation = float(
        np.mean([value.elongation for value in region_descriptions.values()])
    )
    mean_child_extent = float(
        np.mean([value.extent for value in region_descriptions.values()])
    )
    elongation_improvement = float(
        expit((merged_description.elongation - mean_child_elongation) / 0.40)
    )
    extent_improvement = float(
        expit((mean_child_extent - merged_description.extent) / 0.080)
    )
    shape_improvement = _geometric_mean_probability(
        [elongation_improvement, extent_improvement], epsilon
    )
    return absolute_child_shape, shape_improvement


def _smooth_child_volume_probability(volume_fractions: np.ndarray) -> float:
    if len(volume_fractions) <= 1:
        return 1.0
    expected_fraction = 1.0 / len(volume_fractions)
    log_ratio = np.log(
        np.clip(volume_fractions / expected_fraction, 1e-8, None)
    )
    return float(np.mean(np.exp(-0.5 * (log_ratio / 0.85) ** 2)))


def build_watershed_hypothesis(
    mask: np.ndarray,
    watershed_distance: np.ndarray,
    selected_peaks: tuple[PeakCandidate, ...],
) -> np.ndarray:
    """Build a marker-controlled local watershed for selected lobe peaks."""

    marker_image = np.zeros(mask.shape, dtype=np.int32)
    for marker_label, peak in enumerate(selected_peaks, start=1):
        if not mask[peak.position_zyx]:
            raise ValueError(f"marker {peak.position_zyx} is outside the component")
        marker_image[peak.position_zyx] = marker_label
    return watershed(
        -watershed_distance,
        markers=marker_image,
        mask=mask,
        watershed_line=False,
    ).astype(np.int32)


def hard_gate_hypothesis(
    labels: np.ndarray,
    mask: np.ndarray,
    selected_peaks: tuple[PeakCandidate, ...],
    config: SegmentationConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Reject only invalid partitions and clearly unsafe split children."""

    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    if labels.shape != mask.shape:
        return False, ("shape_mismatch",)

    reasons: list[str] = []
    if np.any(labels[~mask] != 0):
        reasons.append("labels_outside_mask")
    if np.any(labels[mask] <= 0):
        reasons.append("mask_not_fully_partitioned")

    positive_labels = tuple(
        int(value) for value in np.unique(labels) if int(value) > 0
    )
    expected_k = len(selected_peaks)
    if len(positive_labels) != expected_k:
        reasons.append("wrong_child_count")

    if expected_k > 1:
        minimum_separation = min(
            physical_distance(
                first.position_zyx,
                second.position_zyx,
                config.voxel_size_zyx_um,
            )
            for first, second in combinations(selected_peaks, 2)
        )
        if minimum_separation < config.hard_min_marker_separation_um:
            reasons.append("sub_resolution_marker_spacing")

    total_voxels = max(int(np.count_nonzero(mask)), 1)
    connectivity = ndimage.generate_binary_structure(3, 1)
    for label_value in positive_labels:
        region = labels == label_value
        voxel_count = int(np.count_nonzero(region))
        _, component_count = ndimage.label(region, structure=connectivity)
        if int(component_count) != 1:
            reasons.append(f"child_{label_value}_disconnected")
        if expected_k <= 1:
            continue
        if voxel_count < config.hard_min_child_voxels:
            reasons.append(f"child_{label_value}_too_small")
            continue
        if voxel_count / total_voxels < config.hard_min_child_fraction(expected_k):
            reasons.append(f"child_{label_value}_fraction_too_small")
        try:
            radius_um = describe_cell_mask(
                region, config.voxel_size_zyx_um
            ).equivalent_radius_um
        except ValueError:
            reasons.append(f"child_{label_value}_not_measurable")
            continue
        if radius_um < config.hard_min_equivalent_radius_um:
            reasons.append(f"child_{label_value}_sub_resolution")

    for marker_label, peak in enumerate(selected_peaks, start=1):
        if not mask[peak.position_zyx]:
            reasons.append(f"marker_{marker_label}_outside_mask")
        elif int(labels[peak.position_zyx]) != marker_label:
            reasons.append(f"marker_{marker_label}_label_mismatch")
    return not reasons, tuple(reasons)


def _invalid_hypothesis(
    labels: np.ndarray,
    selected_peaks: tuple[PeakCandidate, ...],
    reasons: tuple[str, ...],
) -> SplitHypothesis:
    empty_evidence = HypothesisEvidence(*(0.0 for _ in range(8)))
    return SplitHypothesis(
        len(selected_peaks),
        labels,
        selected_peaks,
        False,
        reasons,
        0.0,
        -np.inf,
        -np.inf,
        0.0,
        empty_evidence,
        0.0,
    )


def score_watershed_hypothesis(
    labels: np.ndarray,
    selected_peaks: tuple[PeakCandidate, ...],
    effective_peaks: tuple[PeakCandidate, ...],
    pair_evidence: tuple[PairEvidence, ...],
    distance_um: np.ndarray,
    merged_description: CellShape,
    mask: np.ndarray,
    config: SegmentationConfig,
) -> SplitHypothesis:
    """Combine prior and soft spatial evidence for one watershed hypothesis."""

    hard_valid, hard_reasons = hard_gate_hypothesis(
        labels, mask, selected_peaks, config
    )
    if not hard_valid:
        return _invalid_hypothesis(labels, selected_peaks, hard_reasons)

    epsilon = config.probability_epsilon
    positive_labels = tuple(
        int(value) for value in np.unique(labels) if int(value) > 0
    )
    region_descriptions = {
        value: describe_cell_mask(labels == value, config.voxel_size_zyx_um)
        for value in positive_labels
    }
    k = len(region_descriptions)
    child_volumes = np.asarray(
        [region_descriptions[value].volume_um3 for value in positive_labels],
        dtype=float,
    )
    volume_fractions = child_volumes / max(float(child_volumes.sum()), 1e-12)
    minimum_child_fraction = float(volume_fractions.min())
    soft_fraction_center = {1: 0.50, 2: 0.14, 3: 0.075}.get(k, 0.05)
    fragment_safety = (
        1.0
        if k == 1
        else float(expit((minimum_child_fraction - soft_fraction_center) / 0.045))
    )
    child_volume = _smooth_child_volume_probability(volume_fractions)
    child_shape, shape_improvement = _child_shape_evidence(
        region_descriptions, merged_description, epsilon
    )

    pair_lookup = pair_probability_lookup(pair_evidence)
    selected_probabilities = selected_pair_probabilities(selected_peaks, pair_lookup)
    selected_ids = {peak.peak_id for peak in selected_peaks}
    unexplained_probabilities: list[float] = []
    for candidate in effective_peaks:
        if candidate.peak_id in selected_ids:
            continue
        unexplained_probabilities.append(
            _geometric_mean_probability(
                [
                    pair_lookup.get(
                        frozenset((candidate.peak_id, selected.peak_id)), 0.0
                    )
                    for selected in selected_peaks
                ],
                epsilon,
            )
        )
    maximum_unexplained = max(unexplained_probabilities, default=0.0)
    coverage_support = float(np.clip(1.0 - maximum_unexplained, epsilon, 1.0))

    if k == 1:
        effective_probabilities = selected_pair_probabilities(
            effective_peaks, pair_lookup
        )
        lobe_support = float(
            np.clip(1.0 - max(effective_probabilities, default=0.0), epsilon, 1.0)
        )
        neck_support = 0.75
        shape_improvement = 0.55
    else:
        lobe_support = _geometric_mean_probability(selected_probabilities, epsilon)
        surfaces = {
            value: region_surface_area_um2(
                labels == value, config.voxel_size_zyx_um
            )
            for value in positive_labels
        }
        marker_depth_by_label = {
            marker_label: float(distance_um[peak.position_zyx])
            for marker_label, peak in enumerate(selected_peaks, start=1)
        }
        pair_neck_probabilities: list[float] = []
        for label_a, label_b in combinations(positive_labels, 2):
            interface = pair_interface_statistics(
                labels,
                label_a,
                label_b,
                distance_um,
                config.voxel_size_zyx_um,
            )
            if not interface.contact:
                continue
            smaller_peak_depth = max(
                min(marker_depth_by_label[label_a], marker_depth_by_label[label_b]),
                1e-6,
            )
            saddle_ratio = interface.distance_median_um / smaller_peak_depth
            neck_depth_probability = float(
                expit(((1.0 - saddle_ratio) - 0.22) / 0.10)
            )
            normalized_area = interface.interface_area_um2 / max(
                min(surfaces[label_a], surfaces[label_b]), 1e-6
            )
            interface_probability = float(expit((0.16 - normalized_area) / 0.045))
            pair_neck_probabilities.append(
                _geometric_mean_probability(
                    [neck_depth_probability, interface_probability], epsilon
                )
            )
        neck_support = _geometric_mean_probability(pair_neck_probabilities, epsilon)

    marker_quality = _geometric_mean_probability(
        [peak.persistence_score for peak in selected_peaks], epsilon
    )
    evidence = HypothesisEvidence(
        lobe_support,
        coverage_support,
        marker_quality,
        neck_support,
        child_shape,
        shape_improvement,
        child_volume,
        fragment_safety,
    )
    prior = config.hypothesis_prior(k)
    log_likelihood = sum(
        config.evidence_weights()[name]
        * np.log(float(np.clip(probability, epsilon, 1.0)))
        for name, probability in evidence.as_dict().items()
    )
    log_posterior = float(np.log(max(prior, epsilon)) + log_likelihood)
    return SplitHypothesis(
        k,
        labels,
        selected_peaks,
        True,
        (),
        prior,
        float(log_likelihood),
        log_posterior,
        0.0,
        evidence,
        minimum_child_fraction,
    )


def minimum_marker_separation_um(
    selected_peaks: tuple[PeakCandidate, ...],
    config: SegmentationConfig,
) -> float:
    """Return minimum physical marker separation, or infinity for H1."""

    if len(selected_peaks) < 2:
        return np.inf
    return min(
        physical_distance(
            first.position_zyx,
            second.position_zyx,
            config.voxel_size_zyx_um,
        )
        for first, second in combinations(selected_peaks, 2)
    )


def _combination_prescore(
    selected_peaks: tuple[PeakCandidate, ...],
    pair_lookup: dict[frozenset[int], float],
    epsilon: float,
) -> float:
    marker_quality = _geometric_mean_probability(
        [peak.persistence_score for peak in selected_peaks], epsilon
    )
    lobe_support = _geometric_mean_probability(
        selected_pair_probabilities(selected_peaks, pair_lookup), epsilon
    )
    return float(0.70 * lobe_support + 0.30 * marker_quality)


def _evaluate_combinations(
    k: int,
    candidate_pool: tuple[PeakCandidate, ...],
    mask: np.ndarray,
    distance_um: np.ndarray,
    watershed_distance: np.ndarray,
    pair_evidence: tuple[PairEvidence, ...],
    merged_description: CellShape,
    config: SegmentationConfig,
) -> list[SplitHypothesis]:
    pair_lookup = pair_probability_lookup(pair_evidence)
    ranked: list[tuple[float, tuple[PeakCandidate, ...]]] = []
    for selected in combinations(candidate_pool, k):
        if minimum_marker_separation_um(selected, config) < (
            config.hard_min_marker_separation_um
        ):
            continue
        if k == 3:
            probabilities = selected_pair_probabilities(selected, pair_lookup)
            if len(probabilities) != 3:
                continue
            if min(probabilities) < config.k3_generation_min_pair_probability:
                continue
            if _geometric_mean_probability(
                probabilities, config.probability_epsilon
            ) < config.k3_generation_min_geometric_probability:
                continue
        ranked.append(
            (
                _combination_prescore(selected, pair_lookup, config.probability_epsilon),
                selected,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0],
            tuple(peak.peak_id for peak in item[1]),
        )
    )

    hypotheses: list[SplitHypothesis] = []
    for prescore, selected in ranked[: config.max_combinations_per_k]:
        labels = build_watershed_hypothesis(mask, watershed_distance, selected)
        result = score_watershed_hypothesis(
            labels,
            selected,
            candidate_pool,
            pair_evidence,
            distance_um,
            merged_description,
            mask,
            config,
        )
        hypotheses.append(replace(result, combination_prescore=float(prescore)))
    return hypotheses


def evaluate_spatial_split_hypotheses(
    mask: np.ndarray,
    distance_um: np.ndarray,
    watershed_distance: np.ndarray,
    effective_peaks: tuple[PeakCandidate, ...],
    pair_evidence: tuple[PairEvidence, ...],
    merged_description: CellShape,
    config: SegmentationConfig,
) -> HypothesisEvaluation:
    """Evaluate H1/H2 first, then H3 only for three independent lobes."""

    candidate_pool = effective_peaks[: config.max_candidate_peaks]
    if not candidate_pool:
        raise ValueError("no effective peak remains after same-lobe collapsing")

    h1 = score_watershed_hypothesis(
        mask.astype(np.int32),
        (candidate_pool[0],),
        candidate_pool,
        pair_evidence,
        distance_um,
        merged_description,
        mask,
        config,
    )
    all_hypotheses: list[SplitHypothesis] = [h1]

    h2_hypotheses: list[SplitHypothesis] = []
    if len(candidate_pool) >= 2:
        h2_hypotheses = _evaluate_combinations(
            2,
            candidate_pool,
            mask,
            distance_um,
            watershed_distance,
            pair_evidence,
            merged_description,
            config,
        )
        all_hypotheses.extend(h2_hypotheses)

    # The common one-vs-two decision has now been evaluated. H3 is considered
    # only if a valid H2 exists and all three pairwise lobe probabilities show
    # that a third independently supported branch remains.
    if (
        config.max_cells >= 3
        and len(candidate_pool) >= 3
        and any(hypothesis.hard_valid for hypothesis in h2_hypotheses)
    ):
        all_hypotheses.extend(
            _evaluate_combinations(
                3,
                candidate_pool,
                mask,
                distance_um,
                watershed_distance,
                pair_evidence,
                merged_description,
                config,
            )
        )

    valid = [
        hypothesis
        for hypothesis in all_hypotheses
        if hypothesis.hard_valid
        and np.isfinite(hypothesis.log_posterior_unnormalized)
    ]
    if not valid:
        raise RuntimeError("every spatial hypothesis failed broad safety gates")

    best: list[SplitHypothesis] = []
    for k in sorted({hypothesis.k for hypothesis in valid}):
        best.append(
            max(
                (hypothesis for hypothesis in valid if hypothesis.k == k),
                key=lambda hypothesis: hypothesis.log_posterior_unnormalized,
            )
        )
    log_values = np.asarray(
        [hypothesis.log_posterior_unnormalized for hypothesis in best], dtype=float
    )
    normalized_logs = log_values - logsumexp(log_values)
    normalized_best = tuple(
        replace(hypothesis, posterior_probability=float(np.exp(log_probability)))
        for hypothesis, log_probability in zip(best, normalized_logs)
    )
    return HypothesisEvaluation(normalized_best, tuple(all_hypotheses))


def choose_hierarchical_hypothesis(
    best_hypotheses: tuple[SplitHypothesis, ...],
    config: SegmentationConfig,
) -> HypothesisDecision:
    """Choose H1/H2 first and promote to H3 only with strong extra evidence."""

    by_k = {hypothesis.k: hypothesis for hypothesis in best_hypotheses}
    if 1 not in by_k:
        raise ValueError("an H1 hypothesis is required")
    h1 = by_k[1]
    h2 = by_k.get(2)
    h3 = by_k.get(3)
    epsilon = config.probability_epsilon
    p1 = h1.posterior_probability
    p2 = h2.posterior_probability if h2 is not None else 0.0
    p3 = h3.posterior_probability if h3 is not None else 0.0

    h2_conditional = p2 / max(p1 + p2, epsilon) if h2 is not None else 0.0
    h2_odds = p2 / max(p1, epsilon) if h2 is not None else 0.0
    h2_accepted = bool(
        h2 is not None
        and h2_conditional >= config.h2_min_conditional_probability
        and h2_odds >= config.h2_min_odds_vs_h1
    )
    if h2_accepted:
        chosen = h2
        status = "accepted_two_cell_split"
    else:
        chosen = h1
        status = (
            "uncertain_no_split"
            if h2 is not None
            and h2_conditional > config.h1_confident_conditional_probability
            else "accepted_single"
        )

    h3_conditional = (
        p3 / max(p2 + p3, epsilon) if h3 is not None and h2 is not None else 0.0
    )
    h3_odds = p3 / max(p2, epsilon) if h3 is not None and h2 is not None else 0.0
    h3_accepted = bool(
        h2_accepted
        and h3 is not None
        and h3_conditional >= config.h3_min_conditional_probability
        and h3_odds >= config.h3_min_odds_vs_h2
    )
    if h3_accepted:
        chosen = h3
        status = "accepted_three_cell_split"

    return HypothesisDecision(
        chosen,
        status,
        float(h2_conditional),
        float(h2_odds),
        float(h3_conditional),
        float(h3_odds),
    )
