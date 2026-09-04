"""Cheap peak-based proposals controlling expensive geometric completion.

This module intentionally uses only center-like evidence. It does not inspect
component size or surface geometry and never creates watershed markers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima

from .config import CenterCandidateConfig
from .distance import physical_distance, physical_sigma_voxels, validate_voxel_size
from .models import (
    CandidateDebugArtifacts,
    CenterProposal,
    GeometricCandidateResult,
    ShapePeakCandidate,
)
from .peaks import (
    DistancePeakAnalysis,
    LobeCollapseResult,
    PairEvidence,
    PeakCandidate,
)


@dataclass(frozen=True)
class ShapePeakDetection:
    """One per-scale regional maximum before multiscale consolidation."""

    position_zyx: tuple[int, int, int]
    sigma_um: float
    response: float
    relative_response: float


@dataclass(frozen=True)
class _ProposalSource:
    source: str
    source_id: int
    position_zyx: tuple[float, float, float]
    position_um: tuple[float, float, float]


def _padding_for_sigma(
    sigma_um: float,
    spacing: np.ndarray,
    multiplier: float,
) -> tuple[int, int, int]:
    padding_um = float(multiplier) * float(sigma_um)
    return tuple(max(int(np.ceil(padding_um / value)), 1) for value in spacing)


def _physical_ball_footprint(radius_um: float, spacing: np.ndarray) -> np.ndarray:
    radii_voxels = np.ceil(float(radius_um) / spacing).astype(int)
    z, y, x = np.ogrid[
        -radii_voxels[0] : radii_voxels[0] + 1,
        -radii_voxels[1] : radii_voxels[1] + 1,
        -radii_voxels[2] : radii_voxels[2] + 1,
    ]
    return (
        (z * spacing[0]) ** 2
        + (y * spacing[1]) ** 2
        + (x * spacing[2]) ** 2
        <= float(radius_um) ** 2 + 1e-12
    )


def physical_binary_log_response(
    component_mask: np.ndarray,
    sigma_um: float,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    *,
    padding_sigma_multiplier: float = 4.0,
    padding_reference_sigma_um: float | None = None,
) -> np.ndarray:
    """Return a scale-normalized physical 3-D LoG response on the input crop.

    Gaussian derivatives are calculated in voxel coordinates and divided by
    the squared physical voxel size on each axis before summation. Positive
    responses are retained only inside the binary component.
    """

    mask = np.asarray(component_mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("component_mask must be a non-empty 3-D mask")
    spacing = validate_voxel_size(voxel_size_zyx_um)
    sigma_um = float(sigma_um)
    if not isfinite(sigma_um) or sigma_um <= 0:
        raise ValueError("sigma_um must be finite and positive")
    if not isfinite(padding_sigma_multiplier) or padding_sigma_multiplier <= 0:
        raise ValueError("padding_sigma_multiplier must be finite and positive")

    padding_reference = (
        sigma_um
        if padding_reference_sigma_um is None
        else float(padding_reference_sigma_um)
    )
    if not isfinite(padding_reference) or padding_reference < sigma_um:
        raise ValueError("padding_reference_sigma_um must be at least sigma_um")
    padding = _padding_for_sigma(
        padding_reference, spacing, padding_sigma_multiplier
    )
    padded = np.pad(
        mask,
        tuple((value, value) for value in padding),
        mode="constant",
        constant_values=False,
    )
    mask_float = padded.astype(float, copy=False)
    sigma_vox = physical_sigma_voxels(sigma_um, spacing)
    laplacian = np.zeros(padded.shape, dtype=float)
    derivative_orders = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    for axis, order in enumerate(derivative_orders):
        second_voxel = ndimage.gaussian_filter(
            mask_float,
            sigma=sigma_vox,
            order=order,
            mode="constant",
            cval=0.0,
        )
        laplacian += second_voxel / float(spacing[axis] ** 2)
    response_padded = -(sigma_um**2) * laplacian
    inner = tuple(
        slice(value, -value) if value else slice(None) for value in padding
    )
    response = np.asarray(response_padded[inner], dtype=float)
    return np.where(mask, np.maximum(response, 0.0), 0.0)


def _plateau_representatives(
    response: np.ndarray,
    maxima_mask: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    labels, count = ndimage.label(
        maxima_mask,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    representatives: list[tuple[int, int, int]] = []
    for label_id in range(1, count + 1):
        coordinates = np.argwhere(labels == label_id)
        if not len(coordinates):
            continue
        values = response[tuple(coordinates.T)]
        # np.argwhere is lexicographic and np.argmax returns the first tie.
        coordinate = coordinates[int(np.argmax(values))]
        representatives.append(tuple(int(value) for value in coordinate))
    return tuple(sorted(representatives))


def detect_shape_peaks_at_scale(
    component_mask: np.ndarray,
    sigma_um: float,
    config: CenterCandidateConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, tuple[ShapePeakDetection, ...]]:
    """Calculate one LoG response and deterministic regional maxima."""

    mask = np.asarray(component_mask, dtype=bool)
    response = physical_binary_log_response(
        mask,
        sigma_um,
        voxel_size_zyx_um,
        padding_sigma_multiplier=config.shape_padding_sigma_multiplier,
        padding_reference_sigma_um=max(config.shape_sigma_levels_um),
    )
    positive = response[mask & np.isfinite(response) & (response > 0)]
    if not len(positive):
        return response, ()
    maximum = float(np.max(positive))
    minimum = float(np.min(positive))
    response_range = max(maximum - minimum, maximum, np.finfo(float).eps)
    h_value = max(
        config.shape_peak_h_fraction * response_range,
        np.finfo(float).eps * max(maximum, 1.0),
    )
    maxima_mask = h_maxima(response, h_value) & mask
    detections = []
    for position in _plateau_representatives(response, maxima_mask):
        value = float(response[position])
        relative = value / max(maximum, np.finfo(float).eps)
        if relative + 1e-12 < config.shape_peak_min_relative_response:
            continue
        detections.append(
            ShapePeakDetection(position, float(sigma_um), value, float(relative))
        )
    return response, tuple(detections)


def _shape_detection_sort_key(
    detection: ShapePeakDetection,
) -> tuple[float, ...]:
    return (
        -detection.relative_response,
        -detection.response,
        -detection.sigma_um,
        float(detection.position_zyx[0]),
        float(detection.position_zyx[1]),
        float(detection.position_zyx[2]),
    )


def consolidate_shape_peak_detections(
    detections: tuple[ShapePeakDetection, ...] | list[ShapePeakDetection],
    config: CenterCandidateConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
) -> tuple[ShapePeakCandidate, ...]:
    """Complete-link cluster repeated physical blob centers across scales."""

    spacing = validate_voxel_size(voxel_size_zyx_um)
    clusters: list[list[ShapePeakDetection]] = []
    ordered = sorted(detections, key=_shape_detection_sort_key)
    for detection in ordered:
        compatible: list[tuple[float, int]] = []
        for cluster_index, cluster in enumerate(clusters):
            distances = [
                physical_distance(
                    detection.position_zyx,
                    member.position_zyx,
                    spacing,
                )
                for member in cluster
            ]
            maximum_distance = max(distances, default=0.0)
            if maximum_distance <= config.shape_peak_cluster_radius_um + 1e-12:
                compatible.append((maximum_distance, cluster_index))
        if compatible:
            clusters[min(compatible)[1]].append(detection)
        else:
            clusters.append([detection])

    unnumbered: list[ShapePeakCandidate] = []
    scale_count = max(len(config.shape_sigma_levels_um), 1)
    for cluster in clusters:
        representative = min(cluster, key=_shape_detection_sort_key)
        unique_scales = {record.sigma_um for record in cluster}
        scale_support = len(unique_scales) / scale_count
        if scale_support + 1e-12 < config.shape_peak_min_scale_support:
            continue
        position = representative.position_zyx
        position_um = tuple(float(value) for value in np.asarray(position) * spacing)
        unnumbered.append(
            ShapePeakCandidate(
                0,
                position,
                position_um,
                representative.sigma_um,
                representative.response,
                representative.relative_response,
                float(scale_support),
                len(cluster),
            )
        )
    ordered_candidates = sorted(
        unnumbered,
        key=lambda peak: (
            peak.position_zyx,
            -peak.relative_response,
            -peak.best_scale_um,
        ),
    )
    return tuple(
        replace(peak, peak_id=index)
        for index, peak in enumerate(ordered_candidates, start=1)
    )


def detect_multiscale_shape_peaks(
    component_mask: np.ndarray,
    config: CenterCandidateConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    *,
    retain_debug_artifacts: bool = False,
) -> tuple[
    tuple[ShapePeakCandidate, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    """Detect and consolidate binary-LoG maxima over fixed physical scales."""

    all_detections: list[ShapePeakDetection] = []
    retained_responses: list[np.ndarray] = []
    retained_maxima: list[np.ndarray] = []
    for sigma_um in config.shape_sigma_levels_um:
        response, detections = detect_shape_peaks_at_scale(
            component_mask,
            sigma_um,
            config,
            voxel_size_zyx_um,
        )
        all_detections.extend(detections)
        if retain_debug_artifacts:
            retained_responses.append(response)
            retained_maxima.append(
                np.asarray([item.position_zyx for item in detections], dtype=int).reshape(
                    (-1, 3)
                )
            )
    consolidated = consolidate_shape_peak_detections(
        all_detections,
        config,
        voxel_size_zyx_um,
    )
    return consolidated, tuple(retained_responses), tuple(retained_maxima)


def suppressed_raw_peaks(
    peak_analysis: DistancePeakAnalysis,
    effective_peaks: tuple[PeakCandidate, ...],
) -> tuple[PeakCandidate, ...]:
    """Return consolidated EDT peaks not retained as effective markers."""

    effective_ids = {peak.peak_id for peak in effective_peaks}
    return tuple(
        peak for peak in peak_analysis.peaks if peak.peak_id not in effective_ids
    )


def _complete_link_source_clusters(
    sources: list[_ProposalSource],
    match_radius_um: float,
) -> list[list[_ProposalSource]]:
    clusters: list[list[_ProposalSource]] = []
    ordered = sorted(
        sources,
        key=lambda item: (item.position_um, item.source, item.source_id),
    )
    for source in ordered:
        compatible: list[tuple[float, int]] = []
        source_position = np.asarray(source.position_um, dtype=float)
        for index, cluster in enumerate(clusters):
            maximum = max(
                np.linalg.norm(
                    source_position - np.asarray(member.position_um, dtype=float)
                )
                for member in cluster
            )
            if maximum <= match_radius_um + 1e-12:
                compatible.append((float(maximum), index))
        if compatible:
            clusters[min(compatible)[1]].append(source)
        else:
            clusters.append([source])
    return clusters


def _pair_for_raw_and_effective(
    raw_peak_id: int,
    effective_peak_id: int,
    pair_evidence: tuple[PairEvidence, ...],
) -> PairEvidence | None:
    target = frozenset((int(raw_peak_id), int(effective_peak_id)))
    matches = [
        pair
        for pair in pair_evidence
        if frozenset((pair.peak_id_a, pair.peak_id_b)) == target
    ]
    return min(
        matches,
        key=lambda pair: (
            -pair.branch_persistence,
            -pair.separation_support,
            pair.peak_id_a,
            pair.peak_id_b,
        ),
        default=None,
    )


def _route_reasons(
    has_raw: bool,
    has_shape: bool,
    represented: bool,
    raw: PeakCandidate | None,
    shape: ShapePeakCandidate | None,
    pair: PairEvidence | None,
    raw_depth_ratio: float,
    config: CenterCandidateConfig,
) -> tuple[str | None, bool, tuple[str, ...]]:
    if represented:
        return None, False, ("represented_by_effective_marker",)
    if has_raw and has_shape:
        failures = []
        if raw is None or raw.persistence_score < config.cross_min_raw_persistence:
            failures.append("cross_raw_persistence_too_low")
        if shape is None or shape.relative_response < config.cross_min_shape_relative_response:
            failures.append("cross_shape_response_too_low")
        if shape is None or shape.scale_support < config.cross_min_shape_scale_support:
            failures.append("cross_shape_scale_support_too_low")
        if not failures:
            return (
                "cross_transform",
                True,
                ("cross_transform_unrepresented_center",),
            )
        return None, False, tuple(failures)
    if has_shape:
        failures = []
        if shape is None or shape.relative_response < config.shape_only_min_relative_response:
            failures.append("shape_only_response_too_low")
        if shape is None or shape.scale_support < config.shape_only_min_scale_support:
            failures.append("shape_only_scale_support_too_low")
        if (
            shape is None
            or shape.local_depth_ratio < config.shape_only_min_local_depth_ratio
        ):
            failures.append("shape_only_not_local_edt_center")
        if not failures:
            return "shape_only", True, ("strong_shape_only_center",)
        return None, False, tuple(failures)
    if has_raw:
        failures = []
        if raw is None or raw.persistence_score < config.raw_only_min_persistence:
            failures.append("raw_persistence_too_low")
        if raw is None or raw.setting_support < config.raw_only_min_setting_support:
            failures.append("raw_setting_support_too_low")
        if raw_depth_ratio < config.raw_only_min_depth_ratio:
            failures.append("raw_depth_ratio_too_low")
        branch_pass = (
            pair is not None
            and pair.branch_persistence >= config.raw_only_min_branch_persistence
        )
        separation_pass = (
            pair is not None
            and pair.separation_support >= config.raw_only_min_separation_support
        )
        if not (branch_pass or separation_pass):
            failures.append("raw_branch_and_separation_too_low")
        if not failures:
            return "suppressed_edt", True, ("strong_suppressed_edt_center",)
        return None, False, tuple(failures)
    return None, False, ("proposal_has_no_peak_support",)


def build_center_proposals(
    suppressed_peaks: tuple[PeakCandidate, ...],
    shape_peaks: tuple[ShapePeakCandidate, ...],
    effective_peaks: tuple[PeakCandidate, ...],
    pair_evidence: tuple[PairEvidence, ...],
    config: CenterCandidateConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
) -> tuple[CenterProposal, ...]:
    """Cross-match compact peak collections and evaluate explicit routes."""

    if not effective_peaks:
        raise ValueError("candidate detection requires effective EDT peaks")
    spacing = validate_voxel_size(voxel_size_zyx_um)
    raw_by_id = {peak.peak_id: peak for peak in suppressed_peaks}
    shape_by_id = {peak.peak_id: peak for peak in shape_peaks}
    sources = [
        _ProposalSource(
            "raw",
            peak.peak_id,
            tuple(float(value) for value in peak.position_zyx),
            tuple(float(value) for value in np.asarray(peak.position_zyx) * spacing),
        )
        for peak in suppressed_peaks
    ]
    sources.extend(
        _ProposalSource(
            "shape",
            peak.peak_id,
            tuple(float(value) for value in peak.position_zyx),
            peak.position_um,
        )
        for peak in shape_peaks
    )
    clusters = _complete_link_source_clusters(
        sources, config.cross_transform_match_radius_um
    )
    effective_positions_um = (
        np.asarray([peak.position_zyx for peak in effective_peaks], dtype=float)
        * spacing
    )
    unnumbered: list[CenterProposal] = []
    epsilon = np.finfo(float).eps
    for cluster in clusters:
        raw_ids = tuple(sorted(item.source_id for item in cluster if item.source == "raw"))
        shape_ids = tuple(
            sorted(item.source_id for item in cluster if item.source == "shape")
        )
        raw_records = [raw_by_id[value] for value in raw_ids]
        shape_records = [shape_by_id[value] for value in shape_ids]
        raw = min(
            raw_records,
            key=lambda peak: (-peak.persistence_score, -peak.raw_depth_um, peak.peak_id),
            default=None,
        )
        shape = min(
            shape_records,
            key=lambda peak: (
                -peak.relative_response,
                -peak.scale_support,
                -peak.best_scale_um,
                peak.peak_id,
            ),
            default=None,
        )
        weights = []
        positions_um = []
        for item in cluster:
            if item.source == "raw":
                weight = max(raw_by_id[item.source_id].persistence_score, 0.05)
            else:
                record = shape_by_id[item.source_id]
                weight = max(record.relative_response * record.scale_support, 0.05)
            weights.append(weight)
            positions_um.append(item.position_um)
        proposal_um_array = np.average(
            np.asarray(positions_um, dtype=float),
            axis=0,
            weights=np.asarray(weights, dtype=float),
        )
        proposal_zyx_array = proposal_um_array / spacing
        distances = np.linalg.norm(
            effective_positions_um - proposal_um_array[None, :], axis=1
        )
        nearest_index = min(
            range(len(effective_peaks)),
            key=lambda index: (float(distances[index]), effective_peaks[index].peak_id),
        )
        nearest = effective_peaks[nearest_index]
        nearest_distance = float(distances[nearest_index])
        pair = (
            _pair_for_raw_and_effective(raw.peak_id, nearest.peak_id, pair_evidence)
            if raw is not None
            else None
        )
        raw_depth_ratio = (
            float(raw.raw_depth_um / max(nearest.raw_depth_um, epsilon))
            if raw is not None
            else 0.0
        )
        radius_estimates = []
        if raw is not None:
            radius_estimates.append(raw.raw_depth_um)
        if shape is not None:
            radius_estimates.append(
                config.proposal_radius_from_sigma_factor * shape.best_scale_um
            )
        proposal_radius = max(radius_estimates, default=epsilon)
        normalized_separation = nearest_distance / max(
            proposal_radius + nearest.raw_depth_um, epsilon
        )
        unrepresented = (
            nearest_distance + 1e-12 >= config.proposal_min_absolute_separation_um
            and normalized_separation + 1e-12
            >= config.proposal_min_normalized_separation
        )
        represented = not unrepresented
        route, candidate, reasons = _route_reasons(
            bool(raw_records),
            bool(shape_records),
            represented,
            raw,
            shape,
            pair,
            raw_depth_ratio,
            config,
        )
        support_values = [
            normalized_separation,
            raw.persistence_score if raw is not None else 0.0,
            shape.relative_response if shape is not None else 0.0,
            shape.scale_support if shape is not None else 0.0,
        ]
        score = float(np.clip(np.mean(support_values), 0.0, 1.0))
        unnumbered.append(
            CenterProposal(
                0,
                tuple(float(value) for value in proposal_zyx_array),
                tuple(float(value) for value in proposal_um_array),
                raw_ids,
                shape_ids,
                raw.raw_depth_um if raw is not None else 0.0,
                raw.smoothed_depth_um if raw is not None else 0.0,
                raw_depth_ratio,
                raw.persistence_score if raw is not None else 0.0,
                raw.scale_support if raw is not None else 0.0,
                raw.h_support if raw is not None else 0.0,
                raw.setting_support if raw is not None else 0.0,
                raw.detection_count if raw is not None else 0,
                pair.branch_persistence if pair is not None else 0.0,
                pair.branch_balance if pair is not None else 0.0,
                pair.separation_support if pair is not None else 0.0,
                pair.peak_support if pair is not None else 0.0,
                pair.distinct_lobe_probability if pair is not None else 0.0,
                shape.response if shape is not None else 0.0,
                shape.relative_response if shape is not None else 0.0,
                shape.best_scale_um if shape is not None else 0.0,
                shape.scale_support if shape is not None else 0.0,
                shape.detection_count if shape is not None else 0,
                shape.interior_depth_um if shape is not None else 0.0,
                shape.local_depth_ratio if shape is not None else 0.0,
                nearest.peak_id,
                nearest_distance,
                float(normalized_separation),
                represented,
                route,
                candidate,
                score,
                reasons,
            )
        )
    ordered_proposals = sorted(
        unnumbered,
        key=lambda proposal: (
            proposal.position_zyx,
            proposal.raw_peak_ids,
            proposal.shape_peak_ids,
        ),
    )
    return tuple(
        replace(proposal, proposal_id=index)
        for index, proposal in enumerate(ordered_proposals, start=1)
    )


def _component_reasons(proposals: tuple[CenterProposal, ...]) -> tuple[str, ...]:
    if not proposals:
        return ("no_center_proposals",)
    candidates = [proposal for proposal in proposals if proposal.candidate]
    if candidates:
        return tuple(
            dict.fromkeys(reason for proposal in candidates for reason in proposal.reasons)
        )
    if all(proposal.represented for proposal in proposals):
        return ("all_proposals_represented",)
    unrepresented = [proposal for proposal in proposals if not proposal.represented]
    reasons = []
    if any(proposal.shape_peak_ids for proposal in unrepresented):
        reasons.append("shape_peaks_too_weak")
    if any(proposal.raw_peak_ids for proposal in unrepresented):
        reasons.append("raw_peaks_too_weak")
    reasons.append("no_unrepresented_center_proposal")
    return tuple(dict.fromkeys(reasons))


def detect_geometric_candidate(
    component_mask: np.ndarray,
    peak_analysis: DistancePeakAnalysis,
    pair_evidence: tuple[PairEvidence, ...],
    collapsed: LobeCollapseResult,
    effective_peaks: tuple[PeakCandidate, ...],
    config: CenterCandidateConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    *,
    retain_debug_artifacts: bool = False,
) -> GeometricCandidateResult:
    """Detect unrepresented center proposals without running surface geometry."""

    mask = np.asarray(component_mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("component_mask must be a non-empty 3-D mask")
    if peak_analysis.raw_distance.shape != mask.shape:
        raise ValueError("peak-analysis arrays must align with component_mask")
    if not effective_peaks:
        raise ValueError("candidate detection requires effective EDT peaks")
    effective_ids = {peak.peak_id for peak in effective_peaks}
    collapsed_ids = {peak.peak_id for peak in collapsed.effective_peaks}
    if effective_ids != collapsed_ids:
        raise ValueError("effective_peaks must match the lobe-collapse result")

    shape_peaks, responses, raw_maxima = detect_multiscale_shape_peaks(
        mask,
        config,
        voxel_size_zyx_um,
        retain_debug_artifacts=retain_debug_artifacts,
    )
    spacing = validate_voxel_size(voxel_size_zyx_um)
    center_footprint = _physical_ball_footprint(
        config.shape_center_neighborhood_radius_um,
        spacing,
    )
    local_depth_maximum = ndimage.maximum_filter(
        peak_analysis.raw_distance,
        footprint=center_footprint,
        mode="constant",
        cval=0.0,
    )
    shape_peaks = tuple(
        replace(
            peak,
            interior_depth_um=float(peak_analysis.raw_distance[peak.position_zyx]),
            local_depth_ratio=float(
                peak_analysis.raw_distance[peak.position_zyx]
                / max(local_depth_maximum[peak.position_zyx], np.finfo(float).eps)
            ),
        )
        for peak in shape_peaks
    )
    suppressed = suppressed_raw_peaks(peak_analysis, effective_peaks)
    proposals = build_center_proposals(
        suppressed,
        shape_peaks,
        effective_peaks,
        pair_evidence,
        config,
        voxel_size_zyx_um,
    )
    candidate_ids = tuple(
        proposal.proposal_id for proposal in proposals if proposal.candidate
    )
    reasons = _component_reasons(proposals)
    debug = None
    if retain_debug_artifacts:
        debug = CandidateDebugArtifacts(
            tuple(float(value) for value in config.shape_sigma_levels_um),
            responses,
            raw_maxima,
            shape_peaks,
            proposals,
            candidate_ids,
        )
    return GeometricCandidateResult(
        shape_peaks,
        proposals,
        candidate_ids,
        bool(candidate_ids),
        "processed",
        None,
        reasons,
        debug,
    )


def safely_detect_geometric_candidate(*args, **kwargs) -> GeometricCandidateResult:
    """Isolate candidate failures from successful effective-EDT processing."""

    try:
        return detect_geometric_candidate(*args, **kwargs)
    except Exception as error:  # the safety boundary is an explicit contract
        return GeometricCandidateResult.failed(error)


_SHAPE_COLUMNS = (
    "component_id",
    "shape_peak_id",
    "z",
    "y",
    "x",
    "best_scale_um",
    "response",
    "relative_response",
    "scale_support",
    "detection_count",
    "interior_depth_um",
    "local_depth_ratio",
)


def shape_peaks_dataframe(result: GeometricCandidateResult, component_id: int):
    """Convert consolidated binary-LoG peaks to a reusable diagnostic table."""

    import pandas as pd

    rows = [
        {
            "component_id": int(component_id),
            "shape_peak_id": peak.peak_id,
            "z": peak.position_zyx[0],
            "y": peak.position_zyx[1],
            "x": peak.position_zyx[2],
            "best_scale_um": peak.best_scale_um,
            "response": peak.response,
            "relative_response": peak.relative_response,
            "scale_support": peak.scale_support,
            "detection_count": peak.detection_count,
            "interior_depth_um": peak.interior_depth_um,
            "local_depth_ratio": peak.local_depth_ratio,
        }
        for peak in result.shape_peaks
    ]
    return pd.DataFrame(rows, columns=_SHAPE_COLUMNS)


def center_proposals_dataframe(result: GeometricCandidateResult, component_id: int):
    """Convert every represented, rejected, and candidate proposal to a table."""

    import pandas as pd

    rows = []
    for proposal in result.proposals:
        source_types = ";".join(
            source
            for source, present in (
                ("suppressed_edt", bool(proposal.raw_peak_ids)),
                ("binary_log", bool(proposal.shape_peak_ids)),
            )
            if present
        )
        rows.append(
            {
                "component_id": int(component_id),
                "proposal_id": proposal.proposal_id,
                "source_types": source_types,
                "raw_peak_ids": ";".join(map(str, proposal.raw_peak_ids)),
                "shape_peak_ids": ";".join(map(str, proposal.shape_peak_ids)),
                "z": proposal.position_zyx[0],
                "y": proposal.position_zyx[1],
                "x": proposal.position_zyx[2],
                "raw_depth_um": proposal.raw_depth_um,
                "raw_smoothed_depth_um": proposal.raw_smoothed_depth_um,
                "raw_depth_ratio": proposal.raw_depth_ratio,
                "raw_persistence": proposal.raw_persistence,
                "raw_scale_support": proposal.raw_scale_support,
                "raw_h_support": proposal.raw_h_support,
                "raw_setting_support": proposal.raw_setting_support,
                "raw_detection_count": proposal.raw_detection_count,
                "branch_persistence": proposal.branch_persistence,
                "branch_balance": proposal.branch_balance,
                "separation_support": proposal.separation_support,
                "peak_support": proposal.peak_support,
                "distinct_lobe_probability": proposal.distinct_lobe_probability,
                "shape_response": proposal.shape_response,
                "shape_relative_response": proposal.shape_relative_response,
                "shape_best_scale_um": proposal.shape_best_scale_um,
                "shape_scale_support": proposal.shape_scale_support,
                "shape_detection_count": proposal.shape_detection_count,
                "shape_interior_depth_um": proposal.shape_interior_depth_um,
                "shape_local_depth_ratio": proposal.shape_local_depth_ratio,
                "nearest_effective_peak_id": proposal.nearest_effective_peak_id,
                "physical_separation_um": proposal.nearest_effective_distance_um,
                "normalized_separation": proposal.normalized_effective_separation,
                "represented": proposal.represented,
                "route": proposal.route,
                "candidate": proposal.candidate,
                "score": proposal.score,
                "reasons": ";".join(proposal.reasons),
            }
        )
    columns = (
        "component_id",
        "proposal_id",
        "source_types",
        "raw_peak_ids",
        "shape_peak_ids",
        "z",
        "y",
        "x",
        "raw_depth_um",
        "raw_smoothed_depth_um",
        "raw_depth_ratio",
        "raw_persistence",
        "raw_scale_support",
        "raw_h_support",
        "raw_setting_support",
        "raw_detection_count",
        "branch_persistence",
        "branch_balance",
        "separation_support",
        "peak_support",
        "distinct_lobe_probability",
        "shape_response",
        "shape_relative_response",
        "shape_best_scale_um",
        "shape_scale_support",
        "shape_detection_count",
        "shape_interior_depth_um",
        "shape_local_depth_ratio",
        "nearest_effective_peak_id",
        "physical_separation_um",
        "normalized_separation",
        "represented",
        "route",
        "candidate",
        "score",
        "reasons",
    )
    return pd.DataFrame(rows, columns=columns)


def candidate_summary_dataframe(
    result: GeometricCandidateResult,
    component_id: int,
    raw_peak_count: int,
    effective_peak_count: int,
):
    """Return one compact candidate-selection row without size evidence."""

    import pandas as pd

    return pd.DataFrame(
        [
            {
                "component_id": int(component_id),
                "raw_peak_count": int(raw_peak_count),
                "effective_peak_count": int(effective_peak_count),
                "suppressed_raw_peak_count": max(
                    int(raw_peak_count) - int(effective_peak_count), 0
                ),
                "shape_peak_count": len(result.shape_peaks),
                "proposal_count": len(result.proposals),
                "unrepresented_proposal_count": sum(
                    not proposal.represented for proposal in result.proposals
                ),
                "candidate_proposal_count": len(result.candidate_proposal_ids),
                "candidate": result.candidate,
                "candidate_routes": ";".join(
                    dict.fromkeys(
                        proposal.route
                        for proposal in result.proposals
                        if proposal.candidate and proposal.route is not None
                    )
                ),
                "processing_status": result.processing_status,
                "error": result.error,
                "reasons": ";".join(result.reasons),
            }
        ]
    )


__all__ = [
    "ShapePeakDetection",
    "build_center_proposals",
    "candidate_summary_dataframe",
    "center_proposals_dataframe",
    "consolidate_shape_peak_detections",
    "detect_geometric_candidate",
    "detect_multiscale_shape_peaks",
    "detect_shape_peaks_at_scale",
    "physical_binary_log_response",
    "safely_detect_geometric_candidate",
    "shape_peaks_dataframe",
    "suppressed_raw_peaks",
]
