"""Production entry points for probabilistic 3-D instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import DEFAULT_SEGMENTATION_CONFIG, SegmentationConfig
from .distance import compute_distance_transform
from .geometry import describe_cell_mask
from .hypotheses import (
    HypothesisDecision,
    HypothesisEvaluation,
    SplitHypothesis,
    choose_hierarchical_hypothesis,
    evaluate_spatial_split_hypotheses,
)
from .peaks import (
    DistancePeakAnalysis,
    LobeCollapseResult,
    PairEvidence,
    PeakCandidate,
    build_peak_pair_evidence,
    collapse_same_lobe_peaks,
    detect_persistent_distance_peaks,
)


VOXEL_SIZE = DEFAULT_SEGMENTATION_CONFIG.voxel_size_zyx_um


@dataclass(frozen=True)
class ComponentDiagnostic:
    """Compact record of one component-level split decision."""

    component_id: int
    source_voxels: int
    raw_peak_count: int
    effective_lobe_count: int
    selected_cell_count: int
    posterior_h1: float
    posterior_h2: float
    posterior_h3: float
    decision_status: str
    split_accepted: bool
    marker_positions_zyx: tuple[tuple[int, int, int], ...]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    error: str | None


@dataclass(frozen=True)
class HypothesisDiagnostic:
    """Optional compact evidence record for the best model of one size."""

    component_id: int
    k: int
    posterior_probability: float
    prior_probability: float
    log_likelihood: float
    lobe_support: float
    coverage_support: float
    marker_quality: float
    neck_support: float
    child_shape: float
    shape_improvement: float
    child_volume: float
    fragment_safety: float
    minimum_child_fraction: float
    selected_peak_ids: tuple[int, ...]


@dataclass(frozen=True)
class InstanceComponentMapping:
    """Relationship between deterministic output IDs and source components."""

    component_id: int
    local_child_label: int
    instance_id: int
    voxel_count: int
    decision_status: str


@dataclass(frozen=True)
class SegmentationResult:
    """Detailed segmentation output for diagnostic or notebook workflows."""

    instance_labels: np.ndarray
    markers: np.ndarray
    component_diagnostics: tuple[ComponentDiagnostic, ...]
    hypothesis_diagnostics: tuple[HypothesisDiagnostic, ...]
    instance_component_map: tuple[InstanceComponentMapping, ...]


@dataclass(frozen=True)
class ComponentAnalysis:
    """Internal spatial analysis for one unpadded component crop."""

    labels: np.ndarray
    selected_positions_zyx: tuple[tuple[int, int, int], ...]
    peaks: tuple[PeakCandidate, ...]
    effective_peaks: tuple[PeakCandidate, ...]
    pair_evidence: tuple[PairEvidence, ...]
    evaluation: HypothesisEvaluation
    decision: HypothesisDecision


def analyze_component_crop(
    component_mask: np.ndarray,
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
) -> ComponentAnalysis:
    """Run spatial-only probabilistic inference on one tight component crop."""

    component_mask = np.asarray(component_mask, dtype=bool)
    if component_mask.ndim != 3 or not component_mask.any():
        raise ValueError("component_mask must be a non-empty 3-D mask")

    padding = config.component_padding_voxels
    padded_mask = np.pad(
        component_mask,
        padding,
        mode="constant",
        constant_values=False,
    )
    peak_analysis: DistancePeakAnalysis = detect_persistent_distance_peaks(
        padded_mask, config
    )
    pair_evidence = build_peak_pair_evidence(
        peak_analysis.peaks,
        padded_mask,
        peak_analysis.merge_tree_distance,
        config,
    )
    collapsed: LobeCollapseResult = collapse_same_lobe_peaks(
        peak_analysis.peaks, pair_evidence, config
    )
    merged_description = describe_cell_mask(
        padded_mask, config.voxel_size_zyx_um
    )
    evaluation = evaluate_spatial_split_hypotheses(
        padded_mask,
        peak_analysis.raw_distance,
        peak_analysis.watershed_distance,
        collapsed.effective_peaks,
        pair_evidence,
        merged_description,
        config,
    )
    decision = choose_hierarchical_hypothesis(evaluation.best_by_k, config)

    inner = tuple(
        slice(padding, -padding) if padding > 0 else slice(None)
        for _ in range(3)
    )
    chosen_labels = np.asarray(decision.chosen.labels[inner], dtype=np.int32)
    selected_positions = tuple(
        tuple(coordinate - padding for coordinate in peak.position_zyx)
        for peak in decision.chosen.selected_peaks
    )
    _validate_local_result(component_mask, chosen_labels, selected_positions)
    return ComponentAnalysis(
        chosen_labels,
        selected_positions,
        peak_analysis.peaks,
        collapsed.effective_peaks,
        pair_evidence,
        evaluation,
        decision,
    )


def _validate_local_result(
    component_mask: np.ndarray,
    labels: np.ndarray,
    marker_positions: tuple[tuple[int, int, int], ...],
) -> None:
    """Validate local coverage plus one marker aligned to each child label."""

    if labels.shape != component_mask.shape:
        raise RuntimeError("local labels and component mask have different shapes")
    if np.any(labels[component_mask] <= 0):
        raise RuntimeError("local hypothesis does not cover the source component")
    if np.any(labels[~component_mask] != 0):
        raise RuntimeError("local hypothesis extends outside the source component")

    positive_labels = tuple(
        int(value) for value in np.unique(labels) if int(value) > 0
    )
    if len(marker_positions) != len(positive_labels):
        raise RuntimeError("marker count does not match local child count")
    shape = component_mask.shape
    marker_labels: list[int] = []
    for position in marker_positions:
        if not all(0 <= position[axis] < shape[axis] for axis in range(3)):
            raise RuntimeError("local marker lies outside the crop")
        if not component_mask[position]:
            raise RuntimeError("local marker lies outside the component")
        marker_labels.append(int(labels[position]))
    if tuple(marker_labels) != positive_labels:
        raise RuntimeError("marker IDs are not aligned with local child labels")


def _fallback_component_analysis(
    component_mask: np.ndarray,
    config: SegmentationConfig,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Preserve a failed component and place one marker at its deepest point."""

    padding = config.component_padding_voxels
    padded = np.pad(component_mask, padding, mode="constant", constant_values=False)
    distance = compute_distance_transform(padded, config.voxel_size_zyx_um)
    inner = tuple(
        slice(padding, -padding) if padding > 0 else slice(None)
        for _ in range(3)
    )
    local_distance = distance[inner]
    masked_distance = np.where(component_mask, local_distance, -np.inf)
    position = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(masked_distance)), masked_distance.shape
        )
    )
    labels = component_mask.astype(np.int32)
    positions = (position,)
    _validate_local_result(component_mask, labels, positions)
    return labels, positions


def _prepare_component_output(
    local_labels: np.ndarray,
    selected_positions: tuple[tuple[int, int, int], ...],
    component_slice: tuple[slice, slice, slice],
    global_mask: np.ndarray,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    """Resolve and validate IDs and marker coordinates before global writes."""

    positive_labels = tuple(
        int(value) for value in np.unique(local_labels) if int(value) > 0
    )
    if len(positive_labels) != len(selected_positions):
        raise RuntimeError("marker count does not match local child count")
    starts = np.asarray(
        [axis_slice.start for axis_slice in component_slice], dtype=int
    )
    global_positions = tuple(
        tuple(
            int(value)
            for value in starts + np.asarray(local_position, dtype=int)
        )
        for local_position in selected_positions
    )
    if any(not global_mask[position] for position in global_positions):
        raise RuntimeError("a component marker lies outside the global mask")
    return positive_labels, global_positions


def _posterior_by_k(evaluation: HypothesisEvaluation | None, k: int) -> float:
    if evaluation is None:
        return 0.0
    return next(
        (
            hypothesis.posterior_probability
            for hypothesis in evaluation.best_by_k
            if hypothesis.k == k
        ),
        0.0,
    )


def _hypothesis_diagnostic(
    component_id: int,
    hypothesis: SplitHypothesis,
) -> HypothesisDiagnostic:
    evidence = hypothesis.evidence
    return HypothesisDiagnostic(
        component_id=component_id,
        k=hypothesis.k,
        posterior_probability=hypothesis.posterior_probability,
        prior_probability=hypothesis.prior_probability,
        log_likelihood=hypothesis.log_likelihood,
        lobe_support=evidence.lobe_support,
        coverage_support=evidence.coverage_support,
        marker_quality=evidence.marker_quality,
        neck_support=evidence.neck_support,
        child_shape=evidence.child_shape,
        shape_improvement=evidence.shape_improvement,
        child_volume=evidence.child_volume,
        fragment_safety=evidence.fragment_safety,
        minimum_child_fraction=hypothesis.minimum_child_fraction,
        selected_peak_ids=tuple(peak.peak_id for peak in hypothesis.selected_peaks),
    )


def segment_instances_detailed(
    binary_mask: np.ndarray,
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
    *,
    include_hypothesis_diagnostics: bool = False,
) -> SegmentationResult:
    """Segment a 3-D mask and return aligned markers plus compact diagnostics.

    Every connected component is processed independently in a padded crop. Any
    unexpected component-level failure is isolated: the original component is
    emitted as one instance and the error is recorded.
    """

    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"binary_mask must be 3-D, received shape {mask.shape}")

    component_labels, _ = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(3, 1)
    )
    component_slices = ndimage.find_objects(component_labels)
    instance_labels = np.zeros(mask.shape, dtype=np.int32)
    markers = np.zeros(mask.shape, dtype=np.int32)
    component_diagnostics: list[ComponentDiagnostic] = []
    hypothesis_diagnostics: list[HypothesisDiagnostic] = []
    instance_mapping: list[InstanceComponentMapping] = []
    next_instance_id = 1

    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        local_component = component_labels[component_slice] == component_id
        source_voxels = int(np.count_nonzero(local_component))
        error_message: str | None = None

        try:
            analysis = analyze_component_crop(local_component, config)
            local_labels = analysis.labels
            selected_positions = analysis.selected_positions_zyx
            decision_status = analysis.decision.decision_status
            raw_peak_count = len(analysis.peaks)
            effective_lobe_count = len(analysis.effective_peaks)
            evaluation: HypothesisEvaluation | None = analysis.evaluation
            positive_local_labels, global_positions = _prepare_component_output(
                local_labels,
                selected_positions,
                component_slice,
                mask,
            )
        except Exception as error:  # component isolation is a required safety net
            local_labels, selected_positions = _fallback_component_analysis(
                local_component, config
            )
            decision_status = "fallback_single"
            raw_peak_count = 0
            effective_lobe_count = 1
            evaluation = None
            error_message = f"{type(error).__name__}: {error}"
            positive_local_labels, global_positions = _prepare_component_output(
                local_labels,
                selected_positions,
                component_slice,
                mask,
            )

        target_view = instance_labels[component_slice]

        for local_label, global_position in zip(
            positive_local_labels, global_positions
        ):
            instance_id = next_instance_id
            next_instance_id += 1
            region = local_labels == local_label
            target_view[region] = instance_id
            markers[global_position] = instance_id
            instance_mapping.append(
                InstanceComponentMapping(
                    component_id,
                    local_label,
                    instance_id,
                    int(np.count_nonzero(region)),
                    decision_status,
                )
            )

        bbox = tuple(
            (int(axis_slice.start), int(axis_slice.stop))
            for axis_slice in component_slice
        )
        component_diagnostics.append(
            ComponentDiagnostic(
                component_id=component_id,
                source_voxels=source_voxels,
                raw_peak_count=raw_peak_count,
                effective_lobe_count=effective_lobe_count,
                selected_cell_count=len(positive_local_labels),
                posterior_h1=_posterior_by_k(evaluation, 1),
                posterior_h2=_posterior_by_k(evaluation, 2),
                posterior_h3=_posterior_by_k(evaluation, 3),
                decision_status=decision_status,
                split_accepted=len(positive_local_labels) > 1,
                marker_positions_zyx=global_positions,
                bbox_zyx=bbox,  # type: ignore[arg-type]
                error=error_message,
            )
        )
        if include_hypothesis_diagnostics and evaluation is not None:
            hypothesis_diagnostics.extend(
                _hypothesis_diagnostic(component_id, hypothesis)
                for hypothesis in evaluation.best_by_k
            )

    if np.any(instance_labels[mask] <= 0):
        raise RuntimeError("final labels do not cover the complete binary mask")
    if np.any(instance_labels[~mask] != 0):
        raise RuntimeError("final labels extend outside the binary mask")
    marker_values = markers[markers > 0]
    expected_ids = np.arange(1, int(instance_labels.max()) + 1, dtype=np.int32)
    if not np.array_equal(np.sort(marker_values), expected_ids):
        raise RuntimeError("final marker IDs and instance IDs are not aligned")
    for marker_position in np.argwhere(markers > 0):
        position = tuple(int(value) for value in marker_position)
        if int(markers[position]) != int(instance_labels[position]):
            raise RuntimeError("a final marker ID differs from its instance ID")

    return SegmentationResult(
        instance_labels,
        markers,
        tuple(component_diagnostics),
        tuple(hypothesis_diagnostics),
        tuple(instance_mapping),
    )


def segment_instances(
    binary_mask: np.ndarray,
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
) -> np.ndarray:
    """Convert a 3-D foreground mask into deterministic instance labels.

    The return type and shape match the historical production interface. Use
    :func:`segment_instances_detailed` when markers or diagnostics are needed.
    """

    return segment_instances_detailed(binary_mask, config).instance_labels
