"""Production effective-EDT plus geometric-marker 3-D segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import DEFAULT_SEGMENTATION_CONFIG, SegmentationConfig
from .distance import compute_distance_transform
from .marker_completion import (
    combine_markers,
    convert_effective_peaks_to_markers,
    safely_complete_geometric_markers,
)
from .models import GeometricCompletionResult, InstanceMarker
from .peaks import (
    DistancePeakAnalysis,
    LobeCollapseResult,
    PairEvidence,
    PeakCandidate,
    build_peak_pair_evidence,
    collapse_same_lobe_peaks,
    detect_persistent_distance_peaks,
)
from .watershed import build_marker_watershed


VOXEL_SIZE = DEFAULT_SEGMENTATION_CONFIG.voxel_size_zyx_um


@dataclass(frozen=True)
class ComponentDiagnostic:
    """Compact record of one component-level production result."""

    component_id: int
    source_voxels: int
    raw_peak_count: int
    effective_peak_count: int
    surface_cap_count: int
    body_candidate_count: int
    valid_body_count: int
    selected_body_count: int
    represented_body_count: int
    unrepresented_body_count: int
    supplemental_marker_count: int
    final_marker_count: int
    instance_count: int
    marker_count: int
    marker_positions_zyx: tuple[tuple[int, int, int], ...]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    processing_status: str
    geometry_processing_status: str
    geometry_error: str | None
    error: str | None


@dataclass(frozen=True)
class InstanceComponentMapping:
    """Relationship between deterministic output IDs and source components."""

    component_id: int
    local_child_label: int
    instance_id: int
    voxel_count: int
    processing_status: str


@dataclass(frozen=True)
class SegmentationResult:
    """Detailed segmentation output for diagnostic or notebook workflows."""

    final_labels: np.ndarray
    markers: np.ndarray
    component_diagnostics: tuple[ComponentDiagnostic, ...]
    instance_component_map: tuple[InstanceComponentMapping, ...]
    component_debug_artifacts: tuple["ComponentDebugArtifacts", ...] = ()


@dataclass(frozen=True)
class ComponentDebugArtifacts:
    """Large arrays retained only for an explicitly diagnostic invocation."""

    component_id: int
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    component_mask: np.ndarray
    padded_component_mask: np.ndarray
    raw_distance: np.ndarray
    watershed_distance: np.ndarray
    merge_tree_distance: np.ndarray
    raw_peaks: tuple[PeakCandidate, ...]
    effective_peaks: tuple[PeakCandidate, ...]
    pair_evidence: tuple[PairEvidence, ...]
    final_markers: tuple[InstanceMarker, ...]
    geometric_completion: GeometricCompletionResult
    marker_positions_zyx: tuple[tuple[int, int, int], ...]
    final_labels: np.ndarray


@dataclass(frozen=True)
class ComponentAnalysis:
    """Internal all-effective-peak analysis for one unpadded component crop."""

    final_labels: np.ndarray
    marker_positions_zyx: tuple[tuple[int, int, int], ...]
    raw_peaks: tuple[PeakCandidate, ...]
    effective_peaks: tuple[PeakCandidate, ...]
    pair_evidence: tuple[PairEvidence, ...]
    final_markers: tuple[InstanceMarker, ...]
    geometric_completion: GeometricCompletionResult
    debug_artifacts: ComponentDebugArtifacts | None = None


def analyze_component_crop(
    component_mask: np.ndarray,
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
    *,
    retain_debug_artifacts: bool = False,
    force_geometric_analysis: bool = False,
) -> ComponentAnalysis:
    """Split one tight component crop using every effective peak as a marker."""

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
    effective_peaks = collapsed.effective_peaks
    if not effective_peaks:
        raise RuntimeError("peak collapse produced no effective peaks")

    effective_markers = convert_effective_peaks_to_markers(effective_peaks)
    geometric_completion = safely_complete_geometric_markers(
        padded_mask,
        peak_analysis,
        effective_peaks,
        config.geometric_completion,
        config.voxel_size_zyx_um,
        retain_debug_artifacts=retain_debug_artifacts,
        force_analysis=force_geometric_analysis,
    )

    def labels_for(markers: tuple[InstanceMarker, ...]) -> np.ndarray:
        if len(markers) == 1:
            return padded_mask.astype(np.int32)
        return build_marker_watershed(
            padded_mask,
            peak_analysis.watershed_distance,
            markers,
        )

    try:
        final_markers = combine_markers(
            effective_markers, geometric_completion.supplemental_markers
        )
        padded_labels = labels_for(final_markers)
    except Exception as error:
        # A geometric marker application failure is isolated just like a
        # geometric-analysis failure; the successful EDT result remains valid.
        if not geometric_completion.supplemental_markers:
            raise
        geometric_completion = GeometricCompletionResult.failed(
            RuntimeError(f"geometric marker application failed: {error}")
        )
        final_markers = effective_markers
        padded_labels = labels_for(final_markers)

    inner = tuple(
        slice(padding, -padding) if padding > 0 else slice(None)
        for _ in range(3)
    )
    final_labels = np.asarray(padded_labels[inner], dtype=np.int32)
    marker_positions = tuple(
        tuple(coordinate - padding for coordinate in marker.position_zyx)
        for marker in final_markers
    )
    _validate_local_result(component_mask, final_labels, marker_positions)

    debug_artifacts = None
    if retain_debug_artifacts:
        debug_artifacts = ComponentDebugArtifacts(
            component_id=0,
            bbox_zyx=(
                (0, component_mask.shape[0]),
                (0, component_mask.shape[1]),
                (0, component_mask.shape[2]),
            ),
            component_mask=component_mask,
            padded_component_mask=padded_mask,
            raw_distance=peak_analysis.raw_distance,
            watershed_distance=peak_analysis.watershed_distance,
            merge_tree_distance=peak_analysis.merge_tree_distance,
            raw_peaks=peak_analysis.peaks,
            effective_peaks=effective_peaks,
            pair_evidence=pair_evidence,
            final_markers=final_markers,
            geometric_completion=geometric_completion,
            marker_positions_zyx=marker_positions,
            final_labels=final_labels,
        )
    return ComponentAnalysis(
        final_labels=final_labels,
        marker_positions_zyx=marker_positions,
        raw_peaks=peak_analysis.peaks,
        effective_peaks=effective_peaks,
        pair_evidence=pair_evidence,
        final_markers=final_markers,
        geometric_completion=geometric_completion,
        debug_artifacts=debug_artifacts,
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
        raise RuntimeError("local labels do not cover the source component")
    if np.any(labels[~component_mask] != 0):
        raise RuntimeError("local labels extend outside the source component")

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
    marker_positions: tuple[tuple[int, int, int], ...],
    component_slice: tuple[slice, slice, slice],
    global_mask: np.ndarray,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    """Resolve and validate IDs and marker coordinates before global writes."""

    positive_labels = tuple(
        int(value) for value in np.unique(local_labels) if int(value) > 0
    )
    if len(positive_labels) != len(marker_positions):
        raise RuntimeError("marker count does not match local child count")
    starts = np.asarray(
        [axis_slice.start for axis_slice in component_slice], dtype=int
    )
    global_positions = tuple(
        tuple(
            int(value)
            for value in starts + np.asarray(local_position, dtype=int)
        )
        for local_position in marker_positions
    )
    if any(not global_mask[position] for position in global_positions):
        raise RuntimeError("a component marker lies outside the global mask")
    return positive_labels, global_positions


def segment_instances_detailed(
    binary_mask: np.ndarray,
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
    *,
    retain_debug_artifacts: bool = False,
    force_geometric_analysis: bool = False,
) -> SegmentationResult:
    """Segment a 3-D mask using all effective peaks and return aligned markers.

    Every 6-connected component is processed independently in a padded crop.
    Any unexpected component-level failure is isolated: the original component
    is emitted as one instance and the error is recorded.
    """

    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"binary_mask must be 3-D, received shape {mask.shape}")

    component_labels, _ = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(3, 1)
    )
    component_slices = ndimage.find_objects(component_labels)
    final_labels = np.zeros(mask.shape, dtype=np.int32)
    markers = np.zeros(mask.shape, dtype=np.int32)
    component_diagnostics: list[ComponentDiagnostic] = []
    instance_mapping: list[InstanceComponentMapping] = []
    component_debug_artifacts: list[ComponentDebugArtifacts] = []
    next_instance_id = 1

    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        local_component = component_labels[component_slice] == component_id
        source_voxels = int(np.count_nonzero(local_component))
        error_message: str | None = None

        try:
            analysis = analyze_component_crop(
                local_component,
                config,
                retain_debug_artifacts=retain_debug_artifacts,
                force_geometric_analysis=force_geometric_analysis,
            )
            local_labels = analysis.final_labels
            marker_positions = analysis.marker_positions_zyx
            processing_status = "processed"
            raw_peak_count = len(analysis.raw_peaks)
            effective_peak_count = len(analysis.effective_peaks)
            completion = analysis.geometric_completion
            surface_cap_count = len(completion.surface_caps)
            body_candidate_count = len(completion.body_candidates)
            valid_body_count = sum(
                body.valid for body in completion.body_candidates
            )
            selected_body_count = len(completion.selected_bodies)
            represented_body_count = sum(
                bool(body.represented_by_effective_peak_ids)
                for body in completion.selected_bodies
            )
            unrepresented_body_count = sum(
                not body.represented_by_effective_peak_ids
                for body in completion.selected_bodies
            )
            supplemental_marker_count = len(completion.supplemental_markers)
            final_marker_count = len(analysis.final_markers)
            geometry_processing_status = completion.processing_status
            geometry_error = completion.error
            positive_local_labels, global_positions = _prepare_component_output(
                local_labels,
                marker_positions,
                component_slice,
                mask,
            )
            if len(positive_local_labels) != final_marker_count:
                raise RuntimeError(
                    "successful component instance count differs from final marker count"
                )
        except Exception as error:  # component isolation is a required safety net
            local_labels, marker_positions = _fallback_component_analysis(
                local_component, config
            )
            processing_status = "fallback_single"
            raw_peak_count = 0
            effective_peak_count = 1
            surface_cap_count = 0
            body_candidate_count = 0
            valid_body_count = 0
            selected_body_count = 0
            represented_body_count = 0
            unrepresented_body_count = 0
            supplemental_marker_count = 0
            final_marker_count = 1
            geometry_processing_status = "not_run_due_to_edt_failure"
            geometry_error = None
            error_message = f"{type(error).__name__}: {error}"
            positive_local_labels, global_positions = _prepare_component_output(
                local_labels,
                marker_positions,
                component_slice,
                mask,
            )

        target_view = final_labels[component_slice]
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
                    processing_status,
                )
            )

        bbox = tuple(
            (int(axis_slice.start), int(axis_slice.stop))
            for axis_slice in component_slice
        )
        if error_message is None and analysis.debug_artifacts is not None:
            artifact = analysis.debug_artifacts
            component_debug_artifacts.append(
                ComponentDebugArtifacts(
                    component_id=component_id,
                    bbox_zyx=bbox,  # type: ignore[arg-type]
                    component_mask=artifact.component_mask,
                    padded_component_mask=artifact.padded_component_mask,
                    raw_distance=artifact.raw_distance,
                    watershed_distance=artifact.watershed_distance,
                    merge_tree_distance=artifact.merge_tree_distance,
                    raw_peaks=artifact.raw_peaks,
                    effective_peaks=artifact.effective_peaks,
                    pair_evidence=artifact.pair_evidence,
                    final_markers=artifact.final_markers,
                    geometric_completion=artifact.geometric_completion,
                    marker_positions_zyx=artifact.marker_positions_zyx,
                    final_labels=artifact.final_labels,
                )
            )
        instance_count = len(positive_local_labels)
        marker_count = len(global_positions)
        component_diagnostics.append(
            ComponentDiagnostic(
                component_id=component_id,
                source_voxels=source_voxels,
                raw_peak_count=raw_peak_count,
                effective_peak_count=effective_peak_count,
                surface_cap_count=surface_cap_count,
                body_candidate_count=body_candidate_count,
                valid_body_count=valid_body_count,
                selected_body_count=selected_body_count,
                represented_body_count=represented_body_count,
                unrepresented_body_count=unrepresented_body_count,
                supplemental_marker_count=supplemental_marker_count,
                final_marker_count=final_marker_count,
                instance_count=instance_count,
                marker_count=marker_count,
                marker_positions_zyx=global_positions,
                bbox_zyx=bbox,  # type: ignore[arg-type]
                processing_status=processing_status,
                geometry_processing_status=geometry_processing_status,
                geometry_error=geometry_error,
                error=error_message,
            )
        )
        if error_message is None:
            if final_marker_count != effective_peak_count + supplemental_marker_count:
                raise RuntimeError(
                    "final marker count differs from effective plus supplemental markers"
                )
            if instance_count != final_marker_count:
                raise RuntimeError(
                    "successful component instance count differs from final marker count"
                )
            if marker_count != final_marker_count:
                raise RuntimeError(
                    "successful component marker count differs from final marker count"
                )

    if np.any(final_labels[mask] <= 0):
        raise RuntimeError("final labels do not cover the complete binary mask")
    if np.any(final_labels[~mask] != 0):
        raise RuntimeError("final labels extend outside the binary mask")
    marker_values = markers[markers > 0]
    expected_ids = np.arange(1, int(final_labels.max()) + 1, dtype=np.int32)
    if not np.array_equal(np.sort(marker_values), expected_ids):
        raise RuntimeError("final marker IDs and instance IDs are not aligned")
    for marker_position in np.argwhere(markers > 0):
        position = tuple(int(value) for value in marker_position)
        if int(markers[position]) != int(final_labels[position]):
            raise RuntimeError("a final marker ID differs from its instance ID")

    return SegmentationResult(
        final_labels,
        markers,
        tuple(component_diagnostics),
        tuple(instance_mapping),
        tuple(component_debug_artifacts),
    )


def segment_instances(
    binary_mask: np.ndarray,
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
    *,
    return_diagnostics: bool = False,
):
    """Convert a 3-D foreground mask into deterministic instance labels."""

    result = segment_instances_detailed(binary_mask, config)
    if not return_diagnostics:
        return result.final_labels

    from src.diagnostics import DecisionRecord, StageTrace

    trace = StageTrace(
        stage_name="03_segmentation",
        inputs={"binary_mask": binary_mask},
        outputs={"instance_labels": result.final_labels},
        intermediates={"markers": result.markers},
        metrics={
            "components": len(result.component_diagnostics),
            "instances": int(result.final_labels.max()),
        },
        decisions=[
            DecisionRecord(
                decision_type="component_segmentation",
                outcome=record.processing_status,
                subject_id=record.component_id,
                reason=record.error,
                metrics={
                    "raw_peak_count": record.raw_peak_count,
                    "effective_peak_count": record.effective_peak_count,
                    "surface_cap_count": record.surface_cap_count,
                    "body_candidate_count": record.body_candidate_count,
                    "selected_body_count": record.selected_body_count,
                    "supplemental_marker_count": record.supplemental_marker_count,
                    "final_marker_count": record.final_marker_count,
                    "instance_count": record.instance_count,
                    "marker_count": record.marker_count,
                },
            )
            for record in result.component_diagnostics
        ],
    )
    return result.final_labels, trace
