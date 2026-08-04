"""Effective-marker preservation and conservative geometric completion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from .config import GeometricCompletionConfig
from .distance import physical_distance, validate_voxel_size
from .geometric_bodies import (
    analyze_geometric_bodies,
    ellipsoid_normalized_distance,
)
from .models import (
    GeometricBody,
    GeometricCompletionResult,
    GeometricDebugArtifacts,
    InstanceMarker,
)
from .peaks import DistancePeakAnalysis, PeakCandidate
from .surface_geometry import analyze_surface_geometry


def convert_effective_peaks_to_markers(
    effective_peaks: tuple[PeakCandidate, ...],
) -> tuple[InstanceMarker, ...]:
    """Preserve effective-peak tuple order while removing watershed coupling."""

    return tuple(
        InstanceMarker(
            peak.position_zyx,
            "effective_edt",
            peak.peak_id,
            float(np.clip(peak.persistence_score, 0.0, 1.0)),
        )
        for peak in effective_peaks
    )


def combine_markers(
    effective_markers: tuple[InstanceMarker, ...],
    supplemental_markers: tuple[InstanceMarker, ...],
) -> tuple[InstanceMarker, ...]:
    """Keep EDT order and append deterministic, position-sorted supplements."""

    if any(marker.source != "effective_edt" for marker in effective_markers):
        raise ValueError("effective_markers contains a non-EDT marker")
    if any(
        marker.source != "geometric_completion" for marker in supplemental_markers
    ):
        raise ValueError("supplemental_markers contains a non-geometric marker")
    ordered_supplemental = tuple(
        sorted(
            supplemental_markers,
            key=lambda marker: (
                marker.position_zyx[0],
                marker.position_zyx[1],
                marker.position_zyx[2],
                marker.source_reference_id,
            ),
        )
    )
    positions = [marker.position_zyx for marker in effective_markers]
    positions.extend(marker.position_zyx for marker in ordered_supplemental)
    if len(set(positions)) != len(positions):
        raise ValueError("combined markers contain duplicate positions")
    return effective_markers + ordered_supplemental


def geometric_completion_eligible(
    component_mask: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    config: GeometricCompletionConfig,
    *,
    raw_peak_count: int | None = None,
    effective_marker_positions_zyx: tuple[tuple[int, int, int], ...] = (),
) -> bool:
    """Cheap runtime gate for shapes large enough to support two physical caps.

    This gate does not infer cell count and imposes no candidate/body-count cap.
    Suspicious multi-body components remain eligible regardless of EDT count.
    """

    coordinates = np.argwhere(component_mask)
    if len(coordinates) < 8:
        return False
    spacing = validate_voxel_size(voxel_size_zyx_um)
    extent = (coordinates.max(axis=0) - coordinates.min(axis=0) + 1) * spacing
    if float(np.max(extent)) < config.min_cap_separation_um:
        return False
    marker_count = max(len(effective_marker_positions_zyx), 1)
    if raw_peak_count is not None and int(raw_peak_count) > marker_count:
        return True
    physical_volume_um3 = len(coordinates) * float(np.prod(spacing))
    if (
        physical_volume_um3 / marker_count
        >= config.eligibility_min_volume_per_marker_um3
    ):
        return True
    if (
        float(np.max(extent)) / marker_count
        >= config.eligibility_min_extent_per_marker_um
    ):
        return True
    if effective_marker_positions_zyx:
        marker_um = np.asarray(effective_marker_positions_zyx, dtype=float) * spacing
        coordinates_um = coordinates.astype(float) * spacing
        nearest, _ = cKDTree(marker_um).query(coordinates_um, k=1)
        if (
            float(np.max(nearest))
            >= config.eligibility_max_unrepresented_distance_um
        ):
            return True
    return False


def _represented_peak_ids(
    body: GeometricBody,
    effective_peaks: tuple[PeakCandidate, ...],
    spacing: np.ndarray,
    radius: float,
) -> tuple[int, ...]:
    if not effective_peaks:
        return ()
    points_um = np.asarray(
        [peak.position_zyx for peak in effective_peaks], dtype=float
    ) * spacing
    q = ellipsoid_normalized_distance(
        points_um,
        np.asarray(body.center_um, dtype=float),
        body.rotation_matrix,
        np.asarray(body.semi_axes_um, dtype=float),
    )
    return tuple(
        peak.peak_id
        for peak, value in zip(effective_peaks, q)
        if float(value) <= radius
    )


def _safe_marker_for_body(
    body: GeometricBody,
    component_mask: np.ndarray,
    raw_distance: np.ndarray,
    existing_markers: list[InstanceMarker],
    spacing: np.ndarray,
    config: GeometricCompletionConfig,
) -> InstanceMarker | None:
    center_um = np.asarray(body.center_um, dtype=float)
    candidates_zyx = np.argwhere(component_mask)
    candidates_um = candidates_zyx.astype(float) * spacing
    distance_to_center = np.linalg.norm(candidates_um - center_um, axis=1)
    near = distance_to_center <= config.marker_search_radius_um
    candidates_zyx = candidates_zyx[near]
    candidates_um = candidates_um[near]
    distance_to_center = distance_to_center[near]
    if not len(candidates_zyx):
        return None
    q = ellipsoid_normalized_distance(
        candidates_um,
        center_um,
        body.rotation_matrix,
        np.asarray(body.semi_axes_um, dtype=float),
    )
    supported = q <= 1.0
    candidates_zyx = candidates_zyx[supported]
    distance_to_center = distance_to_center[supported]
    q = q[supported]
    if not len(candidates_zyx):
        return None
    occupied = {marker.position_zyx for marker in existing_markers}
    ranked: list[tuple[tuple[float, ...], tuple[int, int, int]]] = []
    for coordinate, normalized, center_distance in zip(
        candidates_zyx, q, distance_to_center
    ):
        position = tuple(int(value) for value in coordinate)
        if position in occupied:
            continue
        separations = [
            physical_distance(position, marker.position_zyx, spacing)
            for marker in existing_markers
        ]
        minimum_separation = min(separations, default=np.inf)
        if minimum_separation + 1e-12 < config.marker_min_separation_um:
            continue
        depth = float(raw_distance[position])
        key = (
            float(normalized),
            float(center_distance),
            -depth,
            -float(minimum_separation if np.isfinite(minimum_separation) else 1e6),
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        ranked.append((key, position))
    if not ranked:
        return None
    position = min(ranked, key=lambda item: item[0])[1]
    return InstanceMarker(
        position,
        "geometric_completion",
        body.body_id,
        float(np.clip(body.score, 0.0, 1.0)),
    )


def analyze_geometric_completion(
    component_mask: np.ndarray,
    peak_analysis: DistancePeakAnalysis,
    effective_peaks: tuple[PeakCandidate, ...],
    config: GeometricCompletionConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    *,
    retain_debug_artifacts: bool = False,
    force_analysis: bool = False,
) -> GeometricCompletionResult:
    """Detect unrepresented complete bodies and place safe supplemental markers."""

    mask = np.asarray(component_mask, dtype=bool)
    spacing = validate_voxel_size(voxel_size_zyx_um)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("component_mask must be a non-empty 3-D mask")
    if peak_analysis.raw_distance.shape != mask.shape:
        raise ValueError("peak-analysis arrays must align with component_mask")
    if not effective_peaks:
        raise ValueError("geometric completion requires effective EDT peaks")
    if not force_analysis and not geometric_completion_eligible(
        mask,
        spacing,
        config,
        raw_peak_count=len(peak_analysis.peaks),
        effective_marker_positions_zyx=tuple(
            peak.position_zyx for peak in effective_peaks
        ),
    ):
        return GeometricCompletionResult((), (), (), (), "ineligible", None)

    surface = analyze_surface_geometry(mask, spacing, config)
    body_analysis = analyze_geometric_bodies(mask, surface, spacing, config)
    represented_candidates = tuple(
        replace(
            body,
            represented_by_effective_peak_ids=_represented_peak_ids(
                body,
                effective_peaks,
                spacing,
                config.representation_ellipsoid_radius,
            ),
        )
        for body in body_analysis.body_candidates
    )
    candidates_by_id = {body.body_id: body for body in represented_candidates}
    selected = [candidates_by_id[body.body_id] for body in body_analysis.selected_bodies]
    effective_markers = list(convert_effective_peaks_to_markers(effective_peaks))
    supplemental: list[InstanceMarker] = []
    updated_selected: list[GeometricBody] = []
    for body in selected:
        represented_count = len(body.represented_by_effective_peak_ids)
        if represented_count != 0:
            # One marker means represented; multiple markers are retained and
            # conservatively suppress a redundant geometric addition.
            updated_selected.append(body)
            continue
        marker = _safe_marker_for_body(
            body,
            mask,
            peak_analysis.raw_distance,
            effective_markers + supplemental,
            spacing,
            config,
        )
        if marker is None:
            updated_selected.append(
                replace(
                    body,
                    rejection_reasons=tuple(
                        dict.fromkeys(body.rejection_reasons + ("no_valid_marker_position",))
                    ),
                )
            )
            continue
        supplemental.append(marker)
        updated_selected.append(body)

    supplemental_tuple = tuple(
        sorted(
            supplemental,
            key=lambda marker: (
                marker.position_zyx,
                marker.source_reference_id,
            ),
        )
    )
    updated_by_id = {body.body_id: body for body in updated_selected}
    final_candidates = tuple(
        updated_by_id.get(body.body_id, body) for body in represented_candidates
    )
    debug = None
    if retain_debug_artifacts:
        debug = GeometricDebugArtifacts(
            surface.boundary_positions_zyx,
            surface.boundary_normals_zyx,
            surface.samples,
            body_analysis.cap_pairs,
            body_analysis.ellipsoid_support_zyx,
            body_analysis.unique_support_zyx,
        )
    else:
        final_candidates = tuple(
            replace(body, cross_sections=()) for body in final_candidates
        )
        updated_selected = [
            replace(body, cross_sections=()) for body in updated_selected
        ]
    return GeometricCompletionResult(
        surface.caps,
        final_candidates,
        tuple(updated_selected),
        supplemental_tuple,
        "processed",
        None,
        debug,
    )


def safely_complete_geometric_markers(
    component_mask: np.ndarray,
    peak_analysis: DistancePeakAnalysis,
    effective_peaks: tuple[PeakCandidate, ...],
    config: GeometricCompletionConfig,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    *,
    retain_debug_artifacts: bool = False,
    force_analysis: bool = False,
) -> GeometricCompletionResult:
    """Isolate all geometric failures from the already-successful EDT path."""

    try:
        return analyze_geometric_completion(
            component_mask,
            peak_analysis,
            effective_peaks,
            config,
            voxel_size_zyx_um,
            retain_debug_artifacts=retain_debug_artifacts,
            force_analysis=force_analysis,
        )
    except Exception as error:  # the safety boundary is an explicit contract
        return GeometricCompletionResult.failed(error)


def surface_caps_dataframe(result: GeometricCompletionResult, component_id: int):
    """Convert compact cap records to the repository's diagnostic table style."""

    import pandas as pd

    return pd.DataFrame(
        [
            {
                "component_id": int(component_id),
                "cap_id": cap.cap_id,
                "z": cap.center_zyx[0],
                "y": cap.center_zyx[1],
                "x": cap.center_zyx[2],
                "normal_z": cap.mean_normal[0],
                "normal_y": cap.mean_normal[1],
                "normal_x": cap.mean_normal[2],
                "area_proxy_um2": cap.area_proxy_um2,
                "prominence_um": cap.prominence_um,
                "normal_coherence": cap.normal_coherence,
                "curvature_score": cap.curvature_score,
                "scale_support": cap.scale_support,
            }
            for cap in result.surface_caps
        ]
    )


def body_candidates_dataframe(result: GeometricCompletionResult, component_id: int):
    import pandas as pd

    selected_ids = {body.body_id for body in result.selected_bodies}
    return pd.DataFrame(
        [
            {
                "component_id": int(component_id),
                "body_id": body.body_id,
                "cap_id_a": body.cap_ids[0],
                "cap_id_b": body.cap_ids[1],
                "z": body.center_zyx[0],
                "y": body.center_zyx[1],
                "x": body.center_zyx[2],
                "semi_axis_long_um": body.semi_axes_um[0],
                "semi_axis_major_um": body.semi_axes_um[1],
                "semi_axis_minor_um": body.semi_axes_um[2],
                "axis_start_z": body.axis_endpoints_zyx[0][0],
                "axis_start_y": body.axis_endpoints_zyx[0][1],
                "axis_start_x": body.axis_endpoints_zyx[0][2],
                "axis_end_z": body.axis_endpoints_zyx[1][0],
                "axis_end_y": body.axis_endpoints_zyx[1][1],
                "axis_end_x": body.axis_endpoints_zyx[1][2],
                "score": body.score,
                "valid": body.valid,
                "selected": body.body_id in selected_ids,
                "rejection_reasons": ";".join(body.rejection_reasons),
                "represented_effective_peak_ids": ";".join(
                    str(value) for value in body.represented_by_effective_peak_ids
                ),
                **body.evidence.__dict__,
            }
            for body in result.body_candidates
        ]
    )


def cross_sections_dataframe(result: GeometricCompletionResult, component_id: int):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "component_id": int(component_id),
                "body_id": body.body_id,
                "section_index": section_index,
                **section.__dict__,
            }
            for body in result.body_candidates
            for section_index, section in enumerate(body.cross_sections)
        ]
    )


def marker_completion_dataframe(result: GeometricCompletionResult, component_id: int):
    import pandas as pd

    markers = {
        marker.source_reference_id: marker for marker in result.supplemental_markers
    }
    return pd.DataFrame(
        [
            {
                "component_id": int(component_id),
                "body_id": body.body_id,
                "represented_status": (
                    "multiple_effective_markers"
                    if len(body.represented_by_effective_peak_ids) > 1
                    else "represented"
                    if body.represented_by_effective_peak_ids
                    else "unrepresented"
                ),
                "marker_z": markers[body.body_id].position_zyx[0]
                if body.body_id in markers else np.nan,
                "marker_y": markers[body.body_id].position_zyx[1]
                if body.body_id in markers else np.nan,
                "marker_x": markers[body.body_id].position_zyx[2]
                if body.body_id in markers else np.nan,
                "application_status": (
                    "applied" if body.body_id in markers
                    else "not_needed" if body.represented_by_effective_peak_ids
                    else "rejected"
                ),
                "rejection_reason": ";".join(body.rejection_reasons),
            }
            for body in result.selected_bodies
        ]
    )


__all__ = [
    "analyze_geometric_completion",
    "body_candidates_dataframe",
    "combine_markers",
    "convert_effective_peaks_to_markers",
    "cross_sections_dataframe",
    "geometric_completion_eligible",
    "marker_completion_dataframe",
    "safely_complete_geometric_markers",
    "surface_caps_dataframe",
]
