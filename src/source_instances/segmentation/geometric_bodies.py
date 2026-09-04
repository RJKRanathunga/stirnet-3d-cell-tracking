"""Cap-pair body validation, ellipsoid fitting, conflicts, and selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from .config import GeometricCompletionConfig
from .distance import validate_voxel_size
from .models import (
    BodyEvidence,
    CapPairEvidence,
    CrossSectionEvidence,
    GeometricBody,
    SurfaceCap,
)
from .surface_geometry import SurfaceGeometryAnalysis


@dataclass(frozen=True)
class GeometricBodyAnalysis:
    """Body candidates, conflict-free selections, and compact debug supports."""

    cap_pairs: tuple[CapPairEvidence, ...]
    body_candidates: tuple[GeometricBody, ...]
    selected_bodies: tuple[GeometricBody, ...]
    ellipsoid_support_zyx: np.ndarray
    unique_support_zyx: np.ndarray


@dataclass(frozen=True)
class _SectionFit:
    evidence: CrossSectionEvidence
    radial_axes: np.ndarray


def _longest_true_fraction(values: np.ndarray) -> float:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return float(longest / len(values)) if len(values) else 0.0


def _axis_occupancy(
    mask: np.ndarray,
    first_um: np.ndarray,
    second_um: np.ndarray,
    spacing: np.ndarray,
    sample_spacing_um: float,
) -> tuple[float, float]:
    separation = float(np.linalg.norm(second_um - first_um))
    count = max(2, int(np.ceil(separation / sample_spacing_um)) + 1)
    fractions = np.linspace(0.0, 1.0, count)
    points_um = first_um[None, :] + fractions[:, None] * (second_um - first_um)
    coordinates = (points_um / spacing).T
    occupied = ndimage.map_coordinates(
        np.asarray(mask, dtype=np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    return float(np.mean(occupied)), _longest_true_fraction(occupied)


def build_cap_pair_evidence(
    caps: tuple[SurfaceCap, ...],
    component_mask: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    config: GeometricCompletionConfig,
) -> tuple[CapPairEvidence, ...]:
    """Apply cheap physical and foreground-axis gates to every distinct cap pair."""

    mask = np.asarray(component_mask, dtype=bool)
    spacing = validate_voxel_size(voxel_size_zyx_um)
    records: list[CapPairEvidence] = []
    for pair_id, (first, second) in enumerate(combinations(caps, 2), start=1):
        first_um = np.asarray(first.center_um, dtype=float)
        second_um = np.asarray(second.center_um, dtype=float)
        delta = second_um - first_um
        separation = float(np.linalg.norm(delta))
        if separation <= 1e-10:
            axis = np.zeros(3, dtype=float)
        else:
            axis = delta / separation
        first_normal = np.asarray(first.mean_normal, dtype=float)
        second_normal = np.asarray(second.mean_normal, dtype=float)
        opposition = float(-np.dot(first_normal, second_normal))
        # Cap A must face away from B; cap B must face away from A.
        alignment = float(
            min(-np.dot(first_normal, axis), np.dot(second_normal, axis))
        )
        scale_support = float(min(first.scale_support, second.scale_support))
        reasons: list[str] = []
        if not config.min_cap_separation_um <= separation <= config.max_cap_separation_um:
            reasons.append("cap_separation_out_of_range")
        if opposition < config.min_normal_opposition:
            reasons.append("insufficient_cap_opposition")
        if alignment < config.min_axis_alignment:
            reasons.append("insufficient_axis_alignment")
        if scale_support < config.cap_min_scale_support:
            reasons.append("insufficient_cap_scale_support")
        if separation > 1e-10:
            occupancy, consecutive = _axis_occupancy(
                mask,
                first_um,
                second_um,
                spacing,
                config.axis_sample_spacing_um,
            )
        else:
            occupancy = consecutive = 0.0
        if occupancy < config.min_axis_occupancy:
            reasons.append("axis_leaves_component")
        if consecutive < config.min_consecutive_axis_occupancy:
            reasons.append("insufficient_consecutive_axis_support")
        records.append(
            CapPairEvidence(
                pair_id,
                (first.cap_id, second.cap_id),
                tuple(
                    float(value)
                    for value in (
                        np.asarray(first.center_zyx)
                        + np.asarray(second.center_zyx)
                    )
                    / 2.0
                ),
                tuple(float(value) for value in (first_um + second_um) / 2.0),
                tuple(float(value) for value in axis),
                separation,
                opposition,
                alignment,
                scale_support,
                occupancy,
                consecutive,
                not reasons,
                tuple(reasons),
            )
        )
    return tuple(records)


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros(3, dtype=float)
    reference[int(np.argmin(np.abs(axis)))] = 1.0
    first = np.cross(axis, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(axis, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return first, second


def _foreground_axis_extent(
    mask: np.ndarray,
    midpoint_um: np.ndarray,
    axis: np.ndarray,
    spacing: np.ndarray,
    config: GeometricCompletionConfig,
) -> tuple[np.ndarray, float]:
    """Extend inset patch centers to the contiguous foreground axis ends."""

    half_limit = 0.5 * config.max_cap_separation_um
    count = max(
        3,
        int(np.ceil(2.0 * half_limit / config.axis_sample_spacing_um)) + 1,
    )
    t = np.linspace(-half_limit, half_limit, count)
    points_um = midpoint_um[None, :] + t[:, None] * axis
    occupied = ndimage.map_coordinates(
        np.asarray(mask, dtype=np.uint8),
        (points_um / spacing).T,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    center_index = int(np.argmin(np.abs(t)))
    if not occupied[center_index]:
        return midpoint_um, 0.0
    start = center_index
    stop = center_index
    while start > 0 and occupied[start - 1]:
        start -= 1
    while stop + 1 < len(occupied) and occupied[stop + 1]:
        stop += 1
    low = float(t[start])
    high = float(t[stop])
    return midpoint_um + 0.5 * (low + high) * axis, high - low


def _select_section_region(
    projected: np.ndarray,
    pixel_um: float,
    footprint_um: tuple[float, float],
) -> np.ndarray:
    """Raster-connect a slab and retain the group containing/nearest the axis."""

    if len(projected) < 5:
        return np.empty((0, 2), dtype=float)
    grid_points = np.rint(projected / pixel_um).astype(int)
    minimum = grid_points.min(axis=0) - 1
    maximum = grid_points.max(axis=0) + 1
    shape = tuple(int(value) for value in maximum - minimum + 1)
    if any(value <= 0 for value in shape):
        return np.empty((0, 2), dtype=float)
    raster = np.zeros(shape, dtype=bool)
    local = grid_points - minimum
    raster[tuple(local.T)] = True
    # A projected anisotropic voxel occupies more than one min-spacing pixel.
    # Dilation by its physical footprint prevents Z-separated voxel centers
    # from becoming artificial disconnected 2-D islands.
    footprint_pixels = tuple(
        max(1, int(np.ceil(value / pixel_um))) for value in footprint_um
    )
    footprint = np.ones(
        tuple(2 * value + 1 for value in footprint_pixels), dtype=bool
    )
    raster = ndimage.binary_dilation(
        raster,
        structure=footprint,
        iterations=1,
    )
    labels, count = ndimage.label(
        raster, structure=ndimage.generate_binary_structure(2, 2)
    )
    if count <= 0:
        return np.empty((0, 2), dtype=float)
    axis_cell = np.rint(-minimum).astype(int)
    chosen = 0
    if np.all(axis_cell >= 0) and np.all(axis_cell < np.asarray(shape)):
        chosen = int(labels[tuple(axis_cell)])
    if chosen <= 0:
        best: tuple[float, int] | None = None
        for label_id in range(1, int(count) + 1):
            coordinates = np.argwhere(labels == label_id) + minimum
            distance = float(np.min(np.linalg.norm(coordinates, axis=1)))
            key = (distance, label_id)
            if best is None or key < best:
                best = key
        chosen = 0 if best is None else best[1]
    cells = np.argwhere(labels == chosen) + minimum
    return cells.astype(float) * pixel_um


def _measure_cross_section(
    t_um: float,
    selected_points: np.ndarray,
    pixel_um: float,
    config: GeometricCompletionConfig,
) -> _SectionFit:
    if len(selected_points) < 5:
        evidence = CrossSectionEvidence(
            t_um, 0.0, np.inf, 0.0, 0.0, 0.0, np.inf, len(selected_points),
            False, "insufficient_section_support",
        )
        return _SectionFit(evidence, np.eye(2))
    centroid = selected_points.mean(axis=0)
    centered = selected_points - centroid
    covariance = centered.T @ centered / max(len(centered), 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        evidence = CrossSectionEvidence(
            t_um, 0.0, np.inf, 0.0, 0.0, 0.0, np.inf, len(selected_points),
            False, "ellipse_fit_failed",
        )
        return _SectionFit(evidence, np.eye(2))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    radii = 2.0 * np.sqrt(eigenvalues)
    major, minor = float(radii[0]), float(radii[1])
    area = float(len(selected_points) * pixel_um * pixel_um)
    offset = float(np.linalg.norm(centroid))
    if minor <= 0.25 * pixel_um or major <= 0.5 * pixel_um:
        evidence = CrossSectionEvidence(
            t_um, area, offset, major, minor, 0.0, np.inf,
            len(selected_points), False, "degenerate_section",
        )
        return _SectionFit(evidence, eigenvectors)
    local = centered @ eigenvectors
    normalized_radius = np.sqrt(
        (local[:, 0] / major) ** 2 + (local[:, 1] / minor) ** 2
    )
    intersection = float(np.count_nonzero(normalized_radius <= 1.0)) * pixel_um**2
    ellipse_area = float(np.pi * major * minor)
    union = max(area + ellipse_area - intersection, 1e-8)
    ellipse_iou = float(np.clip(intersection / union, 0.0, 1.0))

    grid = np.rint(selected_points / pixel_um).astype(int)
    minimum = grid.min(axis=0)
    local_grid = grid - minimum
    raster = np.zeros(tuple((grid.max(axis=0) - minimum + 1).astype(int)), dtype=bool)
    raster[tuple(local_grid.T)] = True
    boundary = raster ^ ndimage.binary_erosion(raster)
    boundary_points = (np.argwhere(boundary) + minimum) * pixel_um
    boundary_local = (boundary_points - centroid) @ eigenvectors
    boundary_q = np.sqrt(
        (boundary_local[:, 0] / major) ** 2
        + (boundary_local[:, 1] / minor) ** 2
    )
    boundary_error = float(
        np.median(np.abs(boundary_q - 1.0)) * min(major, minor)
    )
    reason: str | None = None
    if ellipse_iou < min(0.35, 0.8 * config.min_median_ellipse_iou):
        reason = "poor_section_ellipse_support"
    elif offset > 1.5 * config.max_centerline_deviation_um:
        reason = "section_centerline_deviation"
    evidence = CrossSectionEvidence(
        t_um,
        area,
        offset,
        major,
        minor,
        ellipse_iou,
        boundary_error,
        len(selected_points),
        reason is None,
        reason,
    )
    return _SectionFit(evidence, eigenvectors)


def perpendicular_cross_sections(
    component_mask: np.ndarray,
    midpoint_um: np.ndarray,
    axis: np.ndarray,
    separation_um: float,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    config: GeometricCompletionConfig,
) -> tuple[_SectionFit, ...]:
    """Measure local connected elliptical sections along one proposed axis."""

    spacing = validate_voxel_size(voxel_size_zyx_um)
    tangent_a, tangent_b = _orthonormal_frame(axis)
    footprint_um = (
        0.5 * float(np.dot(np.abs(tangent_a), spacing)),
        0.5 * float(np.dot(np.abs(tangent_b), spacing)),
    )
    component_points_um = np.argwhere(component_mask).astype(float) * spacing
    relative = component_points_um - midpoint_um
    longitudinal = relative @ axis
    radial_a = relative @ tangent_a
    radial_b = relative @ tangent_b
    half_length = separation_um / 2.0
    count = max(
        config.min_valid_cross_sections,
        int(np.floor(separation_um / config.cross_section_spacing_um)) + 1,
    )
    t_values = np.linspace(
        -0.9 * half_length, 0.9 * half_length, count
    )
    projected_voxel_half_width = 0.5 * float(np.dot(np.abs(axis), spacing))
    half_thickness = 0.5 * config.cross_section_thickness_um + projected_voxel_half_width
    pixel_um = float(min(spacing))
    # Candidate-local crop: a touching neighbour may be connected in the slab,
    # so do not let a section expand over most of the parent component.
    radial_limit = max(0.40 * separation_um, 2.0 * pixel_um)
    projected_by_section: list[np.ndarray] = []
    results: list[_SectionFit] = []
    for t_um in t_values:
        selected = (
            (np.abs(longitudinal - t_um) <= half_thickness)
            & (np.hypot(radial_a, radial_b) <= radial_limit)
        )
        projected = np.column_stack((radial_a[selected], radial_b[selected]))
        projected_by_section.append(projected)
        region = _select_section_region(projected, pixel_um, footprint_um)
        results.append(_measure_cross_section(float(t_um), region, pixel_um, config))
    supported_radii = [
        section.evidence.radius_major_um
        for section in results
        if section.evidence.radius_major_um > 0
    ]
    if supported_radii:
        # Contact sections can balloon into a neighbouring body. End-biased
        # low-quantile scale provides a conservative candidate-local radius.
        refined_limit = float(
            np.clip(
                1.6 * np.percentile(supported_radii, 20.0),
                2.0 * pixel_um,
                radial_limit,
            )
        )
        if refined_limit < 0.95 * radial_limit:
            results = []
            for t_um, projected in zip(t_values, projected_by_section):
                projected = projected[
                    np.linalg.norm(projected, axis=1) <= refined_limit
                ]
                region = _select_section_region(
                    projected, pixel_um, footprint_um
                )
                results.append(
                    _measure_cross_section(float(t_um), region, pixel_um, config)
                )
    return tuple(results)


def _fit_area_profile(
    sections: tuple[_SectionFit, ...],
    half_length: float,
) -> tuple[float, float, float, float, bool]:
    valid = [section.evidence for section in sections if section.evidence.valid]
    if len(valid) < 3:
        return 0.0, 0.0, half_length, 1.0, False
    t = np.asarray([section.t_um for section in valid], dtype=float)
    area = np.asarray([section.area_um2 for section in valid], dtype=float)
    maximum = max(float(area.max()), 1e-8)

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude = np.exp(parameters[0])
        center = parameters[1]
        radius = np.exp(parameters[2])
        prediction = amplitude * np.maximum(0.0, 1.0 - ((t - center) / radius) ** 2)
        return (prediction - area) / maximum

    initial = np.asarray((np.log(maximum), float(t[np.argmax(area)]), np.log(half_length)))
    lower = np.asarray((np.log(maximum * 0.35), -0.35 * half_length, np.log(0.65 * half_length)))
    upper = np.asarray((np.log(maximum * 2.2), 0.35 * half_length, np.log(1.55 * half_length)))
    try:
        fitted = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.15,
            max_nfev=120,
        )
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            return maximum, 0.0, half_length, 1.0, False
    except (ValueError, FloatingPointError):
        return maximum, 0.0, half_length, 1.0, False
    amplitude = float(np.exp(fitted.x[0]))
    center = float(fitted.x[1])
    radius = float(np.exp(fitted.x[2]))
    prediction = amplitude * np.maximum(0.0, 1.0 - ((t - center) / radius) ** 2)
    error = float(np.median(np.abs(prediction - area)) / maximum)
    both_sides = np.count_nonzero(t < center) >= 1 and np.count_nonzero(t > center) >= 1
    maximum_central = abs(float(t[np.argmax(area)] - center)) <= 0.45 * half_length
    return amplitude, center, radius, error, bool(both_sides and maximum_central)


def ellipsoid_normalized_distance(
    points_um: np.ndarray,
    center_um: np.ndarray,
    rotation_matrix: np.ndarray,
    semi_axes_um: np.ndarray,
) -> np.ndarray:
    local = (np.asarray(points_um, dtype=float) - center_um) @ rotation_matrix
    return np.sqrt(np.sum((local / semi_axes_um) ** 2, axis=-1))


def _fit_ellipsoid(
    center_um: np.ndarray,
    rotation: np.ndarray,
    semi_axes: np.ndarray,
    surface: SurfaceGeometryAnalysis,
) -> tuple[np.ndarray, np.ndarray] | None:
    boundary = surface.boundary_positions_um
    if not len(boundary):
        return None
    local = (boundary - center_um) @ rotation
    support = (
        (np.abs(local[:, 0]) <= 1.25 * semi_axes[0])
        & (np.hypot(local[:, 1], local[:, 2]) <= 1.55 * max(semi_axes[1:]))
    )
    points = boundary[support]
    if len(points) < 12:
        return None
    max_center_shift = min(1.0, 0.20 * semi_axes[0])
    lower_axes = np.maximum(0.65 * semi_axes, 0.35)
    # Cross-sections already estimate the radial body scale. Allow a modest
    # robust refinement, but do not let contact-surface points balloon a fit
    # into the neighbouring cell.
    upper_axes = semi_axes * np.asarray((1.30, 1.15, 1.15))

    def residual(parameters: np.ndarray) -> np.ndarray:
        fitted_center = center_um + parameters[:3]
        fitted_axes = np.exp(parameters[3:])
        q = ellipsoid_normalized_distance(points, fitted_center, rotation, fitted_axes)
        return q - 1.0

    initial = np.concatenate((np.zeros(3), np.log(semi_axes)))
    lower = np.concatenate(
        (
            np.full(3, -max_center_shift),
            np.log(lower_axes),
        )
    )
    upper = np.concatenate(
        (
            np.full(3, max_center_shift),
            np.log(upper_axes),
        )
    )
    try:
        fitted = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.12,
            max_nfev=160,
        )
    except (ValueError, FloatingPointError):
        return None
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None
    return center_um + fitted.x[:3], np.exp(fitted.x[3:])


def _candidate_body(
    body_id: int,
    pair: CapPairEvidence,
    caps_by_id: dict[int, SurfaceCap],
    component_mask: np.ndarray,
    spacing: np.ndarray,
    surface: SurfaceGeometryAnalysis,
    config: GeometricCompletionConfig,
) -> tuple[GeometricBody, np.ndarray, np.ndarray]:
    first = caps_by_id[pair.cap_ids[0]]
    second = caps_by_id[pair.cap_ids[1]]
    midpoint_um = np.asarray(pair.midpoint_um, dtype=float)
    axis = np.asarray(pair.axis_unit_um, dtype=float)
    extended_midpoint, extended_separation = _foreground_axis_extent(
        component_mask, midpoint_um, axis, spacing, config
    )
    body_separation = max(pair.separation_um, extended_separation)
    if body_separation <= config.max_cap_separation_um:
        midpoint_um = extended_midpoint
    else:
        body_separation = pair.separation_um
    sections = perpendicular_cross_sections(
        component_mask,
        midpoint_um,
        axis,
        body_separation,
        spacing,
        config,
    )
    valid_sections = [section for section in sections if section.evidence.valid]
    valid_fraction = len(valid_sections) / max(len(sections), 1)
    median_iou = float(
        np.median([section.evidence.ellipse_iou for section in valid_sections])
    ) if valid_sections else 0.0
    max_centerline = float(
        max((section.evidence.centroid_offset_um for section in valid_sections), default=np.inf)
    )
    _, profile_center, profile_radius, profile_error, two_sided_profile = _fit_area_profile(
        sections, body_separation / 2.0
    )
    center_um = midpoint_um + profile_center * axis

    central = sorted(
        valid_sections,
        key=lambda section: (
            abs(section.evidence.t_um - profile_center),
            -section.evidence.area_um2,
        ),
    )
    reasons = list(pair.rejection_reasons)
    if len(valid_sections) < config.min_valid_cross_sections:
        reasons.append("too_few_cross_sections")
    if valid_fraction < config.min_valid_cross_section_fraction:
        reasons.append("insufficient_valid_cross_section_fraction")
    if median_iou < config.min_median_ellipse_iou:
        reasons.append("poor_cross_section_ellipse_support")
    if max_centerline > config.max_centerline_deviation_um:
        reasons.append("excessive_centerline_deviation")
    if profile_error > config.max_area_profile_error or not two_sided_profile:
        reasons.append("non_ellipsoidal_area_profile")

    if central:
        representative = central[0]
        major = max(representative.evidence.radius_major_um, 0.35)
        minor = max(representative.evidence.radius_minor_um, 0.35)
        tangent_a, tangent_b = _orthonormal_frame(axis)
        radial_2d = representative.radial_axes
        radial_major = tangent_a * radial_2d[0, 0] + tangent_b * radial_2d[1, 0]
        radial_minor = tangent_a * radial_2d[0, 1] + tangent_b * radial_2d[1, 1]
    else:
        major = minor = 0.35
        radial_major, radial_minor = _orthonormal_frame(axis)
    long_axis = max(0.5 * body_separation, min(profile_radius, 0.8 * body_separation))
    rotation = np.column_stack((axis, radial_major, radial_minor))
    semi_axes = np.asarray((long_axis, major, minor), dtype=float)
    fitted = _fit_ellipsoid(center_um, rotation, semi_axes, surface)
    if fitted is None:
        reasons.append("ellipsoid_fit_failed")
    else:
        center_um, semi_axes = fitted

    axis_ratio = float(max(semi_axes) / max(min(semi_axes), 1e-8))
    radial_ratio = float(max(semi_axes[1:]) / max(min(semi_axes[1:]), 1e-8))
    if axis_ratio > 8.0 or radial_ratio > 4.0:
        reasons.append("implausible_axis_ratio")

    component_points_zyx = np.argwhere(component_mask)
    component_points_um = component_points_zyx.astype(float) * spacing
    q_component = ellipsoid_normalized_distance(
        component_points_um, center_um, rotation, semi_axes
    )
    supported_component = q_component <= 1.0
    grid = np.indices(component_mask.shape).transpose(1, 2, 3, 0)
    q_grid = ellipsoid_normalized_distance(
        grid.reshape((-1, 3)).astype(float) * spacing,
        center_um,
        rotation,
        semi_axes,
    ).reshape(component_mask.shape)
    ellipsoid_mask = q_grid <= 1.0
    ellipsoid_voxels = max(int(np.count_nonzero(ellipsoid_mask)), 1)
    occupancy = float(
        np.count_nonzero(ellipsoid_mask & component_mask) / ellipsoid_voxels
    )
    boundary_q = ellipsoid_normalized_distance(
        surface.boundary_positions_um, center_um, rotation, semi_axes
    )
    local_boundary = boundary_q <= 1.35
    surface_support = float(
        np.count_nonzero(local_boundary & (np.abs(boundary_q - 1.0) <= 0.30))
        / max(np.count_nonzero(local_boundary), 1)
    )
    if occupancy < config.min_ellipsoid_occupancy:
        reasons.append("insufficient_ellipsoid_occupancy")
    if surface_support < config.min_ellipsoid_surface_support:
        reasons.append("insufficient_surface_support")

    area_profile_score = float(np.clip(1.0 - profile_error, 0.0, 1.0))
    centerline_score = float(
        np.clip(1.0 - max_centerline / config.max_centerline_deviation_um, 0.0, 1.0)
    ) if np.isfinite(max_centerline) else 0.0
    evidence_values = (
        pair.normal_opposition,
        pair.axis_alignment,
        pair.scale_support,
        pair.axis_occupancy,
        pair.consecutive_axis_occupancy,
        valid_fraction,
        median_iou,
        area_profile_score,
        centerline_score,
        surface_support,
        occupancy,
    )
    score = float(np.clip(np.mean(evidence_values), 0.0, 1.0))
    if score < config.min_body_score:
        reasons.append("insufficient_body_score")
    evidence = BodyEvidence(
        pair.normal_opposition,
        pair.axis_alignment,
        pair.scale_support,
        pair.axis_occupancy,
        pair.consecutive_axis_occupancy,
        valid_fraction,
        median_iou,
        area_profile_score,
        centerline_score,
        surface_support,
        occupancy,
        1.0,
        1.0,
    )
    center_zyx = center_um / spacing
    endpoints = (
        tuple(float(value) for value in (center_um - semi_axes[0] * axis) / spacing),
        tuple(float(value) for value in (center_um + semi_axes[0] * axis) / spacing),
    )
    body = GeometricBody(
        body_id,
        pair.cap_ids,
        tuple(float(value) for value in center_zyx),
        tuple(float(value) for value in center_um),
        rotation,
        tuple(float(value) for value in semi_axes),
        endpoints,
        evidence,
        score,
        not reasons,
        tuple(dict.fromkeys(reasons)),
        (),
        tuple(section.evidence for section in sections),
    )
    surface_mask = np.abs(boundary_q - 1.0) <= 0.30
    return body, ellipsoid_mask & component_mask, surface_mask


def _apply_unique_support(
    bodies: list[GeometricBody],
    volume_masks: dict[int, np.ndarray],
    surface_masks: dict[int, np.ndarray],
    config: GeometricCompletionConfig,
) -> list[GeometricBody]:
    """Assign support against higher-ranked fits; shifted duplicates lose ownership."""

    ranked = sorted(
        [body for body in bodies if body.valid],
        key=lambda body: (-body.score, body.cap_ids, body.body_id),
    )
    earlier_volume: list[np.ndarray] = []
    earlier_surface: list[np.ndarray] = []
    updated: dict[int, GeometricBody] = {}
    for body in ranked:
        volume = volume_masks[body.body_id]
        surface = surface_masks[body.body_id]
        occupied_volume = np.logical_or.reduce(earlier_volume) if earlier_volume else np.zeros_like(volume)
        occupied_surface = np.logical_or.reduce(earlier_surface) if earlier_surface else np.zeros_like(surface)
        unique_volume = float(
            np.count_nonzero(volume & ~occupied_volume)
            / max(np.count_nonzero(volume), 1)
        )
        unique_surface = float(
            np.count_nonzero(surface & ~occupied_surface)
            / max(np.count_nonzero(surface), 1)
        )
        evidence = replace(
            body.evidence,
            unique_volume_fraction=unique_volume,
            unique_surface_fraction=unique_surface,
        )
        reasons = list(body.rejection_reasons)
        if (
            unique_volume < config.min_unique_volume_fraction
            and unique_surface < config.min_unique_surface_fraction
        ):
            reasons.append("insufficient_unique_support")
        updated[body.body_id] = replace(
            body,
            evidence=evidence,
            valid=not reasons,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )
        earlier_volume.append(volume)
        earlier_surface.append(surface)
    return [updated.get(body.body_id, body) for body in bodies]


def bodies_conflict(
    first: GeometricBody,
    second: GeometricBody,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    spacing: np.ndarray,
    config: GeometricCompletionConfig,
) -> bool:
    """Return whether two valid fits are redundant or reuse geometric support."""

    if set(first.cap_ids).intersection(second.cap_ids):
        return True
    center_distance = float(
        np.linalg.norm(
            (np.asarray(first.center_zyx) - np.asarray(second.center_zyx)) * spacing
        )
    )
    if center_distance < config.marker_min_separation_um:
        return True
    intersection = int(np.count_nonzero(first_mask & second_mask))
    first_count = max(int(np.count_nonzero(first_mask)), 1)
    second_count = max(int(np.count_nonzero(second_mask)), 1)
    union = max(first_count + second_count - intersection, 1)
    iou = intersection / union
    containment = intersection / min(first_count, second_count)
    return iou >= 0.62 or containment >= 0.84


def select_geometric_bodies(
    bodies: tuple[GeometricBody, ...] | list[GeometricBody],
    volume_masks: dict[int, np.ndarray],
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    config: GeometricCompletionConfig,
) -> tuple[GeometricBody, ...]:
    """Select any number of disjoint complete bodies by deterministic greedy rank."""

    spacing = validate_voxel_size(voxel_size_zyx_um)
    ordered = sorted(
        [body for body in bodies if body.valid],
        key=lambda body: (
            -body.score,
            -max(
                body.evidence.unique_volume_fraction,
                body.evidence.unique_surface_fraction,
            ),
            -body.evidence.cap_scale_support,
            body.cap_ids,
            body.body_id,
        ),
    )
    selected: list[GeometricBody] = []
    for candidate in ordered:
        if any(
            bodies_conflict(
                candidate,
                accepted,
                volume_masks[candidate.body_id],
                volume_masks[accepted.body_id],
                spacing,
                config,
            )
            for accepted in selected
        ):
            continue
        selected.append(candidate)
    return tuple(selected)


def analyze_geometric_bodies(
    component_mask: np.ndarray,
    surface: SurfaceGeometryAnalysis,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    config: GeometricCompletionConfig,
) -> GeometricBodyAnalysis:
    """Fit, hard-gate, de-duplicate, and select arbitrary cap-supported bodies."""

    mask = np.asarray(component_mask, dtype=bool)
    spacing = validate_voxel_size(voxel_size_zyx_um)
    cap_pairs = build_cap_pair_evidence(
        surface.caps, mask, spacing, config
    )
    caps_by_id = {cap.cap_id: cap for cap in surface.caps}
    bodies: list[GeometricBody] = []
    volume_masks: dict[int, np.ndarray] = {}
    surface_masks: dict[int, np.ndarray] = {}
    for body_id, pair in enumerate(cap_pairs, start=1):
        if not pair.valid:
            evidence = BodyEvidence(
                pair.normal_opposition,
                pair.axis_alignment,
                pair.scale_support,
                pair.axis_occupancy,
                pair.consecutive_axis_occupancy,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            )
            axis = np.asarray(pair.axis_unit_um, dtype=float)
            tangent_a, tangent_b = _orthonormal_frame(axis) if np.linalg.norm(axis) > 0 else (np.eye(3)[1], np.eye(3)[2])
            rotation = np.column_stack((axis, tangent_a, tangent_b))
            body = GeometricBody(
                body_id,
                pair.cap_ids,
                pair.midpoint_zyx,
                pair.midpoint_um,
                rotation,
                (max(pair.separation_um / 2.0, 0.1), 0.1, 0.1),
                (pair.midpoint_zyx, pair.midpoint_zyx),
                evidence,
                0.0,
                False,
                pair.rejection_reasons,
            )
            bodies.append(body)
            volume_masks[body_id] = np.zeros(mask.shape, dtype=bool)
            surface_masks[body_id] = np.zeros(len(surface.boundary_positions_zyx), dtype=bool)
            continue
        body, volume_mask, surface_mask = _candidate_body(
            body_id,
            pair,
            caps_by_id,
            mask,
            spacing,
            surface,
            config,
        )
        bodies.append(body)
        volume_masks[body_id] = volume_mask
        surface_masks[body_id] = surface_mask
    bodies = _apply_unique_support(bodies, volume_masks, surface_masks, config)
    selected = select_geometric_bodies(bodies, volume_masks, spacing, config)
    ellipsoid_points = [
        np.argwhere(volume_masks[body.body_id]) for body in selected
    ]
    unique_points: list[np.ndarray] = []
    occupied = np.zeros(mask.shape, dtype=bool)
    for body in selected:
        volume = volume_masks[body.body_id]
        unique_points.append(np.argwhere(volume & ~occupied))
        occupied |= volume
    return GeometricBodyAnalysis(
        cap_pairs,
        tuple(bodies),
        selected,
        (
            np.concatenate(ellipsoid_points, axis=0).astype(float)
            if ellipsoid_points and any(len(value) for value in ellipsoid_points)
            else np.empty((0, 3), dtype=float)
        ),
        (
            np.concatenate(unique_points, axis=0).astype(float)
            if unique_points and any(len(value) for value in unique_points)
            else np.empty((0, 3), dtype=float)
        ),
    )


__all__ = [
    "GeometricBodyAnalysis",
    "analyze_geometric_bodies",
    "bodies_conflict",
    "build_cap_pair_evidence",
    "ellipsoid_normalized_distance",
    "perpendicular_cross_sections",
    "select_geometric_bodies",
]
