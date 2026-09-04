"""Physical signed-distance surface analysis and multiscale cap patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import GeometricCompletionConfig
from .distance import physical_sigma_voxels, validate_voxel_size
from .models import SurfaceCap, SurfaceSample


@dataclass(frozen=True)
class SignedDistanceSurface:
    """A geometry-padded component and its inside-positive physical SDF."""

    padded_mask: np.ndarray
    signed_distance_um: np.ndarray
    padding_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class SurfaceGeometryAnalysis:
    """Compact caps plus boundary detail used by downstream body fitting."""

    signed_surface: SignedDistanceSurface
    boundary_positions_zyx: np.ndarray
    boundary_positions_um: np.ndarray
    boundary_normals_zyx: np.ndarray
    samples: tuple[SurfaceSample, ...]
    caps: tuple[SurfaceCap, ...]


@dataclass(frozen=True)
class _PatchDetection:
    scale_index: int
    center_zyx: np.ndarray
    center_um: np.ndarray
    mean_normal: np.ndarray
    area_proxy_um2: float
    prominence_um: float
    normal_coherence: float
    curvature_score: float
    sample_indices: tuple[int, ...]


def geometry_padding_voxels(
    surface_padding_um: float,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
) -> tuple[int, int, int]:
    """Convert independent physical surface padding to per-axis voxels."""

    spacing = validate_voxel_size(voxel_size_zyx_um)
    if not np.isfinite(surface_padding_um) or surface_padding_um <= 0:
        raise ValueError("surface_padding_um must be finite and positive")
    return tuple(int(value) for value in np.ceil(surface_padding_um / spacing))


def physical_signed_distance(
    component_mask: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    surface_padding_um: float,
) -> SignedDistanceSurface:
    """Build an inside-positive SDF with geometry-specific exterior padding."""

    mask = np.asarray(component_mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("component_mask must be a non-empty 3-D mask")
    spacing = validate_voxel_size(voxel_size_zyx_um)
    padding = geometry_padding_voxels(surface_padding_um, spacing)
    padded = np.pad(mask, tuple((value, value) for value in padding))
    inside = ndimage.distance_transform_edt(padded, sampling=spacing)
    outside = ndimage.distance_transform_edt(~padded, sampling=spacing)
    signed = np.asarray(inside - outside, dtype=np.float32)
    return SignedDistanceSurface(padded, signed, padding)


def _stable_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros(3, dtype=float)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(normal, reference)
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-10:
        raise np.linalg.LinAlgError("cannot construct tangent frame")
    first /= first_norm
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-10)
    return first, second


def _local_surface_measurements(
    index: int,
    positions_um: np.ndarray,
    normals: np.ndarray,
    neighborhood: tuple[int, ...] | list[int],
) -> tuple[float, float, tuple[float, float]] | None:
    """Return coherence, prominence, and stable convex curvatures."""

    neighbor_indices = np.asarray(neighborhood, dtype=int)
    if len(neighbor_indices) < 6:
        return None
    neighbor_positions = positions_um[neighbor_indices]
    neighbor_normals = normals[neighbor_indices]
    mean_normal = neighbor_normals.mean(axis=0)
    coherence = float(np.linalg.norm(mean_normal))
    normal = normals[index]
    prominence = float(
        np.dot(normal, positions_um[index] - neighbor_positions.mean(axis=0))
    )
    try:
        tangent_a, tangent_b = _stable_tangent_frame(normal)
        relative = neighbor_positions - positions_um[index]
        x = relative @ tangent_a
        y = relative @ tangent_b
        z = relative @ normal
        design = np.column_stack((x * x, x * y, y * y, x, y, np.ones(len(x))))
        if np.linalg.cond(design) > 1e7:
            return None
        coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
        hessian = np.asarray(
            (
                (2.0 * coefficients[0], coefficients[1]),
                (coefficients[1], 2.0 * coefficients[2]),
            )
        )
        # With the outward normal, a convex surface bends toward negative z.
        curvature = np.sort(-np.linalg.eigvalsh(hessian))[::-1]
        if not np.all(np.isfinite(curvature)):
            return None
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return coherence, prominence, (float(curvature[0]), float(curvature[1]))


def _surface_voxel_area_proxy(spacing: np.ndarray) -> float:
    face_areas = (
        spacing[0] * spacing[1],
        spacing[0] * spacing[2],
        spacing[1] * spacing[2],
    )
    return float(np.mean(face_areas))


def _detect_scale_patches(
    scale_index: int,
    smoothed_signed_distance: np.ndarray,
    boundary_positions_padded: np.ndarray,
    boundary_positions_um: np.ndarray,
    tree: cKDTree,
    spacing: np.ndarray,
    padding: np.ndarray,
    config: GeometricCompletionConfig,
) -> tuple[list[_PatchDetection], list[SurfaceSample], np.ndarray]:
    gradients = np.gradient(
        smoothed_signed_distance,
        float(spacing[0]),
        float(spacing[1]),
        float(spacing[2]),
    )
    stacked = np.stack(
        [gradient[tuple(boundary_positions_padded.T)] for gradient in gradients],
        axis=1,
    )
    # The SDF is positive inside, so negative gradients face the exterior.
    norms = np.linalg.norm(stacked, axis=1)
    normals = np.zeros_like(stacked, dtype=float)
    safe = norms > 1e-10
    normals[safe] = -stacked[safe] / norms[safe, None]
    neighborhoods = tree.query_ball_point(
        boundary_positions_um, config.boundary_neighborhood_radius_um
    )

    measurements: list[tuple[float, float, tuple[float, float]] | None] = []
    samples: list[SurfaceSample] = []
    for index, neighborhood in enumerate(neighborhoods):
        measured = _local_surface_measurements(
            index, boundary_positions_um, normals, neighborhood
        )
        measurements.append(measured)
        if measured is None:
            continue
        coherence, prominence, curvatures = measured
        original_position = boundary_positions_padded[index] - padding
        samples.append(
            SurfaceSample(
                tuple(int(value) for value in original_position),
                tuple(float(value) for value in boundary_positions_um[index]),
                tuple(float(value) for value in normals[index]),
                curvatures,
                prominence,
                coherence,
                scale_index,
            )
        )

    eligible: list[tuple[float, int]] = []
    for index, measured in enumerate(measurements):
        if measured is None or not safe[index]:
            continue
        coherence, prominence, curvatures = measured
        maximum_curvature = max(curvatures)
        minimum_curvature = min(curvatures)
        locally_convex = (
            maximum_curvature > 0.0
            and minimum_curvature >= -0.35 * maximum_curvature
        )
        if (
            coherence < config.cap_min_normal_coherence
            or prominence < config.cap_min_prominence_um
            or not locally_convex
        ):
            continue
        curvature_score = float(max(np.mean(curvatures), 0.0))
        score = prominence * coherence * max(curvature_score, 1e-6)
        eligible.append((score, index))

    # Physical NMS bounds candidate density without imposing a biological cap.
    seeds: list[int] = []
    for _, index in sorted(
        eligible,
        key=lambda item: (
            -item[0],
            tuple(int(value) for value in boundary_positions_padded[item[1]]),
        ),
    ):
        if all(
            np.linalg.norm(
                boundary_positions_um[index] - boundary_positions_um[other]
            )
            >= config.cap_patch_radius_um
            for other in seeds
        ):
            seeds.append(index)

    area_per_sample = _surface_voxel_area_proxy(spacing)
    patches: list[_PatchDetection] = []
    for seed in seeds:
        nearby = tree.query_ball_point(
            boundary_positions_um[seed], config.cap_patch_radius_um
        )
        patch_indices = tuple(
            sorted(
                int(index)
                for index in nearby
                if measurements[index] is not None
                and float(np.dot(normals[index], normals[seed])) >= 0.72
                and measurements[index][0]
                >= 0.85 * config.cap_min_normal_coherence
                and measurements[index][1]
                >= 0.5 * config.cap_min_prominence_um
                and max(measurements[index][2]) > 0.0
                and min(measurements[index][2])
                >= -0.35 * max(measurements[index][2])
            )
        )
        area = len(patch_indices) * area_per_sample
        if area < config.cap_min_area_um2 or not patch_indices:
            continue
        weights = np.asarray(
            [max(measurements[index][1], 1e-6) for index in patch_indices]
        )
        weights /= float(weights.sum())
        patch_um = boundary_positions_um[np.asarray(patch_indices)]
        patch_zyx = (
            boundary_positions_padded[np.asarray(patch_indices)] - padding
        )
        mean_normal = np.average(
            normals[np.asarray(patch_indices)], axis=0, weights=weights
        )
        mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-10)
        patches.append(
            _PatchDetection(
                scale_index,
                np.average(patch_zyx, axis=0, weights=weights),
                np.average(patch_um, axis=0, weights=weights),
                mean_normal,
                float(area),
                float(
                    np.average(
                        [measurements[index][1] for index in patch_indices],
                        weights=weights,
                    )
                ),
                float(
                    np.average(
                        [measurements[index][0] for index in patch_indices],
                        weights=weights,
                    )
                ),
                float(
                    np.average(
                        [
                            max(np.mean(measurements[index][2]), 0.0)
                            for index in patch_indices
                        ],
                        weights=weights,
                    )
                ),
                patch_indices,
            )
        )
    return patches, samples, normals


def _consolidate_cap_patches(
    detections: list[_PatchDetection],
    scale_count: int,
    config: GeometricCompletionConfig,
) -> tuple[SurfaceCap, ...]:
    groups: list[list[_PatchDetection]] = []
    ordered = sorted(
        detections,
        key=lambda patch: (
            -patch.prominence_um,
            -patch.curvature_score,
            tuple(float(value) for value in patch.center_zyx),
            patch.scale_index,
        ),
    )
    for patch in ordered:
        match: int | None = None
        best_distance = np.inf
        for index, group in enumerate(groups):
            representative = group[0]
            distance = float(
                np.linalg.norm(patch.center_um - representative.center_um)
            )
            normal_agreement = float(
                np.dot(patch.mean_normal, representative.mean_normal)
            )
            overlap = len(
                set(patch.sample_indices).intersection(representative.sample_indices)
            )
            if (
                distance <= config.cap_cluster_radius_um
                and normal_agreement >= 0.65
                and (overlap > 0 or distance <= 0.65 * config.cap_cluster_radius_um)
                and distance < best_distance
            ):
                match = index
                best_distance = distance
        if match is None:
            groups.append([patch])
        else:
            groups[match].append(patch)

    provisional: list[SurfaceCap] = []
    for group in groups:
        support = len({patch.scale_index for patch in group}) / scale_count
        if support + 1e-12 < config.cap_min_scale_support:
            continue
        weights = np.asarray(
            [patch.prominence_um * patch.area_proxy_um2 for patch in group],
            dtype=float,
        )
        weights = np.maximum(weights, 1e-8)
        weights /= float(weights.sum())
        normal = np.average(
            np.asarray([patch.mean_normal for patch in group]),
            axis=0,
            weights=weights,
        )
        normal /= max(float(np.linalg.norm(normal)), 1e-10)
        sample_indices = tuple(
            sorted({value for patch in group for value in patch.sample_indices})
        )
        provisional.append(
            SurfaceCap(
                0,
                tuple(
                    float(value)
                    for value in np.average(
                        np.asarray([patch.center_zyx for patch in group]),
                        axis=0,
                        weights=weights,
                    )
                ),
                tuple(
                    float(value)
                    for value in np.average(
                        np.asarray([patch.center_um for patch in group]),
                        axis=0,
                        weights=weights,
                    )
                ),
                tuple(float(value) for value in normal),
                float(max(patch.area_proxy_um2 for patch in group)),
                float(np.average([patch.prominence_um for patch in group], weights=weights)),
                float(np.average([patch.normal_coherence for patch in group], weights=weights)),
                float(np.average([patch.curvature_score for patch in group], weights=weights)),
                float(support),
                sample_indices,
            )
        )
    ordered_caps = sorted(
        provisional,
        key=lambda cap: (
            cap.center_zyx[0],
            cap.center_zyx[1],
            cap.center_zyx[2],
            -cap.prominence_um,
        ),
    )
    return tuple(
        SurfaceCap(
            index,
            cap.center_zyx,
            cap.center_um,
            cap.mean_normal,
            cap.area_proxy_um2,
            cap.prominence_um,
            cap.normal_coherence,
            cap.curvature_score,
            cap.scale_support,
            cap.sample_indices,
        )
        for index, cap in enumerate(ordered_caps, start=1)
    )


def analyze_surface_geometry(
    component_mask: np.ndarray,
    voxel_size_zyx_um: tuple[float, float, float] | np.ndarray,
    config: GeometricCompletionConfig,
) -> SurfaceGeometryAnalysis:
    """Extract deterministic multiscale convex cap patches in physical space."""

    spacing = validate_voxel_size(voxel_size_zyx_um)
    signed_surface = physical_signed_distance(
        component_mask, spacing, config.surface_padding_um
    )
    structure = ndimage.generate_binary_structure(3, 1)
    boundary_mask = signed_surface.padded_mask ^ ndimage.binary_erosion(
        signed_surface.padded_mask, structure=structure, border_value=0
    )
    boundary_padded = np.argwhere(boundary_mask)
    if len(boundary_padded) < 6:
        return SurfaceGeometryAnalysis(
            signed_surface,
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            (),
            (),
        )
    padding = np.asarray(signed_surface.padding_zyx, dtype=int)
    boundary_zyx = boundary_padded - padding
    boundary_um = boundary_zyx.astype(float) * spacing
    tree = cKDTree(boundary_um)

    detections: list[_PatchDetection] = []
    samples: list[SurfaceSample] = []
    normals_by_scale: list[np.ndarray] = []
    for scale_index, sigma_um in enumerate(config.surface_sigma_levels_um):
        smoothed = ndimage.gaussian_filter(
            signed_surface.signed_distance_um,
            sigma=physical_sigma_voxels(float(sigma_um), spacing),
            mode="nearest",
        )
        patches, scale_samples, normals = _detect_scale_patches(
            scale_index,
            smoothed,
            boundary_padded,
            boundary_um,
            tree,
            spacing,
            padding,
            config,
        )
        detections.extend(patches)
        samples.extend(scale_samples)
        normals_by_scale.append(normals)
    caps = _consolidate_cap_patches(
        detections, len(config.surface_sigma_levels_um), config
    )
    representative_normals = (
        np.mean(np.stack(normals_by_scale, axis=0), axis=0)
        if normals_by_scale
        else np.empty((0, 3), dtype=float)
    )
    normal_lengths = np.linalg.norm(representative_normals, axis=1)
    valid = normal_lengths > 1e-10
    representative_normals[valid] /= normal_lengths[valid, None]
    return SurfaceGeometryAnalysis(
        signed_surface,
        boundary_zyx.astype(float),
        boundary_um,
        representative_normals,
        tuple(samples),
        caps,
    )


__all__ = [
    "SignedDistanceSurface",
    "SurfaceGeometryAnalysis",
    "analyze_surface_geometry",
    "geometry_padding_voxels",
    "physical_signed_distance",
]
