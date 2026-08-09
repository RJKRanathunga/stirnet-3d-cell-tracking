"""Dense foreground, vector, center, and boundary supervision in cubic voxels."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .marker_heatmap import markers_to_heatmap
from .models import Index3D, Spacing3D, TargetBundle

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def relabel_selected_instances(
    labels: np.ndarray,
    selected_instance_ids: tuple[int, ...],
) -> np.ndarray:
    source = np.asarray(labels)
    result = np.zeros(source.shape, dtype=np.int32)
    for local_id, source_id in enumerate(selected_instance_ids, start=1):
        result[source == int(source_id)] = local_id
    return result


def stable_interior_center(
    instance_mask: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    interior_fraction: float = 0.70,
) -> Index3D:
    mask = np.asarray(instance_mask, dtype=bool)
    if not mask.any():
        raise ValueError("instance_mask cannot be empty")
    if not 0 < interior_fraction <= 1:
        raise ValueError("interior_fraction must be in (0,1]")
    distance = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
    max_distance = float(distance.max())
    candidates = np.argwhere(mask & (distance >= interior_fraction * max_distance))
    if candidates.size == 0:
        candidates = np.argwhere(mask)
    all_coords = np.argwhere(mask)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    centroid = np.mean(all_coords * spacing, axis=0)
    candidate_positions = candidates * spacing
    squared = np.sum((candidate_positions - centroid) ** 2, axis=1)
    center = candidates[int(np.argmin(squared))]
    return tuple(int(v) for v in center)


def centers_for_labels(
    instance_labels: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    interior_fraction: float,
) -> tuple[Index3D, ...]:
    ids = [int(v) for v in np.unique(instance_labels) if int(v) > 0]
    return tuple(
        stable_interior_center(
            instance_labels == instance_id,
            spacing_zyx,
            interior_fraction=interior_fraction,
        )
        for instance_id in ids
    )


def vector_targets_canonical(
    instance_labels: np.ndarray,
    centers_zyx: tuple[Index3D, ...],
) -> np.ndarray:
    """Displacement to owning center normalized by canonical axis span."""
    labels = np.asarray(instance_labels)
    vectors = np.zeros((3, *labels.shape), dtype=np.float32)
    span = np.maximum(np.asarray(labels.shape, dtype=np.float32) - 1.0, 1.0)
    for local_id, center in enumerate(centers_zyx, start=1):
        coords = np.argwhere(labels == local_id)
        if coords.size == 0:
            raise ValueError(f"instance {local_id} has no voxels")
        center_array = np.asarray(center, dtype=np.float32)
        normalized = (center_array[None, :] - coords.astype(np.float32)) / span[None, :]
        normalized = np.clip(normalized, -1.0, 1.0)
        z, y, x = coords.T
        vectors[:, z, y, x] = normalized.T
    return vectors


def vectors_to_canonical_displacement(
    vectors_normalized: np.ndarray,
    spatial_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    span = np.maximum(np.asarray(spatial_shape_zyx, dtype=np.float32) - 1.0, 1.0)
    return np.asarray(vectors_normalized, dtype=np.float32) * span[:, None, None, None]


def _ownership_inside_component(
    instance_labels: np.ndarray,
    input_component_mask: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
) -> np.ndarray:
    ids = [int(v) for v in np.unique(instance_labels) if int(v) > 0]
    if not ids:
        return np.zeros(instance_labels.shape, dtype=np.int32)
    distances = [
        ndimage.distance_transform_edt(instance_labels != instance_id, sampling=spacing_zyx)
        for instance_id in ids
    ]
    stack = np.stack(distances, axis=0)
    nearest = np.argmin(stack, axis=0)
    ownership = np.zeros(instance_labels.shape, dtype=np.int32)
    for array_index, instance_id in enumerate(ids):
        ownership[(nearest == array_index) & input_component_mask] = instance_id
    return ownership


def internal_boundary_target(
    instance_labels: np.ndarray,
    input_component_mask: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    radius_vox: float,
) -> np.ndarray:
    labels = np.asarray(instance_labels)
    component = np.asarray(input_component_mask, dtype=bool)
    ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    if len(ids) < 2:
        return np.zeros(labels.shape, dtype=np.float32)
    ownership = _ownership_inside_component(labels, component, spacing_zyx)
    seed = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(None, -1)
        second_slice[axis] = slice(1, None)
        first = tuple(first_slice)
        second = tuple(second_slice)
        a = ownership[first]
        b = ownership[second]
        difference = (a > 0) & (b > 0) & (a != b)
        seed[first] |= difference
        seed[second] |= difference
    if not seed.any():
        return np.zeros(labels.shape, dtype=np.float32)
    distance = ndimage.distance_transform_edt(~seed, sampling=spacing_zyx)
    return ((distance <= float(radius_vox)) & component).astype(np.float32)


def build_targets(
    instance_labels: np.ndarray,
    input_component_mask: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    center_sigma_vox: float | None = None,
    center_interior_fraction: float,
    boundary_radius_vox: float | None = None,
    # Compatibility aliases for old callers.
    center_sigma_um: float | None = None,
    boundary_radius_um: float | None = None,
    vector_max_distance_um: float | None = None,
) -> TargetBundle:
    """Build scale-invariant targets on the normalized cubic lattice."""
    del vector_max_distance_um
    center_sigma = center_sigma_vox if center_sigma_vox is not None else center_sigma_um
    boundary_radius = boundary_radius_vox if boundary_radius_vox is not None else boundary_radius_um
    if center_sigma is None or boundary_radius is None:
        raise ValueError("center sigma and boundary radius are required")

    labels = np.asarray(instance_labels, dtype=np.int32)
    foreground = (labels > 0).astype(np.float32)[None, ...]
    centers = centers_for_labels(
        labels, spacing_zyx, interior_fraction=center_interior_fraction
    )
    vectors = vector_targets_canonical(labels, centers)
    center = markers_to_heatmap(
        labels.shape, centers, spacing_zyx, sigma_vox=float(center_sigma)
    )[None, ...]
    boundary = internal_boundary_target(
        labels,
        input_component_mask,
        spacing_zyx,
        radius_vox=float(boundary_radius),
    )[None, ...]
    return TargetBundle(
        foreground=foreground.astype(np.float32, copy=False),
        vectors_normalized=vectors.astype(np.float32, copy=False),
        boundary=boundary.astype(np.float32, copy=False),
        center=center.astype(np.float32, copy=False),
        instance_labels=labels,
        centers_zyx=centers,
    )
