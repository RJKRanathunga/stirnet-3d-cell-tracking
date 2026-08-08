"""Generate dense foreground, vector, center, and boundary supervision."""

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
    """Keep only selected source IDs and map them deterministically to 1..K."""

    source = np.asarray(labels)
    result = np.zeros(source.shape, dtype=np.int32)
    for local_id, source_id in enumerate(selected_instance_ids, start=1):
        result[source == int(source_id)] = local_id
    return result


def stable_interior_center(
    instance_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    interior_fraction: float = 0.70,
) -> Index3D:
    """Select a deep interior voxel close to the physical centroid."""

    mask = np.asarray(instance_mask, dtype=bool)
    if not mask.any():
        raise ValueError("instance_mask cannot be empty")
    if not 0 < interior_fraction <= 1:
        raise ValueError("interior_fraction must be in (0,1]")
    distance = ndimage.distance_transform_edt(mask, sampling=spacing_zyx_um)
    max_distance = float(distance.max())
    candidates = np.argwhere(mask & (distance >= interior_fraction * max_distance))
    if candidates.size == 0:
        candidates = np.argwhere(mask)
    all_coords = np.argwhere(mask)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    centroid_physical = np.mean(all_coords * spacing, axis=0)
    candidate_physical = candidates * spacing
    squared = np.sum((candidate_physical - centroid_physical) ** 2, axis=1)
    center = candidates[int(np.argmin(squared))]
    return tuple(int(v) for v in center)


def centers_for_labels(
    instance_labels: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    interior_fraction: float,
) -> tuple[Index3D, ...]:
    ids = [int(v) for v in np.unique(instance_labels) if int(v) > 0]
    return tuple(
        stable_interior_center(
            instance_labels == instance_id,
            spacing_zyx_um,
            interior_fraction=interior_fraction,
        )
        for instance_id in ids
    )


def vector_targets(
    instance_labels: np.ndarray,
    centers_zyx: tuple[Index3D, ...],
    spacing_zyx_um: Spacing3D,
    *,
    max_distance_um: float,
) -> np.ndarray:
    """Physical displacement from each GT voxel to its owning stable center."""

    if max_distance_um <= 0:
        raise ValueError("max_distance_um must be positive")
    labels = np.asarray(instance_labels)
    vectors = np.zeros((3, *labels.shape), dtype=np.float32)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    for local_id, center in enumerate(centers_zyx, start=1):
        coords = np.argwhere(labels == local_id)
        if coords.size == 0:
            raise ValueError(f"instance {local_id} has no voxels")
        center_array = np.asarray(center, dtype=np.float32)
        displacement_um = (center_array[None, :] - coords.astype(np.float32)) * spacing
        normalized = np.clip(displacement_um / float(max_distance_um), -1.0, 1.0)
        z, y, x = coords.T
        vectors[:, z, y, x] = normalized.T
    return vectors


def _ownership_inside_component(
    instance_labels: np.ndarray,
    input_component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
) -> np.ndarray:
    ids = [int(v) for v in np.unique(instance_labels) if int(v) > 0]
    if not ids:
        return np.zeros(instance_labels.shape, dtype=np.int32)
    distances = []
    for instance_id in ids:
        distances.append(
            ndimage.distance_transform_edt(
                instance_labels != instance_id,
                sampling=spacing_zyx_um,
            )
        )
    stack = np.stack(distances, axis=0)
    nearest = np.argmin(stack, axis=0)
    ownership = np.zeros(instance_labels.shape, dtype=np.int32)
    for array_index, instance_id in enumerate(ids):
        ownership[(nearest == array_index) & input_component_mask] = instance_id
    return ownership


def internal_boundary_target(
    instance_labels: np.ndarray,
    input_component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    radius_um: float,
) -> np.ndarray:
    """Boundary ridge separating nearest GT ownership within the merged mask.

    This deliberately marks the middle of an artificial bridge even when the
    original GT nuclei have a small background gap between them.
    """

    labels = np.asarray(instance_labels)
    component = np.asarray(input_component_mask, dtype=bool)
    ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    if len(ids) < 2:
        return np.zeros(labels.shape, dtype=np.float32)

    ownership = _ownership_inside_component(labels, component, spacing_zyx_um)
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
    distance = ndimage.distance_transform_edt(~seed, sampling=spacing_zyx_um)
    boundary = (distance <= radius_um) & component
    return boundary.astype(np.float32)


def build_targets(
    instance_labels: np.ndarray,
    input_component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    vector_max_distance_um: float,
    center_sigma_um: float,
    center_interior_fraction: float,
    boundary_radius_um: float,
) -> TargetBundle:
    """Build every dense target required by ``VectorInstanceCNN``."""

    labels = np.asarray(instance_labels, dtype=np.int32)
    foreground = (labels > 0).astype(np.float32)[None, ...]
    centers = centers_for_labels(
        labels,
        spacing_zyx_um,
        interior_fraction=center_interior_fraction,
    )
    vectors = vector_targets(
        labels,
        centers,
        spacing_zyx_um,
        max_distance_um=vector_max_distance_um,
    )
    center = markers_to_heatmap(
        labels.shape,
        centers,
        spacing_zyx_um,
        sigma_um=center_sigma_um,
    )[None, ...]
    boundary = internal_boundary_target(
        labels,
        input_component_mask,
        spacing_zyx_um,
        radius_um=boundary_radius_um,
    )[None, ...]
    return TargetBundle(
        foreground=foreground.astype(np.float32, copy=False),
        vectors_normalized=vectors.astype(np.float32, copy=False),
        boundary=boundary.astype(np.float32, copy=False),
        center=center.astype(np.float32, copy=False),
        instance_labels=labels,
        centers_zyx=centers,
    )
