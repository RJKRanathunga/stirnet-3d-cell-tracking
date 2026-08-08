"""Physical instance-neighbour discovery on native dense annotations."""

from __future__ import annotations

from math import ceil

import numpy as np
from scipy import ndimage

from .models import AdjacencyEdge, Spacing3D


def _expanded_slice(
    source_slice: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    padding_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    expanded: list[slice] = []
    for axis, current in enumerate(source_slice):
        start = max(0, int(current.start or 0) - padding_zyx[axis])
        stop = min(shape[axis], int(current.stop or shape[axis]) + padding_zyx[axis])
        expanded.append(slice(start, stop))
    return tuple(expanded)  # type: ignore[return-value]


def _centroids_physical(
    labels: np.ndarray,
    instance_ids: np.ndarray,
    spacing_zyx_um: Spacing3D,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    for instance_id in instance_ids:
        center = ndimage.center_of_mass(labels == int(instance_id))
        result[int(instance_id)] = np.asarray(center, dtype=np.float64) * spacing
    return result


def build_instance_adjacency(
    instance_labels: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    max_distance_um: float,
    min_instance_voxels: int = 1,
) -> tuple[AdjacencyEdge, ...]:
    """Find nearby GT instances using local physical distance transforms.

    ``separation_um`` is a conservative grid-based estimate. For face-adjacent
    labels it is reported as zero by subtracting one local voxel pitch from the
    nearest voxel-center distance. The value is intended for candidate ranking,
    not precise morphometry.
    """

    labels = np.asarray(instance_labels)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("instance_labels must be a 3-D integer array")
    if max_distance_um <= 0:
        raise ValueError("max_distance_um must be positive")
    spacing = tuple(float(v) for v in spacing_zyx_um)
    if any(v <= 0 for v in spacing):
        raise ValueError("spacing values must be positive")

    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    keep = ids[counts >= int(min_instance_voxels)]
    if keep.size < 2:
        return ()

    max_id = int(labels.max(initial=0))
    object_slices = ndimage.find_objects(labels, max_label=max_id)
    padding = tuple(int(ceil(max_distance_um / s)) + 1 for s in spacing)
    centroids = _centroids_physical(labels, keep, spacing)
    keep_set = {int(v) for v in keep}
    min_pitch = min(spacing)

    pair_distance: dict[tuple[int, int], float] = {}
    for raw_id in keep:
        instance_id = int(raw_id)
        source_slice = object_slices[instance_id - 1]
        if source_slice is None:
            continue
        region_slice = _expanded_slice(source_slice, labels.shape, padding)
        local = labels[region_slice]
        source = local == instance_id
        if not source.any():
            continue
        distance = ndimage.distance_transform_edt(~source, sampling=spacing)
        near = distance <= (max_distance_um + max_pitch(spacing))
        candidates = np.unique(local[near & (local > 0) & (local != instance_id)])
        for candidate_raw in candidates:
            candidate = int(candidate_raw)
            if candidate not in keep_set:
                continue
            key = tuple(sorted((instance_id, candidate)))
            if key[0] == key[1]:
                continue
            candidate_distance = float(np.min(distance[local == candidate]))
            separation = max(0.0, candidate_distance - min_pitch)
            if separation <= max_distance_um:
                previous = pair_distance.get(key)
                if previous is None or separation < previous:
                    pair_distance[key] = separation

    edges: list[AdjacencyEdge] = []
    for (a, b), separation in sorted(pair_distance.items()):
        centroid_distance = float(np.linalg.norm(centroids[a] - centroids[b]))
        edges.append(
            AdjacencyEdge(
                instance_a=a,
                instance_b=b,
                separation_um=separation,
                centroid_distance_um=centroid_distance,
            )
        )
    return tuple(edges)


def max_pitch(spacing_zyx_um: Spacing3D) -> float:
    return float(max(spacing_zyx_um))
