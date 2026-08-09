"""Scalable physical neighbour discovery on native dense annotations."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .models import AdjacencyEdge, Spacing3D

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def _bbox_gap_um(a: tuple[slice, slice, slice], b: tuple[slice, slice, slice], spacing: np.ndarray) -> np.ndarray:
    gap = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        if a[axis].stop <= b[axis].start:
            vox_gap = b[axis].start - a[axis].stop + 1
        elif b[axis].stop <= a[axis].start:
            vox_gap = a[axis].start - b[axis].stop + 1
        else:
            vox_gap = 0
        gap[axis] = max(0, vox_gap) * spacing[axis]
    return gap


def _surface_points_global(
    labels: np.ndarray,
    instance_id: int,
    bbox: tuple[slice, slice, slice],
) -> np.ndarray:
    local = labels[bbox] == instance_id
    surface = local & ~ndimage.binary_erosion(local, structure=_STRUCTURE_6, border_value=0)
    coords = np.argwhere(surface)
    if coords.size == 0:
        coords = np.argwhere(local)
    offset = np.asarray([sl.start for sl in bbox], dtype=np.int64)
    return coords + offset


def build_instance_adjacency(
    instance_labels: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    max_distance_um: float = 2.5,
) -> tuple[AdjacencyEdge, ...]:
    """Return surface-neighbour edges using bbox sweep + exact surface distance."""
    labels = np.asarray(instance_labels)
    if labels.ndim != 3:
        raise ValueError("instance_labels must be 3-D")
    if max_distance_um < 0:
        raise ValueError("max_distance_um cannot be negative")
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    boxes_raw = ndimage.find_objects(labels)
    items: list[tuple[int, tuple[slice, slice, slice]]] = [
        (i + 1, box)
        for i, box in enumerate(boxes_raw)
        if box is not None
    ]
    items.sort(key=lambda item: item[1][2].start * spacing[2])

    surface_cache: dict[int, np.ndarray] = {}
    tree_cache: dict[int, cKDTree] = {}

    def surface(instance_id: int, box: tuple[slice, slice, slice]) -> np.ndarray:
        if instance_id not in surface_cache:
            surface_cache[instance_id] = _surface_points_global(labels, instance_id, box)
        return surface_cache[instance_id]

    def tree(instance_id: int, box: tuple[slice, slice, slice]) -> cKDTree:
        if instance_id not in tree_cache:
            tree_cache[instance_id] = cKDTree(surface(instance_id, box) * spacing)
        return tree_cache[instance_id]

    edges: list[AdjacencyEdge] = []
    for i, (id_a, box_a) in enumerate(items):
        x_stop_um = (box_a[2].stop - 1) * spacing[2]
        coords_a = None
        centroid_a = None
        for id_b, box_b in items[i + 1 :]:
            x_start_um = box_b[2].start * spacing[2]
            if x_start_um - x_stop_um > max_distance_um + spacing[2]:
                break
            gap = _bbox_gap_um(box_a, box_b, spacing)
            if float(np.linalg.norm(gap)) > max_distance_um + float(np.max(spacing)):
                continue
            if coords_a is None:
                coords_a = surface(id_a, box_a) * spacing
                centroid_a = coords_a.mean(axis=0)
            distances, _ = tree(id_b, box_b).query(coords_a, k=1)
            separation = float(np.min(distances))
            # Surface voxel centers one voxel apart can represent touching faces;
            # subtract half-pitches conservatively but never go negative.
            separation = max(0.0, separation - 0.5 * float(np.linalg.norm(spacing)))
            if separation > max_distance_um:
                continue
            coords_b = surface(id_b, box_b) * spacing
            centroid_distance = float(np.linalg.norm(centroid_a - coords_b.mean(axis=0)))
            a, b = sorted((id_a, id_b))
            edges.append(AdjacencyEdge(a, b, separation, centroid_distance))

    edges.sort(key=lambda e: (e.separation_um, e.centroid_distance_um, e.instance_a, e.instance_b))
    return tuple(edges)


def max_pitch(spacing_zyx_um: Spacing3D) -> float:
    return float(max(spacing_zyx_um))
