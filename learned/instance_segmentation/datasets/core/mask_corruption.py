"""Generate Stage-2-like connected masks in canonical ROI coordinates."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .models import Spacing3D

_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def canonical_ball(radius: float, spacing_zyx: Spacing3D) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    radii = np.ceil(float(radius) / spacing).astype(int)
    axes = [np.arange(-r, r + 1) * spacing[i] for i, r in enumerate(radii)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return (zz * zz + yy * yy + xx * xx) <= float(radius) ** 2


# Backward-compatible name.  Radius is interpreted in the coordinate system
# associated with spacing_zyx_um; in this package that is now usually canonical.
physical_ball = canonical_ball


def _surface_coordinates(mask: np.ndarray) -> np.ndarray:
    surface = mask & ~ndimage.binary_erosion(mask, structure=_STRUCTURE_6, border_value=0)
    coords = np.argwhere(surface)
    return coords if coords.size else np.argwhere(mask)


def _closest_component_points(
    component_labels: np.ndarray,
    spacing_zyx: Spacing3D,
) -> tuple[np.ndarray, np.ndarray]:
    component_ids = [int(v) for v in np.unique(component_labels) if int(v) > 0]
    if len(component_ids) < 2:
        raise ValueError("at least two connected components are required")
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    surfaces: dict[int, np.ndarray] = {}
    trees: dict[int, cKDTree] = {}
    for component_id in component_ids:
        coords = _surface_coordinates(component_labels == component_id)
        surfaces[component_id] = coords
        trees[component_id] = cKDTree(coords * spacing)

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for index, first in enumerate(component_ids[:-1]):
        coords_first = surfaces[first]
        physical_first = coords_first * spacing
        for second in component_ids[index + 1 :]:
            distances, indices = trees[second].query(physical_first, k=1)
            local_index = int(np.argmin(distances))
            distance = float(distances[local_index])
            point_a = coords_first[local_index]
            point_b = surfaces[second][int(indices[local_index])]
            if best is None or distance < best[0]:
                best = (distance, point_a, point_b)
    if best is None:
        raise RuntimeError("failed to find closest connected components")
    return best[1], best[2]


def _line_mask(shape: tuple[int, int, int], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end.astype(np.float64) - start.astype(np.float64)
    steps = max(2, int(np.ceil(np.max(np.abs(delta)))) + 1)
    coordinates = np.rint(
        start[None, :] + np.linspace(0.0, 1.0, steps)[:, None] * delta[None, :]
    ).astype(int)
    coordinates = np.clip(coordinates, 0, np.asarray(shape) - 1)
    result = np.zeros(shape, dtype=bool)
    result[tuple(coordinates.T)] = True
    return result


def connect_components_with_bridges(
    mask: np.ndarray,
    spacing_zyx: Spacing3D,
    *,
    bridge_radius: float = 0.45,
) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if not result.any():
        raise ValueError("mask cannot be empty")
    while True:
        components, count = ndimage.label(result, structure=_STRUCTURE_6)
        if count <= 1:
            return result
        start, end = _closest_component_points(components, spacing_zyx)
        bridge = _line_mask(result.shape, start, end)
        if bridge_radius > 0:
            bridge = ndimage.binary_dilation(
                bridge, structure=canonical_ball(bridge_radius, spacing_zyx)
            )
        result |= bridge


def build_stage2_like_component(
    selected_instance_labels: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    bridge_radius_um: float = 0.45,
    closing_radius_um: float = 0.0,
) -> np.ndarray:
    """Turn selected GT instances into one connected canonical input component.

    Parameter names retain ``_um`` for API compatibility, but when called by
    the new SampleBuilder they are canonical-coordinate units.
    """
    selected = np.asarray(selected_instance_labels)
    mask = selected > 0
    if not mask.any():
        raise ValueError("selected_instance_labels contain no foreground")
    result = connect_components_with_bridges(
        mask, spacing_zyx_um, bridge_radius=bridge_radius_um
    )
    if closing_radius_um > 0:
        structure = canonical_ball(closing_radius_um, spacing_zyx_um)
        result = ndimage.binary_closing(result, structure=structure)
        result = connect_components_with_bridges(
            result, spacing_zyx_um, bridge_radius=bridge_radius_um
        )
    return result.astype(bool, copy=False)
