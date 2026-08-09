"""Object-centric isotropic scaling into a fixed cubic CNN lattice."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .models import CanonicalTransform, ComponentBBox, Shape3D, Spacing3D


def bbox_from_instance_slices(
    instance_ids: tuple[int, ...],
    object_slices: Mapping[int, tuple[slice, slice, slice]] | tuple[tuple[slice, slice, slice] | None, ...],
    spacing_zyx_um: Spacing3D,
) -> ComponentBBox:
    boxes: list[tuple[slice, slice, slice]] = []
    for instance_id in instance_ids:
        if isinstance(object_slices, Mapping):
            box = object_slices.get(int(instance_id))
        else:
            index = int(instance_id) - 1
            box = object_slices[index] if 0 <= index < len(object_slices) else None
        if box is None:
            raise ValueError(f"instance {instance_id} does not occur in the label volume")
        boxes.append(box)

    start = np.asarray([min(box[a].start for box in boxes) for a in range(3)], dtype=np.int64)
    stop = np.asarray([max(box[a].stop for box in boxes) for a in range(3)], dtype=np.int64)
    extent_vox = stop - start
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    extent_um = extent_vox.astype(np.float64) * spacing
    center = (start.astype(np.float64) + stop.astype(np.float64) - 1.0) / 2.0
    return ComponentBBox(
        start_zyx=tuple(int(v) for v in start),
        stop_zyx=tuple(int(v) for v in stop),
        extent_vox_zyx=tuple(int(v) for v in extent_vox),
        extent_um_zyx=tuple(float(v) for v in extent_um),
        center_native_zyx=tuple(float(v) for v in center),
    )


def bbox_from_binary_mask(mask: np.ndarray, spacing_zyx_um: Spacing3D) -> ComponentBBox:
    source = np.asarray(mask, dtype=bool)
    coords = np.argwhere(source)
    if coords.size == 0:
        raise ValueError("component mask cannot be empty")
    start = coords.min(axis=0).astype(np.int64)
    stop = coords.max(axis=0).astype(np.int64) + 1
    extent_vox = stop - start
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    extent_um = extent_vox.astype(np.float64) * spacing
    center = (start.astype(np.float64) + stop.astype(np.float64) - 1.0) / 2.0
    return ComponentBBox(
        start_zyx=tuple(int(v) for v in start),
        stop_zyx=tuple(int(v) for v in stop),
        extent_vox_zyx=tuple(int(v) for v in extent_vox),
        extent_um_zyx=tuple(float(v) for v in extent_um),
        center_native_zyx=tuple(float(v) for v in center),
    )


def build_canonical_transform(
    bbox: ComponentBBox,
    *,
    native_spacing_zyx_um: Spacing3D,
    canonical_shape_zyx: Shape3D,
    canonical_spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    component_occupancy: float,
    border_margin_voxels: int = 1,
    min_scale: float = 0.05,
    max_scale: float = 32.0,
) -> CanonicalTransform:
    """Fit a physical group into a cubic canonical ROI with one scalar scale.

    ``min_scale``/``max_scale`` use the transform's historic normalization-scale
    units.  With the production unit canonical spacing they are canonical voxels
    per source micrometre.
    """
    if not 0.0 < component_occupancy <= 1.0:
        raise ValueError("component_occupancy must be in (0,1]")
    if border_margin_voxels < 0:
        raise ValueError("border_margin_voxels cannot be negative")
    if not 0 < min_scale <= max_scale:
        raise ValueError("invalid scale bounds")

    shape = np.asarray(canonical_shape_zyx, dtype=np.int64)
    spacing = np.asarray(canonical_spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("canonical spacing must contain three positive values")
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("canonical spacing must be isotropic for cubic CNN voxels")

    usable_intervals = shape - 1 - 2 * int(border_margin_voxels)
    if np.any(usable_intervals <= 0):
        raise ValueError("border margin leaves no usable canonical span")
    usable_span = usable_intervals.astype(np.float64) * spacing
    desired_span = usable_span * float(component_occupancy)

    extent_um = np.asarray(bbox.extent_um_zyx, dtype=np.float64)
    if np.any(extent_um <= 0):
        raise ValueError("component bbox has non-positive physical extent")

    fit_scale = float(np.min(desired_span / extent_um))
    scale = float(np.clip(fit_scale, min_scale, max_scale))

    return CanonicalTransform(
        native_center_zyx=bbox.center_native_zyx,
        normalization_scale=scale,
        native_spacing_zyx_um=tuple(float(v) for v in native_spacing_zyx_um),
        canonical_spacing_zyx=tuple(float(v) for v in canonical_spacing_zyx),
        canonical_shape_zyx=tuple(int(v) for v in canonical_shape_zyx),
        component_bbox=bbox,
    )


def transformed_bbox_extent_vox(transform: CanonicalTransform) -> tuple[float, float, float]:
    extent_um = np.asarray(transform.component_bbox.extent_um_zyx, dtype=np.float64)
    spacing = np.asarray(transform.canonical_spacing_zyx, dtype=np.float64)
    return tuple(float(v) for v in extent_um * transform.normalization_scale / spacing)


def transformed_bbox_extent_canonical(transform: CanonicalTransform) -> tuple[float, float, float]:
    """Compatibility alias; values are canonical voxel extents for unit cubes."""
    return transformed_bbox_extent_vox(transform)
