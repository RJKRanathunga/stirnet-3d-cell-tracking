"""Reuse Stage 3 effective peaks and convert them into a CNN marker channel."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from typing import Callable

import numpy as np
from scipy import ndimage

from .models import Index3D, Spacing3D

MarkerDetector = Callable[[np.ndarray, Spacing3D], tuple[Index3D, ...]]


def detect_effective_markers_stage3(
    component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
) -> tuple[Index3D, ...]:
    """Run the repository's real effective-peak detector with geometry disabled.

    Dynamic import is required because the existing production package is named
    ``src.03_segmentation``. No GT information enters this detector.
    """

    pipeline = import_module("src.03_segmentation.pipeline")
    config_module = import_module("src.03_segmentation.config")
    config = replace(
        config_module.DEFAULT_SEGMENTATION_CONFIG,
        voxel_size_zyx_um=tuple(float(v) for v in spacing_zyx_um),
        enable_geometric_completion=False,
    )
    analysis = pipeline.analyze_component_crop(
        np.asarray(component_mask, dtype=bool),
        config=config,
        retain_debug_artifacts=False,
        force_geometric_analysis=False,
    )
    return tuple(tuple(int(v) for v in point) for point in analysis.marker_positions_zyx)


def deepest_point_marker(
    component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
) -> tuple[Index3D, ...]:
    """Simple explicit fallback useful for isolated tests, never the default."""

    mask = np.asarray(component_mask, dtype=bool)
    if not mask.any():
        return ()
    distance = ndimage.distance_transform_edt(mask, sampling=spacing_zyx_um)
    point = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return (tuple(int(v) for v in point),)


def markers_to_heatmap(
    shape_zyx: tuple[int, int, int],
    marker_positions_zyx: tuple[Index3D, ...],
    spacing_zyx_um: Spacing3D,
    *,
    sigma_um: float,
) -> np.ndarray:
    """Return max-composed physical Gaussian bumps centered on marker points."""

    if sigma_um <= 0:
        raise ValueError("sigma_um must be positive")
    heatmap = np.zeros(shape_zyx, dtype=np.float32)
    if not marker_positions_zyx:
        return heatmap

    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=int)
    for raw_position in marker_positions_zyx:
        position = np.asarray(raw_position, dtype=int)
        if np.any(position < 0) or np.any(position >= shape):
            raise ValueError(f"marker outside heatmap: {tuple(position)}")
        radius_vox = np.ceil(3.0 * sigma_um / spacing).astype(int)
        lo = np.maximum(0, position - radius_vox)
        hi = np.minimum(shape, position + radius_vox + 1)
        axes = [
            (np.arange(lo[a], hi[a]) - position[a]) * spacing[a]
            for a in range(3)
        ]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        gaussian = np.exp(-(zz * zz + yy * yy + xx * xx) / (2.0 * sigma_um**2))
        target = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        heatmap[target] = np.maximum(heatmap[target], gaussian.astype(np.float32))
    return heatmap


def normalized_physical_edt(
    component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    clip_um: float,
) -> np.ndarray:
    """EDT channel in [0,1] using a fixed physical clipping scale."""

    if clip_um <= 0:
        raise ValueError("clip_um must be positive")
    distance = ndimage.distance_transform_edt(
        np.asarray(component_mask, dtype=bool),
        sampling=spacing_zyx_um,
    )
    return np.clip(distance / float(clip_um), 0.0, 1.0).astype(np.float32)
