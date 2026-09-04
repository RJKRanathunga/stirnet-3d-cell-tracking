"""Canonical EDT and effective-marker evidence on cubic CNN voxels."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from typing import Callable

import numpy as np
from scipy import ndimage

from .models import Index3D, Spacing3D

MarkerDetector = Callable[[np.ndarray, Spacing3D], tuple[Index3D, ...]]


@dataclass(frozen=True)
class CanonicalMarkerConfig:
    """Stage-3 peak parameters expressed in canonical voxel units.

    The production Stage-3 code names these quantities ``*_um`` because it was
    designed for native physical microscopy.  Here we deliberately reuse its
    mature peak logic on a normalized unit-cubic lattice, so these values are
    canonical voxel distances despite the downstream legacy field names.
    """

    sigma_levels_vox: tuple[float, ...] = (0.6, 1.0, 1.5, 2.1, 2.7)
    h_levels_vox: tuple[float, ...] = (0.25, 0.45, 0.70, 1.05, 1.50)
    peak_cluster_radius_vox: float = 2.7
    merge_tree_sigma_vox: float = 0.5
    watershed_sigma_vox: float = 1.25


DEFAULT_CANONICAL_MARKER_CONFIG = CanonicalMarkerConfig()


def detect_effective_markers_stage3(
    component_mask: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    marker_config: CanonicalMarkerConfig = DEFAULT_CANONICAL_MARKER_CONFIG,
) -> tuple[Index3D, ...]:
    """Reuse production Stage-3 peak logic on the normalized cubic ROI."""
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("canonical marker detection requires cubic voxels")

    pipeline = import_module("src.source_instances.segmentation.pipeline")
    config_module = import_module("src.source_instances.segmentation.config")
    config = replace(
        config_module.DEFAULT_SEGMENTATION_CONFIG,
        voxel_size_zyx_um=tuple(float(v) for v in spacing),
        sigma_levels_um=tuple(float(v) for v in marker_config.sigma_levels_vox),
        h_levels_um=tuple(float(v) for v in marker_config.h_levels_vox),
        peak_cluster_radius_um=float(marker_config.peak_cluster_radius_vox),
        merge_tree_sigma_um=float(marker_config.merge_tree_sigma_vox),
        watershed_sigma_um=float(marker_config.watershed_sigma_vox),
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
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
) -> tuple[Index3D, ...]:
    mask = np.asarray(component_mask, dtype=bool)
    if not mask.any():
        return ()
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    distance = ndimage.distance_transform_edt(mask, sampling=spacing)
    point = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return (tuple(int(v) for v in point),)


def markers_to_heatmap(
    shape_zyx: tuple[int, int, int],
    marker_positions_zyx: tuple[Index3D, ...],
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    sigma_vox: float | None = None,
    sigma_um: float | None = None,
) -> np.ndarray:
    """Rasterize Gaussian marker evidence.

    ``sigma_vox`` is the preferred argument.  ``sigma_um`` is retained as a
    compatibility alias and has identical meaning in the cubic canonical space.
    """
    sigma = sigma_vox if sigma_vox is not None else sigma_um
    if sigma is None or sigma <= 0:
        raise ValueError("marker sigma must be positive")
    heatmap = np.zeros(shape_zyx, dtype=np.float32)
    if not marker_positions_zyx:
        return heatmap
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=int)
    for raw_position in marker_positions_zyx:
        position = np.asarray(raw_position, dtype=int)
        if np.any(position < 0) or np.any(position >= shape):
            raise ValueError(f"marker outside heatmap: {tuple(position)}")
        radius_vox = np.ceil(3.0 * float(sigma) / spacing).astype(int)
        lo = np.maximum(0, position - radius_vox)
        hi = np.minimum(shape, position + radius_vox + 1)
        axes = [(np.arange(lo[a], hi[a]) - position[a]) * spacing[a] for a in range(3)]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        gaussian = np.exp(-(zz * zz + yy * yy + xx * xx) / (2.0 * float(sigma) ** 2))
        target = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        heatmap[target] = np.maximum(heatmap[target], gaussian.astype(np.float32))
    return heatmap


def normalized_canonical_edt(
    component_mask: np.ndarray,
    spacing_zyx: Spacing3D = (1.0, 1.0, 1.0),
    *,
    clip_distance: float | None = None,
    clip_distance_vox: float | None = None,
) -> np.ndarray:
    clip = clip_distance_vox if clip_distance_vox is not None else clip_distance
    if clip is None or clip <= 0:
        raise ValueError("clip distance must be positive")
    distance = ndimage.distance_transform_edt(
        np.asarray(component_mask, dtype=bool), sampling=spacing_zyx
    )
    return np.clip(distance / float(clip), 0.0, 1.0).astype(np.float32)


def normalized_physical_edt(
    component_mask: np.ndarray,
    spacing_zyx_um: Spacing3D,
    *,
    clip_um: float,
) -> np.ndarray:
    """Deprecated compatibility helper."""
    return normalized_canonical_edt(
        component_mask, spacing_zyx_um, clip_distance=clip_um
    )
