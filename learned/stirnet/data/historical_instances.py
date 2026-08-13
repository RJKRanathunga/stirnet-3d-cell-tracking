from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage as ndi


TEMPORAL_CACHE_CONTRACT_VERSION = 3
# Compatibility name for callers that imported the earlier history-only label.
HISTORY_CACHE_CONTRACT_VERSION = TEMPORAL_CACHE_CONTRACT_VERSION


@dataclass(frozen=True)
class NativeRegion:
    slices: tuple[slice, slice, slice]
    origin_voxel_zyx: tuple[int, int, int]


def native_region_bounds(
    center_um: Sequence[float],
    extent_um: float,
    spacing_um: Sequence[float],
    shape: Sequence[int],
) -> NativeRegion:
    """Return the clipped native region enclosing a physical, axis-aligned cube."""

    spacing = np.asarray(spacing_um, np.float64)
    center_vox = np.asarray(center_um, np.float64) / spacing
    half_vox = 0.5 * float(extent_um) / spacing
    low = np.maximum(np.floor(center_vox - half_vox).astype(int) - 1, 0)
    high = np.minimum(np.ceil(center_vox + half_vox).astype(int) + 2, np.asarray(shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    return NativeRegion(slices, tuple(int(v) for v in low))


def extract_native_local_region(
    volume: np.ndarray,
    center_um: Sequence[float],
    extent_um: float,
    spacing_um: Sequence[float],
) -> tuple[np.ndarray, NativeRegion]:
    region = native_region_bounds(center_um, extent_um, spacing_um, volume.shape)
    return np.asarray(volume[region.slices]), region


def canonical_grid_points_um(
    center_um: Sequence[float], extent_um: float, grid_size: int
) -> np.ndarray:
    """Return canonical physical zyx coordinates with global acquisition orientation."""

    if grid_size <= 1:
        raise ValueError("history grid_size must be greater than one")
    offsets = np.linspace(-0.5 * extent_um, 0.5 * extent_um, grid_size, dtype=np.float32)
    zz, yy, xx = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    return np.stack([zz, yy, xx], axis=0) + np.asarray(center_um, np.float32)[:, None, None, None]


def _sample_native(
    volume: np.ndarray,
    points_um: np.ndarray,
    spacing_um: Sequence[float],
    *,
    order: int,
    origin_um: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    coords_vox = (
        points_um - np.asarray(origin_um, np.float32)[:, None, None, None]
    ) / np.asarray(spacing_um, np.float32)[:, None, None, None]
    return ndi.map_coordinates(
        np.asarray(volume, np.float32),
        coords_vox,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    ).astype(np.float32)


def build_historical_instance_grid(
    raw_normalized: np.ndarray,
    instance_labels: np.ndarray,
    instance_id: int,
    spacing_um: Sequence[float],
    dref_um: float,
    *,
    center_um: Sequence[float] | None = None,
    grid_size: int = 12,
    extent_dref: float = 2.5,
    sdf_clip_dref: float = 1.0,
) -> tuple[torch.Tensor, bool]:
    """Build ``[4,G,G,G]`` compact history in a fixed physical cube.

    Channels are occupancy, signed physical distance (positive inside), masked
    robust-normalized intensity, and local robust-normalized intensity. The SDF
    is divided by ``dref_um`` and clipped to ``[-sdf_clip_dref,+sdf_clip_dref]``.
    ``center_um`` is absolute physical zyx from the native volume origin.
    """

    raw = np.asarray(raw_normalized, np.float32)
    labels = np.asarray(instance_labels)
    if raw.shape != labels.shape:
        raise ValueError("raw_normalized and instance_labels must have the same shape")
    full_mask = labels == int(instance_id)
    if not full_mask.any() or not np.isfinite(dref_um) or dref_um <= 0:
        return torch.zeros((4, grid_size, grid_size, grid_size), dtype=torch.float32), False
    spacing = np.asarray(spacing_um, np.float32)
    if center_um is None:
        center_um = np.argwhere(full_mask).mean(axis=0) * spacing
    extent_um = float(extent_dref) * float(dref_um)
    points = canonical_grid_points_um(center_um, extent_um, grid_size)

    # Work only in a physically padded local cube. The extra clipped-SDF margin
    # makes local EDT values exact over the canonical sample domain while
    # avoiding full-frame distance volumes.
    local_extent = extent_um + 2.0 * float(sdf_clip_dref) * float(dref_um)
    region = native_region_bounds(center_um, local_extent, spacing, labels.shape)
    local_raw = raw[region.slices]
    mask = full_mask[region.slices]
    origin_um = np.asarray(region.origin_voxel_zyx, np.float32) * spacing

    inside = ndi.distance_transform_edt(mask, sampling=spacing)
    outside = ndi.distance_transform_edt(~mask, sampling=spacing)
    signed = (inside - outside) / float(dref_um)
    signed = np.clip(signed, -float(sdf_clip_dref), float(sdf_clip_dref))
    occupancy = _sample_native(mask.astype(np.float32), points, spacing, order=1, origin_um=origin_um).clip(0, 1)
    sdf = _sample_native(signed, points, spacing, order=1, origin_um=origin_um).clip(
        -float(sdf_clip_dref), float(sdf_clip_dref)
    )
    sampled_raw = _sample_native(local_raw, points, spacing, order=1, origin_um=origin_um)
    masked_raw = _sample_native(local_raw * mask, points, spacing, order=1, origin_um=origin_um)
    grid = np.stack([occupancy, sdf, masked_raw, sampled_raw], axis=0).astype(np.float32)
    return torch.from_numpy(grid), True


def build_node_instance_grids(
    observations: Sequence[tuple[np.ndarray, np.ndarray, int, Sequence[float], float, Sequence[float] | None]],
    *,
    grid_size: int = 12,
    extent_dref: float = 2.5,
    cache_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cache-ready descriptors from plain observation tuples.

    Each tuple is ``(raw_norm, labels, id, spacing_um, dref_um, center_um)``.
    """

    grids, valid = [], []
    for raw, labels, instance_id, spacing, dref, center in observations:
        grid, ok = build_historical_instance_grid(
            raw,
            labels,
            instance_id,
            spacing,
            dref,
            center_um=center,
            grid_size=grid_size,
            extent_dref=extent_dref,
        )
        grids.append(grid.to(cache_dtype))
        valid.append(ok)
    if not grids:
        return (
            torch.zeros((0, 4, grid_size, grid_size, grid_size), dtype=cache_dtype),
            torch.zeros((0,), dtype=torch.bool),
        )
    return torch.stack(grids), torch.tensor(valid, dtype=torch.bool)


def select_nearest_history_support(
    records: Sequence[object],
    tracklet_id: np.ndarray,
    node_instance_grid: torch.Tensor,
    node_history_valid: torch.Tensor,
    *,
    n_tracklets: int,
    dref_um: float,
    extent_dref: float = 2.5,
    support_channels: int = 2,
) -> dict[str, torch.Tensor]:
    """Select at most the nearest valid past and future observation per tracklet."""

    grid_size = int(node_instance_grid.shape[-1]) if node_instance_grid.ndim == 5 else 12
    support = node_instance_grid.new_zeros(
        (n_tracklets, 2, support_channels, grid_size, grid_size, grid_size)
    )
    valid = torch.zeros((n_tracklets, 2), dtype=torch.bool)
    dt = torch.zeros((n_tracklets, 2), dtype=torch.float32)
    center = torch.zeros((n_tracklets, 2, 3), dtype=torch.float32)
    extent = torch.full((n_tracklets, 2), float(extent_dref * dref_um), dtype=torch.float32)
    node_valid_cpu = node_history_valid.detach().cpu().bool()
    for track in range(n_tracklets):
        indices = [
            i for i, r in enumerate(records)
            if int(tracklet_id[i]) == track and bool(node_valid_cpu[i])
        ]
        candidates = (
            [i for i in indices if int(getattr(records[i], "time_offset")) < 0],
            [i for i in indices if int(getattr(records[i], "time_offset")) > 0],
        )
        for side, rows in enumerate(candidates):
            if not rows:
                continue
            chosen = min(rows, key=lambda i: abs(int(getattr(records[i], "time_offset"))))
            support[track, side] = node_instance_grid[chosen, :support_channels]
            valid[track, side] = True
            dt[track, side] = float(getattr(records[chosen], "time_offset"))
            center[track, side] = torch.as_tensor(
                getattr(records[chosen], "position_um"), dtype=torch.float32
            )
    return {
        "history_support": support,
        "history_support_valid": valid,
        "history_support_dt": dt,
        "history_support_center_um": center,
        "history_support_extent_um": extent,
    }


def component_overlap_from_projected_support(
    support: torch.Tensor,
    valid: torch.Tensor,
    projected_center_um: np.ndarray,
    extent_um: torch.Tensor,
    current_labels: np.ndarray,
    spacing_um: Sequence[float],
    *,
    occupancy_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign support to current components on CPU using sparse positive samples."""

    labels = np.asarray(current_labels)
    spacing = np.asarray(spacing_um, np.float32)
    patch_center = 0.5 * (np.asarray(labels.shape, np.float32) - 1) * spacing
    m = support.shape[0]
    best_id = np.full(m, -1, np.int64)
    best = np.zeros(m, np.float32)
    second = np.zeros(m, np.float32)
    available = np.zeros(m, np.bool_)
    grid_size = support.shape[-1]
    for index in range(m):
        component_counts: dict[int, int] = {}
        positive_count = 0
        for side in range(2):
            if not bool(valid[index, side]):
                continue
            occ = support[index, side, 0].detach().float().cpu().numpy()
            rows = np.argwhere(occ > occupancy_threshold)
            if not len(rows):
                continue
            offsets = (rows / max(grid_size - 1, 1) - 0.5) * float(extent_um[index, side])
            voxels = np.rint((projected_center_um[index] + offsets + patch_center) / spacing).astype(int)
            in_bounds = np.all((voxels >= 0) & (voxels < np.asarray(labels.shape)), axis=1)
            if not in_bounds.any():
                continue
            sampled = labels[tuple(voxels[in_bounds].T)]
            positive_count += int(in_bounds.sum())
            for component, count in zip(*np.unique(sampled[sampled > 0], return_counts=True)):
                component_counts[int(component)] = component_counts.get(int(component), 0) + int(count)
        if positive_count == 0:
            continue
        available[index] = True
        ranked = sorted(component_counts.items(), key=lambda item: item[1], reverse=True)
        if ranked:
            best_id[index] = ranked[0][0]
            best[index] = ranked[0][1] / positive_count
        if len(ranked) > 1:
            second[index] = ranked[1][1] / positive_count
    return best_id, best, second, available


def projected_support_overlap(
    support_a: torch.Tensor,
    valid_a: torch.Tensor,
    center_a_um: np.ndarray,
    extent_a_um: torch.Tensor,
    support_b: torch.Tensor,
    valid_b: torch.Tensor,
    center_b_um: np.ndarray,
    extent_b_um: torch.Tensor,
) -> tuple[float, bool]:
    """Approximate projected occupancy Dice using compact physical point sets."""

    point_sets = []
    resolution = np.inf
    for support, valid, center, extent in (
        (support_a, valid_a, center_a_um, extent_a_um),
        (support_b, valid_b, center_b_um, extent_b_um),
    ):
        points = []
        g = support.shape[-1]
        for side in range(2):
            if not bool(valid[side]):
                continue
            rows = np.argwhere(support[side, 0].detach().float().cpu().numpy() > 0.5)
            if len(rows):
                scale = float(extent[side]) / max(g - 1, 1)
                resolution = min(resolution, scale)
                points.append(center + (rows / max(g - 1, 1) - 0.5) * float(extent[side]))
        point_sets.append(np.concatenate(points) if points else np.zeros((0, 3), np.float32))
    if not len(point_sets[0]) or not len(point_sets[1]) or not np.isfinite(resolution):
        return 0.0, False
    sets = [set(map(tuple, np.rint(points / resolution).astype(np.int64))) for points in point_sets]
    intersection = len(sets[0] & sets[1])
    return float(2 * intersection / max(len(sets[0]) + len(sets[1]), 1)), True


def pairwise_convergence_statistics(
    track_a: Sequence[object], track_b: Sequence[object], dref_um: float
) -> dict[str, float | bool | np.ndarray]:
    """Compute common-past separation and a positive-when-closing velocity."""

    by_a = {int(getattr(r, "time_offset")): r for r in track_a if int(getattr(r, "time_offset")) < 0}
    by_b = {int(getattr(r, "time_offset")): r for r in track_b if int(getattr(r, "time_offset")) < 0}
    common = sorted(set(by_a) & set(by_b), reverse=True)
    nearest_valid = len(common) >= 1
    older_valid = len(common) >= 2
    nearest = older = change = closing = 0.0
    rel_velocity = np.zeros(3, np.float32)
    if nearest_valid:
        t1 = common[0]
        pa1 = np.asarray(getattr(by_a[t1], "position_um"), np.float32)
        pb1 = np.asarray(getattr(by_b[t1], "position_um"), np.float32)
        nearest = float(np.linalg.norm(pb1 - pa1) / max(dref_um, 1e-8))
    if older_valid:
        t0 = common[1]
        pa0 = np.asarray(getattr(by_a[t0], "position_um"), np.float32)
        pb0 = np.asarray(getattr(by_b[t0], "position_um"), np.float32)
        older = float(np.linalg.norm(pb0 - pa0) / max(dref_um, 1e-8))
        elapsed = float(t1 - t0)
        change = nearest - older
        closing = (older - nearest) / max(elapsed, 1e-8)
        va = (pa1 - pa0) / max(elapsed, 1e-8)
        vb = (pb1 - pb0) / max(elapsed, 1e-8)
        rel_velocity = (vb - va) / max(dref_um, 1e-8)
    return {
        "nearest_distance": nearest,
        "older_distance": older,
        "distance_change": change,
        "closing_speed": closing,
        "relative_velocity": rel_velocity,
        "nearest_valid": nearest_valid,
        "older_valid": older_valid,
    }
