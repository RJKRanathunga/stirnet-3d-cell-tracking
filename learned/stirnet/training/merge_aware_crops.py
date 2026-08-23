from __future__ import annotations

"""Spacing-aware, merge-first greedy crop planning for STIR-Net training."""

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import torch
from torch import Tensor
from .crops import CropSpec


@dataclass(frozen=True)
class MergeAwareCropRecord:
    batch_index: int
    slices_zyx: tuple[slice, slice, slice]
    complete_cell_ids: tuple[int, ...]
    partial_cell_ids: tuple[int, ...]
    true_boundary_cell_ids: tuple[int, ...]
    merge_source_ids: tuple[int, ...] = ()
    merge_gt_ids: tuple[int, ...] = ()
    candidate_type: str = 'coverage'

    @property
    def covered_cell_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.complete_cell_ids) | set(self.true_boundary_cell_ids)))


@dataclass(frozen=True)
class MergeAwareCropManifest:
    records: tuple[tuple[MergeAwareCropRecord, ...], ...]
    crop_shape_zyx: tuple[int, int, int]
    min_complete_cells: int
    preferred_complete_cells: int
    views_per_cell: int
    context_um: float
    merge_min_overlap_voxels: int
    merge_min_gt_fraction: float
    uncoverable_cell_ids: tuple[tuple[int, ...], ...]
    uncoverable_merge_source_ids: tuple[tuple[int, ...], ...]
    source_signature: tuple


@dataclass(frozen=True)
class _Box:
    cell_id: int
    low: np.ndarray
    high: np.ndarray
    boundary: bool
    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.low + self.high - 1)



# _BOX_INDEX_MANIFEST_PERF_V1
@dataclass(frozen=True)
class _BoxIndex:
    """Vectorized exact AABB index used while constructing crop manifests."""

    ids: np.ndarray
    low: np.ndarray
    high: np.ndarray
    boundary: np.ndarray


def _make_box_index(boxes: dict[int, _Box]) -> _BoxIndex:
    ordered_ids = np.asarray(sorted(boxes), dtype=np.int64)
    if ordered_ids.size == 0:
        return _BoxIndex(
            ids=ordered_ids,
            low=np.empty((0, 3), dtype=np.int64),
            high=np.empty((0, 3), dtype=np.int64),
            boundary=np.empty((0,), dtype=bool),
        )
    ordered = [boxes[int(cell_id)] for cell_id in ordered_ids.tolist()]
    return _BoxIndex(
        ids=ordered_ids,
        low=np.stack([box.low for box in ordered], axis=0),
        high=np.stack([box.high for box in ordered], axis=0),
        boundary=np.asarray([box.boundary for box in ordered], dtype=bool),
    )

def source_signature(gt: Tensor, current: Tensor | None, spacing: Tensor | None) -> tuple:
    return (
        tuple(gt.shape), int(gt.data_ptr()),
        None if current is None else tuple(current.shape),
        -1 if current is None else int(current.data_ptr()),
        () if spacing is None else tuple(float(x) for x in torch.as_tensor(spacing).detach().cpu().reshape(-1).tolist()),
    )


def _boxes(labels: np.ndarray) -> dict[int, _Box]:
    shape = np.asarray(labels.shape)
    out = {}
    for cell_id, bbox in enumerate(ndi.find_objects(labels), 1):
        if bbox is None: continue
        low = np.asarray([int(s.start) for s in bbox], dtype=np.int64)
        high = np.asarray([int(s.stop) for s in bbox], dtype=np.int64)
        out[cell_id] = _Box(cell_id, low, high, bool(np.any(low == 0) or np.any(high == shape)))
    return out


def _fit(low, high, full_shape, crop_shape, spacing, context_um):
    full = np.asarray(full_shape, dtype=np.int64)
    size = np.minimum(full, np.asarray(crop_shape, dtype=np.int64))
    for context in (float(context_um), 0.0):
        margin = np.ceil(context / np.maximum(spacing, 1e-6)).astype(np.int64)
        lo = np.maximum(low - margin, 0)
        hi = np.minimum(high + margin, full)
        if np.any(hi - lo > size):
            continue
        min_start = np.maximum(hi - size, 0)
        max_start = np.minimum(lo, full - size)
        if np.any(min_start > max_start):
            continue
        desired = np.rint(0.5 * (lo + hi) - 0.5 * size).astype(np.int64)
        start = np.minimum(np.maximum(desired, min_start), max_start)
        return tuple(slice(int(a), int(b)) for a, b in zip(start, start + size))
    return None


def _record(
    batch_index,
    crop,
    boxes,
    *,
    box_index: _BoxIndex | None = None,
    kind='coverage',
    merge_sources=(),
    merge_gt=(),
):
    lo = np.asarray([int(s.start) for s in crop], dtype=np.int64)
    hi = np.asarray([int(s.stop) for s in crop], dtype=np.int64)
    index = _make_box_index(boxes) if box_index is None else box_index

    if index.ids.size == 0:
        complete = partial = boundary = ()
    else:
        intersects = (
            np.all(index.high > lo[None, :], axis=1)
            & np.all(index.low < hi[None, :], axis=1)
        )
        hit = np.flatnonzero(intersects)

        if hit.size == 0:
            complete = partial = boundary = ()
        else:
            hit_low = index.low[hit]
            hit_high = index.high[hit]
            contained = (
                np.all(hit_low >= lo[None, :], axis=1)
                & np.all(hit_high <= hi[None, :], axis=1)
            )
            hit_boundary = index.boundary[hit]
            hit_ids = index.ids[hit]

            boundary = tuple(
                int(v) for v in hit_ids[contained & hit_boundary].tolist()
            )
            complete = tuple(
                int(v) for v in hit_ids[contained & ~hit_boundary].tolist()
            )
            partial = tuple(int(v) for v in hit_ids[~contained].tolist())

    return MergeAwareCropRecord(
        batch_index,
        crop,
        complete,
        partial,
        boundary,
        tuple(sorted(set(merge_sources))),
        tuple(sorted(set(merge_gt))),
        kind,
    )

def _union(rows: Iterable[_Box]):
    rows = list(rows)
    return np.min(np.stack([r.low for r in rows]), 0), np.max(np.stack([r.high for r in rows]), 0)


def _merge_rows(
    batch_index,
    gt,
    current,
    boxes,
    box_index,
    full_shape,
    crop_shape,
    spacing,
    context_um,
    min_overlap,
    min_gt_fraction,
):
    if current is None or not np.any(current > 0):
        return [], ()

    pos = gt[gt > 0]
    ids, counts = (
        np.unique(pos, return_counts=True)
        if pos.size
        else (np.asarray([]), np.asarray([]))
    )
    gt_count = {
        int(i): int(n) for i, n in zip(ids.tolist(), counts.tolist())
    }

    aggregated = {}
    unfit = []

    for source_id, bbox in enumerate(ndi.find_objects(current), 1):
        if bbox is None:
            continue
        mask = current[bbox] == source_id
        overlap = gt[bbox][mask]
        overlap = overlap[overlap > 0]
        if not overlap.size:
            continue

        gids, nums = np.unique(overlap, return_counts=True)
        meaningful = [
            int(g)
            for g, n in zip(gids.tolist(), nums.tolist())
            if int(n) >= min_overlap
            and float(n) / max(gt_count.get(int(g), 0), 1)
            >= min_gt_fraction
        ]
        if len(meaningful) < 2:
            continue

        low = np.asarray([int(s.start) for s in bbox])
        high = np.asarray([int(s.stop) for s in bbox])
        for gid in meaningful:
            if gid in boxes:
                low = np.minimum(low, boxes[gid].low)
                high = np.maximum(high, boxes[gid].high)

        crop = _fit(
            low,
            high,
            full_shape,
            crop_shape,
            spacing,
            context_um,
        )
        if crop is None:
            unfit.append(source_id)
            continue

        key = tuple((int(s.start), int(s.stop)) for s in crop)
        if key not in aggregated:
            aggregated[key] = [crop, set(), set()]
        aggregated[key][1].add(int(source_id))
        aggregated[key][2].update(int(gid) for gid in meaningful)

    rows = [
        _record(
            batch_index,
            crop,
            boxes,
            box_index=box_index,
            kind='merge',
            merge_sources=tuple(sorted(source_ids)),
            merge_gt=tuple(sorted(gt_ids)),
        )
        for _, (crop, source_ids, gt_ids) in sorted(aggregated.items())
    ]
    return rows, tuple(sorted(unfit))

def _coverage_rows(
    batch_index,
    boxes,
    box_index,
    full_shape,
    crop_shape,
    spacing,
    context_um,
    min_cells,
    preferred_cells,
):
    if not boxes:
        return []

    ordered = [boxes[i] for i in sorted(boxes)]
    centers = np.stack([b.center * spacing for b in ordered])
    k = min(
        len(ordered),
        max(preferred_cells + 2, min_cells, 1),
    )
    tree = cKDTree(centers)
    _, nbr = tree.query(centers, k=k)
    if k == 1:
        nbr = np.asarray(nbr)[:, None]

    unique_crops = {}
    for anchor in range(len(ordered)):
        order = [
            int(v)
            for v in np.asarray(nbr[anchor]).reshape(-1).tolist()
        ]
        for n in range(1, len(order) + 1):
            low, high = _union(ordered[j] for j in order[:n])
            crop = _fit(
                low,
                high,
                full_shape,
                crop_shape,
                spacing,
                context_um,
            )
            if crop is None:
                continue
            key = tuple((int(s.start), int(s.stop)) for s in crop)
            unique_crops.setdefault(key, crop)

    return [
        _record(
            batch_index,
            crop,
            boxes,
            box_index=box_index,
        )
        for _, crop in sorted(unique_crops.items())
    ]

def _key(row):
    return tuple((int(s.start), int(s.stop)) for s in row.slices_zyx)


def _dedupe(rows):
    out = {}
    for row in rows:
        key = _key(row)
        if key not in out: out[key] = row; continue
        old = out[key]
        ms = tuple(sorted(set(old.merge_source_ids) | set(row.merge_source_ids)))
        mg = tuple(sorted(set(old.merge_gt_ids) | set(row.merge_gt_ids)))
        out[key] = MergeAwareCropRecord(old.batch_index, old.slices_zyx, old.complete_cell_ids,
            old.partial_cell_ids, old.true_boundary_cell_ids, ms, mg, 'merge' if ms else old.candidate_type)
    return [out[k] for k in sorted(out)]


def _select(
    candidates,
    target_ids,
    boundary_ids,
    min_cells,
    preferred_cells,
    views,
):
    remaining = {i: views for i in target_ids}
    selected = []
    used = set()

    covered = {
        _key(row): frozenset(row.covered_cell_ids)
        for row in candidates
    }

    merge = sorted(
        (r for r in candidates if r.merge_source_ids),
        key=lambda r: (
            -len(r.merge_gt_ids),
            -len(r.complete_cell_ids),
            len(r.partial_cell_ids),
            _key(r),
        ),
    )
    for row in merge:
        key = _key(row)
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        for i in row.covered_cell_ids:
            if remaining.get(i, 0) > 0:
                remaining[i] -= 1

    pool = [r for r in candidates if _key(r) not in used]
    interior_count = len(target_ids - boundary_ids)

    while any(v > 0 for v in remaining.values()):
        needed = frozenset(i for i, v in remaining.items() if v > 0)

        gains = {
            _key(row): len(covered[_key(row)] & needed)
            for row in pool
        }
        useful = [row for row in pool if gains[_key(row)] > 0]
        if not useful:
            break

        preferred = [
            row
            for row in useful
            if len(row.complete_cell_ids) >= preferred_cells
        ]
        minimum = [
            row
            for row in useful
            if len(row.complete_cell_ids) >= min_cells
        ]
        boundary = [
            row
            for row in useful
            if covered[_key(row)] & boundary_ids & needed
        ]
        eligible = (
            preferred
            or minimum
            or boundary
            or (useful if interior_count < min_cells else [])
        )
        if not eligible:
            break

        best = max(
            eligible,
            key=lambda row: (
                gains[_key(row)],
                len(row.complete_cell_ids),
                -len(row.partial_cell_ids),
                tuple(-int(s.start) for s in row.slices_zyx),
            ),
        )
        selected.append(best)
        pool.remove(best)

        for i in best.covered_cell_ids:
            if remaining.get(i, 0) > 0:
                remaining[i] -= 1

    return selected, tuple(
        sorted(i for i, v in remaining.items() if v > 0)
    )

def build_merge_aware_crop_manifest(gt_labels: Tensor, *, current_labels: Tensor | None = None, spacing_um: Tensor | None = None,
        crop_shape_zyx=(32, 192, 192), min_complete_cells=3, preferred_complete_cells=4, views_per_cell=1,
        context_um=4.0, merge_min_overlap_voxels=8, merge_min_gt_fraction=0.05):
    gt = torch.as_tensor(gt_labels).detach().cpu().long()
    if gt.ndim != 4: raise ValueError('GT must be [B,Z,Y,X]')
    current = None if current_labels is None else torch.as_tensor(current_labels).detach().cpu().long()
    if current is not None and current.shape != gt.shape: raise ValueError('current_labels must align with GT')
    spacing = torch.ones((gt.shape[0], 3)) if spacing_um is None else torch.as_tensor(spacing_um).detach().cpu().float()
    if spacing.ndim == 1: spacing = spacing[None].expand(gt.shape[0], -1)
    if spacing.shape != (gt.shape[0], 3) or bool((spacing <= 0).any()): raise ValueError('spacing_um must be positive [B,3]')
    if min_complete_cells < 1 or preferred_complete_cells < min_complete_cells: raise ValueError('invalid complete-cell thresholds')
    full_shape = tuple(int(v) for v in gt.shape[-3:])
    records, uncells, unmerges = [], [], []
    for b in range(gt.shape[0]):
        g = gt[b].numpy(); c = None if current is None else current[b].numpy(); boxes = _boxes(g); sp = spacing[b].numpy().astype(np.float64)
        box_index = _make_box_index(boxes)
        merge_rows, unfit = _merge_rows(b, g, c, boxes, box_index, full_shape, crop_shape_zyx, sp, context_um, merge_min_overlap_voxels, merge_min_gt_fraction)
        cover_rows = _coverage_rows(b, boxes, box_index, full_shape, crop_shape_zyx, sp, context_um, min_complete_cells, preferred_complete_cells)
        candidates = _dedupe([*merge_rows, *cover_rows])
        boundary_ids = {i for i, box in boxes.items() if box.boundary}
        chosen, missing = _select(candidates, set(boxes), boundary_ids, min_complete_cells, preferred_complete_cells, views_per_cell)
        if not chosen:
            size = np.minimum(np.asarray(full_shape), np.asarray(crop_shape_zyx)); start = (np.asarray(full_shape) - size) // 2
            crop = tuple(slice(int(a), int(bb)) for a, bb in zip(start, start + size))
            chosen = [_record(b, crop, boxes, box_index=box_index, kind='background')]
        records.append(tuple(chosen)); uncells.append(missing); unmerges.append(unfit)
    return MergeAwareCropManifest(tuple(records), tuple(crop_shape_zyx), min_complete_cells, preferred_complete_cells,
        views_per_cell, float(context_um), merge_min_overlap_voxels, float(merge_min_gt_fraction), tuple(uncells), tuple(unmerges),
        source_signature(gt_labels, current_labels, spacing_um))


def _crop_spec_from_record(row: MergeAwareCropRecord, *, batch_index: int,
        full_shape: tuple[int, int, int], spacing: Tensor) -> CropSpec:
    lower = torch.tensor([s.start for s in row.slices_zyx], dtype=torch.float32)
    size = torch.tensor([s.stop - s.start for s in row.slices_zyx], dtype=torch.float32)
    shift = (lower + 0.5 * (size - 1) - 0.5 * (torch.tensor(full_shape).float() - 1)) * spacing
    return CropSpec(batch_index, row.slices_zyx, full_shape, shift, row.candidate_type,
        row.complete_cell_ids, row.partial_cell_ids, row.true_boundary_cell_ids, row.merge_source_ids)


def _take_unique_cyclic(rows, *, count: int, start: int, used: set[tuple]):
    chosen = []
    if count <= 0 or not rows: return chosen
    for offset in range(len(rows)):
        row = rows[(start + offset) % len(rows)]; key = _key(row)
        if key in used: continue
        chosen.append(row); used.add(key)
        if len(chosen) >= count: break
    return chosen


def _balanced_batch_rows(rows, *, crop_batch_size: int, merge_fraction: float, iteration: int):
    if crop_batch_size < 1: raise ValueError('crop_batch_size must be positive')
    if not 0.0 <= merge_fraction <= 1.0: raise ValueError('merge_fraction must be in [0,1]')
    if not rows: return []
    # Exact historical cycling for B=1.
    if crop_batch_size == 1: return [rows[iteration % len(rows)]]
    merge_rows = [r for r in rows if r.merge_source_ids]
    coverage_rows = [r for r in rows if not r.merge_source_ids]
    merge_target = min(crop_batch_size, max(0, int(crop_batch_size * merge_fraction + 0.5)))
    coverage_target = crop_batch_size - merge_target
    used = set(); chosen = []
    chosen += _take_unique_cyclic(merge_rows, count=merge_target,
        start=(iteration * max(merge_target, 1)) % max(len(merge_rows), 1), used=used)
    chosen += _take_unique_cyclic(coverage_rows, count=coverage_target,
        start=(iteration * max(coverage_target, 1)) % max(len(coverage_rows), 1), used=used)
    all_rows = list(rows); fill_start = (iteration * crop_batch_size) % len(all_rows)
    chosen += _take_unique_cyclic(all_rows, count=crop_batch_size-len(chosen),
        start=fill_start, used=used)
    # Duplicate only if the whole manifest has fewer unique rows than requested B.
    offset = 0
    while len(chosen) < crop_batch_size:
        chosen.append(all_rows[(fill_start + offset) % len(all_rows)]); offset += 1
    return chosen


def sample_merge_aware_crop_specs(gt_labels: Tensor, spacing_um: Tensor,
        manifest: MergeAwareCropManifest, *, crops_per_step: int, global_step: int,
        crop_batch_size: int = 1, merge_fraction: float = 0.5):
    """Return deterministic true crop-batch rounds with merge/coverage balance."""
    spacing = torch.as_tensor(spacing_um).detach().cpu().float(); labels = torch.as_tensor(gt_labels)
    if spacing.ndim == 1: spacing = spacing[None]
    full_shape = tuple(int(v) for v in labels.shape[-3:]); rounds = []
    for crop_round in range(crops_per_step):
        specs = []; iteration = global_step * crops_per_step + crop_round
        for b, rows in enumerate(manifest.records):
            for row in _balanced_batch_rows(rows, crop_batch_size=crop_batch_size,
                    merge_fraction=merge_fraction, iteration=iteration):
                specs.append(_crop_spec_from_record(row, batch_index=b,
                    full_shape=full_shape, spacing=spacing[b]))
        rounds.append(specs)
    return rounds




__all__ = ['MergeAwareCropManifest', 'MergeAwareCropRecord', 'build_merge_aware_crop_manifest', 'sample_merge_aware_crop_specs', 'source_signature']
