from __future__ import annotations

"""Deterministic coverage-driven 3-D crop planning."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import Tensor

from .crops import CropSpec


@dataclass(frozen=True)
class CoverageCropRecord:
    batch_index: int
    slices_zyx: tuple[slice, slice, slice]
    complete_cell_ids: tuple[int, ...]
    partial_cell_ids: tuple[int, ...]
    true_boundary_cell_ids: tuple[int, ...]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(s.stop) - int(s.start) for s in self.slices_zyx)


@dataclass(frozen=True)
class CoverageCropManifest:
    records: tuple[tuple[CoverageCropRecord, ...], ...]
    crop_shape_zyx: tuple[int, int, int]
    min_complete_cells: int
    views_per_cell: int
    uncoverable_cell_ids: tuple[tuple[int, ...], ...]
    source_signature: tuple


@dataclass(frozen=True)
class _CellBox:
    cell_id: int
    low: tuple[int, int, int]
    high: tuple[int, int, int]
    true_boundary: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(hi - lo for lo, hi in zip(self.low, self.high))


def _boxes(labels: np.ndarray) -> list[_CellBox]:
    shape = labels.shape
    rows: list[_CellBox] = []
    for cell_id, bbox in enumerate(ndi.find_objects(labels), 1):
        if bbox is None:
            continue
        low = tuple(int(s.start) for s in bbox)
        high = tuple(int(s.stop) for s in bbox)
        boundary = any(
            low[a] == 0 or high[a] == shape[a] for a in range(3)
        )
        rows.append(_CellBox(cell_id, low, high, boundary))
    return rows


def _crop_for_box(box, full_shape, crop_shape):
    if any(e > c for e, c in zip(box.shape, crop_shape)):
        return None
    starts = []
    for a in range(3):
        size = min(int(crop_shape[a]), int(full_shape[a]))
        center = 0.5 * (box.low[a] + box.high[a] - 1)
        start = int(round(center - 0.5 * (size - 1)))
        starts.append(min(max(start, 0), full_shape[a] - size))
    return tuple(
        slice(starts[a], starts[a] + min(crop_shape[a], full_shape[a]))
        for a in range(3)
    )


def _intersects(box, crop) -> bool:
    return all(
        box.high[a] > int(crop[a].start)
        and box.low[a] < int(crop[a].stop)
        for a in range(3)
    )


def _inside(box, crop) -> bool:
    return all(
        box.low[a] >= int(crop[a].start)
        and box.high[a] <= int(crop[a].stop)
        for a in range(3)
    )


def _record(batch_index, crop, boxes):
    complete, partial, true_boundary = [], [], []
    for box in boxes:
        if not _intersects(box, crop):
            continue
        if _inside(box, crop):
            (true_boundary if box.true_boundary else complete).append(box.cell_id)
        else:
            partial.append(box.cell_id)
    return CoverageCropRecord(
        batch_index,
        crop,
        tuple(sorted(complete)),
        tuple(sorted(partial)),
        tuple(sorted(true_boundary)),
    )


def _dedupe(rows: Iterable[CoverageCropRecord]):
    result, seen = [], set()
    for row in rows:
        key = tuple((int(s.start), int(s.stop)) for s in row.slices_zyx)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _greedy(candidates, target_ids, min_complete_cells, views_per_cell):
    remaining = {cell_id: int(views_per_cell) for cell_id in target_ids}
    selected, unused = [], list(candidates)
    while any(v > 0 for v in remaining.values()):
        eligible = [
            row for row in unused
            if any(remaining.get(i, 0) > 0 for i in row.complete_cell_ids)
        ]
        if not eligible:
            eligible = [
                row for row in candidates
                if any(remaining.get(i, 0) > 0 for i in row.complete_cell_ids)
            ]
        if not eligible:
            break
        preferred = [
            row for row in eligible
            if len(row.complete_cell_ids) >= min_complete_cells
        ]
        pool = preferred or eligible
        best = max(
            pool,
            key=lambda row: (
                sum(remaining.get(i, 0) > 0 for i in row.complete_cell_ids),
                len(row.complete_cell_ids),
                -len(row.partial_cell_ids),
            ),
        )
        selected.append(best)
        if best in unused:
            unused.remove(best)
        for i in best.complete_cell_ids:
            if remaining.get(i, 0) > 0:
                remaining[i] -= 1
    return selected


def build_coverage_crop_manifest(
    gt_labels: Tensor,
    *,
    crop_shape_zyx: tuple[int, int, int],
    min_complete_cells: int = 10,
    views_per_cell: int = 1,
) -> CoverageCropManifest:
    labels = torch.as_tensor(gt_labels).detach().cpu().long()
    if labels.ndim == 3:
        labels = labels[None]
    if labels.ndim != 4:
        raise ValueError("gt_labels must be [Z,Y,X] or [B,Z,Y,X]")
    if any(v < 1 for v in crop_shape_zyx):
        raise ValueError("crop dimensions must be positive")
    if min_complete_cells < 1 or views_per_cell < 1:
        raise ValueError("coverage counts must be positive")

    batches, uncoverable_batches = [], []
    for b in range(labels.shape[0]):
        volume = labels[b].numpy()
        full_shape = tuple(int(v) for v in volume.shape)
        boxes = _boxes(volume)
        candidates, target_ids, uncoverable = [], set(), set()
        for box in boxes:
            crop = _crop_for_box(box, full_shape, crop_shape_zyx)
            if crop is None:
                if not box.true_boundary:
                    uncoverable.add(box.cell_id)
                continue
            candidates.append(_record(b, crop, boxes))
            if not box.true_boundary:
                target_ids.add(box.cell_id)
        candidates = _dedupe(candidates)
        if not candidates:
            crop = tuple(
                slice(
                    max((full_shape[a] - min(full_shape[a], crop_shape_zyx[a])) // 2, 0),
                    max((full_shape[a] - min(full_shape[a], crop_shape_zyx[a])) // 2, 0)
                    + min(full_shape[a], crop_shape_zyx[a]),
                )
                for a in range(3)
            )
            candidates = [_record(b, crop, boxes)]
        selected = _greedy(
            candidates, target_ids, min_complete_cells, views_per_cell
        ) or candidates[:1]
        covered = {i for row in selected for i in row.complete_cell_ids}
        uncoverable.update(target_ids - covered)
        batches.append(tuple(selected))
        uncoverable_batches.append(tuple(sorted(uncoverable)))

    return CoverageCropManifest(
        records=tuple(batches),
        crop_shape_zyx=tuple(int(v) for v in crop_shape_zyx),
        min_complete_cells=int(min_complete_cells),
        views_per_cell=int(views_per_cell),
        uncoverable_cell_ids=tuple(uncoverable_batches),
        source_signature=(tuple(labels.shape), int(labels.data_ptr())),
    )


def sample_coverage_crop_specs(
    gt_labels: Tensor,
    spacing_um: Tensor,
    manifest: CoverageCropManifest,
    *,
    crops_per_step: int,
    global_step: int,
) -> list[list[CropSpec]]:
    labels = torch.as_tensor(gt_labels)
    if labels.ndim == 3:
        labels = labels[None]
    if tuple(labels.shape) != manifest.source_signature[0]:
        raise ValueError("coverage manifest does not match GT shape")
    spacing = torch.as_tensor(spacing_um).detach().cpu().float()
    if spacing.ndim == 1:
        spacing = spacing[None]
    full_shape = tuple(int(v) for v in labels.shape[-3:])
    rounds = []
    for crop_round in range(crops_per_step):
        ordinal = global_step * crops_per_step + crop_round
        specs = []
        for b in range(labels.shape[0]):
            rows = manifest.records[b]
            row = rows[ordinal % len(rows)]
            lower = torch.tensor([int(s.start) for s in row.slices_zyx]).float()
            size = torch.tensor(row.shape_zyx).float()
            full_center = 0.5 * (torch.tensor(full_shape).float() - 1)
            crop_center = lower + 0.5 * (size - 1)
            specs.append(
                CropSpec(
                    batch_index=b,
                    slices_zyx=row.slices_zyx,
                    full_shape_zyx=full_shape,
                    center_shift_um=(crop_center - full_center) * spacing[b],
                    candidate_type="coverage",
                )
            )
        rounds.append(specs)
    return rounds


__all__ = [
    "CoverageCropManifest",
    "CoverageCropRecord",
    "build_coverage_crop_manifest",
    "sample_coverage_crop_specs",
]
