from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor, nn

from ..config import RefinementConfig
from ..types import (
    GeometryLike,
    GeometryState,
    RefinedGeometryView,
    SupervoxelStatistics,
    geometry_field_crop,
)
from .watershed import LearnedGeometryWatershed


@dataclass(frozen=True)
class LocalPartitionUpdateResult:
    supervoxel_labels: List[Tensor]
    used_fallback: bool
    fallback_reason: str = ""
    updated_box_count: int = 0
    updated_boxes: List[tuple[int, tuple[slice, slice, slice]]] | None = None
    updated_voxel_count: int = 0


def _boxes_touch(
    left: tuple[slice, slice, slice], right: tuple[slice, slice, slice]
) -> bool:
    return all(
        int(left[axis].start) <= int(right[axis].stop)
        and int(right[axis].start) <= int(left[axis].stop)
        for axis in range(3)
    )


def _merge_boxes(
    boxes: list[tuple[slice, slice, slice]],
) -> list[tuple[slice, slice, slice]]:
    pending = list(boxes)
    merged: list[tuple[slice, slice, slice]] = []
    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            keep = []
            for other in pending:
                if _boxes_touch(current, other):
                    current = tuple(
                        slice(
                            min(int(current[a].start), int(other[a].start)),
                            max(int(current[a].stop), int(other[a].stop)),
                        )
                        for a in range(3)
                    )
                    changed = True
                else:
                    keep.append(other)
            pending = keep
        merged.append(current)
    return merged


def reconcile_local_labels(
    global_labels: Tensor,
    local_labels: Tensor,
    box: tuple[slice, slice, slice],
    core_mask: Tensor,
    *,
    copy_output: bool = True,
    next_label_id: int | None = None,
) -> tuple[Tensor | None, str]:
    """Conservatively reconcile one halo watershed into a global label field."""
    old_crop = global_labels[box]
    if old_crop.shape != local_labels.shape or core_mask.shape != local_labels.shape:
        raise ValueError("local labels/core mask must align with the update box")
    shell = ~core_mask
    local_ids = torch.unique(local_labels[core_mask])
    local_ids = local_ids[local_ids > 0]
    assignments: dict[int, int] = {}
    shell_owners: dict[int, set[int]] = {}
    next_id = (
        int(global_labels.max().item()) + 1
        if next_label_id is None
        else int(next_label_id)
    )

    for local_id_tensor in local_ids:
        local_id = int(local_id_tensor.item())
        component = local_labels == local_id
        boundary_ids = torch.unique(old_crop[component & shell])
        boundary_ids = boundary_ids[boundary_ids > 0]
        if boundary_ids.numel() > 1:
            return None, "local component touches multiple external labels"
        if boundary_ids.numel() == 1:
            owner = int(boundary_ids.item())
            assignments[local_id] = owner
            shell_owners.setdefault(owner, set()).add(local_id)
            continue

        # A component clipped by the halo without an owner is unsafe.
        touches_box = bool(
            (int(box[0].start) > 0 and component[0].any())
            or (int(box[0].stop) < global_labels.shape[0] and component[-1].any())
            or (int(box[1].start) > 0 and component[:, 0].any())
            or (int(box[1].stop) < global_labels.shape[1] and component[:, -1].any())
            or (int(box[2].start) > 0 and component[:, :, 0].any())
            or (int(box[2].stop) < global_labels.shape[2] and component[:, :, -1].any())
        )
        if touches_box:
            return None, "unowned component touches halo boundary"
        overlap_ids = torch.unique(old_crop[component & core_mask])
        overlap_ids = overlap_ids[overlap_ids > 0]
        if overlap_ids.numel() > 1:
            return None, "local component would merge existing labels"
        assignments[local_id] = (
            int(overlap_ids.item()) if overlap_ids.numel() == 1 else next_id
        )
        if overlap_ids.numel() == 0:
            next_id += 1

    # Two proposed components cannot both reconnect to the same untouched
    # external component; they would still be connected outside the ROI.
    if any(len(local_set) > 1 for local_set in shell_owners.values()):
        return None, "split components reconnect through an external label"

    # If several interior components overlap one old ID, retain it for the
    # largest and allocate fresh IDs to the remaining genuine split pieces.
    by_assignment: dict[int, list[int]] = {}
    for local_id, old_id in assignments.items():
        by_assignment.setdefault(old_id, []).append(local_id)
    for old_id, proposed_ids in by_assignment.items():
        if len(proposed_ids) <= 1 or old_id in shell_owners:
            continue
        sizes = {
            local_id: int(((local_labels == local_id) & core_mask).sum().item())
            for local_id in proposed_ids
        }
        keep = max(proposed_ids, key=lambda value: (sizes[value], -value))
        for local_id in proposed_ids:
            if local_id != keep:
                assignments[local_id] = next_id
                next_id += 1

    mapped = torch.zeros_like(local_labels)
    for local_id, global_id in assignments.items():
        mapped[local_labels == local_id] = global_id
    updated = global_labels.clone() if copy_output else global_labels
    target = updated[box]
    target[core_mask] = mapped[core_mask]
    updated[box] = target
    return updated, ""


class LocalPartitionUpdater(nn.Module):
    # Conservative local refinement with bounded regional retries before
    # the trusted full-frame watershed fallback.
    _COMPONENT_RETRY_MAX_VOLUME_FRACTION = 0.50
    _COMPONENT_CLOSURE_STEPS = 2

    def __init__(
        self,
        watershed: LearnedGeometryWatershed,
        cfg: RefinementConfig,
    ):
        super().__init__()
        self.watershed = watershed
        self.cfg = cfg

    @staticmethod
    def _expand_box(
        box: tuple[slice, slice, slice],
        halo_voxels: Tensor,
        shape: tuple[int, int, int],
    ) -> tuple[slice, slice, slice]:
        return tuple(
            slice(
                max(0, int(box[axis].start) - int(halo_voxels[axis])),
                min(shape[axis], int(box[axis].stop) + int(halo_voxels[axis])),
            )
            for axis in range(3)
        )

    @staticmethod
    def _box_volume(box: tuple[slice, slice, slice]) -> int:
        return (
            (int(box[0].stop) - int(box[0].start))
            * (int(box[1].stop) - int(box[1].start))
            * (int(box[2].stop) - int(box[2].start))
        )

    @staticmethod
    def _same_box(
        left: tuple[slice, slice, slice],
        right: tuple[slice, slice, slice],
    ) -> bool:
        return all(
            int(left[a].start) == int(right[a].start)
            and int(left[a].stop) == int(right[a].stop)
            for a in range(3)
        )

    @staticmethod
    def _core_mask(
        box: tuple[slice, slice, slice],
        rois,
        device: torch.device,
    ) -> Tensor:
        shape = tuple(int(axis.stop) - int(axis.start) for axis in box)
        core_mask = torch.zeros(shape, device=device, dtype=torch.bool)
        for roi in rois:
            start = [
                max(int(roi.slices_zyx[a].start), int(box[a].start))
                for a in range(3)
            ]
            stop = [
                min(int(roi.slices_zyx[a].stop), int(box[a].stop))
                for a in range(3)
            ]
            if any(b <= a for a, b in zip(start, stop)):
                continue
            local = tuple(
                slice(
                    start[a] - int(box[a].start),
                    stop[a] - int(box[a].start),
                )
                for a in range(3)
            )
            core_mask[local] = True
        return core_mask

    @staticmethod
    def _component_retry_box(
        base: Tensor,
        seed_box: tuple[slice, slice, slice],
        statistics: SupervoxelStatistics | None,
        halo_voxels: Tensor,
    ) -> tuple[slice, slice, slice] | None:
        if statistics is None or statistics.counts.numel() == 0:
            return None

        full_shape = tuple(int(v) for v in base.shape)
        candidate = seed_box
        max_rows = int(statistics.counts.shape[0])

        for _ in range(LocalPartitionUpdater._COMPONENT_CLOSURE_STEPS):
            touched = torch.unique(base[candidate]).long()
            touched = touched[(touched > 0) & (touched <= max_rows)]
            if touched.numel() == 0:
                return None

            rows = touched - 1
            rows = rows[statistics.counts[rows] > 0]
            if rows.numel() == 0:
                return None

            lower = statistics.min_voxel[rows].amin(dim=0)
            upper = statistics.max_voxel[rows].amax(dim=0) + 1

            object_box = tuple(
                slice(
                    min(int(candidate[a].start), int(lower[a].item())),
                    max(int(candidate[a].stop), int(upper[a].item())),
                )
                for a in range(3)
            )
            expanded = LocalPartitionUpdater._expand_box(
                object_box,
                halo_voxels,
                full_shape,
            )
            if LocalPartitionUpdater._same_box(candidate, expanded):
                break
            candidate = expanded

        return candidate

    def _fallback(
        self,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None,
        reason: str,
    ) -> LocalPartitionUpdateResult:
        return LocalPartitionUpdateResult(
            supervoxel_labels=self.watershed(
                geometry, spacing_um, dref_um, padding_mask
            ),
            used_fallback=True,
            fallback_reason=reason,
        )

    def _attempt_box(
        self,
        updated: Tensor,
        geometry: RefinedGeometryView,
        batch_index: int,
        box: tuple[slice, slice, slice],
        rois,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None,
        next_label_id: int,
    ) -> tuple[Tensor | None, str, int, int]:
        core_mask = self._core_mask(box, rois, updated.device)
        if not bool(core_mask.any()):
            return None, "regional retry contains no refinement core", 0, next_label_id

        crop = GeometryState(
            foreground_logits=geometry_field_crop(
                geometry, "foreground_logits", batch_index, box
            )[None],
            surface_logits=geometry_field_crop(
                geometry, "surface_logits", batch_index, box
            )[None],
            separator_logits=geometry_field_crop(
                geometry, "separator_logits", batch_index, box
            )[None],
            sdf=geometry_field_crop(geometry, "sdf", batch_index, box)[None],
            flow=geometry_field_crop(geometry, "flow", batch_index, box)[None],
            centroid_offset=geometry_field_crop(
                geometry, "centroid_offset", batch_index, box
            )[None],
            seed_logits=geometry_field_crop(
                geometry, "seed_logits", batch_index, box
            )[None],
            features=None,
        )
        local_padding = (
            None
            if padding_mask is None
            else padding_mask[batch_index : batch_index + 1][
                (slice(None), *box)
            ]
        )
        local_labels = self.watershed(
            crop,
            spacing_um[batch_index : batch_index + 1],
            dref_um[batch_index : batch_index + 1],
            local_padding,
        )[0]

        reconciled, reason = reconcile_local_labels(
            updated,
            local_labels,
            box,
            core_mask,
            copy_output=False,
            next_label_id=next_label_id,
        )
        if reconciled is None:
            return None, reason, 0, next_label_id

        changed_voxels = int(core_mask.sum().item())
        next_label_id = max(
            next_label_id,
            int(reconciled[box].max().item()) + 1,
        )
        return reconciled, "", changed_voxels, next_label_id

    @torch.no_grad()
    def forward(
        self,
        initial_labels: List[Tensor],
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None = None,
        *,
        statistics: Sequence[SupervoxelStatistics] | None = None,
    ) -> LocalPartitionUpdateResult:
        if not isinstance(geometry, RefinedGeometryView) or geometry.delta.is_empty:
            return self._fallback(
                geometry,
                spacing_um,
                dref_um,
                padding_mask,
                "no sparse delta",
            )

        output = [labels.clone() for labels in initial_labels]
        total_boxes = 0
        updated_voxels = 0
        updated_boxes: list[
            tuple[int, tuple[slice, slice, slice]]
        ] = []

        for batch_index, base in enumerate(initial_labels):
            rois = [
                roi
                for roi in geometry.delta.rois
                if roi.batch_index == batch_index
            ]
            if not rois:
                continue

            halo_voxels = torch.ceil(
                self.cfg.partition_halo_dref
                * dref_um[batch_index].float()
                / spacing_um[batch_index].float().clamp_min(1e-6)
            ).long()

            boxes = []
            for roi in rois:
                boxes.append(
                    tuple(
                        slice(
                            max(
                                0,
                                int(roi.slices_zyx[axis].start)
                                - int(halo_voxels[axis]),
                            ),
                            min(
                                base.shape[axis],
                                int(roi.slices_zyx[axis].stop)
                                + int(halo_voxels[axis]),
                            ),
                        )
                        for axis in range(3)
                    )
                )
            boxes = _merge_boxes(boxes)
            total_boxes += len(boxes)

            updated = output[batch_index]
            next_label_id = int(base.max().item()) + 1
            batch_stats = (
                statistics[batch_index]
                if statistics is not None
                and batch_index < len(statistics)
                else None
            )
            full_voxels = max(base.numel(), 1)

            for base_box in boxes:
                attempts = [base_box]

                expanded_box = self._expand_box(
                    base_box,
                    halo_voxels,
                    tuple(int(v) for v in base.shape),
                )
                if not self._same_box(base_box, expanded_box):
                    attempts.append(expanded_box)

                last_reason = ""
                succeeded = False
                changed_voxels = 0

                for attempt_box in attempts:
                    (
                        reconciled,
                        reason,
                        attempt_changed,
                        next_after,
                    ) = self._attempt_box(
                        updated,
                        geometry,
                        batch_index,
                        attempt_box,
                        rois,
                        spacing_um,
                        dref_um,
                        padding_mask,
                        next_label_id,
                    )
                    if reconciled is not None:
                        updated = reconciled
                        next_label_id = next_after
                        changed_voxels = attempt_changed
                        succeeded = True
                        break
                    last_reason = reason

                if not succeeded:
                    seed_box = attempts[-1]
                    component_box = self._component_retry_box(
                        base,
                        seed_box,
                        batch_stats,
                        halo_voxels,
                    )
                    if (
                        component_box is not None
                        and not self._same_box(component_box, seed_box)
                        and (
                            self._box_volume(component_box)
                            / float(full_voxels)
                        )
                        <= self._COMPONENT_RETRY_MAX_VOLUME_FRACTION
                    ):
                        (
                            reconciled,
                            reason,
                            attempt_changed,
                            next_after,
                        ) = self._attempt_box(
                            updated,
                            geometry,
                            batch_index,
                            component_box,
                            rois,
                            spacing_um,
                            dref_um,
                            padding_mask,
                            next_label_id,
                        )
                        if reconciled is not None:
                            updated = reconciled
                            next_label_id = next_after
                            changed_voxels = attempt_changed
                            succeeded = True
                        else:
                            last_reason = reason

                if not succeeded:
                    return self._fallback(
                        geometry,
                        spacing_um,
                        dref_um,
                        padding_mask,
                        (
                            f"{last_reason}; regional retries exhausted"
                            if last_reason
                            else "regional retries exhausted"
                        ),
                    )

                updated_boxes.append((batch_index, base_box))
                updated_voxels += changed_voxels

            output[batch_index] = updated

        return LocalPartitionUpdateResult(
            supervoxel_labels=output,
            used_fallback=False,
            updated_box_count=total_boxes,
            updated_boxes=updated_boxes,
            updated_voxel_count=updated_voxels,
        )


__all__ = [
    "LocalPartitionUpdateResult",
    "LocalPartitionUpdater",
    "reconcile_local_labels",
]
