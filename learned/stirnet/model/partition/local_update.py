from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List

import torch
from torch import Tensor, nn

from ..config import RefinementConfig
from ..types import (
    GeometryLike,
    GeometryState,
    RefinedGeometryView,
    geometry_field_crop,
)
from .watershed import LearnedGeometryWatershed


@dataclass(frozen=True)
class LocalPartitionUpdateResult:
    supervoxel_labels: List[Tensor]
    used_fallback: bool
    fallback_reason: str = ""
    fallback_reason_code: int = 0
    fallback_batch_index: int = -1
    fallback_box_index: int = -1
    fallback_box_shape_zyx: tuple[int, int, int] | None = None
    fallback_box_voxel_count: int = 0
    fallback_core_voxel_count: int = 0
    fallback_local_component_count: int = 0
    fallback_old_core_label_count: int = 0
    fallback_old_shell_label_count: int = 0
    fallback_conflicting_old_label_ids: List[int] | None = None
    updated_box_count: int = 0
    updated_boxes: List[tuple[int, tuple[slice, slice, slice]]] | None = None
    updated_voxel_count: int = 0


_FALLBACK_REASON_CODES = {
    "": 0,
    "no sparse delta": 1,
    "local component touches multiple external labels": 2,
    "unowned component touches halo boundary": 3,
    "local component would merge existing labels": 4,
    "split components reconnect through an external label": 5,
}


def _positive_ids(value: Tensor) -> list[int]:
    ids = torch.unique(value)
    ids = ids[ids > 0]
    return [int(item) for item in ids.detach().cpu().tolist()]


def _fallback_context(
    global_labels: Tensor,
    local_labels: Tensor,
    box: tuple[slice, slice, slice],
    core_mask: Tensor,
    reason: str,
) -> dict[str, object]:
    old_crop = global_labels[box]
    shell = ~core_mask
    local_ids = _positive_ids(local_labels[core_mask])
    old_core_ids = _positive_ids(old_crop[core_mask])
    old_shell_ids = _positive_ids(old_crop[shell])

    conflicting_old_ids: list[int] = []
    if reason == "local component touches multiple external labels":
        for local_id in local_ids:
            component = local_labels == local_id
            owners = _positive_ids(old_crop[component & shell])
            if len(owners) > 1:
                conflicting_old_ids = owners
                break
    elif reason == "local component would merge existing labels":
        for local_id in local_ids:
            component = local_labels == local_id
            owners = _positive_ids(old_crop[component & core_mask])
            if len(owners) > 1:
                conflicting_old_ids = owners
                break
    elif reason == "split components reconnect through an external label":
        owner_to_local: dict[int, list[int]] = {}
        for local_id in local_ids:
            component = local_labels == local_id
            owners = _positive_ids(old_crop[component & shell])
            if len(owners) == 1:
                owner_to_local.setdefault(owners[0], []).append(local_id)
        for owner, members in owner_to_local.items():
            if len(members) > 1:
                conflicting_old_ids = [owner]
                break

    box_shape = tuple(int(axis.stop) - int(axis.start) for axis in box)
    return {
        "box_shape_zyx": box_shape,
        "box_voxel_count": int(box_shape[0] * box_shape[1] * box_shape[2]),
        "core_voxel_count": int(core_mask.sum().item()),
        "local_component_count": len(local_ids),
        "old_core_label_count": len(old_core_ids),
        "old_shell_label_count": len(old_shell_ids),
        "conflicting_old_label_ids": conflicting_old_ids,
    }


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
    def __init__(
        self,
        watershed: LearnedGeometryWatershed,
        cfg: RefinementConfig,
    ):
        super().__init__()
        self.watershed = watershed
        self.cfg = cfg

    def _fallback(
        self,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None,
        reason: str,
        *,
        batch_index: int = -1,
        box_index: int = -1,
        context: dict[str, object] | None = None,
        updated_box_count: int = 0,
        updated_voxel_count: int = 0,
    ) -> LocalPartitionUpdateResult:
        context = {} if context is None else context
        reason_code = _FALLBACK_REASON_CODES.get(reason, 99)
        event = {
            "event": "local_partition_fallback",
            "reason": reason,
            "reason_code": reason_code,
            "batch_index": int(batch_index),
            "box_index": int(box_index),
            "box_shape_zyx": context.get("box_shape_zyx"),
            "box_voxel_count": int(context.get("box_voxel_count", 0)),
            "core_voxel_count": int(context.get("core_voxel_count", 0)),
            "local_component_count": int(context.get("local_component_count", 0)),
            "old_core_label_count": int(context.get("old_core_label_count", 0)),
            "old_shell_label_count": int(context.get("old_shell_label_count", 0)),
            "conflicting_old_label_ids": list(
                context.get("conflicting_old_label_ids", [])
            ),
        }
        print(
            "[stirnet-local-partition-fallback] "
            + json.dumps(event, sort_keys=True),
            flush=True,
        )
        return LocalPartitionUpdateResult(
            supervoxel_labels=self.watershed(
                geometry, spacing_um, dref_um, padding_mask
            ),
            used_fallback=True,
            fallback_reason=reason,
            fallback_reason_code=reason_code,
            fallback_batch_index=int(batch_index),
            fallback_box_index=int(box_index),
            fallback_box_shape_zyx=context.get("box_shape_zyx"),
            fallback_box_voxel_count=int(context.get("box_voxel_count", 0)),
            fallback_core_voxel_count=int(context.get("core_voxel_count", 0)),
            fallback_local_component_count=int(
                context.get("local_component_count", 0)
            ),
            fallback_old_core_label_count=int(
                context.get("old_core_label_count", 0)
            ),
            fallback_old_shell_label_count=int(
                context.get("old_shell_label_count", 0)
            ),
            fallback_conflicting_old_label_ids=list(
                context.get("conflicting_old_label_ids", [])
            ),
            updated_box_count=int(updated_box_count),
            updated_voxel_count=int(updated_voxel_count),
        )

    @torch.no_grad()
    def forward(
        self,
        initial_labels: List[Tensor],
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None = None,
    ) -> LocalPartitionUpdateResult:
        if not isinstance(geometry, RefinedGeometryView) or geometry.delta.is_empty:
            return self._fallback(
                geometry, spacing_um, dref_um, padding_mask, "no sparse delta"
            )
        output = [labels.clone() for labels in initial_labels]
        total_boxes = 0
        updated_voxels = 0
        updated_boxes: list[tuple[int, tuple[slice, slice, slice]]] = []
        for batch_index, base in enumerate(initial_labels):
            rois = [
                roi for roi in geometry.delta.rois if roi.batch_index == batch_index
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
                            max(0, int(roi.slices_zyx[axis].start) - int(halo_voxels[axis])),
                            min(
                                base.shape[axis],
                                int(roi.slices_zyx[axis].stop) + int(halo_voxels[axis]),
                            ),
                        )
                        for axis in range(3)
                    )
                )
            boxes = _merge_boxes(boxes)
            total_boxes += len(boxes)
            updated_boxes.extend((batch_index, box) for box in boxes)
            updated = output[batch_index]
            next_label_id = int(base.max().item()) + 1
            for box_index, box in enumerate(boxes):
                shape = tuple(int(axis.stop) - int(axis.start) for axis in box)
                core_mask = torch.zeros(shape, device=base.device, dtype=torch.bool)
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
                        slice(start[a] - int(box[a].start), stop[a] - int(box[a].start))
                        for a in range(3)
                    )
                    core_mask[local] = True
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
                    context = _fallback_context(
                        updated,
                        local_labels,
                        box,
                        core_mask,
                        reason,
                    )
                    return self._fallback(
                        geometry,
                        spacing_um,
                        dref_um,
                        padding_mask,
                        reason,
                        batch_index=batch_index,
                        box_index=box_index,
                        context=context,
                        updated_box_count=total_boxes,
                        updated_voxel_count=updated_voxels,
                    )
                updated = reconciled
                updated_voxels += int(core_mask.sum().item())
                next_label_id = max(
                    next_label_id, int(updated[box].max().item()) + 1
                )
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
