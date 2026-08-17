from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List

import torch
from skimage.measure import label as skimage_label
from torch import Tensor, nn

from ..config import RefinementConfig
from ..types import (
    GeometryLike,
    GeometryState,
    InstanceState,
    RAGState,
    RefinedGeometryView,
    RefinementRequest,
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
    "owned component touches protected external label": 6,
    "request/roi mismatch": 7,
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


@dataclass(frozen=True)
class _LocalPartitionTask:
    batch_index: int
    request_index: int
    request_kind: str
    source_index: int
    core_box: tuple[slice, slice, slice]
    dependency_box: tuple[slice, slice, slice]
    owned_label_ids: tuple[int, ...]


def _boxes_overlap(
    left: tuple[slice, slice, slice],
    right: tuple[slice, slice, slice],
) -> bool:
    """True only when two boxes share at least one voxel."""
    return all(
        int(left[axis].start) < int(right[axis].stop)
        and int(right[axis].start) < int(left[axis].stop)
        for axis in range(3)
    )


def _union_boxes(
    boxes: list[tuple[slice, slice, slice]],
) -> tuple[slice, slice, slice]:
    if not boxes:
        raise ValueError("cannot union an empty box list")
    return tuple(
        slice(
            min(int(box[axis].start) for box in boxes),
            max(int(box[axis].stop) for box in boxes),
        )
        for axis in range(3)
    )


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


def _owned_supervoxel_ids(
    request: RefinementRequest,
    rag: RAGState,
    instances: InstanceState,
) -> tuple[int, ...]:
    """Map request semantics to the exact old supervoxels it may change."""
    if request.kind == "recovery":
        return ()

    if request.kind == "edge":
        edge_row = int(request.source_index)
        if edge_row < 0 or edge_row >= rag.edge_index.shape[1]:
            return ()
        endpoints = rag.edge_index[:, edge_row].long()
        if bool((endpoints < 0).any()) or bool(
            (endpoints >= rag.node_supervoxel_id.numel()).any()
        ):
            return ()
        endpoint_batches = rag.node_batch[endpoints]
        if bool((endpoint_batches != int(request.batch_index)).any()):
            return ()
        labels = torch.unique(rag.node_supervoxel_id[endpoints].long())
        labels = labels[labels > 0]
        return tuple(
            sorted(int(value) for value in labels.detach().cpu().tolist())
        )

    if request.kind != "split":
        return ()

    node_to_instance = instances.node_to_instance
    if node_to_instance.numel() != rag.node_supervoxel_id.numel():
        return ()
    same_batch = rag.node_batch == int(request.batch_index)
    belongs = node_to_instance == int(request.source_index)
    node_rows = torch.nonzero(same_batch & belongs, as_tuple=False).flatten()
    if node_rows.numel() == 0:
        return ()
    labels = torch.unique(rag.node_supervoxel_id[node_rows].long())
    labels = labels[labels > 0]
    return tuple(
        sorted(int(value) for value in labels.detach().cpu().tolist())
    )


def _owned_bbox_from_statistics(
    rag: RAGState,
    batch_index: int,
    owned_label_ids: tuple[int, ...],
) -> tuple[slice, slice, slice] | None:
    """Return complete cached owned-supervoxel bounds without a volume scan."""
    if not owned_label_ids or rag.statistics is None:
        return None
    if batch_index < 0 or batch_index >= len(rag.statistics):
        return None
    stats = rag.statistics[batch_index]
    rows = torch.tensor(
        [label_id - 1 for label_id in owned_label_ids],
        device=stats.counts.device,
        dtype=torch.long,
    )
    rows = rows[(rows >= 0) & (rows < stats.counts.shape[0])]
    if rows.numel() == 0:
        return None
    rows = rows[stats.counts[rows] > 0]
    if rows.numel() == 0:
        return None
    lower = stats.min_voxel[rows].amin(dim=0)
    upper = stats.max_voxel[rows].amax(dim=0) + 1
    return tuple(
        slice(int(lower[axis].item()), int(upper[axis].item()))
        for axis in range(3)
    )


def _cluster_request_tasks(
    tasks: list[_LocalPartitionTask],
) -> list[list[_LocalPartitionTask]]:
    """Cluster only genuinely shared ownership; recovery is always singleton."""
    if not tasks:
        return []

    parent = list(range(len(tasks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    owned_sets = [set(task.owned_label_ids) for task in tasks]
    for left in range(len(tasks)):
        for right in range(left + 1, len(tasks)):
            if tasks[left].batch_index != tasks[right].batch_index:
                continue
            if (
                tasks[left].request_kind == "recovery"
                or tasks[right].request_kind == "recovery"
            ):
                continue
            if (
                owned_sets[left]
                and owned_sets[right]
                and owned_sets[left].intersection(owned_sets[right])
            ):
                union(left, right)

    groups: dict[int, list[_LocalPartitionTask]] = {}
    for index, task in enumerate(tasks):
        groups.setdefault(find(index), []).append(task)
    return [
        sorted(group, key=lambda task: task.request_index)
        for _, group in sorted(
            groups.items(),
            key=lambda item: min(task.request_index for task in item[1]),
        )
    ]


def _mask_in_box(
    box: tuple[slice, slice, slice],
    sub_box: tuple[slice, slice, slice],
    *,
    device: torch.device,
) -> Tensor:
    shape = tuple(int(axis.stop) - int(axis.start) for axis in box)
    result = torch.zeros(shape, device=device, dtype=torch.bool)
    lower = [max(int(box[a].start), int(sub_box[a].start)) for a in range(3)]
    upper = [min(int(box[a].stop), int(sub_box[a].stop)) for a in range(3)]
    if any(stop <= start for start, stop in zip(lower, upper)):
        return result
    local = tuple(
        slice(lower[a] - int(box[a].start), upper[a] - int(box[a].start))
        for a in range(3)
    )
    result[local] = True
    return result


def _minimal_global_box(
    mask: Tensor,
    parent_box: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice] | None:
    points = torch.nonzero(mask, as_tuple=False)
    if points.numel() == 0:
        return None
    lower = points.amin(dim=0)
    upper = points.amax(dim=0) + 1
    return tuple(
        slice(
            int(parent_box[a].start) + int(lower[a].item()),
            int(parent_box[a].start) + int(upper[a].item()),
        )
        for a in range(3)
    )


def _touches_nonvolume_box_boundary(
    component: Tensor,
    box: tuple[slice, slice, slice],
    full_shape: tuple[int, int, int],
) -> bool:
    return bool(
        (int(box[0].start) > 0 and component[0].any())
        or (int(box[0].stop) < full_shape[0] and component[-1].any())
        or (int(box[1].start) > 0 and component[:, 0].any())
        or (int(box[1].stop) < full_shape[1] and component[:, -1].any())
        or (int(box[2].start) > 0 and component[:, :, 0].any())
        or (int(box[2].stop) < full_shape[2] and component[:, :, -1].any())
    )


def _label_writable_components(
    local_labels: Tensor,
    writable_mask: Tensor,
) -> Tensor:
    """Label 6-connected equal-ID writable components after masking protection."""
    if local_labels.shape != writable_mask.shape:
        raise ValueError("local labels/writable mask must have identical shape")
    if not bool(writable_mask.any()):
        return torch.zeros_like(local_labels, dtype=torch.long)

    packed = torch.where(
        writable_mask,
        local_labels,
        torch.zeros((), device=local_labels.device, dtype=local_labels.dtype),
    )
    packed_cpu = packed.detach().to(
        device="cpu", dtype=torch.int32
    ).numpy()
    components_cpu = skimage_label(
        packed_cpu,
        background=0,
        connectivity=1,
    )
    return torch.from_numpy(components_cpu).to(
        device=local_labels.device,
        dtype=torch.long,
    )


def reconcile_request_local_labels(
    global_labels: Tensor,
    local_labels: Tensor,
    box: tuple[slice, slice, slice],
    writable_mask: Tensor,
    owned_label_ids: tuple[int, ...],
    *,
    copy_output: bool = True,
    next_label_id: int | None = None,
) -> tuple[Tensor | None, str]:
    """Reconcile connected writable pieces while protected labels stay fixed."""
    old_crop = global_labels[box]
    if (
        old_crop.shape != local_labels.shape
        or writable_mask.shape != local_labels.shape
    ):
        raise ValueError(
            "local labels/writable mask must align with the update box"
        )
    if not bool(writable_mask.any()):
        return global_labels.clone() if copy_output else global_labels, ""

    owned_set = set(int(value) for value in owned_label_ids)
    component_labels = _label_writable_components(
        local_labels, writable_mask
    )
    component_ids = torch.unique(component_labels)
    component_ids = component_ids[component_ids > 0]
    assignments: dict[int, int] = {}
    next_id = (
        int(global_labels.max().item()) + 1
        if next_label_id is None
        else int(next_label_id)
    )

    for component_id_tensor in component_ids:
        component_id = int(component_id_tensor.item())
        component = component_labels == component_id

        if _touches_nonvolume_box_boundary(
            component, box, tuple(global_labels.shape)
        ):
            return None, "unowned component touches halo boundary"

        overlap_ids = torch.unique(old_crop[component])
        overlap_ids = overlap_ids[overlap_ids > 0]
        if overlap_ids.numel():
            overlap_counts: dict[int, int] = {}
            for value in overlap_ids.detach().cpu().tolist():
                old_id = int(value)
                if old_id not in owned_set:
                    return None, "local component would merge existing labels"
                overlap_counts[old_id] = int(
                    (component & (old_crop == old_id)).sum().item()
                )
            owner = max(
                overlap_counts,
                key=lambda old_id: (overlap_counts[old_id], -old_id),
            )
            assignments[component_id] = owner
        else:
            assignments[component_id] = next_id
            next_id += 1

    by_assignment: dict[int, list[int]] = {}
    for component_id, old_id in assignments.items():
        by_assignment.setdefault(old_id, []).append(component_id)
    for old_id, proposed_ids in by_assignment.items():
        if old_id not in owned_set or len(proposed_ids) <= 1:
            continue
        sizes = {
            component_id: int(
                (component_labels == component_id).sum().item()
            )
            for component_id in proposed_ids
        }
        keep = max(
            proposed_ids,
            key=lambda value: (sizes[value], -value),
        )
        for component_id in proposed_ids:
            if component_id != keep:
                assignments[component_id] = next_id
                next_id += 1

    mapped = torch.zeros_like(local_labels)
    for component_id, global_id in assignments.items():
        mapped[component_labels == component_id] = global_id

    updated = global_labels.clone() if copy_output else global_labels
    target = updated[box]
    target[writable_mask] = mapped[writable_mask]
    updated[box] = target
    return updated, ""


def _request_fallback_context(
    global_labels: Tensor,
    local_labels: Tensor,
    box: tuple[slice, slice, slice],
    core_mask: Tensor,
    writable_mask: Tensor,
    reason: str,
    *,
    owned_label_ids: tuple[int, ...],
    request_kinds: list[str],
    cluster_request_count: int,
) -> dict[str, object]:
    old_crop = global_labels[box]
    protected_mask = ~writable_mask
    component_labels = _label_writable_components(
        local_labels, writable_mask
    )
    component_ids = torch.unique(component_labels)
    component_ids = component_ids[component_ids > 0]
    conflicting_old_ids: list[int] = []

    if reason == "local component would merge existing labels":
        owned_set = set(owned_label_ids)
        for component_id_tensor in component_ids:
            component = component_labels == int(component_id_tensor.item())
            overlap = _positive_ids(old_crop[component])
            unexpected = [
                value for value in overlap if value not in owned_set
            ]
            if unexpected:
                conflicting_old_ids = sorted(set(unexpected))
                break
    elif reason == "local component touches multiple external labels":
        local_ids = _positive_ids(local_labels[writable_mask])
        for local_id in local_ids:
            component = local_labels == local_id
            owners = _positive_ids(old_crop[component & protected_mask])
            if len(owners) > 1:
                conflicting_old_ids = owners
                break

    shape = tuple(int(axis.stop) - int(axis.start) for axis in box)
    return {
        "box_shape_zyx": shape,
        "box_voxel_count": int(shape[0] * shape[1] * shape[2]),
        "core_voxel_count": int(core_mask.sum().item()),
        "writable_voxel_count": int(writable_mask.sum().item()),
        "local_component_count": int(component_ids.numel()),
        "old_core_label_count": len(_positive_ids(old_crop[core_mask])),
        "old_shell_label_count": len(
            _positive_ids(old_crop[protected_mask])
        ),
        "conflicting_old_label_ids": conflicting_old_ids,
        "owned_label_ids": list(owned_label_ids),
        "request_kinds": list(request_kinds),
        "cluster_request_count": int(cluster_request_count),
    }


class LocalPartitionUpdater(nn.Module):
    def __init__(self, watershed: LearnedGeometryWatershed, cfg: RefinementConfig):
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
            "writable_voxel_count": int(context.get("writable_voxel_count", 0)),
            "local_component_count": int(context.get("local_component_count", 0)),
            "old_core_label_count": int(context.get("old_core_label_count", 0)),
            "old_shell_label_count": int(context.get("old_shell_label_count", 0)),
            "conflicting_old_label_ids": list(context.get("conflicting_old_label_ids", [])),
            "owned_label_ids": list(context.get("owned_label_ids", [])),
            "request_kinds": list(context.get("request_kinds", [])),
            "cluster_request_count": int(context.get("cluster_request_count", 0)),
        }
        print(
            "[stirnet-local-partition-fallback] " + json.dumps(event, sort_keys=True),
            flush=True,
        )
        return LocalPartitionUpdateResult(
            supervoxel_labels=self.watershed(geometry, spacing_um, dref_um, padding_mask),
            used_fallback=True,
            fallback_reason=reason,
            fallback_reason_code=reason_code,
            fallback_batch_index=int(batch_index),
            fallback_box_index=int(box_index),
            fallback_box_shape_zyx=context.get("box_shape_zyx"),
            fallback_box_voxel_count=int(context.get("box_voxel_count", 0)),
            fallback_core_voxel_count=int(context.get("core_voxel_count", 0)),
            fallback_local_component_count=int(context.get("local_component_count", 0)),
            fallback_old_core_label_count=int(context.get("old_core_label_count", 0)),
            fallback_old_shell_label_count=int(context.get("old_shell_label_count", 0)),
            fallback_conflicting_old_label_ids=list(
                context.get("conflicting_old_label_ids", [])
            ),
            updated_box_count=int(updated_box_count),
            updated_voxel_count=int(updated_voxel_count),
        )

    def _build_tasks(
        self,
        geometry: RefinedGeometryView,
        initial_labels: List[Tensor],
        spacing_um: Tensor,
        dref_um: Tensor,
        requests: List[RefinementRequest] | None,
        rag: RAGState | None,
        instances: InstanceState | None,
    ) -> tuple[list[_LocalPartitionTask] | None, str]:
        all_rois = list(geometry.delta.rois)
        if requests is None:
            selected_rois = all_rois
            selected_requests: list[RefinementRequest | None] = [None] * len(selected_rois)
        else:
            if len(all_rois) < len(requests):
                return None, "request/roi mismatch"
            selected_rois = all_rois[-len(requests):] if requests else []
            selected_requests = list(requests)
            if len(selected_rois) != len(selected_requests):
                return None, "request/roi mismatch"

        tasks: list[_LocalPartitionTask] = []
        for request_index, (roi, request) in enumerate(zip(selected_rois, selected_requests)):
            batch_index = int(roi.batch_index)
            if batch_index < 0 or batch_index >= len(initial_labels):
                return None, "request/roi mismatch"
            if request is not None and int(request.batch_index) != batch_index:
                return None, "request/roi mismatch"

            owned: tuple[int, ...] = ()
            kind = "unknown"
            source_index = -1
            if request is not None:
                kind = request.kind
                source_index = int(request.source_index)
                if rag is not None and instances is not None:
                    owned = _owned_supervoxel_ids(request, rag, instances)

            halo_voxels = torch.ceil(
                self.cfg.partition_halo_dref
                * dref_um[batch_index].float()
                / spacing_um[batch_index].float().clamp_min(1e-6)
            ).long()
            shape = tuple(initial_labels[batch_index].shape)
            dependency_seed = roi.slices_zyx
            if rag is not None and owned:
                owned_bbox = _owned_bbox_from_statistics(rag, batch_index, owned)
                if owned_bbox is not None:
                    dependency_seed = _union_boxes([dependency_seed, owned_bbox])
            dependency_box = _expand_box(dependency_seed, halo_voxels, shape)
            tasks.append(
                _LocalPartitionTask(
                    batch_index=batch_index,
                    request_index=request_index,
                    request_kind=kind,
                    source_index=source_index,
                    core_box=roi.slices_zyx,
                    dependency_box=dependency_box,
                    owned_label_ids=owned,
                )
            )
        return tasks, ""

    @staticmethod
    def _geometry_crop(
        geometry: GeometryLike,
        batch_index: int,
        box: tuple[slice, slice, slice],
    ) -> GeometryState:
        return GeometryState(
            foreground_logits=geometry_field_crop(geometry, "foreground_logits", batch_index, box)[None],
            surface_logits=geometry_field_crop(geometry, "surface_logits", batch_index, box)[None],
            separator_logits=geometry_field_crop(geometry, "separator_logits", batch_index, box)[None],
            sdf=geometry_field_crop(geometry, "sdf", batch_index, box)[None],
            flow=geometry_field_crop(geometry, "flow", batch_index, box)[None],
            centroid_offset=geometry_field_crop(geometry, "centroid_offset", batch_index, box)[None],
            seed_logits=geometry_field_crop(geometry, "seed_logits", batch_index, box)[None],
            features=None,
        )

    @torch.no_grad()
    def forward(
        self,
        initial_labels: List[Tensor],
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None = None,
        *,
        requests: List[RefinementRequest] | None = None,
        rag: RAGState | None = None,
        instances: InstanceState | None = None,
    ) -> LocalPartitionUpdateResult:
        if not isinstance(geometry, RefinedGeometryView) or geometry.delta.is_empty:
            return self._fallback(
                geometry, spacing_um, dref_um, padding_mask, "no sparse delta"
            )

        tasks, task_error = self._build_tasks(
            geometry, initial_labels, spacing_um, dref_um, requests, rag, instances
        )
        if tasks is None:
            return self._fallback(
                geometry, spacing_um, dref_um, padding_mask, task_error
            )

        clusters = _cluster_request_tasks(tasks)
        output = [labels.clone() for labels in initial_labels]
        updated_boxes: list[tuple[int, tuple[slice, slice, slice]]] = []
        updated_voxels = 0

        for cluster_index, cluster in enumerate(clusters):
            batch_index = cluster[0].batch_index
            if any(task.batch_index != batch_index for task in cluster):
                return self._fallback(
                    geometry, spacing_um, dref_um, padding_mask, "request/roi mismatch"
                )
            base = output[batch_index]
            box = _union_boxes([task.dependency_box for task in cluster])
            core_mask = torch.zeros(
                tuple(int(axis.stop) - int(axis.start) for axis in box),
                device=base.device,
                dtype=torch.bool,
            )
            for task in cluster:
                core_mask |= _mask_in_box(box, task.core_box, device=base.device)

            owned = tuple(
                sorted({label_id for task in cluster for label_id in task.owned_label_ids})
            )
            old_crop = base[box]
            if owned:
                owned_tensor = torch.tensor(owned, device=base.device, dtype=base.dtype)
                owned_mask = torch.isin(old_crop, owned_tensor)
            else:
                owned_mask = torch.zeros_like(old_crop, dtype=torch.bool)

            # Only complete request-owned old supervoxels plus background
            # inside the actual refined ROI are mutable. Background is not
            # expanded merely because it shares a watershed basin ID.
            writable_mask = owned_mask | (core_mask & (old_crop == 0))
            if not bool(writable_mask.any()):
                continue

            crop = self._geometry_crop(geometry, batch_index, box)
            local_padding = (
                None
                if padding_mask is None
                else padding_mask[
                    batch_index : batch_index + 1
                ][(slice(None), *box)]
            )
            local_labels = self.watershed(
                crop,
                spacing_um[batch_index : batch_index + 1],
                dref_um[batch_index : batch_index + 1],
                local_padding,
            )[0]

            next_label_id = max(int(labels.max().item()) for labels in output) + 1
            reconciled, reason = reconcile_request_local_labels(
                base,
                local_labels,
                box,
                writable_mask,
                owned,
                copy_output=False,
                next_label_id=next_label_id,
            )
            if reconciled is None:
                context = _request_fallback_context(
                    base,
                    local_labels,
                    box,
                    core_mask,
                    writable_mask,
                    reason,
                    owned_label_ids=owned,
                    request_kinds=[task.request_kind for task in cluster],
                    cluster_request_count=len(cluster),
                )
                return self._fallback(
                    geometry,
                    spacing_um,
                    dref_um,
                    padding_mask,
                    reason,
                    batch_index=batch_index,
                    box_index=cluster_index,
                    context=context,
                    updated_box_count=len(updated_boxes),
                    updated_voxel_count=updated_voxels,
                )

            output[batch_index] = reconciled
            changed_box = _minimal_global_box(writable_mask, box)
            if changed_box is not None:
                updated_boxes.append((batch_index, changed_box))
            updated_voxels += int(writable_mask.sum().item())

        return LocalPartitionUpdateResult(
            supervoxel_labels=output,
            used_fallback=False,
            updated_box_count=len(updated_boxes),
            updated_boxes=updated_boxes,
            updated_voxel_count=updated_voxels,
        )

__all__ = [
    "LocalPartitionUpdateResult",
    "LocalPartitionUpdater",
    "reconcile_local_labels",
    "reconcile_request_local_labels",
]
