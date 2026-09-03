from __future__ import annotations

# STIRNET_TINY_SUPERVOXEL_AGGLOMERATION_V1

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class TinySupervoxelAgglomerationDiagnostics:
    input_supervoxel_count: int
    output_supervoxel_count: int
    input_tiny_supervoxel_count: int
    merge_operation_count: int
    deleted_region_count: int
    deleted_voxel_count: int
    remaining_tiny_supervoxel_count: int


def _face_areas_zyx_um2(
    spacing_zyx_um: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    if spacing.shape != (3,):
        raise ValueError(
            "spacing_zyx_um must contain exactly three values (z, y, x)"
        )
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_zyx_um values must be finite and positive")

    sz, sy, sx = (float(v) for v in spacing)
    return (
        sy * sx,  # Z-normal face
        sz * sx,  # Y-normal face
        sz * sy,  # X-normal face
    )


def _build_physical_adjacency(
    labels: np.ndarray,
    *,
    face_areas_um2: tuple[float, float, float],
) -> list[dict[int, float]]:
    """Build positive-label 6-neighbour adjacency weighted by physical area."""
    max_label = int(labels.max()) if labels.size else 0
    adjacency: list[dict[int, float]] = [
        {} for _ in range(max_label + 1)
    ]
    if max_label <= 1:
        return adjacency

    base = max_label + 1

    for axis, face_area in enumerate(face_areas_um2):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)

        a = labels[tuple(lower)]
        b = labels[tuple(upper)]
        valid = (a > 0) & (b > 0) & (a != b)
        if not valid.any():
            continue

        av = a[valid].astype(np.int64, copy=False)
        bv = b[valid].astype(np.int64, copy=False)
        lo = np.minimum(av, bv)
        hi = np.maximum(av, bv)
        packed = lo * base + hi

        keys, counts = np.unique(
            packed,
            return_counts=True,
        )
        pair_lo = keys // base
        pair_hi = keys % base
        areas = counts.astype(np.float64) * float(face_area)

        for left, right, area in zip(
            pair_lo.tolist(),
            pair_hi.tolist(),
            areas.tolist(),
        ):
            left = int(left)
            right = int(right)
            area = float(area)
            adjacency[left][right] = (
                adjacency[left].get(right, 0.0) + area
            )
            adjacency[right][left] = (
                adjacency[right].get(left, 0.0) + area
            )

    return adjacency


def _compact_from_parent(
    labels: np.ndarray,
    parent: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    max_label = int(parent.shape[0] - 1)

    def resolve(label: int) -> int:
        path: list[int] = []
        current = int(label)
        while current > 0 and int(parent[current]) != current:
            path.append(current)
            current = int(parent[current])
        root = int(current)
        for item in path:
            parent[item] = root
        return root

    old_to_root = np.zeros(max_label + 1, dtype=np.int32)
    for old_id in present.tolist():
        old_to_root[int(old_id)] = resolve(int(old_id))

    surviving_roots = np.unique(old_to_root[old_to_root > 0])
    root_to_dense = np.zeros(max_label + 1, dtype=np.int32)
    root_to_dense[surviving_roots] = np.arange(
        1,
        surviving_roots.size + 1,
        dtype=np.int32,
    )

    old_to_dense = root_to_dense[old_to_root]
    return old_to_dense[labels].astype(np.int32, copy=False)


def agglomerate_tiny_supervoxels(
    labels: np.ndarray,
    spacing_zyx_um: tuple[float, float, float] | np.ndarray,
    *,
    max_voxels: int = 50,
) -> tuple[np.ndarray, TinySupervoxelAgglomerationDiagnostics]:
    """
    Remove pathological tiny atomic supervoxels before RAG construction.

    Every positive region with size <= max_voxels is processed iteratively.

    - If it has positive 6-neighbour regions, merge into the neighbour with
      greatest PHYSICAL shared interface area.
    - Interface ties prefer the larger neighbour, then the smaller label ID.
    - If it has no positive neighbour, delete it to background.
    - Region sizes and interfaces are updated after every merge, so chains of
      tiny supervoxels resolve transitively instead of leaving tiny remnants.

    Output positive labels are compact and consecutive.
    """
    source = np.asarray(labels)
    if source.ndim != 3:
        raise ValueError(
            f"labels must be a 3-D [Z,Y,X] array, got {source.shape}"
        )
    if int(max_voxels) < 1:
        raise ValueError("max_voxels must be >= 1")
    if source.size and int(source.min()) < 0:
        raise ValueError("labels cannot contain negative IDs")

    work = np.asarray(source, dtype=np.int32)
    if work.size == 0 or int(work.max()) <= 0:
        empty = work.copy()
        return empty, TinySupervoxelAgglomerationDiagnostics(
            input_supervoxel_count=0,
            output_supervoxel_count=0,
            input_tiny_supervoxel_count=0,
            merge_operation_count=0,
            deleted_region_count=0,
            deleted_voxel_count=0,
            remaining_tiny_supervoxel_count=0,
        )

    max_label = int(work.max())
    sizes = np.bincount(
        work.ravel(),
        minlength=max_label + 1,
    ).astype(np.int64, copy=False)
    present = np.flatnonzero(sizes > 0)
    present = present[present > 0]

    input_count = int(present.size)
    tiny_input = present[
        sizes[present] <= int(max_voxels)
    ]
    tiny_input_count = int(tiny_input.size)

    if tiny_input_count == 0:
        return work.copy(), TinySupervoxelAgglomerationDiagnostics(
            input_supervoxel_count=input_count,
            output_supervoxel_count=input_count,
            input_tiny_supervoxel_count=0,
            merge_operation_count=0,
            deleted_region_count=0,
            deleted_voxel_count=0,
            remaining_tiny_supervoxel_count=0,
        )

    face_areas = _face_areas_zyx_um2(spacing_zyx_um)
    adjacency = _build_physical_adjacency(
        work,
        face_areas_um2=face_areas,
    )

    active = sizes > 0
    active[0] = False
    parent = np.arange(max_label + 1, dtype=np.int32)

    heap: list[tuple[int, int]] = [
        (int(sizes[label]), int(label))
        for label in tiny_input.tolist()
    ]
    heapq.heapify(heap)

    merge_count = 0
    deleted_count = 0
    deleted_voxels = 0

    while heap:
        queued_size, source_id = heapq.heappop(heap)

        if not bool(active[source_id]):
            continue
        current_size = int(sizes[source_id])
        if current_size != int(queued_size):
            continue
        if current_size > int(max_voxels):
            continue

        stale = [
            neighbour
            for neighbour in adjacency[source_id]
            if not bool(active[neighbour])
        ]
        for neighbour in stale:
            adjacency[source_id].pop(neighbour, None)

        if not adjacency[source_id]:
            active[source_id] = False
            parent[source_id] = 0
            deleted_count += 1
            deleted_voxels += current_size
            sizes[source_id] = 0
            continue

        candidates = list(adjacency[source_id].items())
        target_id, _ = max(
            candidates,
            key=lambda item: (
                float(item[1]),
                int(sizes[int(item[0])]),
                -int(item[0]),
            ),
        )
        target_id = int(target_id)

        if target_id == source_id or not bool(active[target_id]):
            raise RuntimeError(
                "Tiny-supervoxel adjacency became inconsistent during agglomeration"
            )

        source_neighbours = dict(adjacency[source_id])
        adjacency[target_id].pop(source_id, None)

        for neighbour, area in source_neighbours.items():
            neighbour = int(neighbour)
            if neighbour == target_id:
                continue

            adjacency[neighbour].pop(source_id, None)
            if not bool(active[neighbour]):
                continue

            combined = (
                float(adjacency[target_id].get(neighbour, 0.0))
                + float(area)
            )
            adjacency[target_id][neighbour] = combined
            adjacency[neighbour][target_id] = combined

        adjacency[source_id].clear()

        parent[source_id] = target_id
        active[source_id] = False
        sizes[target_id] += sizes[source_id]
        sizes[source_id] = 0
        merge_count += 1

        if int(sizes[target_id]) <= int(max_voxels):
            heapq.heappush(
                heap,
                (int(sizes[target_id]), target_id),
            )

    output = _compact_from_parent(
        work,
        parent,
        present,
    )

    output_counts = np.bincount(output.ravel())
    output_present = np.flatnonzero(output_counts > 0)
    output_present = output_present[output_present > 0]
    remaining_tiny = int(
        np.count_nonzero(
            output_counts[output_present] <= int(max_voxels)
        )
    )

    if remaining_tiny:
        raise RuntimeError(
            "Tiny-supervoxel agglomeration invariant failed: "
            f"{remaining_tiny} positive regions remain <= {max_voxels} voxels"
        )

    return output, TinySupervoxelAgglomerationDiagnostics(
        input_supervoxel_count=input_count,
        output_supervoxel_count=int(output_present.size),
        input_tiny_supervoxel_count=tiny_input_count,
        merge_operation_count=int(merge_count),
        deleted_region_count=int(deleted_count),
        deleted_voxel_count=int(deleted_voxels),
        remaining_tiny_supervoxel_count=remaining_tiny,
    )
