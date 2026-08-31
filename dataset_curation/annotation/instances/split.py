from __future__ import annotations

"""Exact weighted-supervoxel contact graph and multi-source split algorithm for merged-instance correction."""

import heapq

from dataclasses import dataclass

import numpy as np

DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

class AnnotationError(RuntimeError):
    pass

@dataclass(frozen=True)
class SplitResult:
    timepoint: int
    original_instance_id: int
    output_instance_ids: tuple[int, ...]
    seed_groups: tuple[tuple[int, ...], ...]
    groups: tuple[tuple[int, ...], ...]

def _supervoxel_contact_graph(
    sv_frame: np.ndarray,
    allowed_ids: set[int],
) -> dict[int, dict[int, float]]:
    """
    Build a 6-neighbour graph for the allowed supervoxels.

    Edge weight stored in the graph is physical contact area in um^2.
    """
    graph: dict[int, dict[int, float]] = {
        int(sv_id): {} for sv_id in allowed_ids
    }

    if len(allowed_ids) <= 1:
        return graph

    max_sv = int(np.max(sv_frame)) if sv_frame.size else 0
    allowed_lookup = np.zeros(max_sv + 1, dtype=bool)

    valid_ids = np.asarray(
        sorted(sv_id for sv_id in allowed_ids if 0 < sv_id <= max_sv),
        dtype=np.int64,
    )
    allowed_lookup[valid_ids] = True

    sz, sy, sx = DEFAULT_SPACING_ZYX_UM
    face_area_by_axis = (
        sy * sx,  # z-neighbour face has Y*X area
        sz * sx,  # y-neighbour face has Z*X area
        sz * sy,  # x-neighbour face has Z*Y area
    )

    for axis, face_area in enumerate(face_area_by_axis):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)

        a = sv_frame[tuple(left)].astype(np.int64, copy=False)
        b = sv_frame[tuple(right)].astype(np.int64, copy=False)

        valid = (
            (a > 0)
            & (b > 0)
            & (a != b)
            & allowed_lookup[a]
            & allowed_lookup[b]
        )

        if not np.any(valid):
            continue

        aa = a[valid]
        bb = b[valid]

        low = np.minimum(aa, bb)
        high = np.maximum(aa, bb)
        pairs = np.stack([low, high], axis=1)

        unique_pairs, counts = np.unique(
            pairs,
            axis=0,
            return_counts=True,
        )

        for (sv_a, sv_b), count in zip(
            unique_pairs.tolist(),
            counts.tolist(),
        ):
            sv_a = int(sv_a)
            sv_b = int(sv_b)
            area = float(count) * float(face_area)

            graph[sv_a][sv_b] = graph[sv_a].get(sv_b, 0.0) + area
            graph[sv_b][sv_a] = graph[sv_b].get(sv_a, 0.0) + area

    return graph

def _expand_seed_groups_by_contact_graph(
    *,
    sv_frame: np.ndarray,
    parent_supervoxels: set[int],
    seed_groups: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """
    Assign every SV in one merged instance to one seed group.

    We perform multi-source Dijkstra on the SV contact graph. Traversing a broad
    contact is cheap and traversing a narrow contact is expensive:

        traversal_cost = 1 / physical_contact_area

    Therefore the eventual group boundary tends to fall on narrow contact necks.
    """
    graph = _supervoxel_contact_graph(
        sv_frame,
        parent_supervoxels,
    )

    # best[sv] = (distance, group_index)
    best: dict[int, tuple[float, int]] = {}
    queue: list[tuple[float, int, int]] = []

    for group_index, seeds in enumerate(seed_groups):
        for sv_id in seeds:
            best[int(sv_id)] = (0.0, group_index)
            heapq.heappush(
                queue,
                (0.0, group_index, int(sv_id)),
            )

    while queue:
        distance, group_index, sv_id = heapq.heappop(queue)

        current = best.get(sv_id)
        if current is None:
            continue

        current_distance, current_group = current
        if (
            distance > current_distance + 1e-12
            or group_index != current_group
        ):
            continue

        for neighbour, contact_area in graph[sv_id].items():
            # Broad intra-cell contacts are preferred; narrow necks are costly.
            edge_cost = 1.0 / max(float(contact_area), 1e-12)
            candidate_distance = distance + edge_cost

            previous = best.get(neighbour)

            should_update = (
                previous is None
                or candidate_distance < previous[0] - 1e-12
                or (
                    abs(candidate_distance - previous[0]) <= 1e-12
                    and group_index < previous[1]
                )
            )

            if should_update:
                best[neighbour] = (
                    candidate_distance,
                    group_index,
                )
                heapq.heappush(
                    queue,
                    (
                        candidate_distance,
                        group_index,
                        neighbour,
                    ),
                )

    unreachable = sorted(
        int(sv_id)
        for sv_id in parent_supervoxels
        if int(sv_id) not in best
    )

    if unreachable:
        preview = ", ".join(str(v) for v in unreachable[:30])
        suffix = " ..." if len(unreachable) > 30 else ""
        raise AnnotationError(
            "The current predicted instance is not fully connected in the "
            "6-neighbour supervoxel contact graph. Unreachable SVs: "
            f"{preview}{suffix}"
        )

    expanded: list[list[int]] = [
        [] for _ in seed_groups
    ]

    for sv_id in sorted(parent_supervoxels):
        _, group_index = best[int(sv_id)]
        expanded[group_index].append(int(sv_id))

    # Every seed group necessarily remains non-empty, but validate anyway.
    if any(len(group) == 0 for group in expanded):
        raise AnnotationError(
            "Internal error: automatic seed expansion produced an empty cell."
        )

    return tuple(
        tuple(group)
        for group in expanded
    )
