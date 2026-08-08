"""Deterministic selection of instance groups for sample generation."""

from __future__ import annotations

from collections.abc import Iterable

from .models import AdjacencyEdge, InstanceGroup


def pair_groups(
    edges: Iterable[AdjacencyEdge],
    *,
    max_separation_um: float | None = None,
) -> tuple[InstanceGroup, ...]:
    """Convert adjacency edges into nearest-first two-instance sample groups."""

    selected: list[InstanceGroup] = []
    for edge in edges:
        if max_separation_um is not None and edge.separation_um > max_separation_um:
            continue
        # Smaller separation gets a larger ranking score.
        score = 1.0 / (1.0 + edge.separation_um)
        selected.append(
            InstanceGroup(
                instance_ids=(edge.instance_a, edge.instance_b),
                kind="pair_merge",
                score=score,
            )
        )
    selected.sort(key=lambda group: (-group.score, group.instance_ids))
    return tuple(selected)


def single_groups(instance_ids: Iterable[int]) -> tuple[InstanceGroup, ...]:
    """Create ordinary single-cell groups. Hard-negative ranking comes later."""

    return tuple(
        InstanceGroup(instance_ids=(int(instance_id),), kind="single", score=0.0)
        for instance_id in sorted(set(int(v) for v in instance_ids if int(v) > 0))
    )
