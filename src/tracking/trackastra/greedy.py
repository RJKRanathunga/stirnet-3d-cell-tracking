"""Greedy graph solver equivalent to Trackastra 0.5.5 selection semantics."""

from __future__ import annotations

import math
import time
from typing import Any

import networkx as nx


def solve_greedy(
    candidate_graph,
    *,
    allow_divisions: bool,
    threshold: float = 0.5,
    edge_attr: str = "weight",
) -> tuple[nx.DiGraph, dict[str, object]]:
    started = time.perf_counter()

    edges = sorted(
        candidate_graph.edges(data=True),
        key=lambda edge: edge[2][edge_attr],
        reverse=True,
    )
    max_out = 2 if allow_divisions else 1
    targets_with_parent: set[Any] = set()
    source_out_count: dict[Any, int] = {}
    accepted: list[tuple[Any, Any, dict[str, Any]]] = []
    eligible = 0

    for node_in, node_out, attributes in edges:
        weight = float(attributes[edge_attr])
        if weight < float(threshold):
            break
        if not math.isfinite(weight) or weight > 1.0:
            raise RuntimeError(
                f"Invalid Trackastra candidate weight {weight!r} "
                f"for {node_in!r}->{node_out!r}"
            )
        eligible += 1
        if node_out in targets_with_parent:
            continue
        if source_out_count.get(node_in, 0) >= max_out:
            continue

        targets_with_parent.add(node_out)
        source_out_count[node_in] = source_out_count.get(node_in, 0) + 1
        accepted.append((node_in, node_out, dict(attributes)))

    solution = nx.DiGraph()
    for node_in, node_out, attributes in accepted:
        if node_in not in solution:
            solution.add_node(node_in, **dict(candidate_graph.nodes[node_in]))
        if node_out not in solution:
            solution.add_node(node_out, **dict(candidate_graph.nodes[node_out]))
        solution.add_edge(node_in, node_out, **attributes)

    return solution, {
        "solver": "trackastra_equivalent_local_greedy",
        "candidate_edges_total": int(len(edges)),
        "candidate_edges_eligible": int(eligible),
        "accepted_edges": int(len(accepted)),
        "allow_divisions": bool(allow_divisions),
        "threshold": float(threshold),
        "seconds": float(time.perf_counter() - started),
    }


__all__ = ["solve_greedy"]
