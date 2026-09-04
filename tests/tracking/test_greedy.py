from __future__ import annotations

import networkx as nx
import numpy as np

from src.tracking.trackastra.greedy import solve_greedy


def _candidate_graph():
    graph = nx.DiGraph()
    for node_id, frame in ((1, 0), (2, 0), (3, 1), (4, 1), (5, 1)):
        graph.add_node(
            node_id,
            time=frame,
            label=node_id,
            coords=np.asarray([float(node_id), 0.0, 0.0]),
        )
    graph.add_edge(1, 3, weight=0.95)
    graph.add_edge(2, 3, weight=0.90)
    graph.add_edge(1, 4, weight=0.85)
    graph.add_edge(1, 5, weight=0.80)
    graph.add_edge(2, 5, weight=0.40)
    return graph


def test_greedy_capacity_rules_with_divisions():
    solution, summary = solve_greedy(
        _candidate_graph(),
        allow_divisions=True,
    )
    assert set(solution.edges) == {(1, 3), (1, 4)}
    assert summary["candidate_edges_eligible"] == 4
    assert summary["accepted_edges"] == 2


def test_greedy_nodiv_allows_only_one_outgoing():
    solution, _ = solve_greedy(
        _candidate_graph(),
        allow_divisions=False,
    )
    assert set(solution.edges) == {(1, 3)}
