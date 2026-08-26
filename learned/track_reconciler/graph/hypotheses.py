"""Sparse division-hypothesis enumeration after candidate-edge gating."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..config import CandidateConfig


def enumerate_division_hypotheses(
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    edge_score: np.ndarray,
    *,
    target_start_frame: np.ndarray | None = None,
    max_children_per_parent: int | None = None,
) -> list[tuple[int, int]]:
    """Return pairs of edge indices sharing one parent.

    Only top candidate children are expanded; this prevents an O(k^2) daughter
    search from becoming the learned module's problem.  If target start frames
    are supplied, daughters must start simultaneously.
    """

    edge_source = np.asarray(edge_source)
    edge_target = np.asarray(edge_target)
    edge_score = np.asarray(edge_score, dtype=float)
    limit = max_children_per_parent or CandidateConfig().maximum_division_children_per_parent
    result: list[tuple[int, int]] = []
    for parent in np.unique(edge_source):
        idx = np.flatnonzero(edge_source == parent)
        idx = idx[np.argsort(-edge_score[idx], kind="stable")[:limit]]
        for first, second in combinations(idx.tolist(), 2):
            if edge_target[first] == edge_target[second]:
                continue
            if target_start_frame is not None and target_start_frame[first] != target_start_frame[second]:
                continue
            result.append((first, second))
    return result
