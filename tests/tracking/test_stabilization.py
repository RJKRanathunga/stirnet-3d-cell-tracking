from __future__ import annotations

import networkx as nx
import numpy as np

from src.tracking.trackastra.config import GlobalMotionEstimate
from src.tracking.trackastra.stabilization import (
    build_stabilized_movies,
    pad_frame_to_canvas,
    restore_graph_coordinates,
)


def _estimate() -> GlobalMotionEstimate:
    return GlobalMotionEstimate(
        pairwise_float_zyx=np.asarray([[1.0, 0.0, -2.0]]),
        cumulative_float_zyx=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, -2.0]]
        ),
        align_int_zyx=np.asarray([[0, 0, 0], [-1, 0, 2]], dtype=np.int64),
        placement_zyx=np.asarray([[1, 0, 0], [0, 0, 2]], dtype=np.int64),
        canvas_shape_zyx=(3, 3, 6),
        pair_counts=np.asarray([12], dtype=np.int64),
        inlier_counts=np.asarray([12], dtype=np.int64),
        gate_physical=np.asarray([2.0]),
        median_residual_physical=np.asarray([0.0]),
        p90_residual_physical=np.asarray([0.0]),
    )


def test_zero_padded_translation_never_wraps():
    frame = np.zeros((2, 3, 4), dtype=np.uint16)
    frame[0, 0, 0] = 7
    shifted = pad_frame_to_canvas(
        frame,
        placement_zyx=(1, 0, 2),
        canvas_shape_zyx=(3, 3, 6),
    )
    assert shifted.dtype == np.uint16
    assert shifted[1, 0, 2] == 7
    assert int(np.count_nonzero(shifted)) == 1
    assert shifted[0, -1, -1] == 0


def test_lazy_stabilization_preserves_ids_and_shape():
    raw = np.zeros((2, 2, 3, 4), dtype=np.uint16)
    labels = np.zeros_like(raw)
    labels[0, 0, 0, 0] = 11
    labels[1, 1, 2, 3] = 29

    shifted_raw, shifted_labels = build_stabilized_movies(
        raw,
        labels,
        _estimate(),
    )
    assert tuple(int(v) for v in shifted_raw.shape) == (2, 3, 3, 6)
    materialized = shifted_labels.compute()
    assert materialized[0, 1, 0, 0] == 11
    assert materialized[1, 1, 2, 5] == 29
    assert set(np.unique(materialized).tolist()) == {0, 11, 29}


def test_graph_coordinates_restore_to_source_space():
    graph = nx.DiGraph()
    graph.add_node(1, time=0, label=11, coords=np.asarray([5.0, 6.0, 7.0]))
    graph.add_node(2, time=1, label=29, coords=np.asarray([8.0, 9.0, 10.0]))
    graph.add_edge(1, 2, weight=0.9)

    restored = restore_graph_coordinates(graph, _estimate())
    np.testing.assert_allclose(restored.nodes[1]["coords"], [4.0, 6.0, 7.0])
    np.testing.assert_allclose(restored.nodes[2]["coords"], [8.0, 9.0, 8.0])
    np.testing.assert_allclose(graph.nodes[2]["coords"], [8.0, 9.0, 10.0])
