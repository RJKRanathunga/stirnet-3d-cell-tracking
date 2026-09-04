from __future__ import annotations

import networkx as nx
import numpy as np

from src.tracking.trackastra import GlobalMotionConfig
from src.tracking.trackastra.global_motion import estimate_global_motion


def _node(graph, node_id, *, time, label, coords):
    graph.add_node(
        node_id,
        time=int(time),
        label=int(label),
        coords=np.asarray(coords, dtype=np.float64),
    )


def test_bootstrap_motion_recovers_common_translation_with_outlier():
    graph = nx.DiGraph()
    shift = np.asarray([2.0, -3.0, 4.0], dtype=np.float64)

    for index in range(15):
        left_id = index
        right_id = 100 + index
        left = np.asarray([10.0 + index, 30.0, 40.0])
        displacement = (
            np.asarray([80.0, -60.0, 50.0])
            if index == 14
            else shift
        )
        _node(graph, left_id, time=0, label=index + 1, coords=left)
        _node(
            graph,
            right_id,
            time=1,
            label=index + 1,
            coords=left + displacement,
        )
        graph.add_edge(left_id, right_id)

    estimate = estimate_global_motion(
        graph,
        frame_count=2,
        spatial_shape_zyx=(64, 256, 256),
        config=GlobalMotionConfig(
            voxel_size_zyx=(1.0, 1.0, 1.0),
            minimum_pairs=10,
            mad_scale=4.0,
            minimum_residual_gate_physical=2.0,
        ),
    )
    np.testing.assert_allclose(
        estimate.pairwise_float_zyx[0],
        shift,
        atol=1e-12,
    )
    assert int(estimate.pair_counts[0]) == 15
    assert int(estimate.inlier_counts[0]) >= 10


def test_division_parent_is_excluded():
    graph = nx.DiGraph()
    shift = np.asarray([1.0, 2.0, 3.0])

    for index in range(10):
        left_id = index
        right_id = 100 + index
        left = np.asarray([index, 20.0, 30.0], dtype=np.float64)
        _node(graph, left_id, time=0, label=index + 1, coords=left)
        _node(
            graph,
            right_id,
            time=1,
            label=index + 1,
            coords=left + shift,
        )
        graph.add_edge(left_id, right_id)

    _node(graph, 1000, time=0, label=1000, coords=[10, 10, 10])
    _node(graph, 1001, time=1, label=1001, coords=[100, 100, 100])
    _node(graph, 1002, time=1, label=1002, coords=[120, 110, 90])
    graph.add_edge(1000, 1001)
    graph.add_edge(1000, 1002)

    estimate = estimate_global_motion(
        graph,
        frame_count=2,
        spatial_shape_zyx=(64, 256, 256),
        config=GlobalMotionConfig(minimum_pairs=10),
    )
    assert int(estimate.pair_counts[0]) == 10
    np.testing.assert_allclose(estimate.pairwise_float_zyx[0], shift)
