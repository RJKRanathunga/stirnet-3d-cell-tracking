from __future__ import annotations

import unittest

import numpy as np

from graph_tracking import GraphTrackingConfig
from graph_tracking.spatial_graph import build_spatial_graph, neighbor_indices, relative_vector


class SpatialGraphTests(unittest.TestCase):
    def test_unique_edges_and_vector_direction(self) -> None:
        graph = build_spatial_graph(
            frame=0,
            node_ids=np.asarray([10, 20, 30]),
            positions_zyx_um=np.asarray([[0, 0, 0], [0, 2, 0], [0, 4, 0]], dtype=float),
            volumes=np.asarray([100, 120, 90], dtype=float),
            touches_boundary=np.asarray([False, False, False]),
            boundary_faces=("", "", ""),
            config=GraphTrackingConfig(maximum_radius_um=3.0, maximum_neighbors=2),
        )
        self.assertEqual(graph.edge_count, 2)
        self.assertEqual(neighbor_indices(graph, 1).tolist(), [0, 2])
        np.testing.assert_allclose(relative_vector(graph, 0, 1), [0, 2, 0])
        np.testing.assert_allclose(relative_vector(graph, 1, 0), [0, -2, 0])

    def test_empty_graph(self) -> None:
        graph = build_spatial_graph(
            frame=0,
            node_ids=np.empty(0, dtype=int),
            positions_zyx_um=np.empty((0, 3)),
            volumes=np.empty(0),
            touches_boundary=np.empty(0, dtype=bool),
            boundary_faces=(),
            config=GraphTrackingConfig(),
        )
        self.assertEqual(graph.node_count, 0)
        self.assertEqual(graph.edge_count, 0)


if __name__ == "__main__":
    unittest.main()
