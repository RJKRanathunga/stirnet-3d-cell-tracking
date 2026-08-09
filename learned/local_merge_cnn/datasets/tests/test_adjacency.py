import numpy as np

from ..core.adjacency import build_instance_adjacency


def test_close_instances_are_neighbors():
    labels = np.zeros((12, 30, 30), np.int32)
    labels[3:8, 3:10, 3:10] = 1
    labels[3:8, 11:18, 3:10] = 2
    labels[3:8, 22:28, 22:28] = 3
    edges = build_instance_adjacency(labels, (1.0, 1.0, 1.0), max_distance_um=2.5)
    pairs = {(e.instance_a, e.instance_b) for e in edges}
    assert (1, 2) in pairs
    assert (1, 3) not in pairs
