import numpy as np

from learned.instance_segmentation.datasets.core.adjacency import build_instance_adjacency


def test_adjacency_finds_near_pair_not_distant_cell() -> None:
    labels = np.zeros((12, 40, 40), dtype=np.int32)
    labels[3:8, 5:10, 5:10] = 1
    labels[3:8, 5:10, 11:16] = 2
    labels[3:8, 28:33, 28:33] = 3

    edges = build_instance_adjacency(
        labels,
        (1.0, 1.0, 1.0),
        max_distance_um=2.0,
    )
    pairs = {(edge.instance_a, edge.instance_b) for edge in edges}
    assert (1, 2) in pairs
    assert (1, 3) not in pairs
    assert (2, 3) not in pairs
