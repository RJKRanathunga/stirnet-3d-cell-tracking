from ..core.models import AdjacencyEdge
from ..core.sampling import pair_groups


def test_pair_groups_nearest_first():
    groups = pair_groups([
        AdjacencyEdge(1, 2, 2.0, 5.0),
        AdjacencyEdge(2, 3, 0.5, 3.0),
    ])
    assert groups[0].instance_ids == (2, 3)
