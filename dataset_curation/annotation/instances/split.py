
from dataset_curation._compat.instance_annotator import (
    SplitResult,
    _expand_seed_groups_by_contact_graph,
    _supervoxel_contact_graph,
)

supervoxel_contact_graph = _supervoxel_contact_graph
expand_seed_groups_by_contact_graph = _expand_seed_groups_by_contact_graph

__all__ = [
    "SplitResult",
    "supervoxel_contact_graph",
    "expand_seed_groups_by_contact_graph",
]
