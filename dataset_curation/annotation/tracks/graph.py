
from dataset_curation._compat.track_annotator import (
    Edge,
    Node,
    _canonical_edge,
    build_trackastra_detection_edges,
)

canonical_edge = _canonical_edge
__all__ = ["Node", "Edge", "canonical_edge", "build_trackastra_detection_edges"]
