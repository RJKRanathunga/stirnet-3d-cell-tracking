"""Shared object-centric dataset machinery."""

from .adjacency import build_instance_adjacency
from .component_transform import (
    bbox_from_binary_mask,
    bbox_from_instance_slices,
    build_canonical_transform,
    transformed_bbox_extent_canonical,
    transformed_bbox_extent_vox,
)
from .models import (
    AdjacencyEdge,
    AnnotatedVolume,
    CanonicalTransform,
    ComponentBBox,
    InstanceGroup,
    TargetBundle,
    TrainingSample,
    VolumeRecord,
)
from .sample_builder import SampleBuildError, SampleBuilder
from .sample_selection import GroupRejection, ValidSampleSelection, select_valid_sample
from .sampling import pair_groups, single_groups

__all__ = [
    "AdjacencyEdge", "AnnotatedVolume", "CanonicalTransform", "ComponentBBox",
    "GroupRejection", "InstanceGroup", "SampleBuildError", "SampleBuilder",
    "TargetBundle", "TrainingSample", "ValidSampleSelection", "VolumeRecord",
    "bbox_from_binary_mask", "bbox_from_instance_slices", "build_canonical_transform",
    "build_instance_adjacency", "pair_groups", "select_valid_sample", "single_groups",
    "transformed_bbox_extent_canonical", "transformed_bbox_extent_vox",
]
