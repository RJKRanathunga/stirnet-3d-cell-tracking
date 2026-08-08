"""Dataset-independent sample construction primitives."""

from .adjacency import build_instance_adjacency
from .models import (
    AdjacencyEdge,
    AnnotatedVolume,
    InstanceGroup,
    TargetBundle,
    TrainingSample,
    VolumeRecord,
)
from .sample_builder import SampleBuildError, SampleBuilder
from .sample_selection import GroupRejection, ValidSampleSelection, select_valid_sample

__all__ = [
    "AdjacencyEdge",
    "AnnotatedVolume",
    "InstanceGroup",
    "SampleBuildError",
    "SampleBuilder",
    "TargetBundle",
    "TrainingSample",
    "VolumeRecord",
    "build_instance_adjacency",
    "GroupRejection",
    "ValidSampleSelection",
    "select_valid_sample",
]
