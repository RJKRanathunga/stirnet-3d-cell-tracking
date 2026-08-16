"""Spatial-first STIR-Net model package.

The package is intentionally self-contained so it can be merged into the
existing learned/stirnet/model tree by Codex after architecture review.
"""

from .config import ModelConfig, StirNetConfig
from .geometry import GeometryCriterion, GeometryTargets, build_geometry_targets
from .partition import RAGCriterion
from .stir_net import StirNet
from .types import (
    GeometryState,
    GeometryForwardOutput,
    InstanceState,
    PartitionState,
    RAGState,
    ReasoningState,
    StirNetOutput,
    SpatialForwardOutput,
    TemporalInput,
    TemporalState,
)

__all__ = [
    "StirNet",
    "ModelConfig",
    "StirNetConfig",
    "StirNetOutput",
    "GeometryForwardOutput",
    "SpatialForwardOutput",
    "GeometryState",
    "RAGState",
    "PartitionState",
    "InstanceState",
    "TemporalInput",
    "TemporalState",
    "ReasoningState",
    "GeometryTargets",
    "build_geometry_targets",
    "GeometryCriterion",
    "RAGCriterion",
]
