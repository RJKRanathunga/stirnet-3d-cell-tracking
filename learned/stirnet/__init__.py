"""Public API for the spatial-first STIR-Net implementation."""

from .model import (
    GeometryCriterion,
    GeometryTargets,
    GeometryForwardOutput,
    ModelConfig,
    RAGCriterion,
    StirNet,
    StirNetConfig,
    StirNetOutput,
    SpatialForwardOutput,
    TemporalInput,
    build_geometry_targets,
)

__all__ = [
    "StirNet",
    "ModelConfig",
    "StirNetConfig",
    "StirNetOutput",
    "GeometryForwardOutput",
    "SpatialForwardOutput",
    "TemporalInput",
    "GeometryTargets",
    "build_geometry_targets",
    "GeometryCriterion",
    "RAGCriterion",
]
