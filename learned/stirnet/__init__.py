"""Public API for the spatial-first STIR-Net implementation."""

from .model import (
    GeometryCriterion,
    GeometryTargets,
    ModelConfig,
    RAGCriterion,
    StirNet,
    StirNetConfig,
    StirNetOutput,
    TemporalInput,
    build_geometry_targets,
)

__all__ = [
    "StirNet",
    "ModelConfig",
    "StirNetConfig",
    "StirNetOutput",
    "TemporalInput",
    "GeometryTargets",
    "build_geometry_targets",
    "GeometryCriterion",
    "RAGCriterion",
]
