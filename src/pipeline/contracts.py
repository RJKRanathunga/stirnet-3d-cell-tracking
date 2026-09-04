"""Stable orchestration contracts for the current production architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PipelineRequest:
    source: Path
    output_directory: Path
    sample_id: str | None = None


@dataclass
class PipelineState:
    source_instances: Any | None = None
    spatial_refinement: Any | None = None
    primary_tracking: Any | None = None
    track_stitching: Any | None = None
    exported: Any | None = None


SourceInstanceStage = Callable[[PipelineRequest], Any]
SpatialRefinementStage = Callable[[PipelineRequest, Any], Any]
PrimaryTrackingStage = Callable[[PipelineRequest, Any], Any]
TrackStitchingStage = Callable[[PipelineRequest, Any, Any], Any]
ExportStage = Callable[[PipelineRequest, Any, Any], Any]


@dataclass(frozen=True)
class PipelineStages:
    """Concrete adapters for all current architecture stages.

    Track stitching is injected until the learned reconciler's production
    checkpoint/inference contract is finalized.
    """

    source_instances: SourceInstanceStage
    spatial_refinement: SpatialRefinementStage
    primary_tracking: PrimaryTrackingStage
    track_stitching: TrackStitchingStage
    export: ExportStage


__all__ = [
    "ExportStage",
    "PipelineRequest",
    "PipelineStages",
    "PipelineState",
    "PrimaryTrackingStage",
    "SourceInstanceStage",
    "SpatialRefinementStage",
    "TrackStitchingStage",
]
