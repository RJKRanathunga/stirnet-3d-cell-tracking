"""Effective-EDT plus conservative geometric marker segmentation."""

from .config import (
    DEFAULT_SEGMENTATION_CONFIG,
    GeometricCompletionConfig,
    SegmentationConfig,
)
from .marker_completion import (
    analyze_geometric_completion,
    combine_markers,
    convert_effective_peaks_to_markers,
    safely_complete_geometric_markers,
)
from .models import GeometricBody, GeometricCompletionResult, InstanceMarker, SurfaceCap
from .pipeline import (
    ComponentDiagnostic,
    ComponentDebugArtifacts,
    InstanceComponentMapping,
    SegmentationResult,
    segment_instances,
    segment_instances_detailed,
)

__all__ = [
    "ComponentDiagnostic",
    "ComponentDebugArtifacts",
    "DEFAULT_SEGMENTATION_CONFIG",
    "GeometricBody",
    "GeometricCompletionConfig",
    "GeometricCompletionResult",
    "InstanceComponentMapping",
    "InstanceMarker",
    "SegmentationConfig",
    "SegmentationResult",
    "SurfaceCap",
    "analyze_geometric_completion",
    "combine_markers",
    "convert_effective_peaks_to_markers",
    "safely_complete_geometric_markers",
    "segment_instances",
    "segment_instances_detailed",
]
