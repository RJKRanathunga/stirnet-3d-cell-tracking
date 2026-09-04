"""Effective-EDT plus conservative geometric marker segmentation."""

from .config import (
    CenterCandidateConfig,
    DEFAULT_SEGMENTATION_CONFIG,
    GeometricCompletionConfig,
    SegmentationConfig,
)
from .candidate_detection import (
    detect_geometric_candidate,
    safely_detect_geometric_candidate,
)
from .marker_completion import (
    analyze_geometric_completion,
    combine_markers,
    convert_effective_peaks_to_markers,
    safely_complete_geometric_markers,
)
from .models import (
    CenterProposal,
    GeometricBody,
    GeometricCandidateResult,
    GeometricCompletionResult,
    InstanceMarker,
    ShapePeakCandidate,
    SurfaceCap,
)
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
    "CenterCandidateConfig",
    "CenterProposal",
    "DEFAULT_SEGMENTATION_CONFIG",
    "GeometricBody",
    "GeometricCandidateResult",
    "GeometricCompletionConfig",
    "GeometricCompletionResult",
    "InstanceComponentMapping",
    "InstanceMarker",
    "SegmentationConfig",
    "SegmentationResult",
    "ShapePeakCandidate",
    "SurfaceCap",
    "analyze_geometric_completion",
    "combine_markers",
    "convert_effective_peaks_to_markers",
    "detect_geometric_candidate",
    "safely_detect_geometric_candidate",
    "safely_complete_geometric_markers",
    "segment_instances",
    "segment_instances_detailed",
]
