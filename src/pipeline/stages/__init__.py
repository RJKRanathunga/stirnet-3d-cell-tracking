"""Adapters for semantic stages of the current pipeline."""

from .source_instances import SourceFrameResult, prepare_source_frame
from .spatial_refinement import (
    SpatialInferenceConfig,
    load_spatial_runtime,
    run_parallel_spatial_volume,
)
from .primary_tracking import (
    GlobalMotionConfig,
    GlobalMotionEstimate,
    TrackastraConfig,
    TrackastraResult,
    run_trackastra,
)
from .track_stitching import (
    LearnedTrackStitchingNotFinalized,
    require_track_stitcher,
)

__all__ = [
    "GlobalMotionConfig",
    "GlobalMotionEstimate",
    "LearnedTrackStitchingNotFinalized",
    "SourceFrameResult",
    "SpatialInferenceConfig",
    "TrackastraConfig",
    "TrackastraResult",
    "load_spatial_runtime",
    "prepare_source_frame",
    "require_track_stitcher",
    "run_parallel_spatial_volume",
    "run_trackastra",
]
