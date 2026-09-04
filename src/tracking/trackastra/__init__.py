"""Production Trackastra tracking with bootstrap global-motion compensation."""

from .config import (
    GlobalMotionConfig,
    GlobalMotionEstimate,
    TrackastraConfig,
    TrackastraResult,
)
from .global_motion import GlobalMotionEstimationError, estimate_global_motion
from .runner import BOOTSTRAP_STRATEGY, TRACKING_SCHEMA_VERSION, run_trackastra

__all__ = [
    "BOOTSTRAP_STRATEGY",
    "GlobalMotionConfig",
    "GlobalMotionEstimate",
    "GlobalMotionEstimationError",
    "TRACKING_SCHEMA_VERSION",
    "TrackastraConfig",
    "TrackastraResult",
    "estimate_global_motion",
    "run_trackastra",
]
