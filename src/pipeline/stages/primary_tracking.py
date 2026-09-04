"""Stage 3: Trackastra primary tracking with bootstrap global-motion removal."""

from src.tracking.trackastra import (
    GlobalMotionConfig,
    GlobalMotionEstimate,
    TrackastraConfig,
    TrackastraResult,
    run_trackastra,
)

__all__ = [
    "GlobalMotionConfig",
    "GlobalMotionEstimate",
    "TrackastraConfig",
    "TrackastraResult",
    "run_trackastra",
]
