# STIRNET_TEMPORAL_GLOBAL_MOTION_V1
"""Pipeline bridge for global-motion-compensated STIR-Net temporal evidence.

The adapter consumes the float global-motion estimate already produced by
production Trackastra. It does not restabilize the movie and does not alter the
Trackastra graph, which remains in source/BioHub coordinates.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from learned.stirnet.data.graph_builder import (
    AssociationRecord,
    DetectionRecord,
    build_temporal_graph,
    sequence_available_time_offsets,
)
from learned.stirnet.data.temporal_motion import (
    target_relative_global_motion_um,
)
from src.tracking.trackastra import TrackastraResult


def temporal_motion_by_offset_um(
    tracking_result: TrackastraResult,
    *,
    target_frame: int,
    spacing_zyx_um: Sequence[float],
    available_time_offsets: Iterable[int],
) -> dict[int, tuple[float, float, float]]:
    """Convert Trackastra bootstrap motion into target-relative STIR-Net motion."""

    estimate = tracking_result.global_motion
    if estimate is None:
        raise RuntimeError(
            "STIR-Net temporal reasoning requires Trackastra global motion. "
            "Run primary tracking with GlobalMotionConfig(enabled=True)."
        )

    return target_relative_global_motion_um(
        estimate.cumulative_float_zyx,
        spacing_zyx_um,
        target_frame=int(target_frame),
        time_offsets=available_time_offsets,
    )


def build_motion_compensated_temporal_graph(
    records: Iterable[DetectionRecord],
    associations: Iterable[AssociationRecord],
    *,
    tracking_result: TrackastraResult,
    target_frame: int,
    spacing_zyx_um: Sequence[float],
    dref_um: float,
    temporal_radius: int = 2,
    available_time_offsets: Iterable[int] | None = None,
    **graph_kwargs,
) -> dict:
    """Build the temporal graph in stabilized target-frame geometry.

    DetectionRecord positions supplied by callers remain source-coordinate
    positions relative to the current target patch center. The same float
    cumulative motion used by production Trackastra is converted to offsets
    relative to the current target frame and passed to STIR-Net's graph builder.
    """

    estimate = tracking_result.global_motion
    if estimate is None:
        raise RuntimeError(
            "Cannot build motion-compensated STIR-Net temporal input because "
            "TrackastraResult.global_motion is None."
        )

    frame_count = int(estimate.cumulative_float_zyx.shape[0])
    if available_time_offsets is None:
        offsets = sequence_available_time_offsets(
            int(target_frame),
            frame_count,
            int(temporal_radius),
        )
    else:
        offsets = tuple(sorted({int(v) for v in available_time_offsets}))

    relative_motion = temporal_motion_by_offset_um(
        tracking_result,
        target_frame=int(target_frame),
        spacing_zyx_um=spacing_zyx_um,
        available_time_offsets=offsets,
    )

    return build_temporal_graph(
        records,
        associations,
        dref_um=float(dref_um),
        temporal_radius=int(temporal_radius),
        available_time_offsets=offsets,
        global_motion_by_offset_um=relative_motion,
        **graph_kwargs,
    )


__all__ = [
    "build_motion_compensated_temporal_graph",
    "temporal_motion_by_offset_um",
]
