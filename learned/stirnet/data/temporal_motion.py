# STIRNET_TEMPORAL_GLOBAL_MOTION_V1
"""Global-motion normalization for STIR-Net temporal detection records.

The spatial model and persisted Trackastra graph remain in source coordinates.
Only temporal detection geometry is expressed in a target-frame-anchored
coordinate system.

For target frame t0 and observation t:

    p'(t | t0) = p(t) - (G(t) - G(t0))

``p(t)`` is already relative to the current target patch center. ``G`` is the
cumulative common displacement in physical Z,Y,X units.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

import numpy as np


STIRNET_TEMPORAL_GLOBAL_MOTION_CONTRACT = "target_frame_anchored_v1"


def _zyx(value: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite Z,Y,X values")
    return result


def target_relative_global_motion_um(
    cumulative_motion_zyx_vox: np.ndarray,
    spacing_zyx_um: Sequence[float],
    *,
    target_frame: int,
    time_offsets: Iterable[int],
) -> dict[int, tuple[float, float, float]]:
    """Return ``G(t0+dt)-G(t0)`` in physical units for requested offsets.

    Integer padded-canvas placement is deliberately not used. STIR-Net temporal
    reasoning retains the sub-voxel float motion estimate from Trackastra.
    """

    cumulative = np.asarray(cumulative_motion_zyx_vox, dtype=np.float64)
    if cumulative.ndim != 2 or cumulative.shape[1] != 3:
        raise ValueError(
            "cumulative_motion_zyx_vox must have shape [T,3], "
            f"got {cumulative.shape}"
        )
    if not np.all(np.isfinite(cumulative)):
        raise ValueError("cumulative global motion contains non-finite values")

    spacing = _zyx(spacing_zyx_um, name="spacing_zyx_um")
    if np.any(spacing <= 0):
        raise ValueError("spacing_zyx_um must be strictly positive")

    target = int(target_frame)
    if not 0 <= target < int(cumulative.shape[0]):
        raise ValueError(
            f"target_frame={target} is outside motion length={cumulative.shape[0]}"
        )

    offsets = tuple(sorted({int(value) for value in time_offsets}))
    if not offsets:
        raise ValueError("time_offsets cannot be empty")
    if 0 not in offsets:
        raise ValueError("time_offsets must contain target offset 0")

    base = cumulative[target]
    result: dict[int, tuple[float, float, float]] = {}
    for offset in offsets:
        frame = target + int(offset)
        if not 0 <= frame < int(cumulative.shape[0]):
            raise ValueError(
                f"target_frame + offset is outside movie: {target}+{offset}"
            )
        delta_um = (cumulative[frame] - base) * spacing
        result[int(offset)] = tuple(float(v) for v in delta_um.tolist())

    if not np.allclose(result[0], 0.0, atol=1e-7, rtol=0.0):
        raise RuntimeError("target-frame global-motion offset must be zero")
    return result


def compensate_detection_records(
    records: Iterable[object],
    associations: Iterable[object],
    global_motion_by_offset_um: Mapping[int, Sequence[float]],
) -> list[object]:
    """Return detection records with common/global translation removed.

    Positions are shifted by target-relative global motion. Backward/forward
    velocity fields are then recomputed from accepted non-division temporal
    Trackastra continuations in the compensated coordinate system. This avoids
    leaking global motion back through the velocity features after positions
    were stabilized.
    """

    source_records = list(records)
    source_associations = list(associations)
    if not source_records:
        return []

    motion = {
        int(offset): _zyx(value, name=f"global_motion_by_offset_um[{offset}]")
        for offset, value in dict(global_motion_by_offset_um).items()
    }
    if 0 not in motion:
        raise ValueError("global_motion_by_offset_um must contain offset 0")
    if not np.allclose(motion[0], 0.0, atol=1e-6, rtol=0.0):
        raise ValueError(
            "global_motion_by_offset_um must be target-frame anchored: offset 0 == 0"
        )

    missing_offsets = sorted(
        {
            int(getattr(record, "time_offset"))
            for record in source_records
            if int(getattr(record, "time_offset")) not in motion
        }
    )
    if missing_offsets:
        raise ValueError(
            "global motion is missing DetectionRecord time offsets: "
            f"{missing_offsets}"
        )

    transformed: list[object] = []
    node_to_index: dict[int, int] = {}
    position_by_node: dict[int, np.ndarray] = {}

    for index, record in enumerate(source_records):
        node_id = int(getattr(record, "node_id"))
        if node_id in node_to_index:
            raise ValueError(f"duplicate temporal node_id={node_id}")

        offset = int(getattr(record, "time_offset"))
        source_position = _zyx(
            getattr(record, "position_um"),
            name=f"record[{node_id}].position_um",
        )
        stabilized_position = source_position - motion[offset]
        transformed.append(
            replace(
                record,
                position_um=tuple(float(v) for v in stabilized_position.tolist()),
                backward_velocity_um=(0.0, 0.0, 0.0),
                forward_velocity_um=(0.0, 0.0, 0.0),
            )
        )
        node_to_index[node_id] = index
        position_by_node[node_id] = stabilized_position

    incoming: dict[int, list[np.ndarray]] = {}
    outgoing: dict[int, list[np.ndarray]] = {}

    for association in source_associations:
        if str(getattr(association, "relation", "temporal")) != "temporal":
            continue

        source_id = int(getattr(association, "src_node_id"))
        destination_id = int(getattr(association, "dst_node_id"))
        if source_id not in node_to_index or destination_id not in node_to_index:
            continue

        source_record = source_records[node_to_index[source_id]]
        destination_record = source_records[node_to_index[destination_id]]
        dt = (
            int(getattr(destination_record, "time_offset"))
            - int(getattr(source_record, "time_offset"))
        )
        if dt <= 0:
            continue

        velocity = (
            position_by_node[destination_id] - position_by_node[source_id]
        ) / float(dt)
        outgoing.setdefault(source_id, []).append(velocity)
        incoming.setdefault(destination_id, []).append(velocity)

    result: list[object] = []
    for record in transformed:
        node_id = int(getattr(record, "node_id"))
        backward = (
            np.mean(np.stack(incoming[node_id]), axis=0)
            if node_id in incoming
            else np.zeros(3, dtype=np.float64)
        )
        forward = (
            np.mean(np.stack(outgoing[node_id]), axis=0)
            if node_id in outgoing
            else np.zeros(3, dtype=np.float64)
        )
        result.append(
            replace(
                record,
                backward_velocity_um=tuple(float(v) for v in backward.tolist()),
                forward_velocity_um=tuple(float(v) for v in forward.tolist()),
            )
        )
    return result


__all__ = [
    "STIRNET_TEMPORAL_GLOBAL_MOTION_CONTRACT",
    "compensate_detection_records",
    "target_relative_global_motion_um",
]
