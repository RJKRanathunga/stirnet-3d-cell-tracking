from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from learned.stirnet.data.graph_builder import (
    AssociationRecord,
    DetectionRecord,
    build_temporal_graph,
)
from learned.stirnet.data.temporal_motion import (
    compensate_detection_records,
    target_relative_global_motion_um,
)
from src.pipeline.stages.temporal_evidence import (
    build_motion_compensated_temporal_graph,
)
from src.tracking.trackastra import GlobalMotionEstimate


def _motion(cumulative):
    cumulative = np.asarray(cumulative, dtype=np.float64)
    pairwise = np.diff(cumulative, axis=0)
    align = -np.rint(cumulative).astype(np.int64)
    minimum = align.min(axis=0)
    placement = align - minimum[None, :]
    return GlobalMotionEstimate(
        pairwise_float_zyx=pairwise,
        cumulative_float_zyx=cumulative,
        align_int_zyx=align,
        placement_zyx=placement,
        canvas_shape_zyx=(32, 32, 32),
        pair_counts=np.full((len(cumulative) - 1,), 20, np.int64),
        inlier_counts=np.full((len(cumulative) - 1,), 20, np.int64),
        gate_physical=np.ones((len(cumulative) - 1,), np.float64),
        median_residual_physical=np.zeros((len(cumulative) - 1,), np.float64),
        p90_residual_physical=np.zeros((len(cumulative) - 1,), np.float64),
    )


def test_target_relative_motion_keeps_current_frame_fixed():
    cumulative = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 5.0, 0.0]]
    )
    result = target_relative_global_motion_um(
        cumulative,
        (1.0, 0.5, 1.0),
        target_frame=1,
        time_offsets=(-1, 0, 1),
    )
    assert np.allclose(result[-1], [0.0, -1.0, 0.0])
    assert np.allclose(result[0], [0.0, 0.0, 0.0])
    assert np.allclose(result[1], [0.0, 1.5, 0.0])


def test_compensation_removes_pure_global_motion_from_positions_and_velocity():
    records = [
        DetectionRecord(10, -1, (0.0, -2.0, 0.0), 100.0,
                        backward_velocity_um=(0.0, 2.0, 0.0),
                        forward_velocity_um=(0.0, 2.0, 0.0)),
        DetectionRecord(11, 0, (0.0, 0.0, 0.0), 100.0,
                        backward_velocity_um=(0.0, 2.0, 0.0),
                        forward_velocity_um=(0.0, 2.0, 0.0)),
        DetectionRecord(12, 1, (0.0, 2.0, 0.0), 100.0,
                        backward_velocity_um=(0.0, 2.0, 0.0),
                        forward_velocity_um=(0.0, 2.0, 0.0)),
    ]
    associations = [
        AssociationRecord(10, 11, score=0.9),
        AssociationRecord(11, 12, score=0.9),
    ]
    motion = {-1: (0.0, -2.0, 0.0), 0: (0.0, 0.0, 0.0), 1: (0.0, 2.0, 0.0)}
    result = compensate_detection_records(records, associations, motion)

    assert all(np.allclose(record.position_um, 0.0) for record in result)
    assert np.allclose(result[0].forward_velocity_um, 0.0)
    assert np.allclose(result[1].backward_velocity_um, 0.0)
    assert np.allclose(result[1].forward_velocity_um, 0.0)
    assert np.allclose(result[2].backward_velocity_um, 0.0)
    assert np.allclose(records[0].position_um, [0.0, -2.0, 0.0])


def test_build_temporal_graph_uses_compensated_geometry_everywhere():
    records = [
        DetectionRecord(1, -1, (0.0, -2.0, 0.0), 100.0),
        DetectionRecord(2, 0, (0.0, 0.0, 0.0), 100.0),
        DetectionRecord(3, 1, (0.0, 2.0, 0.0), 100.0),
    ]
    associations = [AssociationRecord(1, 2, score=0.9), AssociationRecord(2, 3, score=0.9)]
    graph = build_temporal_graph(
        records,
        associations,
        dref_um=1.0,
        temporal_radius=1,
        available_time_offsets=(-1, 0, 1),
        candidate_graph_enabled=False,
        global_motion_by_offset_um={
            -1: (0.0, -2.0, 0.0),
            0: (0.0, 0.0, 0.0),
            1: (0.0, 2.0, 0.0),
        },
    )
    assert np.allclose(graph["graph_x"][:, 1:4].numpy(), 0.0, atol=1e-6)
    assert np.allclose(graph["graph_x"][:, 17:23].numpy(), 0.0, atol=1e-6)
    assert graph["temporal_ref_um"].shape == (1, 3)
    assert np.allclose(graph["temporal_ref_um"].numpy(), 0.0, atol=1e-6)


def test_pipeline_bridge_uses_trackastra_float_cumulative_motion():
    estimate = _motion([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])
    tracking = SimpleNamespace(global_motion=estimate)
    records = [
        DetectionRecord(1, -1, (0.0, -2.0, 0.0), 100.0),
        DetectionRecord(2, 0, (0.0, 0.0, 0.0), 100.0),
        DetectionRecord(3, 1, (0.0, 2.0, 0.0), 100.0),
    ]
    associations = [AssociationRecord(1, 2, score=0.9), AssociationRecord(2, 3, score=0.9)]
    graph = build_motion_compensated_temporal_graph(
        records,
        associations,
        tracking_result=tracking,
        target_frame=1,
        spacing_zyx_um=(1.0, 1.0, 1.0),
        dref_um=1.0,
        temporal_radius=1,
        candidate_graph_enabled=False,
    )
    assert np.allclose(graph["graph_x"][:, 1:4].numpy(), 0.0, atol=1e-6)


def test_pipeline_bridge_refuses_missing_global_motion():
    tracking = SimpleNamespace(global_motion=None)
    try:
        build_motion_compensated_temporal_graph(
            [], [], tracking_result=tracking, target_frame=0,
            spacing_zyx_um=(1.0, 1.0, 1.0), dref_um=1.0, temporal_radius=0,
        )
    except RuntimeError as exc:
        assert "global_motion" in str(exc)
    else:
        raise AssertionError("temporal pipeline must not silently use raw coordinates")
