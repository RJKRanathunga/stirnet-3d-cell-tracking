from __future__ import annotations

import pytest
import torch

from learned.stirnet.data.historical_instances import TEMPORAL_CACHE_CONTRACT_VERSION
from learned.stirnet.data.graph_builder import (
    TEMPORAL_STATUS_FUTURE_CONTEXT_COLUMN,
    TEMPORAL_STATUS_PAST_CONTEXT_COLUMN,
    AssociationRecord,
    DetectionRecord,
    build_temporal_graph,
    sequence_available_time_offsets,
)


def _track_graph(
    times: tuple[int, ...],
    *,
    available_time_offsets: tuple[int, ...],
):
    records = [
        DetectionRecord(
            node_id=index,
            time_offset=time,
            position_um=(float(time), 0.0, 0.0),
            physical_volume_um3=100.0,
        )
        for index, time in enumerate(times)
    ]
    associations = [
        AssociationRecord(
            src_node_id=index,
            dst_node_id=index + 1,
            score=0.95,
        )
        for index in range(len(records) - 1)
    ]
    return build_temporal_graph(
        records,
        associations,
        dref_um=1.0,
        temporal_radius=2,
        available_time_offsets=available_time_offsets,
        candidate_graph_enabled=False,
    )


def test_temporal_cache_contract_bumped_for_availability_semantics():
    assert TEMPORAL_CACHE_CONTRACT_VERSION == 4


def test_sequence_available_time_offsets_for_20_frame_movie():
    assert sequence_available_time_offsets(0, 20, 2) == (0, 1, 2)
    assert sequence_available_time_offsets(1, 20, 2) == (-1, 0, 1, 2)
    assert sequence_available_time_offsets(2, 20, 2) == (-2, -1, 0, 1, 2)
    assert sequence_available_time_offsets(17, 20, 2) == (-2, -1, 0, 1, 2)
    assert sequence_available_time_offsets(18, 20, 2) == (-2, -1, 0, 1)
    assert sequence_available_time_offsets(19, 20, 2) == (-2, -1, 0)


def test_movie_start_is_not_false_interior_track_start():
    graph = _track_graph(
        (0, 1, 2),
        available_time_offsets=(0, 1, 2),
    )
    status = graph["temporal_status"][0]

    assert float(status[0]) == pytest.approx(1.0)
    assert float(status[1]) == pytest.approx(0.0)
    assert float(status[2]) == pytest.approx(0.0)
    assert float(status[TEMPORAL_STATUS_PAST_CONTEXT_COLUMN]) == pytest.approx(0.0)
    assert float(status[TEMPORAL_STATUS_FUTURE_CONTEXT_COLUMN]) == pytest.approx(1.0)

    current_row = int(torch.nonzero(graph["node_time_offset"] == 0)[0].item())
    assert float(graph["graph_x"][current_row, 28]) == pytest.approx(0.0)


def test_movie_end_is_not_false_interior_track_end():
    graph = _track_graph(
        (-2, -1, 0),
        available_time_offsets=(-2, -1, 0),
    )
    status = graph["temporal_status"][0]

    assert float(status[0]) == pytest.approx(1.0)
    assert float(status[1]) == pytest.approx(0.0)
    assert float(status[2]) == pytest.approx(0.0)
    assert float(status[TEMPORAL_STATUS_PAST_CONTEXT_COLUMN]) == pytest.approx(1.0)
    assert float(status[TEMPORAL_STATUS_FUTURE_CONTEXT_COLUMN]) == pytest.approx(0.0)

    current_row = int(torch.nonzero(graph["node_time_offset"] == 0)[0].item())
    assert float(graph["graph_x"][current_row, 29]) == pytest.approx(0.0)


def test_real_interior_start_stays_detectable_when_past_frames_exist():
    graph = _track_graph(
        (0, 1, 2),
        available_time_offsets=(-2, -1, 0, 1, 2),
    )
    status = graph["temporal_status"][0]

    assert float(status[0]) == pytest.approx(0.0)
    assert float(status[1]) == pytest.approx(1.0)
    assert float(status[TEMPORAL_STATUS_PAST_CONTEXT_COLUMN]) == pytest.approx(1.0)
    assert float(status[TEMPORAL_STATUS_FUTURE_CONTEXT_COLUMN]) == pytest.approx(1.0)

    current_row = int(torch.nonzero(graph["node_time_offset"] == 0)[0].item())
    assert float(graph["graph_x"][current_row, 28]) == pytest.approx(1.0)


def test_one_sided_context_is_a_valid_explicit_contract():
    future_only = _track_graph(
        (0, 1, 2),
        available_time_offsets=(0, 1, 2),
    )
    past_only = _track_graph(
        (-2, -1, 0),
        available_time_offsets=(-2, -1, 0),
    )

    assert future_only["graph_x"].shape[1] == 32
    assert past_only["graph_x"].shape[1] == 32
    assert future_only["temporal_status"].shape[1] == 10
    assert past_only["temporal_status"].shape[1] == 10


def test_unavailable_middle_frame_does_not_manufacture_track_gap():
    graph = _track_graph(
        (-2, 0, 1, 2),
        available_time_offsets=(-2, 0, 1, 2),
    )
    status = graph["temporal_status"][0]

    assert float(status[3]) == pytest.approx(0.0)
    assert float(status[0]) == pytest.approx(1.0)


def test_missing_detection_in_available_middle_frame_is_a_real_gap():
    graph = _track_graph(
        (-2, 0, 1, 2),
        available_time_offsets=(-2, -1, 0, 1, 2),
    )
    status = graph["temporal_status"][0]

    assert float(status[3]) == pytest.approx(1.0)
    assert float(status[0]) == pytest.approx(0.0)


def test_records_outside_declared_availability_are_rejected():
    with pytest.raises(ValueError, match="outside available_time_offsets"):
        _track_graph(
            (-1, 0, 1),
            available_time_offsets=(0, 1),
        )


def test_available_time_offsets_are_validated():
    with pytest.raises(ValueError, match="must contain 0"):
        _track_graph(
            (1, 2),
            available_time_offsets=(1, 2),
        )

    with pytest.raises(ValueError, match="outside temporal_radius"):
        _track_graph(
            (0, 1),
            available_time_offsets=(0, 1, 3),
        )
