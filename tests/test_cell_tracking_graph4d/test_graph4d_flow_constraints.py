from __future__ import annotations

from .helpers import GraphTrackingConfig, moving_frames, run_cell_tracking


def test_selected_graph_has_one_predecessor_and_successor() -> None:
    result = run_cell_tracking(
        moving_frames(),
        sample_id="flow",
        graph_config=GraphTrackingConfig(mode="apply", algorithm="windowed_4d"),
    )
    selected = result.graph4d_temporal_edges.loc[
        result.graph4d_temporal_edges["optimized_selected"].astype(bool)
    ]
    assert selected.groupby("source_node").size().max() <= 1
    assert selected.groupby("target_node").size().max() <= 1
    assert selected["hard_safety_valid"].astype(bool).all()
