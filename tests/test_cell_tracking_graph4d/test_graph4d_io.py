from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pandas as pd

from src.io import PipelinePaths, load_stage7_outputs, save_tracking_result

from .helpers import FourDGraphConfig, GraphTrackingConfig, moving_frames, run_cell_tracking


def test_graph4d_io_round_trip_and_legacy_loading(tmp_path: Path) -> None:
    result = run_cell_tracking(
        moving_frames(),
        sample_id="io",
        graph_config=GraphTrackingConfig(mode="shadow", algorithm="windowed_4d"),
    )
    paths = PipelinePaths(tmp_path)
    save_tracking_result(result, paths.stage7_tracking)
    loaded = load_stage7_outputs(paths=paths)
    pd.testing.assert_frame_equal(
        loaded.graph4d_temporal_edges,
        result.graph4d_temporal_edges,
        check_dtype=False,
    )
    graph4d_files = tuple(paths.stage7_tracking.glob("graph4d_*.csv"))
    assert len(graph4d_files) == 7
    for path in graph4d_files:
        path.unlink()
    legacy = load_stage7_outputs(paths=paths)
    assert legacy.graph4d_temporal_edges.empty
    assert legacy.graph4d_component_summary.empty


def test_detailed_npz_artifacts_are_opt_in(tmp_path: Path) -> None:
    result = run_cell_tracking(
        moving_frames(),
        sample_id="debug-io",
        graph_config=GraphTrackingConfig(
            mode="shadow",
            algorithm="windowed_4d",
            four_d=replace(
                FourDGraphConfig(), save_detailed_debug_artifacts=True
            ),
        ),
    )
    paths = PipelinePaths(tmp_path)
    save_tracking_result(result, paths.stage7_tracking)
    assert {path.name for path in paths.stage7_tracking.glob("graph4d_*.npz")} == {
        "graph4d_spatial_edges.npz",
        "graph4d_pair_factors.npz",
        "graph4d_relation_histories.npz",
    }
