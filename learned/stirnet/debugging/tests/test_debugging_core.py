from __future__ import annotations

import numpy as np

from learned.stirnet.debugging.core import DebugConfig, DebugTrace
from learned.stirnet.debugging.io import load_debug_trace, save_debug_trace
from learned.stirnet.debugging.probes.queries import select_queries_for_deep_probe


def test_debug_config_presets():
    assert not DebugConfig.light().capture_native_masks
    assert DebugConfig.deep().capture_native_masks
    assert DebugConfig.deep().capture_scene_arrays


def test_query_selection_respects_explicit_and_limit():
    rows = [
        {
            "query": 1,
            "matched": True,
            "center_error_um": 8.0,
            "layer3_coarse_dice": 0.1,
            "query_type": "primary",
            "survives_final_exist": True,
            "layer3_exist_prob": 0.9,
        },
        {
            "query": 2,
            "matched": True,
            "center_error_um": 2.0,
            "layer3_coarse_dice": 0.8,
            "query_type": "temporal",
            "survives_final_exist": True,
            "layer3_exist_prob": 0.8,
        },
    ]
    selected = select_queries_for_deep_probe(
        rows,
        explicit=(2,),
        max_queries=2,
        worst_center=1,
        worst_coarse_dice=1,
        top_temporal=1,
        top_split=0,
    )
    assert selected == [2, 1]


def test_trace_roundtrip(tmp_path):
    trace = DebugTrace(
        metadata={"step": 25},
        tables={"queries": [{"query": 3, "matched": True}]},
        arrays={"scene/raw": np.arange(8, dtype=np.float32).reshape(2, 2, 2)},
    )
    save_debug_trace(trace, tmp_path)
    loaded = load_debug_trace(tmp_path)
    assert loaded.metadata["step"] == 25
    assert loaded.tables["queries"][0]["query"] == 3
    np.testing.assert_array_equal(loaded.arrays["scene/raw"], trace.arrays["scene/raw"])
