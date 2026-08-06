from __future__ import annotations

from dataclasses import replace

from .helpers import (
    FourDGraphConfig,
    GraphTrackingConfig,
    ambiguous_gap_frames,
    run_cell_tracking,
)


def test_oversized_component_uses_deterministic_iterative_fallback() -> None:
    four_d = replace(
        FourDGraphConfig(),
        maximum_exact_component_nodes=1,
        maximum_exact_component_edges=1,
    )
    result = run_cell_tracking(
        ambiguous_gap_frames(),
        sample_id="fallback",
        graph_config=GraphTrackingConfig(
            mode="shadow", algorithm="windowed_4d", four_d=four_d
        ),
    )
    assert not result.graph4d_component_summary.empty
    assert result.graph4d_component_summary["fallback_used"].astype(bool).any()
    assert "iterative_fallback" in set(result.graph4d_solver_diagnostics["solver_type"])
