from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .helpers import (
    FourDGraphConfig,
    GraphTrackingConfig,
    ambiguous_gap_frames,
    run_cell_tracking,
)


def test_exact_milp_is_sparse_deterministic_and_integral() -> None:
    result = run_cell_tracking(
        ambiguous_gap_frames(),
        sample_id="milp",
        graph_config=GraphTrackingConfig(mode="shadow", algorithm="windowed_4d"),
    )
    exact = result.graph4d_solver_diagnostics.loc[
        result.graph4d_solver_diagnostics["solver_type"] == "exact_milp"
    ]
    assert not exact.empty
    assert exact["success"].astype(bool).all()
    assert (exact["constraint_count"] > 0).all()
    assert (exact["variable_count"] > 0).all()
    assert (exact["mip_gap"].fillna(0.0) == 0.0).all()


def test_solver_failure_falls_back_only_to_provisional_component(monkeypatch) -> None:
    module = __import__(
        "importlib"
    ).import_module("src.07_cell_tracking.graph_tracking.four_d.milp_model")

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic HiGHS failure")

    monkeypatch.setattr(module, "milp", fail)
    frames = ambiguous_gap_frames()
    provisional = run_cell_tracking(frames, sample_id="milp-failure")
    result = run_cell_tracking(
        frames,
        sample_id="milp-failure",
        graph_config=GraphTrackingConfig(mode="apply", algorithm="windowed_4d"),
    )
    assert result.metadata["graph_tracking"]["solver_failures"] > 0
    assert "provisional_fallback" in set(result.graph4d_component_summary["status"])
    assert len(result.tracks) == len(provisional.tracks)


def test_all_solver_paths_receive_a_finite_time_limit(monkeypatch) -> None:
    module = __import__(
        "importlib"
    ).import_module("src.07_cell_tracking.graph_tracking.four_d.milp_model")
    observed: list[tuple[str, float]] = []
    original_milp = module.milp
    original_linprog = module.linprog

    def capture_milp(*args, **kwargs):
        observed.append(("exact", float(kwargs["options"]["time_limit"])))
        return original_milp(*args, **kwargs)

    def capture_linprog(*args, **kwargs):
        observed.append(("fallback", float(kwargs["options"]["time_limit"])))
        return original_linprog(*args, **kwargs)

    monkeypatch.setattr(module, "milp", capture_milp)
    monkeypatch.setattr(module, "linprog", capture_linprog)
    frames = ambiguous_gap_frames()
    limit = 0.75
    run_cell_tracking(
        frames,
        sample_id="exact-time-limit",
        graph_config=GraphTrackingConfig(
            mode="shadow",
            algorithm="windowed_4d",
            four_d=replace(FourDGraphConfig(), solver_time_limit_seconds=limit),
        ),
    )
    run_cell_tracking(
        frames,
        sample_id="fallback-time-limit",
        graph_config=GraphTrackingConfig(
            mode="shadow",
            algorithm="windowed_4d",
            four_d=replace(
                FourDGraphConfig(),
                maximum_exact_component_nodes=1,
                maximum_exact_component_edges=1,
                solver_time_limit_seconds=limit,
            ),
        ),
    )
    assert ("exact", limit) in observed
    assert ("fallback", limit) in observed


def test_iterative_solver_failure_restores_provisional_tracks(monkeypatch) -> None:
    module = __import__(
        "importlib"
    ).import_module("src.07_cell_tracking.graph_tracking.four_d.milp_model")

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic fallback HiGHS failure")

    monkeypatch.setattr(module, "linprog", fail)
    frames = ambiguous_gap_frames()
    provisional = run_cell_tracking(frames, sample_id="fallback-failure")
    result = run_cell_tracking(
        frames,
        sample_id="fallback-failure",
        graph_config=GraphTrackingConfig(
            mode="apply",
            algorithm="windowed_4d",
            four_d=replace(
                FourDGraphConfig(),
                maximum_exact_component_nodes=1,
                maximum_exact_component_edges=1,
            ),
        ),
    )
    assert result.metadata["graph_tracking"]["solver_failures"] > 0
    assert "provisional_fallback" in set(result.graph4d_component_summary["status"])
    identity_columns = ["frame", "cell", "track_id"]
    pd.testing.assert_frame_equal(
        result.tracks[identity_columns],
        provisional.tracks[identity_columns],
    )
