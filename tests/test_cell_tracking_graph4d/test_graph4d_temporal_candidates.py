from __future__ import annotations

from importlib import import_module

import numpy as np

from .helpers import FourDGraphConfig, moving_frames, run_cell_tracking


build_observations = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.observations"
).build_observations
build_spatial_relations = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.spatial_relations"
).build_spatial_relations
expanded_pairs = import_module(
    "src.07_cell_tracking.graph_tracking.four_d.temporal_candidates"
)._expanded_pairs


def test_graph_expansion_adds_consensus_candidate_outside_existing_set() -> None:
    frames = moving_frames(2)
    provisional = run_cell_tracking(frames, sample_id="expansion")
    config = FourDGraphConfig(graph_expansion_radius_um=2.0)
    observations = build_observations(
        time_frames=frames,
        provisional_tracks=provisional.tracks,
        spatial_shape_zyx=np.asarray([64, 256, 256]),
        voxel_size_zyx_um=np.asarray([1.625, 0.40625, 0.40625]),
        config=config,
    )
    graphs = build_spatial_relations(observations, config)
    # The two neighbour continuations vote for source node 0's target node 3.
    selected = {(1, 4), (2, 5)}
    additions = expanded_pairs(
        observations=observations,
        spatial_graphs=graphs,
        existing=set(selected),
        selected=selected,
        config=config,
    )
    assert (0, 3) in additions


def test_hard_safety_gate_is_never_bypassed() -> None:
    result = run_cell_tracking(moving_frames(), sample_id="safety")
    assert all(evidence.safety_invalid.dtype == bool for evidence in result.transition_evidence)

