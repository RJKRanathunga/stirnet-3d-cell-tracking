from __future__ import annotations

import numpy as np
from importlib import import_module

build_observations = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.observations"
).build_observations
build_spatial_relations = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.spatial_relations"
).build_spatial_relations

from .helpers import FourDGraphConfig, moving_frames, run_cell_tracking


def test_spatial_edges_are_unique_and_capped() -> None:
    frames = moving_frames(1)
    provisional = run_cell_tracking(frames, sample_id="spatial")
    config = FourDGraphConfig(spatial_maximum_neighbors=2)
    observations = build_observations(
        time_frames=frames,
        provisional_tracks=provisional.tracks,
        spatial_shape_zyx=np.asarray([64, 256, 256]),
        voxel_size_zyx_um=np.asarray([1.625, 0.40625, 0.40625]),
        config=config,
    )
    graph = build_spatial_relations(observations, config)[0]
    pairs = list(zip(graph.edge_sources.tolist(), graph.edge_targets.tolist()))
    assert len(pairs) == len(set(pairs))
    assert all(source < target for source, target in pairs)
