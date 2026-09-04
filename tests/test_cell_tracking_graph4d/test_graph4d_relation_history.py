from __future__ import annotations

import numpy as np
from importlib import import_module

build_observations = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.observations"
).build_observations
build_relation_histories = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.relation_history"
).build_relation_histories
build_spatial_relations = import_module(
    "legacy.classical_pipeline.tracking.graph_tracking.four_d.spatial_relations"
).build_spatial_relations

from .helpers import FourDGraphConfig, moving_frames, run_cell_tracking


def test_persistent_identity_history_beats_transient_rank() -> None:
    frames = moving_frames(4)
    provisional = run_cell_tracking(frames, sample_id="history")
    config = FourDGraphConfig()
    observations = build_observations(
        time_frames=frames,
        provisional_tracks=provisional.tracks,
        spatial_shape_zyx=np.asarray([64, 256, 256]),
        voxel_size_zyx_um=np.asarray([1.625, 0.40625, 0.40625]),
        config=config,
    )
    graphs = build_spatial_relations(observations, config)
    histories = build_relation_histories(
        observations=observations, spatial_graphs=graphs, config=config
    )
    assert histories
    assert max(history.persistence_count for history in histories.values()) >= 2
    assert all(0.0 <= history.confidence <= 1.0 for history in histories.values())
