from __future__ import annotations

from dataclasses import replace

import torch

from learned.stirnet import StirNet
from learned.stirnet.model.temporal.observer import TemporalSpatialObserver

from .conftest import small_model_config, synthetic_batch
from .test_v2_observer_sampling import _observer_inputs


def test_full_output_drops_hidden_geometry_and_reuses_compact_observations():
    config = small_model_config()
    config.refinement.split_threshold = -1.0
    config.refinement.recovery_threshold = -1.0
    model = StirNet(config).eval()
    calls = {name: 0 for name in ("d1_proj", "d2_proj", "geometry_proj", "geometry_field_proj")}
    handles = [
        getattr(model.temporal_observer, name).register_forward_hook(
            lambda _, __, ___, key=name: calls.__setitem__(key, calls[key] + 1)
        )
        for name in calls
    ]
    encoder_calls = 0

    def count_encoder(*_):
        nonlocal encoder_calls
        encoder_calls += 1

    handles.append(model.temporal_encoder.register_forward_hook(count_encoder))
    try:
        batch = synthetic_batch(temporal=True)
        with torch.no_grad():
            output = model(
                batch["spatial_inputs"],
                batch["spacing_um"],
                batch["dref_um"],
                graph_x=batch["graph_x"],
                graph_edge_index=batch["graph_edge_index"],
                graph_edge_attr=batch["graph_edge_attr"],
                tracklet_id=batch["tracklet_id"],
                temporal_ref_um=batch["temporal_ref_um"],
                temporal_status=batch["temporal_status"],
                temporal_batch=batch["temporal_batch"],
                node_instance_grid=batch["node_instance_grid"],
                node_history_valid=batch["node_history_valid"],
                execution_stage="refinement",
                apply_existence_filter=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert output.refinement is not None and output.refinement.applied_count > 0
    assert output.geometry.features is None
    assert output.initial_geometry.features is None
    assert calls["d1_proj"] == 1
    assert calls["d2_proj"] == 1
    assert calls["geometry_proj"] == 1
    assert calls["geometry_field_proj"] == 2
    assert encoder_calls == 1


def test_cached_hidden_observation_still_responds_to_refined_explicit_geometry():
    cfg, temporal, decoded, geometry, spacings, spacing, dref = _observer_inputs()
    observer = TemporalSpatialObserver(cfg.temporal, cfg.spatial, cfg.geometry).eval()
    cache = observer.build_cache(
        temporal, decoded, geometry, spacings, spacing, dref
    )
    initial = observer(
        temporal,
        decoded,
        replace(geometry, features=None),
        spacings,
        spacing,
        dref,
        cache=cache,
    )
    refined_geometry = replace(
        geometry,
        foreground_logits=geometry.foreground_logits + 2.0,
        sdf=geometry.sdf - 1.5,
        features=None,
    )
    refined = observer(
        temporal,
        decoded,
        refined_geometry,
        spacings,
        spacing,
        dref,
        cache=cache,
    )
    assert not torch.allclose(initial.tokens, refined.tokens)
    assert torch.isfinite(refined.tokens).all()

