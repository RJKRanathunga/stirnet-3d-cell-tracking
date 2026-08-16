from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from learned.stirnet.model.temporal import observer as observer_module
from learned.stirnet.model.temporal.observer import (
    TemporalSpatialObserver,
    _sample_local_grid,
)
from learned.stirnet.model.types import (
    GeometryState,
    SpatialDecodeState,
    TemporalState,
)

from .conftest import small_model_config


def _observer_inputs(*, requires_grad: bool = False):
    torch.manual_seed(91)
    cfg = small_model_config()

    def source(shape):
        return torch.randn(shape, requires_grad=requires_grad)

    decoded = SpatialDecodeState(
        d2=source((1, cfg.spatial.channels[2], 3, 4, 4)),
        d1=source((1, cfg.spatial.channels[1], 4, 7, 7)),
        d0=source((1, cfg.spatial.channels[0], 6, 12, 12)),
    )
    native = (6, 12, 12)
    geometry = GeometryState(
        foreground_logits=source((1, 1, *native)),
        surface_logits=source((1, 1, *native)),
        separator_logits=source((1, 1, *native)),
        sdf=source((1, 1, *native)),
        flow=source((1, 3, *native)),
        centroid_offset=source((1, 3, *native)),
        seed_logits=source((1, 1, *native)),
        features=source((1, cfg.geometry.hidden_channels, *native)),
    )
    temporal = TemporalState(
        tokens=torch.randn(3, cfg.temporal.d_model),
        ref_um=torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.4, -0.4], [-0.8, -0.4, 0.4]]
        ),
        batch_index=torch.zeros(3, dtype=torch.long),
        salience=torch.ones(3, 1),
        reliability=torch.tensor([[1.0], [0.8], [0.6]]),
        status=torch.zeros(3, cfg.temporal.status_dim),
    )
    spacings = [
        torch.tensor([[1.6, 0.4, 0.4]]),
        torch.tensor([[1.6, 0.8, 0.8]]),
        torch.tensor([[3.2, 1.6, 1.6]]),
    ]
    spacing = torch.tensor([[1.6, 0.4, 0.4]])
    dref = torch.tensor([4.0])
    return cfg, temporal, decoded, geometry, spacings, spacing, dref


def _dense_projection(linear: torch.nn.Linear, feature: torch.Tensor) -> torch.Tensor:
    return F.conv3d(feature, linear.weight[:, :, None, None, None])


def _legacy_dense_observer_forward(
    observer,
    temporal,
    decoded,
    geometry,
    spatial_spacings_um,
    spacing_um,
    dref_um,
):
    """Small-tensor reference for the algebraically equivalent old flow."""
    d1 = _dense_projection(observer.d1_proj, decoded.d1)
    d2 = _dense_projection(observer.d2_proj, decoded.d2)
    probabilities = geometry.probabilities()
    explicit = torch.cat(
        [
            probabilities["foreground"],
            probabilities["surface"],
            probabilities["separator"],
            geometry.sdf,
            geometry.flow,
            geometry.centroid_offset,
            probabilities["seed"],
        ],
        dim=1,
    )
    geo = _dense_projection(
        observer.geometry_proj, geometry.features
    ) + _dense_projection(observer.geometry_field_proj, explicit)
    messages = torch.zeros_like(temporal.tokens)
    for batch_index in range(decoded.d0.shape[0]):
        idx = torch.nonzero(
            temporal.batch_index == batch_index, as_tuple=False
        ).flatten()
        if not idx.numel():
            continue
        refs = temporal.ref_um[idx]
        radius = (
            dref_um[batch_index] * observer.cfg.observation_radius_dref
        ).expand(len(idx))
        p1 = _sample_local_grid(
            d1[batch_index], refs, spatial_spacings_um[1][batch_index], radius
        )
        p2 = _sample_local_grid(
            d2[batch_index], refs, spatial_spacings_um[2][batch_index], radius
        )
        pg = _sample_local_grid(
            geo[batch_index], refs, spacing_um[batch_index], radius
        )
        messages[idx] = observer.message(torch.cat([p1, p2, pg], dim=-1))
    gate = observer.gate(
        torch.cat([temporal.tokens, messages, temporal.reliability], dim=-1)
    )
    return replace(
        temporal, tokens=observer.norm(temporal.tokens + gate * messages)
    )


def test_sample_first_observer_matches_dense_projection_contract():
    cfg, temporal, decoded, geometry, spacings, spacing, dref = _observer_inputs()
    observer = TemporalSpatialObserver(
        cfg.temporal, cfg.spatial, cfg.geometry
    ).eval()
    expected = _legacy_dense_observer_forward(
        observer, temporal, decoded, geometry, spacings, spacing, dref
    )
    actual = observer(
        temporal, decoded, geometry, spacings, spacing, dref
    )
    assert actual.tokens.shape == temporal.tokens.shape
    assert torch.isfinite(actual.tokens).all()
    torch.testing.assert_close(actual.tokens, expected.tokens, atol=2e-6, rtol=2e-6)


def test_bounded_sampler_matches_grid_sample_reference():
    torch.manual_seed(12)
    feature = torch.randn(5, 4, 7, 8)
    refs = torch.tensor([[0.0, 0.0, 0.0], [1.8, -0.7, 0.9]])
    spacing = torch.tensor([1.6, 0.4, 0.3])
    radius = torch.tensor([1.2, 0.8])
    actual = _sample_local_grid(feature, refs, spacing, radius)

    shape = feature.shape[-3:]
    extent = refs.new_tensor(
        [shape[0] - 1, shape[1] - 1, shape[2] - 1]
    ) * spacing
    unit = torch.tensor([-1.0, 0.0, 1.0])
    base = torch.stack(
        torch.meshgrid(unit, unit, unit, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    points = refs[:, None] + base[None] * radius.reshape(-1, 1, 1)
    normalized_zyx = points / (0.5 * extent[None, None]).clamp_min(1e-6)
    reference = F.grid_sample(
        feature[None],
        normalized_zyx[..., [2, 1, 0]].reshape(1, len(refs), 27, 1, 3),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, :, :, 0].permute(1, 0, 2).mean(dim=-1)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)


def test_observer_strictly_loads_legacy_pointwise_conv_weights():
    cfg = small_model_config()
    source = TemporalSpatialObserver(cfg.temporal, cfg.spatial, cfg.geometry)
    legacy_state = source.state_dict()
    projection_names = (
        "d1_proj",
        "d2_proj",
        "geometry_proj",
        "geometry_field_proj",
    )
    for name in projection_names:
        key = f"{name}.weight"
        legacy_state[key] = legacy_state[key][..., None, None, None]
    restored = TemporalSpatialObserver(cfg.temporal, cfg.spatial, cfg.geometry)
    restored.load_state_dict(legacy_state, strict=True)
    for name in projection_names:
        torch.testing.assert_close(
            getattr(restored, name).weight, getattr(source, name).weight
        )


def test_observer_projects_only_sampled_vectors_and_never_native_d_model():
    cfg, temporal, decoded, geometry, spacings, spacing, dref = _observer_inputs()
    observer = TemporalSpatialObserver(
        cfg.temporal, cfg.spatial, cfg.geometry
    ).eval()
    projection_inputs: dict[str, tuple[int, ...]] = {}
    handles = []
    for name in ("d1_proj", "d2_proj", "geometry_proj", "geometry_field_proj"):
        module = getattr(observer, name)
        handles.append(
            module.register_forward_hook(
                lambda _, args, __, projection=name: projection_inputs.__setitem__(
                    projection, tuple(args[0].shape)
                )
            )
        )
    sampled_inputs: list[tuple[int, ...]] = []
    original_sampler = observer_module._sample_local_grid

    def tracked_sampler(feature, *args, **kwargs):
        sampled_inputs.append((1, *feature.shape))
        return original_sampler(feature, *args, **kwargs)

    try:
        with patch.object(
            observer_module, "_sample_local_grid", side_effect=tracked_sampler
        ):
            output = observer(
                temporal, decoded, geometry, spacings, spacing, dref
            )
    finally:
        for handle in handles:
            handle.remove()

    assert torch.isfinite(output.tokens).all()
    assert set(projection_inputs) == {
        "d1_proj",
        "d2_proj",
        "geometry_proj",
        "geometry_field_proj",
    }
    assert all(shape[0] == len(temporal.tokens) for shape in projection_inputs.values())
    assert all(len(shape) == 2 for shape in projection_inputs.values())
    native_shape = tuple(geometry.features.shape[-3:])
    assert sampled_inputs
    assert not any(
        shape[1] == cfg.temporal.d_model and shape[-3:] == native_shape
        for shape in sampled_inputs
    )


def test_observer_backward_reaches_all_sources_and_sampled_projections():
    cfg, temporal, decoded, geometry, spacings, spacing, dref = _observer_inputs(
        requires_grad=True
    )
    observer = TemporalSpatialObserver(
        cfg.temporal, cfg.spatial, cfg.geometry
    ).train()
    output = observer(
        temporal, decoded, geometry, spacings, spacing, dref
    )
    weights = torch.linspace(
        -1.0, 1.0, output.tokens.numel(), dtype=output.tokens.dtype
    ).reshape_as(output.tokens)
    (output.tokens * weights).sum().backward()

    sources = {
        "decoded.d1": decoded.d1,
        "decoded.d2": decoded.d2,
        "geometry.features": geometry.features,
        "foreground": geometry.foreground_logits,
        "surface": geometry.surface_logits,
        "separator": geometry.separator_logits,
        "sdf": geometry.sdf,
        "flow": geometry.flow,
        "centroid_offset": geometry.centroid_offset,
        "seed": geometry.seed_logits,
    }
    for name, source in sources.items():
        assert source.grad is not None, name
        assert torch.isfinite(source.grad).all(), name
        assert source.grad.abs().sum() > 0, name
    for name in ("d1_proj", "d2_proj", "geometry_proj", "geometry_field_proj"):
        gradient = getattr(observer, name).weight.grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
