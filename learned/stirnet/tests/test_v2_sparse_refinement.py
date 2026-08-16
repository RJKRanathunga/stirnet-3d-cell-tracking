from __future__ import annotations

import inspect
from unittest.mock import patch

import torch

from learned.stirnet.model.refinement.local_refiner import LocalGeometryRefiner
from learned.stirnet.model.types import (
    GeometryState,
    RefinedGeometryView,
    RefinementRequest,
)

from .conftest import small_model_config


def _inputs(*, requires_grad: bool = False):
    torch.manual_seed(123)
    cfg = small_model_config()
    shape = (5, 9, 9)

    def field(channels: int):
        return torch.randn((1, channels, *shape), requires_grad=requires_grad)

    geometry = GeometryState(
        foreground_logits=field(1),
        surface_logits=field(1),
        separator_logits=field(1),
        sdf=field(1),
        flow=field(3),
        centroid_offset=field(3),
        seed_logits=field(1),
        features=None,
    )
    d0 = field(cfg.spatial.channels[0])
    spatial_inputs = field(cfg.spatial.in_channels)
    request = RefinementRequest(
        batch_index=0,
        center_um=torch.zeros(3),
        query_token=torch.randn(cfg.instances.d_model),
        kind="split",
        source_index=0,
        score=1.0,
    )
    refiner = LocalGeometryRefiner(
        cfg.refinement,
        cfg.spatial,
        cfg.geometry.hidden_channels,
        cfg.instances.d_model,
    )
    return cfg, refiner, d0, spatial_inputs, geometry, request


def test_sparse_refinement_matches_dense_overlay_reference():
    cfg, refiner, d0, spatial_inputs, geometry, request = _inputs()
    raw_delta = torch.linspace(-0.7, 0.7, 11)[:, None, None, None]

    def deterministic(local, _token):
        return raw_delta.expand(11, *local.shape[-3:])

    with patch.object(refiner, "_decode_crop", side_effect=deterministic):
        result = refiner(
            d0,
            spatial_inputs,
            geometry,
            torch.tensor([[1.5, 0.4, 0.4]]),
            torch.tensor([3.0]),
            [request],
        )

    assert isinstance(result.geometry, RefinedGeometryView)
    assert len(result.geometry.delta.rois) == 1
    roi = result.geometry.delta.rois[0]
    expected_delta = cfg.refinement.residual_scale * torch.tanh(
        raw_delta.expand_as(roi.delta)
    )
    torch.testing.assert_close(roi.delta, expected_delta)
    field_rows = (
        ("foreground_logits", slice(0, 1)),
        ("surface_logits", slice(1, 2)),
        ("separator_logits", slice(2, 3)),
        ("sdf", slice(3, 4)),
        ("flow", slice(4, 7)),
        ("centroid_offset", slice(7, 10)),
        ("seed_logits", slice(10, 11)),
    )
    for name, channels in field_rows:
        base = getattr(geometry, name)
        actual = result.geometry.materialize_field(name)
        dense_reference = torch.zeros_like(base)
        zyx = roi.slices_zyx
        dense_reference[0, :, zyx[0], zyx[1], zyx[2]] = expected_delta[channels]
        torch.testing.assert_close(actual, base + dense_reference)
        torch.testing.assert_close(
            result.geometry.field_crop(name, 0, zyx),
            actual[0, :, zyx[0], zyx[1], zyx[2]],
        )


def test_sparse_refinement_stores_no_dense_clones_and_backpropagates():
    _, refiner, d0, spatial_inputs, geometry, request = _inputs(
        requires_grad=True
    )
    assert ".clone()" not in inspect.getsource(LocalGeometryRefiner.forward)
    result = refiner(
        d0,
        spatial_inputs,
        geometry,
        torch.tensor([[1.5, 0.4, 0.4]]),
        torch.tensor([3.0]),
        [request],
    )
    assert isinstance(result.geometry, RefinedGeometryView)
    assert set(vars(result.geometry)) == {"base", "delta"}
    loss = (
        result.geometry.foreground_logits.square().mean()
        + result.geometry.separator_logits.square().mean()
        + result.geometry.sdf.square().mean()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in refiner.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0

