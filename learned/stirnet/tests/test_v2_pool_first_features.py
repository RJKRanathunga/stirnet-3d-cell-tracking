from __future__ import annotations

import torch
import torch.nn.functional as F

from learned.stirnet.model.instances.tokenizer import InstanceTokenizer
from learned.stirnet.model.partition.rag import RAGBuilder
from learned.stirnet.model.types import (
    GeometryState,
    PartitionState,
    RAGState,
    SpatialDecodeState,
)
from learned.stirnet.model.utils.tensor_ops import pool_labeled_features

from .conftest import small_model_config


def _geometry(shape: tuple[int, int, int], *, requires_grad: bool = False) -> GeometryState:
    def field(channels: int) -> torch.Tensor:
        return torch.randn((1, channels, *shape), requires_grad=requires_grad)

    cfg = small_model_config()
    return GeometryState(
        foreground_logits=field(1),
        surface_logits=field(1),
        separator_logits=field(1),
        sdf=field(1),
        flow=field(3),
        centroid_offset=field(3),
        seed_logits=field(1),
        features=field(cfg.geometry.hidden_channels),
    )


def _two_labels(shape: tuple[int, int, int]) -> torch.Tensor:
    labels = torch.zeros(shape, dtype=torch.long)
    labels[:, :, : shape[2] // 2] = 1
    labels[:, :, shape[2] // 2 :] = 2
    return labels


def test_rag_pool_first_mean_matches_dense_projection_reference():
    torch.manual_seed(3)
    cfg = small_model_config()
    builder = RAGBuilder(cfg.partition, cfg.spatial)
    shape = (3, 5, 6)
    labels = _two_labels(shape)
    d0 = torch.randn((1, cfg.spatial.channels[0], *shape), requires_grad=True)
    spatial_inputs = torch.randn((1, cfg.spatial.in_channels, *shape))
    geometry = _geometry(shape, requires_grad=True)

    dense_projected = F.conv3d(
        d0, builder.node_projection.weight[:, :, None, None, None]
    )
    reference, _ = pool_labeled_features(dense_projected[0], labels)
    projection_inputs: list[tuple[int, ...]] = []
    handle = builder.node_projection.register_forward_hook(
        lambda _, args, __: projection_inputs.append(tuple(args[0].shape))
    )
    try:
        actual = builder(
            [labels],
            d0,
            spatial_inputs,
            geometry,
            torch.tensor([[1.5, 0.4, 0.3]]),
            torch.tensor([3.0]),
        )
    finally:
        handle.remove()

    width = cfg.partition.node_feature_channels
    torch.testing.assert_close(
        actual.node_features[:, :width], reference[:, :width], atol=1e-6, rtol=1e-6
    )
    assert projection_inputs == [(2, cfg.spatial.channels[0])] * 2
    assert torch.isfinite(actual.node_features).all()
    actual.node_features.square().mean().backward()
    assert builder.node_projection.weight.grad is not None
    assert d0.grad is not None and d0.grad.abs().sum() > 0


def test_rag_and_tokenizer_strictly_load_legacy_conv_projection_weights():
    cfg = small_model_config()
    for module, names in (
        (RAGBuilder(cfg.partition, cfg.spatial), ("node_projection",)),
        (InstanceTokenizer(cfg.instances, cfg.spatial), ("proj_d0", "proj_d1", "proj_d2")),
    ):
        state = module.state_dict()
        expected = {}
        for name in names:
            key = f"{name}.weight"
            expected[key] = state[key].clone()
            state[key] = state[key][..., None, None, None]
        restored = type(module)(
            cfg.partition if isinstance(module, RAGBuilder) else cfg.instances,
            cfg.spatial,
        )
        restored.load_state_dict(state, strict=True)
        for key, value in expected.items():
            torch.testing.assert_close(restored.state_dict()[key], value)


def test_instance_tokenizer_pools_before_projection_and_backpropagates():
    torch.manual_seed(9)
    cfg = small_model_config()
    tokenizer = InstanceTokenizer(cfg.instances, cfg.spatial)
    shape = (4, 8, 8)
    labels = _two_labels(shape)
    decoded = SpatialDecodeState(
        d0=torch.randn((1, cfg.spatial.channels[0], *shape), requires_grad=True),
        d1=torch.randn((1, cfg.spatial.channels[1], 4, 4, 4), requires_grad=True),
        d2=torch.randn((1, cfg.spatial.channels[2], 2, 2, 2), requires_grad=True),
    )
    geometry = _geometry(shape, requires_grad=True)
    rag = RAGState(
        node_features=torch.zeros((2, 1)),
        node_embeddings=torch.zeros((2, 1)),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2]),
        node_centroid_um=torch.zeros((2, 3)),
        node_volume_voxels=torch.ones(2),
        edge_index=torch.tensor([[0], [1]]),
        edge_features=torch.zeros((1, 1)),
        edge_embeddings=torch.zeros((1, 1)),
        spatial_edge_logits=torch.zeros(1),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, 2]),
    )
    partition = PartitionState(
        labels=[labels],
        node_component=torch.tensor([0, 1]),
        node_component_global=torch.tensor([0, 1]),
        component_count_per_batch=torch.tensor([2]),
        edge_logits=torch.zeros(1),
    )
    projection_inputs: dict[str, list[tuple[int, ...]]] = {}
    handles = []
    for name in ("proj_d0", "proj_d1", "proj_d2"):
        projection_inputs[name] = []
        handles.append(
            getattr(tokenizer, name).register_forward_hook(
                lambda _, args, __, key=name: projection_inputs[key].append(
                    tuple(args[0].shape)
                )
            )
        )
    try:
        result = tokenizer(
            partition,
            rag,
            decoded,
            geometry,
            torch.tensor([[1.5, 0.4, 0.4]]),
            torch.tensor([3.0]),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert result.tokens.shape == (2, cfg.instances.d_model)
    assert result.local_ids.tolist() == [1, 2]
    assert result.node_to_instance.tolist() == [0, 1]
    assert all(
        shapes == [(2, channels), (2, channels)]
        for shapes, channels in zip(
            projection_inputs.values(), cfg.spatial.channels[:3]
        )
    )
    loss = result.tokens.square().mean() + result.quality_logits.square().mean()
    loss.backward()
    for name in ("proj_d0", "proj_d1", "proj_d2"):
        gradient = getattr(tokenizer, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    for feature in (decoded.d0, decoded.d1, decoded.d2):
        assert feature.grad is not None and feature.grad.abs().sum() > 0

