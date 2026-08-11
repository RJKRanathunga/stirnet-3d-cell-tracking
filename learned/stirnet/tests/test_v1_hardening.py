from __future__ import annotations

from dataclasses import fields, is_dataclass

import numpy as np
import pytest
import torch

from learned.stirnet import RefinementCriterion, StirNet, StirNetConfig
from learned.stirnet.data.graph_builder import DetectionRecord, build_temporal_graph
from learned.stirnet.data.targets import build_gt_targets, extract_instance_metadata
from learned.stirnet.model.attention import LocalPhysicalCrossAttention
from learned.stirnet.model.blocks import AxisFactorizedConv
from learned.stirnet.model.config import CoReasoningConfig, QueryConfig
from learned.stirnet.model.graph_encoder import DetectionGraphEncoder, segment_softmax
from learned.stirnet.model.matcher import (
    HungarianMatcher3D,
    build_cost_matrix,
    target_masks_at_shape,
)
from learned.stirnet.model.query_builder import InstanceQueryBuilder
from learned.stirnet.model.query_decoder import QueryCrossAttention, _cap_feature_tokens
from learned.stirnet.model.types import TemporalState


def _geometry_fixture() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    labels = np.zeros((10, 14, 18), dtype=np.int32)
    labels[0, 0, 0] = 1                       # one voxel
    labels[2, 2, 2:16] = 2                   # line-like
    labels[4, 3:12, 3:15] = 3               # one-voxel-thick plane
    labels[6:9, 5:9, 2:17] = 4              # strongly elongated volume
    raw = np.linspace(0, 1, labels.size, dtype=np.float32).reshape(labels.shape)
    return labels, raw, (2.0, 0.25, 0.25)


def _all_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif is_dataclass(value):
        for item in fields(value):
            yield from _all_tensors(getattr(value, item.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_tensors(item)


def test_degenerate_geometry_features_are_finite_and_scaled() -> None:
    labels, raw, spacing = _geometry_fixture()
    metadata = extract_instance_metadata(labels, raw, spacing, dref_um=8.0)
    assert torch.isfinite(metadata.features).all()
    assert float(metadata.features[:, 7:9].min()) >= 0
    assert float(metadata.features[:, 7:9].max()) < 20

    records = []
    for row, instance_id in enumerate(metadata.ids.tolist()):
        feature = metadata.features[row]
        records.append(
            DetectionRecord(
                node_id=instance_id,
                time_offset=0,
                position_um=tuple(metadata.centroids_um[row].tolist()),
                physical_volume_um3=float((labels == instance_id).sum() * np.prod(spacing)),
                bbox_um=tuple((feature[1:4] * 8.0).tolist()),
                pca_axes_um=tuple((feature[4:7] * 8.0).tolist()),
                elongation=float(feature[7]),
                flatness=float(feature[8]),
            )
        )
    graph = build_temporal_graph(records, [], dref_um=8.0)
    assert torch.isfinite(graph["graph_x"]).all()
    assert float(graph["graph_x"][:, 11:13].max()) < 20


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_softmax_cpu_is_finite_and_normalized(dtype: torch.dtype) -> None:
    scores = torch.tensor(
        [[1000.0, -1000.0], [999.0, -999.0], [2.0, 3.0], [4.0, 1.0], [-2.0, 8.0], [7.0, -7.0]],
        dtype=dtype,
    )
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    result = segment_softmax(scores, index, 3)
    assert result.dtype == dtype
    assert torch.isfinite(result.float()).all()
    sums = torch.zeros(3, 2).index_add_(0, index, result.float())
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_segment_softmax_cuda_amp_is_finite_and_normalized(dtype: torch.dtype) -> None:
    scores = torch.randn(32, 4, device="cuda", dtype=dtype) * 20
    index = torch.arange(32, device="cuda") % 5
    result = segment_softmax(scores, index, 5)
    assert result.dtype == dtype
    assert torch.isfinite(result.float()).all()
    sums = torch.zeros(5, 4, device="cuda").index_add_(0, index, result.float())
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=3e-3, rtol=3e-3)


def test_axis_factorized_conv_matches_three_branch_reference() -> None:
    torch.manual_seed(3)
    module = AxisFactorizedConv(4, acquisition_dim=6)
    x = torch.randn(2, 4, 3, 5, 7, requires_grad=True)
    acquisition = torch.randn(2, 6)
    gates = torch.sigmoid(module.gate(acquisition)).view(2, 3, 4, 1, 1, 1)
    expected = module.fuse(
        module.conv_z(x) * gates[:, 0]
        + module.conv_y(x) * gates[:, 1]
        + module.conv_x(x) * gates[:, 2]
    )
    actual = module(x, acquisition)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def _attention_config(**chunks: int) -> CoReasoningConfig:
    return CoReasoningConfig(
        d_model=8,
        heads=2,
        dropout=0.0,
        position_bias_hidden=4,
        **chunks,
    )


def test_chunked_local_attention_is_chunk_boundary_invariant_and_local() -> None:
    torch.manual_seed(4)
    large = LocalPhysicalCrossAttention(
        _attention_config(
            temporal_query_chunk_size=32,
            spatial_query_chunk_size=32,
            spatial_key_chunk_size=32,
        )
    )
    small = LocalPhysicalCrossAttention(
        _attention_config(
            temporal_query_chunk_size=1,
            spatial_query_chunk_size=2,
            spatial_key_chunk_size=3,
        )
    )
    small.load_state_dict(large.state_dict())
    spatial = torch.randn(1, 7, 8)
    positions = torch.tensor(
        [[[-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )
    temporal = torch.randn(3, 8)
    refs = torch.tensor([[-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    salience = torch.tensor([[0.0], [0.3], [0.8]])
    reliability = torch.tensor([[0.9], [0.7], [0.5]])
    batch = torch.zeros(3, dtype=torch.long)
    dref = torch.tensor([1.0])

    temporal_large = large.temporal_reads_spatial(
        temporal, refs, salience, spatial, positions, batch, dref
    )
    temporal_small = small.temporal_reads_spatial(
        temporal, refs, salience, spatial, positions, batch, dref
    )
    spatial_large = large.spatial_reads_temporal(
        spatial, positions, temporal, refs, salience, reliability, batch, dref
    )
    spatial_small = small.spatial_reads_temporal(
        spatial, positions, temporal, refs, salience, reliability, batch, dref
    )
    torch.testing.assert_close(temporal_small, temporal_large, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(spatial_small, spatial_large, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(temporal_small).all()
    assert torch.isfinite(spatial_small).all()

    far_refs = refs + 1000
    assert torch.count_nonzero(
        small.temporal_reads_spatial(
            temporal, far_refs, salience, spatial, positions, batch, dref
        )
    ) == 0
    assert torch.count_nonzero(
        small.spatial_reads_temporal(
            spatial, positions, temporal, far_refs, salience, reliability, batch, dref
        )
    ) == 0


def test_checkpointed_local_attention_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(5)
    cfg = _attention_config(
        temporal_query_chunk_size=2,
        spatial_query_chunk_size=3,
        spatial_key_chunk_size=2,
    )
    direct = LocalPhysicalCrossAttention(cfg).train()
    checkpointed = LocalPhysicalCrossAttention(
        cfg, activation_checkpointing=True
    ).train()
    checkpointed.load_state_dict(direct.state_dict())

    spatial_direct = torch.randn(1, 7, 8, requires_grad=True)
    temporal_direct = torch.randn(4, 8, requires_grad=True)
    spatial_checkpointed = spatial_direct.detach().clone().requires_grad_()
    temporal_checkpointed = temporal_direct.detach().clone().requires_grad_()
    positions = torch.randn(1, 7, 3)
    refs = torch.randn(4, 3)
    salience = torch.rand(4, 1)
    reliability = torch.rand(4, 1).clamp_min(0.1)
    batch = torch.zeros(4, dtype=torch.long)
    dref = torch.tensor([2.0])

    def run(
        module: LocalPhysicalCrossAttention,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal_output = module.temporal_reads_spatial(
            temporal, refs, salience, spatial, positions, batch, dref
        )
        spatial_output = module.spatial_reads_temporal(
            spatial,
            positions,
            temporal,
            refs,
            salience,
            reliability,
            batch,
            dref,
        )
        return temporal_output, spatial_output

    direct_outputs = run(direct, spatial_direct, temporal_direct)
    checkpointed_outputs = run(
        checkpointed, spatial_checkpointed, temporal_checkpointed
    )
    for actual, expected in zip(checkpointed_outputs, direct_outputs):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    sum(output.square().mean() for output in direct_outputs).backward()
    sum(output.square().mean() for output in checkpointed_outputs).backward()
    torch.testing.assert_close(
        spatial_checkpointed.grad, spatial_direct.grad, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        temporal_checkpointed.grad, temporal_direct.grad, atol=2e-6, rtol=2e-5
    )
    direct_grads = dict(direct.named_parameters())
    checkpointed_grads = dict(checkpointed.named_parameters())
    assert direct_grads.keys() == checkpointed_grads.keys()
    for name, parameter in direct_grads.items():
        assert parameter.grad is not None, name
        assert checkpointed_grads[name].grad is not None, name
        torch.testing.assert_close(
            checkpointed_grads[name].grad,
            parameter.grad,
            atol=3e-6,
            rtol=3e-5,
        )


def test_query_cross_attention_cpu_autocast_preserves_destination_dtype() -> None:
    cfg = StirNetConfig().decoder
    cfg.d_model = 8
    cfg.heads = 2
    cfg.dropout = 0.0
    attention = QueryCrossAttention(cfg)
    q = torch.randn(1, 3, 8)
    spatial = torch.randn(1, 5, 8)
    positions = torch.randn(1, 5, 3)
    refs = torch.randn(1, 3, 3)
    support = torch.ones(1, 3, 5, dtype=torch.bool)
    padding = torch.zeros(1, 3, dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = attention(q, spatial, positions, refs, support, padding, torch.ones(1))
    assert output.dtype == q.dtype
    assert torch.isfinite(output).all()


def test_hungarian_cost_is_fp32_and_finite_for_saturated_logits() -> None:
    label_map = torch.zeros(4, 5, 6, dtype=torch.int32)
    label_map[:, :3, :3] = 11
    label_map[:, 3:, 3:] = 29
    target = {
        "ids": torch.tensor([11, 29]),
        "label_map": label_map,
        "centers_cellscale": torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    gt = target_masks_at_shape(target, (2, 3, 3), torch.device("cpu"))
    logits = torch.tensor(
        [[[[100.0] * 3] * 3] * 2, [[[-100.0] * 3] * 3] * 2, [[[80.0] * 3] * 3] * 2],
        dtype=torch.float16,
    )
    exist = torch.tensor([100.0, -100.0, 80.0], dtype=torch.float16)
    centers = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]], dtype=torch.float16)
    cost = build_cost_matrix(exist, logits, centers, gt, target["centers_cellscale"])
    assert cost.dtype == torch.float32
    assert torch.isfinite(cost).all()
    matcher = HungarianMatcher3D()
    result = matcher(
        {
            "exist_logits": exist[None],
            "coarse_mask_logits": logits[None],
            "centers_cellscale": centers[None],
        },
        torch.zeros(1, 3, dtype=torch.bool),
        [target],
    )
    assert result[0].pred_indices.numel() == 2


def _empty_temporal(batch_size: int, d_model: int) -> TemporalState:
    return TemporalState(
        tokens=torch.zeros(0, d_model),
        ref_um=torch.zeros(0, 3),
        ref_cellscale=torch.zeros(0, 3),
        salience=torch.zeros(0, 1),
        reliability=torch.zeros(0, 1),
        status=torch.zeros(0, 10),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 8),
        batch_index=torch.zeros(0, dtype=torch.long),
    )


def test_query_capacity_is_dynamic_and_padded_only_to_batch_maximum() -> None:
    cfg = QueryConfig(d_model=32, max_queries=None)
    builder = InstanceQueryBuilder(cfg, feature_channels=16)
    instance_count = 36
    temporal_count = 52
    temporal = _empty_temporal(1, 32)
    temporal = TemporalState(
        tokens=torch.randn(temporal_count, 32),
        ref_um=torch.zeros(temporal_count, 3),
        ref_cellscale=torch.zeros(temporal_count, 3),
        salience=torch.ones(temporal_count, 1),
        reliability=torch.ones(temporal_count, 1),
        status=torch.zeros(temporal_count, 10),
        edge_index=temporal.edge_index,
        edge_attr=temporal.edge_attr,
        batch_index=torch.zeros(temporal_count, dtype=torch.long),
    )
    state = builder(
        torch.randn(1, 16, 2, 2, 2),
        torch.ones(1, 3),
        torch.zeros(1, 4, 4, 4, dtype=torch.long),
        torch.zeros(instance_count, 14),
        torch.arange(1, instance_count + 1),
        torch.zeros(instance_count, dtype=torch.long),
        torch.zeros(instance_count, 3),
        torch.ones(1),
        temporal,
    )
    assert state.embeddings.shape[1] == 2 * 36 + 52 + 8
    assert not state.padding_mask.any()


def test_token_capping_preserves_physical_grid_extent() -> None:
    feature = torch.randn(1, 4, 5, 9, 13)
    spacing = torch.tensor([[2.0, 0.5, 0.25]])
    pooled, effective_spacing = _cap_feature_tokens(feature, spacing, 80)
    original_extent = (torch.tensor(feature.shape[-3:]) - 1) * spacing[0]
    pooled_extent = (torch.tensor(pooled.shape[-3:]) - 1) * effective_spacing[0]
    torch.testing.assert_close(pooled_extent, original_extent)


def _reduced_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.spatial.channels = (4, 8, 16, 32)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.mask_dim = 8
    cfg.temporal.d_model = 32
    cfg.temporal.graph_ffn_dim = 64
    cfg.coreasoning.d_model = 32
    cfg.coreasoning.dropout = 0.0
    cfg.queries.d_model = 32
    cfg.queries.discovery_queries = 2
    cfg.decoder.d_model = 32
    cfg.decoder.ffn_dim = 64
    cfg.decoder.dropout = 0.0
    cfg.decoder.mask_dim = 8
    cfg.decoder.max_spatial_tokens = 128
    cfg.losses.native_chunk_voxels = 256
    cfg.losses.dense_chunk_voxels = 256
    return cfg


def test_reduced_width_full_forward_matching_and_label_map_loss_are_finite() -> None:
    torch.manual_seed(9)
    cfg = _reduced_config()
    model = StirNet(cfg).eval()
    assert model.query_builder.feature_proj.in_features == 2 * cfg.spatial.channels[2]
    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:5, 2:7, 2:7] = 1
    labels[3:7, 9:14, 9:14] = 2
    raw = np.random.default_rng(4).random(labels.shape, dtype=np.float32)
    metadata = extract_instance_metadata(labels, raw, (1.5, 0.5, 0.5), 6.0)
    target = build_gt_targets(labels, (1.5, 0.5, 0.5), 6.0)
    assert "label_map" in target and "masks" not in target

    graph = build_temporal_graph(
        [
            DetectionRecord(
                node_id=int(instance_id),
                time_offset=0,
                position_um=tuple(metadata.centroids_um[row].tolist()),
                physical_volume_um3=float((labels == int(instance_id)).sum() * 0.375),
                elongation=float(metadata.features[row, 7]),
                flatness=float(metadata.features[row, 8]),
            )
            for row, instance_id in enumerate(metadata.ids)
        ],
        [],
        dref_um=6.0,
    )
    observed: dict[str, bool] = {}
    handles = []
    for name in ("graph_encoder", "tracklet_pooler", "cr1", "cr2", "query_builder", "query_decoder"):
        module = model.get_submodule(name)

        def hook(_module, _inputs, output, module_name=name):
            tensors = list(_all_tensors(output))
            observed[module_name] = bool(tensors) and all(
                bool(torch.isfinite(tensor.float()).all()) for tensor in tensors if tensor.numel()
            )

        handles.append(module.register_forward_hook(hook))

    spatial = torch.stack(
        [
            torch.as_tensor(raw),
            torch.as_tensor(labels > 0).float(),
            torch.zeros(labels.shape),
            torch.zeros(labels.shape),
            torch.zeros(labels.shape),
        ]
    )[None]
    output = model(
        spatial,
        torch.as_tensor(labels)[None],
        torch.tensor([[1.5, 0.5, 0.5]]),
        torch.tensor([6.0]),
        metadata.features,
        metadata.ids,
        torch.zeros(len(metadata.ids), dtype=torch.long),
        metadata.centroids_um,
        graph["graph_x"],
        graph["graph_edge_index"],
        graph["graph_edge_attr"],
        graph["tracklet_id"],
        graph["temporal_ref_um"],
        graph["temporal_status"],
        graph["hypothesis_edge_index"],
        graph["hypothesis_edge_attr"],
        torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long),
    )
    for handle in handles:
        handle.remove()
    assert observed and all(observed.values())
    for tensor in (
        output.exist_logits,
        output.centers_cellscale,
        output.coarse_mask_logits,
        output.query_embeddings,
    ):
        assert torch.isfinite(tensor.float()).all()
    losses = RefinementCriterion(cfg.losses, cfg.queries)(output, [target])
    assert losses and all(torch.isfinite(value.float()) for value in losses.values())


def test_mismatched_representation_widths_fail_early() -> None:
    cfg = StirNetConfig()
    cfg.decoder.d_model = 64
    with pytest.raises(ValueError, match="shares one representation width"):
        StirNet(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_encoder_is_finite_under_cuda_fp16_autocast() -> None:
    cfg = _reduced_config().temporal
    encoder = DetectionGraphEncoder(cfg).cuda().eval()
    graph_x = torch.randn(12, 32, device="cuda")
    graph_x[:, 11:13] = torch.rand(12, 2, device="cuda") * 10
    edges = torch.stack(
        [torch.arange(12, device="cuda"), torch.arange(12, device="cuda").roll(1)]
    )
    edge_attr = torch.randn(12, 14, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = encoder(graph_x, edges, edge_attr)
    assert torch.isfinite(output.float()).all()
