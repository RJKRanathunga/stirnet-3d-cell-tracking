from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from learned.stirnet.model.config import InferenceConfig, PartitionConfig
from learned.stirnet.model.partition.watershed import LearnedGeometryWatershed
from learned.stirnet.model.types import (
    GeometryForwardOutput,
    GeometryState,
    SpatialDecodeState,
    SpatialPyramid,
)
from learned.stirnet.model.utils.tensor_ops import pool_labeled_features
from learned.stirnet.inference.tiled_dense import (
    infer_dense_geometry,
    stream_tiled_label_feature_stats,
    tiled_dense_geometry,
    tiled_spatial_inference,
    tiled_temporal_inference,
)
from learned.stirnet import StirNet
from learned.stirnet.model.types import TemporalInput
from .conftest import small_model_config, synthetic_batch


class PointwiseGeometryModel(nn.Module):
    def __init__(self, inference: InferenceConfig):
        super().__init__()
        self.cfg = SimpleNamespace(
            inference=inference,
            spatial=SimpleNamespace(channels=(2, 3, 4, 5)),
        )

    def forward(self, spatial_inputs, spacing_um, dref_um, **_):
        x = spatial_inputs
        geometry = GeometryState(
            foreground_logits=2 * x[:, 0:1] - 1,
            surface_logits=x[:, 0:1] + x[:, 1:2],
            separator_logits=x[:, 1:2] - x[:, 0:1],
            sdf=x[:, 0:1] - 0.25,
            flow=torch.cat([x[:, 0:1], x[:, 1:2], x[:, 2:3]], dim=1),
            centroid_offset=torch.cat(
                [x[:, 2:3], x[:, 3:4], x[:, 4:5]], dim=1
            ),
            seed_logits=x[:, 0:1] + 0.5,
            features=x[:, :1],
        )
        d0 = x[:, :2]
        d1 = F.interpolate(x[:, :3], scale_factor=0.5, mode="nearest")
        d2 = F.interpolate(x[:, :4], scale_factor=0.25, mode="nearest")
        decoded = SpatialDecodeState(d0=d0, d1=d1, d2=d2)
        pyramid = SpatialPyramid(
            features=[d0, d1, d2, d2],
            spacings_um=[spacing_um] * 4,
            strides=[(1, 1, 1)] * 3,
        )
        return GeometryForwardOutput(geometry, pyramid, decoded)


def _config() -> InferenceConfig:
    return InferenceConfig(
        mode="tiled",
        tiled_dense_enabled=True,
        tile_shape_zyx=(4, 8, 8),
        tile_overlap_zyx=(2, 4, 4),
        tile_halo_zyx=(1, 2, 2),
        tile_batch_size=2,
    )


def _packed(geometry: GeometryState) -> torch.Tensor:
    return torch.cat(
        [
            geometry.foreground_logits,
            geometry.surface_logits,
            geometry.separator_logits,
            geometry.sdf,
            geometry.flow,
            geometry.centroid_offset,
            geometry.seed_logits,
        ],
        dim=1,
    )


def test_tiled_dense_geometry_matches_full_pointwise_prediction_without_seams():
    torch.manual_seed(44)
    config = _config()
    model = PointwiseGeometryModel(config).eval()
    spatial = torch.randn((1, 5, 6, 14, 14))
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])
    full = model(spatial, spacing, dref, execution_stage="geometry").geometry
    tiled = tiled_dense_geometry(
        model, spatial, spacing, dref, config=config
    )
    assert tiled.tile_count > 1
    assert tiled.geometry.features is None
    assert torch.all(tiled.blend_weight_sum > 0)
    torch.testing.assert_close(_packed(tiled.geometry), _packed(full))

    # Explicit seam planes are no less accurate than the rest of the field.
    difference = (_packed(tiled.geometry) - _packed(full)).abs()
    assert difference[..., 2, :, :].max() < 1e-6
    assert difference[..., :, 4, :].max() < 1e-6
    assert difference[..., :, :, 4].max() < 1e-6


def test_tiled_geometry_preserves_one_global_watershed_partition():
    config = _config()
    model = PointwiseGeometryModel(config).eval()
    spatial = torch.zeros((1, 5, 6, 14, 14))
    spatial[:, 0, 1:5, 2:6, 2:6] = 1
    spatial[:, 0, 1:5, 8:12, 8:12] = 1
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])
    full = infer_dense_geometry(
        model,
        spatial,
        spacing,
        dref,
        config=InferenceConfig(mode="full"),
    ).geometry
    tiled = tiled_dense_geometry(model, spatial, spacing, dref, config=config).geometry
    partition_cfg = PartitionConfig(
        foreground_threshold=0.5,
        seed_threshold=0.1,
        min_supervoxel_voxels=1,
    )
    watershed = LearnedGeometryWatershed(partition_cfg)
    full_labels = watershed(full, spacing, dref)[0]
    tiled_labels = watershed(tiled, spacing, dref)[0]
    assert torch.equal(tiled_labels, full_labels)


def test_second_pass_streams_features_directly_into_global_label_rows():
    torch.manual_seed(7)
    config = _config()
    model = PointwiseGeometryModel(config).eval()
    spatial = torch.randn((1, 5, 6, 14, 14))
    labels = torch.zeros((6, 14, 14), dtype=torch.long)
    labels[:, :, :7] = 1
    labels[:, :, 7:] = 2
    spacing = torch.tensor([[1.5, 0.4, 0.4]])
    dref = torch.tensor([3.0])
    tiled = tiled_dense_geometry(model, spatial, spacing, dref, config=config)
    stats = stream_tiled_label_feature_stats(
        model,
        spatial,
        spacing,
        dref,
        [labels],
        tiled.blend_weight_sum,
        config=config,
    )
    full = model(spatial, spacing, dref, execution_stage="geometry")
    expected_d0, _ = pool_labeled_features(full.decoded_spatial.d0[0], labels)
    torch.testing.assert_close(
        stats.pooled_scales[0][0], expected_d0, atol=2e-6, rtol=2e-6
    )
    assert stats.pooled_scales[1][0].shape == (2, 6)
    assert stats.pooled_scales[2][0].shape == (2, 8)
    assert all(torch.isfinite(rows[0]).all() for rows in stats.pooled_scales)


def test_tiled_spatial_path_builds_global_rag_and_tokens_from_streamed_rows():
    config = small_model_config()
    config.inference.mode = "tiled"
    config.inference.tiled_dense_enabled = True
    config.inference.tile_shape_zyx = (6, 8, 8)
    config.inference.tile_overlap_zyx = (0, 4, 4)
    config.inference.tile_halo_zyx = (0, 2, 2)
    config.inference.tile_batch_size = 2
    config.validate()
    model = StirNet(config).eval()
    batch = synthetic_batch(temporal=False)
    result = tiled_spatial_inference(
        model,
        batch["spatial_inputs"],
        batch["spacing_um"],
        batch["dref_um"],
        config=config.inference,
    )
    assert result.dense.tile_count > 1
    assert result.supervoxel_labels[0].shape == batch["spatial_inputs"].shape[-3:]
    assert result.spatial_partition.labels[0].shape == result.supervoxel_labels[0].shape
    assert result.rag.node_features.shape[0] == int(
        result.supervoxel_labels[0].max().item()
    )
    assert result.provisional_instances.tokens.shape[0] == int(
        result.spatial_partition.labels[0].max().item()
    )
    assert torch.isfinite(result.rag.node_features).all()
    assert torch.isfinite(result.provisional_instances.tokens).all()


def test_tiled_path_streams_temporal_observations_and_runs_local_refinement():
    config = small_model_config()
    config.inference.mode = "tiled"
    config.inference.tiled_dense_enabled = True
    config.inference.tile_shape_zyx = (6, 8, 8)
    config.inference.tile_overlap_zyx = (0, 4, 4)
    config.inference.tile_halo_zyx = (0, 2, 2)
    config.inference.tile_batch_size = 2
    config.refinement.split_threshold = -1.0
    config.refinement.recovery_threshold = -1.0
    config.validate()
    model = StirNet(config).eval()
    batch = synthetic_batch(temporal=True)
    temporal_input = TemporalInput(
        graph_x=batch["graph_x"],
        graph_edge_index=batch["graph_edge_index"],
        graph_edge_attr=batch["graph_edge_attr"],
        tracklet_id=batch["tracklet_id"],
        temporal_ref_um=batch["temporal_ref_um"],
        temporal_status=batch["temporal_status"],
        temporal_batch=batch["temporal_batch"],
    )
    result = tiled_temporal_inference(
        model,
        batch["spatial_inputs"],
        batch["spacing_um"],
        batch["dref_um"],
        temporal_input=temporal_input,
        config=config.inference,
        run_refinement=True,
        apply_existence_filter=False,
    )
    assert result.temporal.tokens.shape == (2, config.temporal.d_model)
    assert torch.isfinite(result.temporal.tokens).all()
    assert result.refinement is not None
    assert result.refinement.applied_count > 0
    assert result.final_labels[0].shape == batch["spatial_inputs"].shape[-3:]
    assert result.centers_um[0].shape[0] == int(result.final_labels[0].max())
