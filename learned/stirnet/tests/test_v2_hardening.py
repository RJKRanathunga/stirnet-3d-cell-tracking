from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch
from scipy import ndimage as ndi

from learned.stirnet import (
    GeometryForwardOutput,
    RAGCriterion,
    SpatialForwardOutput,
    StirNet,
    build_geometry_targets,
)
from learned.stirnet.data.sample_builder import build_cached_sample
from learned.stirnet.model.geometry.targets import _boundaries, _soft_interface_target
from learned.stirnet.model.geometry import targets as geometry_targets_module
from learned.stirnet.model.refinement.local_refiner import LocalGeometryRefiner
from learned.stirnet.model.types import RAGState, RefinementRequest, TemporalInput
from learned.stirnet.training import Trainer, build_instance_targets
from learned.stirnet.training.metrics import instance_iou_matrix
from learned.stirnet.training.trainer import gt_labels_from_batch

from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def _slow_geometry_reference(labels, spacing, dref, cfg):
    labels = np.asarray(labels, dtype=np.int64)
    foreground = labels > 0
    surface_interface, separator_interface = _boundaries(labels)
    surface = _soft_interface_target(
        surface_interface, spacing, cfg.surface_target_sigma_um
    )
    separator = _soft_interface_target(
        separator_interface, spacing, cfg.separator_target_sigma_um
    )
    sdf_um = np.zeros(labels.shape, np.float32)
    if (~foreground).any():
        background = ndi.distance_transform_edt(~foreground, sampling=spacing)
        sdf_um[~foreground] = -background[~foreground]
    flow = np.zeros((3, *labels.shape), np.float32)
    offset = np.zeros_like(flow)
    seed = np.zeros(labels.shape, np.float32)
    coordinates = np.indices(labels.shape, dtype=np.float32)
    for instance_id in np.unique(labels[labels > 0]):
        mask = labels == instance_id
        distance = ndi.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
        sdf_um[mask] = distance[mask]
        gradients = np.gradient(distance, *spacing, edge_order=1)
        norm = np.sqrt(sum(gradient.astype(np.float32) ** 2 for gradient in gradients)) + 1e-6
        for axis, gradient in enumerate(gradients):
            flow[axis, mask] = (gradient / norm)[mask]
        voxels = np.argwhere(mask).astype(np.float32)
        center_um = voxels.mean(axis=0) * spacing
        for axis in range(3):
            coordinate_um = coordinates[axis] * spacing[axis]
            offset[axis, mask] = (center_um[axis] - coordinate_um[mask]) / dref
        maximum = float(distance[mask].max())
        seed[mask] = distance[mask] / maximum
    return {
        "foreground": foreground.astype(np.float32)[None],
        "surface": surface[None],
        "separator": separator[None],
        "sdf": np.clip(sdf_um / dref, -cfg.sdf_clip_dref, cfg.sdf_clip_dref)[None],
        "flow": flow,
        "centroid_offset": offset,
        "seed": seed[None],
    }


def _three_node_rag() -> RAGState:
    labels = torch.zeros((3, 6, 6), dtype=torch.long)
    labels[:, :, :2] = 1
    labels[:, :, 2:4] = 2
    labels[:, :, 4:] = 3
    return RAGState(
        node_features=torch.randn(3, 4),
        node_embeddings=torch.zeros(3, 4),
        node_batch=torch.zeros(3, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2, 3]),
        node_centroid_um=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        node_volume_voxels=torch.full((3,), 36.0),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_features=torch.zeros(2, 2),
        edge_embeddings=torch.zeros(2, 4),
        spatial_edge_logits=torch.zeros(2),
        edge_batch=torch.zeros(2, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, 3]),
    )


def test_stage_execution_really_skips_disabled_modules():
    model = StirNet(small_model_config()).eval()
    batch = synthetic_batch(temporal=True)
    with (
        patch.object(model.watershed, "forward", wraps=model.watershed.forward) as watershed,
        patch.object(model.rag_network, "forward", wraps=model.rag_network.forward) as rag,
        patch.object(model.temporal_encoder, "forward", wraps=model.temporal_encoder.forward) as temporal,
    ):
        output = model(
            batch["spatial_inputs"], batch["spacing_um"], batch["dref_um"],
            execution_stage="geometry",
        )
        assert isinstance(output, GeometryForwardOutput)
        assert watershed.call_count == rag.call_count == temporal.call_count == 0
    with (
        patch.object(model.temporal_encoder, "forward", wraps=model.temporal_encoder.forward) as temporal,
        patch.object(model.local_refiner, "forward", wraps=model.local_refiner.forward) as refiner,
        patch.object(model.instance_tokenizer, "forward", wraps=model.instance_tokenizer.forward) as tokenizer,
    ):
        output = model(
            batch["spatial_inputs"], batch["spacing_um"], batch["dref_um"],
            execution_stage="spatial",
        )
        assert isinstance(output, SpatialForwardOutput)
        assert temporal.call_count == refiner.call_count == tokenizer.call_count == 0
    with patch.object(
        model.local_refiner, "forward", wraps=model.local_refiner.forward
    ) as refiner:
        model(
            batch["spatial_inputs"], batch["spacing_um"], batch["dref_um"],
            execution_stage="temporal",
            graph_x=batch["graph_x"],
            graph_edge_index=batch["graph_edge_index"],
            graph_edge_attr=batch["graph_edge_attr"],
            tracklet_id=batch["tracklet_id"],
            temporal_ref_um=batch["temporal_ref_um"],
            temporal_status=batch["temporal_status"],
            temporal_batch=batch["temporal_batch"],
        )
        assert refiner.call_count == 0


def test_local_geometry_targets_match_slow_reference():
    cfg = small_model_config().geometry
    labels = np.zeros((7, 15, 17), np.int64)
    labels[1:5, 2:8, 2:8] = 1
    labels[2:6, 8:13, 8:15] = 2
    spacing = np.asarray([1.7, 0.4, 0.3], np.float32)
    reference = _slow_geometry_reference(labels, spacing, 4.2, cfg)
    target = build_geometry_targets(
        torch.from_numpy(labels)[None],
        torch.from_numpy(spacing)[None],
        torch.tensor([4.2]),
        sdf_clip_dref=cfg.sdf_clip_dref,
        surface_target_sigma_um=cfg.surface_target_sigma_um,
        separator_target_sigma_um=cfg.separator_target_sigma_um,
    )
    for name, expected in reference.items():
        actual = getattr(target, name)[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_per_instance_edt_is_bounding_box_local():
    labels = torch.zeros((1, 20, 30, 30), dtype=torch.long)
    labels[:, 2:5, 3:7, 3:7] = 1
    labels[:, 12:16, 20:25, 21:26] = 2
    shapes = []
    original = geometry_targets_module.ndi.distance_transform_edt

    def tracked(array, *args, **kwargs):
        shapes.append(tuple(array.shape))
        return original(array, *args, **kwargs)

    with patch.object(
        geometry_targets_module.ndi,
        "distance_transform_edt",
        side_effect=tracked,
    ):
        build_geometry_targets(
            labels,
            torch.tensor([[2.0, 0.3, 0.3]]),
            torch.tensor([4.0]),
        )
    full_shape = tuple(labels.shape[-3:])
    # Surface band + background SDF are allowed volume-wide; neither per-cell
    # transform is allowed to have the native volume shape.
    assert sum(shape == full_shape for shape in shapes) <= 2
    assert sum(shape != full_shape for shape in shapes) == 2


def test_precomputed_geometry_targets_are_reused():
    trainer = Trainer(
        StirNet(small_model_config()),
        fixed_stage_training("geometry_bootstrap"),
        device="cpu",
    )
    batch = synthetic_batch(temporal=False)
    labels = gt_labels_from_batch(batch)
    with patch.object(
        trainer.criterion,
        "build_geometry_targets",
        wraps=trainer.criterion.build_geometry_targets,
    ) as builder:
        target = trainer.prepare_geometry_targets(batch, gt_labels=labels)
        trainer.train_step(
            batch, gt_labels=labels, precomputed_geometry_targets=target
        )
        trainer.train_step(
            batch, gt_labels=labels, precomputed_geometry_targets=target
        )
        assert builder.call_count == 1


def test_geometry_criterion_does_not_build_rag_or_instance_targets():
    trainer = Trainer(
        StirNet(small_model_config()),
        fixed_stage_training("geometry_bootstrap"),
        device="cpu",
    )
    batch = synthetic_batch(temporal=False)
    output = trainer.model(
        batch["spatial_inputs"],
        batch["spacing_um"],
        batch["dref_um"],
        execution_stage="geometry",
    )
    with patch.object(
        trainer.criterion.rag,
        "forward",
        side_effect=AssertionError("RAG target path must remain disabled"),
    ):
        losses = trainer.criterion(
            output,
            gt_labels_from_batch(batch),
            batch["spacing_um"],
            batch["dref_um"],
            stage="geometry_bootstrap",
        )
    assert torch.isfinite(losses["loss"])


def test_vectorized_instance_targets_and_iou_match_slow_references():
    generator = np.random.default_rng(42)
    pred = generator.integers(0, 5, size=(4, 7, 8), dtype=np.int64)
    gt = generator.integers(0, 4, size=pred.shape, dtype=np.int64)
    matrix, pids, gids = instance_iou_matrix(pred, gt)
    slow = np.zeros_like(matrix)
    for i, pid in enumerate(pids):
        for j, gid in enumerate(gids):
            intersection = np.count_nonzero((pred == pid) & (gt == gid))
            union = np.count_nonzero((pred == pid) | (gt == gid))
            slow[i, j] = intersection / union
    np.testing.assert_allclose(matrix, slow)

    predicted = torch.from_numpy(pred)
    truth = torch.from_numpy(gt)
    targets = build_instance_targets(
        [predicted], truth[None],
        existence_min_precision=0.2,
        existence_min_gt_coverage=0.1,
        split_min_pred_fraction=0.1,
        split_min_gt_coverage=0.2,
        device=torch.device("cpu"),
    )
    slow_exist, slow_split = [], []
    for pid in range(1, int(predicted.max()) + 1):
        mask = predicted == pid
        intersections = []
        for gid in torch.unique(truth[truth > 0]):
            intersection = (mask & (truth == gid)).sum().float()
            intersections.append(
                (intersection / mask.sum().clamp_min(1), intersection / (truth == gid).sum())
            )
        slow_exist.append(float(any(a >= 0.2 and b >= 0.1 for a, b in intersections)))
        slow_split.append(float(sum(a >= 0.1 and b >= 0.2 for a, b in intersections) >= 2))
    assert targets.existence.tolist() == slow_exist
    assert targets.split.tolist() == slow_split


def test_rag_targets_are_vectorized_and_impure_edges_are_invalid():
    rag = _three_node_rag()
    gt = torch.zeros((1, 3, 6, 6), dtype=torch.long)
    gt[:, :, :, :3] = 1
    gt[:, :, :, 3:] = 2
    criterion = RAGCriterion(small_model_config().partition)
    targets = criterion.build_targets(rag, gt)
    assert targets.dominant_gt.tolist() == [1, 1, 2]
    assert torch.allclose(targets.node_purity, torch.tensor([1.0, 0.5, 1.0]))
    assert targets.valid.tolist() == [False, False]
    metrics = criterion(rag, gt)
    assert metrics["rag_valid_edge_fraction"] == 0
    assert metrics["rag_impure_node_fraction"] > 0


def test_refined_explicit_geometry_changes_temporal_observation():
    model = StirNet(small_model_config()).eval()
    batch = synthetic_batch(temporal=True)
    geometry_output = model(
        batch["spatial_inputs"], batch["spacing_um"], batch["dref_um"],
        execution_stage="geometry",
    )
    data = TemporalInput(
        graph_x=batch["graph_x"],
        graph_edge_index=batch["graph_edge_index"],
        graph_edge_attr=batch["graph_edge_attr"],
        tracklet_id=batch["tracklet_id"],
        temporal_ref_um=batch["temporal_ref_um"],
        temporal_status=batch["temporal_status"],
        temporal_batch=batch["temporal_batch"],
    )
    base = model.temporal_encoder(data)
    initial = model.temporal_observer(
        base,
        geometry_output.decoded_spatial,
        geometry_output.geometry,
        geometry_output.spatial_pyramid.spacings_um,
        batch["spacing_um"],
        batch["dref_um"],
    )
    changed_geometry = replace(
        geometry_output.geometry,
        separator_logits=geometry_output.geometry.separator_logits + 5.0,
    )
    changed = model.temporal_observer(
        base,
        geometry_output.decoded_spatial,
        changed_geometry,
        geometry_output.spatial_pyramid.spacings_um,
        batch["spacing_um"],
        batch["dref_um"],
    )
    assert not torch.allclose(initial.tokens, changed.tokens)


def test_refinement_encodes_temporal_graph_once():
    model = StirNet(small_model_config()).train()
    batch = synthetic_batch(temporal=True)

    def teacher(instances, rag, temporal, reasoning, dref):
        return [
            RefinementRequest(
                batch_index=0,
                center_um=temporal.ref_um[0],
                query_token=temporal.tokens[0],
                kind="recovery",
                source_index=0,
                score=3.0,
                selection_source="teacher",
            )
        ]

    with patch.object(
        model.temporal_encoder, "forward", wraps=model.temporal_encoder.forward
    ) as encoder:
        output = model(
            batch["spatial_inputs"], batch["spacing_um"], batch["dref_um"],
            execution_stage="refinement",
            teacher_request_builder=teacher,
            graph_x=batch["graph_x"],
            graph_edge_index=batch["graph_edge_index"],
            graph_edge_attr=batch["graph_edge_attr"],
            tracklet_id=batch["tracklet_id"],
            temporal_ref_um=batch["temporal_ref_um"],
            temporal_status=batch["temporal_status"],
            temporal_batch=batch["temporal_batch"],
        )
        assert output.refinement is not None
        assert output.refinement.applied_count >= 1
        assert encoder.call_count == 1


def test_local_refiner_relative_physical_coordinates_affect_residuals():
    cfg = small_model_config()
    refiner = LocalGeometryRefiner(
        cfg.refinement, cfg.spatial, cfg.geometry.hidden_channels, cfg.instances.d_model
    ).eval()
    shape = (5, 9, 9)
    crop = (slice(0, 5), slice(0, 9), slice(0, 9))
    spacing = torch.tensor([2.0, 0.2, 0.2])
    first = refiner._relative_coordinates(
        shape, crop, spacing, torch.zeros(3), torch.tensor(4.0),
        device=torch.device("cpu"), dtype=torch.float32,
    )
    second = refiner._relative_coordinates(
        shape, crop, spacing, torch.tensor([0.5, 0.0, 0.0]), torch.tensor(4.0),
        device=torch.device("cpu"), dtype=torch.float32,
    )
    base_channels = cfg.spatial.channels[0] + cfg.spatial.in_channels + 11
    assert refiner.spatial[0].in_channels == base_channels + 4
    base = torch.zeros((base_channels, *shape))
    token = torch.randn(cfg.instances.d_model)
    residual_a = refiner._decode_crop(torch.cat([base, first]), token)
    residual_b = refiner._decode_crop(torch.cat([base, second]), token)
    assert not torch.allclose(first, second)
    assert not torch.allclose(residual_a, residual_b)


def test_physical_boundary_band_is_isotropic_in_micrometres():
    interface = np.zeros((5, 31, 31), dtype=bool)
    interface[2, 15, 15] = True
    band = _soft_interface_target(
        interface, np.asarray([2.0, 0.2, 0.2]), sigma_um=1.0
    )
    assert np.isclose(band[3, 15, 15], band[2, 25, 15], atol=1e-6)
    assert band[2, 16, 15] > band[3, 15, 15]


def test_model_dref_is_current_segmentation_only():
    current = np.zeros((5, 12, 12), np.int64)
    current[1:4, 2:7, 2:7] = 1
    gt_a = current.copy()
    gt_b = np.zeros_like(current)
    gt_b[:, 1:11, 1:11] = 9
    raw = np.zeros(current.shape, np.float32)
    sample_a = build_cached_sample(
        raw, current, gt_a, (1.5, 0.4, 0.4), normalize_raw=False
    )
    sample_b = build_cached_sample(
        raw, current, gt_b, (1.5, 0.4, 0.4), normalize_raw=False
    )
    assert sample_a["dref_um"] == sample_b["dref_um"]
    assert sample_a["metadata"]["model_dref_source"] == "current_segmentation"
