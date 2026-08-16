from __future__ import annotations

import torch
import torch.nn.functional as F

from learned.stirnet import RAGCriterion, StirNet
from learned.stirnet.model.refinement.local_refiner import LocalGeometryRefiner
from learned.stirnet.model.types import (
    GeometryState,
    InstanceState,
    RAGState,
    RefinementRequest,
    TemporalInput,
)
from learned.stirnet.training import StirNetCriterion, Trainer, build_instance_targets
from learned.stirnet.training.trainer import gt_labels_from_batch
from learned.stirnet.training.criterion import teacher_forcing_fraction

from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def _has_nonzero_gradient(module) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


def test_geometry_backward_reaches_spatial_modules():
    cfg = small_model_config()
    model = StirNet(cfg).train()
    batch = synthetic_batch(temporal=False)
    output = model(
        batch["spatial_inputs"], batch["spacing_um"], batch["dref_um"],
        execution_stage="geometry",
    )
    criterion = StirNetCriterion(cfg)
    losses = criterion(
        output,
        gt_labels_from_batch(batch),
        batch["spacing_um"],
        batch["dref_um"],
        stage="geometry_bootstrap",
    )
    losses["loss"].backward()
    assert _has_nonzero_gradient(model.evidence_stem)
    assert _has_nonzero_gradient(model.spatial_backbone)
    assert _has_nonzero_gradient(model.geometry_decoder)


def _two_node_rag(model: StirNet) -> RAGState:
    labels = torch.zeros((3, 4, 4), dtype=torch.long)
    labels[:, :, :2] = 1
    labels[:, :, 2:] = 2
    node_dim = model.rag_builder.node_feature_dim
    edge_dim = model.rag_builder.edge_feature_dim
    return RAGState(
        node_features=torch.randn(2, node_dim),
        node_embeddings=torch.zeros(2, model.cfg.partition.rag_hidden_dim),
        node_batch=torch.zeros(2, dtype=torch.long),
        node_supervoxel_id=torch.tensor([1, 2]),
        node_centroid_um=torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
        node_volume_voxels=torch.tensor([24.0, 24.0]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.randn(1, edge_dim),
        edge_embeddings=torch.zeros(1, model.cfg.partition.rag_hidden_dim),
        spatial_edge_logits=torch.zeros(1),
        edge_batch=torch.zeros(1, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, 2], dtype=torch.long),
    )


def test_spatial_rag_loss_reaches_rag_network():
    model = StirNet(small_model_config())
    rag = model.rag_network(_two_node_rag(model))
    gt = torch.ones((1, 3, 4, 4), dtype=torch.long)
    loss = RAGCriterion()(rag, gt)["rag_bce"]
    loss.backward()
    assert _has_nonzero_gradient(model.rag_network)


def test_final_rag_loss_reaches_temporal_encoder_and_reasoner():
    model = StirNet(small_model_config())
    rag = model.rag_network(_two_node_rag(model))
    instances = InstanceState(
        tokens=torch.randn(2, model.cfg.instances.d_model),
        ref_um=rag.node_centroid_um.clone(),
        batch_index=torch.zeros(2, dtype=torch.long),
        local_ids=torch.tensor([1, 2]),
        quality_logits=torch.zeros(2),
        labels=rag.supervoxel_labels,
        token_offsets=torch.tensor([0, 2]),
        node_to_instance=torch.tensor([0, 1]),
    )
    temporal_input = TemporalInput(
        graph_x=torch.randn(2, 32),
        graph_edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        graph_edge_attr=torch.randn(2, 15),
        tracklet_id=torch.tensor([0, 1]),
        temporal_ref_um=rag.node_centroid_um.clone(),
        temporal_status=torch.zeros(2, 10),
        temporal_batch=torch.zeros(2, dtype=torch.long),
        node_history_embedding=model.history_encoder(
            torch.randn(2, 4, 8, 8, 8).half(),
            torch.tensor([True, False]),
        ),
    )
    temporal = model.temporal_encoder(temporal_input)
    reasoning = model.instance_temporal(instances, rag, temporal, torch.tensor([4.0]))
    loss = F.binary_cross_entropy_with_logits(
        reasoning.final_edge_logits, torch.ones_like(reasoning.final_edge_logits)
    )
    loss.backward()
    assert _has_nonzero_gradient(model.temporal_encoder)
    assert _has_nonzero_gradient(model.instance_temporal)
    assert _has_nonzero_gradient(model.history_encoder)


def test_existence_supervision_reaches_instance_tokenizer():
    model = StirNet(small_model_config()).train()
    with torch.no_grad():
        model.geometry_decoder.foreground.bias.fill_(4.0)
    batch = synthetic_batch(temporal=False)
    output = model(
        batch["spatial_inputs"],
        batch["spacing_um"],
        batch["dref_um"],
        run_refinement=False,
        apply_existence_filter=False,
    )
    assert output.reasoning.instance_exist_logits.numel() > 0
    loss = F.binary_cross_entropy_with_logits(
        output.reasoning.instance_exist_logits,
        torch.ones_like(output.reasoning.instance_exist_logits),
    )
    loss.backward()
    assert _has_nonzero_gradient(model.instance_tokenizer)


def test_existence_and_split_target_derivation_ignores_tiny_contact():
    predicted = torch.zeros((3, 8, 8), dtype=torch.long)
    predicted[:, 1:7, 1:7] = 1
    predicted[:, 0, 0] = 2
    gt = torch.zeros_like(predicted)
    gt[:, 1:4, 1:4] = 1
    gt[:, 4:7, 4:7] = 2
    targets = build_instance_targets(
        [predicted],
        gt[None],
        existence_min_precision=0.20,
        existence_min_gt_coverage=0.10,
        split_min_pred_fraction=0.10,
        split_min_gt_coverage=0.20,
        device=torch.device("cpu"),
    )
    assert targets.existence.tolist() == [1.0, 0.0]
    assert targets.split.tolist() == [1.0, 0.0]


def test_local_refined_geometry_loss_reaches_refiner():
    cfg = small_model_config()
    refiner = LocalGeometryRefiner(
        cfg.refinement,
        cfg.spatial,
        cfg.geometry.hidden_channels,
        cfg.instances.d_model,
    )
    shape = (5, 9, 9)
    zeros1 = torch.zeros((1, 1, *shape))
    zeros3 = torch.zeros((1, 3, *shape))
    geometry = GeometryState(
        foreground_logits=zeros1.clone(),
        surface_logits=zeros1.clone(),
        separator_logits=zeros1.clone(),
        sdf=zeros1.clone(),
        flow=zeros3.clone(),
        centroid_offset=zeros3.clone(),
        seed_logits=zeros1.clone(),
        features=torch.zeros((1, cfg.geometry.hidden_channels, *shape)),
    )
    request = RefinementRequest(
        batch_index=0,
        center_um=torch.zeros(3),
        query_token=torch.randn(cfg.instances.d_model),
        kind="split",
        source_index=0,
        score=1.0,
    )
    refined = refiner(
        torch.randn(1, cfg.spatial.channels[0], *shape),
        torch.randn(1, 5, *shape),
        geometry,
        torch.tensor([[1.5, 0.5, 0.5]]),
        torch.tensor([4.0]),
        [request],
    )
    loss = (
        refined.geometry.foreground_logits.square().mean()
        + refined.geometry.separator_logits.square().mean()
        + refined.geometry.sdf.square().mean()
    )
    loss.backward()
    assert refined.applied_count == 1
    assert _has_nonzero_gradient(refiner)


def test_one_complete_training_step_is_finite():
    model_config = small_model_config()
    model_config.refinement.split_threshold = 1.1
    model_config.refinement.recovery_threshold = 1.1
    model = StirNet(model_config)
    with torch.no_grad():
        model.geometry_decoder.foreground.bias.fill_(4.0)
    trainer = Trainer(
        model,
        fixed_stage_training("refinement_joint"),
        device="cpu",
    )
    metrics = trainer.train_step(synthetic_batch(temporal=True))
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["optimizer_step_skipped"] == 0
    assert metrics["grad_geometry_spatial"] > 0
    assert metrics["grad_partition"] > 0
    assert metrics["grad_instances"] > 0
    assert metrics["grad_temporal"] > 0
    assert metrics["grad_refinement"] > 0
    assert metrics["refinement_teacher_requests"] > 0
    assert metrics["refinement_total_requests"] <= model_config.refinement.max_rois_per_batch


def test_refinement_teacher_forcing_can_be_disabled():
    model_config = small_model_config()
    model_config.refinement.split_threshold = 1.1
    model_config.refinement.recovery_threshold = 1.1
    model_config.refinement.ambiguity_logit_abs_max = -1.0
    training = fixed_stage_training("refinement_joint")
    training.refinement_teacher_forcing_start = 0.0
    training.refinement_teacher_forcing_end = 0.0
    trainer = Trainer(StirNet(model_config), training, device="cpu")
    metrics = trainer.train_step(synthetic_batch(temporal=True))
    assert metrics["refinement_teacher_forcing_fraction"] == 0.0
    assert metrics["refinement_teacher_requests"] == 0.0
    assert metrics["refinement_total_requests"] == 0.0


def test_refinement_teacher_forcing_schedule_decays_to_model_only():
    training = fixed_stage_training("refinement_joint")
    training.refinement_teacher_forcing_start = 1.0
    training.refinement_teacher_forcing_end = 0.0
    training.refinement_teacher_forcing_decay_steps = 100
    assert teacher_forcing_fraction(training, 0) == 1.0
    assert teacher_forcing_fraction(training, 50) == 0.5
    assert teacher_forcing_fraction(training, 100) == 0.0
    assert teacher_forcing_fraction(training, 1_000) == 0.0
