from __future__ import annotations

from unittest.mock import patch

import torch

from learned.stirnet import StirNet
from learned.stirnet.training.trainer import (
    Trainer,
    model_forward_from_batch,
    move_batch_to_device,
)

from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def _split_trainer() -> Trainer:
    config = small_model_config()
    config.partition.rag_min_node_gt_support = 0.0
    config.refinement.split_threshold = 1.1
    config.refinement.recovery_threshold = 1.1
    model = StirNet(config)
    with torch.no_grad():
        model.geometry_decoder.foreground.bias.fill_(4.0)
    return Trainer(model, fixed_stage_training("refinement_joint"), device="cpu")


def test_split_refinement_routes_gradients_and_steps_optimizer_once():
    trainer = _split_trainer()
    original_step = trainer.optimizer.step
    with patch.object(trainer.optimizer, "step", wraps=original_step) as step:
        metrics = trainer.train_step(synthetic_batch(temporal=True))

    assert step.call_count == 1
    assert trainer.global_step == 1
    assert trainer.refinement_stage_step == 1
    assert metrics["phase_a_grad_geometry_spatial"] > 0
    assert metrics["phase_b_accumulated_grad_geometry_spatial"] == metrics[
        "phase_a_grad_geometry_spatial"
    ]
    assert metrics["grad_partition"] > 0
    assert metrics["grad_instances"] > 0
    assert metrics["grad_temporal"] > 0
    assert metrics["grad_refinement"] > 0
    assert metrics["split_objective_abs_error"] < 1e-5


def test_split_and_monolithic_select_same_requests_and_hard_partitions():
    torch.manual_seed(77)
    config = small_model_config()
    config.refinement.split_threshold = -1.0
    config.refinement.recovery_threshold = -1.0
    model = StirNet(config).train()
    batch = move_batch_to_device(synthetic_batch(temporal=True), torch.device("cpu"))
    rng_state = torch.random.get_rng_state()
    monolithic = model_forward_from_batch(
        model,
        batch,
        execution_stage="refinement",
        apply_existence_filter=False,
    )

    torch.random.set_rng_state(rng_state)
    with torch.no_grad():
        dense = model_forward_from_batch(
            model,
            batch,
            execution_stage="geometry",
            apply_existence_filter=False,
        )
    split = model_forward_from_batch(
        model,
        batch,
        execution_stage="refinement",
        apply_existence_filter=False,
        precomputed_geometry=dense,
    )

    monolithic_requests = [
        (request.kind, request.source_index, request.batch_index)
        for request in monolithic.refinement.requests
    ]
    split_requests = [
        (request.kind, request.source_index, request.batch_index)
        for request in split.refinement.requests
    ]
    assert split_requests == monolithic_requests
    for actual, expected in zip(
        split.initial_spatial_partition.labels,
        monolithic.initial_spatial_partition.labels,
    ):
        assert torch.equal(actual, expected)
    for actual, expected in zip(split.final_labels, monolithic.final_labels):
        assert torch.equal(actual, expected)

