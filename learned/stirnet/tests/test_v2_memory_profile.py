from __future__ import annotations

import torch

from learned.stirnet import StirNet
from learned.stirnet.training.profiler import StageProfiler
from learned.stirnet.training.trainer import Trainer

from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def test_stage_profiler_is_opt_in_and_records_required_fields():
    disabled = StageProfiler(enabled=False, device="cpu")
    with disabled.profile("noop"):
        torch.ones(1)
    assert disabled.records == []

    profiler = StageProfiler(enabled=True, device="cpu")
    with profiler.profile("example"):
        torch.ones(4).square()
    assert len(profiler.records) == 1
    record = profiler.records[0]
    assert record.name == "example"
    assert record.elapsed_seconds >= 0
    assert record.allocated_mb_before == 0
    assert record.allocated_mb_after == 0
    assert record.reserved_mb == 0
    assert record.peak_allocated_mb == 0


def test_profiled_refinement_step_reports_coarse_stage_metrics():
    config = small_model_config()
    config.partition.rag_min_node_gt_support = 0.0
    config.refinement.split_threshold = 1.1
    config.refinement.recovery_threshold = 1.1
    training = fixed_stage_training("refinement_joint")
    training.profile_memory = True
    trainer = Trainer(StirNet(config), training, device="cpu")
    metrics = trainer.train_step(synthetic_batch(temporal=True))
    required = {
        "evidence_stem",
        "backbone_encoder",
        "backbone_decoder",
        "geometry",
        "initial_watershed",
        "initial_rag_build",
        "initial_rag_network",
        "instance_tokenizer",
        "temporal_encoder",
        "temporal_observer",
        "temporal_reasoning",
        "local_refinement",
        "criterion",
        "backward",
    }
    recorded = {record.name for record in trainer.stage_profiler.records}
    assert required.issubset(recorded)
    for name in required:
        assert f"profile_{name}_elapsed_seconds" in metrics
        assert f"profile_{name}_peak_allocated_mb" in metrics
    assert metrics["profile_criterion_calls"] == 2
    assert metrics["profile_backward_calls"] == 2

