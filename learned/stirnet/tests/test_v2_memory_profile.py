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


def test_profiled_refinement_step_reports_crop_and_detached_full_stages():
    config = small_model_config()
    config.partition.rag_min_node_gt_support = 0.0
    config.refinement.split_threshold = 1.1
    config.refinement.recovery_threshold = 1.1
    training = fixed_stage_training("refinement_joint")
    training.profile_memory = True
    trainer = Trainer(StirNet(config), training, device="cpu")
    metrics = trainer.train_step(synthetic_batch(temporal=True))
    required = {
        "batch_to_device",
        "geometry_targets_prepare",
        "phase_a_crop_select",
        "phase_a_crop_prepare",
        "phase_a_evidence_stem",
        "phase_a_encoder_level0",
        "phase_a_down0",
        "phase_a_encoder_level3",
        "phase_a_decoder_up0",
        "phase_a_geometry_input_projection",
        "phase_a_geometry_trunk",
        "phase_a_geometry_heads",
        "phase_a_geometry_criterion",
        "phase_a_pre_backward",
        "phase_a_backward",
        "phase_a_release",
        "phase_b_full_evidence_stem",
        "phase_b_full_encoder_level0",
        "phase_b_full_decoder_up0",
        "phase_b_full_geometry_input_projection",
        "phase_b_full_geometry_trunk",
        "phase_b_full_geometry_heads",
        "phase_b_initial_watershed",
        "phase_b_initial_rag_build",
        "phase_b_initial_rag_network",
        "phase_b_instance_tokenizer",
        "phase_b_temporal_encoder",
        "phase_b_temporal_observer",
        "phase_b_temporal_reasoning",
        "phase_b_teacher_request_build",
        "phase_b_local_refinement",
        "phase_b_refined_partition_update",
        "phase_b_refined_rag_network",
        "phase_b_refined_tokenizer",
        "phase_b_refined_temporal_observer",
        "phase_b_refined_temporal_reasoning",
        "phase_b_criterion",
        "phase_b_pre_backward",
        "phase_b_backward",
        "gradient_unscale",
        "gradient_metrics_clip",
        "optimizer_step",
    }
    recorded = {record.name for record in trainer.stage_profiler.records}
    assert required.issubset(recorded)
    for name in required:
        assert f"profile_{name}_elapsed_seconds" in metrics
        assert f"profile_{name}_peak_allocated_mb" in metrics
    assert metrics["profile_phase_a_geometry_criterion_calls"] == 1
    assert metrics["profile_phase_b_criterion_calls"] == 1
    assert metrics["profile_phase_a_backward_calls"] == 1
    assert metrics["profile_phase_b_backward_calls"] == 1
