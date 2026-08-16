from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from learned.stirnet import StirNet
from learned.stirnet.model.geometry.targets import build_geometry_targets
from learned.stirnet.training.crops import (
    build_crop_candidate_cache,
    prepare_crop_batch,
    sample_mixed_crop_specs,
    shift_crop_centered_points,
)
from learned.stirnet.training.profiler import StageProfiler
from learned.stirnet.training.trainer import Trainer

from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def _crop_trainer(*, profile: bool = False) -> Trainer:
    config = small_model_config()
    config.partition.rag_min_node_gt_support = 0.0
    config.refinement.split_threshold = 1.1
    config.refinement.recovery_threshold = 1.1
    training = fixed_stage_training("refinement_joint")
    training.profile_memory = profile
    training.curriculum.refinement_crop_shape_zyx = (4, 8, 8)
    return Trainer(StirNet(config), training, device="cpu")


def test_refinement_dense_grad_forward_is_crop_only_and_phase_b_is_detached():
    trainer = _crop_trainer()
    calls: list[tuple[bool, tuple[int, ...]]] = []

    def record(_, args):
        calls.append((torch.is_grad_enabled(), tuple(args[0].shape[-3:])))

    handle = trainer.model.geometry_decoder.register_forward_pre_hook(record)
    try:
        metrics = trainer.train_step(synthetic_batch(temporal=True))
    finally:
        handle.remove()

    grad_shapes = [shape for grad, shape in calls if grad]
    detached_shapes = [shape for grad, shape in calls if not grad]
    assert grad_shapes == [(4, 8, 8)]
    assert detached_shapes == [(6, 12, 12)]
    assert metrics["phase_a_grad_geometry_spatial"] > 0
    assert metrics["phase_b_accumulated_grad_geometry_spatial"] == metrics[
        "phase_a_grad_geometry_spatial"
    ]


def test_phase_b_preserves_phase_a_geometry_gradients_and_trains_downstream():
    trainer = _crop_trainer()
    original = trainer._detached_refinement_phase_b_backward
    snapshots: dict[str, list[torch.Tensor | None]] = {}

    def wrapped(*args, **kwargs):
        parameters = list(trainer.model.geometry_decoder.parameters())
        snapshots["before"] = [
            None if value.grad is None else value.grad.detach().clone()
            for value in parameters
        ]
        result = original(*args, **kwargs)
        snapshots["after"] = [
            None if value.grad is None else value.grad.detach().clone()
            for value in parameters
        ]
        return result

    original_step = trainer.optimizer.step
    with patch.object(
        trainer,
        "_detached_refinement_phase_b_backward",
        side_effect=wrapped,
    ), patch.object(trainer.optimizer, "step", wraps=original_step) as step:
        metrics = trainer.train_step(synthetic_batch(temporal=True))

    assert step.call_count == 1
    assert trainer.global_step == 1
    assert trainer.refinement_stage_step == 1
    for before, after in zip(snapshots["before"], snapshots["after"]):
        if before is None:
            assert after is None
        else:
            assert torch.equal(before, after)
    assert metrics["grad_partition"] > 0
    assert metrics["grad_instances"] > 0
    assert metrics["grad_temporal"] > 0
    assert metrics["grad_refinement"] > 0


def test_crop_inputs_current_gt_targets_and_physical_coordinates_align():
    batch = synthetic_batch(temporal=False)
    gt = torch.as_tensor(batch["targets"][0]["label_map"])[None].long()
    current = batch["instance_labels"]
    targets = build_geometry_targets(
        gt, batch["spacing_um"], batch["dref_um"], device=torch.device("cpu")
    )
    cache = build_crop_candidate_cache(
        gt, current, batch["spacing_um"], batch["dref_um"]
    )
    rounds = sample_mixed_crop_specs(
        gt,
        batch["spacing_um"],
        cache,
        crop_shape_zyx=(4, 8, 8),
        crops_per_step=1,
        global_step=0,
        seed=123,
        min_foreground_fraction=0.0,
    )
    crop = prepare_crop_batch(batch, gt, rounds[0], geometry_targets=targets)
    spec = rounds[0][0]
    expected_gt = gt[0][spec.slices_zyx]
    expected_current = current[0][spec.slices_zyx]
    assert torch.equal(crop.gt_labels[0], expected_gt)
    assert torch.equal(crop.batch["instance_labels"][0], expected_current)
    assert torch.equal(
        crop.geometry_targets.foreground[0, 0], (expected_gt > 0).float()
    )
    assert crop.batch["dref_um"].item() == batch["dref_um"].item()

    point_voxel = torch.tensor([2.0, 5.0, 5.0])
    full_center = 0.5 * (torch.tensor(gt.shape[-3:]).float() - 1)
    full_point_um = (point_voxel - full_center) * batch["spacing_um"][0]
    shifted = shift_crop_centered_points(full_point_um, spec)
    crop_lower = torch.tensor(
        [axis.start for axis in spec.slices_zyx], dtype=torch.float32
    )
    crop_center = crop_lower + 0.5 * (
        torch.tensor(spec.shape_zyx).float() - 1
    )
    expected = (point_voxel - crop_center) * batch["spacing_um"][0]
    torch.testing.assert_close(shifted, expected)


def test_mixed_crop_sampling_is_deterministic_and_handles_small_empty_volumes():
    labels = torch.zeros((1, 3, 5, 5), dtype=torch.long)
    current = torch.zeros_like(labels)
    spacing = torch.tensor([[2.0, 0.4, 0.4]])
    dref = torch.tensor([4.0])
    cache = build_crop_candidate_cache(labels, current, spacing, dref)
    kwargs = dict(
        crop_shape_zyx=(32, 192, 192),
        crops_per_step=2,
        global_step=9,
        seed=77,
        min_foreground_fraction=0.5,
    )
    first = sample_mixed_crop_specs(labels, spacing, cache, **kwargs)
    second = sample_mixed_crop_specs(labels, spacing, cache, **kwargs)
    assert [row[0].slices_zyx for row in first] == [
        row[0].slices_zyx for row in second
    ]
    assert all(row[0].shape_zyx == (3, 5, 5) for row in first)
    assert all(
        row[0].candidate_type in {"background", "random"} for row in first
    )


def test_full_frame_spatial_gradient_reference_remains_explicit_debug_path():
    trainer = _crop_trainer()
    trainer.training_config.curriculum.full_frame_spatial_grad = True
    calls: list[tuple[bool, tuple[int, ...]]] = []
    handle = trainer.model.geometry_decoder.register_forward_pre_hook(
        lambda _, args: calls.append(
            (torch.is_grad_enabled(), tuple(args[0].shape[-3:]))
        )
    )
    try:
        trainer.train_step(synthetic_batch(temporal=True))
    finally:
        handle.remove()
    assert (True, (6, 12, 12)) in calls


@pytest.mark.parametrize("stage", ["geometry_bootstrap", "spatial_partition"])
def test_early_dense_stages_support_optional_crop_training(stage):
    config = small_model_config()
    training = fixed_stage_training(stage)
    training.curriculum.refinement_crop_shape_zyx = (4, 8, 8)
    training.curriculum.geometry_bootstrap_crop_enabled = True
    training.curriculum.spatial_partition_crop_enabled = True
    trainer = Trainer(StirNet(config), training, device="cpu")
    grad_shapes = []
    handle = trainer.model.geometry_decoder.register_forward_pre_hook(
        lambda _, args: grad_shapes.append(
            (torch.is_grad_enabled(), tuple(args[0].shape[-3:]))
        )
    )
    try:
        metrics = trainer.train_step(synthetic_batch(temporal=False))
    finally:
        handle.remove()
    assert grad_shapes == [(True, (4, 8, 8))]
    assert metrics["grad_geometry_spatial"] > 0
    assert trainer.global_step == 1


def test_instance_temporal_can_use_detached_full_frame_spatial_state():
    config = small_model_config()
    config.partition.rag_min_node_gt_support = 0.0
    training = fixed_stage_training("instance_temporal")
    training.curriculum.instance_temporal_detached_spatial = True
    trainer = Trainer(StirNet(config), training, device="cpu")
    calls = []
    handle = trainer.model.geometry_decoder.register_forward_pre_hook(
        lambda _, args: calls.append(torch.is_grad_enabled())
    )
    try:
        metrics = trainer.train_step(synthetic_batch(temporal=True))
    finally:
        handle.remove()
    assert calls == [False]
    assert metrics["grad_geometry_spatial"] == 0
    assert metrics["grad_partition"] > 0
    assert metrics["grad_instances"] > 0
    assert metrics["grad_temporal"] > 0


def test_streaming_profiler_flushes_success_and_failure_without_masking(
    tmp_path, capsys
):
    path = tmp_path / "memory_profile.jsonl"
    profiler = StageProfiler(True, "cpu", output_path=path)
    with profiler.phase_scope("phase_a"):
        with profiler.profile("complete"):
            torch.ones(1)
    original = RuntimeError("synthetic later failure")
    with pytest.raises(RuntimeError) as caught:
        with profiler.phase_scope("phase_b"):
            with profiler.profile("fails"):
                raise original
    assert caught.value is original
    emitted = capsys.readouterr().out
    assert "stage=phase_a_complete" in emitted
    assert "status=ok" in emitted
    assert "stage=phase_b_fails" in emitted
    assert "status=failed" in emitted
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(
        row["stage"] == "phase_a_complete" and row["status"] == "ok"
        for row in rows
    )
    assert any(
        row["stage"] == "phase_b_fails" and row["status"] == "failed"
        for row in rows
    )


def test_profiler_never_masks_simulated_oom_and_disabled_path_is_inert(tmp_path):
    profiler = StageProfiler(True, "cpu", output_path=tmp_path / "oom.jsonl")
    oom = torch.OutOfMemoryError("simulated CUDA out of memory")
    with pytest.raises(torch.OutOfMemoryError) as caught:
        with profiler.profile("oom"):
            raise oom
    assert caught.value is oom
    assert profiler.records[-1].status == "failed"

    disabled_path = tmp_path / "disabled.jsonl"
    disabled = StageProfiler(False, "cpu", output_path=disabled_path)
    with disabled.profile("noop"):
        torch.ones(1)
    assert disabled.records == []
    assert not disabled_path.exists()


def test_profiler_treats_checkpoint_early_stop_as_success(tmp_path):
    profiler = StageProfiler(True, "cpu", output_path=tmp_path / "checkpoint.jsonl")
    value = torch.randn(4, requires_grad=True)

    def profiled_square(item: torch.Tensor) -> torch.Tensor:
        with profiler.profile("checkpoint_body"):
            return item.square()

    checkpoint(profiled_square, value, use_reentrant=False).sum().backward()

    completed = [
        record
        for record in profiler.records
        if record.stage == "checkpoint_body"
    ]
    assert len(completed) == 2
    assert all(record.status == "ok" for record in completed)
    assert all(not record.error_type for record in completed)
