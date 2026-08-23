from __future__ import annotations

import numpy as np
import torch

from learned.stirnet import StirNet
from learned.stirnet.training.crop_target_cache import StaticCropTargetCache
from learned.stirnet.training.crops import CropSpec, prepare_crop_batch
from learned.stirnet.training.raw_source import (
    materialize_raw_source_crop_batch,
    prepare_raw_source_volume_cache,
    prepare_raw_training_batch,
)
from learned.stirnet.training.trainer import Trainer
from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def _manual_raw_batch():
    shape = (8, 24, 24)
    # PyTorch does not implement boolean masked assignment for UInt16 on CPU.
    # Build this tiny synthetic fixture in a supported integer dtype, then cast
    # to the real raw-volume dtype used by production training.
    raw = torch.zeros((1, *shape), dtype=torch.int32)
    gt = torch.zeros((1, *shape), dtype=torch.long)
    gt[0, 2:6, 5:10, 5:10] = 1
    gt[0, 2:6, 14:19, 14:19] = 2
    current = gt.to(torch.int32)
    raw[0][gt[0] == 1] = 1000
    raw[0][gt[0] == 2] = 800
    raw = raw.to(torch.uint16)
    return {
        "raw_volume": raw,
        "raw_normalization_bounds": torch.tensor([[0.0, 1000.0]]),
        "instance_labels": current,
        "gt_labels": gt,
        "targets": [{"label_map": gt[0]}],
        "spacing_um": torch.tensor([[2.0, 0.4, 0.4]]),
        "dref_um": torch.tensor([4.0]),
        "source_ids": ("toy-frame",),
    }


def test_raw_frame_preprocessing_can_cache_source_state(tmp_path):
    raw = np.zeros((8, 24, 24), dtype=np.uint16)
    raw[2:6, 6:18, 6:18] = 1200
    gt = np.zeros_like(raw, dtype=np.int32)
    gt[2:6, 6:12, 6:12] = 1
    gt[2:6, 12:18, 12:18] = 2
    cache = tmp_path / "source.pt"
    first = prepare_raw_training_batch(raw, gt, (2.0, 0.4, 0.4), source_id="toy-source", source_cache_path=cache)
    second = prepare_raw_training_batch(raw, gt, (2.0, 0.4, 0.4), source_id="toy-source", source_cache_path=cache)
    assert cache.exists()
    assert "spatial_inputs" not in first
    assert first["raw_volume"].dtype == torch.uint16
    assert first["instance_labels"].shape == first["gt_labels"].shape
    assert not first["source_preprocessing_metadata"][0]["source_cache_hit"]
    assert second["source_preprocessing_metadata"][0]["source_cache_hit"]
    assert torch.equal(first["instance_labels"], second["instance_labels"])
    torch.testing.assert_close(first["raw_normalization_bounds"], second["raw_normalization_bounds"])


def test_raw_frame_preprocessing_accepts_big_endian_uint16(tmp_path):
    raw_native = np.zeros((8, 24, 24), dtype=np.uint16)
    raw_native[2:6, 6:18, 6:18] = 1200
    raw_big_endian = raw_native.astype(">u2")

    gt = np.zeros(raw_native.shape, dtype=np.int32)
    gt[2:6, 6:12, 6:12] = 1
    gt[2:6, 12:18, 12:18] = 2

    batch = prepare_raw_training_batch(
        raw_big_endian,
        gt,
        (2.0, 0.4, 0.4),
        source_id="toy-big-endian-source",
        source_cache_path=tmp_path / "source.pt",
    )

    raw_tensor = batch["raw_volume"][0]
    assert raw_tensor.dtype == torch.uint16
    assert raw_tensor.shape == raw_native.shape
    np.testing.assert_array_equal(
        raw_tensor.to(torch.int32).numpy(),
        raw_native.astype(np.int32),
    )


def test_missing_cell_rebuilds_all_source_priors_and_preserves_raw_gt():
    batch = _manual_raw_batch()
    gt = batch["gt_labels"]
    spec = CropSpec(0, (slice(0, 8), slice(0, 24), slice(0, 24)), (8, 24, 24), torch.zeros(3), "coverage", (1, 2), (), (), ())
    crop = prepare_crop_batch(batch, gt, [spec])
    materialized = materialize_raw_source_crop_batch(
        batch, crop,
        source_halo_um=0.0,
        dropout_probability=1.0,
        dropout_max_instances=1,
        dropout_seed=7,
        dropout_min_purity=0.8,
        dropout_min_gt_coverage=0.5,
    )
    dropped = materialized.batch["source_dropout_ids"][0]
    assert len(dropped) == 1
    original_mask = batch["instance_labels"][0] == dropped[0]
    assert not bool(materialized.batch["instance_labels"][0][original_mask].any())
    raw_channel = materialized.batch["spatial_inputs"][0, 0]
    assert float(raw_channel[original_mask].max()) > 0
    assert not bool(materialized.batch["spatial_inputs"][0, 1][original_mask].any())
    assert not bool(materialized.batch["spatial_inputs"][0, 2][original_mask].any())
    assert not bool(materialized.batch["spatial_inputs"][0, 4][original_mask].any())
    assert torch.equal(materialized.gt_labels, gt)


def test_real_merge_crop_is_never_synthetically_deleted():
    batch = _manual_raw_batch()
    gt = batch["gt_labels"]
    merged = batch["instance_labels"].clone()
    merged[merged == 2] = 1
    batch["instance_labels"] = merged
    spec = CropSpec(0, (slice(0, 8), slice(0, 24), slice(0, 24)), (8, 24, 24), torch.zeros(3), "merge", (1, 2), (), (), (1,))
    crop = prepare_crop_batch(batch, gt, [spec])
    materialized = materialize_raw_source_crop_batch(
        batch, crop,
        source_halo_um=0.0,
        dropout_probability=1.0,
        dropout_max_instances=1,
        dropout_seed=7,
        dropout_min_purity=0.8,
        dropout_min_gt_coverage=0.5,
    )
    assert materialized.batch["source_dropout_ids"] == ((),)
    assert torch.equal(materialized.batch["instance_labels"], merged)


def _materialize_for_test(batch, spec, *, dropout_probability: float, seed: int = 7):
    crop = prepare_crop_batch(batch, batch["gt_labels"], [spec])
    return materialize_raw_source_crop_batch(
        batch,
        crop,
        source_halo_um=0.0,
        dropout_probability=dropout_probability,
        dropout_max_instances=1,
        dropout_seed=seed,
        dropout_min_purity=0.8,
        dropout_min_gt_coverage=0.5,
    )


def test_source_ram_cache_preserves_exact_channels_without_dropout():
    baseline_batch = _manual_raw_batch()
    cached_batch = prepare_raw_source_volume_cache(
        baseline_batch,
        release_raw_volume=True,
    )
    spec = CropSpec(
        0,
        (slice(0, 8), slice(0, 24), slice(0, 24)),
        (8, 24, 24),
        torch.zeros(3),
        "coverage",
        (1, 2),
        (),
        (),
        (),
    )

    baseline = _materialize_for_test(
        baseline_batch,
        spec,
        dropout_probability=0.0,
    )
    accelerated = _materialize_for_test(
        cached_batch,
        spec,
        dropout_probability=0.0,
    )

    assert cached_batch["raw_volume"] is None
    assert "raw_normalized_volume" in cached_batch
    assert "source_edt_prior_volume" in cached_batch
    assert "current_foreground_prior" not in cached_batch
    assert "current_boundary_prior" not in cached_batch
    assert "current_marker_prior" not in cached_batch
    assert accelerated.batch["source_materialization_modes"] == ("ram_cache",)
    assert accelerated.batch["source_ram_cache_recomputed_edt_label_counts"] == (0,)
    assert torch.equal(
        baseline.batch["spatial_inputs"],
        accelerated.batch["spatial_inputs"],
    )
    assert torch.equal(
        baseline.batch["instance_labels"],
        accelerated.batch["instance_labels"],
    )


def test_source_ram_cache_preserves_exact_channels_with_missing_cell_dropout():
    baseline_batch = _manual_raw_batch()
    cached_batch = prepare_raw_source_volume_cache(
        baseline_batch,
        release_raw_volume=True,
    )
    spec = CropSpec(
        0,
        (slice(0, 8), slice(0, 24), slice(0, 24)),
        (8, 24, 24),
        torch.zeros(3),
        "coverage",
        (1, 2),
        (),
        (),
        (),
    )

    baseline = _materialize_for_test(
        baseline_batch,
        spec,
        dropout_probability=1.0,
        seed=19,
    )
    accelerated = _materialize_for_test(
        cached_batch,
        spec,
        dropout_probability=1.0,
        seed=19,
    )

    assert baseline.batch["source_dropout_ids"] == accelerated.batch["source_dropout_ids"]
    assert torch.equal(
        baseline.batch["spatial_inputs"],
        accelerated.batch["spatial_inputs"],
    )
    assert torch.equal(
        baseline.batch["instance_labels"],
        accelerated.batch["instance_labels"],
    )


def test_source_ram_cache_exactly_repairs_cell_clipped_by_halo():
    baseline_batch = _manual_raw_batch()
    cached_batch = prepare_raw_source_volume_cache(
        baseline_batch,
        release_raw_volume=True,
    )
    # This crop cuts source instance 1 in Y/X. With a zero halo, its cached
    # full-volume EDT must not be used blindly; the accelerated path repairs
    # just that source instance using the historical cropped-halo definition.
    spec = CropSpec(
        0,
        (slice(2, 6), slice(5, 8), slice(5, 8)),
        (8, 24, 24),
        torch.zeros(3),
        "coverage",
        (),
        (),
        (),
        (),
    )

    baseline = _materialize_for_test(
        baseline_batch,
        spec,
        dropout_probability=0.0,
    )
    accelerated = _materialize_for_test(
        cached_batch,
        spec,
        dropout_probability=0.0,
    )

    assert accelerated.batch["source_materialization_modes"] == ("ram_cache",)
    assert accelerated.batch["source_ram_cache_recomputed_edt_label_counts"][0] >= 1
    assert torch.equal(
        baseline.batch["spatial_inputs"],
        accelerated.batch["spatial_inputs"],
    )
    assert torch.equal(
        baseline.batch["instance_labels"],
        accelerated.batch["instance_labels"],
    )




def test_static_gt_crop_cache_is_independent_of_source_state(tmp_path):
    batch = _manual_raw_batch()
    gt = batch["gt_labels"]
    spec = CropSpec(0, (slice(0, 8), slice(0, 24), slice(0, 24)), (8, 24, 24), torch.zeros(3), "coverage", (1, 2), (), (), ())
    cache = StaticCropTargetCache(max_memory_entries=1, disk_dir=tmp_path)
    geometry = small_model_config().geometry
    first, stats1 = cache.get_or_build_batch(
        batch, gt, [spec], spacing_um=batch["spacing_um"], dref_um=batch["dref_um"],
        geometry_config=geometry, backend="scipy", gpu_min_voxels=1, halo_um=0.0,
    )
    changed = dict(batch)
    current = batch["instance_labels"].clone()
    current[current == 1] = 0
    changed["instance_labels"] = current
    second, stats2 = cache.get_or_build_batch(
        changed, gt, [spec], spacing_um=batch["spacing_um"], dref_um=batch["dref_um"],
        geometry_config=geometry, backend="scipy", gpu_min_voxels=1, halo_um=0.0,
    )
    assert stats1["misses"] == 1
    assert stats2["memory_hits"] == 1
    for name in first.__dict__:
        torch.testing.assert_close(getattr(first, name), getattr(second, name))


def test_trainer_accepts_raw_only_frame_and_builds_true_crop_batch():
    base = synthetic_batch(temporal=False)
    raw_float = base["spatial_inputs"][:, 0].clamp(0, 1)
    gt = torch.stack([torch.as_tensor(target["label_map"]).long() for target in base["targets"]])
    batch = dict(base)
    batch.pop("spatial_inputs")
    batch["raw_volume"] = (raw_float * 65535).to(torch.uint16)
    batch["raw_normalization_bounds"] = torch.tensor([[0.0, 65535.0]])
    batch["gt_labels"] = gt
    batch["source_ids"] = ("synthetic-raw-frame",)
    config = small_model_config()
    config.partition.rag_min_node_gt_support = 0.0
    training = fixed_stage_training("spatial_partition")
    training.geometry_target_backend = "scipy"
    training.crop_static_target_memory_entries = 2
    training.curriculum.refinement_crop_shape_zyx = (4, 8, 8)
    training.curriculum.refinement_crop_batch_size = 2
    training.curriculum.refinement_crop_merge_fraction = 0.5
    training.curriculum.refinement_crops_per_step = 1
    training.curriculum.refinement_crop_source_halo_um = 0.0
    training.curriculum.refinement_crop_target_halo_um = 0.0
    training.curriculum.refinement_crop_source_dropout_probability = 0.0
    trainer = Trainer(StirNet(config), training, device="cpu")
    seen_batches = []
    handle = trainer.model.geometry_decoder.register_forward_pre_hook(
        lambda _, args: seen_batches.append(int(args[0].shape[0]))
    )
    try:
        metrics = trainer.train_step(batch)
    finally:
        handle.remove()
    assert seen_batches == [2]
    assert metrics["crop_true_batch_size"] == 2
    assert metrics["phase_a_crop_effective_batch_size"] == 2
    assert (
        metrics["phase_a_static_target_misses"]
        + metrics["phase_a_static_target_memory_hits"]
        + metrics["phase_a_static_target_disk_hits"]
    ) == 2
    assert metrics["phase_a_static_target_misses"] >= 1
    assert metrics["grad_geometry_spatial"] > 0
