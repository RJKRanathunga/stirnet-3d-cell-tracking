from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from learned.stirnet import RAGCriterion, StirNet, build_geometry_targets
from learned.stirnet.data.dataset import CachedStirNetDataset
from learned.stirnet.data.sample_builder import build_cached_sample
from learned.stirnet.data.targets import estimate_model_dref_um
from learned.stirnet.data.trackastra_cache import save_cache
from learned.stirnet.model.geometry.losses import GeometryCriterion
from learned.stirnet.model.types import RAGState
from learned.stirnet.training import Trainer
from learned.stirnet.training.checkpoint import save_checkpoint
from learned.stirnet.training.criterion import teacher_forcing_fraction

from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def test_sdf_supervision_uses_unclipped_distance_and_keeps_interior():
    labels = torch.zeros((1, 9, 17, 17), dtype=torch.long)
    labels[:, 2:7, 5:12, 5:12] = 1
    target = build_geometry_targets(
        labels,
        torch.tensor([[2.0, 0.5, 0.5]]),
        torch.tensor([2.0]),
        sdf_clip_dref=2.5,
        sdf_supervision_radius_dref=1.25,
    )
    valid = target.sdf_valid[0, 0]
    assert not valid[0, 0, 0]
    assert valid[2, 4, 8]
    assert valid[4, 8, 8]
    assert valid[labels[0] > 0].all()
    assert valid.float().mean() < 1.0


def test_sdf_loss_ignores_predictions_only_in_invalid_background():
    cfg = small_model_config()
    cfg.geometry.sdf_supervision_radius_dref = 0.10
    model = StirNet(cfg).eval()
    batch = synthetic_batch(temporal=False)
    output = model(
        batch["spatial_inputs"],
        batch["spacing_um"],
        batch["dref_um"],
        execution_stage="geometry",
    )
    labels = batch["targets"][0]["label_map"][None]
    target = build_geometry_targets(
        labels,
        batch["spacing_um"],
        batch["dref_um"],
        sdf_clip_dref=cfg.geometry.sdf_clip_dref,
        sdf_supervision_radius_dref=cfg.geometry.sdf_supervision_radius_dref,
        surface_target_sigma_um=cfg.geometry.surface_target_sigma_um,
        separator_target_sigma_um=cfg.geometry.separator_target_sigma_um,
    )
    invalid = ~target.sdf_valid
    assert invalid.any()
    criterion = GeometryCriterion(cfg.geometry)
    base = criterion(
        output.geometry, target, batch["spacing_um"], batch["dref_um"]
    )["sdf"]
    changed = replace(
        output.geometry,
        sdf=output.geometry.sdf + invalid.to(output.geometry.sdf) * 100.0,
    )
    changed_loss = criterion(
        changed, target, batch["spacing_um"], batch["dref_um"]
    )["sdf"]
    torch.testing.assert_close(changed_loss, base)


def test_masked_sdf_loss_backpropagates_to_sdf_head():
    cfg = small_model_config()
    cfg.geometry.sdf_supervision_radius_dref = 0.25
    model = StirNet(cfg)
    trainer = Trainer(
        model, fixed_stage_training("geometry_bootstrap"), device="cpu"
    )
    metrics = trainer.train_step(synthetic_batch(temporal=False))
    gradient = model.geometry_decoder.sdf.weight.grad
    assert metrics["sdf_valid_fraction"] < 1.0
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


@pytest.mark.parametrize(
    ("cell_cell", "field"),
    [(False, "surface"), (True, "separator")],
)
def test_anisotropic_interfaces_are_face_centered_in_full_target_pipeline(
    cell_cell: bool,
    field: str,
):
    spacing = torch.tensor([[2.0, 0.2, 0.2]])
    z_plane = torch.ones((1, 7, 41, 41), dtype=torch.long) if cell_cell else torch.zeros(
        (1, 7, 41, 41), dtype=torch.long
    )
    y_plane = z_plane.clone()
    z_plane[:, 3:] = 2 if cell_cell else 1
    y_plane[:, :, 20:] = 2 if cell_cell else 1
    kwargs = {
        "surface_target_sigma_um": 1.0,
        "separator_target_sigma_um": 1.0,
    }
    z_target = getattr(
        build_geometry_targets(z_plane, spacing, torch.tensor([4.0]), **kwargs),
        field,
    )[0, 0]
    y_target = getattr(
        build_geometry_targets(y_plane, spacing, torch.tensor([4.0]), **kwargs),
        field,
    )[0, 0]

    expected_z = np.exp(-0.5 * 1.0**2)
    expected_y_near = np.exp(-0.5 * 0.9**2)
    expected_y_far = np.exp(-0.5 * 1.1**2)
    assert float(z_target[2, 10, 10]) == pytest.approx(expected_z, abs=1e-6)
    assert float(y_target[2, 24, 10]) == pytest.approx(expected_y_near, abs=1e-6)
    assert float(y_target[2, 25, 10]) == pytest.approx(expected_y_far, abs=1e-6)
    # Symmetric samples around the same 1 um physical distance agree with the
    # anisotropic z sample, and neither adjacent center is mislabeled as 1.
    y_bracket = 0.5 * (y_target[2, 24, 10] + y_target[2, 25, 10])
    assert float(y_bracket) == pytest.approx(float(z_target[2, 10, 10]), abs=2e-3)
    assert float(z_target[2, 10, 10]) < 1.0


def _support_case_rag() -> tuple[RAGState, torch.Tensor]:
    supervoxels = torch.zeros((1, 4, 35), dtype=torch.long)
    for node in range(7):
        supervoxels[:, :, node * 5 : (node + 1) * 5] = node + 1
    gt = torch.zeros_like(supervoxels)
    gt[:, :, 0:1] = 1  # A: pure overlap, but only 20% GT support.
    gt[:, :, 5:10] = 2  # B: well-supported and pure.
    gt[:, :, 10:12] = 2
    gt[:, :, 12:15] = 3  # C: supported, but 60/40 mixed.
    # D (node 4): no positive overlap.
    gt[:, :, 20:30] = 4
    gt[:, :, 30:35] = 5  # E: clean same/different pairs.
    edges = torch.tensor(
        [[0, 1, 1, 1, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long
    )
    rag = RAGState(
        node_features=torch.zeros(7, 4),
        node_embeddings=torch.zeros(7, 4),
        node_batch=torch.zeros(7, dtype=torch.long),
        node_supervoxel_id=torch.arange(1, 8),
        node_centroid_um=torch.zeros(7, 3),
        node_volume_voxels=torch.full((7,), 20.0),
        edge_index=edges,
        edge_features=torch.zeros(6, 2),
        edge_embeddings=torch.zeros(6, 4),
        spatial_edge_logits=torch.zeros(6),
        edge_batch=torch.zeros(6, dtype=torch.long),
        supervoxel_labels=[supervoxels],
        node_offsets=torch.tensor([0, 7]),
    )
    return rag, gt[None]


def test_rag_targets_gate_purity_and_gt_support_independently():
    cfg = small_model_config().partition
    cfg.rag_min_node_purity = 0.8
    cfg.rag_min_node_gt_support = 0.5
    criterion = RAGCriterion(cfg)
    rag, gt = _support_case_rag()
    targets = criterion.build_targets(rag, gt)
    torch.testing.assert_close(
        targets.node_gt_support,
        torch.tensor([0.2, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        targets.node_purity,
        torch.tensor([1.0, 1.0, 0.6, 0.0, 1.0, 1.0, 1.0]),
    )
    assert targets.valid.tolist() == [False, False, False, True, True, True]
    assert targets.target.tolist()[-2:] == [1.0, 0.0]
    metrics = criterion(rag, gt)
    assert metrics["rag_mean_node_gt_support"] == pytest.approx(5.2 / 7)
    assert metrics["rag_low_support_node_fraction"] == pytest.approx(2 / 7)


def _native_payload(gt_variant: int = 1) -> tuple[dict, float]:
    labels = np.zeros((5, 12, 12), np.int64)
    labels[1:4, 3:9, 3:9] = 1
    gt = labels.copy()
    if gt_variant == 2:
        gt = np.zeros_like(labels)
        gt[:, 1:11, 1:11] = 7
    spacing = (1.5, 0.4, 0.4)
    expected = estimate_model_dref_um(labels, spacing)
    return {
        "raw": np.zeros_like(labels, dtype=np.float32),
        "instance_labels": labels,
        "gt_labels": gt,
        "spacing_um": spacing,
    }, expected


def _load_native(tmp_path, name: str, payload: dict) -> dict:
    path = tmp_path / f"{name}.pt"
    save_cache(path, payload)
    return CachedStirNetDataset([path])[0]


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("fixed_acquisition_prior", 8.0),
        ("explicit_non_gt", 9.0),
    ],
)
def test_native_dataset_accepts_only_explicit_trusted_dref_sources(
    tmp_path, source: str, value: float
):
    payload, _ = _native_payload()
    payload.update(
        {"dref_um": value, "metadata": {"model_dref_source": source}}
    )
    sample = _load_native(tmp_path, source, payload)
    assert float(sample["dref_um"]) == value
    assert sample["metadata"]["model_dref_source"] == source


def test_native_dataset_trusts_provenanced_current_scale(tmp_path):
    payload, expected = _native_payload()
    payload.update(
        {
            "dref_um": expected,
            "metadata": {"model_dref_source": "current_segmentation"},
        }
    )
    sample = _load_native(tmp_path, "trusted_current", payload)
    assert float(sample["dref_um"]) == pytest.approx(expected)


def test_native_dataset_recomputes_unprovenanced_dref_and_ignores_gt(tmp_path):
    first, expected = _native_payload(gt_variant=1)
    second, _ = _native_payload(gt_variant=2)
    first["dref_um"] = 999.0
    second["dref_um"] = 123.0
    sample_a = _load_native(tmp_path, "untrusted_a", first)
    sample_b = _load_native(tmp_path, "untrusted_b", second)
    assert float(sample_a["dref_um"]) == pytest.approx(expected)
    assert float(sample_b["dref_um"]) == pytest.approx(expected)
    assert sample_a["metadata"]["model_dref_source"] == "current_segmentation"
    assert sample_a["metadata"]["legacy_cached_dref_um"] == 999.0


def test_explicit_builder_dref_requires_explicit_non_gt_provenance():
    payload, expected = _native_payload()
    untrusted = build_cached_sample(
        payload["raw"],
        payload["instance_labels"],
        payload["gt_labels"],
        payload["spacing_um"],
        dref_um=77.0,
        normalize_raw=False,
    )
    trusted = build_cached_sample(
        payload["raw"],
        payload["instance_labels"],
        payload["gt_labels"],
        payload["spacing_um"],
        dref_um=77.0,
        model_dref_source="explicit_non_gt",
        normalize_raw=False,
    )
    assert float(untrusted["dref_um"]) == pytest.approx(expected)
    assert float(trusted["dref_um"]) == 77.0


def test_prematerialized_cache_without_provenance_is_renormalized(tmp_path):
    payload, expected = _native_payload()
    cached = build_cached_sample(
        payload["raw"],
        payload["instance_labels"],
        payload["gt_labels"],
        payload["spacing_um"],
        dref_um=99.0,
        model_dref_source="explicit_non_gt",
        normalize_raw=False,
    )
    cached.pop("metadata")
    path = tmp_path / "prematerialized_missing_metadata.pt"
    save_cache(path, cached)
    sample = CachedStirNetDataset([path])[0]
    assert float(sample["dref_um"]) == pytest.approx(expected)
    assert sample["metadata"]["model_dref_source"] == "current_segmentation"
    assert sample["metadata"]["legacy_cached_dref_um"] == 99.0


def test_refinement_teacher_forcing_uses_stage_local_step_at_high_global_step():
    training = fixed_stage_training("refinement_joint")
    training.refinement_teacher_forcing_decay_steps = 2
    trainer = Trainer(StirNet(small_model_config()), training, device="cpu")
    trainer.global_step = 5_000
    first = trainer.train_step(synthetic_batch(temporal=True))
    second = trainer.train_step(synthetic_batch(temporal=True))
    assert first["refinement_stage_step"] == 0
    assert first["refinement_teacher_forcing_fraction"] == 1.0
    assert second["refinement_stage_step"] == 1
    assert second["refinement_teacher_forcing_fraction"] == 0.5
    assert trainer.refinement_stage_step == 2


def test_refinement_progress_warm_start_and_resume_semantics(tmp_path):
    trainer = Trainer(
        StirNet(small_model_config()),
        fixed_stage_training("refinement_joint"),
        device="cpu",
    )
    non_refinement_checkpoint = {
        "global_step": 5_000,
        "extra": {
            "curriculum_stage": "instance_temporal",
            "refinement_stage_step": 900,
        },
    }
    trainer.restore_training_progress(non_refinement_checkpoint, resume=False)
    assert trainer.global_step == 5_000
    assert trainer.refinement_stage_step == 0

    refinement_checkpoint = {
        "global_step": 5_500,
        "extra": {
            "curriculum_stage": "refinement_joint",
            "refinement_stage_step": 500,
        },
    }
    trainer.restore_training_progress(refinement_checkpoint, resume=True)
    assert trainer.refinement_stage_step == 500
    metadata = trainer.checkpoint_metadata()
    assert metadata["refinement_stage_step"] == 500
    checkpoint_path = tmp_path / "progress.pt"
    save_checkpoint(
        checkpoint_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        step=trainer.global_step,
        model_config=trainer.model.cfg,
        training_config=trainer.training_config,
        extra=metadata,
    )
    payload = torch.load(checkpoint_path, weights_only=False)
    assert payload["extra"]["refinement_stage_step"] == 500


def test_teacher_forcing_schedule_is_monotonic_and_reaches_configured_end():
    training = fixed_stage_training("refinement_joint")
    training.refinement_teacher_forcing_start = 0.9
    training.refinement_teacher_forcing_end = 0.1
    training.refinement_teacher_forcing_decay_steps = 4
    fractions = [teacher_forcing_fraction(training, step) for step in range(7)]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] == 0.9
    assert fractions[4:] == pytest.approx([0.1, 0.1, 0.1])
