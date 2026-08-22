from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import time
from typing import Any

import torch

from ..model import StirNet
from ..model.geometry.targets import GeometryTargets
from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .criterion import (
    StirNetCriterion,
    build_teacher_refinement_requests,
    teacher_forcing_fraction,
)
from .crops import (
    CropCandidateCache,
    build_crop_candidate_cache,
    crop_source_signature,
    prepare_crop_batch,
    sample_mixed_crop_specs,
)
from .curriculum import (
    CurriculumController,
    model_parameter_groups,
    optimizer_parameter_groups,
)
from .profiler import StageProfiler


MODEL_INPUT_KEYS = frozenset(
    {
        "spatial_inputs",
        "spacing_um",
        "dref_um",
        "spatial_padding_mask",
        "graph_x",
        "graph_edge_index",
        "graph_edge_attr",
        "hypothesis_edge_index",
        "hypothesis_edge_attr",
        "tracklet_id",
        "temporal_ref_um",
        "temporal_status",
        "temporal_batch",
        "node_instance_grid",
        "node_history_valid",
    }
)


def move_to_device(value: Any, device: torch.device | str):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move only V2 model inputs; noisy labels and large GT maps remain on CPU."""
    moved = dict(batch)
    for key in MODEL_INPUT_KEYS:
        if key in batch and batch[key] is not None:
            moved[key] = move_to_device(batch[key], device)
    return moved


def gt_labels_from_batch(batch: dict) -> torch.Tensor:
    targets = batch.get("targets")
    if not targets:
        raise ValueError("V2 training requires batch['targets'][b]['label_map']")
    labels = [torch.as_tensor(target["label_map"]).long() for target in targets]
    return torch.stack(labels)


def model_forward_from_batch(
    model: StirNet,
    batch: dict,
    *,
    use_temporal: bool = True,
    run_refinement: bool = False,
    execution_stage: str | None = None,
    teacher_request_builder=None,
    apply_existence_filter: bool = False,
    return_debug: bool = False,
    precomputed_geometry=None,
    stage_profiler=None,
):
    if execution_stage is None:
        execution_stage = (
            "refinement"
            if run_refinement
            else ("temporal" if use_temporal else "spatial")
        )
    temporal_kwargs = {}
    if use_temporal:
        temporal_kwargs = {
            key: batch.get(key)
            for key in (
                "graph_x",
                "graph_edge_index",
                "graph_edge_attr",
                "hypothesis_edge_index",
                "hypothesis_edge_attr",
                "tracklet_id",
                "temporal_ref_um",
                "temporal_status",
                "temporal_batch",
                "node_instance_grid",
                "node_history_valid",
            )
        }
    return model(
        batch["spatial_inputs"],
        batch["spacing_um"],
        batch["dref_um"],
        spatial_padding_mask=batch.get("spatial_padding_mask"),
        execution_stage=execution_stage,
        teacher_request_builder=teacher_request_builder,
        apply_existence_filter=apply_existence_filter,
        return_debug=return_debug,
        precomputed_geometry=precomputed_geometry,
        stage_profiler=stage_profiler,
        **temporal_kwargs,
    )


def _group_gradient_norms(model: StirNet) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, parameters in model_parameter_groups(model).items():
        norm = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = float(
                torch.linalg.vector_norm(parameter.grad.detach().float()).cpu()
            )
            norm = math.hypot(norm, value)
        result[f"grad_{name}"] = norm
    return result


def _refinement_fallback_metrics(output, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    refinement = getattr(output, "refinement", None)
    if refinement is None:
        values = {
            "refinement_partition_fallback": 0.0,
            "refinement_partition_fallback_reason_code": 0.0,
            "refinement_partition_fallback_batch_index": -1.0,
            "refinement_partition_fallback_box_index": -1.0,
            "refinement_partition_fallback_box_voxels": 0.0,
            "refinement_partition_fallback_core_voxels": 0.0,
            "refinement_partition_fallback_local_components": 0.0,
            "refinement_partition_fallback_old_core_labels": 0.0,
            "refinement_partition_fallback_old_shell_labels": 0.0,
            "refinement_partition_fallback_conflicting_old_labels": 0.0,
            "refinement_local_update_box_count": 0.0,
            "refinement_local_update_voxel_fraction": 0.0,
        }
    else:
        values = {
            "refinement_partition_fallback": float(refinement.partition_fallback),
            "refinement_partition_fallback_reason_code": float(
                refinement.partition_fallback_reason_code
            ),
            "refinement_partition_fallback_batch_index": float(
                refinement.partition_fallback_batch_index
            ),
            "refinement_partition_fallback_box_index": float(
                refinement.partition_fallback_box_index
            ),
            "refinement_partition_fallback_box_voxels": float(
                refinement.partition_fallback_box_voxel_count
            ),
            "refinement_partition_fallback_core_voxels": float(
                refinement.partition_fallback_core_voxel_count
            ),
            "refinement_partition_fallback_local_components": float(
                refinement.partition_fallback_local_component_count
            ),
            "refinement_partition_fallback_old_core_labels": float(
                refinement.partition_fallback_old_core_label_count
            ),
            "refinement_partition_fallback_old_shell_labels": float(
                refinement.partition_fallback_old_shell_label_count
            ),
            "refinement_partition_fallback_conflicting_old_labels": float(
                len(refinement.partition_fallback_conflicting_old_label_ids)
            ),
            "refinement_local_update_box_count": float(
                refinement.local_update_box_count
            ),
            "refinement_local_update_voxel_fraction": float(
                refinement.local_update_voxel_fraction
            ),
        }
    return {key: reference.new_tensor(value) for key, value in values.items()}


class Trainer:
    def __init__(
        self,
        model: StirNet,
        training_config: TrainingConfig | None = None,
        device=None,
    ):
        self.model = model
        self.training_config = training_config or TrainingConfig()
        self.training_config.validate()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if (
            self.device.type == "cuda"
            and self.training_config.amp_dtype == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError(
                "BF16 training was requested but this CUDA device does not support it; "
                "select amp_dtype='fp16' or 'fp32'."
            )
        self.model.to(self.device)
        self.criterion = StirNetCriterion(
            model.cfg, self.training_config.loss
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, self.training_config.lr),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )
        self.curriculum = CurriculumController(
            model,
            self.optimizer,
            self.training_config.curriculum,
            self.training_config.lr,
        )
        self.curriculum_stage = self.curriculum.apply(0)
        self.scheduler = None
        amp_dtype = self.training_config.amp_dtype
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda" and amp_dtype == "fp16"
        )
        self.global_step = 0
        self.refinement_stage_step = 0
        self.stage_profiler = StageProfiler(
            enabled=self.training_config.profile_memory,
            device=self.device,
            output_path=self.training_config.memory_profile_path,
        )
        self._crop_candidate_cache: CropCandidateCache | None = None

    def checkpoint_metadata(self) -> dict[str, int | str]:
        """Return the stage-local progress required for an exact resume."""
        return {
            "curriculum_stage": self.curriculum_stage.name,
            "refinement_stage_step": int(self.refinement_stage_step),
        }

    def restore_training_progress(
        self,
        checkpoint: dict,
        *,
        resume: bool,
    ) -> None:
        """Restore global progress, preserving local refinement only on resume."""
        self.global_step = int(checkpoint.get("global_step", 0))
        self.curriculum_stage = self.curriculum.apply(self.global_step)
        extra = dict(checkpoint.get("extra", {}))
        checkpoint_stage = extra.get("curriculum_stage", extra.get("stage"))
        if resume and checkpoint_stage == "refinement_joint":
            self.refinement_stage_step = max(
                0, int(extra.get("refinement_stage_step", 0))
            )
        else:
            self.refinement_stage_step = 0

    def _autocast(self):
        if self.device.type != "cuda" or self.training_config.amp_dtype == "fp32":
            return nullcontext()
        dtype = (
            torch.float16
            if self.training_config.amp_dtype == "fp16"
            else torch.bfloat16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def prepare_geometry_targets(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None = None,
    ) -> GeometryTargets:
        labels = gt_labels_from_batch(batch) if gt_labels is None else gt_labels
        with self.stage_profiler.profile(
            "geometry_targets_prepare", qualify=False
        ):
            return self.criterion.build_geometry_targets(
                labels,
                batch["spacing_um"],
                batch["dref_um"],
                current_labels=batch.get("instance_labels"),
                device=torch.device("cpu"),
            )

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward_and_loss(
        self,
        batch: dict,
        *,
        return_debug: bool = False,
        gt_labels: torch.Tensor | None = None,
        precomputed_geometry_targets: GeometryTargets | None = None,
    ):
        labels = gt_labels_from_batch(batch) if gt_labels is None else gt_labels
        fraction = (
            teacher_forcing_fraction(
                self.training_config, self.refinement_stage_step
            )
            if self.curriculum_stage.name == "refinement_joint"
            else 0.0
        )
        teacher_builder = None
        discrete_target_cache: dict[str, object] = {}
        if fraction > 0:
            def teacher_builder(instances, rag, temporal, reasoning, dref_um):
                return build_teacher_refinement_requests(
                    instances,
                    rag,
                    temporal,
                    reasoning,
                    dref_um,
                    gt_labels=labels,
                    spacing_um=batch["spacing_um"],
                    loss_config=self.training_config.loss,
                    rag_criterion=self.criterion.rag,
                    fraction=fraction,
                    ambiguity_logit_abs_max=self.model.cfg.refinement.ambiguity_logit_abs_max,
                    target_cache=discrete_target_cache,
                )
        self._sync_device()
        forward_started = time.perf_counter()
        output = model_forward_from_batch(
            self.model,
            batch,
            execution_stage=self.curriculum_stage.execution_stage,
            teacher_request_builder=teacher_builder,
            apply_existence_filter=False,
            return_debug=return_debug,
            stage_profiler=self.stage_profiler,
        )
        self._sync_device()
        forward_seconds = time.perf_counter() - forward_started
        target_started = time.perf_counter()
        with self.stage_profiler.profile("criterion"):
            losses = self.criterion(
                output,
                labels,
                batch["spacing_um"],
                batch["dref_um"],
                stage=self.curriculum_stage.name,
                current_labels=batch.get("instance_labels"),
                precomputed_geometry_targets=precomputed_geometry_targets,
                precomputed_discrete_targets=(
                    discrete_target_cache if discrete_target_cache else None
                ),
            )
        self._sync_device()
        target_seconds = time.perf_counter() - target_started
        return output, losses, {
            "forward_seconds": forward_seconds,
            "target_seconds": target_seconds,
            "refinement_teacher_forcing_fraction": fraction,
            "refinement_stage_step": float(self.refinement_stage_step),
        }

    def _full_frame_split_refinement_backward(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None,
        precomputed_geometry_targets: GeometryTargets | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Small-volume debug/reference path with full-frame Phase A gradients."""
        labels = gt_labels_from_batch(batch) if gt_labels is None else gt_labels
        fraction = teacher_forcing_fraction(
            self.training_config, self.refinement_stage_step
        )
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )

        forward_seconds = 0.0
        target_seconds = 0.0
        backward_seconds = 0.0

        self._sync_device()
        started = time.perf_counter()
        with self._autocast():
            base_output = model_forward_from_batch(
                self.model,
                batch,
                execution_stage="temporal",
                apply_existence_filter=False,
                stage_profiler=self.stage_profiler,
            )
        self._sync_device()
        forward_seconds += time.perf_counter() - started

        started = time.perf_counter()
        with self._autocast():
            with self.stage_profiler.profile("criterion"):
                phase_a_metrics = self.criterion(
                    base_output,
                    labels,
                    batch["spacing_um"],
                    batch["dref_um"],
                    stage="refinement_joint",
                    current_labels=batch.get("instance_labels"),
                    precomputed_geometry_targets=precomputed_geometry_targets,
                )
            phase_a_loss = self.criterion.refinement_phase_a_objective(
                phase_a_metrics
            )
        self._sync_device()
        target_seconds += time.perf_counter() - started
        if not bool(torch.isfinite(phase_a_loss)):
            raise FloatingPointError(
                f"Non-finite STIR-Net V2 Phase A loss: {phase_a_loss.detach()}"
            )
        phase_a_geometry_loss = phase_a_metrics["geometry_loss"].detach()
        phase_a_report = {
            key: value.detach() for key, value in phase_a_metrics.items()
        }
        started = time.perf_counter()
        with self.stage_profiler.profile("backward"):
            self.scaler.scale(phase_a_loss).backward()
        self._sync_device()
        backward_seconds += time.perf_counter() - started
        phase_a_gradient_norms = _group_gradient_norms(self.model)
        del base_output, phase_a_metrics

        # Replaying the pre-forward RNG state makes all stochastic base choices
        # (prior dropout and MLP dropout) identical in both contributions.
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        self._sync_device()
        started = time.perf_counter()
        with torch.no_grad(), self._autocast():
            detached_dense = model_forward_from_batch(
                self.model,
                batch,
                execution_stage="geometry",
                apply_existence_filter=False,
                stage_profiler=self.stage_profiler,
            )

        discrete_target_cache: dict[str, object] = {}

        def teacher_builder(instances, rag, temporal, reasoning, dref_um):
            return build_teacher_refinement_requests(
                instances,
                rag,
                temporal,
                reasoning,
                dref_um,
                gt_labels=labels,
                spacing_um=batch["spacing_um"],
                loss_config=self.training_config.loss,
                rag_criterion=self.criterion.rag,
                fraction=fraction,
                ambiguity_logit_abs_max=self.model.cfg.refinement.ambiguity_logit_abs_max,
                target_cache=discrete_target_cache,
            )

        with self._autocast():
            refined_output = model_forward_from_batch(
                self.model,
                batch,
                execution_stage="refinement",
                teacher_request_builder=teacher_builder if fraction > 0 else None,
                apply_existence_filter=False,
                precomputed_geometry=detached_dense,
                stage_profiler=self.stage_profiler,
            )
        del detached_dense
        self._sync_device()
        forward_seconds += time.perf_counter() - started

        started = time.perf_counter()
        with self._autocast():
            with self.stage_profiler.profile("criterion"):
                phase_b_metrics = self.criterion(
                    refined_output,
                    labels,
                    batch["spacing_um"],
                    batch["dref_um"],
                    stage="refinement_joint",
                    current_labels=batch.get("instance_labels"),
                    precomputed_geometry_targets=precomputed_geometry_targets,
                    precomputed_discrete_targets=(
                        discrete_target_cache if discrete_target_cache else None
                    ),
                )
            refinement_applied = bool(
                refined_output.refinement is not None
                and refined_output.refinement.applied_count
            )
            phase_b_loss = self.criterion.refinement_phase_b_objective(
                phase_b_metrics,
                phase_a_geometry_loss=phase_a_geometry_loss,
                refinement_applied=refinement_applied,
            )
        self._sync_device()
        target_seconds += time.perf_counter() - started
        if not bool(torch.isfinite(phase_b_loss)):
            raise FloatingPointError(
                f"Non-finite STIR-Net V2 Phase B loss: {phase_b_loss.detach()}"
            )
        started = time.perf_counter()
        with self.stage_profiler.profile("backward"):
            self.scaler.scale(phase_b_loss).backward()
        self._sync_device()
        backward_seconds += time.perf_counter() - started
        phase_b_gradient_norms = _group_gradient_norms(self.model)

        combined = phase_a_loss.detach() + phase_b_loss.detach()
        monolithic_reference = phase_b_metrics["loss"].detach()
        reported = {key: value for key, value in phase_b_metrics.items()}
        reported.update(
            _refinement_fallback_metrics(refined_output, phase_b_loss.detach())
        )
        reported.update(
            {
                "loss": combined,
                "split_phase_a_loss": phase_a_loss.detach(),
                "split_phase_b_loss": phase_b_loss.detach(),
                "split_monolithic_reference_loss": monolithic_reference,
                "split_objective_abs_error": (
                    combined - monolithic_reference
                ).abs(),
                "phase_a_geometry_loss": phase_a_report["geometry_loss"],
            }
        )
        timing = {
            "forward_seconds": forward_seconds,
            "target_seconds": target_seconds,
            "backward_seconds": backward_seconds,
            "refinement_teacher_forcing_fraction": fraction,
            "refinement_stage_step": float(self.refinement_stage_step),
            "phase_a_grad_geometry_spatial": phase_a_gradient_norms[
                "grad_geometry_spatial"
            ],
            "phase_b_accumulated_grad_geometry_spatial": phase_b_gradient_norms[
                "grad_geometry_spatial"
            ],
        }
        return reported, timing

    def _geometry_targets_for_step(
        self,
        batch: dict,
        labels: torch.Tensor,
        precomputed: GeometryTargets | None,
    ) -> GeometryTargets:
        if precomputed is not None:
            return precomputed
        return self.prepare_geometry_targets(batch, gt_labels=labels)

    def _crop_cache(
        self,
        batch: dict,
        labels: torch.Tensor,
    ) -> CropCandidateCache:
        current = batch.get("instance_labels")
        if current is None:
            current = torch.zeros_like(labels)
        signature = crop_source_signature(labels, current)
        if (
            self._crop_candidate_cache is None
            or self._crop_candidate_cache.source_signature != signature
        ):
            self._crop_candidate_cache = build_crop_candidate_cache(
                labels,
                current,
                batch["spacing_um"],
                batch["dref_um"],
            )
        return self._crop_candidate_cache

    def _crop_phase_a_backward(
        self,
        batch: dict,
        *,
        labels: torch.Tensor,
        geometry_targets: GeometryTargets,
        stage: str,
        profile_phase: str = "phase_a",
        geometry_scale: float | None = None,
        rag_scale: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Accumulate dense spatial gradients from bounded crop graphs only."""
        cfg = self.training_config.curriculum
        geometry_scale = (
            cfg.refinement_crop_geometry_weight
            if geometry_scale is None
            else geometry_scale
        )
        rag_scale = (
            cfg.refinement_crop_rag_weight
            if rag_scale is None
            else rag_scale
        )
        execution_stage = "spatial" if rag_scale > 0 else "geometry"
        criterion_stage = "spatial_partition" if rag_scale > 0 else "geometry_bootstrap"
        phase_loss = batch["spatial_inputs"].new_zeros(())
        geometry_report = batch["spatial_inputs"].new_zeros(())
        rag_report = batch["spatial_inputs"].new_zeros(())
        forward_seconds = 0.0
        target_seconds = 0.0
        backward_seconds = 0.0
        candidate_types: list[str] = []

        with self.stage_profiler.phase_scope(profile_phase):
            with self.stage_profiler.profile("crop_select"):
                cache = self._crop_cache(batch, labels)
                crop_rounds = sample_mixed_crop_specs(
                    labels,
                    batch["spacing_um"],
                    cache,
                    crop_shape_zyx=cfg.refinement_crop_shape_zyx,
                    crops_per_step=cfg.refinement_crops_per_step,
                    global_step=self.global_step,
                    seed=cfg.refinement_crop_seed,
                    min_foreground_fraction=(
                        cfg.refinement_crop_min_foreground_fraction
                    ),
                )
            for crop_index, specs in enumerate(crop_rounds):
                candidate_types.extend(spec.candidate_type for spec in specs)
                with self.stage_profiler.profile(
                    "crop_prepare",
                    metadata={
                        "crop_index": crop_index,
                        "crop_shape_zyx": list(specs[0].shape_zyx),
                        "candidate_types": [
                            spec.candidate_type for spec in specs
                        ],
                    },
                ):
                    crop = prepare_crop_batch(
                        batch,
                        labels,
                        specs,
                        geometry_targets=geometry_targets,
                    )
                started = time.perf_counter()
                with self._autocast():
                    output = model_forward_from_batch(
                        self.model,
                        crop.batch,
                        use_temporal=False,
                        execution_stage=execution_stage,
                        apply_existence_filter=False,
                        stage_profiler=self.stage_profiler,
                    )
                self._sync_device()
                forward_seconds += time.perf_counter() - started

                started = time.perf_counter()
                with self._autocast():
                    with self.stage_profiler.profile("geometry_criterion"):
                        metrics = self.criterion(
                            output,
                            crop.gt_labels,
                            crop.batch["spacing_um"],
                            crop.batch["dref_um"],
                            stage=criterion_stage,
                            precomputed_geometry_targets=crop.geometry_targets,
                        )
                    if rag_scale:
                        with self.stage_profiler.profile("rag"):
                            crop_rag = metrics["spatial_rag_bce"]
                    else:
                        crop_rag = metrics["geometry_loss"] * 0
                    loss = self.criterion.crop_phase_a_objective(
                        metrics,
                        geometry_scale=geometry_scale,
                        rag_scale=rag_scale,
                    ) / len(crop_rounds)
                self._sync_device()
                target_seconds += time.perf_counter() - started
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite crop Phase A loss: {loss.detach()}"
                    )
                with self.stage_profiler.profile("pre_backward"):
                    pass
                started = time.perf_counter()
                with self.stage_profiler.profile("backward"):
                    self.scaler.scale(loss).backward()
                self._sync_device()
                backward_seconds += time.perf_counter() - started
                phase_loss = phase_loss + loss.detach()
                geometry_report = geometry_report + (
                    metrics["geometry_loss"].detach() / len(crop_rounds)
                )
                rag_report = rag_report + (
                    crop_rag.detach() / len(crop_rounds)
                )
                del output, metrics, loss, crop
            with self.stage_profiler.profile("release"):
                del crop_rounds

        return (
            {
                "crop_phase_a_loss": phase_loss,
                "crop_geometry_loss": geometry_report,
                "crop_spatial_rag_bce": rag_report,
                "crop_count": phase_loss.new_tensor(
                    cfg.refinement_crops_per_step * labels.shape[0]
                ),
            },
            {
                "phase_a_crop_forward_seconds": forward_seconds,
                "phase_a_crop_target_seconds": target_seconds,
                "phase_a_crop_backward_seconds": backward_seconds,
                "phase_a_crop_candidate_count": float(len(candidate_types)),
            },
        )

    def _detached_refinement_phase_b_backward(
        self,
        batch: dict,
        *,
        labels: torch.Tensor,
        geometry_targets: GeometryTargets,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Run full-frame dense state detached, then train downstream modules."""
        fraction = teacher_forcing_fraction(
            self.training_config, self.refinement_stage_step
        )
        forward_seconds = 0.0
        target_seconds = 0.0
        backward_seconds = 0.0

        started = time.perf_counter()
        with self.stage_profiler.phase_scope("phase_b_full"):
            with torch.no_grad(), self._autocast():
                detached_dense = model_forward_from_batch(
                    self.model,
                    batch,
                    execution_stage="geometry",
                    apply_existence_filter=False,
                    stage_profiler=self.stage_profiler,
                )
        self._sync_device()
        forward_seconds += time.perf_counter() - started
        if self.stage_profiler.enabled:
            dense_tensors = (
                detached_dense.decoded_spatial.d0,
                detached_dense.decoded_spatial.d1,
                detached_dense.decoded_spatial.d2,
                detached_dense.geometry.sdf,
            )
            if any(value.requires_grad for value in dense_tensors):
                raise RuntimeError("detached Phase B dense tensors require gradients")

        discrete_target_cache: dict[str, object] = {}

        def teacher_builder(instances, rag, temporal, reasoning, dref_um):
            with self.stage_profiler.profile("teacher_request_build"):
                return build_teacher_refinement_requests(
                    instances,
                    rag,
                    temporal,
                    reasoning,
                    dref_um,
                    gt_labels=labels,
                    spacing_um=batch["spacing_um"],
                    loss_config=self.training_config.loss,
                    rag_criterion=self.criterion.rag,
                    fraction=fraction,
                    ambiguity_logit_abs_max=(
                        self.model.cfg.refinement.ambiguity_logit_abs_max
                    ),
                    target_cache=discrete_target_cache,
                )

        started = time.perf_counter()
        with self.stage_profiler.phase_scope("phase_b"):
            with self._autocast():
                output = model_forward_from_batch(
                    self.model,
                    batch,
                    execution_stage="refinement",
                    teacher_request_builder=(
                        teacher_builder if fraction > 0 else None
                    ),
                    apply_existence_filter=False,
                    precomputed_geometry=detached_dense,
                    stage_profiler=self.stage_profiler,
                )
        del detached_dense
        self._sync_device()
        forward_seconds += time.perf_counter() - started

        started = time.perf_counter()
        with self.stage_profiler.phase_scope("phase_b"):
            with self._autocast():
                with self.stage_profiler.profile("criterion"):
                    local_geometry, valid_fraction = (
                        self.criterion.refined_local_geometry_losses(
                            output,
                            geometry_targets,
                            batch["spacing_um"],
                            batch["dref_um"],
                        )
                    )
                    metrics = self.criterion(
                        output,
                        labels,
                        batch["spacing_um"],
                        batch["dref_um"],
                        stage="refinement_joint",
                        precomputed_geometry_targets=None,
                        precomputed_discrete_targets=(
                            discrete_target_cache
                            if discrete_target_cache
                            else None
                        ),
                        geometry_losses_override=local_geometry,
                        geometry_valid_fraction_override=valid_fraction,
                    )
                    loss = metrics["loss"]
        self._sync_device()
        target_seconds += time.perf_counter() - started
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite detached Phase B loss: {loss.detach()}"
            )
        with self.stage_profiler.phase_scope("phase_b"):
            with self.stage_profiler.profile("pre_backward"):
                pass
            started = time.perf_counter()
            with self.stage_profiler.profile("backward"):
                self.scaler.scale(loss).backward()
        self._sync_device()
        backward_seconds += time.perf_counter() - started
        reported = {key: value for key, value in metrics.items()}
        reported["phase_b_loss"] = loss.detach()
        reported["phase_b_refined_local_geometry_loss"] = metrics[
            "geometry_loss"
        ].detach()
        reported.update(_refinement_fallback_metrics(output, loss.detach()))
        del output, metrics, local_geometry
        return reported, {
            "phase_b_forward_seconds": forward_seconds,
            "phase_b_target_seconds": target_seconds,
            "phase_b_backward_seconds": backward_seconds,
            "refinement_teacher_forcing_fraction": fraction,
            "refinement_stage_step": float(self.refinement_stage_step),
        }

    def _crop_bounded_refinement_backward(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None,
        precomputed_geometry_targets: GeometryTargets | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        labels = gt_labels_from_batch(batch) if gt_labels is None else gt_labels
        geometry_targets = self._geometry_targets_for_step(
            batch, labels, precomputed_geometry_targets
        )
        crop_metrics, crop_timing = self._crop_phase_a_backward(
            batch,
            labels=labels,
            geometry_targets=geometry_targets,
            stage="refinement_joint",
        )
        phase_a_gradients = _group_gradient_norms(self.model)
        phase_b_metrics, phase_b_timing = (
            self._detached_refinement_phase_b_backward(
                batch,
                labels=labels,
                geometry_targets=geometry_targets,
            )
        )
        phase_b_gradients = _group_gradient_norms(self.model)
        combined = (
            crop_metrics["crop_phase_a_loss"]
            + phase_b_metrics["phase_b_loss"]
        )
        reported = {**phase_b_metrics, **crop_metrics, "loss": combined}
        timing = {
            **crop_timing,
            **phase_b_timing,
            "forward_seconds": (
                crop_timing["phase_a_crop_forward_seconds"]
                + phase_b_timing["phase_b_forward_seconds"]
            ),
            "target_seconds": (
                crop_timing["phase_a_crop_target_seconds"]
                + phase_b_timing["phase_b_target_seconds"]
            ),
            "backward_seconds": (
                crop_timing["phase_a_crop_backward_seconds"]
                + phase_b_timing["phase_b_backward_seconds"]
            ),
            "phase_a_grad_geometry_spatial": phase_a_gradients[
                "grad_geometry_spatial"
            ],
            "phase_b_accumulated_grad_geometry_spatial": phase_b_gradients[
                "grad_geometry_spatial"
            ],
        }
        return reported, timing

    def _detached_instance_temporal_backward(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None,
        precomputed_geometry_targets: GeometryTargets | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Optional large-frame instance/temporal training with detached CNN."""
        labels = gt_labels_from_batch(batch) if gt_labels is None else gt_labels
        geometry_targets = self._geometry_targets_for_step(
            batch, labels, precomputed_geometry_targets
        )
        started = time.perf_counter()
        with self.stage_profiler.phase_scope("detached_full"):
            with torch.no_grad(), self._autocast():
                dense = model_forward_from_batch(
                    self.model,
                    batch,
                    execution_stage="geometry",
                    stage_profiler=self.stage_profiler,
                )
        with self.stage_profiler.phase_scope("instance_temporal"):
            with self._autocast():
                output = model_forward_from_batch(
                    self.model,
                    batch,
                    execution_stage="temporal",
                    precomputed_geometry=dense,
                    stage_profiler=self.stage_profiler,
                )
        del dense
        self._sync_device()
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with self.stage_profiler.phase_scope("instance_temporal"):
            with self._autocast():
                with self.stage_profiler.profile("criterion"):
                    zero_geometry, valid_fraction = (
                        self.criterion.refined_local_geometry_losses(
                            output,
                            geometry_targets,
                            batch["spacing_um"],
                            batch["dref_um"],
                        )
                    )
                    metrics = self.criterion(
                        output,
                        labels,
                        batch["spacing_um"],
                        batch["dref_um"],
                        stage="instance_temporal",
                        geometry_losses_override=zero_geometry,
                        geometry_valid_fraction_override=valid_fraction,
                    )
                    loss = metrics["loss"]
        self._sync_device()
        target_seconds = time.perf_counter() - started
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite detached instance/temporal loss: {loss.detach()}"
            )
        started = time.perf_counter()
        with self.stage_profiler.phase_scope("instance_temporal"):
            with self.stage_profiler.profile("backward"):
                self.scaler.scale(loss).backward()
        self._sync_device()
        backward_seconds = time.perf_counter() - started
        return metrics, {
            "forward_seconds": forward_seconds,
            "target_seconds": target_seconds,
            "backward_seconds": backward_seconds,
            "refinement_teacher_forcing_fraction": 0.0,
            "refinement_stage_step": float(self.refinement_stage_step),
        }

    def _train_step_impl(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None = None,
        precomputed_geometry_targets: GeometryTargets | None = None,
    ) -> dict[str, float]:
        total_started = time.perf_counter()
        self.curriculum_stage = self.curriculum.apply(self.global_step)
        self.model.train()
        self.criterion.train()
        self.stage_profiler.clear()
        self.stage_profiler.set_context(
            global_step=self.global_step,
            refinement_stage_step=self.refinement_stage_step,
            phase="setup",
            metadata={"curriculum_stage": self.curriculum_stage.name},
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with self.stage_profiler.profile("batch_to_device", qualify=False):
            moved = move_batch_to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if self.curriculum_stage.name == "refinement_joint":
            if (
                self.training_config.curriculum.full_frame_spatial_grad
                or not self.training_config.curriculum.refinement_crop_enabled
            ):
                losses, timing = self._full_frame_split_refinement_backward(
                    moved,
                    gt_labels=gt_labels,
                    precomputed_geometry_targets=precomputed_geometry_targets,
                )
            else:
                losses, timing = self._crop_bounded_refinement_backward(
                    moved,
                    gt_labels=gt_labels,
                    precomputed_geometry_targets=precomputed_geometry_targets,
                )
            backward_seconds = timing["backward_seconds"]
        elif (
            self.curriculum_stage.name == "geometry_bootstrap"
            and self.training_config.curriculum.geometry_bootstrap_crop_enabled
        ) or (
            self.curriculum_stage.name == "spatial_partition"
            and self.training_config.curriculum.spatial_partition_crop_enabled
        ):
            labels = gt_labels_from_batch(moved) if gt_labels is None else gt_labels
            geometry_targets = self._geometry_targets_for_step(
                moved, labels, precomputed_geometry_targets
            )
            rag_scale = float(
                self.curriculum_stage.name == "spatial_partition"
            )
            losses, timing = self._crop_phase_a_backward(
                moved,
                labels=labels,
                geometry_targets=geometry_targets,
                stage=self.curriculum_stage.name,
                profile_phase="crop_train",
                geometry_scale=1.0,
                rag_scale=rag_scale,
            )
            losses["loss"] = losses["crop_phase_a_loss"]
            backward_seconds = timing["phase_a_crop_backward_seconds"]
            timing.update(
                {
                    "forward_seconds": timing[
                        "phase_a_crop_forward_seconds"
                    ],
                    "target_seconds": timing[
                        "phase_a_crop_target_seconds"
                    ],
                    "backward_seconds": backward_seconds,
                }
            )
        elif (
            self.curriculum_stage.name == "instance_temporal"
            and self.training_config.curriculum.instance_temporal_detached_spatial
        ):
            losses, timing = self._detached_instance_temporal_backward(
                moved,
                gt_labels=gt_labels,
                precomputed_geometry_targets=precomputed_geometry_targets,
            )
            backward_seconds = timing["backward_seconds"]
        else:
            with self._autocast():
                _, losses, timing = self._forward_and_loss(
                    moved,
                    gt_labels=gt_labels,
                    precomputed_geometry_targets=precomputed_geometry_targets,
                )
                loss = losses["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite STIR-Net V2 loss: {loss.detach()}"
                )
            self._sync_device()
            backward_started = time.perf_counter()
            with self.stage_profiler.profile("backward"):
                self.scaler.scale(loss).backward()
            self._sync_device()
            backward_seconds = time.perf_counter() - backward_started
        with self.stage_profiler.phase_scope("optimizer"):
            with self.stage_profiler.profile(
                "gradient_unscale", qualify=False
            ):
                self.scaler.unscale_(self.optimizer)
            with self.stage_profiler.profile(
                "gradient_metrics_clip", qualify=False
            ):
                grad_metrics = _group_gradient_norms(self.model)
                total_grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_config.max_grad_norm
                )
        finite_grad = bool(torch.isfinite(torch.as_tensor(total_grad)))
        optimizer_step_skipped = not finite_grad
        if finite_grad:
            with self.stage_profiler.phase_scope("optimizer"):
                with self.stage_profiler.profile(
                    "optimizer_step", qualify=False
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                    if self.curriculum_stage.name == "refinement_joint":
                        self.refinement_stage_step += 1
        else:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.update(new_scale=max(self.scaler.get_scale() * 0.5, 1.0))
        metrics = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
        metrics.update(grad_metrics)
        metrics["grad_norm"] = float(torch.as_tensor(total_grad).detach().cpu())
        metrics["optimizer_step_skipped"] = float(optimizer_step_skipped)
        metrics.update(timing)
        metrics["backward_seconds"] = backward_seconds
        metrics["total_step_seconds"] = time.perf_counter() - total_started
        if self.device.type == "cuda":
            metrics["peak_allocated_mb"] = (
                self.stage_profiler.overall_peak_allocated_mb
                if self.stage_profiler.enabled
                else torch.cuda.max_memory_allocated(self.device) / (1024**2)
            )
            metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(
                self.device
            ) / (1024**2)
        if self.stage_profiler.enabled:
            for stage_name, row in self.stage_profiler.summary().items():
                for key, value in row.items():
                    metrics[f"profile_{stage_name}_{key}"] = float(value)
        return metrics

    def _print_oom_diagnostic(self) -> None:
        values = {
            "phase": self.stage_profiler.last_profile_phase,
            "last_profile_stage": self.stage_profiler.last_profile_stage,
            "profile_path": str(self.stage_profiler.output_path or ""),
        }
        if self.device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                values.update(
                    {
                        "allocated_mb": torch.cuda.memory_allocated(self.device)
                        / (1024**2),
                        "reserved_mb": torch.cuda.memory_reserved(self.device)
                        / (1024**2),
                        "max_allocated_mb": torch.cuda.max_memory_allocated(
                            self.device
                        )
                        / (1024**2),
                        "free_mb": free_bytes / (1024**2),
                    }
                )
            except BaseException:
                pass
        for key, value in values.items():
            try:
                print(f"[OOM] {key}={value}", flush=True)
            except BaseException:
                pass

    def train_step(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None = None,
        precomputed_geometry_targets: GeometryTargets | None = None,
    ) -> dict[str, float]:
        try:
            return self._train_step_impl(
                batch,
                gt_labels=gt_labels,
                precomputed_geometry_targets=precomputed_geometry_targets,
            )
        except BaseException as error:
            is_oom = isinstance(error, torch.OutOfMemoryError) or (
                isinstance(error, RuntimeError)
                and "out of memory" in str(error).lower()
            )
            if is_oom:
                self._print_oom_diagnostic()
            raise

    @torch.no_grad()
    def eval_step(
        self,
        batch: dict,
        *,
        gt_labels: torch.Tensor | None = None,
        precomputed_geometry_targets: GeometryTargets | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        self.stage_profiler.clear()
        moved = move_batch_to_device(batch, self.device)
        with self._autocast():
            _, losses, timing = self._forward_and_loss(
                moved,
                gt_labels=gt_labels,
                precomputed_geometry_targets=precomputed_geometry_targets,
            )
        result = {
            key: float(value.detach().float().cpu()) for key, value in losses.items()
        }
        result.update(timing)
        return result

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        out_dir="runs/stirnet_v2",
        checkpoint_every: int = 1,
    ) -> None:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(epochs):
            sums: dict[str, float] = {}
            count = 0
            for batch in train_loader:
                metrics = self.train_step(batch)
                count += 1
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value
            train_average = {key: value / max(count, 1) for key, value in sums.items()}
            print(
                f"epoch {epoch + 1} stage={self.curriculum_stage.name}: "
                f"train {train_average}"
            )
            if val_loader is not None:
                validation: dict[str, float] = {}
                validation_count = 0
                for batch in val_loader:
                    metrics = self.eval_step(batch)
                    validation_count += 1
                    for key, value in metrics.items():
                        validation[key] = validation.get(key, 0.0) + value
                print(
                    f"epoch {epoch + 1}: val "
                    f"{ {key: value / max(validation_count, 1) for key, value in validation.items()} }"
                )
            if (epoch + 1) % checkpoint_every == 0:
                save_checkpoint(
                    output_dir / f"epoch_{epoch + 1:04d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    step=self.global_step,
                    epoch=epoch + 1,
                    model_config=self.model.cfg,
                    training_config=self.training_config,
                    extra=self.checkpoint_metadata(),
                )


__all__ = [
    "Trainer",
    "gt_labels_from_batch",
    "model_forward_from_batch",
    "move_batch_to_device",
    "move_to_device",
]
