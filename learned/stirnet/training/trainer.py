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
from .curriculum import (
    CurriculumController,
    model_parameter_groups,
    optimizer_parameter_groups,
)


MODEL_INPUT_KEYS = frozenset(
    {
        "spatial_inputs",
        "spacing_um",
        "dref_um",
        "spatial_padding_mask",
        "graph_x",
        "graph_edge_index",
        "graph_edge_attr",
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
        return self.criterion.build_geometry_targets(
            labels,
            batch["spacing_um"],
            batch["dref_um"],
            device=self.device,
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
        )
        self._sync_device()
        forward_seconds = time.perf_counter() - forward_started
        target_started = time.perf_counter()
        losses = self.criterion(
            output,
            labels,
            batch["spacing_um"],
            batch["dref_um"],
            stage=self.curriculum_stage.name,
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

    def train_step(
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
        moved = move_batch_to_device(batch, self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            _, losses, timing = self._forward_and_loss(
                moved,
                gt_labels=gt_labels,
                precomputed_geometry_targets=precomputed_geometry_targets,
            )
            loss = losses["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite STIR-Net V2 loss: {loss.detach()}")
        self._sync_device()
        backward_started = time.perf_counter()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_metrics = _group_gradient_norms(self.model)
        total_grad = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.training_config.max_grad_norm
        )
        finite_grad = bool(torch.isfinite(torch.as_tensor(total_grad)))
        optimizer_step_skipped = not finite_grad
        if finite_grad:
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
        self._sync_device()
        backward_seconds = time.perf_counter() - backward_started
        metrics = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
        metrics.update(grad_metrics)
        metrics["grad_norm"] = float(torch.as_tensor(total_grad).detach().cpu())
        metrics["optimizer_step_skipped"] = float(optimizer_step_skipped)
        metrics.update(timing)
        metrics["backward_seconds"] = backward_seconds
        metrics["total_step_seconds"] = time.perf_counter() - total_started
        if self.device.type == "cuda":
            metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(
                self.device
            ) / (1024**2)
            metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(
                self.device
            ) / (1024**2)
        return metrics

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
