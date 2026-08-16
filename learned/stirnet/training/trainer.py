from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

import torch

from ..model import StirNet
from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .criterion import StirNetCriterion
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
    apply_existence_filter: bool = False,
    return_debug: bool = False,
):
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
        run_refinement=run_refinement,
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

    def _autocast(self):
        if self.device.type != "cuda" or self.training_config.amp_dtype == "fp32":
            return nullcontext()
        dtype = (
            torch.float16
            if self.training_config.amp_dtype == "fp16"
            else torch.bfloat16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _forward_and_loss(self, batch: dict, *, return_debug: bool = False):
        output = model_forward_from_batch(
            self.model,
            batch,
            use_temporal=self.curriculum_stage.use_temporal,
            run_refinement=self.curriculum_stage.run_refinement,
            apply_existence_filter=False,
            return_debug=return_debug,
        )
        gt_labels = gt_labels_from_batch(batch)
        losses = self.criterion(
            output,
            gt_labels,
            batch["spacing_um"],
            batch["dref_um"],
            stage=self.curriculum_stage.name,
        )
        return output, losses

    def train_step(self, batch: dict) -> dict[str, float]:
        self.curriculum_stage = self.curriculum.apply(self.global_step)
        self.model.train()
        self.criterion.train()
        moved = move_batch_to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            _, losses = self._forward_and_loss(moved)
            loss = losses["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite STIR-Net V2 loss: {loss.detach()}")
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
        else:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.update(new_scale=max(self.scaler.get_scale() * 0.5, 1.0))
        metrics = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
        metrics.update(grad_metrics)
        metrics["grad_norm"] = float(torch.as_tensor(total_grad).detach().cpu())
        metrics["optimizer_step_skipped"] = float(optimizer_step_skipped)
        return metrics

    @torch.no_grad()
    def eval_step(self, batch: dict) -> dict[str, float]:
        self.model.eval()
        self.criterion.eval()
        moved = move_batch_to_device(batch, self.device)
        with self._autocast():
            _, losses = self._forward_and_loss(moved)
        return {
            key: float(value.detach().float().cpu()) for key, value in losses.items()
        }

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
                    extra={"curriculum_stage": self.curriculum_stage.name},
                )


__all__ = [
    "Trainer",
    "gt_labels_from_batch",
    "model_forward_from_batch",
    "move_batch_to_device",
    "move_to_device",
]
