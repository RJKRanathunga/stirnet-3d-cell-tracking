from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..model import GeometryCriterion, RAGCriterion, StirNetOutput
from ..model.geometry.targets import build_geometry_targets
from ..model.utils.physical import physical_crop_slices
from .config import LossConfig


@dataclass
class InstanceTargets:
    existence: Tensor
    split: Tensor


@dataclass
class RecoveryTargets:
    target: Tensor
    valid: Tensor


def _safe_binary_metrics(logits: Tensor, target: Tensor) -> dict[str, Tensor]:
    if target.numel() == 0:
        zero = logits.sum() * 0
        return {
            "accuracy": zero.detach(),
            "precision": zero.detach(),
            "recall": zero.detach(),
        }
    pred = logits.sigmoid() >= 0.5
    truth = target.bool()
    tp = (pred & truth).sum().float()
    fp = (pred & ~truth).sum().float()
    fn = (~pred & truth).sum().float()
    return {
        "accuracy": (pred == truth).float().mean().detach(),
        "precision": (tp / (tp + fp).clamp_min(1)).detach(),
        "recall": (tp / (tp + fn).clamp_min(1)).detach(),
    }


def build_instance_targets(
    provisional_labels: Iterable[Tensor],
    gt_labels: Tensor,
    *,
    existence_min_precision: float,
    existence_min_gt_coverage: float,
    split_min_pred_fraction: float,
    split_min_gt_coverage: float,
    device: torch.device,
) -> InstanceTargets:
    existence_rows: list[Tensor] = []
    split_rows: list[Tensor] = []
    for batch_index, predicted in enumerate(provisional_labels):
        gt = gt_labels[batch_index].to(predicted.device).long()
        for instance_id in range(1, int(predicted.max().item()) + 1):
            mask = predicted == instance_id
            pred_count = mask.sum().float().clamp_min(1)
            values = gt[mask]
            values = values[values > 0]
            if values.numel() == 0:
                existence_rows.append(pred_count.new_tensor(0.0))
                split_rows.append(pred_count.new_tensor(0.0))
                continue
            ids, intersections = torch.unique(values, return_counts=True)
            intersections = intersections.float()
            gt_counts = torch.stack([(gt == gt_id).sum() for gt_id in ids]).float()
            pred_fraction = intersections / pred_count
            gt_coverage = intersections / gt_counts.clamp_min(1)
            meaningful = (
                (pred_fraction >= existence_min_precision)
                & (gt_coverage >= existence_min_gt_coverage)
            )
            split_parts = (
                (pred_fraction >= split_min_pred_fraction)
                & (gt_coverage >= split_min_gt_coverage)
            )
            existence_rows.append(meaningful.any().float())
            split_rows.append((split_parts.sum() >= 2).float())
    if not existence_rows:
        empty = torch.zeros((0,), device=device)
        return InstanceTargets(empty, empty)
    return InstanceTargets(
        torch.stack(existence_rows).to(device=device),
        torch.stack(split_rows).to(device=device),
    )


def build_recovery_targets(
    output: StirNetOutput,
    gt_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    search_radius_dref: float,
    missing_gt_coverage: float,
) -> RecoveryTargets:
    count = output.temporal.tokens.shape[0]
    target = output.temporal.tokens.new_zeros((count,))
    valid = torch.zeros(count, device=target.device, dtype=torch.bool)
    for row in range(count):
        batch_index = int(output.temporal.batch_index[row].item())
        status = output.temporal.status[row]
        if status.shape[0] > 5 and bool(status[5] > 0.5):
            continue
        gt = gt_labels[batch_index].to(target.device).long()
        radius_um = search_radius_dref * dref_um[batch_index]
        slices = physical_crop_slices(
            tuple(gt.shape),
            spacing_um[batch_index],
            output.temporal.ref_um[row],
            radius_um,
        )
        local = gt[slices]
        points = torch.nonzero(local > 0, as_tuple=False)
        if points.numel() == 0:
            continue
        origin = torch.tensor(
            [axis.start for axis in slices], device=points.device, dtype=torch.float32
        )
        absolute = points.float() + origin
        extent = (gt.new_tensor(gt.shape).float() - 1) * spacing_um[batch_index]
        points_um = absolute * spacing_um[batch_index] - 0.5 * extent
        distances = torch.linalg.vector_norm(
            points_um - output.temporal.ref_um[row].float(), dim=-1
        )
        nearest = int(distances.argmin().item())
        if bool(distances[nearest] > radius_um):
            continue
        voxel = absolute[nearest].long()
        gt_id = int(gt[voxel[0], voxel[1], voxel[2]].item())
        if gt_id <= 0:
            continue
        gt_mask = gt == gt_id
        provisional = output.initial_provisional_instances.labels[batch_index]
        overlaps = provisional[gt_mask]
        overlaps = overlaps[overlaps > 0]
        coverage = 0.0
        if overlaps.numel():
            _, counts = torch.unique(overlaps, return_counts=True)
            coverage = float(counts.max().item()) / max(int(gt_mask.sum().item()), 1)
        target[row] = float(coverage < missing_gt_coverage)
        valid[row] = True
    return RecoveryTargets(target, valid)


class StirNetCriterion(nn.Module):
    """Differentiable V2 objective applied before hard watershed/union-find decisions."""

    _STAGE_LOSSES = {
        "geometry_bootstrap": frozenset({"geometry"}),
        "spatial_partition": frozenset({"geometry", "spatial_rag"}),
        "instance_temporal": frozenset(
            {"geometry", "spatial_rag", "final_rag", "existence", "split", "recovery"}
        ),
        "refinement_joint": frozenset(
            {"geometry", "spatial_rag", "final_rag", "existence", "split", "recovery"}
        ),
    }

    def __init__(self, model_config, loss_config: LossConfig | None = None):
        super().__init__()
        self.model_config = model_config
        self.cfg = loss_config or LossConfig()
        self.geometry = GeometryCriterion(model_config.geometry)
        self.rag = RAGCriterion()

    def forward(
        self,
        output: StirNetOutput,
        gt_labels: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        stage: str = "refinement_joint",
    ) -> dict[str, Tensor]:
        if stage not in self._STAGE_LOSSES:
            raise ValueError(f"Unknown V2 training stage: {stage}")
        active = self._STAGE_LOSSES[stage]
        device = output.geometry.sdf.device
        gt_cpu = gt_labels.detach().cpu().long()
        geometry_target = build_geometry_targets(
            gt_cpu,
            spacing_um.detach().cpu(),
            dref_um.detach().cpu(),
            sdf_clip_dref=self.model_config.geometry.sdf_clip_dref,
            device=device,
        )
        geometry_losses = self.geometry(
            output.geometry, geometry_target, spacing_um, dref_um
        )
        geometry_total = torch.stack(list(geometry_losses.values())).sum()

        spatial_rag = self.rag(output.rag, gt_labels)
        final_rag = self.rag(
            output.rag, gt_labels, logits=output.reasoning.final_edge_logits
        )
        initial_spatial_rag = self.rag(output.initial_rag, gt_labels)
        initial_final_rag = self.rag(
            output.initial_rag,
            gt_labels,
            logits=output.initial_reasoning.final_edge_logits,
        )
        refinement_applied = bool(
            output.refinement is not None and output.refinement.applied_count
        )
        spatial_rag_loss = (
            0.5 * (spatial_rag["rag_bce"] + initial_spatial_rag["rag_bce"])
            if refinement_applied
            else spatial_rag["rag_bce"]
        )
        final_rag_loss = (
            0.5 * (final_rag["rag_bce"] + initial_final_rag["rag_bce"])
            if refinement_applied
            else final_rag["rag_bce"]
        )
        instance_target = build_instance_targets(
            output.initial_provisional_instances.labels,
            gt_labels,
            existence_min_precision=self.cfg.existence_min_precision,
            existence_min_gt_coverage=self.cfg.existence_min_gt_coverage,
            split_min_pred_fraction=self.cfg.split_min_pred_fraction,
            split_min_gt_coverage=self.cfg.split_min_gt_coverage,
            device=device,
        )
        request_reasoning = output.initial_reasoning
        if request_reasoning.instance_exist_logits.numel():
            existence_loss = F.binary_cross_entropy_with_logits(
                request_reasoning.instance_exist_logits, instance_target.existence
            )
            split_loss = F.binary_cross_entropy_with_logits(
                request_reasoning.split_logits, instance_target.split
            )
        else:
            existence_loss = output.geometry.sdf.sum() * 0
            split_loss = output.geometry.sdf.sum() * 0
        existence_metrics = _safe_binary_metrics(
            request_reasoning.instance_exist_logits, instance_target.existence
        )
        split_metrics = _safe_binary_metrics(
            request_reasoning.split_logits, instance_target.split
        )

        recovery_target = build_recovery_targets(
            output,
            gt_labels,
            spacing_um,
            dref_um,
            search_radius_dref=self.cfg.recovery_search_radius_dref,
            missing_gt_coverage=self.cfg.recovery_missing_gt_coverage,
        )
        if recovery_target.valid.any():
            recovery_loss = F.binary_cross_entropy_with_logits(
                request_reasoning.recovery_logits[recovery_target.valid],
                recovery_target.target[recovery_target.valid],
            )
            recovery_metrics = _safe_binary_metrics(
                request_reasoning.recovery_logits[recovery_target.valid],
                recovery_target.target[recovery_target.valid],
            )
        else:
            recovery_loss = request_reasoning.recovery_logits.sum() * 0
            recovery_metrics = _safe_binary_metrics(
                request_reasoning.recovery_logits[:0], recovery_target.target[:0]
            )

        weighted = {
            "geometry": self.cfg.geometry_weight * geometry_total,
            "spatial_rag": self.cfg.spatial_rag_weight * spatial_rag_loss,
            "final_rag": self.cfg.final_rag_weight * final_rag_loss,
            "existence": self.cfg.existence_weight * existence_loss,
            "split": self.cfg.split_weight * split_loss,
            "recovery": self.cfg.recovery_weight * recovery_loss,
        }
        total = sum(weighted[name] for name in active)
        metrics: dict[str, Tensor] = {"loss": total}
        metrics.update({f"geometry_{name}": value for name, value in geometry_losses.items()})
        metrics.update(
            {
                "geometry_loss": geometry_total,
                "spatial_rag_bce": spatial_rag_loss,
                "spatial_rag_accuracy": spatial_rag["rag_accuracy"],
                "initial_spatial_rag_bce": initial_spatial_rag["rag_bce"],
                "final_rag_bce": final_rag_loss,
                "final_rag_accuracy": final_rag["rag_accuracy"],
                "initial_final_rag_bce": initial_final_rag["rag_bce"],
                "existence_bce": existence_loss,
                "existence_accuracy": existence_metrics["accuracy"],
                "existence_precision": existence_metrics["precision"],
                "existence_recall": existence_metrics["recall"],
                "split_bce": split_loss,
                "split_accuracy": split_metrics["accuracy"],
                "recovery_bce": recovery_loss,
                "recovery_accuracy": recovery_metrics["accuracy"],
                "recovery_valid_count": recovery_target.valid.sum().detach().float(),
                "refined_geometry_loss": geometry_total
                if output.refinement is not None and output.refinement.applied_count
                else geometry_total.detach() * 0,
            }
        )
        return metrics


__all__ = [
    "InstanceTargets",
    "RecoveryTargets",
    "StirNetCriterion",
    "build_instance_targets",
    "build_recovery_targets",
]
