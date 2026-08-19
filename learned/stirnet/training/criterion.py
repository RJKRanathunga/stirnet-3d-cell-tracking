from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..model import GeometryCriterion, RAGCriterion
from ..model.geometry.targets import GeometryTargets, build_geometry_targets
from ..model.types import (
    GeometryForwardOutput,
    GeometryState,
    InstanceState,
    RAGState,
    ReasoningState,
    RefinementRequest,
    SpatialForwardOutput,
    StirNetOutput,
    TemporalState,
    geometry_field_crop,
)
from ..model.utils.contingency import label_contingency
from ..model.utils.physical import physical_crop_slices
from .config import LossConfig, TrainingConfig


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
    """Derive instance targets from one contingency scan per batch item."""
    existence_rows: list[Tensor] = []
    split_rows: list[Tensor] = []
    for batch_index, predicted in enumerate(provisional_labels):
        gt = gt_labels[batch_index].to(predicted.device).long()
        count = int(predicted.max().item())
        existence = torch.zeros(count, device=predicted.device)
        split = torch.zeros_like(existence)
        table = label_contingency(predicted, gt)
        if table.row_ids.numel() and table.column_ids.numel():
            intersections = table.intersections.float()
            pred_fraction = intersections / table.row_counts[:, None].clamp_min(1)
            gt_coverage = intersections / table.column_counts[None, :].clamp_min(1)
            meaningful = (
                (pred_fraction >= existence_min_precision)
                & (gt_coverage >= existence_min_gt_coverage)
            )
            split_parts = (
                (pred_fraction >= split_min_pred_fraction)
                & (gt_coverage >= split_min_gt_coverage)
            )
            rows = table.row_ids - 1
            existence[rows] = meaningful.any(dim=1).float()
            split[rows] = (split_parts.sum(dim=1) >= 2).float()
        existence_rows.append(existence)
        split_rows.append(split)
    if not existence_rows:
        empty = torch.zeros((0,), device=device)
        return InstanceTargets(empty, empty)
    return InstanceTargets(
        torch.cat(existence_rows).to(device=device),
        torch.cat(split_rows).to(device=device),
    )


def _recovery_targets_from_states(
    temporal: TemporalState,
    provisional_labels: list[Tensor],
    gt_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    search_radius_dref: float,
    missing_gt_coverage: float,
    min_provisional_precision: float,
) -> RecoveryTargets:
    count = temporal.tokens.shape[0]
    target = temporal.tokens.new_zeros((count,))
    valid = torch.zeros(count, device=target.device, dtype=torch.bool)
    for batch_index, provisional in enumerate(provisional_labels):
        gt = gt_labels[batch_index].to(target.device).long()
        table = label_contingency(provisional.to(target.device), gt)
        best_coverage = target.new_zeros((table.column_ids.numel(),))
        if table.intersections.numel():
            precision = (
                table.intersections.float()
                / table.row_counts[:, None].clamp_min(1).float()
            )
            coverage = (
                table.intersections.float()
                / table.column_counts[None, :].clamp_min(1).float()
            )
            best_coverage = (
                coverage
                * (precision >= min_provisional_precision).to(coverage.dtype)
            ).max(dim=0).values
        rows = torch.nonzero(
            temporal.batch_index == batch_index, as_tuple=False
        ).flatten()
        radius_um = search_radius_dref * dref_um[batch_index]
        for row in rows.tolist():
            status = temporal.status[row]
            if status.shape[0] > 5 and bool(status[5] > 0.5):
                continue
            slices = physical_crop_slices(
                tuple(gt.shape),
                spacing_um[batch_index],
                temporal.ref_um[row],
                radius_um,
            )
            local = gt[slices]
            points = torch.nonzero(local > 0, as_tuple=False)
            if points.numel() == 0:
                continue
            origin = torch.as_tensor(
                [axis.start for axis in slices],
                device=points.device,
                dtype=torch.float32,
            )
            absolute = points.float() + origin
            extent = (
                torch.as_tensor(gt.shape, device=points.device).float() - 1
            ) * spacing_um[batch_index]
            points_um = absolute * spacing_um[batch_index] - 0.5 * extent
            distances = torch.linalg.vector_norm(
                points_um - temporal.ref_um[row].float(), dim=-1
            )
            nearest = distances.argmin()
            if bool(distances[nearest] > radius_um):
                continue
            voxel = absolute[nearest].long()
            gt_id = gt[voxel[0], voxel[1], voxel[2]]
            column = torch.searchsorted(table.column_ids, gt_id)
            coverage = target.new_tensor(0.0)
            if (
                column < table.column_ids.numel()
                and table.column_ids[column] == gt_id
            ):
                coverage = best_coverage[column]
            target[row] = (coverage < missing_gt_coverage).float()
            valid[row] = True
    return RecoveryTargets(target, valid)


def build_recovery_targets(
    output: StirNetOutput,
    gt_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    search_radius_dref: float,
    missing_gt_coverage: float,
    min_provisional_precision: float = 0.10,
) -> RecoveryTargets:
    return _recovery_targets_from_states(
        output.temporal,
        output.initial_provisional_instances.labels,
        gt_labels,
        spacing_um,
        dref_um,
        search_radius_dref=search_radius_dref,
        missing_gt_coverage=missing_gt_coverage,
        min_provisional_precision=min_provisional_precision,
    )


def teacher_forcing_fraction(config: TrainingConfig, step: int) -> float:
    duration = config.refinement_teacher_forcing_decay_steps
    if duration <= 0:
        return float(config.refinement_teacher_forcing_end)
    progress = min(max(step, 0) / duration, 1.0)
    return float(
        config.refinement_teacher_forcing_start
        + progress
        * (
            config.refinement_teacher_forcing_end
            - config.refinement_teacher_forcing_start
        )
    )


def build_teacher_refinement_requests(
    instances: InstanceState,
    rag: RAGState,
    temporal: TemporalState,
    reasoning: ReasoningState,
    dref_um: Tensor,
    *,
    gt_labels: Tensor,
    spacing_um: Tensor,
    loss_config: LossConfig,
    rag_criterion: RAGCriterion,
    fraction: float,
    ambiguity_logit_abs_max: float,
    target_cache: dict[str, object] | None = None,
) -> list[RefinementRequest]:
    """Use GT only to select useful training ROIs, never as refiner input."""
    if fraction <= 0:
        return []
    requests: list[RefinementRequest] = []
    instance_targets = build_instance_targets(
        instances.labels,
        gt_labels,
        existence_min_precision=loss_config.existence_min_precision,
        existence_min_gt_coverage=loss_config.existence_min_gt_coverage,
        split_min_pred_fraction=loss_config.split_min_pred_fraction,
        split_min_gt_coverage=loss_config.split_min_gt_coverage,
        device=instances.tokens.device,
    )
    for row in torch.nonzero(instance_targets.split > 0.5, as_tuple=False).flatten().tolist():
        requests.append(
            RefinementRequest(
                batch_index=int(instances.batch_index[row].item()),
                center_um=instances.ref_um[row],
                query_token=reasoning.instance_tokens[row],
                kind="split",
                source_index=row,
                score=3.0,
                selection_source="teacher",
            )
        )
    recovery = _recovery_targets_from_states(
        temporal,
        instances.labels,
        gt_labels,
        spacing_um,
        dref_um,
        search_radius_dref=loss_config.recovery_search_radius_dref,
        missing_gt_coverage=loss_config.recovery_missing_gt_coverage,
        min_provisional_precision=loss_config.recovery_min_pred_precision,
    )
    recovery_rows = torch.nonzero(
        recovery.valid & (recovery.target > 0.5), as_tuple=False
    ).flatten()
    for row in recovery_rows.tolist():
        requests.append(
            RefinementRequest(
                batch_index=int(temporal.batch_index[row].item()),
                center_um=temporal.ref_um[row],
                query_token=temporal.tokens[row],
                kind="recovery",
                source_index=row,
                score=3.0,
                selection_source="teacher",
            )
        )
    rag_targets = rag_criterion.build_targets(rag, gt_labels)
    if target_cache is not None:
        target_cache.update(
            {
                "instance": instance_targets,
                "recovery": recovery,
                "rag": rag_targets,
            }
        )
    if rag.edge_index.shape[1] and reasoning.instance_tokens.numel():
        predicted = rag.spatial_edge_logits.detach() >= 0
        useful = rag_targets.valid & (
            (predicted != rag_targets.target.bool())
            | (rag.spatial_edge_logits.detach().abs() <= ambiguity_logit_abs_max)
        )
        for edge_row in torch.nonzero(useful, as_tuple=False).flatten().tolist():
            a = int(rag.edge_index[0, edge_row].item())
            b = int(rag.edge_index[1, edge_row].item())
            ia = int(instances.node_to_instance[a].item())
            ib = int(instances.node_to_instance[b].item())
            requests.append(
                RefinementRequest(
                    batch_index=int(rag.edge_batch[edge_row].item()),
                    center_um=0.5
                    * (rag.node_centroid_um[a] + rag.node_centroid_um[b]),
                    query_token=0.5
                    * (
                        reasoning.instance_tokens[ia]
                        + reasoning.instance_tokens[ib]
                    ),
                    kind="edge",
                    source_index=edge_row,
                    score=2.0,
                    selection_source="teacher",
                )
            )
    requests.sort(key=lambda request: request.score, reverse=True)
    keep = min(len(requests), max(1, math.ceil(fraction * len(requests))))
    return requests[:keep]


class StirNetCriterion(nn.Module):
    """Stage-aware V2 supervision before hard watershed/union-find decisions."""

    def __init__(self, model_config, loss_config: LossConfig | None = None):
        super().__init__()
        self.model_config = model_config
        self.cfg = loss_config or LossConfig()
        self.geometry = GeometryCriterion(model_config.geometry)
        self.rag = RAGCriterion(model_config.partition)

    def build_geometry_targets(
        self,
        gt_labels: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        current_labels: Tensor | None = None,
        device: torch.device | None = None,
    ) -> GeometryTargets:
        geometry_cfg = self.model_config.geometry
        return build_geometry_targets(
            gt_labels.detach().cpu().long(),
            spacing_um.detach().cpu(),
            dref_um.detach().cpu(),
            current_labels=(
                None
                if current_labels is None
                else current_labels.detach().cpu().long()
            ),
            sdf_clip_dref=geometry_cfg.sdf_clip_dref,
            sdf_supervision_radius_dref=geometry_cfg.sdf_supervision_radius_dref,
            surface_target_sigma_um=geometry_cfg.surface_target_sigma_um,
            separator_target_sigma_um=geometry_cfg.separator_target_sigma_um,
            separator_source_conditioned=geometry_cfg.separator_source_conditioned,
            separator_source_min_overlap_voxels=(
                geometry_cfg.separator_source_min_overlap_voxels
            ),
            separator_source_min_gt_fraction=(
                geometry_cfg.separator_source_min_gt_fraction
            ),
            device=device,
        )

    def crop_phase_a_objective(
        self,
        metrics: dict[str, Tensor],
        *,
        geometry_scale: float = 1.0,
        rag_scale: float = 0.0,
    ) -> Tensor:
        """Dense crop estimator used before detached full-frame reasoning."""
        result = (
            geometry_scale
            * self.cfg.geometry_weight
            * metrics["geometry_loss"]
        )
        if rag_scale:
            result = result + (
                rag_scale
                * self.cfg.spatial_rag_weight
                * metrics["spatial_rag_bce"]
            )
        return result

    def refined_local_geometry_losses(
        self,
        output: StirNetOutput,
        targets: GeometryTargets,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Supervise only corrected ROIs, retaining gradients to sparse deltas."""
        refinement = output.refinement
        rois = (
            []
            if refinement is None
            else list(refinement.geometry.delta.rois)
            if hasattr(refinement.geometry, "delta")
            else []
        )
        if not rois:
            zero = output.initial_geometry.sdf.sum() * 0
            names = (
                "foreground_bce",
                "foreground_dice",
                "surface_bce",
                "surface_dice",
                "separator_bce",
                "separator_dice",
                "sdf",
                "flow_direction",
                "flow_l1",
                "flow_magnitude",
                "flow_background",
                "centroid_offset",
                "seed",
                "flow_sdf_consistency",
                "eikonal",
            )
            return {name: zero for name in names}, zero.detach()

        weighted: dict[str, Tensor] = {}
        valid_numerator = output.initial_geometry.sdf.new_zeros(())
        total_voxels = sum(math.prod(roi.delta.shape[-3:]) for roi in rois)
        for roi in rois:
            b = roi.batch_index
            crop = roi.slices_zyx
            prediction = GeometryState(
                foreground_logits=geometry_field_crop(
                    output.geometry, "foreground_logits", b, crop
                )[None],
                surface_logits=geometry_field_crop(
                    output.geometry, "surface_logits", b, crop
                )[None],
                separator_logits=geometry_field_crop(
                    output.geometry, "separator_logits", b, crop
                )[None],
                sdf=geometry_field_crop(output.geometry, "sdf", b, crop)[None],
                flow=geometry_field_crop(output.geometry, "flow", b, crop)[None],
                centroid_offset=geometry_field_crop(
                    output.geometry, "centroid_offset", b, crop
                )[None],
                seed_logits=geometry_field_crop(
                    output.geometry, "seed_logits", b, crop
                )[None],
                features=None,
            )
            target = GeometryTargets(
                **{
                    name: value[
                        b : b + 1,
                        :,
                        crop[0],
                        crop[1],
                        crop[2],
                    ].to(prediction.sdf.device)
                    for name, value in targets.__dict__.items()
                }
            )
            row = self.geometry(
                prediction,
                target,
                spacing_um[b : b + 1],
                dref_um[b : b + 1],
            )
            voxels = math.prod(roi.delta.shape[-3:])
            weight = voxels / max(total_voxels, 1)
            for name, value in row.items():
                weighted[name] = weighted.get(name, value * 0) + weight * value
            valid_numerator = valid_numerator + (
                weight * target.sdf_valid.float().mean()
            )
        return weighted, valid_numerator.detach()

    def refinement_phase_a_objective(self, metrics: dict[str, Tensor]) -> Tensor:
        """Base contribution for memory-bounded refinement training."""
        return (
            self.cfg.geometry_weight * metrics["geometry_loss"]
            + 0.5 * self.cfg.spatial_rag_weight * metrics["working_spatial_rag_bce"]
            + 0.5 * self.cfg.final_rag_weight * metrics["working_final_rag_bce"]
            + self.cfg.existence_weight * metrics["existence_bce"]
            + self.cfg.split_weight * metrics["split_bce"]
            + self.cfg.recovery_weight * metrics["recovery_bce"]
        )

    def refinement_phase_b_objective(
        self,
        metrics: dict[str, Tensor],
        *,
        phase_a_geometry_loss: Tensor,
        refinement_applied: bool,
    ) -> Tensor:
        """Detached-dense refinement contribution.

        The geometry difference makes Phase A + Phase B reproduce the refined
        geometry scalar while routing the base gradient only through Phase A.
        Initial request losses intentionally do not appear a second time.
        """
        geometry_delta = (
            metrics["geometry_loss"] - phase_a_geometry_loss.detach()
            if refinement_applied
            else metrics["geometry_loss"] * 0
        )
        return (
            self.cfg.geometry_weight * geometry_delta
            + 0.5 * self.cfg.spatial_rag_weight * metrics["working_spatial_rag_bce"]
            + 0.5 * self.cfg.final_rag_weight * metrics["working_final_rag_bce"]
        )

    def forward(
        self,
        output: GeometryForwardOutput | SpatialForwardOutput | StirNetOutput,
        gt_labels: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        stage: str = "refinement_joint",
        current_labels: Tensor | None = None,
        precomputed_geometry_targets: GeometryTargets | None = None,
        precomputed_discrete_targets: dict[str, object] | None = None,
        geometry_losses_override: dict[str, Tensor] | None = None,
        geometry_valid_fraction_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if stage not in {
            "geometry_bootstrap",
            "spatial_partition",
            "instance_temporal",
            "refinement_joint",
        }:
            raise ValueError(f"Unknown V2 training stage: {stage}")
        reference_tensor = (
            output.initial_geometry.sdf
            if isinstance(output, StirNetOutput)
            else output.geometry.sdf
        )
        device = reference_tensor.device
        geometry_target = None
        if geometry_losses_override is None:
            geometry_target = (
                self.build_geometry_targets(
                    gt_labels,
                    spacing_um,
                    dref_um,
                    current_labels=current_labels,
                    device=device,
                )
                if precomputed_geometry_targets is None
                else precomputed_geometry_targets.to(device)
            )
            geometry_losses = self.geometry(
                output.geometry, geometry_target, spacing_um, dref_um
            )
            sdf_valid_fraction = geometry_target.sdf_valid.float().mean().detach()
        else:
            geometry_losses = geometry_losses_override
            sdf_valid_fraction = (
                output.initial_geometry.sdf.new_zeros(())
                if geometry_valid_fraction_override is None
                else geometry_valid_fraction_override
            )
        geometry_total = torch.stack(list(geometry_losses.values())).sum()
        metrics: dict[str, Tensor] = {
            "geometry_loss": geometry_total,
            "sdf_valid_fraction": sdf_valid_fraction,
            **{f"geometry_{name}": value for name, value in geometry_losses.items()},
        }
        if stage == "geometry_bootstrap":
            metrics["loss"] = self.cfg.geometry_weight * geometry_total
            return metrics
        if not isinstance(output, (SpatialForwardOutput, StirNetOutput)):
            raise TypeError("spatial and later stages require a spatial output")
        cached_rag_targets = (
            precomputed_discrete_targets.get("rag")
            if precomputed_discrete_targets is not None
            else None
        )
        can_reuse_initial = isinstance(output, StirNetOutput) and (
            output.rag is output.initial_rag
        )
        spatial_targets = (
            cached_rag_targets
            if cached_rag_targets is not None and can_reuse_initial
            else self.rag.build_targets(output.rag, gt_labels)
        )
        spatial_rag = self.rag(
            output.rag, gt_labels, targets=spatial_targets
        )
        metrics.update(
            {
                "spatial_rag_bce": spatial_rag["rag_bce"],
                "spatial_rag_accuracy": spatial_rag["rag_accuracy"],
                "rag_valid_edge_fraction": spatial_rag["rag_valid_edge_fraction"],
                "rag_mean_node_purity": spatial_rag["rag_mean_node_purity"],
                "rag_impure_node_fraction": spatial_rag["rag_impure_node_fraction"],
                "rag_mean_node_gt_support": spatial_rag[
                    "rag_mean_node_gt_support"
                ],
                "rag_low_support_node_fraction": spatial_rag[
                    "rag_low_support_node_fraction"
                ],
            }
        )
        if stage == "spatial_partition":
            metrics["loss"] = (
                self.cfg.geometry_weight * geometry_total
                + self.cfg.spatial_rag_weight * spatial_rag["rag_bce"]
            )
            return metrics
        if not isinstance(output, StirNetOutput):
            raise TypeError("temporal and refinement stages require a full output")
        final_rag = self.rag(
            output.rag,
            gt_labels,
            logits=output.reasoning.final_edge_logits,
            targets=spatial_targets,
        )
        refinement_applied = bool(
            output.refinement is not None and output.refinement.applied_count
        )
        initial_spatial_loss = spatial_rag["rag_bce"]
        initial_final_loss = final_rag["rag_bce"]
        if refinement_applied:
            initial_targets = (
                cached_rag_targets
                if cached_rag_targets is not None
                else self.rag.build_targets(output.initial_rag, gt_labels)
            )
            initial_spatial = self.rag(
                output.initial_rag, gt_labels, targets=initial_targets
            )
            initial_final = self.rag(
                output.initial_rag,
                gt_labels,
                logits=output.initial_reasoning.final_edge_logits,
                targets=initial_targets,
            )
            spatial_rag_loss = 0.5 * (
                spatial_rag["rag_bce"] + initial_spatial["rag_bce"]
            )
            final_rag_loss = 0.5 * (
                final_rag["rag_bce"] + initial_final["rag_bce"]
            )
            initial_spatial_loss = initial_spatial["rag_bce"]
            initial_final_loss = initial_final["rag_bce"]
        else:
            spatial_rag_loss = spatial_rag["rag_bce"]
            final_rag_loss = final_rag["rag_bce"]
        instance_target = (
            precomputed_discrete_targets["instance"]
            if precomputed_discrete_targets is not None
            and "instance" in precomputed_discrete_targets
            else build_instance_targets(
                output.initial_provisional_instances.labels,
                gt_labels,
                existence_min_precision=self.cfg.existence_min_precision,
                existence_min_gt_coverage=self.cfg.existence_min_gt_coverage,
                split_min_pred_fraction=self.cfg.split_min_pred_fraction,
                split_min_gt_coverage=self.cfg.split_min_gt_coverage,
                device=device,
            )
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
            existence_loss = reference_tensor.sum() * 0
            split_loss = reference_tensor.sum() * 0
        existence_metrics = _safe_binary_metrics(
            request_reasoning.instance_exist_logits, instance_target.existence
        )
        split_metrics = _safe_binary_metrics(
            request_reasoning.split_logits, instance_target.split
        )
        recovery_target = (
            precomputed_discrete_targets["recovery"]
            if precomputed_discrete_targets is not None
            and "recovery" in precomputed_discrete_targets
            else build_recovery_targets(
                output,
                gt_labels,
                spacing_um,
                dref_um,
                search_radius_dref=self.cfg.recovery_search_radius_dref,
                missing_gt_coverage=self.cfg.recovery_missing_gt_coverage,
                min_provisional_precision=self.cfg.recovery_min_pred_precision,
            )
        )
        if recovery_target.valid.any():
            recovery_logits = request_reasoning.recovery_logits[recovery_target.valid]
            recovery_truth = recovery_target.target[recovery_target.valid]
            recovery_loss = F.binary_cross_entropy_with_logits(
                recovery_logits, recovery_truth
            )
            recovery_metrics = _safe_binary_metrics(recovery_logits, recovery_truth)
        else:
            recovery_loss = request_reasoning.recovery_logits.sum() * 0
            recovery_metrics = _safe_binary_metrics(
                request_reasoning.recovery_logits[:0], recovery_target.target[:0]
            )
        total = (
            self.cfg.geometry_weight * geometry_total
            + self.cfg.spatial_rag_weight * spatial_rag_loss
            + self.cfg.final_rag_weight * final_rag_loss
            + self.cfg.existence_weight * existence_loss
            + self.cfg.split_weight * split_loss
            + self.cfg.recovery_weight * recovery_loss
        )
        metrics.update(
            {
                "loss": total,
                "spatial_rag_bce": spatial_rag_loss,
                "final_rag_bce": final_rag_loss,
                "working_spatial_rag_bce": spatial_rag["rag_bce"],
                "working_final_rag_bce": final_rag["rag_bce"],
                "initial_spatial_rag_bce": initial_spatial_loss,
                "initial_final_rag_bce": initial_final_loss,
                "final_rag_accuracy": final_rag["rag_accuracy"],
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
                if refinement_applied
                else geometry_total.detach() * 0,
                "refinement_teacher_requests": reference_tensor.new_tensor(
                    0 if output.refinement is None else output.refinement.teacher_request_count
                ),
                "refinement_model_requests": reference_tensor.new_tensor(
                    0 if output.refinement is None else output.refinement.model_request_count
                ),
                "refinement_total_requests": reference_tensor.new_tensor(
                    0 if output.refinement is None else len(output.refinement.requests)
                ),
            }
        )
        return metrics


__all__ = [
    "InstanceTargets",
    "RecoveryTargets",
    "StirNetCriterion",
    "build_instance_targets",
    "build_recovery_targets",
    "build_teacher_refinement_requests",
    "teacher_forcing_fraction",
]
