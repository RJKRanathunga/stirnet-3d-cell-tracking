from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import LossConfig, QueryConfig
from .matcher import (
    HungarianMatcher3D,
    MatchResult,
    target_ids,
    target_masks_at_shape,
    valid_target_indices,
)
from .query_builder import QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL
from .types import StirNetOutput


def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - pt).pow(gamma) * ce
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    probability = logits.sigmoid().flatten(1)
    target = targets.float().flatten(1)
    score = (2 * (probability * target).sum(-1) + eps) / (
        probability.sum(-1) + target.sum(-1) + eps
    )
    return (1 - score).mean() if score.numel() else logits.sum() * 0


def _target_count(target: dict) -> int:
    return int(valid_target_indices(target).numel())


def _tensor_chunk(tensor: Tensor, start: int, end: int, device: torch.device) -> Tensor:
    flat = tensor.reshape(-1)
    return flat[start:end].to(device=device, non_blocking=True)


class RefinementCriterion(nn.Module):
    def __init__(self, loss_cfg: LossConfig, query_cfg: QueryConfig):
        super().__init__()
        self.cfg = loss_cfg
        self.query_cfg = query_cfg
        self.matcher = HungarianMatcher3D()

    def _coarse_targets(self, out: dict[str, Tensor], targets: list[dict]) -> list[Tensor]:
        shape = tuple(int(v) for v in out["coarse_mask_logits"].shape[-3:])
        device = out["coarse_mask_logits"].device
        result = []
        for target in targets:
            all_indices = torch.arange(len(target_ids(target)), dtype=torch.long)
            result.append(
                target_masks_at_shape(
                    target, shape, device, target_indices=all_indices
                )
            )
        return result

    def _match(
        self,
        out: dict[str, Tensor],
        padding: Tensor,
        targets: list[dict],
        coarse_targets: list[Tensor],
    ) -> list[MatchResult]:
        valid_masks = []
        for target, masks in zip(targets, coarse_targets):
            indices = valid_target_indices(target).to(masks.device)
            valid_masks.append(masks[indices])
        return self.matcher(out, padding, targets, valid_masks)

    def _existence_loss(
        self, logits: Tensor, padding: Tensor, matches: list[MatchResult]
    ) -> Tensor:
        target = torch.zeros_like(logits)
        valid = ~padding
        for b, match in enumerate(matches):
            target[b, match.pred_indices] = 1
        if not valid.any():
            return logits.sum() * 0
        return binary_focal_loss_with_logits(
            logits[valid],
            target[valid],
            alpha=self.cfg.exist_focal_alpha_pos,
            gamma=self.cfg.exist_focal_gamma,
        )

    def _coarse_losses(
        self,
        out: dict[str, Tensor],
        matches: list[MatchResult],
        targets: list[dict],
        coarse_targets: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        masks_pred = []
        masks_target = []
        centers_pred = []
        centers_target = []
        for b, match in enumerate(matches):
            if match.pred_indices.numel() == 0:
                continue
            pred = out["coarse_mask_logits"][b, match.pred_indices]
            masks_pred.append(pred)
            masks_target.append(coarse_targets[b][match.target_indices])
            centers_pred.append(out["centers_cellscale"][b, match.pred_indices])
            target_centers = torch.as_tensor(targets[b]["centers_cellscale"])
            centers_target.append(
                target_centers[match.target_indices.to(target_centers.device)].to(
                    pred.device, dtype=out["centers_cellscale"].dtype, non_blocking=True
                )
            )
        zero = out["exist_logits"].sum() * 0
        if not masks_pred:
            return zero, zero, zero
        pred = torch.cat(masks_pred)
        target = torch.cat(masks_target)
        pred_centers = torch.cat(centers_pred)
        target_centers = torch.cat(centers_target)
        return (
            dice_loss(pred, target),
            binary_focal_loss_with_logits(pred, target),
            F.smooth_l1_loss(pred_centers, target_centers),
        )

    def _count_loss(self, logits: Tensor, padding: Tensor, targets: list[dict]) -> Tensor:
        probability = torch.sigmoid(logits).masked_fill(padding, 0)
        predicted = probability.sum(dim=-1)
        target_count = torch.tensor(
            [_target_count(target) for target in targets],
            device=logits.device,
            dtype=logits.dtype,
        )
        return (
            F.smooth_l1_loss(predicted, target_count, reduction="none")
            / target_count.clamp_min(1)
        ).mean()

    def _overlap_loss(self, coarse: Tensor, matches: list[MatchResult]) -> Tensor:
        values = []
        for b, match in enumerate(matches):
            if match.pred_indices.numel() < 2:
                continue
            summed = coarse[b, match.pred_indices].sigmoid().sum(dim=0)
            values.append(F.relu(summed - 1).pow(2).mean())
        return torch.stack(values).mean() if values else coarse.sum() * 0

    def _native_target_chunk(
        self,
        target: dict,
        matched_target_indices: Tensor,
        start: int,
        end: int,
        device: torch.device,
    ) -> Tensor:
        indices_cpu = matched_target_indices.detach().cpu().long()
        if "label_map" in target:
            labels = _tensor_chunk(torch.as_tensor(target["label_map"]), start, end, device)
            all_ids = target_ids(target)
            ids = all_ids[indices_cpu.to(all_ids.device)].to(device)
            return (labels[None] == ids[:, None]).float()
        masks = torch.as_tensor(target["masks"])
        selected = masks[indices_cpu.to(masks.device)].reshape(len(indices_cpu), -1)
        return selected[:, start:end].to(device=device, dtype=torch.float32, non_blocking=True)

    def _native_mask_losses(
        self,
        outputs: StirNetOutput,
        targets: list[dict],
        matches: list[MatchResult],
    ) -> tuple[Tensor, Tensor]:
        """Exact matched native Dice/focal losses with bounded spatial chunks."""
        dice_values = []
        focal_values = []
        _, _, z_size, y_size, x_size = outputs.mask_features.shape
        voxel_count = z_size * y_size * x_size
        spatial_chunk = max(1, int(self.cfg.native_chunk_voxels))
        query_chunk = 4

        for b, match in enumerate(matches):
            if match.pred_indices.numel() == 0:
                continue
            feature_flat = outputs.mask_features[b].reshape(outputs.mask_features.shape[1], -1)
            current_labels = outputs.instance_labels[b].reshape(-1)
            spacing = outputs.spacing_um[b].float()
            extent = torch.tensor(
                [z_size - 1, y_size - 1, x_size - 1],
                device=feature_flat.device,
                dtype=torch.float32,
            ) * spacing

            for query_start in range(0, match.pred_indices.numel(), query_chunk):
                query_end = min(query_start + query_chunk, match.pred_indices.numel())
                pred_indices = match.pred_indices[query_start:query_end]
                gt_indices = match.target_indices[query_start:query_end]
                embeddings = outputs.native_mask_embeddings[b, pred_indices]
                query_types = outputs.query_types[b, pred_indices]
                source_ids = outputs.source_instance_ids[b, pred_indices]
                refs_um = outputs.centers_cellscale[b, pred_indices].float() * outputs.dref_um[b].float()
                count = query_end - query_start
                intersection = torch.zeros(count, device=feature_flat.device, dtype=torch.float32)
                probability_sum = torch.zeros_like(intersection)
                target_sum = torch.zeros_like(intersection)
                focal_sum = torch.zeros_like(intersection)

                for start in range(0, voxel_count, spatial_chunk):
                    end = min(start + spatial_chunk, voxel_count)
                    logits = torch.einsum("qc,cv->qv", embeddings, feature_flat[:, start:end])
                    prior = torch.zeros_like(logits)

                    seeded = (query_types == QUERY_PRIMARY) | (query_types == QUERY_SPLIT)
                    if seeded.any():
                        inside = current_labels[start:end][None] == source_ids[seeded, None]
                        seeded_prior = torch.where(
                            inside,
                            logits.new_tensor(self.query_cfg.prior_inside_logit),
                            logits.new_tensor(self.query_cfg.prior_outside_logit),
                        )
                        prior[seeded] = seeded_prior.to(prior.dtype)

                    temporal = query_types == QUERY_TEMPORAL
                    if temporal.any():
                        linear = torch.arange(start, end, device=feature_flat.device)
                        z_coord = torch.div(linear, y_size * x_size, rounding_mode="floor")
                        remainder = linear.remainder(y_size * x_size)
                        y_coord = torch.div(remainder, x_size, rounding_mode="floor")
                        x_coord = remainder.remainder(x_size)
                        coords = torch.stack([z_coord, y_coord, x_coord], dim=-1).float()
                        coords_um = coords * spacing[None] - 0.5 * extent[None]
                        delta = coords_um[None] - refs_um[temporal, None]
                        sigma = self.query_cfg.temporal_gaussian_sigma_dref * outputs.dref_um[b].float()
                        temporal_prior = self.query_cfg.prior_inside_logit * torch.exp(
                            -0.5 * delta.square().sum(dim=-1) / sigma.clamp_min(1e-6).square()
                        )
                        prior[temporal] = temporal_prior.to(prior.dtype)

                    target_chunk = self._native_target_chunk(
                        targets[b], gt_indices, start, end, feature_flat.device
                    )
                    with torch.autocast(device_type=feature_flat.device.type, enabled=False):
                        work_logits = (logits + prior).float()
                        probability = work_logits.sigmoid()
                        intersection = intersection + (probability * target_chunk).sum(dim=-1)
                        probability_sum = probability_sum + probability.sum(dim=-1)
                        target_sum = target_sum + target_chunk.sum(dim=-1)
                        focal_sum = focal_sum + binary_focal_loss_with_logits(
                            work_logits, target_chunk, reduction="none"
                        ).sum(dim=-1)

                dice_values.append(
                    1 - (2 * intersection + 1e-6) / (
                        probability_sum + target_sum + 1e-6
                    )
                )
                focal_values.append(focal_sum / voxel_count)

        zero = outputs.exist_logits.sum() * 0
        if not dice_values:
            return zero, zero
        return torch.cat(dice_values).mean(), torch.cat(focal_values).mean()

    def _dense_target_chunk(
        self,
        target: dict,
        key: str,
        start: int,
        end: int,
        device: torch.device,
    ) -> Tensor | None:
        if key in target:
            return _tensor_chunk(torch.as_tensor(target[key]), start, end, device).float()
        if key == "foreground" and "label_map" in target:
            labels = _tensor_chunk(torch.as_tensor(target["label_map"]), start, end, device)
            return (labels > 0).float()
        return None

    def _stream_dense_loss(
        self,
        logits: Tensor,
        targets: list[dict],
        key: str,
        *,
        focal: bool = False,
        pos_weight: float | None = None,
        include_dice: bool = False,
    ) -> Tensor:
        flat_logits = logits[:, 0].reshape(logits.shape[0], -1)
        voxel_count = flat_logits.shape[1]
        chunk_size = max(1, int(self.cfg.dense_chunk_voxels))
        element_sum = logits.sum() * 0
        element_count = 0
        dice_values = []
        for b, target in enumerate(targets):
            intersection = logits.sum() * 0
            probability_sum = logits.sum() * 0
            target_sum = logits.sum() * 0
            has_target = True
            for start in range(0, voxel_count, chunk_size):
                end = min(start + chunk_size, voxel_count)
                target_chunk = self._dense_target_chunk(
                    target, key, start, end, logits.device
                )
                if target_chunk is None:
                    has_target = False
                    break
                with torch.autocast(device_type=logits.device.type, enabled=False):
                    pred = flat_logits[b, start:end].float()
                    if focal:
                        value = binary_focal_loss_with_logits(
                            pred, target_chunk, reduction="sum"
                        )
                    else:
                        weight = None
                        if pos_weight is not None:
                            weight = torch.tensor(
                                pos_weight, device=pred.device, dtype=pred.dtype
                            )
                        value = F.binary_cross_entropy_with_logits(
                            pred, target_chunk, pos_weight=weight, reduction="sum"
                        )
                    element_sum = element_sum + value
                    element_count += end - start
                    if include_dice:
                        probability = pred.sigmoid()
                        intersection = intersection + (probability * target_chunk).sum()
                        probability_sum = probability_sum + probability.sum()
                        target_sum = target_sum + target_chunk.sum()
            if has_target and include_dice:
                dice_values.append(
                    1 - (2 * intersection + 1e-6) / (
                        probability_sum + target_sum + 1e-6
                    )
                )
        if element_count == 0:
            return logits.sum() * 0
        result = element_sum / element_count
        if include_dice and dice_values:
            result = result + torch.stack(dice_values).mean()
        return result

    def _dense_losses(
        self, dense: dict[str, Tensor], targets: list[dict]
    ) -> tuple[Tensor, Tensor, Tensor]:
        foreground = self._stream_dense_loss(
            dense["foreground_logits"], targets, "foreground", include_dice=True
        )
        center = self._stream_dense_loss(
            dense["center_heatmap_logits"], targets, "center_heatmap", focal=True
        )
        boundary = self._stream_dense_loss(
            dense["boundary_logits"],
            targets,
            "boundary",
            pos_weight=self.cfg.boundary_pos_weight,
            include_dice=True,
        )
        return foreground, center, boundary

    def forward(self, outputs: StirNetOutput, targets: list[dict]) -> dict[str, Tensor]:
        final = {
            "exist_logits": outputs.exist_logits,
            "centers_cellscale": outputs.centers_cellscale,
            "coarse_mask_logits": outputs.coarse_mask_logits,
        }
        final_targets = self._coarse_targets(final, targets)
        matches = self._match(final, outputs.query_padding_mask, targets, final_targets)
        loss_exist = self._existence_loss(
            outputs.exist_logits, outputs.query_padding_mask, matches
        )
        loss_coarse_dice, loss_coarse_focal, loss_center = self._coarse_losses(
            final, matches, targets, final_targets
        )
        loss_count = self._count_loss(
            outputs.exist_logits, outputs.query_padding_mask, targets
        )
        loss_overlap = self._overlap_loss(outputs.coarse_mask_logits, matches)
        loss_high_dice, loss_high_focal = self._native_mask_losses(
            outputs, targets, matches
        )
        loss_foreground, loss_heatmap, loss_boundary = self._dense_losses(
            outputs.dense_outputs, targets
        )

        total = (
            self.cfg.exist * loss_exist
            + self.cfg.dice_hi * loss_high_dice
            + self.cfg.focal_hi * loss_high_focal
            + self.cfg.dice_coarse * loss_coarse_dice
            + self.cfg.focal_coarse * loss_coarse_focal
            + self.cfg.center * loss_center
            + self.cfg.count * loss_count
            + self.cfg.overlap * loss_overlap
            + self.cfg.foreground * loss_foreground
            + self.cfg.center_heatmap * loss_heatmap
            + self.cfg.boundary * loss_boundary
        )

        zero = outputs.exist_logits.sum() * 0
        aux_total = zero
        for aux in outputs.aux_outputs:
            aux_targets = self._coarse_targets(aux, targets)
            aux_matches = self._match(
                aux, outputs.query_padding_mask, targets, aux_targets
            )
            aux_exist = self._existence_loss(
                aux["exist_logits"], outputs.query_padding_mask, aux_matches
            )
            aux_dice, aux_focal, aux_center = self._coarse_losses(
                aux, aux_matches, targets, aux_targets
            )
            aux_total = aux_total + self.cfg.aux_layer * (
                self.cfg.exist * aux_exist
                + self.cfg.dice_coarse * aux_dice
                + self.cfg.focal_coarse * aux_focal
                + self.cfg.center * aux_center
            )
        total = total + aux_total
        return {
            "loss": total,
            "exist": loss_exist,
            "dice_hi": loss_high_dice,
            "focal_hi": loss_high_focal,
            "dice_coarse": loss_coarse_dice,
            "focal_coarse": loss_coarse_focal,
            "center": loss_center,
            "count": loss_count,
            "overlap": loss_overlap,
            "foreground": loss_foreground,
            "center_heatmap": loss_heatmap,
            "boundary": loss_boundary,
            "aux": aux_total,
        }
