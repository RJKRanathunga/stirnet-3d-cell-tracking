from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .coordinates import feature_grid_coordinates_um, resize_label_map_nearest
from .query_builder import (
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
)


def target_ids(target: dict) -> Tensor:
    if "ids" in target:
        return torch.as_tensor(target["ids"], dtype=torch.long)
    if "masks" in target:
        masks = torch.as_tensor(target["masks"])
        return torch.arange(len(masks), dtype=torch.long, device=masks.device)
    raise KeyError("A STIR-Net target requires 'ids' with 'label_map', or legacy 'masks'")


def valid_target_indices(target: dict) -> Tensor:
    ids = target_ids(target)
    valid = target.get("valid", target.get("gt_valid"))
    if valid is None:
        return torch.arange(len(ids), dtype=torch.long, device=ids.device)
    return torch.nonzero(torch.as_tensor(valid, dtype=torch.bool), as_tuple=False).flatten()


def target_masks_at_shape(
    target: dict,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
    *,
    target_indices: Tensor | None = None,
) -> Tensor:
    """Materialize only coarse target masks, preferably from one label map."""
    if target_indices is None:
        target_indices = valid_target_indices(target)
    target_indices_cpu = target_indices.detach().cpu().long()
    if "label_map" in target:
        labels = torch.as_tensor(target["label_map"])
        labels = resize_label_map_nearest(labels, spatial_shape)
        all_ids = target_ids(target)
        ids = all_ids[target_indices_cpu.to(all_ids.device)].to(labels.device)
        masks = labels.unsqueeze(0) == ids[:, None, None, None]
    elif "masks" in target:
        masks = torch.as_tensor(target["masks"])
        masks = masks[target_indices_cpu.to(masks.device)]
        masks = resize_label_map_nearest(masks, spatial_shape)
    else:
        raise KeyError("Target needs either 'label_map' plus 'ids', or legacy 'masks'")
    return masks.to(device=device, dtype=torch.float32, non_blocking=True)


def build_local_support_masks(
    gt_masks: Tensor,
    gt_centers_cellscale: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    radius_dref: float,
) -> Tensor:
    """Return coarse GT-local physical supports, always retaining positives."""
    if gt_masks.shape[0] == 0:
        return torch.zeros_like(gt_masks, dtype=torch.bool)
    shape = tuple(int(v) for v in gt_masks.shape[-3:])
    spacing = spacing_um.to(device=gt_masks.device, dtype=torch.float32)
    coords_um = feature_grid_coordinates_um(
        shape, spacing[None], relative_to_center=True
    )[0]
    centers_um = gt_centers_cellscale.to(
        device=gt_masks.device, dtype=torch.float32
    ) * torch.as_tensor(dref_um, device=gt_masks.device, dtype=torch.float32)
    radius_um = float(radius_dref) * torch.as_tensor(
        dref_um, device=gt_masks.device, dtype=torch.float32
    )
    radial = (
        coords_um[None] - centers_um[:, None]
    ).square().sum(dim=-1) <= radius_um.square()
    return (radial | gt_masks.bool().flatten(1)).reshape_as(gt_masks)


def _pairwise_dice_cost(
    pred_logits: Tensor,
    gt: Tensor,
    support: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """FP32 pairwise Dice cost for pred [Q,V] and gt [K,V]."""
    pred_logits = pred_logits.float()
    gt = gt.float()
    p = pred_logits.sigmoid()
    inter = 2 * torch.einsum("qv,kv->qk", p, gt)
    if support is None:
        pred_sum = p.sum(-1)[:, None].expand(-1, gt.shape[0])
    else:
        pred_sum = torch.einsum("qv,kv->qk", p, support.float())
    denom = pred_sum + gt.sum(-1)[None, :]
    return 1 - (inter + eps) / (denom + eps)


def _pairwise_focal_cost(
    pred_logits: Tensor,
    gt: Tensor,
    support: Tensor | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    """Stable FP32 focal matching cost using logit-space log probabilities."""
    logits = pred_logits.float()
    gt = gt.float()
    probability = logits.sigmoid()
    positive = -alpha * (1 - probability).pow(gamma) * F.logsigmoid(logits)
    negative = -(1 - alpha) * probability.pow(gamma) * F.logsigmoid(-logits)
    costs = []
    for k in range(gt.shape[0]):
        target = gt[k][None]
        value = positive * target + negative * (1 - target)
        if support is None:
            costs.append(value.mean(dim=-1))
        else:
            local = support[k].float()[None]
            costs.append((value * local).sum(dim=-1) / local.sum().clamp_min(1))
    return torch.stack(costs, dim=1) if costs else logits.new_zeros((logits.shape[0], 0))


def build_cost_matrix(
    exist_logits: Tensor,
    coarse_mask_logits: Tensor,
    centers_cellscale: Tensor,
    gt_masks: Tensor,
    gt_centers_cellscale: Tensor,
    *,
    gt_support: Tensor | None = None,
    w_exist: float = 2.0,
    w_dice: float = 5.0,
    w_focal: float = 2.0,
    w_center: float = 2.0,
    mask_focal_alpha_pos: float = 0.75,
    mask_focal_gamma: float = 2.0,
) -> Tensor:
    """Construct the complete Hungarian cost explicitly in FP32."""
    with torch.autocast(device_type=coarse_mask_logits.device.type, enabled=False):
        masks = coarse_mask_logits.float().flatten(1)
        gt = gt_masks.float().flatten(1)
        support = None if gt_support is None else gt_support.bool().flatten(1)
        dice = _pairwise_dice_cost(masks, gt, support)
        focal = _pairwise_focal_cost(
            masks,
            gt,
            support,
            alpha=mask_focal_alpha_pos,
            gamma=mask_focal_gamma,
        )
        exist = -F.logsigmoid(exist_logits.float())[:, None].expand(-1, gt.shape[0])
        center = torch.cdist(
            centers_cellscale.float(), gt_centers_cellscale.float(), p=1
        )
        cost = w_exist * exist + w_dice * dice + w_focal * focal + w_center * center
    if not torch.isfinite(cost).all():
        bad = ~torch.isfinite(cost)
        raise ValueError(
            "Hungarian cost contains non-finite entries after FP32 construction: "
            f"shape={tuple(cost.shape)}, nonfinite={int(bad.sum())}, "
            f"exist_finite={bool(torch.isfinite(exist).all())}, "
            f"dice_finite={bool(torch.isfinite(dice).all())}, "
            f"focal_finite={bool(torch.isfinite(focal).all())}, "
            f"center_finite={bool(torch.isfinite(center).all())}"
        )
    return cost


@dataclass
class MatchResult:
    pred_indices: Tensor
    target_indices: Tensor


class HungarianMatcher3D(nn.Module):
    def __init__(
        self,
        w_exist: float = 2.0,
        w_dice: float = 5.0,
        w_focal: float = 2.0,
        w_center: float = 2.0,
        mask_supervision_radius_dref: float = 1.5,
        mask_focal_alpha_pos: float = 0.75,
        mask_focal_gamma: float = 2.0,
        temporal_match_radius_dref: float = 1.0,
        discovery_match_radius_dref: float = 1.5,
    ):
        super().__init__()
        self.w_exist = w_exist
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_center = w_center
        self.mask_supervision_radius_dref = mask_supervision_radius_dref
        self.mask_focal_alpha_pos = mask_focal_alpha_pos
        self.mask_focal_gamma = mask_focal_gamma
        self.temporal_match_radius_dref = temporal_match_radius_dref
        self.discovery_match_radius_dref = discovery_match_radius_dref

    @staticmethod
    def _assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
        if cost.numel() == 0:
            empty = torch.empty(0, device=cost.device, dtype=torch.long)
            return empty, empty
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(row, device=cost.device, dtype=torch.long),
            torch.as_tensor(col, device=cost.device, dtype=torch.long),
        )

    @classmethod
    def _eligible_assignment(
        cls, cost: Tensor, eligible: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Maximum-cardinality, then minimum-cost assignment on eligible edges.

        The augmented square problem gives every real query and GT its own
        dummy. Its unmatched penalty dominates every possible change in real
        edge costs, so cardinality is optimized before cost. Ineligible real
        edges are more expensive than remaining unmatched and are filtered as
        a final invariant.
        """
        if cost.shape != eligible.shape:
            raise ValueError("cost and eligibility matrices must have one shape")
        query_count, target_count = cost.shape
        if query_count == 0 or target_count == 0 or not bool(eligible.any()):
            empty = torch.empty(0, device=cost.device, dtype=torch.long)
            return empty, empty
        work = cost.float()
        eligible_device = eligible.to(device=cost.device, dtype=torch.bool)
        valid_costs = work[eligible_device]
        shifted = work - valid_costs.min()
        cost_span = (valid_costs.max() - valid_costs.min()).clamp_min(1.0)
        max_cardinality = min(query_count, target_count)
        unmatched_cost = cost_span * (max_cardinality + 1)
        forbidden_cost = unmatched_cost * (query_count + target_count + 2)
        size = query_count + target_count
        augmented = work.new_full((size, size), forbidden_cost)
        augmented[:query_count, :target_count] = torch.where(
            eligible_device, shifted, forbidden_cost
        )
        augmented[:query_count, target_count:] = unmatched_cost
        augmented[query_count:, :target_count] = unmatched_cost
        augmented[query_count:, target_count:] = 0.0
        rows, cols = cls._assignment(augmented)
        real = (rows < query_count) & (cols < target_count)
        rows, cols = rows[real], cols[real]
        keep = eligible_device[rows, cols]
        rows, cols = rows[keep], cols[keep]
        if rows.unique().numel() != rows.numel() or cols.unique().numel() != cols.numel():
            raise AssertionError("Eligible assignment is not one-to-one")
        return rows, cols

    def _structured_assignment(
        self,
        cost: Tensor,
        valid_q: Tensor,
        valid_gt: Tensor,
        query_types: Tensor,
        source_instance_ids: Tensor,
        initial_references_cellscale: Tensor,
        final_centers_cellscale: Tensor,
        gt_centers_cellscale: Tensor,
        target: dict,
    ) -> tuple[Tensor, Tensor] | None:
        """Two-stage source-aware assignment in valid-query/valid-GT space."""
        if "source_ids" not in target or "source_gt_overlap" not in target:
            return None
        source_ids = torch.as_tensor(target["source_ids"], dtype=torch.long).cpu()
        overlap = torch.as_tensor(target["source_gt_overlap"], dtype=torch.long).cpu()
        all_gt_count = len(target_ids(target))
        if overlap.shape != (len(source_ids), all_gt_count):
            raise ValueError(
                "source_gt_overlap must have shape [num_sources, num_targets]; "
                f"got {tuple(overlap.shape)}, expected {(len(source_ids), all_gt_count)}"
            )
        overlap = overlap[:, valid_gt.detach().cpu().long()] > 0
        source_row = {int(source_id): row for row, source_id in enumerate(source_ids.tolist())}
        qtypes = query_types[valid_q].detach().cpu().long()
        query_sources = source_instance_ids[valid_q].detach().cpu().long()
        eligible = torch.zeros((len(valid_q), len(valid_gt)), dtype=torch.bool)
        for row, (query_type, source_id) in enumerate(
            zip(qtypes.tolist(), query_sources.tolist())
        ):
            source_index = source_row.get(int(source_id))
            if source_index is None:
                continue
            compatible = overlap[source_index]
            compatible_count = int(compatible.sum())
            if query_type == QUERY_PRIMARY and compatible_count >= 1:
                eligible[row] = compatible
            elif query_type == QUERY_SPLIT and compatible_count >= 2:
                eligible[row] = compatible

        # Stage A: compatible primary/split candidates only.
        seeded_rows = torch.nonzero(eligible.any(dim=1), as_tuple=False).flatten()
        seeded_cols = torch.nonzero(eligible.any(dim=0), as_tuple=False).flatten()
        matched_rows: list[Tensor] = []
        matched_cols: list[Tensor] = []
        used_cols = torch.zeros(len(valid_gt), dtype=torch.bool)
        if seeded_rows.numel() and seeded_cols.numel():
            eligible_sub = eligible[seeded_rows][:, seeded_cols].to(cost.device)
            seeded_cost = cost[seeded_rows.to(cost.device)][:, seeded_cols.to(cost.device)]
            row, col = self._eligible_assignment(seeded_cost, eligible_sub)
            row = seeded_rows.to(cost.device)[row]
            col = seeded_cols.to(cost.device)[col]
            if row.numel():
                matched_rows.append(row)
                matched_cols.append(col)
                used_cols[col.detach().cpu()] = True

        # Stage B1: temporal clues use their immutable initial references.
        temporal_rows = torch.nonzero(qtypes == QUERY_TEMPORAL, as_tuple=False).flatten()
        remaining_cols = torch.nonzero(~used_cols, as_tuple=False).flatten()
        if temporal_rows.numel() and remaining_cols.numel():
            temporal_refs = initial_references_cellscale[valid_q][
                temporal_rows.to(valid_q.device)
            ].float()
            remaining_centers = gt_centers_cellscale[
                remaining_cols.to(gt_centers_cellscale.device)
            ].float()
            temporal_eligible = torch.cdist(
                temporal_refs, remaining_centers, p=2
            ) <= float(self.temporal_match_radius_dref)
            temporal_cost = cost[temporal_rows.to(cost.device)][:, remaining_cols.to(cost.device)]
            row, col = self._eligible_assignment(temporal_cost, temporal_eligible)
            row = temporal_rows.to(cost.device)[row]
            col = remaining_cols.to(cost.device)[col]
            if row.numel():
                matched_rows.append(row)
                matched_cols.append(col)
                used_cols[col.detach().cpu()] = True

        # Stage B2: discovery uses decoded centers and sees only B1 leftovers.
        discovery_rows = torch.nonzero(qtypes == QUERY_DISCOVERY, as_tuple=False).flatten()
        remaining_cols = torch.nonzero(~used_cols, as_tuple=False).flatten()
        if discovery_rows.numel() and remaining_cols.numel():
            discovery_centers = final_centers_cellscale[valid_q][
                discovery_rows.to(valid_q.device)
            ].float()
            remaining_centers = gt_centers_cellscale[
                remaining_cols.to(gt_centers_cellscale.device)
            ].float()
            discovery_eligible = torch.cdist(
                discovery_centers, remaining_centers, p=2
            ) <= float(self.discovery_match_radius_dref)
            discovery_cost = cost[discovery_rows.to(cost.device)][:, remaining_cols.to(cost.device)]
            row, col = self._eligible_assignment(discovery_cost, discovery_eligible)
            row = discovery_rows.to(cost.device)[row]
            col = remaining_cols.to(cost.device)[col]
            if row.numel():
                matched_rows.append(row)
                matched_cols.append(col)
                used_cols[col.detach().cpu()] = True

        if not matched_rows:
            empty = torch.empty(0, device=cost.device, dtype=torch.long)
            return empty, empty
        rows = torch.cat(matched_rows)
        cols = torch.cat(matched_cols)
        order = torch.argsort(rows)
        rows, cols = rows[order], cols[order]
        if rows.unique().numel() != rows.numel() or cols.unique().numel() != cols.numel():
            raise AssertionError("Structured Hungarian assignment is not one-to-one")
        seeded = (qtypes[rows.cpu()] == QUERY_PRIMARY) | (
            qtypes[rows.cpu()] == QUERY_SPLIT
        )
        if seeded.any() and not eligible[rows.cpu()[seeded], cols.cpu()[seeded]].all():
            raise AssertionError("Structured Hungarian produced a source-incompatible seeded match")
        temporal = qtypes[rows.cpu()] == QUERY_TEMPORAL
        if temporal.any():
            rows_cpu = rows.detach().cpu()
            cols_cpu = cols.detach().cpu()
            initial_cpu = initial_references_cellscale.detach().float().cpu()
            gt_centers_cpu = gt_centers_cellscale.detach().float().cpu()
            distance = torch.linalg.vector_norm(
                initial_cpu[valid_q.detach().cpu()][rows_cpu[temporal]]
                - gt_centers_cpu[cols_cpu[temporal]],
                dim=-1,
            )
            if bool((distance > self.temporal_match_radius_dref + 1e-6).any()):
                raise AssertionError("Temporal match violates initial-reference radius")
        discovery = qtypes[rows.cpu()] == QUERY_DISCOVERY
        if discovery.any():
            rows_cpu = rows.detach().cpu()
            cols_cpu = cols.detach().cpu()
            final_cpu = final_centers_cellscale.detach().float().cpu()
            gt_centers_cpu = gt_centers_cellscale.detach().float().cpu()
            distance = torch.linalg.vector_norm(
                final_cpu[valid_q.detach().cpu()][rows_cpu[discovery]]
                - gt_centers_cpu[cols_cpu[discovery]],
                dim=-1,
            )
            if bool((distance > self.discovery_match_radius_dref + 1e-6).any()):
                raise AssertionError("Discovery match violates decoded-center radius")
        return rows, cols

    @torch.no_grad()
    def forward(
        self,
        output: dict[str, Tensor],
        query_padding_mask: Tensor,
        targets: list[dict],
        coarse_target_masks: list[Tensor] | None = None,
    ) -> list[MatchResult]:
        batch_size, _ = output["exist_logits"].shape
        results = []
        coarse = output["coarse_mask_logits"]
        for b in range(batch_size):
            valid_q = torch.nonzero(~query_padding_mask[b], as_tuple=False).flatten()
            valid_gt = valid_target_indices(targets[b])
            if valid_gt.numel() == 0 or valid_q.numel() == 0:
                results.append(
                    MatchResult(
                        valid_q[:0],
                        torch.empty(0, device=coarse.device, dtype=torch.long),
                    )
                )
                continue
            gt_masks = (
                coarse_target_masks[b]
                if coarse_target_masks is not None
                else target_masks_at_shape(
                    targets[b], coarse.shape[-3:], coarse.device, target_indices=valid_gt
                )
            )
            gt_centers = torch.as_tensor(targets[b]["centers_cellscale"])[valid_gt]
            gt_centers = gt_centers.to(coarse.device, dtype=torch.float32, non_blocking=True)
            gt_support = None
            if "coarse_spacing_um" in output and "dref_um" in output:
                spacing = output["coarse_spacing_um"]
                spacing_b = spacing[b] if spacing.ndim == 2 else spacing
                dref = output["dref_um"]
                dref_b = dref[b] if dref.ndim else dref
                gt_support = build_local_support_masks(
                    gt_masks,
                    gt_centers,
                    spacing_b,
                    dref_b,
                    self.mask_supervision_radius_dref,
                )
            cost = build_cost_matrix(
                output["exist_logits"][b, valid_q],
                coarse[b, valid_q],
                output["centers_cellscale"][b, valid_q],
                gt_masks,
                gt_centers,
                gt_support=gt_support,
                w_exist=self.w_exist,
                w_dice=self.w_dice,
                w_focal=self.w_focal,
                w_center=self.w_center,
                mask_focal_alpha_pos=self.mask_focal_alpha_pos,
                mask_focal_gamma=self.mask_focal_gamma,
            )
            structured = None
            if "query_types" in output and "source_instance_ids" in output:
                structured = self._structured_assignment(
                    cost,
                    valid_q,
                    valid_gt,
                    output["query_types"][b],
                    output["source_instance_ids"][b],
                    output.get(
                        "query_initial_references_cellscale",
                        output["centers_cellscale"],
                    )[b],
                    output["centers_cellscale"][b],
                    gt_centers,
                    targets[b],
                )
            row_t, col_t = (
                self._assignment(cost) if structured is None else structured
            )
            results.append(
                MatchResult(
                    valid_q[row_t],
                    valid_gt.to(coarse.device)[col_t],
                )
            )
        return results
