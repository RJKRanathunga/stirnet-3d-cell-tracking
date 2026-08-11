from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...model.query_builder import QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL


@dataclass
class MaskMetricState:
    intersection: float = 0.0
    probability_sum: float = 0.0
    target_sum: float = 0.0
    hard_intersection: int = 0
    hard_predicted: int = 0
    hard_target: int = 0
    inside_probability_sum: float = 0.0
    inside_count: int = 0
    outside_probability_sum: float = 0.0
    outside_count: int = 0

    def update(self, probability: torch.Tensor, target: torch.Tensor, threshold: float):
        p = probability.float()
        t = target.bool()
        self.intersection += float((p * t.float()).sum().cpu())
        self.probability_sum += float(p.sum().cpu())
        self.target_sum += float(t.sum().cpu())
        hard = p > threshold
        self.hard_intersection += int((hard & t).sum().cpu())
        self.hard_predicted += int(hard.sum().cpu())
        self.hard_target += int(t.sum().cpu())
        if t.any():
            self.inside_probability_sum += float(p[t].sum().cpu())
            self.inside_count += int(t.sum().cpu())
        outside = ~t
        if outside.any():
            self.outside_probability_sum += float(p[outside].sum().cpu())
            self.outside_count += int(outside.sum().cpu())

    def summary(self, prefix: str):
        soft_dice = (2.0 * self.intersection + 1e-6) / (self.probability_sum + self.target_sum + 1e-6)
        hard_dice = (2.0 * self.hard_intersection + 1e-6) / (self.hard_predicted + self.hard_target + 1e-6)
        return {
            f"{prefix}_soft_dice": float(soft_dice),
            f"{prefix}_hard_dice": float(hard_dice),
            f"{prefix}_predicted_voxels": int(self.hard_predicted),
            f"{prefix}_target_voxels": int(self.hard_target),
            f"{prefix}_volume_ratio": self.hard_predicted / max(self.hard_target, 1),
            f"{prefix}_mean_prob_inside_gt": self.inside_probability_sum / max(self.inside_count, 1),
            f"{prefix}_mean_prob_outside_gt": self.outside_probability_sum / max(self.outside_count, 1),
        }


def _coords(start, end, *, y_size, x_size, device):
    linear = torch.arange(start, end, device=device, dtype=torch.long)
    z = torch.div(linear, y_size * x_size, rounding_mode="floor")
    rem = linear.remainder(y_size * x_size)
    y = torch.div(rem, x_size, rounding_mode="floor")
    x = rem.remainder(x_size)
    return torch.stack([z, y, x], dim=-1)


def _prior_logits_chunk(outputs, b, q, start, end, *, prior_inside_logit, prior_outside_logit, temporal_sigma_dref):
    device = outputs.mask_features.device
    dtype = outputs.mask_features.dtype
    _, _, z_size, y_size, x_size = outputs.mask_features.shape
    qtype = int(outputs.query_types[b, q].item())
    source_id = int(outputs.source_instance_ids[b, q].item())
    prior = torch.zeros(end - start, device=device, dtype=dtype)

    if qtype in (int(QUERY_PRIMARY), int(QUERY_SPLIT)) and source_id >= 0:
        labels = outputs.instance_labels[b].reshape(-1)[start:end]
        return torch.where(
            labels == source_id,
            prior.new_tensor(prior_inside_logit),
            prior.new_tensor(prior_outside_logit),
        )

    if qtype == int(QUERY_TEMPORAL):
        coords = _coords(start, end, y_size=y_size, x_size=x_size, device=device).float()
        spacing = outputs.spacing_um[b].float()
        extent = torch.tensor([z_size - 1, y_size - 1, x_size - 1], device=device, dtype=torch.float32) * spacing
        coords_um = coords * spacing[None] - 0.5 * extent[None]
        ref_um = outputs.centers_cellscale[b, q].float() * outputs.dref_um[b].float()
        sigma = temporal_sigma_dref * outputs.dref_um[b].float()
        delta = coords_um - ref_um[None]
        return (prior_inside_logit * torch.exp(-0.5 * delta.square().sum(-1) / sigma.clamp_min(1e-6).square())).to(dtype)

    return prior


def _target_chunk(target, target_index, start, end, device):
    if target_index is None:
        return torch.zeros(end - start, device=device, dtype=torch.bool)
    ids = torch.as_tensor(target["ids"], dtype=torch.long)
    gt_id = int(ids[target_index])
    labels = torch.as_tensor(target["label_map"]).reshape(-1)[start:end].to(device=device, non_blocking=True)
    return labels == gt_id


def _crop_bounds(outputs, target, target_index, q, *, margin_dref, unmatched_radius_dref):
    shape = np.asarray(outputs.instance_labels.shape[-3:], dtype=int)
    spacing = outputs.spacing_um[0].detach().float().cpu().numpy()
    dref = float(outputs.dref_um[0].detach().cpu())
    extent = (shape - 1) * spacing
    center_rel = outputs.centers_cellscale[0, q].detach().float().cpu().numpy() * dref
    center_vox = (center_rel + 0.5 * extent) / spacing

    coords = np.empty((0, 3), dtype=int)
    if target_index is not None:
        ids = torch.as_tensor(target["ids"], dtype=torch.long)
        gt_id = int(ids[target_index])
        label_map = torch.as_tensor(target["label_map"]).cpu().numpy()
        coords = np.argwhere(label_map == gt_id)

    if len(coords):
        low = coords.min(0)
        high = coords.max(0) + 1
        margin = np.ceil((margin_dref * dref) / spacing).astype(int)
        low = np.minimum(low, np.floor(center_vox).astype(int)) - margin
        high = np.maximum(high, np.ceil(center_vox).astype(int) + 1) + margin
    else:
        radius = np.ceil((unmatched_radius_dref * dref) / spacing).astype(int)
        c = np.rint(center_vox).astype(int)
        low, high = c - radius, c + radius + 1

    low = np.maximum(low, 0)
    high = np.minimum(high, shape)
    high = np.maximum(high, low + 1)
    return low.astype(int), high.astype(int), center_vox.astype(np.float32)


@torch.no_grad()
def _render_crop(outputs, target, target_index, q, low, high, *, prior_inside_logit, prior_outside_logit, temporal_sigma_dref, mask_threshold, out_dtype):
    b = 0
    z0, y0, x0 = map(int, low)
    z1, y1, x1 = map(int, high)
    feature = outputs.mask_features[b, :, z0:z1, y0:y1, x0:x1]
    embedding = outputs.native_mask_embeddings[b, q]
    learned = torch.einsum("c,czyx->zyx", embedding, feature).float()

    shape = outputs.instance_labels.shape[-3:]
    spacing = outputs.spacing_um[b].float()
    dref = outputs.dref_um[b].float()
    extent = torch.tensor([shape[0]-1, shape[1]-1, shape[2]-1], device=feature.device, dtype=torch.float32) * spacing
    zz, yy, xx = torch.meshgrid(
        torch.arange(z0, z1, device=feature.device),
        torch.arange(y0, y1, device=feature.device),
        torch.arange(x0, x1, device=feature.device),
        indexing="ij",
    )
    coords_um = torch.stack([zz, yy, xx], dim=-1).float() * spacing - 0.5 * extent
    qtype = int(outputs.query_types[b, q].item())
    source_id = int(outputs.source_instance_ids[b, q].item())
    prior = torch.zeros_like(learned)

    if qtype in (int(QUERY_PRIMARY), int(QUERY_SPLIT)) and source_id >= 0:
        labels = outputs.instance_labels[b, z0:z1, y0:y1, x0:x1]
        prior = torch.where(labels == source_id, learned.new_tensor(prior_inside_logit), learned.new_tensor(prior_outside_logit))
    elif qtype == int(QUERY_TEMPORAL):
        ref_um = outputs.centers_cellscale[b, q].float() * dref
        delta = coords_um - ref_um
        sigma = temporal_sigma_dref * dref
        prior = prior_inside_logit * torch.exp(-0.5 * delta.square().sum(-1) / sigma.clamp_min(1e-6).square())

    learned_prob = torch.sigmoid(learned)
    prior_prob = torch.sigmoid(prior)
    combined_prob = torch.sigmoid(learned + prior)

    if target_index is not None:
        ids = torch.as_tensor(target["ids"], dtype=torch.long)
        gt_id = int(ids[target_index])
        gt = (torch.as_tensor(target["label_map"])[z0:z1, y0:y1, x0:x1] == gt_id).numpy()
    else:
        gt = np.zeros((z1-z0, y1-y0, x1-x0), dtype=bool)

    np_dtype = np.float16 if out_dtype == "float16" else np.float32
    return {
        "learned_prob": learned_prob.cpu().numpy().astype(np_dtype, copy=False),
        "prior_prob": prior_prob.cpu().numpy().astype(np_dtype, copy=False),
        "combined_prob": combined_prob.cpu().numpy().astype(np_dtype, copy=False),
        "learned_binary": (learned_prob > mask_threshold).cpu().numpy(),
        "prior_binary": (prior_prob > mask_threshold).cpu().numpy(),
        "combined_binary": (combined_prob > mask_threshold).cpu().numpy(),
        "gt_mask": gt.astype(bool, copy=False),
    }


@torch.no_grad()
def probe_native_masks(outputs, target, selected_queries, query_to_target, *, mask_threshold, chunk_voxels, prior_inside_logit, prior_outside_logit, temporal_sigma_dref, crop_margin_dref, unmatched_crop_radius_dref, out_dtype):
    """Stream full-scene mask metrics; store only compact query-centric crops."""

    b = 0
    feature_flat = outputs.mask_features[b].reshape(outputs.mask_features.shape[1], -1)
    voxel_count = feature_flat.shape[1]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for q in selected_queries:
        q = int(q)
        target_index = query_to_target.get((b, q))
        learned_state, prior_state, combined_state = MaskMetricState(), MaskMetricState(), MaskMetricState()
        embedding = outputs.native_mask_embeddings[b, q]

        for start in range(0, voxel_count, chunk_voxels):
            end = min(start + chunk_voxels, voxel_count)
            learned_logits = torch.einsum("c,cv->v", embedding, feature_flat[:, start:end]).float()
            prior_logits = _prior_logits_chunk(
                outputs, b, q, start, end,
                prior_inside_logit=prior_inside_logit,
                prior_outside_logit=prior_outside_logit,
                temporal_sigma_dref=temporal_sigma_dref,
            ).float()
            target_chunk = _target_chunk(target, target_index, start, end, learned_logits.device)
            learned_state.update(torch.sigmoid(learned_logits), target_chunk, mask_threshold)
            prior_state.update(torch.sigmoid(prior_logits), target_chunk, mask_threshold)
            combined_state.update(torch.sigmoid(learned_logits + prior_logits), target_chunk, mask_threshold)

        low, high, center_vox = _crop_bounds(
            outputs, target, target_index, q,
            margin_dref=crop_margin_dref,
            unmatched_radius_dref=unmatched_crop_radius_dref,
        )
        crop = _render_crop(
            outputs, target, target_index, q, low, high,
            prior_inside_logit=prior_inside_logit,
            prior_outside_logit=prior_outside_logit,
            temporal_sigma_dref=temporal_sigma_dref,
            mask_threshold=mask_threshold,
            out_dtype=out_dtype,
        )
        prefix = f"masks/q{q:03d}"
        for name, value in crop.items():
            arrays[f"{prefix}/{name}"] = np.asarray(value)

        row = {
            "query": q,
            "target_index": int(target_index) if target_index is not None else -1,
            "crop_z0": int(low[0]), "crop_y0": int(low[1]), "crop_x0": int(low[2]),
            "crop_z1": int(high[0]), "crop_y1": int(high[1]), "crop_x1": int(high[2]),
            "pred_center_z_vox": float(center_vox[0]),
            "pred_center_y_vox": float(center_vox[1]),
            "pred_center_x_vox": float(center_vox[2]),
        }
        row.update(learned_state.summary("learned"))
        row.update(prior_state.summary("prior"))
        row.update(combined_state.summary("combined"))
        rows.append(row)

    return rows, arrays
