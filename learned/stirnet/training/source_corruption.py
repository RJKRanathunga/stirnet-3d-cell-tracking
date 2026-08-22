from __future__ import annotations

"""Structured deletion of trustworthy source instances for crop training."""

from dataclasses import replace
import torch
from torch import Tensor
import torch.nn.functional as F
from ..data.sample_builder import SPATIAL_CHANNEL_NAMES
from .crops import CropBatch

_CH = {name: i for i, name in enumerate(SPATIAL_CHANNEL_NAMES)}


def _touches_edge(mask: Tensor) -> bool:
    return bool(mask.any() and (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any() or mask[:,:,0].any() or mask[:,:,-1].any()))


def _boundary(labels: Tensor) -> Tensor:
    out = torch.zeros_like(labels, dtype=torch.bool)
    for axis in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[axis] = slice(0, -1); b[axis] = slice(1, None)
        x, y = labels[tuple(a)], labels[tuple(b)]
        diff = (x != y) & ((x > 0) | (y > 0))
        out[tuple(a)] |= diff; out[tuple(b)] |= diff
    return out


def _eligible(current: Tensor, gt: Tensor, valid: Tensor, complete_ids: tuple[int, ...]):
    complete = set(int(v) for v in complete_ids); rows = []
    for sid in [int(v) for v in torch.unique(current).tolist() if int(v) > 0]:
        mask = current == sid
        if _touches_edge(mask): continue
        gids = {int(v) for v in torch.unique(gt[mask & valid]).tolist() if int(v) > 0}
        ignored = {int(v) for v in torch.unique(gt[mask & ~valid]).tolist() if int(v) > 0}
        if len(gids) == 1 and next(iter(gids)) in complete and not ignored: rows.append(sid)
    return rows


def apply_source_instance_dropout(crop: CropBatch, *, probability: float, max_instances: int, seed: int) -> CropBatch:
    if not 0.0 <= probability <= 1.0: raise ValueError('source dropout probability must be in [0,1]')
    batch = dict(crop.batch)
    if probability <= 0 or max_instances <= 0 or batch.get('instance_labels') is None:
        batch['source_dropout_ids'] = tuple(() for _ in crop.specs)
        return replace(crop, batch=batch)
    current = torch.as_tensor(batch['instance_labels']).clone(); spatial = torch.as_tensor(batch['spatial_inputs']).clone()
    valid_batch = batch.get('supervision_valid_mask'); dropped = []
    for row, spec in enumerate(crop.specs):
        if spec.merge_source_ids:
            dropped.append(()); continue
        valid = torch.ones_like(crop.gt_labels[row], dtype=torch.bool) if valid_batch is None else torch.as_tensor(valid_batch[row]).detach().cpu().bool()
        candidates = _eligible(current[row].detach().cpu(), crop.gt_labels[row].detach().cpu(), valid, spec.complete_cell_ids)
        if not candidates:
            dropped.append(()); continue
        gen = torch.Generator(device='cpu'); gen.manual_seed(int(seed) + 104729 * row)
        if float(torch.rand((), generator=gen)) >= probability:
            dropped.append(()); continue
        order = torch.randperm(len(candidates), generator=gen).tolist(); chosen = tuple(candidates[i] for i in order[:min(max_instances, len(candidates))]); dropped.append(chosen)
        mask_cpu = torch.zeros_like(current[row], dtype=torch.bool)
        for sid in chosen: mask_cpu |= current[row] == sid
        current[row][mask_cpu] = 0
        mask = mask_cpu.to(spatial.device)
        spatial[row, _CH['current_foreground_prior']][mask] = 0
        spatial[row, _CH['current_edt_prior']][mask] = 0
        marker_mask = F.max_pool3d(mask[None,None].float(), 3, stride=1, padding=1)[0,0] > 0
        spatial[row, _CH['current_marker_prior']][marker_mask] = 0
        spatial[row, _CH['current_boundary_prior']] = _boundary(current[row].detach().cpu()).to(spatial.device, spatial.dtype)
    batch['instance_labels'] = current; batch['spatial_inputs'] = spatial; batch['source_dropout_ids'] = tuple(dropped)
    return replace(crop, batch=batch)


__all__ = ['apply_source_instance_dropout']
