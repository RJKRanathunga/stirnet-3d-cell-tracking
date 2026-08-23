from __future__ import annotations

"""Realistic missing-source-cell corruption for focused crop training.

Production spatial training does not fabricate merges or over-segmentation.
Natural under-segmentation comes from the real upstream source mask. The only
synthetic topology error here is deletion of a trustworthy one-to-one source
cell while raw image and GT remain unchanged.
"""

from dataclasses import replace
import torch
from torch import Tensor

from ..data.sample_builder import SPATIAL_CHANNEL_NAMES, build_spatial_channels
from .crops import CropBatch

_CH = {name: i for i, name in enumerate(SPATIAL_CHANNEL_NAMES)}


def _touches_edge(mask: Tensor) -> bool:
    return bool(
        mask.any()
        and (
            mask[0].any() or mask[-1].any()
            or mask[:, 0].any() or mask[:, -1].any()
            or mask[:, :, 0].any() or mask[:, :, -1].any()
        )
    )


def eligible_source_instance_ids(
    current: Tensor,
    gt: Tensor,
    valid: Tensor,
    complete_ids: tuple[int, ...],
    *,
    min_purity: float = 0.80,
    min_gt_coverage: float = 0.50,
) -> tuple[int, ...]:
    if not 0.0 <= min_purity <= 1.0:
        raise ValueError("min_purity must be in [0,1]")
    if not 0.0 <= min_gt_coverage <= 1.0:
        raise ValueError("min_gt_coverage must be in [0,1]")
    current = torch.as_tensor(current).detach().cpu()
    gt = torch.as_tensor(gt).detach().cpu()
    valid = torch.as_tensor(valid).detach().cpu().bool()
    complete = {int(value) for value in complete_ids}
    eligible: list[int] = []
    source_ids = sorted(
        int(value) for value in torch.unique(current).tolist() if int(value) > 0
    )
    for source_id in source_ids:
        source = current == source_id
        if _touches_edge(source):
            continue
        valid_source = source & valid
        if not bool(valid_source.any()):
            continue
        ignored_gt = {
            int(value)
            for value in torch.unique(gt[source & ~valid]).tolist()
            if int(value) > 0
        }
        if ignored_gt:
            continue
        gt_ids = sorted(
            int(value)
            for value in torch.unique(gt[valid_source]).tolist()
            if int(value) > 0
        )
        if len(gt_ids) != 1:
            continue
        gt_id = gt_ids[0]
        if gt_id not in complete:
            continue
        overlap = int((valid_source & (gt == gt_id)).sum().item())
        source_voxels = int(valid_source.sum().item())
        gt_voxels = int(((gt == gt_id) & valid).sum().item())
        purity = overlap / max(source_voxels, 1)
        coverage = overlap / max(gt_voxels, 1)
        if purity >= min_purity and coverage >= min_gt_coverage:
            eligible.append(source_id)
    return tuple(eligible)


def select_source_instance_dropout_ids(
    crop: CropBatch,
    *,
    probability: float,
    max_instances: int,
    seed: int,
    min_purity: float = 0.80,
    min_gt_coverage: float = 0.50,
) -> tuple[tuple[int, ...], ...]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("source dropout probability must be in [0,1]")
    if max_instances < 0:
        raise ValueError("max_instances cannot be negative")
    current = crop.batch.get("instance_labels")
    if probability <= 0 or max_instances == 0 or current is None:
        return tuple(() for _ in crop.specs)
    valid_batch = crop.batch.get("supervision_valid_mask")
    selected: list[tuple[int, ...]] = []
    for row, spec in enumerate(crop.specs):
        if spec.merge_source_ids:
            selected.append(())
            continue
        valid = (
            torch.ones_like(crop.gt_labels[row], dtype=torch.bool)
            if valid_batch is None
            else torch.as_tensor(valid_batch[row]).detach().cpu().bool()
        )
        candidates = eligible_source_instance_ids(
            torch.as_tensor(current[row]),
            crop.gt_labels[row],
            valid,
            spec.complete_cell_ids,
            min_purity=min_purity,
            min_gt_coverage=min_gt_coverage,
        )
        if not candidates:
            selected.append(())
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 104_729 * row)
        if float(torch.rand((), generator=generator)) >= probability:
            selected.append(())
            continue
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected.append(
            tuple(candidates[index] for index in order[: min(max_instances, len(candidates))])
        )
    return tuple(selected)


def apply_selected_source_dropout_labels(
    crop: CropBatch,
    selected: tuple[tuple[int, ...], ...],
) -> CropBatch:
    if len(selected) != len(crop.specs):
        raise ValueError("selected dropout rows must align with crop specs")
    batch = dict(crop.batch)
    current_value = batch.get("instance_labels")
    if current_value is None:
        batch["source_dropout_ids"] = tuple(selected)
        return replace(crop, batch=batch)
    current = torch.as_tensor(current_value).clone()
    for row, source_ids in enumerate(selected):
        for source_id in source_ids:
            current[row][current[row] == int(source_id)] = 0
    batch["instance_labels"] = current
    batch["source_dropout_ids"] = tuple(selected)
    return replace(crop, batch=batch)


def _legacy_rebuild_without_geometry_metadata(
    spatial: Tensor,
    original_current: Tensor,
    selected: tuple[tuple[int, ...], ...],
) -> Tensor:
    result = spatial.clone()
    for row, source_ids in enumerate(selected):
        if not source_ids:
            continue
        removed = torch.zeros_like(original_current[row], dtype=torch.bool)
        for source_id in source_ids:
            removed |= original_current[row] == int(source_id)
        labels = original_current[row].clone()
        labels[removed] = 0
        result[row, _CH["current_foreground_prior"]][removed] = 0
        result[row, _CH["current_edt_prior"]][removed] = 0
        boundary = torch.zeros_like(labels, dtype=torch.bool)
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            x, y = labels[tuple(a)], labels[tuple(b)]
            diff = (x != y) & ((x > 0) | (y > 0))
            boundary[tuple(a)] |= diff
            boundary[tuple(b)] |= diff
        result[row, _CH["current_boundary_prior"]] = boundary.to(result.dtype)
        result[row, _CH["current_marker_prior"]][removed] = 0
    return result


def apply_source_instance_dropout(
    crop: CropBatch,
    *,
    probability: float,
    max_instances: int,
    seed: int,
    min_purity: float = 0.80,
    min_gt_coverage: float = 0.50,
) -> CropBatch:
    """Compatibility path for already-materialized five-channel crop inputs."""
    selected = select_source_instance_dropout_ids(
        crop,
        probability=probability,
        max_instances=max_instances,
        seed=seed,
        min_purity=min_purity,
        min_gt_coverage=min_gt_coverage,
    )
    changed = apply_selected_source_dropout_labels(crop, selected)
    if not any(selected) or changed.batch.get("spatial_inputs") is None:
        return changed
    batch = dict(changed.batch)
    spatial = torch.as_tensor(batch["spatial_inputs"]).clone()
    spacing = batch.get("spacing_um")
    dref = batch.get("dref_um")
    if spacing is None or dref is None:
        batch["spatial_inputs"] = _legacy_rebuild_without_geometry_metadata(
            spatial, torch.as_tensor(crop.batch["instance_labels"]), selected
        )
        return replace(changed, batch=batch)
    current = torch.as_tensor(batch["instance_labels"])
    for row, source_ids in enumerate(selected):
        if not source_ids:
            continue
        raw_norm = spatial[row, _CH["raw"]].detach().cpu().numpy()
        rebuilt = build_spatial_channels(
            raw_norm,
            current[row].detach().cpu().numpy(),
            torch.as_tensor(spacing[row]).detach().cpu().numpy(),
            float(torch.as_tensor(dref[row]).detach().cpu()),
            derive_marker=True,
        )
        spatial[row] = torch.from_numpy(rebuilt).to(spatial.device, spatial.dtype)
    batch["spatial_inputs"] = spatial
    return replace(changed, batch=batch)


__all__ = [
    "apply_selected_source_dropout_labels",
    "apply_source_instance_dropout",
    "eligible_source_instance_ids",
    "select_source_instance_dropout_ids",
]
