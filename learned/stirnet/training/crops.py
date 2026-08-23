from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor
import torch.nn.functional as F

from ..model.geometry.targets import (
    GeometryTargets,
    build_source_conditioned_separator_interface,
)
from ..model.utils.tensor_ops import reduce_labeled_voxels


_CANDIDATE_ORDER = (
    "foreground",
    "separator",
    "disagreement",
    "absent_prior",
    "crowded",
    "background",
    "random",
)
_MAX_CANDIDATES_PER_KIND = 4096


@dataclass(frozen=True)
class CropSpec:
    batch_index: int
    slices_zyx: tuple[slice, slice, slice]
    full_shape_zyx: tuple[int, int, int]
    center_shift_um: Tensor
    candidate_type: str
    complete_cell_ids: tuple[int, ...] = ()
    partial_cell_ids: tuple[int, ...] = ()
    true_boundary_cell_ids: tuple[int, ...] = ()
    merge_source_ids: tuple[int, ...] = ()

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(
            int(axis.stop) - int(axis.start) for axis in self.slices_zyx
        )


@dataclass(frozen=True)
class CropCandidateCache:
    candidates: List[Dict[str, Tensor]]
    source_signature: tuple


@dataclass(frozen=True)
class CropBatch:
    batch: dict
    gt_labels: Tensor
    geometry_targets: GeometryTargets | None
    specs: List[CropSpec]


def _bounded_candidates(points: Tensor, maximum: int = _MAX_CANDIDATES_PER_KIND) -> Tensor:
    points = points.detach().cpu().long().reshape(-1, 3)
    if points.shape[0] <= maximum:
        return points
    indices = torch.linspace(0, points.shape[0] - 1, maximum).round().long()
    return points[indices]


def _interface_candidates(labels: Tensor) -> Tensor:
    rows: list[Tensor] = []
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        lower = labels[tuple(lower_slice)]
        upper = labels[tuple(upper_slice)]
        interface = (lower > 0) & (upper > 0) & (lower != upper)
        points = torch.nonzero(interface, as_tuple=False)
        if points.numel():
            # Alternating the two face-adjacent voxel centres avoids a fixed
            # low-side bias while retaining integer crop centres.
            points[1::2, axis] += 1
            rows.append(points)
    if not rows:
        return labels.new_zeros((0, 3)).cpu()
    return _bounded_candidates(torch.cat(rows, dim=0))



def _source_conditioned_interface_candidates(
    labels: Tensor,
    current_labels: Tensor,
    spacing_um: Tensor,
) -> Tensor:
    """Sample both direct GT contacts and current-source corrective sheets."""
    direct = _interface_candidates(labels)
    corrective_mask = build_source_conditioned_separator_interface(
        labels.detach().cpu().numpy(),
        current_labels.detach().cpu().numpy(),
        spacing_um.detach().cpu().numpy(),
    )
    corrective = torch.nonzero(
        torch.from_numpy(corrective_mask), as_tuple=False
    ).long()
    rows = [row for row in (direct, corrective) if row.numel()]
    if not rows:
        return labels.new_zeros((0, 3)).cpu()
    return _bounded_candidates(torch.cat(rows, dim=0))


def _disagreement_candidates(gt: Tensor, current: Tensor) -> Tensor:
    """Find foreground and topology errors without comparing arbitrary IDs."""
    rows = [
        _bounded_candidates(
            torch.nonzero((gt > 0) != (current > 0), as_tuple=False),
            _MAX_CANDIDATES_PER_KIND // 2,
        )
    ]
    per_axis = max(_MAX_CANDIDATES_PER_KIND // 6, 1)
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        gt_edge = gt[tuple(lower_slice)] != gt[tuple(upper_slice)]
        current_edge = (
            current[tuple(lower_slice)] != current[tuple(upper_slice)]
        )
        points = torch.nonzero(gt_edge != current_edge, as_tuple=False)
        if points.numel():
            points[1::2, axis] += 1
            rows.append(_bounded_candidates(points, per_axis))
    nonempty = [row for row in rows if row.numel()]
    return (
        _bounded_candidates(torch.cat(nonempty, dim=0))
        if nonempty
        else gt.new_zeros((0, 3)).cpu()
    )


def _background_candidates(labels: Tensor) -> Tensor:
    """Sample a coarse deterministic background lattice with O(K) storage."""
    flat = labels.reshape(-1)
    stride = max(flat.numel() // _MAX_CANDIDATES_PER_KIND, 1)
    indices = torch.arange(0, flat.numel(), stride, dtype=torch.long)
    indices = indices[flat[indices] == 0][:_MAX_CANDIDATES_PER_KIND]
    y_size, x_size = labels.shape[1:]
    z = torch.div(indices, y_size * x_size, rounding_mode="floor")
    remainder = indices % (y_size * x_size)
    y = torch.div(remainder, x_size, rounding_mode="floor")
    x = remainder % x_size
    return torch.stack([z, y, x], dim=-1)


def _cell_candidates(
    labels: Tensor,
    current_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
) -> dict[str, Tensor]:
    stats = reduce_labeled_voxels(labels.cpu(), spacing_um.cpu())
    valid = stats.counts > 0
    foreground = torch.div(
        stats.min_voxel[valid] + stats.max_voxel[valid],
        2,
        rounding_mode="floor",
    ).long()

    max_id = int(labels.max().item())
    if max_id:
        flattened = labels.reshape(-1).long()
        positive = flattened > 0
        ids = flattened[positive]
        gt_counts = torch.bincount(ids, minlength=max_id + 1)[1:].float()
        current_foreground = current_labels.reshape(-1)[positive] > 0
        overlap = torch.bincount(
            ids[current_foreground], minlength=max_id + 1
        )[1:].float()
        represented = overlap / gt_counts.clamp_min(1)
        poor = valid & (represented < 0.50)
        absent_prior = torch.div(
            stats.min_voxel[poor] + stats.max_voxel[poor],
            2,
            rounding_mode="floor",
        ).long()
    else:
        absent_prior = labels.new_zeros((0, 3)).cpu()

    centroids = stats.centroid_voxel[valid]
    crowded_rows: list[Tensor] = []
    if centroids.shape[0] > 1:
        physical = centroids * spacing_um.cpu().float()[None]
        distances = torch.cdist(physical, physical)
        distances.fill_diagonal_(torch.inf)
        nearest_distance, nearest_index = distances.min(dim=1)
        crowded = nearest_distance <= 3.0 * float(dref_um)
        if crowded.any():
            midpoint = 0.5 * (
                centroids[crowded]
                + centroids[nearest_index[crowded]]
            )
            crowded_rows.append(midpoint.round().long())
    crowded_points = (
        torch.cat(crowded_rows, dim=0)
        if crowded_rows
        else labels.new_zeros((0, 3)).cpu()
    )
    return {
        "foreground": _bounded_candidates(foreground),
        "absent_prior": _bounded_candidates(absent_prior),
        "crowded": _bounded_candidates(crowded_points),
    }


def crop_source_signature(gt_labels: Tensor, current_labels: Tensor) -> tuple:
    return (
        tuple(gt_labels.shape),
        int(gt_labels.data_ptr()),
        tuple(current_labels.shape),
        int(current_labels.data_ptr()),
    )


def build_crop_candidate_cache(
    gt_labels: Tensor,
    current_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
) -> CropCandidateCache:
    """Precompute compact training-only crop candidates once per source batch."""
    if gt_labels.ndim != 4 or current_labels.shape != gt_labels.shape:
        raise ValueError("GT/current labels must align as [B,Z,Y,X]")
    candidates: list[dict[str, Tensor]] = []
    for batch_index in range(gt_labels.shape[0]):
        gt = gt_labels[batch_index].detach().cpu().long()
        current = current_labels[batch_index].detach().cpu().long()
        rows = _cell_candidates(
            gt,
            current,
            spacing_um[batch_index],
            dref_um[batch_index],
        )
        rows["separator"] = _source_conditioned_interface_candidates(
            gt,
            current,
            spacing_um[batch_index],
        )
        rows["disagreement"] = _disagreement_candidates(gt, current)
        rows["background"] = _background_candidates(gt)
        rows["random"] = gt.new_zeros((0, 3)).cpu()
        candidates.append(rows)
    return CropCandidateCache(
        candidates=candidates,
        source_signature=crop_source_signature(gt_labels, current_labels),
    )


def _crop_slices(
    center: Tensor,
    full_shape: tuple[int, int, int],
    requested_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    full = torch.tensor(full_shape, dtype=torch.long)
    size = torch.minimum(full, torch.tensor(requested_shape, dtype=torch.long))
    lower = torch.minimum(
        torch.maximum(
            center.cpu().long() - torch.div(size, 2, rounding_mode="floor"),
            torch.zeros(3, dtype=torch.long),
        ),
        full - size,
    )
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, lower + size)
    )  # type: ignore[return-value]


def _foreground_fraction(
    labels: Tensor, slices_zyx: tuple[slice, slice, slice]
) -> float:
    crop = labels[slices_zyx]
    return float((crop > 0).float().mean()) if crop.numel() else 0.0


def sample_mixed_crop_specs(
    gt_labels: Tensor,
    spacing_um: Tensor,
    cache: CropCandidateCache,
    *,
    crop_shape_zyx: tuple[int, int, int],
    crops_per_step: int,
    global_step: int,
    seed: int,
    min_foreground_fraction: float,
) -> list[list[CropSpec]]:
    """Return deterministic mixed crop rounds, one crop per batch row/round."""
    if cache.source_signature[0] != tuple(gt_labels.shape):
        raise ValueError("crop candidate cache does not match GT shape")
    full_shape = tuple(int(value) for value in gt_labels.shape[-3:])
    rounds: list[list[CropSpec]] = []
    for crop_round in range(crops_per_step):
        round_specs: list[CropSpec] = []
        for batch_index in range(gt_labels.shape[0]):
            ordinal = (
                global_step * crops_per_step * gt_labels.shape[0]
                + crop_round * gt_labels.shape[0]
                + batch_index
            )
            preferred = _CANDIDATE_ORDER[ordinal % len(_CANDIDATE_ORDER)]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1_000_003 * global_step + 97 * ordinal)
            available = cache.candidates[batch_index].get(preferred)
            candidate_type = preferred
            if available is None or not available.numel():
                for fallback in _CANDIDATE_ORDER:
                    rows = cache.candidates[batch_index].get(fallback)
                    if rows is not None and rows.numel():
                        available = rows
                        candidate_type = fallback
                        break
            selected_center: Tensor | None = None
            if available is not None and available.numel():
                order = torch.randperm(available.shape[0], generator=generator)
                for index in order[: min(16, len(order))]:
                    center = available[index]
                    slices = _crop_slices(center, full_shape, crop_shape_zyx)
                    if (
                        min_foreground_fraction <= 0
                        or _foreground_fraction(
                            gt_labels[batch_index].cpu(), slices
                        )
                        >= min_foreground_fraction
                    ):
                        selected_center = center
                        break
                if selected_center is None:
                    selected_center = available[order[0]]
            if selected_center is None:
                selected_center = torch.tensor(
                    [
                        int(
                            torch.randint(
                                0,
                                max(size, 1),
                                (1,),
                                generator=generator,
                            ).item()
                        )
                        for size in full_shape
                    ],
                    dtype=torch.long,
                )
                candidate_type = "random"
            slices = _crop_slices(
                selected_center, full_shape, crop_shape_zyx
            )
            lower = torch.tensor(
                [int(axis.start) for axis in slices], dtype=torch.float32
            )
            size = torch.tensor(
                [int(axis.stop) - int(axis.start) for axis in slices],
                dtype=torch.float32,
            )
            full_center = 0.5 * (
                torch.tensor(full_shape, dtype=torch.float32) - 1
            )
            crop_center = lower + 0.5 * (size - 1)
            center_shift_um = (
                crop_center - full_center
            ) * spacing_um[batch_index].detach().cpu().float()
            round_specs.append(
                CropSpec(
                    batch_index=batch_index,
                    slices_zyx=slices,
                    full_shape_zyx=full_shape,
                    center_shift_um=center_shift_um,
                    candidate_type=candidate_type,
                )
            )
        rounds.append(round_specs)
    return rounds


def crop_geometry_targets(
    targets: GeometryTargets,
    specs: List[CropSpec],
) -> GeometryTargets:
    rows = {}
    for name, value in targets.__dict__.items():
        rows[name] = torch.cat(
            [
                value[
                    spec.batch_index : spec.batch_index + 1,
                    :,
                    spec.slices_zyx[0],
                    spec.slices_zyx[1],
                    spec.slices_zyx[2],
                ]
                for spec in specs
            ],
            dim=0,
        )
    return GeometryTargets(**rows)


def _crop_supervision_valid_mask(gt_crop: Tensor, spec: CropSpec, spacing_um: Tensor, *, margin_um: float, device: torch.device) -> Tensor:
    if not spec.partial_cell_ids: return torch.ones(gt_crop.shape, device=device, dtype=torch.bool)
    ids=torch.tensor(spec.partial_cell_ids,device=gt_crop.device,dtype=gt_crop.dtype); ignored=torch.isin(gt_crop,ids).to(device)
    if margin_um > 0 and bool(ignored.any()):
        spacing=spacing_um.detach().cpu().float().clamp_min(1e-6); radii=torch.ceil(float(margin_um)/spacing).long(); kernel=tuple(int(2*r.item()+1) for r in radii); pad=tuple(int(r.item()) for r in radii)
        ignored=F.max_pool3d(ignored[None,None].float(),kernel_size=kernel,stride=1,padding=pad)[0,0]>0
    return ~ignored


def prepare_crop_batch(
    batch: dict,
    gt_labels: Tensor,
    specs: List[CropSpec],
    *,
    geometry_targets: GeometryTargets | None = None,
    partial_ignore_margin_um: float = 0.0,
) -> CropBatch:
    """Crop aligned state for cached-input or raw-source training."""
    cropped = {
        "spacing_um": torch.stack(
            [batch["spacing_um"][spec.batch_index] for spec in specs]
        ),
        "dref_um": torch.stack(
            [batch["dref_um"][spec.batch_index] for spec in specs]
        ),
    }
    if batch.get("spatial_inputs") is not None:
        cropped["spatial_inputs"] = torch.cat(
            [
                batch["spatial_inputs"][
                    spec.batch_index : spec.batch_index + 1,
                    :,
                    spec.slices_zyx[0],
                    spec.slices_zyx[1],
                    spec.slices_zyx[2],
                ]
                for spec in specs
            ],
            dim=0,
        )
    if batch.get("instance_labels") is not None:
        cropped["instance_labels"] = torch.stack(
            [batch["instance_labels"][spec.batch_index][spec.slices_zyx] for spec in specs]
        )
    if batch.get("spatial_padding_mask") is not None:
        cropped["spatial_padding_mask"] = torch.stack(
            [batch["spatial_padding_mask"][spec.batch_index][spec.slices_zyx] for spec in specs]
        )
    if batch.get("raw_normalization_bounds") is not None:
        cropped["raw_normalization_bounds"] = torch.stack(
            [torch.as_tensor(batch["raw_normalization_bounds"][spec.batch_index]) for spec in specs]
        )
    if batch.get("source_ids") is not None:
        cropped["source_ids"] = tuple(batch["source_ids"][spec.batch_index] for spec in specs)
    cropped_gt = torch.stack(
        [gt_labels[spec.batch_index][spec.slices_zyx] for spec in specs]
    )
    mask_device = (
        cropped["spatial_inputs"].device if "spatial_inputs" in cropped else cropped_gt.device
    )
    cropped["supervision_valid_mask"] = torch.stack([
        _crop_supervision_valid_mask(
            cropped_gt[row],
            spec,
            cropped["spacing_um"][row],
            margin_um=float(partial_ignore_margin_um),
            device=mask_device,
        )
        for row, spec in enumerate(specs)
    ])
    cropped_targets = None if geometry_targets is None else crop_geometry_targets(geometry_targets, specs)
    return CropBatch(
        batch=cropped,
        gt_labels=cropped_gt,
        geometry_targets=cropped_targets,
        specs=specs,
    )


def shift_crop_centered_points(points_um: Tensor, spec: CropSpec) -> Tensor:
    """Convert full-frame-centred physical points to crop-centred points."""
    return points_um - spec.center_shift_um.to(
        device=points_um.device, dtype=points_um.dtype
    )


__all__ = [
    "CropBatch",
    "CropCandidateCache",
    "CropSpec",
    "build_crop_candidate_cache",
    "crop_geometry_targets",
    "crop_source_signature",
    "prepare_crop_batch",
    "sample_mixed_crop_specs",
    "shift_crop_centered_points",
]
