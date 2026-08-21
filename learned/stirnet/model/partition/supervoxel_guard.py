from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch import Tensor, nn

from ..config import PartitionConfig
from ..types import GeometryDerivedCache, GeometryLike, geometry_field


@dataclass(frozen=True)
class SupervoxelGuardDiagnostics:
    preliminary_count: int
    final_count: int
    split_supervoxel_count: int
    added_supervoxel_count: int
    barrier_voxel_count: int


def _neighbor_vector_disagreement(
    vectors: np.ndarray,
    *,
    normalizer: float = 1.0,
    cosine: bool = False,
    minimum_vector_norm: float = 0.0,
) -> np.ndarray:
    """Maximum 6-neighbor vector disagreement, written to both voxels."""
    if vectors.ndim != 4 or vectors.shape[0] != 3:
        raise ValueError("vectors must have shape [3,Z,Y,X]")
    out = np.zeros(vectors.shape[1:], dtype=np.float32)
    scale = max(float(normalizer), 1e-6)

    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)

        va = vectors[(slice(None), *lower)]
        vb = vectors[(slice(None), *upper)]

        if cosine:
            na = np.sqrt(np.sum(va * va, axis=0))
            nb = np.sqrt(np.sum(vb * vb, axis=0))
            valid = (na >= minimum_vector_norm) & (nb >= minimum_vector_norm)
            dot = np.sum(va * vb, axis=0)
            denom = np.maximum(na * nb, 1e-6)
            disagreement = np.zeros_like(dot, dtype=np.float32)
            disagreement[valid] = np.clip(
                1.0 - dot[valid] / denom[valid], 0.0, 2.0
            )
        else:
            disagreement = (
                np.sqrt(np.sum((va - vb) ** 2, axis=0)) / scale
            ).astype(np.float32, copy=False)

        left = out[tuple(lower)]
        right = out[tuple(upper)]
        np.maximum(left, disagreement, out=left)
        np.maximum(right, disagreement, out=right)

    return out


def _absolute_centroid_votes(
    centroid_offset: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
) -> np.ndarray:
    """Convert relative centroid offsets into absolute physical centroid votes."""
    if centroid_offset.ndim != 4 or centroid_offset.shape[0] != 3:
        raise ValueError("centroid_offset must have shape [3,Z,Y,X]")
    shape = centroid_offset.shape[1:]
    votes = np.empty_like(centroid_offset, dtype=np.float32)
    for axis in range(3):
        coordinate = np.arange(shape[axis], dtype=np.float32) * float(spacing_um[axis])
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        votes[axis] = (
            coordinate.reshape(reshape)
            + centroid_offset[axis].astype(np.float32, copy=False) * float(dref_um)
        )
    return votes


def build_supervoxel_barrier(
    separator: np.ndarray,
    centroid_offset: np.ndarray,
    flow: np.ndarray,
    seed: np.ndarray,
    sdf_normalized: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    cfg: PartitionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a conservative hard barrier from independent geometry cues.

    Separator is primary. Medium separator evidence becomes hard when
    corroborated by centroid-vote disagreement and either flow disagreement or
    a seed/SDF valley. A geometry-only fallback requires a stronger centroid
    discontinuity together with both flow and valley support.
    """
    separator = np.asarray(separator, dtype=np.float32)
    seed = np.asarray(seed, dtype=np.float32)
    sdf_normalized = np.asarray(sdf_normalized, dtype=np.float32)
    spacing_um = np.asarray(spacing_um, dtype=np.float32)

    centroid_votes = _absolute_centroid_votes(
        np.asarray(centroid_offset, dtype=np.float32), spacing_um, dref_um
    )
    centroid_disagreement = _neighbor_vector_disagreement(
        centroid_votes, normalizer=dref_um
    )
    flow_disagreement = _neighbor_vector_disagreement(
        np.asarray(flow, dtype=np.float32),
        cosine=True,
        minimum_vector_norm=cfg.supervoxel_guard_flow_min_norm,
    )

    strong_separator = separator >= cfg.supervoxel_guard_separator_high
    weak_separator = separator >= cfg.supervoxel_guard_separator_low
    centroid_support = (
        centroid_disagreement >= cfg.supervoxel_guard_centroid_disagreement_dref
    )
    strong_centroid_support = (
        centroid_disagreement
        >= cfg.supervoxel_guard_centroid_strong_disagreement_dref
    )
    flow_support = flow_disagreement >= cfg.supervoxel_guard_flow_disagreement
    valley_support = (
        (seed <= cfg.supervoxel_guard_seed_valley_max)
        | (sdf_normalized <= cfg.supervoxel_guard_sdf_valley_max)
    )

    separator_corroborated = (
        weak_separator & centroid_support & (flow_support | valley_support)
    )
    geometry_only = strong_centroid_support & flow_support & valley_support
    barrier = strong_separator | separator_corroborated | geometry_only

    if cfg.supervoxel_guard_closing_iterations > 0 and barrier.any():
        structure = ndi.generate_binary_structure(3, 1)
        closed = ndi.binary_closing(
            barrier,
            structure=structure,
            iterations=cfg.supervoxel_guard_closing_iterations,
        )
        support = (
            (separator >= 0.5 * cfg.supervoxel_guard_separator_low)
            | centroid_support
            | flow_support
        )
        barrier = barrier | (closed & support)

    evidence = {
        "centroid_vote_disagreement": centroid_disagreement,
        "flow_disagreement": flow_disagreement,
        "strong_separator": strong_separator,
        "separator_corroborated": separator_corroborated,
        "geometry_only": geometry_only,
        "valley_support": valley_support,
    }
    return barrier.astype(bool, copy=False), evidence


def _meaningful_component_ids(
    components: np.ndarray,
    *,
    supervoxel_voxels: int,
    cfg: PartitionConfig,
) -> np.ndarray:
    counts = np.bincount(components.ravel())
    if counts.size <= 1:
        return np.zeros((0,), dtype=np.int32)
    minimum = max(
        int(cfg.supervoxel_guard_min_fragment_voxels),
        int(
            np.ceil(
                float(supervoxel_voxels)
                * float(cfg.supervoxel_guard_min_fragment_fraction)
            )
        ),
    )
    ids = np.flatnonzero(counts >= minimum)
    ids = ids[ids > 0]
    if ids.size <= cfg.supervoxel_guard_max_fragments:
        return ids.astype(np.int32, copy=False)
    order = np.lexsort((ids, -counts[ids]))
    return ids[order[: cfg.supervoxel_guard_max_fragments]].astype(
        np.int32, copy=False
    )


def split_preliminary_supervoxels(
    labels: np.ndarray,
    separator: np.ndarray,
    centroid_offset: np.ndarray,
    flow: np.ndarray,
    seed: np.ndarray,
    sdf_normalized: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    cfg: PartitionConfig,
) -> tuple[np.ndarray, SupervoxelGuardDiagnostics]:
    """Split preliminary watershed basins wherever geometry forbids crossing.

    This function never merges preliminary labels. Barrier voxels remain
    foreground: they are removed only to discover safe connected cores, then a
    local flat watershed assigns them back to one of those cores.
    """
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 3:
        raise ValueError("labels must be [Z,Y,X]")
    if not cfg.supervoxel_guard_enabled or labels.max() <= 0:
        count = int(labels.max())
        return labels.copy(), SupervoxelGuardDiagnostics(
            preliminary_count=count,
            final_count=count,
            split_supervoxel_count=0,
            added_supervoxel_count=0,
            barrier_voxel_count=0,
        )

    barrier, _ = build_supervoxel_barrier(
        separator,
        centroid_offset,
        flow,
        seed,
        sdf_normalized,
        spacing_um,
        dref_um,
        cfg,
    )

    structure = ndi.generate_binary_structure(3, 1)
    objects = ndi.find_objects(labels)
    output = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    split_count = 0

    for old_id, box in enumerate(objects, 1):
        if box is None:
            continue
        local_labels = labels[box]
        local_mask = local_labels == old_id
        if not local_mask.any():
            continue

        local_barrier = barrier[box] & local_mask
        if not local_barrier.any():
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue

        core = local_mask & ~local_barrier
        components, _ = ndi.label(core, structure=structure)
        meaningful = _meaningful_component_ids(
            components,
            supervoxel_voxels=int(local_mask.sum()),
            cfg=cfg,
        )
        if meaningful.size < 2:
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue

        markers = np.zeros_like(components, dtype=np.int32)
        for marker_id, component_id in enumerate(meaningful.tolist(), 1):
            markers[components == int(component_id)] = marker_id

        mask_cc, mask_cc_count = ndi.label(local_mask, structure=structure)
        marker_next = int(markers.max()) + 1
        for cc_id in range(1, mask_cc_count + 1):
            cc = mask_cc == cc_id
            if np.any(markers[cc] > 0):
                continue
            core_points = np.argwhere(cc & core)
            points = core_points if len(core_points) else np.argwhere(cc)
            if len(points) == 0:
                continue
            point = tuple(points[len(points) // 2])
            markers[point] = marker_next
            marker_next += 1

        split = watershed(
            np.zeros(local_mask.shape, dtype=np.float32),
            markers=markers,
            mask=local_mask,
            connectivity=1,
        ).astype(np.int32)

        fragment_ids = np.unique(split[local_mask])
        fragment_ids = fragment_ids[fragment_ids > 0]
        if fragment_ids.size < 2:
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue

        split_count += 1
        target = output[box]
        for fragment_id in fragment_ids.tolist():
            target[split == fragment_id] = next_id
            next_id += 1
        output[box] = target

    final_count = next_id - 1
    preliminary_count = int(labels.max())
    return output, SupervoxelGuardDiagnostics(
        preliminary_count=preliminary_count,
        final_count=final_count,
        split_supervoxel_count=split_count,
        added_supervoxel_count=max(final_count - preliminary_count, 0),
        barrier_voxel_count=int((barrier & (labels > 0)).sum()),
    )


class SupervoxelSafetyGuard(nn.Module):
    """Post-watershed guard that can only preserve or split preliminary basins."""

    def __init__(self, cfg: PartitionConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        labels: np.ndarray,
        geometry: GeometryLike,
        derived_cache: GeometryDerivedCache,
        batch_index: int,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> np.ndarray:
        if not self.cfg.supervoxel_guard_enabled:
            return np.asarray(labels, dtype=np.int32)

        separator = (
            derived_cache.separator_prob[batch_index, 0]
            .detach().float().cpu().numpy()
        )
        seed = (
            derived_cache.seed_prob[batch_index, 0]
            .detach().float().cpu().numpy()
        )
        sdf_normalized = (
            derived_cache.sdf_normalized[batch_index, 0]
            .detach().float().cpu().numpy()
        )
        centroid_offset = (
            geometry_field(geometry, "centroid_offset")[batch_index]
            .detach().float().cpu().numpy()
        )
        flow = (
            geometry_field(geometry, "flow")[batch_index]
            .detach().float().cpu().numpy()
        )
        spacing = spacing_um.detach().float().cpu().numpy().astype(np.float32)
        dref = float(dref_um.detach().float().cpu().item())

        guarded, _ = split_preliminary_supervoxels(
            np.asarray(labels, dtype=np.int32),
            separator,
            centroid_offset,
            flow,
            seed,
            sdf_normalized,
            spacing,
            dref,
            self.cfg,
        )
        return guarded


__all__ = [
    "SupervoxelGuardDiagnostics",
    "SupervoxelSafetyGuard",
    "build_supervoxel_barrier",
    "split_preliminary_supervoxels",
]
