from __future__ import annotations

from contextlib import nullcontext
from typing import List

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch import Tensor, nn

from ..config import PartitionConfig
from ..geometry.derived import build_geometry_derived_cache
from ..types import GeometryDerivedCache, GeometryLike, geometry_field
from .seeds import build_markers, build_markers_fast
from .supervoxel_guard import SupervoxelSafetyGuard


def _profile(profiler, name: str):
    return nullcontext() if profiler is None else profiler.profile(name)


def _merge_tiny_regions(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    """Merge tiny watershed regions without repeated full-volume relabel scans.

    Semantics are intentionally identical to the previous implementation:
    tiny regions merge into the positive neighboring label with the largest
    shared 6-connected interface, with the smallest target label breaking ties.
    Surviving positive labels are then compacted in ascending old-label order.
    """
    if min_voxels <= 1 or labels.max() <= 1:
        return labels

    labels = labels.astype(np.int32, copy=True)
    max_label = int(labels.max())

    counts = np.bincount(labels.ravel(), minlength=max_label + 1)
    tiny = np.flatnonzero((counts > 0) & (counts < min_voxels))
    tiny = tiny[tiny > 0]

    if tiny.size:
        # The old implementation materialized every directed positive-label
        # interface in the full volume, although only interfaces whose source
        # label is tiny can affect the remap. Restrict collection to those
        # sources while preserving the exact directed interface counts.
        is_tiny = np.zeros(max_label + 1, dtype=bool)
        is_tiny[tiny] = True
        packed_chunks: list[np.ndarray] = []
        base = max_label + 1

        for axis in range(3):
            left_slice = [slice(None)] * 3
            right_slice = [slice(None)] * 3
            left_slice[axis] = slice(0, -1)
            right_slice[axis] = slice(1, None)

            left = labels[tuple(left_slice)]
            right = labels[tuple(right_slice)]
            valid = (left > 0) & (right > 0) & (left != right)

            left_source = valid & is_tiny[left]
            if left_source.any():
                packed_chunks.append(
                    left[left_source].astype(np.int64) * base
                    + right[left_source].astype(np.int64)
                )

            right_source = valid & is_tiny[right]
            if right_source.any():
                packed_chunks.append(
                    right[right_source].astype(np.int64) * base
                    + left[right_source].astype(np.int64)
                )

        if packed_chunks:
            packed = np.concatenate(packed_chunks)
            keys, interface_counts = np.unique(packed, return_counts=True)
            source = keys // base
            target = keys % base

            remap = np.arange(base, dtype=np.int32)
            for region_id in tiny.tolist():
                candidates = np.flatnonzero(source == region_id)
                if candidates.size:
                    # Same tie-breaking as before:
                    # 1) largest interface count
                    # 2) smallest target label
                    best = candidates[
                        np.lexsort(
                            (
                                target[candidates],
                                -interface_counts[candidates],
                            )
                        )[0]
                    ]
                    remap[region_id] = int(target[best])

            labels = remap[labels]

    # The previous code rescanned the whole volume once for every surviving
    # label:
    #
    #   for new_id, old_id in enumerate(unique, 1):
    #       out[labels == old_id] = new_id
    #
    # A lookup table gives the exact same ascending-label compaction in one
    # indexed pass over the volume.
    present_counts = np.bincount(labels.ravel())
    present = np.flatnonzero(present_counts > 0)
    present = present[present > 0]

    if present.size == 0:
        return np.zeros_like(labels, dtype=np.int32)

    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lut[labels]


def _component_bounded_watershed(
    energy: np.ndarray,
    markers: np.ndarray,
    foreground: np.ndarray,
    *,
    halo: int,
) -> np.ndarray:
    components, count = ndi.label(foreground)
    if count <= 1:
        return watershed(energy, markers=markers, mask=foreground, connectivity=1).astype(np.int32)
    output = np.zeros(foreground.shape, dtype=np.int32)
    next_id = 1
    objects = ndi.find_objects(components)
    shape = foreground.shape
    for component_id, raw_box in enumerate(objects, 1):
        if raw_box is None:
            continue
        box = tuple(
            slice(max(0, int(axis.start) - halo), min(shape[i], int(axis.stop) + halo))
            for i, axis in enumerate(raw_box)
        )
        local_component = components[box] == component_id
        local_markers = np.where(local_component, markers[box], 0)
        local = watershed(
            energy[box], markers=local_markers, mask=local_component, connectivity=1
        ).astype(np.int32)
        positive = local > 0
        if positive.any():
            unique = np.unique(local[positive])
            mapping = np.zeros(int(local.max()) + 1, dtype=np.int32)
            mapping[unique] = np.arange(next_id, next_id + unique.size, dtype=np.int32)
            target = output[box]
            target[positive] = mapping[local[positive]]
            output[box] = target
            next_id += unique.size
    return output


class LearnedGeometryWatershed(nn.Module):
    """Paper-backed non-differentiable instance proposal stage.

    NucMM/NISNet3D motivate learned geometry + marker-controlled watershed.
    PlantSeg motivates deliberately oversegmenting first, then resolving the
    watershed basins with a region-adjacency graph.
    """

    def __init__(self, cfg: PartitionConfig):
        super().__init__()
        self.cfg = cfg
        self.safety_guard = SupervoxelSafetyGuard(cfg)

    @torch.no_grad()
    def forward(
        self,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None = None,
        *,
        derived_cache: GeometryDerivedCache | None = None,
        stage_profiler=None,
        profile_prefix: str = "watershed",
    ) -> List[Tensor]:
        results: List[Tensor] = []
        with _profile(stage_profiler, f"{profile_prefix}_geometry_probability_prepare"):
            cache = derived_cache or build_geometry_derived_cache(
                geometry,
                self.cfg,
                padding_mask=padding_mask,
                stage_profiler=stage_profiler,
                profile_prefix=profile_prefix,
            )
        batch_size = cache.foreground_prob.shape[0]
        for b in range(batch_size):
            fg_tensor = cache.foreground_mask[b]
            fg = fg_tensor.cpu().numpy()
            if not fg.any():
                results.append(
                    torch.zeros_like(cache.sdf[b, 0], dtype=torch.long)
                )
                continue
            radius_um = (
                self.cfg.seed_min_distance_dref * float(dref_um[b].item())
            )
            with _profile(stage_profiler, f"{profile_prefix}_marker_nms"):
                if self.cfg.watershed_backend == "fast":
                    markers = build_markers_fast(
                        cache.seed_score[b, 0], fg_tensor, spacing_um[b], radius_um,
                        self.cfg.seed_threshold, self.cfg.max_supervoxels,
                        stage_profiler=stage_profiler,
                        profile_prefix=profile_prefix,
                    )
                else:
                    markers = build_markers(
                        cache.seed_score[b, 0].float().cpu().numpy(), fg,
                        spacing_um[b].detach().cpu().numpy().astype(np.float32),
                        radius_um, self.cfg.seed_threshold, self.cfg.max_supervoxels,
                    )
            with _profile(stage_profiler, f"{profile_prefix}_gpu_to_cpu_transfer"):
                energy = cache.watershed_energy[b, 0].float().cpu().numpy()
            with _profile(stage_profiler, f"{profile_prefix}_actual_watershed"):
                if self.cfg.watershed_backend == "fast" and self.cfg.component_bounded_watershed:
                    labels = _component_bounded_watershed(
                        energy, markers, fg, halo=self.cfg.watershed_component_halo_voxels
                    )
                else:
                    labels = watershed(energy, markers=markers, mask=fg, connectivity=1)
            with _profile(stage_profiler, f"{profile_prefix}_tiny_region_cleanup"):
                labels = _merge_tiny_regions(
                    labels.astype(np.int32), self.cfg.min_supervoxel_voxels
                )
            # Final proposal safety step: preserve or split only. Nothing after
            # this point may merge a guarded fragment back across dense geometry
            # that indicates distinct cell identity.
            with _profile(stage_profiler, f"{profile_prefix}_supervoxel_safety_guard"):
                labels = self.safety_guard(
                    labels,
                    geometry,
                    cache,
                    b,
                    spacing_um[b],
                    dref_um[b],
                )
            if int(labels.max()) > self.cfg.max_supervoxels:
                raise RuntimeError(
                    f"Watershed created {int(labels.max())} supervoxels, exceeding "
                    f"max_supervoxels={self.cfg.max_supervoxels}."
                )
            with _profile(stage_profiler, f"{profile_prefix}_cpu_to_gpu_transfer"):
                results.append(torch.from_numpy(labels).to(device=cache.sdf.device, dtype=torch.long))
        return results
