from __future__ import annotations

from typing import List

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch import Tensor, nn

from ..config import PartitionConfig
from ..types import GeometryState
from .seeds import build_markers


def _merge_tiny_regions(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or labels.max() <= 1:
        return labels
    labels = labels.copy()
    counts = np.bincount(labels.ravel())
    tiny = [i for i in range(1, len(counts)) if 0 < counts[i] < min_voxels]
    for region_id in tiny:
        mask = labels == region_id
        if not mask.any():
            continue
        dilated = ndi.binary_dilation(mask, iterations=1)
        neighbors = labels[dilated & ~mask]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size:
            values, n = np.unique(neighbors, return_counts=True)
            labels[mask] = values[np.argmax(n)]
    unique = np.unique(labels)
    unique = unique[unique > 0]
    out = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(unique, 1):
        out[labels == old_id] = new_id
    return out


class LearnedGeometryWatershed(nn.Module):
    """Paper-backed non-differentiable instance proposal stage.

    NucMM/NISNet3D motivate learned geometry + marker-controlled watershed.
    PlantSeg motivates deliberately oversegmenting first, then resolving the
    watershed basins with a region-adjacency graph.
    """

    def __init__(self, cfg: PartitionConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        geometry: GeometryState,
        spacing_um: Tensor,
        dref_um: Tensor,
        padding_mask: Tensor | None = None,
    ) -> List[Tensor]:
        probs = geometry.probabilities()
        results: List[Tensor] = []
        for b in range(geometry.foreground_logits.shape[0]):
            fg_prob = probs["foreground"][b, 0].float().cpu().numpy()
            surface = probs["surface"][b, 0].float().cpu().numpy()
            separator = probs["separator"][b, 0].float().cpu().numpy()
            seed_head = probs["seed"][b, 0].float().cpu().numpy()
            sdf = geometry.sdf[b, 0].float().cpu().numpy()
            fg = fg_prob >= self.cfg.foreground_threshold
            if padding_mask is not None:
                fg &= ~padding_mask[b].detach().cpu().numpy().astype(bool)
            if not fg.any():
                results.append(
                    torch.zeros_like(geometry.sdf[b, 0], dtype=torch.long)
                )
                continue

            sdf_pos = np.clip(sdf, 0.0, None)
            sdf_norm = sdf_pos / max(float(sdf_pos[fg].max()), 1e-6)
            # Separator evidence suppresses false seeds near an inter-cell
            # interface; SDF and the learned marker head carry complementary
            # medial-geometry information.
            seed_score = (
                self.cfg.seed_sdf_weight * sdf_norm
                + self.cfg.seed_head_weight * seed_head
            ) * (1.0 - separator)
            radius_um = (
                self.cfg.seed_min_distance_dref * float(dref_um[b].item())
            )
            markers = build_markers(
                seed_score,
                fg,
                spacing_um[b].detach().cpu().numpy().astype(np.float32),
                radius_um,
                self.cfg.seed_threshold,
                self.cfg.max_supervoxels,
            )

            # Low energy = object interior. Separator dominates because it is
            # specifically trained on inter-instance interfaces; surface is a
            # weaker term and positive SDF stabilizes basin interiors.
            energy = (
                self.cfg.watershed_separator_weight * separator
                + self.cfg.watershed_surface_weight * surface
                + self.cfg.watershed_sdf_weight * (1.0 - sdf_norm)
            ).astype(np.float32)
            labels = watershed(energy, markers=markers, mask=fg, connectivity=1)
            labels = _merge_tiny_regions(labels.astype(np.int32), self.cfg.min_supervoxel_voxels)
            if int(labels.max()) > self.cfg.max_supervoxels:
                raise RuntimeError(
                    f"Watershed created {int(labels.max())} supervoxels, exceeding "
                    f"max_supervoxels={self.cfg.max_supervoxels}."
                )
            results.append(
                torch.from_numpy(labels).to(
                    device=geometry.sdf.device, dtype=torch.long
                )
            )
        return results
