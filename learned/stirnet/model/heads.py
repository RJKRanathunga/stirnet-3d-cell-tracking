from __future__ import annotations

import torch
from torch import Tensor, nn

from .native_masks import (
    compose_native_query_logits,
    native_chunk_coordinates_um,
    source_dilation_support_chunk,
)
from .query_builder import QUERY_SPATIAL_PROPOSAL


class ExistenceHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))

    def forward(self, q: Tensor) -> Tensor:
        return self.net(q).squeeze(-1)


class CenterHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q: Tensor) -> Tensor:
        return self.net(q)


class MaskEmbeddingHead(nn.Module):
    def __init__(self, d_model: int = 128, mask_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, mask_dim))

    def forward(self, q: Tensor) -> Tensor:
        return self.net(q)


class DenseAuxiliaryHeads(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.foreground = nn.Conv3d(in_channels, 1, 1)
        self.center = nn.Conv3d(in_channels, 1, 1)
        self.boundary = nn.Conv3d(in_channels, 1, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {
            "foreground_logits": self.foreground(x),
            "center_heatmap_logits": self.center(x),
            "boundary_logits": self.boundary(x),
        }


def dot_mask_logits(query_mask_embeddings: Tensor, mask_features: Tensor) -> Tensor:
    """query [B,Q,C], features [B,C,Z,Y,X] -> [B,Q,Z,Y,X]."""
    return torch.einsum("bqc,bczyx->bqzyx", query_mask_embeddings, mask_features)


def render_native_masks(
    mask_features: Tensor,
    native_mask_embeddings: Tensor,
    selected_indices: list[Tensor],
    query_types: Tensor,
    source_instance_ids: Tensor,
    refs_cellscale: Tensor,
    instance_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    prior_inside_logit: float = 1.5,
    prior_outside_logit: float = -1.5,
    temporal_sigma_dref: float = 0.75,
    native_support_radius_dref: float = 1.5,
    native_source_dilation_dref: float = 0.5,
    native_background_logit: float = -20.0,
    proposal_native_support_radius_dref: float | None = None,
) -> list[Tensor]:
    """Render only selected native-resolution masks; returns one tensor per batch item."""
    B, C, Z, Y, X = mask_features.shape
    voxel_count = Z * Y * X
    results = []
    for b in range(B):
        idx = selected_indices[b]
        if idx.numel() == 0:
            results.append(mask_features.new_zeros((0, Z, Y, X)))
            continue
        emb = native_mask_embeddings[b, idx]
        learned = torch.einsum("qc,cv->qv", emb, mask_features[b].flatten(1))
        selected_types = query_types[b, idx]
        selected_sources = source_instance_ids[b, idx]
        support_sources = torch.where(
            selected_types == QUERY_SPATIAL_PROPOSAL,
            torch.full_like(selected_sources, -1),
            selected_sources,
        )
        coords_um = native_chunk_coordinates_um(
            (Z, Y, X), spacing_um[b], 0, voxel_count
        )
        source_support = source_dilation_support_chunk(
            instance_labels[b],
            support_sources,
            spacing_um[b],
            dref_um[b],
            native_source_dilation_dref,
            0,
            voxel_count,
        )
        logits, _, _ = compose_native_query_logits(
            learned,
            selected_types,
            selected_sources,
            refs_cellscale[b, idx].float() * dref_um[b].float(),
            instance_labels[b].flatten(),
            source_support,
            coords_um,
            dref_um[b],
            support_radius_dref=native_support_radius_dref,
            temporal_sigma_dref=temporal_sigma_dref,
            prior_inside_logit=prior_inside_logit,
            prior_outside_logit=prior_outside_logit,
            background_logit=native_background_logit,
            proposal_support_radius_dref=proposal_native_support_radius_dref,
        )
        results.append(logits.reshape(len(idx), Z, Y, X))
    return results
