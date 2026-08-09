from __future__ import annotations

import torch
from torch import Tensor, nn


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
) -> list[Tensor]:
    """Render only selected native-resolution masks; returns one tensor per batch item."""
    from .query_builder import QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL
    from .coordinates import feature_grid_coordinates_um

    B, C, Z, Y, X = mask_features.shape
    pos = feature_grid_coordinates_um((Z, Y, X), spacing_um, relative_to_center=True).reshape(B, Z, Y, X, 3)
    results = []
    for b in range(B):
        idx = selected_indices[b]
        if idx.numel() == 0:
            results.append(mask_features.new_zeros((0, Z, Y, X)))
            continue
        emb = native_mask_embeddings[b, idx]
        logits = torch.einsum("qc,czyx->qzyx", emb, mask_features[b])
        for local, qi in enumerate(idx.tolist()):
            qtype = int(query_types[b, qi].item())
            sid = int(source_instance_ids[b, qi].item())
            if qtype in (QUERY_PRIMARY, QUERY_SPLIT) and sid >= 0:
                inside = instance_labels[b] == sid
                prior = torch.full_like(logits[local], prior_outside_logit)
                prior[inside] = prior_inside_logit
                logits[local] = logits[local] + prior
            elif qtype == QUERY_TEMPORAL:
                ref_um = refs_cellscale[b, qi] * dref_um[b]
                delta = pos[b] - ref_um
                dist2 = (delta * delta).sum(dim=-1)
                sigma = temporal_sigma_dref * dref_um[b]
                logits[local] = logits[local] + prior_inside_logit * torch.exp(-0.5 * dist2 / sigma.clamp_min(1e-6).pow(2))
        results.append(logits)
    return results
