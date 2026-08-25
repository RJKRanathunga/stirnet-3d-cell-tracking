# STIRNET_LEARNED_SEPARATOR_BARRIER_V1
from __future__ import annotations

"""Learned separation-only barrier for spatial RAG edges.

final_merge_logit = base_merge_logit - barrier, with barrier >= 0.

The branch sees exact six-neighbour interface evidence:
[separator mean/max/std, coverages .50/.70/.85/.95, surface mean/max,
 separator*surface mean, |SDF|/dref, log contact area/dref^2].
"""

import torch
from torch import Tensor, nn

from ..types import GeometryLike, RAGState, geometry_field, geometry_probability


SEPARATOR_BARRIER_FEATURE_DIM = 12
SEPARATOR_BARRIER_FEATURE_NAMES = (
    "separator_mean",
    "separator_max",
    "separator_std",
    "separator_coverage_050",
    "separator_coverage_070",
    "separator_coverage_085",
    "separator_coverage_095",
    "surface_mean",
    "surface_max",
    "separator_surface_mean",
    "abs_sdf_mean_dref",
    "log_contact_area_dref2",
)


class SeparatorAwareBarrier(nn.Module):
    """Non-negative learned veto; it can never increase merge confidence."""

    def __init__(
        self,
        *,
        morphology_dim: int,
        hidden_dim: int,
        max_barrier_logit: float,
        initial_gate_bias: float,
        score_scale: float,
    ):
        super().__init__()
        self.explicit_norm = nn.LayerNorm(SEPARATOR_BARRIER_FEATURE_DIM)
        self.explicit_encoder = nn.Sequential(
            nn.Linear(SEPARATOR_BARRIER_FEATURE_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.morphology_dim = int(morphology_dim)
        if self.morphology_dim > 0:
            self.morphology_norm = nn.LayerNorm(self.morphology_dim)
            self.morphology_encoder = nn.Sequential(
                nn.Linear(self.morphology_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
            )
            fusion_input = 64
        else:
            self.morphology_norm = None
            self.morphology_encoder = None
            fusion_input = 32

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
        )
        self.final = nn.Linear(32, 1)
        nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.final.bias)

        self.max_barrier_logit = float(max_barrier_logit)
        self.initial_gate_bias = float(initial_gate_bias)
        self.score_scale = float(score_scale)

    def forward(
        self,
        explicit_features: Tensor,
        edge_morphology: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if explicit_features.shape[-1] != SEPARATOR_BARRIER_FEATURE_DIM:
            raise ValueError("separator barrier expects [E,12] explicit features")
        explicit = self.explicit_encoder(
            self.explicit_norm(explicit_features.float())
        )
        if self.morphology_dim > 0:
            if edge_morphology is None:
                raise ValueError("separator barrier expects edge morphology")
            assert self.morphology_norm is not None
            assert self.morphology_encoder is not None
            morphology = self.morphology_encoder(
                self.morphology_norm(edge_morphology.float())
            )
            fused = torch.cat([explicit, morphology], dim=-1)
        else:
            fused = explicit

        score = self.final(self.fusion(fused)).squeeze(-1)
        gate = self.initial_gate_bias + self.score_scale * score
        correction = self.max_barrier_logit * torch.sigmoid(gate)
        return (
            score.to(explicit_features.dtype),
            correction.to(explicit_features.dtype),
        )


def build_separator_barrier_features(
    rag: RAGState,
    geometry: GeometryLike,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    detach_geometry: bool = True,
) -> Tensor:
    """Exact physical interface features aligned one-to-one with RAG edges."""
    edge_count = int(rag.edge_index.shape[1])
    reference = rag.edge_features
    if edge_count == 0:
        return reference.new_zeros((0, SEPARATOR_BARRIER_FEATURE_DIM))

    result = reference.new_zeros(
        (edge_count, SEPARATOR_BARRIER_FEATURE_DIM),
        dtype=torch.float32,
    )
    thresholds = (0.50, 0.70, 0.85, 0.95)

    for batch_index, labels in enumerate(rag.supervoxel_labels):
        edge_rows = torch.nonzero(
            rag.edge_batch == batch_index,
            as_tuple=False,
        ).flatten()
        if edge_rows.numel() == 0:
            continue

        start = int(rag.node_offsets[batch_index].item())
        stop = int(rag.node_offsets[batch_index + 1].item())
        node_count = stop - start
        local_edges = rag.edge_index[:, edge_rows].long() - start
        edge_lo = torch.minimum(local_edges[0], local_edges[1])
        edge_hi = torch.maximum(local_edges[0], local_edges[1])
        target_packed = edge_lo * node_count + edge_hi

        separator = geometry_probability(
            geometry, "separator"
        )[batch_index, 0].float()
        surface = geometry_probability(
            geometry, "surface"
        )[batch_index, 0].float()
        sdf_abs = geometry_field(
            geometry, "sdf"
        )[batch_index, 0].float().abs()
        if detach_geometry:
            separator = separator.detach()
            surface = surface.detach()
            sdf_abs = sdf_abs.detach()

        spacing = spacing_um[batch_index].float()
        face_areas = (
            spacing[1] * spacing[2],
            spacing[0] * spacing[2],
            spacing[0] * spacing[1],
        )

        packed_chunks = []
        sep_chunks = []
        surf_chunks = []
        sdf_chunks = []
        area_chunks = []

        for axis in range(3):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis] = slice(0, -1)
            right[axis] = slice(1, None)
            left = tuple(left)
            right = tuple(right)

            la = labels[left]
            lb = labels[right]
            valid = (la > 0) & (lb > 0) & (la != lb)
            if not bool(valid.any()):
                continue

            ida = la[valid].long() - 1
            idb = lb[valid].long() - 1
            lo = torch.minimum(ida, idb)
            hi = torch.maximum(ida, idb)
            packed_chunks.append(lo * node_count + hi)

            def pair(field: Tensor) -> Tensor:
                return 0.5 * (
                    field[left][valid] + field[right][valid]
                )

            sep = pair(separator)
            sep_chunks.append(sep)
            surf_chunks.append(pair(surface))
            sdf_chunks.append(pair(sdf_abs))
            area_chunks.append(
                torch.ones_like(sep, dtype=torch.float32)
                * face_areas[axis]
            )

        if not packed_chunks:
            raise RuntimeError("RAG edges exist but no interface faces were found")

        packed = torch.cat(packed_chunks)
        sep = torch.cat(sep_chunks).float()
        surf = torch.cat(surf_chunks).float()
        sdf = torch.cat(sdf_chunks).float()
        area = torch.cat(area_chunks).float()

        order = torch.argsort(packed)
        packed = packed[order]
        sep = sep[order]
        surf = surf[order]
        sdf = sdf[order]
        area = area[order]

        unique, inverse = torch.unique_consecutive(
            packed,
            return_inverse=True,
        )
        groups = int(unique.numel())
        area_sum = area.new_zeros(groups)
        area_sum.index_add_(0, inverse, area)
        denom = area_sum.clamp_min(1e-8)

        def weighted_mean(value: Tensor) -> Tensor:
            total = value.new_zeros(groups)
            total.index_add_(0, inverse, area * value)
            return total / denom

        sep_mean = weighted_mean(sep)
        sep_std = (
            weighted_mean(sep.square()) - sep_mean.square()
        ).clamp_min(0).sqrt()
        surf_mean = weighted_mean(surf)
        sep_surf_mean = weighted_mean(sep * surf)
        sdf_mean = weighted_mean(sdf)

        sep_max = sep.new_full((groups,), -torch.inf)
        sep_max.scatter_reduce_(
            0, inverse, sep, reduce="amax", include_self=True
        )
        surf_max = surf.new_full((groups,), -torch.inf)
        surf_max.scatter_reduce_(
            0, inverse, surf, reduce="amax", include_self=True
        )
        coverage = [
            weighted_mean((sep >= threshold).to(sep.dtype))
            for threshold in thresholds
        ]

        dref = dref_um[batch_index].float().clamp_min(1e-6)
        features = torch.stack(
            [
                sep_mean,
                sep_max,
                sep_std,
                coverage[0],
                coverage[1],
                coverage[2],
                coverage[3],
                surf_mean,
                surf_max,
                sep_surf_mean,
                sdf_mean / dref,
                torch.log1p(area_sum / dref.square()),
            ],
            dim=-1,
        )

        positions = torch.searchsorted(unique, target_packed)
        in_bounds = positions < unique.numel()
        matched = torch.zeros_like(in_bounds)
        if bool(in_bounds.any()):
            matched[in_bounds] = (
                unique[positions[in_bounds]]
                == target_packed[in_bounds]
            )
        if not bool(matched.all()):
            raise RuntimeError(
                "A RAG edge is missing from separator interface statistics"
            )
        result[edge_rows] = features[positions]

    return result.to(reference.dtype)
