from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import PartitionConfig, SpatialConfig
from ..types import GeometryState, RAGState
from ..utils.physical import relative_grid_coordinates_um
from ..utils.tensor_ops import pool_labeled_features


def _adjacent_pairs_and_stats(
    labels: Tensor,
    separator: Tensor,
    surface: Tensor,
    foreground: Tensor,
    sdf: Tensor,
    flow: Tensor,
    centroid_offset: Tensor,
) -> tuple[Tensor, Tensor]:
    """Extract undirected RAG edges and 8 interface features on-device."""
    pair_chunks = []
    stat_chunks = []
    for axis, dim in enumerate((0, 1, 2)):
        a_slice = [slice(None)] * 3
        b_slice = [slice(None)] * 3
        a_slice[dim] = slice(0, -1)
        b_slice[dim] = slice(1, None)
        a = labels[tuple(a_slice)]
        b = labels[tuple(b_slice)]
        valid = (a > 0) & (b > 0) & (a != b)
        if not valid.any():
            continue
        la = a[valid].long() - 1
        lb = b[valid].long() - 1
        lo = torch.minimum(la, lb)
        hi = torch.maximum(la, lb)
        pair_chunks.append(torch.stack([lo, hi], dim=-1))

        def pair_scalar(x: Tensor) -> Tensor:
            xa = x[tuple(a_slice)][valid]
            xb = x[tuple(b_slice)][valid]
            return 0.5 * (xa + xb)

        sep = pair_scalar(separator)
        surf = pair_scalar(surface)
        fg = pair_scalar(foreground)
        sdf_abs = pair_scalar(sdf.abs())

        fa = flow[:, tuple(a_slice)[0], tuple(a_slice)[1], tuple(a_slice)[2]][:, valid].T
        fb = flow[:, tuple(b_slice)[0], tuple(b_slice)[1], tuple(b_slice)[2]][:, valid].T
        flow_disagree = 1.0 - F.cosine_similarity(fa, fb, dim=-1, eps=1e-6)
        oa = centroid_offset[:, tuple(a_slice)[0], tuple(a_slice)[1], tuple(a_slice)[2]][:, valid].T
        ob = centroid_offset[:, tuple(b_slice)[0], tuple(b_slice)[1], tuple(b_slice)[2]][:, valid].T
        offset_disagree = torch.linalg.vector_norm(oa - ob, dim=-1)
        # Duplicate sep/surface values are aggregated as mean and max later.
        stat_chunks.append(
            torch.stack(
                [sep, surf, fg, sdf_abs, flow_disagree, offset_disagree], dim=-1
            )
        )
    if not pair_chunks:
        return labels.new_zeros((2, 0)), separator.new_zeros((0, 8))
    pairs = torch.cat(pair_chunks, dim=0)
    stats = torch.cat(stat_chunks, dim=0)
    n_nodes = int(labels.max().item())
    packed = pairs[:, 0] * max(n_nodes, 1) + pairs[:, 1]
    unique, inverse = torch.unique(packed, sorted=True, return_inverse=True)
    edge_count = len(unique)
    means = stats.new_zeros((edge_count, stats.shape[1]))
    counts = stats.new_zeros((edge_count,))
    means.index_add_(0, inverse, stats)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=stats.dtype))
    means = means / counts.clamp_min(1)[:, None]
    maxima = stats.new_full((edge_count, 2), -torch.inf)
    maxima.scatter_reduce_(
        0,
        inverse[:, None].expand(-1, 2),
        stats[:, :2],
        reduce="amax",
        include_self=True,
    )
    lo = torch.div(unique, max(n_nodes, 1), rounding_mode="floor")
    hi = unique % max(n_nodes, 1)
    edges = torch.stack([lo, hi], dim=0)
    # [sep_mean, sep_max, surface_mean, surface_max, fg_mean,
    #  abs_sdf_mean, flow_disagreement, centroid-offset disagreement]
    edge_features = torch.stack(
        [
            means[:, 0],
            maxima[:, 0],
            means[:, 1],
            maxima[:, 1],
            means[:, 2],
            means[:, 3],
            means[:, 4],
            means[:, 5],
        ],
        dim=-1,
    )
    return edges, edge_features


class RAGBuilder(nn.Module):
    """Construct a supervoxel RAG without compressing topology into a center token."""

    def __init__(self, cfg: PartitionConfig, spatial_cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        self.node_projection = nn.Conv3d(
            spatial_cfg.channels[0], cfg.node_feature_channels, 1, bias=False
        )
        # 2*C D0 summary + 2*12 dense/raw summary + centroid(3)+log-volume(1)
        self.node_feature_dim = 2 * cfg.node_feature_channels + 24 + 4
        self.edge_feature_dim = 8

    def forward(
        self,
        supervoxel_labels: List[Tensor],
        d0: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryState,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> RAGState:
        d0_small = self.node_projection(d0)
        all_nodes = []
        all_centroids = []
        all_volumes = []
        all_node_batch = []
        all_node_ids = []
        all_edges = []
        all_edge_features = []
        all_edge_batch = []
        node_offsets = [0]

        probs = geometry.probabilities()
        for b, labels in enumerate(supervoxel_labels):
            n = int(labels.max().item())
            if n == 0:
                node_offsets.append(node_offsets[-1])
                continue
            pooled_d0, counts = pool_labeled_features(d0_small[b], labels)
            raw0 = spatial_inputs[b, :1]
            dense = torch.cat(
                [
                    raw0,
                    probs["foreground"][b],
                    probs["surface"][b],
                    probs["separator"][b],
                    geometry.sdf[b],
                    geometry.flow[b],
                    geometry.centroid_offset[b],
                    probs["seed"][b],
                ],
                dim=0,
            )
            pooled_dense, _ = pool_labeled_features(dense, labels)
            coords = relative_grid_coordinates_um(
                tuple(labels.shape), spacing_um[b], device=labels.device
            ).permute(3, 0, 1, 2)
            pooled_coords, _ = pool_labeled_features(coords, labels, include_max=False)
            volume_feature = torch.log1p(counts)[:, None] - torch.log1p(
                counts.median().clamp_min(1)
            )
            node_feat = torch.cat(
                [pooled_d0, pooled_dense, pooled_coords / dref_um[b].clamp_min(1e-6), volume_feature],
                dim=-1,
            )
            all_nodes.append(node_feat)
            all_centroids.append(pooled_coords)
            all_volumes.append(counts)
            all_node_batch.append(
                torch.full((n,), b, device=labels.device, dtype=torch.long)
            )
            all_node_ids.append(torch.arange(1, n + 1, device=labels.device))

            edge_local, edge_feat = _adjacent_pairs_and_stats(
                labels,
                probs["separator"][b, 0],
                probs["surface"][b, 0],
                probs["foreground"][b, 0],
                geometry.sdf[b, 0],
                geometry.flow[b],
                geometry.centroid_offset[b],
            )
            offset = node_offsets[-1]
            if edge_local.shape[1]:
                all_edges.append(edge_local + offset)
                all_edge_features.append(edge_feat)
                all_edge_batch.append(
                    torch.full(
                        (edge_local.shape[1],), b, device=labels.device, dtype=torch.long
                    )
                )
            node_offsets.append(offset + n)

        device = d0.device
        node_features = (
            torch.cat(all_nodes, dim=0)
            if all_nodes
            else d0.new_zeros((0, self.node_feature_dim))
        )
        node_centroid = (
            torch.cat(all_centroids, dim=0) if all_centroids else d0.new_zeros((0, 3))
        )
        node_volume = (
            torch.cat(all_volumes, dim=0) if all_volumes else d0.new_zeros((0,))
        )
        node_batch = (
            torch.cat(all_node_batch) if all_node_batch else torch.zeros(0, device=device, dtype=torch.long)
        )
        node_ids = (
            torch.cat(all_node_ids) if all_node_ids else torch.zeros(0, device=device, dtype=torch.long)
        )
        edge_index = (
            torch.cat(all_edges, dim=1)
            if all_edges
            else torch.zeros((2, 0), device=device, dtype=torch.long)
        )
        edge_features = (
            torch.cat(all_edge_features, dim=0)
            if all_edge_features
            else d0.new_zeros((0, self.edge_feature_dim))
        )
        edge_batch = (
            torch.cat(all_edge_batch)
            if all_edge_batch
            else torch.zeros(0, device=device, dtype=torch.long)
        )
        return RAGState(
            node_features=node_features,
            node_embeddings=d0.new_zeros((node_features.shape[0], self.cfg.rag_hidden_dim)),
            node_batch=node_batch,
            node_supervoxel_id=node_ids,
            node_centroid_um=node_centroid,
            node_volume_voxels=node_volume,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_embeddings=d0.new_zeros((edge_features.shape[0], self.cfg.rag_hidden_dim)),
            spatial_edge_logits=d0.new_zeros((edge_features.shape[0],)),
            edge_batch=edge_batch,
            supervoxel_labels=supervoxel_labels,
            node_offsets=torch.tensor(node_offsets, device=device, dtype=torch.long),
        )


class RAGCriterion(nn.Module):
    """Supervise whether adjacent supervoxels belong to the same GT instance."""

    def forward(self, rag: RAGState, gt_labels: Tensor) -> Dict[str, Tensor]:
        if rag.edge_index.shape[1] == 0:
            zero = rag.node_features.sum() * 0
            return {"rag_bce": zero, "rag_accuracy": zero.detach()}
        dominant = torch.zeros(
            rag.node_features.shape[0], device=rag.node_features.device, dtype=torch.long
        )
        for b, supervox in enumerate(rag.supervoxel_labels):
            gt = gt_labels[b].to(supervox.device).long()
            start = int(rag.node_offsets[b].item())
            n = int(supervox.max().item())
            for local_id in range(1, n + 1):
                values = gt[supervox == local_id]
                values = values[values > 0]
                if values.numel():
                    uniq, counts = torch.unique(values, return_counts=True)
                    dominant[start + local_id - 1] = uniq[counts.argmax()]
        a = dominant[rag.edge_index[0]]
        b = dominant[rag.edge_index[1]]
        target = ((a > 0) & (a == b)).float()
        positives = target.sum()
        negatives = target.numel() - positives
        pos_weight = (negatives / positives.clamp_min(1)).clamp(0.5, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            rag.spatial_edge_logits, target, pos_weight=pos_weight
        )
        accuracy = ((rag.spatial_edge_logits.sigmoid() >= 0.5) == target.bool()).float().mean()
        return {"rag_bce": loss, "rag_accuracy": accuracy.detach()}
