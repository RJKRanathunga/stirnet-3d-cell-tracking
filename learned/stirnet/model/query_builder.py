from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import QueryConfig
from .coordinates import resize_label_map_nearest
from .types import QueryState, TemporalState


QUERY_PRIMARY = 0
QUERY_SPLIT = 1
QUERY_TEMPORAL = 2
QUERY_DISCOVERY = 3
NUM_QUERY_TYPES = 4


class InstanceQueryBuilder(nn.Module):
    def __init__(self, cfg: QueryConfig, feature_channels: int = 64):
        super().__init__()
        self.cfg = cfg
        self.feature_proj = nn.Linear(2 * feature_channels, cfg.d_model)
        self.geom_proj = nn.Sequential(
            nn.Linear(cfg.instance_feature_dim, 64), nn.SiLU(), nn.Linear(64, cfg.d_model)
        )
        self.type_embedding = nn.Embedding(NUM_QUERY_TYPES, cfg.d_model)
        if cfg.split_companions_per_instance < 0:
            raise ValueError("split_companions_per_instance must be non-negative")
        if cfg.max_split_companions_per_instance < cfg.split_companions_per_instance:
            raise ValueError(
                "max_split_companions_per_instance must be at least the baseline "
                "split_companions_per_instance"
            )
        if cfg.split_volume_ratio_per_hypothesis <= 0:
            raise ValueError("split_volume_ratio_per_hypothesis must be positive")
        self.split_slot_embedding = nn.Embedding(
            cfg.max_split_companions_per_instance, cfg.d_model
        )
        nn.init.normal_(self.split_slot_embedding.weight, std=0.02)
        self.discovery_queries = nn.Parameter(torch.randn(cfg.discovery_queries, cfg.d_model) * 0.02)
        self.discovery_refs = nn.Parameter(torch.empty(cfg.discovery_queries, 3))
        nn.init.uniform_(self.discovery_refs, -2.0, 2.0)

    @torch.no_grad()
    def split_companion_counts(
        self,
        instance_labels: Tensor,
        instance_ids: Tensor,
        instance_batch: Tensor,
    ) -> Tensor:
        """Estimate per-source split multiplicity from current-source volume only.

        Voxel-volume ratios equal physical-volume ratios within one batch item
        because spacing is constant over that volume. The robust within-batch
        median defines one typical cell, and no GT information is consumed.
        """
        counts = torch.zeros_like(instance_ids, dtype=torch.float32)
        for batch_index in range(instance_labels.shape[0]):
            rows = torch.nonzero(
                instance_batch == batch_index, as_tuple=False
            ).flatten()
            if rows.numel() == 0:
                continue
            ids = instance_ids[rows]
            max_id = int(ids.max().item())
            labels = instance_labels[batch_index]
            if max_id <= max(4096, 16 * int(rows.numel())):
                histogram = torch.histc(
                    labels.float(),
                    bins=max_id + 1,
                    min=-0.5,
                    max=max_id + 0.5,
                )
                volumes = histogram[ids.long()].float()
            else:
                unique_ids, unique_counts = torch.unique(
                    labels, return_counts=True
                )
                order = torch.argsort(unique_ids)
                unique_ids = unique_ids[order]
                unique_counts = unique_counts[order]
                positions = torch.searchsorted(unique_ids, ids.to(unique_ids.dtype))
                valid = positions < len(unique_ids)
                safe_positions = positions.clamp_max(max(len(unique_ids) - 1, 0))
                valid = valid & (unique_ids[safe_positions] == ids.to(unique_ids.dtype))
                volumes = torch.zeros(len(ids), device=labels.device, dtype=torch.float32)
                volumes[valid] = unique_counts[safe_positions[valid]].float()
            positive = volumes[volumes > 0]
            median = (
                positive.median()
                if positive.numel()
                else volumes.new_tensor(1.0)
            )
            ratio = volumes / median.clamp_min(1.0)
            estimated_total = torch.ceil(
                ratio / float(self.cfg.split_volume_ratio_per_hypothesis)
            ).long().clamp_min(1)
            companions = torch.maximum(
                estimated_total - 1,
                torch.full_like(
                    estimated_total, self.cfg.split_companions_per_instance
                ),
            ).clamp_max(self.cfg.max_split_companions_per_instance)
            counts[rows] = companions.float()
        return counts.long()

    def _pool_instances(
        self,
        feature: Tensor,
        feature_spacing_um: Tensor,
        instance_labels: Tensor,
        instance_ids: Tensor,
        instance_batch: Tensor,
        instance_centroids_um: Tensor,
    ) -> Tensor:
        B, C = feature.shape[:2]
        labels_ds = resize_label_map_nearest(instance_labels, feature.shape[-3:])
        pooled = feature.new_zeros((instance_ids.shape[0], 2 * C))
        for idx in range(instance_ids.shape[0]):
            b = int(instance_batch[idx].item())
            label = int(instance_ids[idx].item())
            mask = labels_ds[b] == label
            flat = feature[b].permute(1,2,3,0)[mask]
            if flat.numel() == 0:
                # Centroid fallback in relative physical coordinates.
                shape = torch.tensor(feature.shape[-3:], device=feature.device, dtype=feature.dtype)
                spacing = feature_spacing_um[b]
                extent = (shape - 1) * spacing
                vox = (instance_centroids_um[idx] + 0.5 * extent) / spacing.clamp_min(1e-8)
                vox = torch.round(vox).long()
                vox = torch.minimum(torch.maximum(vox, torch.zeros_like(vox)), shape.long() - 1)
                f = feature[b, :, vox[0], vox[1], vox[2]]
                pooled[idx] = torch.cat([f, f], dim=0).to(pooled.dtype)
            else:
                pooled[idx] = torch.cat([flat.mean(dim=0), flat.max(dim=0).values], dim=0).to(pooled.dtype)
        return pooled

    def forward(
        self,
        feature: Tensor,
        feature_spacing_um: Tensor,
        instance_labels: Tensor,
        instance_features: Tensor,
        instance_ids: Tensor,
        instance_batch: Tensor,
        instance_centroids_um: Tensor,
        dref_um: Tensor,
        temporal: TemporalState,
    ) -> QueryState:
        B = feature.shape[0]
        pooled = self._pool_instances(
            feature, feature_spacing_um, instance_labels, instance_ids, instance_batch, instance_centroids_um
        )
        inst_emb = self.feature_proj(pooled) + self.geom_proj(instance_features)
        split_counts = self.split_companion_counts(
            instance_labels, instance_ids, instance_batch
        )

        per_batch = []
        for b in range(B):
            parts, refs, types, srcids, sal, rel = [], [], [], [], [], []
            inst_idx = torch.nonzero(instance_batch == b, as_tuple=False).flatten()
            if inst_idx.numel():
                p = inst_emb[inst_idx]
                pref = instance_centroids_um[inst_idx] / dref_um[b].clamp_min(1e-8)
                primary = p + self.type_embedding.weight[QUERY_PRIMARY].to(p.dtype)
                parts.append(primary)
                refs.append(pref)
                types.append(torch.full((len(inst_idx),), QUERY_PRIMARY, device=feature.device, dtype=torch.long))
                srcids.append(instance_ids[inst_idx])
                zeros = feature.new_zeros((len(inst_idx), 1))
                sal.append(zeros)
                rel.append(zeros)
                per_instance_splits = split_counts[inst_idx]
                for slot in range(self.cfg.max_split_companions_per_instance):
                    eligible = per_instance_splits > slot
                    if not eligible.any():
                        continue
                    split = (
                        p[eligible]
                        + self.type_embedding.weight[QUERY_SPLIT].to(p.dtype)
                        + self.split_slot_embedding.weight[slot].to(p.dtype)
                    )
                    split_count = int(eligible.sum())
                    parts.append(split)
                    refs.append(pref[eligible])
                    types.append(torch.full((split_count,), QUERY_SPLIT, device=feature.device, dtype=torch.long))
                    srcids.append(instance_ids[inst_idx][eligible])
                    sal.append(feature.new_zeros((split_count, 1)))
                    rel.append(feature.new_zeros((split_count, 1)))

            tidx = torch.nonzero(temporal.batch_index == b, as_tuple=False).flatten() if not temporal.is_empty else torch.empty(0, dtype=torch.long, device=feature.device)
            if tidx.numel():
                tq = temporal.tokens[tidx]
                tq = tq + self.type_embedding.weight[QUERY_TEMPORAL].to(tq.dtype)
                parts.append(tq)
                refs.append(temporal.ref_cellscale[tidx])
                types.append(torch.full((len(tidx),), QUERY_TEMPORAL, device=feature.device, dtype=torch.long))
                srcids.append(torch.full((len(tidx),), -1, device=feature.device, dtype=torch.long))
                sal.append(temporal.salience[tidx])
                rel.append(temporal.reliability[tidx])

            dq = self.discovery_queries.to(feature.dtype)
            dq = dq + self.type_embedding.weight[QUERY_DISCOVERY].to(dq.dtype)
            parts.append(dq)
            refs.append(self.discovery_refs)
            types.append(torch.full((self.cfg.discovery_queries,), QUERY_DISCOVERY, device=feature.device, dtype=torch.long))
            srcids.append(torch.full((self.cfg.discovery_queries,), -1, device=feature.device, dtype=torch.long))
            sal.append(feature.new_zeros((self.cfg.discovery_queries, 1)))
            rel.append(feature.new_zeros((self.cfg.discovery_queries, 1)))

            q = torch.cat(parts, dim=0)
            if self.cfg.max_queries is not None and q.shape[0] > self.cfg.max_queries:
                raise RuntimeError(
                    f"Query count {q.shape[0]} exceeds the explicit safety limit "
                    f"max_queries={self.cfg.max_queries}; raise or disable that limit "
                    "to retain the complete all-cell sample."
                )
            per_batch.append((q, torch.cat(refs), torch.cat(types), torch.cat(srcids), torch.cat(sal), torch.cat(rel)))

        qmax = max(x[0].shape[0] for x in per_batch)
        query_dtype = per_batch[0][0].dtype
        embeddings = torch.zeros((B, qmax, self.cfg.d_model), device=feature.device, dtype=query_dtype)
        references = torch.zeros((B, qmax, 3), device=feature.device, dtype=dref_um.dtype)
        query_types = torch.full((B, qmax), -1, device=feature.device, dtype=torch.long)
        source_ids = torch.full((B, qmax), -1, device=feature.device, dtype=torch.long)
        salience = torch.zeros((B, qmax, 1), device=feature.device, dtype=query_dtype)
        reliability = torch.zeros((B, qmax, 1), device=feature.device, dtype=query_dtype)
        padding = torch.ones((B, qmax), device=feature.device, dtype=torch.bool)
        for b, (q, r, t, s, sa, re) in enumerate(per_batch):
            n = q.shape[0]
            embeddings[b, :n] = q.to(embeddings.dtype)
            references[b, :n] = r.to(references.dtype)
            query_types[b, :n] = t
            source_ids[b, :n] = s
            salience[b, :n] = sa.to(salience.dtype)
            reliability[b, :n] = re.to(reliability.dtype)
            padding[b, :n] = False
        return QueryState(embeddings, references, query_types, padding, source_ids, salience, reliability)
