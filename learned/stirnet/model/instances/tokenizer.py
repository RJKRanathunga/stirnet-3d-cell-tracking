from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from ..config import InstanceConfig, SpatialConfig
from ..types import GeometryState, InstanceState, PartitionState, RAGState, SpatialDecodeState
from ..utils.physical import relative_grid_coordinates_um
from ..utils.tensor_ops import pool_labeled_features, resize_labels_nearest


def centers_from_labels(
    labels: List[Tensor], spacing_um: Tensor, sdf: Tensor | None = None
) -> List[Tensor]:
    """Exactly one *interior* representative point per connected instance.

    The default final center is the maximum-SDF voxel, not a free learned query
    coordinate. Therefore a returned center is guaranteed to belong to its mask.
    """
    result: List[Tensor] = []
    for b, lab in enumerate(labels):
        n = int(lab.max().item())
        if n == 0:
            result.append(spacing_um.new_zeros((0, 3)))
            continue
        coords = relative_grid_coordinates_um(
            tuple(lab.shape), spacing_um[b], device=lab.device
        )
        centers = []
        for instance_id in range(1, n + 1):
            mask = lab == instance_id
            points = torch.nonzero(mask, as_tuple=False)
            if points.numel() == 0:
                centers.append(spacing_um.new_zeros((3,)))
                continue
            if sdf is not None:
                values = sdf[b, 0][mask]
                voxel = points[values.argmax()]
            else:
                # Pick the in-mask voxel nearest the geometric centroid.
                centroid = points.float().mean(0)
                voxel = points[torch.linalg.vector_norm(points.float() - centroid, dim=-1).argmin()]
            centers.append(coords[voxel[0], voxel[1], voxel[2]])
        result.append(torch.stack(centers))
    return result


def _pad_pooled(pooled: Tensor, n: int) -> Tensor:
    if pooled.shape[0] == n:
        return pooled
    if pooled.shape[0] > n:
        return pooled[:n]
    return torch.cat([pooled, pooled.new_zeros((n - pooled.shape[0], pooled.shape[1]))], dim=0)


def _shape_features(
    labels: Tensor,
    geometry: GeometryState,
    b: int,
    spacing_um: Tensor,
    dref_um: Tensor,
) -> Tensor:
    n = int(labels.max().item())
    if n == 0:
        return geometry.sdf.new_zeros((0, 12))
    coords = relative_grid_coordinates_um(
        tuple(labels.shape), spacing_um, device=labels.device
    )
    probs = geometry.probabilities()
    rows = []
    counts_all = torch.stack([(labels == i).sum() for i in range(1, n + 1)]).float()
    median_count = counts_all[counts_all > 0].median().clamp_min(1)
    for instance_id in range(1, n + 1):
        mask = labels == instance_id
        xyz = coords[mask]
        if xyz.numel() == 0:
            rows.append(geometry.sdf.new_zeros((12,)))
            continue
        extent = (xyz.max(0).values - xyz.min(0).values) / dref_um.clamp_min(1e-6)
        var = xyz.var(0, unbiased=False) / dref_um.square().clamp_min(1e-6)
        sdf_vals = geometry.sdf[b, 0][mask]
        sep_vals = probs["separator"][b, 0][mask]
        fg_vals = probs["foreground"][b, 0][mask]
        row = torch.cat(
            [
                torch.log1p(mask.sum().float())[None] - torch.log1p(median_count)[None],
                extent,
                var,
                sdf_vals.mean()[None],
                sdf_vals.max()[None],
                sep_vals.mean()[None],
                sep_vals.max()[None],
                fg_vals.mean()[None],
            ]
        )
        rows.append(row)
    return torch.stack(rows)


class InstanceTokenizer(nn.Module):
    """Create tokens only after a connected spatial object already exists."""

    def __init__(self, cfg: InstanceConfig, spatial_cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.pooled_feature_dim
        self.proj_d0 = nn.Conv3d(spatial_cfg.channels[0], p, 1, bias=False)
        self.proj_d1 = nn.Conv3d(spatial_cfg.channels[1], p, 1, bias=False)
        self.proj_d2 = nn.Conv3d(spatial_cfg.channels[2], p, 1, bias=False)
        # Geometry pooled channels: fg, surface, separator, sdf, flow(3),
        # centroid offset(3), seed = 11; mean+max -> 22.
        in_dim = 3 * (2 * p) + 22 + cfg.shape_feature_dim
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * cfg.d_model),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )
        self.quality = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1)
        )

    def forward(
        self,
        partition: PartitionState,
        rag: RAGState,
        decoded: SpatialDecodeState,
        geometry: GeometryState,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> InstanceState:
        d0 = self.proj_d0(decoded.d0)
        d1 = self.proj_d1(decoded.d1)
        d2 = self.proj_d2(decoded.d2)
        probs = geometry.probabilities()
        tokens = []
        refs = []
        batches = []
        local_ids = []
        offsets = [0]
        for b, labels in enumerate(partition.labels):
            n = int(labels.max().item())
            if n == 0:
                offsets.append(offsets[-1])
                continue
            pooled_scales = []
            for feature in (d0[b], d1[b], d2[b]):
                lab_scale = resize_labels_nearest(labels, tuple(feature.shape[-3:]))
                pooled, _ = pool_labeled_features(feature, lab_scale)
                pooled_scales.append(_pad_pooled(pooled, n))
            dense = torch.cat(
                [
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
            pooled_geometry, _ = pool_labeled_features(dense, labels)
            pooled_geometry = _pad_pooled(pooled_geometry, n)
            shape = _shape_features(
                labels, geometry, b, spacing_um[b], dref_um[b]
            )
            feature = torch.cat([*pooled_scales, pooled_geometry, shape], dim=-1)
            tokens.append(self.token_mlp(feature))
            refs.append(centers_from_labels([labels], spacing_um[b:b+1], geometry.sdf[b:b+1])[0])
            batches.append(torch.full((n,), b, device=labels.device, dtype=torch.long))
            local_ids.append(torch.arange(1, n + 1, device=labels.device, dtype=torch.long))
            offsets.append(offsets[-1] + n)
        device = decoded.d0.device
        if tokens:
            token = torch.cat(tokens)
            ref = torch.cat(refs)
            batch = torch.cat(batches)
            ids = torch.cat(local_ids)
        else:
            token = decoded.d0.new_zeros((0, self.cfg.d_model))
            ref = spacing_um.new_zeros((0, 3))
            batch = torch.zeros(0, device=device, dtype=torch.long)
            ids = torch.zeros(0, device=device, dtype=torch.long)
        return InstanceState(
            tokens=token,
            ref_um=ref,
            batch_index=batch,
            local_ids=ids,
            quality_logits=self.quality(token).squeeze(-1) if token.numel() else token.new_zeros((0,)),
            labels=partition.labels,
            token_offsets=torch.tensor(offsets, device=device, dtype=torch.long),
            node_to_instance=partition.node_component_global,
            spatial_tokens=token,
        )
