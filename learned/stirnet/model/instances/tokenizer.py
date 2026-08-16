from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from ..config import InstanceConfig, SpatialConfig
from ..types import (
    GeometryLike,
    InstanceState,
    PartitionState,
    RAGState,
    SpatialDecodeState,
    geometry_field,
    geometry_probability,
)
from ..utils.tensor_ops import (
    LabeledVoxelStats,
    pool_labeled_features,
    project_pooled_mean_max,
    reduce_labeled_voxels,
    resize_labels_nearest,
)


def _physical_points_from_flat_indices(
    flat_index: Tensor,
    shape: tuple[int, int, int],
    spacing_um: Tensor,
) -> Tensor:
    y_size, x_size = shape[1:]
    z = torch.div(flat_index, y_size * x_size, rounding_mode="floor")
    remainder = flat_index % (y_size * x_size)
    y = torch.div(remainder, x_size, rounding_mode="floor")
    x = remainder % x_size
    voxel = torch.stack([z, y, x], dim=-1).to(torch.float32)
    center = 0.5 * (
        torch.as_tensor(shape, device=flat_index.device, dtype=torch.float32) - 1
    )
    spacing = spacing_um.to(device=flat_index.device, dtype=torch.float32)
    return (voxel - center) * spacing


def centers_from_labels(
    labels: List[Tensor], spacing_um: Tensor, sdf: Tensor | None = None
) -> List[Tensor]:
    """Exactly one *interior* representative point per connected instance.

    The default final center is the maximum-SDF voxel, not a free learned query
    coordinate. Therefore a returned center is guaranteed to belong to its mask.
    """
    result: List[Tensor] = []
    for b, lab in enumerate(labels):
        stats = reduce_labeled_voxels(
            lab,
            spacing_um[b],
            argmax_field=None if sdf is None else sdf[b, 0],
        )
        if stats.counts.numel() == 0:
            result.append(spacing_um.new_zeros((0, 3)))
            continue
        flat_index = (
            stats.argmax_flat_index
            if sdf is not None
            else stats.nearest_centroid_flat_index
        )
        if flat_index is None:
            raise RuntimeError("argmax center reduction was not produced")
        centers = _physical_points_from_flat_indices(
            flat_index, tuple(lab.shape), spacing_um[b]
        )
        centers = centers.to(device=spacing_um.device, dtype=spacing_um.dtype)
        centers[stats.counts.to(centers.device) == 0] = 0
        result.append(centers)
    return result


def _pad_pooled(pooled: Tensor, n: int) -> Tensor:
    if pooled.shape[0] == n:
        return pooled
    if pooled.shape[0] > n:
        return pooled[:n]
    return torch.cat([pooled, pooled.new_zeros((n - pooled.shape[0], pooled.shape[1]))], dim=0)


def _shape_features(
    labels: Tensor,
    geometry: GeometryLike,
    b: int,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    stats: LabeledVoxelStats | None = None,
) -> Tensor:
    n = int(labels.max().item())
    if n == 0:
        return geometry_field(geometry, "sdf").new_zeros((0, 12))
    stats = stats or _geometry_stats(labels, geometry, b, spacing_um)
    positive = stats.counts > 0
    median_count = stats.counts[positive].median().clamp_min(1)
    extent = (
        (stats.max_voxel - stats.min_voxel).to(torch.float32)
        * spacing_um.float()[None]
        / dref_um.float().clamp_min(1e-6)
    )
    variance = stats.variance_um2 / dref_um.float().square().clamp_min(1e-6)
    result = torch.cat(
        [
            (
                torch.log1p(stats.counts) - torch.log1p(median_count)
            )[:, None],
            extent,
            variance,
            stats.field_means["sdf"][:, None],
            stats.field_maxima["sdf"][:, None],
            stats.field_means["separator"][:, None],
            stats.field_maxima["separator"][:, None],
            stats.field_means["foreground"][:, None],
        ],
        dim=-1,
    )
    return torch.where(positive[:, None], result, torch.zeros_like(result))


def _geometry_stats(
    labels: Tensor,
    geometry: GeometryLike,
    batch_index: int,
    spacing_um: Tensor,
) -> LabeledVoxelStats:
    sdf = geometry_field(geometry, "sdf")[batch_index, 0]
    return reduce_labeled_voxels(
        labels,
        spacing_um,
        fields={
            "sdf": sdf,
            "separator": geometry_probability(geometry, "separator")[
                batch_index, 0
            ],
            "foreground": geometry_probability(geometry, "foreground")[
                batch_index, 0
            ],
        },
        argmax_field=sdf,
    )


class InstanceTokenizer(nn.Module):
    """Create tokens only after a connected spatial object already exists."""

    def __init__(self, cfg: InstanceConfig, spatial_cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.pooled_feature_dim
        self.proj_d0 = nn.Linear(spatial_cfg.channels[0], p, bias=False)
        self.proj_d1 = nn.Linear(spatial_cfg.channels[1], p, bias=False)
        self.proj_d2 = nn.Linear(spatial_cfg.channels[2], p, bias=False)
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in ("proj_d0", "proj_d1", "proj_d2"):
            key = f"{prefix}{name}.weight"
            weight = state_dict.get(key)
            if weight is not None and weight.ndim == 5 and weight.shape[-3:] == (1, 1, 1):
                state_dict[key] = weight[..., 0, 0, 0]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        partition: PartitionState,
        rag: RAGState,
        decoded: SpatialDecodeState,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        pooled_supervoxel_scales: tuple[List[Tensor], List[Tensor], List[Tensor]] | None = None,
        pooled_supervoxel_counts: tuple[List[Tensor], List[Tensor], List[Tensor]] | None = None,
    ) -> InstanceState:
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
            if pooled_supervoxel_scales is None:
                for feature, projection in zip(
                    (decoded.d0[b], decoded.d1[b], decoded.d2[b]),
                    (self.proj_d0, self.proj_d1, self.proj_d2),
                ):
                    lab_scale = resize_labels_nearest(
                        labels, tuple(feature.shape[-3:])
                    )
                    pooled, _ = pool_labeled_features(feature, lab_scale)
                    pooled_scales.append(
                        _pad_pooled(
                            project_pooled_mean_max(pooled, projection), n
                        )
                    )
            else:
                if pooled_supervoxel_counts is None:
                    raise ValueError(
                        "streamed pooled scales require matching count rows"
                    )
                start = int(rag.node_offsets[b].item())
                stop = int(rag.node_offsets[b + 1].item())
                component = partition.node_component[start:stop]
                for scale, projection in enumerate(
                    (self.proj_d0, self.proj_d1, self.proj_d2)
                ):
                    node_pooled = pooled_supervoxel_scales[scale][b]
                    node_counts = pooled_supervoxel_counts[scale][b]
                    node_pooled = _pad_pooled(node_pooled, stop - start)
                    node_counts = torch.cat(
                        [
                            node_counts[: stop - start],
                            node_counts.new_zeros(
                                (max(stop - start - node_counts.shape[0], 0),)
                            ),
                        ]
                    )
                    mean, maximum = node_pooled.chunk(2, dim=-1)
                    instance_counts = node_counts.new_zeros((n,))
                    instance_counts.index_add_(0, component, node_counts)
                    instance_sums = mean.new_zeros((n, mean.shape[1]))
                    instance_sums.index_add_(
                        0, component, mean * node_counts[:, None]
                    )
                    instance_mean = instance_sums / instance_counts.clamp_min(1)[:, None]
                    instance_max = maximum.new_full((n, maximum.shape[1]), -torch.inf)
                    valid_nodes = node_counts > 0
                    if valid_nodes.any():
                        instance_max.scatter_reduce_(
                            0,
                            component[valid_nodes, None].expand(
                                -1, maximum.shape[1]
                            ),
                            maximum[valid_nodes],
                            reduce="amax",
                            include_self=True,
                        )
                    instance_max = torch.where(
                        torch.isfinite(instance_max),
                        instance_max,
                        torch.zeros_like(instance_max),
                    )
                    pooled_scales.append(
                        project_pooled_mean_max(
                            torch.cat([instance_mean, instance_max], dim=-1),
                            projection,
                        )
                    )
            geometry_means: list[Tensor] = []
            geometry_maxima: list[Tensor] = []

            def append_geometry(dense_field: Tensor) -> None:
                pooled_field, _ = pool_labeled_features(dense_field, labels)
                mean, maximum = pooled_field.chunk(2, dim=-1)
                geometry_means.append(mean)
                geometry_maxima.append(maximum)

            append_geometry(geometry_probability(geometry, "foreground")[b])
            append_geometry(geometry_probability(geometry, "surface")[b])
            append_geometry(geometry_probability(geometry, "separator")[b])
            append_geometry(geometry_field(geometry, "sdf")[b])
            append_geometry(geometry_field(geometry, "flow")[b])
            append_geometry(geometry_field(geometry, "centroid_offset")[b])
            append_geometry(geometry_probability(geometry, "seed")[b])
            pooled_geometry = torch.cat(
                [*geometry_means, *geometry_maxima], dim=-1
            )
            pooled_geometry = _pad_pooled(pooled_geometry, n)
            stats = _geometry_stats(
                labels, geometry, b, spacing_um[b]
            )
            shape = _shape_features(
                labels,
                geometry,
                b,
                spacing_um[b],
                dref_um[b],
                stats=stats,
            )
            feature = torch.cat([*pooled_scales, pooled_geometry, shape], dim=-1)
            tokens.append(self.token_mlp(feature))
            if stats.argmax_flat_index is None:
                raise RuntimeError("SDF argmax centers were not reduced")
            refs.append(
                _physical_points_from_flat_indices(
                    stats.argmax_flat_index,
                    tuple(labels.shape),
                    spacing_um[b],
                ).to(spacing_um.dtype)
            )
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
