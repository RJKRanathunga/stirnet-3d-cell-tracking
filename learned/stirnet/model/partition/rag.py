from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import PartitionConfig, SpatialConfig
from ..types import (
    GeometryLike,
    GeometryDerivedCache,
    GeometryState,
    RAGState,
    SpatialDecodeState,
    SupervoxelStatistics,
    geometry_field,
    geometry_field_crop,
    geometry_probability,
)
from ..utils.tensor_ops import (
    pool_labeled_features,
    project_pooled_mean_max,
    reduce_labeled_voxels,
    resize_labels_nearest,
)
from ..utils.contingency import label_contingency
from .statistics import (
    NATIVE_FIELD_ORDER,
    local_dependency_box,
    local_refresh_groups,
    update_supervoxel_statistics_local,
)
from .morphology import RAGMorphologyEmbeddingBuilder
from .separator_barrier import build_separator_barrier_features


def _profile(profiler, name: str):
    return nullcontext() if profiler is None else profiler.profile(name)


def _adjacent_pairs_and_stats(
    labels: Tensor,
    geometry: GeometryLike,
    batch_index: int,
    *,
    derived_cache: GeometryDerivedCache | None = None,
    stage_profiler=None,
    profile_prefix: str = "rag",
) -> tuple[Tensor, Tensor]:
    """Extract undirected RAG edges and 8 interface features on-device."""
    with _profile(stage_profiler, f"{profile_prefix}_adjacency_field_gather"):
        if derived_cache is None:
            separator_field = geometry_probability(geometry, "separator")[batch_index, 0]
            surface_field = geometry_probability(geometry, "surface")[batch_index, 0]
            foreground_field = geometry_probability(geometry, "foreground")[batch_index, 0]
            sdf_field = geometry_field(geometry, "sdf")[batch_index, 0]
        else:
            separator_field = derived_cache.separator_prob[batch_index, 0]
            surface_field = derived_cache.surface_prob[batch_index, 0]
            foreground_field = derived_cache.foreground_prob[batch_index, 0]
            sdf_field = derived_cache.sdf[batch_index, 0]
        flow_field = geometry_field(geometry, "flow")[batch_index]
        centroid_field = geometry_field(geometry, "centroid_offset")[batch_index]
    pair_chunks = []
    stat_chunks = []
    with _profile(stage_profiler, f"{profile_prefix}_adjacency_pair_extract"):
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

        sep = pair_scalar(separator_field)
        surf = pair_scalar(surface_field)
        fg = pair_scalar(foreground_field)
        sdf_abs = pair_scalar(sdf_field.abs())

        fa = flow_field[:, tuple(a_slice)[0], tuple(a_slice)[1], tuple(a_slice)[2]][:, valid].T
        fb = flow_field[:, tuple(b_slice)[0], tuple(b_slice)[1], tuple(b_slice)[2]][:, valid].T
        flow_disagree = 1.0 - F.cosine_similarity(fa, fb, dim=-1, eps=1e-6)
        oa = centroid_field[:, tuple(a_slice)[0], tuple(a_slice)[1], tuple(a_slice)[2]][:, valid].T
        ob = centroid_field[:, tuple(b_slice)[0], tuple(b_slice)[1], tuple(b_slice)[2]][:, valid].T
        offset_disagree = torch.linalg.vector_norm(oa - ob, dim=-1)
        # Duplicate sep/surface values are aggregated as mean and max later.
        stat_chunks.append(
            torch.stack(
                [sep, surf, fg, sdf_abs, flow_disagree, offset_disagree], dim=-1
            )
        )
    if not pair_chunks:
        reference = geometry_field(geometry, "sdf")
        return labels.new_zeros((2, 0)), reference.new_zeros((0, 8))
    pairs = torch.cat(pair_chunks, dim=0)
    stats = torch.cat(stat_chunks, dim=0)
    n_nodes = int(labels.max().item())
    packed = pairs[:, 0] * max(n_nodes, 1) + pairs[:, 1]
    with _profile(stage_profiler, f"{profile_prefix}_edge_pair_unique_or_sort"):
        order = torch.argsort(packed)
        packed = packed[order]
        stats = stats[order]
        unique, inverse = torch.unique_consecutive(packed, return_inverse=True)
    edge_count = len(unique)
    with _profile(stage_profiler, f"{profile_prefix}_edge_reduce"):
        means = stats.new_zeros((edge_count, stats.shape[1]))
        counts = stats.new_zeros((edge_count,))
        means.index_add_(0, inverse, stats)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=stats.dtype))
        means = means / counts.clamp_min(1)[:, None]
        maxima = stats.new_full((edge_count, 2), -torch.inf)
        maxima.scatter_reduce_(
            0, inverse[:, None].expand(-1, 2), stats[:, :2],
            reduce="amax", include_self=True,
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


def _dedupe_local_edges(
    edges: Tensor,
    edge_features: Tensor,
    node_count: int,
) -> tuple[Tensor, Tensor]:
    """Deterministically remove duplicate incident edges from overlapping jobs."""
    if edges.shape[1] <= 1:
        return edges, edge_features
    packed = edges[0].long() * max(int(node_count), 1) + edges[1].long()
    order = torch.argsort(packed)
    packed = packed[order]
    ordered_edges = edges[:, order]
    ordered_features = edge_features[order]
    keep = torch.ones(
        packed.shape[0],
        device=packed.device,
        dtype=torch.bool,
    )
    keep[1:] = packed[1:] != packed[:-1]
    return ordered_edges[:, keep], ordered_features[keep]


class RAGBuilder(nn.Module):
    """Construct a supervoxel RAG without compressing topology into a center token."""

    def __init__(self, cfg: PartitionConfig, spatial_cfg: SpatialConfig):
        super().__init__()
        self.cfg = cfg
        self.node_projection = nn.Linear(
            spatial_cfg.channels[0], cfg.node_feature_channels, bias=False
        )
        # 2*C D0 summary + 2*12 dense/raw summary + centroid(3)+log-volume(1)
        self.node_feature_dim = 2 * cfg.node_feature_channels + 24 + 4
        self.edge_feature_dim = 8
        self.morphology_builder = (
            RAGMorphologyEmbeddingBuilder(cfg)
            if cfg.rag_morphology_enabled
            else None
        )

    def _attach_morphology(
        self,
        rag: RAGState,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        stage_profiler=None,
        profile_prefix: str = "rag",
    ) -> RAGState:
        # STIRNET_SEPARATOR_BARRIER_FEATURE_ATTACHMENT_FIX_V1
        #
        # The production graph network consumes separator_barrier_features
        # whenever rag_separator_barrier_enabled=True. These features belong
        # to RAG construction and must be attached BEFORE
        # SpatialRAGNetwork.forward().
        #
        # Keep this independent of morphology: barrier configurations that do
        # not use morphology still require the exact-contact features.
        if self.cfg.rag_separator_barrier_enabled:
            with _profile(
                stage_profiler,
                f"{profile_prefix}_separator_barrier_features",
            ):
                separator_features = build_separator_barrier_features(
                    rag,
                    geometry,
                    spacing_um,
                    dref_um,
                    detach_geometry=(
                        self.cfg.rag_separator_barrier_detach_geometry
                    ),
                )
            rag = replace(
                rag,
                separator_barrier_features=separator_features,
            )

        if self.morphology_builder is None:
            return rag

        with _profile(stage_profiler, f"{profile_prefix}_morphology_embedding"):
            node_morphology, edge_morphology = self.morphology_builder(
                rag,
                spatial_inputs,
                geometry,
                spacing_um,
                dref_um,
            )
        return replace(
            rag,
            node_morphology_embeddings=node_morphology,
            edge_morphology_embeddings=edge_morphology,
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
        key = f"{prefix}node_projection.weight"
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

    @staticmethod
    def _label_bbox(
        labels: Tensor,
        label_ids: Tensor,
        *,
        halo: int = 0,
    ) -> tuple[slice, slice, slice]:
        mask = torch.isin(labels, label_ids)
        points = torch.nonzero(mask, as_tuple=False)
        if points.numel() == 0:
            return tuple(slice(0, size) for size in labels.shape)  # type: ignore[return-value]
        lower = (points.min(dim=0).values - halo).clamp_min(0)
        upper = torch.minimum(
            points.max(dim=0).values + halo + 1,
            torch.as_tensor(labels.shape, device=labels.device),
        )
        return tuple(
            slice(int(start), int(stop))
            for start, stop in zip(lower.tolist(), upper.tolist())
        )  # type: ignore[return-value]

    def _node_rows_from_statistics(
        self,
        statistics: SupervoxelStatistics,
        reference: Tensor,
        dref_um: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        counts = statistics.counts.to(reference.dtype)
        pooled_d0 = project_pooled_mean_max(
            statistics.scales[0].mean_max, self.node_projection
        )
        dense = torch.cat(
            [
                torch.stack(
                    [statistics.field_means(name) for name in NATIVE_FIELD_ORDER], dim=-1
                ),
                torch.stack(
                    [statistics.field_maxima[name] for name in NATIVE_FIELD_ORDER], dim=-1
                ),
            ],
            dim=-1,
        )
        centroids = statistics.centroid_um.to(reference.dtype)
        positive = counts > 0
        median = counts[positive].median().clamp_min(1) if positive.any() else counts.new_tensor(1)
        volume = (torch.log1p(counts) - torch.log1p(median))[:, None]
        nodes = torch.cat(
            [pooled_d0, dense, centroids / dref_um.clamp_min(1e-6), volume], dim=-1
        )
        nodes = torch.where(positive[:, None], nodes, torch.zeros_like(nodes))
        return nodes, centroids, counts

    def _update_local_cached(
        self,
        initial_rag: RAGState,
        supervoxel_labels: List[Tensor],
        updated_boxes: List[tuple[int, tuple[slice, slice, slice]]],
        decoded: SpatialDecodeState,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        updated_label_ids: List[Tensor] | None = None,
        stage_profiler=None,
        profile_prefix: str = "rag_local",
    ) -> RAGState:
        if initial_rag.statistics is None:
            raise ValueError("cached local update requires initial statistics")

        with _profile(
            stage_profiler,
            f"{profile_prefix}_statistics_refresh",
        ):
            statistics, affected_by_batch = update_supervoxel_statistics_local(
                initial_rag.statistics,
                initial_rag.supervoxel_labels,
                supervoxel_labels,
                updated_boxes,
                spatial_inputs,
                geometry,
                spacing_um,
                (decoded.d0, decoded.d1, decoded.d2),
                updated_label_ids=updated_label_ids,
                stage_profiler=stage_profiler,
                profile_prefix=f"{profile_prefix}_statistics",
            )
            refresh_groups = local_refresh_groups(
                initial_rag.supervoxel_labels,
                supervoxel_labels,
                updated_boxes,
                updated_label_ids=updated_label_ids,
            )

        groups_by_batch: dict[
            int,
            list[tuple[tuple[slice, slice, slice], Tensor]],
        ] = {}
        for batch_index, box, affected in refresh_groups:
            groups_by_batch.setdefault(batch_index, []).append(
                (box, affected)
            )

        all_nodes: list[Tensor] = []
        all_centroids: list[Tensor] = []
        all_volumes: list[Tensor] = []
        all_node_batch: list[Tensor] = []
        all_node_ids: list[Tensor] = []
        all_edges: list[Tensor] = []
        all_edge_features: list[Tensor] = []
        all_edge_batch: list[Tensor] = []
        offsets = [0]

        for b, (labels, stats, affected_all) in enumerate(
            zip(supervoxel_labels, statistics, affected_by_batch)
        ):
            with _profile(
                stage_profiler,
                f"{profile_prefix}_node_assembly",
            ):
                nodes, centroids, volumes = self._node_rows_from_statistics(
                    stats,
                    decoded.d0,
                    dref_um[b],
                )

            n = nodes.shape[0]
            old_start = int(initial_rag.node_offsets[b].item())
            old_edges = torch.nonzero(
                initial_rag.edge_batch == b,
                as_tuple=False,
            ).flatten()
            if old_edges.numel():
                old_local = initial_rag.edge_index[:, old_edges] - old_start
                keep = (
                    (old_local[0] < n)
                    & (old_local[1] < n)
                    & ~torch.isin(old_local[0] + 1, affected_all)
                    & ~torch.isin(old_local[1] + 1, affected_all)
                )
                kept_edges = old_local[:, keep]
                kept_features = initial_rag.edge_features[old_edges[keep]]
            else:
                kept_edges = labels.new_zeros((2, 0))
                kept_features = decoded.d0.new_zeros(
                    (0, self.edge_feature_dim)
                )

            refreshed_edge_chunks: list[Tensor] = []
            refreshed_feature_chunks: list[Tensor] = []
            with _profile(
                stage_profiler,
                f"{profile_prefix}_edge_refresh",
            ):
                for changed_box, group_affected in groups_by_batch.get(b, []):
                    current = group_affected[
                        group_affected <= stats.counts.shape[0]
                    ]
                    if current.numel():
                        rows = current - 1
                        current = current[stats.counts[rows] > 0]
                    if current.numel() == 0:
                        continue

                    edge_box = local_dependency_box(
                        stats,
                        current,
                        changed_box,
                        tuple(labels.shape),
                        halo=1,
                    )
                    edge_geometry = GeometryState(
                        foreground_logits=geometry_field_crop(
                            geometry,
                            "foreground_logits",
                            b,
                            edge_box,
                        )[None],
                        surface_logits=geometry_field_crop(
                            geometry,
                            "surface_logits",
                            b,
                            edge_box,
                        )[None],
                        separator_logits=geometry_field_crop(
                            geometry,
                            "separator_logits",
                            b,
                            edge_box,
                        )[None],
                        sdf=geometry_field_crop(
                            geometry,
                            "sdf",
                            b,
                            edge_box,
                        )[None],
                        flow=geometry_field_crop(
                            geometry,
                            "flow",
                            b,
                            edge_box,
                        )[None],
                        centroid_offset=geometry_field_crop(
                            geometry,
                            "centroid_offset",
                            b,
                            edge_box,
                        )[None],
                        seed_logits=geometry_field_crop(
                            geometry,
                            "seed_logits",
                            b,
                            edge_box,
                        )[None],
                        features=None,
                    )
                    candidate_edges, candidate_features = (
                        _adjacent_pairs_and_stats(
                            labels[edge_box],
                            edge_geometry,
                            0,
                        )
                    )
                    if candidate_edges.shape[1] == 0:
                        continue
                    incident = (
                        torch.isin(
                            candidate_edges[0] + 1,
                            current,
                        )
                        | torch.isin(
                            candidate_edges[1] + 1,
                            current,
                        )
                    )
                    if incident.any():
                        refreshed_edge_chunks.append(
                            candidate_edges[:, incident]
                        )
                        refreshed_feature_chunks.append(
                            candidate_features[incident]
                        )

                if refreshed_edge_chunks:
                    refreshed_edges = torch.cat(
                        refreshed_edge_chunks,
                        dim=1,
                    )
                    refreshed_features = torch.cat(
                        refreshed_feature_chunks,
                        dim=0,
                    )
                    refreshed_edges, refreshed_features = (
                        _dedupe_local_edges(
                            refreshed_edges,
                            refreshed_features,
                            n,
                        )
                    )
                else:
                    refreshed_edges = labels.new_zeros((2, 0))
                    refreshed_features = decoded.d0.new_zeros(
                        (0, self.edge_feature_dim)
                    )

            edges = torch.cat(
                [kept_edges, refreshed_edges],
                dim=1,
            )
            edge_features = torch.cat(
                [kept_features, refreshed_features],
                dim=0,
            )

            offset = offsets[-1]
            if edges.shape[1]:
                all_edges.append(edges + offset)
                all_edge_features.append(edge_features)
                all_edge_batch.append(
                    torch.full(
                        (edges.shape[1],),
                        b,
                        device=labels.device,
                        dtype=torch.long,
                    )
                )
            all_nodes.append(nodes)
            all_centroids.append(centroids)
            all_volumes.append(volumes)
            all_node_batch.append(
                torch.full(
                    (n,),
                    b,
                    device=labels.device,
                    dtype=torch.long,
                )
            )
            all_node_ids.append(
                torch.arange(
                    1,
                    n + 1,
                    device=labels.device,
                    dtype=torch.long,
                )
            )
            offsets.append(offset + n)

        node_features = (
            torch.cat(all_nodes)
            if all_nodes
            else decoded.d0.new_zeros((0, self.node_feature_dim))
        )
        edge_features = (
            torch.cat(all_edge_features)
            if all_edge_features
            else decoded.d0.new_zeros((0, self.edge_feature_dim))
        )
        edge_index = (
            torch.cat(all_edges, dim=1)
            if all_edges
            else torch.zeros(
                (2, 0),
                device=decoded.d0.device,
                dtype=torch.long,
            )
        )
        edge_batch = (
            torch.cat(all_edge_batch)
            if all_edge_batch
            else torch.zeros(
                (0,),
                device=decoded.d0.device,
                dtype=torch.long,
            )
        )

        rag = RAGState(
            node_features=node_features,
            node_embeddings=decoded.d0.new_zeros(
                (node_features.shape[0], self.cfg.rag_hidden_dim)
            ),
            node_batch=(
                torch.cat(all_node_batch)
                if all_node_batch
                else torch.zeros(
                    (0,),
                    device=decoded.d0.device,
                    dtype=torch.long,
                )
            ),
            node_supervoxel_id=(
                torch.cat(all_node_ids)
                if all_node_ids
                else torch.zeros(
                    (0,),
                    device=decoded.d0.device,
                    dtype=torch.long,
                )
            ),
            node_centroid_um=(
                torch.cat(all_centroids)
                if all_centroids
                else decoded.d0.new_zeros((0, 3))
            ),
            node_volume_voxels=(
                torch.cat(all_volumes)
                if all_volumes
                else decoded.d0.new_zeros((0,))
            ),
            edge_index=edge_index,
            edge_features=edge_features,
            edge_embeddings=decoded.d0.new_zeros(
                (edge_features.shape[0], self.cfg.rag_hidden_dim)
            ),
            spatial_edge_logits=decoded.d0.new_zeros(
                (edge_features.shape[0],)
            ),
            edge_batch=edge_batch,
            supervoxel_labels=supervoxel_labels,
            node_offsets=torch.tensor(
                offsets,
                device=decoded.d0.device,
                dtype=torch.long,
            ),
            statistics=statistics,
        )
        return self._attach_morphology(
            rag,
            spatial_inputs,
            geometry,
            spacing_um,
            dref_um,
            stage_profiler=stage_profiler,
            profile_prefix=profile_prefix,
        )

    def update_local(
        self,
        initial_rag: RAGState,
        supervoxel_labels: List[Tensor],
        updated_boxes: List[tuple[int, tuple[slice, slice, slice]]],
        d0: Tensor | SpatialDecodeState,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        updated_label_ids: List[Tensor] | None = None,
        stage_profiler=None,
        profile_prefix: str = "rag_local",
    ) -> RAGState:
        """Refresh only nodes/edges incident to locally changed supervoxels."""
        if initial_rag.statistics is not None and isinstance(d0, SpatialDecodeState):
            return self._update_local_cached(
                initial_rag,
                supervoxel_labels,
                updated_boxes,
                d0,
                spatial_inputs,
                geometry,
                spacing_um,
                dref_um,
                updated_label_ids=updated_label_ids,
                stage_profiler=stage_profiler,
                profile_prefix=profile_prefix,
            )
        if isinstance(d0, SpatialDecodeState):
            d0 = d0.d0
        boxes_by_batch: dict[int, list[tuple[slice, slice, slice]]] = {}
        for batch_index, box in updated_boxes:
            boxes_by_batch.setdefault(batch_index, []).append(box)

        all_nodes: list[Tensor] = []
        all_centroids: list[Tensor] = []
        all_volumes: list[Tensor] = []
        all_node_batch: list[Tensor] = []
        all_node_ids: list[Tensor] = []
        all_edges: list[Tensor] = []
        all_edge_features: list[Tensor] = []
        all_edge_batch: list[Tensor] = []
        node_offsets = [0]

        for b, labels in enumerate(supervoxel_labels):
            n = int(labels.max().item())
            old_start = int(initial_rag.node_offsets[b].item())
            old_stop = int(initial_rag.node_offsets[b + 1].item())
            old_n = old_stop - old_start
            nodes = d0.new_zeros((n, self.node_feature_dim))
            centroids = d0.new_zeros((n, 3))
            volumes = d0.new_zeros((n,))
            copy_count = min(n, old_n)
            if copy_count:
                nodes[:copy_count] = initial_rag.node_features[
                    old_start : old_start + copy_count
                ]
                centroids[:copy_count] = initial_rag.node_centroid_um[
                    old_start : old_start + copy_count
                ]
                volumes[:copy_count] = initial_rag.node_volume_voxels[
                    old_start : old_start + copy_count
                ]

            affected_ids = []
            for box in boxes_by_batch.get(b, []):
                affected_ids.append(torch.unique(labels[box]))
                affected_ids.append(
                    torch.unique(initial_rag.supervoxel_labels[b][box])
                )
            if affected_ids:
                affected_all = torch.unique(torch.cat(affected_ids))
                affected_all = affected_all[affected_all > 0]
                affected = affected_all[affected_all <= n]
            else:
                affected_all = labels.new_zeros((0,))
                affected = labels.new_zeros((0,))

            if affected.numel():
                bbox = self._label_bbox(labels, affected)
                label_crop = labels[bbox]
                pooled_means: list[Tensor] = []
                pooled_maxima: list[Tensor] = []

                def append_pooled(field: Tensor) -> None:
                    pooled, _ = pool_labeled_features(field, label_crop)
                    mean, maximum = pooled.chunk(2, dim=-1)
                    pooled_means.append(mean)
                    pooled_maxima.append(maximum)

                append_pooled(spatial_inputs[b, :1, bbox[0], bbox[1], bbox[2]])
                append_pooled(
                    geometry_field_crop(geometry, "foreground_logits", b, bbox).sigmoid()
                )
                append_pooled(
                    geometry_field_crop(geometry, "surface_logits", b, bbox).sigmoid()
                )
                append_pooled(
                    geometry_field_crop(geometry, "separator_logits", b, bbox).sigmoid()
                )
                append_pooled(geometry_field_crop(geometry, "sdf", b, bbox))
                append_pooled(geometry_field_crop(geometry, "flow", b, bbox))
                append_pooled(
                    geometry_field_crop(geometry, "centroid_offset", b, bbox)
                )
                append_pooled(
                    geometry_field_crop(geometry, "seed_logits", b, bbox).sigmoid()
                )
                pooled_dense = torch.cat(
                    [*pooled_means, *pooled_maxima], dim=-1
                )

                labels_d0 = resize_labels_nearest(labels, tuple(d0.shape[-3:]))
                affected_d0 = affected.to(labels_d0.device)
                d0_bbox = self._label_bbox(labels_d0, affected_d0)
                pooled_raw_d0, _ = pool_labeled_features(
                    d0[b, :, d0_bbox[0], d0_bbox[1], d0_bbox[2]],
                    labels_d0[d0_bbox],
                )
                pooled_d0 = project_pooled_mean_max(
                    pooled_raw_d0, self.node_projection
                )
                stats = reduce_labeled_voxels(label_crop, spacing_um[b])
                crop_center = torch.tensor(
                    [
                        0.5 * (int(axis.start) + int(axis.stop) - 1)
                        for axis in bbox
                    ],
                    device=labels.device,
                    dtype=torch.float32,
                )
                global_center = 0.5 * (
                    torch.as_tensor(labels.shape, device=labels.device).float() - 1
                )
                centroid = stats.centroid_um + (
                    crop_center - global_center
                ) * spacing_um[b].float()
                counts = stats.counts.to(d0.dtype)
                global_counts = torch.bincount(
                    labels.reshape(-1).long(), minlength=n + 1
                )[1:].to(d0.dtype)
                median = global_counts.median().clamp_min(1)
                volume_feature = (
                    torch.log1p(counts) - torch.log1p(median)
                )[:, None]
                width = max(
                    pooled_d0.shape[0],
                    pooled_dense.shape[0],
                    centroid.shape[0],
                )

                def pad_rows(value: Tensor) -> Tensor:
                    if value.shape[0] >= width:
                        return value[:width]
                    return torch.cat(
                        [
                            value,
                            value.new_zeros((width - value.shape[0], value.shape[1])),
                        ],
                        dim=0,
                    )

                refreshed = torch.cat(
                    [
                        pad_rows(pooled_d0),
                        pad_rows(pooled_dense),
                        pad_rows(centroid.to(d0.dtype))
                        / dref_um[b].clamp_min(1e-6),
                        pad_rows(volume_feature),
                    ],
                    dim=-1,
                )
                affected_rows = affected.long() - 1
                nodes[affected_rows] = refreshed[affected_rows]
                centroids[affected_rows] = centroid.to(d0.dtype)[affected_rows]
                volumes[affected_rows] = counts[affected_rows]

            new_offset = node_offsets[-1]
            old_edges = torch.nonzero(
                initial_rag.edge_batch == b, as_tuple=False
            ).flatten()
            if old_edges.numel():
                old_local = initial_rag.edge_index[:, old_edges] - old_start
                keep = (
                    (old_local[0] < n)
                    & (old_local[1] < n)
                    & ~torch.isin(old_local[0] + 1, affected_all)
                    & ~torch.isin(old_local[1] + 1, affected_all)
                )
                kept_edges = old_local[:, keep]
                kept_features = initial_rag.edge_features[old_edges[keep]]
            else:
                kept_edges = labels.new_zeros((2, 0))
                kept_features = d0.new_zeros((0, self.edge_feature_dim))

            incident_edges = labels.new_zeros((2, 0))
            incident_features = d0.new_zeros((0, self.edge_feature_dim))
            if affected.numel():
                edge_bbox = self._label_bbox(labels, affected, halo=1)
                edge_geometry = GeometryState(
                    foreground_logits=geometry_field_crop(
                        geometry, "foreground_logits", b, edge_bbox
                    )[None],
                    surface_logits=geometry_field_crop(
                        geometry, "surface_logits", b, edge_bbox
                    )[None],
                    separator_logits=geometry_field_crop(
                        geometry, "separator_logits", b, edge_bbox
                    )[None],
                    sdf=geometry_field_crop(geometry, "sdf", b, edge_bbox)[None],
                    flow=geometry_field_crop(geometry, "flow", b, edge_bbox)[None],
                    centroid_offset=geometry_field_crop(
                        geometry, "centroid_offset", b, edge_bbox
                    )[None],
                    seed_logits=geometry_field_crop(
                        geometry, "seed_logits", b, edge_bbox
                    )[None],
                    features=None,
                )
                candidate_edges, candidate_features = _adjacent_pairs_and_stats(
                    labels[edge_bbox], edge_geometry, 0
                )
                incident = (
                    torch.isin(candidate_edges[0] + 1, affected)
                    | torch.isin(candidate_edges[1] + 1, affected)
                )
                incident_edges = candidate_edges[:, incident]
                incident_features = candidate_features[incident]

            edges = torch.cat([kept_edges, incident_edges], dim=1)
            edge_features = torch.cat(
                [kept_features, incident_features], dim=0
            )
            if edges.shape[1]:
                all_edges.append(edges + new_offset)
                all_edge_features.append(edge_features)
                all_edge_batch.append(
                    torch.full(
                        (edges.shape[1],),
                        b,
                        device=labels.device,
                        dtype=torch.long,
                    )
                )
            all_nodes.append(nodes)
            all_centroids.append(centroids)
            all_volumes.append(volumes)
            all_node_batch.append(
                torch.full((n,), b, device=labels.device, dtype=torch.long)
            )
            all_node_ids.append(
                torch.arange(1, n + 1, device=labels.device, dtype=torch.long)
            )
            node_offsets.append(new_offset + n)

        node_features = torch.cat(all_nodes) if all_nodes else d0.new_zeros((0, self.node_feature_dim))
        edge_features = torch.cat(all_edge_features) if all_edge_features else d0.new_zeros((0, self.edge_feature_dim))
        edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.zeros((2, 0), device=d0.device, dtype=torch.long)
        edge_batch = torch.cat(all_edge_batch) if all_edge_batch else torch.zeros((0,), device=d0.device, dtype=torch.long)
        rag = RAGState(
            node_features=node_features,
            node_embeddings=d0.new_zeros((node_features.shape[0], self.cfg.rag_hidden_dim)),
            node_batch=torch.cat(all_node_batch) if all_node_batch else torch.zeros((0,), device=d0.device, dtype=torch.long),
            node_supervoxel_id=torch.cat(all_node_ids) if all_node_ids else torch.zeros((0,), device=d0.device, dtype=torch.long),
            node_centroid_um=torch.cat(all_centroids) if all_centroids else d0.new_zeros((0, 3)),
            node_volume_voxels=torch.cat(all_volumes) if all_volumes else d0.new_zeros((0,)),
            edge_index=edge_index,
            edge_features=edge_features,
            edge_embeddings=d0.new_zeros((edge_features.shape[0], self.cfg.rag_hidden_dim)),
            spatial_edge_logits=d0.new_zeros((edge_features.shape[0],)),
            edge_batch=edge_batch,
            supervoxel_labels=supervoxel_labels,
            node_offsets=torch.tensor(node_offsets, device=d0.device, dtype=torch.long),
        )
        return self._attach_morphology(
            rag,
            spatial_inputs,
            geometry,
            spacing_um,
            dref_um,
            stage_profiler=stage_profiler,
            profile_prefix=profile_prefix,
        )

    def forward(
        self,
        supervoxel_labels: List[Tensor],
        d0: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
        pooled_d0_by_batch: List[Tensor] | None = None,
        statistics_by_batch: List[SupervoxelStatistics] | None = None,
        derived_cache: GeometryDerivedCache | None = None,
        stage_profiler=None,
        profile_prefix: str = "rag",
    ) -> RAGState:
        all_nodes = []
        all_centroids = []
        all_volumes = []
        all_node_batch = []
        all_node_ids = []
        all_edges = []
        all_edge_features = []
        all_edge_batch = []
        node_offsets = [0]

        for b, labels in enumerate(supervoxel_labels):
            n = int(labels.max().item())
            if n == 0:
                node_offsets.append(node_offsets[-1])
                continue
            cached_stats = (
                None if statistics_by_batch is None else statistics_by_batch[b]
            )
            if cached_stats is not None:
                pooled_raw_d0 = cached_stats.scales[0].mean_max
            elif pooled_d0_by_batch is None:
                labels_d0 = resize_labels_nearest(labels, tuple(d0.shape[-3:]))
                pooled_raw_d0, _ = pool_labeled_features(d0[b], labels_d0)
            else:
                pooled_raw_d0 = pooled_d0_by_batch[b]
            if pooled_raw_d0.shape[0] < n:
                pooled_raw_d0 = torch.cat(
                    [
                        pooled_raw_d0,
                        pooled_raw_d0.new_zeros(
                            (n - pooled_raw_d0.shape[0], pooled_raw_d0.shape[1])
                        ),
                    ],
                    dim=0,
                )
            label_stats = (
                None if cached_stats is not None
                else reduce_labeled_voxels(labels, spacing_um[b], need_nearest_centroid=False)
            )
            counts = (
                cached_stats.counts.to(d0.dtype)
                if cached_stats is not None
                else label_stats.counts.to(d0.dtype)
            )
            pooled_d0 = project_pooled_mean_max(
                pooled_raw_d0, self.node_projection
            )
            pooled_means: list[Tensor] = []
            pooled_maxima: list[Tensor] = []

            def append_pooled(dense_field: Tensor) -> None:
                pooled_field, _ = pool_labeled_features(dense_field, labels)
                mean, maximum = pooled_field.chunk(2, dim=-1)
                pooled_means.append(mean)
                pooled_maxima.append(maximum)

            if cached_stats is not None:
                pooled_dense = torch.stack(
                    [cached_stats.field_means(name) for name in NATIVE_FIELD_ORDER], dim=-1
                )
                pooled_dense = torch.cat(
                    [
                        pooled_dense,
                        torch.stack(
                            [cached_stats.field_maxima[name] for name in NATIVE_FIELD_ORDER], dim=-1
                        ),
                    ],
                    dim=-1,
                )
                pooled_coords = cached_stats.centroid_um.to(d0.dtype)
            else:
                append_pooled(spatial_inputs[b, :1])
                append_pooled(geometry_probability(geometry, "foreground")[b])
                append_pooled(geometry_probability(geometry, "surface")[b])
                append_pooled(geometry_probability(geometry, "separator")[b])
                append_pooled(geometry_field(geometry, "sdf")[b])
                append_pooled(geometry_field(geometry, "flow")[b])
                append_pooled(geometry_field(geometry, "centroid_offset")[b])
                append_pooled(geometry_probability(geometry, "seed")[b])
                pooled_dense = torch.cat([*pooled_means, *pooled_maxima], dim=-1)
                pooled_coords = label_stats.centroid_um.to(d0.dtype)
            with _profile(stage_profiler, f"{profile_prefix}_node_feature_assembly"):
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
                geometry,
                b,
                derived_cache=derived_cache,
                stage_profiler=stage_profiler,
                profile_prefix=profile_prefix,
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
        rag = RAGState(
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
            statistics=statistics_by_batch,
        )
        return self._attach_morphology(
            rag,
            spatial_inputs,
            geometry,
            spacing_um,
            dref_um,
            stage_profiler=stage_profiler,
            profile_prefix=profile_prefix,
        )


@dataclass(frozen=True)
class RAGTargets:
    target: Tensor
    valid: Tensor
    weight: Tensor
    node_purity: Tensor
    node_gt_support: Tensor
    dominant_gt: Tensor


class RAGCriterion(nn.Module):
    """Supervise whether adjacent supervoxels belong to the same GT instance."""

    def __init__(self, cfg: PartitionConfig | None = None):
        super().__init__()
        self.cfg = cfg or PartitionConfig()

    def build_targets(self, rag: RAGState, gt_labels: Tensor, *, valid_mask: Tensor | None = None) -> RAGTargets:
        dominant = torch.zeros(
            rag.node_features.shape[0], device=rag.node_features.device, dtype=torch.long
        )
        purity = rag.node_features.new_zeros((rag.node_features.shape[0],))
        support = rag.node_features.new_zeros((rag.node_features.shape[0],))
        for b, supervox in enumerate(rag.supervoxel_labels):
            gt = gt_labels[b].to(supervox.device).long()
            valid = None if valid_mask is None else torch.as_tensor(valid_mask[b], device=supervox.device).bool()
            if valid is not None and valid.ndim == 4 and valid.shape[0] == 1: valid = valid[0]
            start = int(rag.node_offsets[b].item())
            table = label_contingency(supervox, gt, valid_mask=valid)
            if not table.row_ids.numel():
                continue
            foreground_overlap = table.intersections.sum(dim=1)
            node_rows = start + table.row_ids - 1
            support[node_rows] = (
                foreground_overlap.float() / table.row_counts.clamp_min(1).float()
            )
            if not table.column_ids.numel():
                continue
            dominant_count, dominant_column = table.intersections.max(dim=1)
            has_foreground = foreground_overlap > 0
            dominant[node_rows[has_foreground]] = table.column_ids[
                dominant_column[has_foreground]
            ]
            purity[node_rows] = (
                dominant_count.float() / foreground_overlap.clamp_min(1).float()
            )
        if rag.edge_index.shape[1] == 0:
            empty = rag.node_features.new_zeros((0,))
            return RAGTargets(
                target=empty,
                valid=empty.bool(),
                weight=empty,
                node_purity=purity,
                node_gt_support=support,
                dominant_gt=dominant,
            )
        a = dominant[rag.edge_index[0]]
        b = dominant[rag.edge_index[1]]
        target = ((a > 0) & (a == b)).float()
        valid = (
            (a > 0)
            & (b > 0)
            & (purity[rag.edge_index[0]] >= self.cfg.rag_min_node_purity)
            & (purity[rag.edge_index[1]] >= self.cfg.rag_min_node_purity)
            & (
                support[rag.edge_index[0]]
                >= self.cfg.rag_min_node_gt_support
            )
            & (
                support[rag.edge_index[1]]
                >= self.cfg.rag_min_node_gt_support
            )
        )
        return RAGTargets(
            target=target,
            valid=valid,
            weight=torch.ones_like(target),
            node_purity=purity,
            node_gt_support=support,
            dominant_gt=dominant,
        )

    def forward(
        self,
        rag: RAGState,
        gt_labels: Tensor,
        *,
        logits: Tensor | None = None,
        targets: RAGTargets | None = None,
        batch_balanced: bool = False,
    ) -> Dict[str, Tensor]:
        targets = self.build_targets(rag, gt_labels) if targets is None else targets
        target = targets.target
        predictions = rag.spatial_edge_logits if logits is None else logits
        if predictions.shape != target.shape:
            raise ValueError("RAG logits must align one-to-one with RAG targets")
        valid = targets.valid
        if target.numel() == 0 or not valid.any():
            zero = rag.node_features.sum() * 0
            valid_fraction = valid.float().mean() if valid.numel() else zero.detach()
            mean_purity = (
                targets.node_purity.mean()
                if targets.node_purity.numel()
                else zero.detach()
            )
            impure = (
                (targets.node_purity < self.cfg.rag_min_node_purity).float().mean()
                if targets.node_purity.numel()
                else zero.detach()
            )
            mean_support = (
                targets.node_gt_support.mean()
                if targets.node_gt_support.numel()
                else zero.detach()
            )
            low_support = (
                (
                    targets.node_gt_support
                    < self.cfg.rag_min_node_gt_support
                ).float().mean()
                if targets.node_gt_support.numel()
                else zero.detach()
            )
            return {
                "rag_bce": zero,
                "rag_accuracy": zero.detach(),
                "rag_valid_edge_fraction": valid_fraction.detach(),
                "rag_mean_node_purity": mean_purity.detach(),
                "rag_impure_node_fraction": impure.detach(),
                "rag_mean_node_gt_support": mean_support.detach(),
                "rag_low_support_node_fraction": low_support.detach(),
            }
        selected_target = target[valid]
        selected_predictions = predictions[valid]
        if batch_balanced:
            # Each crop with valid RAG supervision contributes one scalar loss,
            # regardless of how many valid edges that crop generated.
            per_batch_losses: list[Tensor] = []
            for batch_index in torch.unique(rag.edge_batch[valid]):
                row_valid = valid & (rag.edge_batch == batch_index)
                row_target = target[row_valid]
                row_predictions = predictions[row_valid]
                positives = row_target.sum()
                negatives = row_target.numel() - positives
                pos_weight = (negatives / positives.clamp_min(1)).clamp(0.5, 20.0)
                per_batch_losses.append(F.binary_cross_entropy_with_logits(
                    row_predictions, row_target, pos_weight=pos_weight
                ))
            loss = torch.stack(per_batch_losses).mean()
        else:
            positives = selected_target.sum()
            negatives = selected_target.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1)).clamp(0.5, 20.0)
            loss = F.binary_cross_entropy_with_logits(
                selected_predictions, selected_target, pos_weight=pos_weight
            )
        accuracy = (
            (selected_predictions.sigmoid() >= 0.5) == selected_target.bool()
        ).float().mean()
        return {
            "rag_bce": loss,
            "rag_accuracy": accuracy.detach(),
            "rag_valid_edge_fraction": valid.float().mean().detach(),
            "rag_mean_node_purity": targets.node_purity.mean().detach(),
            "rag_impure_node_fraction": (
                targets.node_purity < self.cfg.rag_min_node_purity
            ).float().mean().detach(),
            "rag_mean_node_gt_support": targets.node_gt_support.mean().detach(),
            "rag_low_support_node_fraction": (
                targets.node_gt_support < self.cfg.rag_min_node_gt_support
            ).float().mean().detach(),
        }
