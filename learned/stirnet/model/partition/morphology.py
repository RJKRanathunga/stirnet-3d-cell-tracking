# STIRNET_MORPHOLOGY_AWARE_RAG_V1
from __future__ import annotations

"""Bounded 3-D morphology evidence for STIR-Net's spatial RAG.

The legacy RAG statistics are retained. This module adds complementary spatial
information:

* node embedding: complete supervoxel morphology with aspect ratio preserved;
* edge embedding: a physical cube centered on the exact A<->B interface;
* inputs: raw appearance, learned scalar geometry, learned vector geometry,
  and explicit topology masks.

Patches are processed in bounded chunks. Dense geometry can be detached so a
RAG-only fine-tuning stage cannot perturb the known-good geometry network.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import PartitionConfig
from ..types import GeometryLike, RAGState, geometry_field_crop
from ..utils.tensor_ops import reduce_labeled_voxels


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Stem3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )


class MorphologyPatchEncoder(nn.Module):
    """Modality-specific shallow stems followed by joint 3-D reasoning."""

    def __init__(self, topology_channels: int, output_dim: int):
        super().__init__()
        self.appearance = _Stem3D(1, 8)
        self.scalar_geometry = _Stem3D(5, 16)
        self.vector_geometry = _Stem3D(6, 16)
        self.topology = _Stem3D(topology_channels, 8)
        self.fusion = nn.Sequential(
            nn.Conv3d(48, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(48), 48),
            nn.SiLU(),
            nn.Conv3d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Linear(96, output_dim),
        )

    def forward(
        self,
        appearance: Tensor,
        scalar_geometry: Tensor,
        vector_geometry: Tensor,
        topology: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [
                self.appearance(appearance),
                self.scalar_geometry(scalar_geometry),
                self.vector_geometry(vector_geometry),
                self.topology(topology),
            ],
            dim=1,
        )
        features = self.fusion(features)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool3d(features, 1),
                F.adaptive_max_pool3d(features, 1),
            ],
            dim=1,
        ).flatten(1)
        return self.head(pooled)


class RAGMorphologyEmbeddingBuilder(nn.Module):
    """Build node/edge embeddings from bounded physical 3-D patches."""

    def __init__(self, cfg: PartitionConfig):
        super().__init__()
        self.cfg = cfg
        self.node_encoder = MorphologyPatchEncoder(
            topology_channels=1,
            output_dim=cfg.rag_node_morphology_dim,
        )
        self.edge_encoder = MorphologyPatchEncoder(
            topology_channels=2,
            output_dim=cfg.rag_edge_morphology_dim,
        )

    @staticmethod
    def _padding_from_bounds(
        start: tuple[int, int, int],
        stop: tuple[int, int, int],
        shape: tuple[int, int, int],
    ) -> tuple[
        tuple[slice, slice, slice],
        tuple[int, int, int, int, int, int],
    ]:
        clipped_start = tuple(max(0, value) for value in start)
        clipped_stop = tuple(min(shape[i], stop[i]) for i in range(3))
        slices = tuple(
            slice(clipped_start[i], clipped_stop[i]) for i in range(3)
        )
        pad_low = tuple(clipped_start[i] - start[i] for i in range(3))
        pad_high = tuple(stop[i] - clipped_stop[i] for i in range(3))
        padding = (
            pad_low[2],
            pad_high[2],
            pad_low[1],
            pad_high[1],
            pad_low[0],
            pad_high[0],
        )
        return slices, padding

    @classmethod
    def _physical_cube(
        cls,
        center_voxel: Tensor,
        radius_um: float,
        spacing_um: Tensor,
        shape: tuple[int, int, int],
    ):
        center = center_voxel.detach().float().cpu().tolist()
        spacing = spacing_um.detach().float().cpu().tolist()
        radius_um = max(float(radius_um), 1e-6)
        start = []
        stop = []
        for axis in range(3):
            half = radius_um / max(float(spacing[axis]), 1e-6)
            lo = math.floor(float(center[axis]) - half)
            hi = math.ceil(float(center[axis]) + half) + 1
            if hi <= lo:
                hi = lo + 1
            start.append(lo)
            stop.append(hi)
        return cls._padding_from_bounds(tuple(start), tuple(stop), shape)

    @staticmethod
    def _pad(value: Tensor, padding) -> Tensor:
        if any(int(v) for v in padding):
            value = F.pad(value, padding, mode="constant", value=0.0)
        return value

    @staticmethod
    def _resize_continuous(value: Tensor, shape: tuple[int, int, int]) -> Tensor:
        return F.interpolate(
            value[None].float(),
            size=shape,
            mode="trilinear",
            align_corners=False,
        )[0]

    @staticmethod
    def _resize_nearest(value: Tensor, shape: tuple[int, int, int]) -> Tensor:
        return F.interpolate(
            value[None].float(),
            size=shape,
            mode="nearest",
        )[0]

    def _modalities(
        self,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        batch_index: int,
        crop: tuple[slice, slice, slice],
        padding,
        patch_shape: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        appearance = spatial_inputs[
            batch_index, :1, crop[0], crop[1], crop[2]
        ]
        foreground = geometry_field_crop(
            geometry, "foreground_logits", batch_index, crop
        ).sigmoid()
        surface = geometry_field_crop(
            geometry, "surface_logits", batch_index, crop
        ).sigmoid()
        separator = geometry_field_crop(
            geometry, "separator_logits", batch_index, crop
        ).sigmoid()
        sdf = geometry_field_crop(geometry, "sdf", batch_index, crop)
        seed = geometry_field_crop(
            geometry, "seed_logits", batch_index, crop
        ).sigmoid()
        flow = geometry_field_crop(geometry, "flow", batch_index, crop)
        offset = geometry_field_crop(
            geometry, "centroid_offset", batch_index, crop
        )

        if self.cfg.rag_morphology_detach_geometry:
            appearance = appearance.detach()
            foreground = foreground.detach()
            surface = surface.detach()
            separator = separator.detach()
            sdf = sdf.detach()
            seed = seed.detach()
            flow = flow.detach()
            offset = offset.detach()

        appearance = self._resize_continuous(
            self._pad(appearance, padding), patch_shape
        )
        scalar = self._resize_continuous(
            self._pad(
                torch.cat([foreground, surface, separator, sdf, seed], dim=0),
                padding,
            ),
            patch_shape,
        )
        vector = self._resize_continuous(
            self._pad(torch.cat([flow, offset], dim=0), padding),
            patch_shape,
        )
        return appearance, scalar, vector

    @staticmethod
    def _node_bounds(
        rag: RAGState,
        batch_index: int,
        labels: Tensor,
        spacing_um: Tensor,
    ) -> tuple[Tensor, Tensor]:
        stats = None if rag.statistics is None else rag.statistics[batch_index]
        n = int(labels.max().item())
        if (
            stats is not None
            and stats.min_voxel.shape[0] >= n
            and stats.max_voxel.shape[0] >= n
        ):
            return stats.min_voxel[:n], stats.max_voxel[:n]
        reduced = reduce_labeled_voxels(
            labels,
            spacing_um,
            need_nearest_centroid=False,
        )
        return reduced.min_voxel[:n], reduced.max_voxel[:n]

    def _node_patch(
        self,
        labels: Tensor,
        node_id: int,
        lower: Tensor,
        upper: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        batch_index: int,
        spacing_um: Tensor,
        dref_um: Tensor,
    ):
        center = 0.5 * (lower.float() + upper.float())
        extent_um = (upper.float() - lower.float() + 1.0) * spacing_um.float()
        radius_um = (
            0.5 * float(extent_um.max().detach().cpu())
            + self.cfg.rag_node_context_dref * float(dref_um.detach().cpu())
        )
        crop, padding = self._physical_cube(
            center, radius_um, spacing_um, tuple(int(v) for v in labels.shape)
        )
        appearance, scalar, vector = self._modalities(
            spatial_inputs,
            geometry,
            batch_index,
            crop,
            padding,
            self.cfg.rag_node_patch_shape_zyx,
        )
        topology = (labels[crop] == node_id).float()[None]
        topology = self._resize_nearest(
            self._pad(topology, padding), self.cfg.rag_node_patch_shape_zyx
        )
        return appearance, scalar, vector, topology

    @staticmethod
    def _union_bbox(
        lower_a: Tensor,
        upper_a: Tensor,
        lower_b: Tensor,
        upper_b: Tensor,
        shape: tuple[int, int, int],
        halo: int = 1,
    ) -> tuple[slice, slice, slice]:
        lower = torch.minimum(lower_a, lower_b).long() - int(halo)
        upper = torch.maximum(upper_a, upper_b).long() + int(halo) + 1
        lower = lower.clamp_min(0)
        upper = torch.minimum(
            upper,
            torch.as_tensor(shape, device=upper.device),
        )
        return tuple(
            slice(int(lo), int(hi))
            for lo, hi in zip(lower.tolist(), upper.tolist())
        )

    @staticmethod
    def _interface_center(
        labels: Tensor,
        label_a: int,
        label_b: int,
        bbox: tuple[slice, slice, slice],
        fallback: Tensor,
    ) -> Tensor:
        local = labels[bbox]
        origin = torch.tensor(
            [int(axis.start) for axis in bbox],
            device=labels.device,
            dtype=torch.float32,
        )
        points = []
        for axis in range(3):
            lower_slice = [slice(None)] * 3
            upper_slice = [slice(None)] * 3
            lower_slice[axis] = slice(0, -1)
            upper_slice[axis] = slice(1, None)
            lower = local[tuple(lower_slice)]
            upper = local[tuple(upper_slice)]
            touch = (
                ((lower == label_a) & (upper == label_b))
                | ((lower == label_b) & (upper == label_a))
            )
            if not touch.any():
                continue
            face = torch.nonzero(touch, as_tuple=False).float()
            face[:, axis] += 0.5
            points.append(face + origin)
        if not points:
            return fallback.float()
        return torch.cat(points, dim=0).mean(dim=0)

    @staticmethod
    def _edge_topology(
        labels: Tensor,
        crop: tuple[slice, slice, slice],
        padding,
        label_a: int,
        label_b: int,
        patch_shape: tuple[int, int, int],
    ) -> Tensor:
        local = labels[crop]
        union = ((local == label_a) | (local == label_b)).float()
        interface = torch.zeros_like(union, dtype=torch.bool)
        for axis in range(3):
            lower_slice = [slice(None)] * 3
            upper_slice = [slice(None)] * 3
            lower_slice[axis] = slice(0, -1)
            upper_slice[axis] = slice(1, None)
            lower = local[tuple(lower_slice)]
            upper = local[tuple(upper_slice)]
            touch = (
                ((lower == label_a) & (upper == label_b))
                | ((lower == label_b) & (upper == label_a))
            )
            if touch.any():
                interface[tuple(lower_slice)] |= touch
                interface[tuple(upper_slice)] |= touch

        stacked = torch.stack([union, interface.float()], dim=0)
        stacked = RAGMorphologyEmbeddingBuilder._pad(stacked, padding)
        stacked = RAGMorphologyEmbeddingBuilder._resize_nearest(
            stacked, patch_shape
        )
        stacked[1:2] = F.max_pool3d(
            stacked[1:2][None], kernel_size=3, stride=1, padding=1
        )[0]
        return stacked

    def _edge_patch(
        self,
        labels: Tensor,
        local_a: int,
        local_b: int,
        lower: Tensor,
        upper: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        batch_index: int,
        spacing_um: Tensor,
        dref_um: Tensor,
    ):
        label_a = local_a + 1
        label_b = local_b + 1
        shape = tuple(int(v) for v in labels.shape)
        pair_bbox = self._union_bbox(
            lower[local_a], upper[local_a], lower[local_b], upper[local_b], shape
        )
        fallback = 0.25 * (
            lower[local_a].float()
            + upper[local_a].float()
            + lower[local_b].float()
            + upper[local_b].float()
        )
        center = self._interface_center(
            labels, label_a, label_b, pair_bbox, fallback
        )
        radius_um = self.cfg.rag_edge_radius_dref * float(dref_um.detach().cpu())
        crop, padding = self._physical_cube(center, radius_um, spacing_um, shape)
        appearance, scalar, vector = self._modalities(
            spatial_inputs,
            geometry,
            batch_index,
            crop,
            padding,
            self.cfg.rag_edge_patch_shape_zyx,
        )
        topology = self._edge_topology(
            labels,
            crop,
            padding,
            label_a,
            label_b,
            self.cfg.rag_edge_patch_shape_zyx,
        )
        return appearance, scalar, vector, topology

    @staticmethod
    def _stack_rows(rows):
        return tuple(
            torch.stack([row[column] for row in rows], dim=0)
            for column in range(4)
        )

    def _encode_nodes(
        self,
        rag: RAGState,
        batch_index: int,
        labels: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> Tensor:
        n = int(labels.max().item())
        if n == 0:
            return spatial_inputs.new_zeros((0, self.cfg.rag_node_morphology_dim))
        lower, upper = self._node_bounds(rag, batch_index, labels, spacing_um)
        outputs = []
        chunk = self.cfg.rag_morphology_chunk_size
        for start in range(0, n, chunk):
            rows = [
                self._node_patch(
                    labels,
                    row + 1,
                    lower[row],
                    upper[row],
                    spatial_inputs,
                    geometry,
                    batch_index,
                    spacing_um,
                    dref_um,
                )
                for row in range(start, min(start + chunk, n))
            ]
            outputs.append(self.node_encoder(*self._stack_rows(rows)))
        return torch.cat(outputs, dim=0)

    def _encode_edges(
        self,
        rag: RAGState,
        batch_index: int,
        labels: Tensor,
        local_edges: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> Tensor:
        count = int(local_edges.shape[1])
        if count == 0:
            return spatial_inputs.new_zeros((0, self.cfg.rag_edge_morphology_dim))
        lower, upper = self._node_bounds(rag, batch_index, labels, spacing_um)
        outputs = []
        chunk = self.cfg.rag_morphology_chunk_size
        for start in range(0, count, chunk):
            rows = [
                self._edge_patch(
                    labels,
                    int(local_edges[0, row].item()),
                    int(local_edges[1, row].item()),
                    lower,
                    upper,
                    spatial_inputs,
                    geometry,
                    batch_index,
                    spacing_um,
                    dref_um,
                )
                for row in range(start, min(start + chunk, count))
            ]
            outputs.append(self.edge_encoder(*self._stack_rows(rows)))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        rag: RAGState,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> tuple[Tensor, Tensor]:
        node_outputs = []
        edge_outputs = []
        for batch_index, labels in enumerate(rag.supervoxel_labels):
            start = int(rag.node_offsets[batch_index].item())
            stop = int(rag.node_offsets[batch_index + 1].item())
            if stop > start:
                node_outputs.append(
                    self._encode_nodes(
                        rag,
                        batch_index,
                        labels,
                        spatial_inputs,
                        geometry,
                        spacing_um[batch_index],
                        dref_um[batch_index],
                    )
                )

            edge_rows = torch.nonzero(
                rag.edge_batch == batch_index, as_tuple=False
            ).flatten()
            if edge_rows.numel():
                local_edges = rag.edge_index[:, edge_rows] - start
                edge_outputs.append(
                    self._encode_edges(
                        rag,
                        batch_index,
                        labels,
                        local_edges,
                        spatial_inputs,
                        geometry,
                        spacing_um[batch_index],
                        dref_um[batch_index],
                    )
                )

        node_embedding = (
            torch.cat(node_outputs, dim=0)
            if node_outputs
            else spatial_inputs.new_zeros((0, self.cfg.rag_node_morphology_dim))
        )
        edge_embedding = (
            torch.cat(edge_outputs, dim=0)
            if edge_outputs
            else spatial_inputs.new_zeros((0, self.cfg.rag_edge_morphology_dim))
        )
        if node_embedding.shape[0] != rag.node_features.shape[0]:
            raise RuntimeError(
                "Node morphology rows do not align with RAG nodes: "
                f"{node_embedding.shape[0]} != {rag.node_features.shape[0]}"
            )
        if edge_embedding.shape[0] != rag.edge_features.shape[0]:
            raise RuntimeError(
                "Edge morphology rows do not align with RAG edges: "
                f"{edge_embedding.shape[0]} != {rag.edge_features.shape[0]}"
            )
        return node_embedding, edge_embedding


__all__ = ["MorphologyPatchEncoder", "RAGMorphologyEmbeddingBuilder"]
