# STIRNET_MORPHOLOGY_AWARE_RAG_V2
from __future__ import annotations

"""Bounded 3-D morphology evidence for STIR-Net's spatial RAG.

Node evidence remains whole-supervoxel morphology.

Edge evidence is v2:
* every A<->B touching voxel is first located;
* the COMPLETE contact bounding box is retained;
* two joint A+B ROIs are extracted:
    - local:  50% contact-bbox headroom per side by default;
    - broad: 100% contact-bbox headroom per side by default;
* a minimum physical headroom in dref units prevents a one-voxel-thick contact
  from producing a nearly planar crop;
* A and B are separate topology channels in the SAME ROI;
* A/B ordering is made exactly symmetric by a shared member stem followed by
  sum and absolute-difference fusion;
* union/interface/proximity are encoded by a relation stem;
* topology directly conditions context features;
* global and interface-aware pooling preserve thin-neck information;
* local+broad embeddings are fused with physical contact/ROI metadata.

Dense geometry can be detached so RAG-only optimization cannot perturb the
known-good geometry network.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import PartitionConfig
from ..types import GeometryLike, RAGState, geometry_field_crop
from ..utils.tensor_ops import reduce_labeled_voxels


EDGE_METADATA_DIM = 16


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
    """Legacy-shape node morphology encoder."""

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


class EdgeMorphologyPatchEncoder(nn.Module):
    """Joint A/B edge encoder with strong, symmetric topology conditioning.

    topology channels:
        0: member A
        1: member B
        2: A union B
        3: visible A<->B interface
        4: wider interface-proximity support
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.appearance = _Stem3D(1, 8)
        self.scalar_geometry = _Stem3D(5, 16)
        self.vector_geometry = _Stem3D(6, 16)

        self.member = _Stem3D(1, 12)
        self.relation = _Stem3D(3, 16)

        self.context_projection = nn.Sequential(
            nn.Conv3d(40, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(48), 48),
            nn.SiLU(),
        )
        self.topology_projection = nn.Sequential(
            nn.Conv3d(40, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(48), 48),
            nn.SiLU(),
        )
        self.topology_gate = nn.Sequential(
            nn.Conv3d(48, 48, 1),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(48, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Linear(96, output_dim),
        )

    @staticmethod
    def _interface_pool(
        features: Tensor,
        proximity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        weights = F.interpolate(
            proximity.float(),
            size=tuple(int(v) for v in features.shape[-3:]),
            mode="trilinear",
            align_corners=False,
        ).to(dtype=features.dtype)
        weights = weights.clamp(0.0, 1.0)

        denominator = weights.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
        local_avg = (
            (features * weights).sum(dim=(-3, -2, -1))
            / denominator
        )

        mask = weights > 0.05
        expanded_mask = mask.expand_as(features)
        very_negative = torch.finfo(features.dtype).min
        local_max = features.masked_fill(
            ~expanded_mask,
            very_negative,
        ).amax(dim=(-3, -2, -1))

        has_support = mask.flatten(1).any(dim=1)
        global_max = features.amax(dim=(-3, -2, -1))
        local_max = torch.where(
            has_support[:, None],
            local_max,
            global_max,
        )
        return local_avg, local_max

    def forward(
        self,
        appearance: Tensor,
        scalar_geometry: Tensor,
        vector_geometry: Tensor,
        topology: Tensor,
    ) -> Tensor:
        if topology.shape[1] != 5:
            raise ValueError(
                "Edge topology must be [A, B, union, interface, proximity]"
            )

        member_a = self.member(topology[:, 0:1])
        member_b = self.member(topology[:, 1:2])
        member_sum = member_a + member_b
        member_absdiff = (member_a - member_b).abs()
        relation = self.relation(topology[:, 2:5])

        context = torch.cat(
            [
                self.appearance(appearance),
                self.scalar_geometry(scalar_geometry),
                self.vector_geometry(vector_geometry),
            ],
            dim=1,
        )
        topology_features = torch.cat(
            [member_sum, member_absdiff, relation],
            dim=1,
        )

        context = self.context_projection(context)
        topology_features = self.topology_projection(topology_features)

        gate = self.topology_gate(topology_features)
        fused = context * (1.0 + gate) + topology_features
        features = self.fusion(fused)

        global_avg = features.mean(dim=(-3, -2, -1))
        global_max = features.amax(dim=(-3, -2, -1))
        interface_avg, interface_max = self._interface_pool(
            features,
            topology[:, 4:5],
        )

        pooled = torch.cat(
            [global_avg, global_max, interface_avg, interface_max],
            dim=-1,
        )
        return self.head(pooled)


@dataclass(frozen=True)
class EdgeScalePatch:
    appearance: Tensor
    scalar_geometry: Tensor
    vector_geometry: Tensor
    topology: Tensor
    requested_start_zyx: Tensor
    requested_stop_zyx: Tensor
    roi_extent_um: Tensor


@dataclass(frozen=True)
class EdgePairPatch:
    broad: EdgeScalePatch
    local: EdgeScalePatch
    metadata: Tensor
    contact_lower_zyx: Tensor
    contact_upper_zyx: Tensor
    contact_face_counts_zyx: Tensor


class RAGMorphologyEmbeddingBuilder(nn.Module):
    """Build node and edge embeddings from bounded physical 3-D patches."""

    def __init__(self, cfg: PartitionConfig):
        super().__init__()
        self.cfg = cfg

        self.node_encoder = MorphologyPatchEncoder(
            topology_channels=1,
            output_dim=cfg.rag_node_morphology_dim,
        )
        self.edge_encoder = EdgeMorphologyPatchEncoder(
            output_dim=cfg.rag_edge_morphology_dim,
        )
        self.edge_scale_fusion = nn.Sequential(
            nn.Linear(
                2 * cfg.rag_edge_morphology_dim + EDGE_METADATA_DIM,
                128,
            ),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Linear(96, cfg.rag_edge_morphology_dim),
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
    def _resize_continuous(
        value: Tensor,
        shape: tuple[int, int, int],
    ) -> Tensor:
        return F.interpolate(
            value[None].float(),
            size=shape,
            mode="trilinear",
            align_corners=False,
        )[0]

    @staticmethod
    def _resize_nearest(
        value: Tensor,
        shape: tuple[int, int, int],
    ) -> Tensor:
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
        sdf = geometry_field_crop(
            geometry, "sdf", batch_index, crop
        )
        seed = geometry_field_crop(
            geometry, "seed_logits", batch_index, crop
        ).sigmoid()
        flow = geometry_field_crop(
            geometry, "flow", batch_index, crop
        )
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
                torch.cat(
                    [foreground, surface, separator, sdf, seed],
                    dim=0,
                ),
                padding,
            ),
            patch_shape,
        )
        vector = self._resize_continuous(
            self._pad(
                torch.cat([flow, offset], dim=0),
                padding,
            ),
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
        extent_um = (
            upper.float() - lower.float() + 1.0
        ) * spacing_um.float()
        radius_um = (
            0.5 * float(extent_um.max().detach().cpu())
            + self.cfg.rag_node_context_dref
            * float(dref_um.detach().cpu())
        )
        crop, padding = self._physical_cube(
            center,
            radius_um,
            spacing_um,
            tuple(int(v) for v in labels.shape),
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
            self._pad(topology, padding),
            self.cfg.rag_node_patch_shape_zyx,
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
    def _contact_voxel_bounds(
        labels: Tensor,
        label_a: int,
        label_b: int,
        bbox: tuple[slice, slice, slice],
        fallback_center: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Inclusive global bounds of ALL A/B touching endpoint voxels."""
        local = labels[bbox]
        origin = torch.tensor(
            [int(axis.start) for axis in bbox],
            device=labels.device,
            dtype=torch.long,
        )

        points: list[Tensor] = []
        face_counts = torch.zeros(
            (3,),
            device=labels.device,
            dtype=torch.float32,
        )

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
            if not bool(touch.any()):
                continue

            coordinates = torch.nonzero(
                touch,
                as_tuple=False,
            ).long()
            first = coordinates + origin
            second = first.clone()
            second[:, axis] += 1
            points.extend([first, second])
            face_counts[axis] = float(coordinates.shape[0])

        if not points:
            center = fallback_center.round().long()
            maximum = torch.as_tensor(
                labels.shape,
                device=labels.device,
                dtype=torch.long,
            ) - 1
            center = torch.minimum(
                center.clamp_min(0),
                maximum,
            )
            return center, center, face_counts

        point_cloud = torch.cat(points, dim=0)
        return (
            point_cloud.amin(dim=0),
            point_cloud.amax(dim=0),
            face_counts,
        )

    @classmethod
    def _contact_roi(
        cls,
        contact_lower: Tensor,
        contact_upper: Tensor,
        *,
        spacing_um: Tensor,
        dref_um: Tensor,
        shape: tuple[int, int, int],
        headroom_fraction: float,
        minimum_headroom_dref: float,
    ):
        spacing = spacing_um.float()
        contact_extent_vox = (
            contact_upper.long()
            - contact_lower.long()
            + 1
        ).clamp_min(1)
        contact_extent_um = contact_extent_vox.float() * spacing

        minimum_um = (
            float(minimum_headroom_dref)
            * float(dref_um.detach().cpu())
        )
        margin_um = torch.maximum(
            contact_extent_um * float(headroom_fraction),
            torch.full_like(contact_extent_um, minimum_um),
        )
        margin_vox = torch.ceil(
            margin_um / spacing.clamp_min(1e-6)
        ).long()

        requested_start = contact_lower.long() - margin_vox
        requested_stop = contact_upper.long() + margin_vox + 1
        roi_extent_um = (
            requested_stop - requested_start
        ).float() * spacing

        crop, padding = cls._padding_from_bounds(
            tuple(
                int(v)
                for v in requested_start.detach().cpu().tolist()
            ),
            tuple(
                int(v)
                for v in requested_stop.detach().cpu().tolist()
            ),
            shape,
        )
        return (
            crop,
            padding,
            requested_start,
            requested_stop,
            roi_extent_um,
        )

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
        mask_a = local == label_a
        mask_b = local == label_b
        union = mask_a | mask_b
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
            if bool(touch.any()):
                interface[tuple(lower_slice)] |= touch
                interface[tuple(upper_slice)] |= touch

        base = torch.stack(
            [
                mask_a.float(),
                mask_b.float(),
                union.float(),
                interface.float(),
            ],
            dim=0,
        )
        base = RAGMorphologyEmbeddingBuilder._pad(base, padding)
        base = RAGMorphologyEmbeddingBuilder._resize_nearest(
            base,
            patch_shape,
        )

        visible_interface = F.max_pool3d(
            base[3:4][None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[0]
        proximity = F.max_pool3d(
            visible_interface[None],
            kernel_size=5,
            stride=1,
            padding=2,
        )[0]

        return torch.cat(
            [
                base[0:3],
                visible_interface,
                proximity,
            ],
            dim=0,
        )

    def _edge_scale_patch(
        self,
        *,
        labels: Tensor,
        label_a: int,
        label_b: int,
        contact_lower: Tensor,
        contact_upper: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        batch_index: int,
        spacing_um: Tensor,
        dref_um: Tensor,
        headroom_fraction: float,
    ) -> EdgeScalePatch:
        shape = tuple(int(v) for v in labels.shape)
        (
            crop,
            padding,
            requested_start,
            requested_stop,
            roi_extent_um,
        ) = self._contact_roi(
            contact_lower,
            contact_upper,
            spacing_um=spacing_um,
            dref_um=dref_um,
            shape=shape,
            headroom_fraction=headroom_fraction,
            minimum_headroom_dref=(
                self.cfg.rag_edge_min_headroom_dref
            ),
        )

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
        return EdgeScalePatch(
            appearance=appearance,
            scalar_geometry=scalar,
            vector_geometry=vector,
            topology=topology,
            requested_start_zyx=requested_start,
            requested_stop_zyx=requested_stop,
            roi_extent_um=roi_extent_um,
        )

    @staticmethod
    def _edge_metadata(
        *,
        contact_lower: Tensor,
        contact_upper: Tensor,
        contact_face_counts: Tensor,
        broad: EdgeScalePatch,
        local: EdgeScalePatch,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> Tensor:
        dref = dref_um.float().reshape(()).clamp_min(1e-6)
        spacing = spacing_um.float()

        contact_extent_um = (
            contact_upper.float()
            - contact_lower.float()
            + 1.0
        ) * spacing

        face_areas = torch.stack(
            [
                spacing[1] * spacing[2],
                spacing[0] * spacing[2],
                spacing[0] * spacing[1],
            ]
        )
        contact_area_um2 = (
            contact_face_counts.float()
            * face_areas
        ).sum()

        broad_topology = broad.topology.float()
        local_topology = local.topology.float()

        broad_a = broad_topology[0].mean()
        broad_b = broad_topology[1].mean()
        local_a = local_topology[0].mean()
        local_b = local_topology[1].mean()

        metadata = torch.cat(
            [
                contact_extent_um / dref,
                broad.roi_extent_um.float() / dref,
                local.roi_extent_um.float() / dref,
                torch.log1p(
                    contact_area_um2 / (dref * dref)
                )[None],
                broad_topology[2].mean()[None],
                local_topology[2].mean()[None],
                (broad_a - broad_b).abs()[None],
                (local_a - local_b).abs()[None],
                broad_topology[4].mean()[None],
                local_topology[4].mean()[None],
            ],
            dim=0,
        )

        if metadata.numel() != EDGE_METADATA_DIM:
            raise RuntimeError(
                f"Expected {EDGE_METADATA_DIM} edge metadata "
                f"values, got {metadata.numel()}"
            )
        return metadata

    def _edge_pair_patch(
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
    ) -> EdgePairPatch:
        label_a = local_a + 1
        label_b = local_b + 1
        shape = tuple(int(v) for v in labels.shape)

        pair_bbox = self._union_bbox(
            lower[local_a],
            upper[local_a],
            lower[local_b],
            upper[local_b],
            shape,
        )
        fallback = 0.25 * (
            lower[local_a].float()
            + upper[local_a].float()
            + lower[local_b].float()
            + upper[local_b].float()
        )
        (
            contact_lower,
            contact_upper,
            face_counts,
        ) = self._contact_voxel_bounds(
            labels,
            label_a,
            label_b,
            pair_bbox,
            fallback,
        )

        broad = self._edge_scale_patch(
            labels=labels,
            label_a=label_a,
            label_b=label_b,
            contact_lower=contact_lower,
            contact_upper=contact_upper,
            spatial_inputs=spatial_inputs,
            geometry=geometry,
            batch_index=batch_index,
            spacing_um=spacing_um,
            dref_um=dref_um,
            headroom_fraction=(
                self.cfg.rag_edge_contact_headroom_fraction
            ),
        )
        local = self._edge_scale_patch(
            labels=labels,
            label_a=label_a,
            label_b=label_b,
            contact_lower=contact_lower,
            contact_upper=contact_upper,
            spatial_inputs=spatial_inputs,
            geometry=geometry,
            batch_index=batch_index,
            spacing_um=spacing_um,
            dref_um=dref_um,
            headroom_fraction=(
                self.cfg.rag_edge_local_headroom_fraction
            ),
        )

        metadata = self._edge_metadata(
            contact_lower=contact_lower,
            contact_upper=contact_upper,
            contact_face_counts=face_counts,
            broad=broad,
            local=local,
            spacing_um=spacing_um,
            dref_um=dref_um,
        )
        return EdgePairPatch(
            broad=broad,
            local=local,
            metadata=metadata,
            contact_lower_zyx=contact_lower,
            contact_upper_zyx=contact_upper,
            contact_face_counts_zyx=face_counts,
        )

    @staticmethod
    def _stack_node_rows(rows):
        return tuple(
            torch.stack(
                [row[column] for row in rows],
                dim=0,
            )
            for column in range(4)
        )

    @staticmethod
    def _stack_edge_scale(
        rows: list[EdgeScalePatch],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.stack(
                [row.appearance for row in rows],
                dim=0,
            ),
            torch.stack(
                [row.scalar_geometry for row in rows],
                dim=0,
            ),
            torch.stack(
                [row.vector_geometry for row in rows],
                dim=0,
            ),
            torch.stack(
                [row.topology for row in rows],
                dim=0,
            ),
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
            return spatial_inputs.new_zeros(
                (0, self.cfg.rag_node_morphology_dim)
            )
        lower, upper = self._node_bounds(
            rag,
            batch_index,
            labels,
            spacing_um,
        )
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
                for row in range(
                    start,
                    min(start + chunk, n),
                )
            ]
            outputs.append(
                self.node_encoder(
                    *self._stack_node_rows(rows)
                )
            )
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
            return spatial_inputs.new_zeros(
                (0, self.cfg.rag_edge_morphology_dim)
            )

        lower, upper = self._node_bounds(
            rag,
            batch_index,
            labels,
            spacing_um,
        )
        outputs = []
        chunk = self.cfg.rag_morphology_chunk_size

        for start in range(0, count, chunk):
            pair_rows = [
                self._edge_pair_patch(
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
                for row in range(
                    start,
                    min(start + chunk, count),
                )
            ]

            broad = self.edge_encoder(
                *self._stack_edge_scale(
                    [row.broad for row in pair_rows]
                )
            )
            local = self.edge_encoder(
                *self._stack_edge_scale(
                    [row.local for row in pair_rows]
                )
            )
            metadata = torch.stack(
                [row.metadata for row in pair_rows],
                dim=0,
            ).to(
                device=broad.device,
                dtype=broad.dtype,
            )

            outputs.append(
                self.edge_scale_fusion(
                    torch.cat(
                        [broad, local, metadata],
                        dim=-1,
                    )
                )
            )

        return torch.cat(outputs, dim=0)

    def edge_debug_patch(
        self,
        rag: RAGState,
        edge_row: int,
        spatial_inputs: Tensor,
        geometry: GeometryLike,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> EdgePairPatch:
        """Return exact production local/broad CNN inputs for one RAG edge."""
        edge_row = int(edge_row)
        if not 0 <= edge_row < rag.edge_index.shape[1]:
            raise IndexError(
                f"edge_row {edge_row} outside "
                f"[0, {rag.edge_index.shape[1]})"
            )

        batch_index = int(
            rag.edge_batch[edge_row].item()
        )
        start = int(
            rag.node_offsets[batch_index].item()
        )
        labels = rag.supervoxel_labels[batch_index]
        lower, upper = self._node_bounds(
            rag,
            batch_index,
            labels,
            spacing_um[batch_index],
        )

        local_a = (
            int(rag.edge_index[0, edge_row].item())
            - start
        )
        local_b = (
            int(rag.edge_index[1, edge_row].item())
            - start
        )

        return self._edge_pair_patch(
            labels,
            local_a,
            local_b,
            lower,
            upper,
            spatial_inputs,
            geometry,
            batch_index,
            spacing_um[batch_index],
            dref_um[batch_index],
        )

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

        for batch_index, labels in enumerate(
            rag.supervoxel_labels
        ):
            start = int(
                rag.node_offsets[batch_index].item()
            )
            stop = int(
                rag.node_offsets[batch_index + 1].item()
            )

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
                rag.edge_batch == batch_index,
                as_tuple=False,
            ).flatten()
            if edge_rows.numel():
                local_edges = (
                    rag.edge_index[:, edge_rows]
                    - start
                )
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
            else spatial_inputs.new_zeros(
                (0, self.cfg.rag_node_morphology_dim)
            )
        )
        edge_embedding = (
            torch.cat(edge_outputs, dim=0)
            if edge_outputs
            else spatial_inputs.new_zeros(
                (0, self.cfg.rag_edge_morphology_dim)
            )
        )

        if node_embedding.shape[0] != rag.node_features.shape[0]:
            raise RuntimeError(
                "Node morphology rows do not align with RAG nodes: "
                f"{node_embedding.shape[0]} "
                f"!= {rag.node_features.shape[0]}"
            )
        if edge_embedding.shape[0] != rag.edge_features.shape[0]:
            raise RuntimeError(
                "Edge morphology rows do not align with RAG edges: "
                f"{edge_embedding.shape[0]} "
                f"!= {rag.edge_features.shape[0]}"
            )

        return node_embedding, edge_embedding


__all__ = [
    "EDGE_METADATA_DIM",
    "EdgeMorphologyPatchEncoder",
    "EdgePairPatch",
    "EdgeScalePatch",
    "MorphologyPatchEncoder",
    "RAGMorphologyEmbeddingBuilder",
]
