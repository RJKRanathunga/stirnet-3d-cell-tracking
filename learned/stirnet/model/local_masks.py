from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import LocalMaskConfig


@dataclass
class PhysicalLocalCrop:
    """Geometry and prediction for one anchor-local native-resolution crop."""

    slices: tuple[slice, slice, slice]
    support: Tensor
    relative_xyz_dref: Tensor
    logits: Tensor | None = None


def physical_local_crop(
    spatial_shape: tuple[int, int, int],
    spacing_um: Tensor,
    anchor_cellscale: Tensor,
    dref_um: Tensor,
    support_radius_dref: float,
) -> PhysicalLocalCrop:
    """Build a clipped native bbox and exact physical sphere around an anchor."""
    if len(spatial_shape) != 3 or any(int(size) <= 0 for size in spatial_shape):
        raise ValueError(f"spatial_shape must contain three positive sizes: {spatial_shape}")
    if support_radius_dref <= 0:
        raise ValueError("local support radius must be positive")
    device = anchor_cellscale.device
    spacing = spacing_um.to(device=device, dtype=torch.float32)
    anchor_um = anchor_cellscale.to(dtype=torch.float32) * dref_um.to(
        device=device, dtype=torch.float32
    )
    shape_tensor = torch.tensor(spatial_shape, device=device, dtype=torch.long)
    extent_um = (shape_tensor.float() - 1) * spacing
    anchor_voxel = (anchor_um + 0.5 * extent_um) / spacing.clamp_min(1e-8)
    radius_um = float(support_radius_dref) * dref_um.to(
        device=device, dtype=torch.float32
    )
    radius_voxels = torch.ceil(radius_um / spacing.clamp_min(1e-8)).long()
    center_voxel = torch.round(anchor_voxel).long()
    lower = torch.maximum(center_voxel - radius_voxels, torch.zeros_like(center_voxel))
    upper = torch.minimum(
        center_voxel + radius_voxels + 1, shape_tensor
    )
    if bool((upper <= lower).any()):
        raise ValueError("proposal anchor has no local crop intersection with the volume")
    slices = tuple(
        slice(int(start.item()), int(stop.item()))
        for start, stop in zip(lower, upper)
    )
    axes_um = [
        torch.arange(
            int(start.item()),
            int(stop.item()),
            device=device,
            dtype=torch.float32,
        )
        * spacing[axis]
        - 0.5 * extent_um[axis]
        for axis, (start, stop) in enumerate(zip(lower, upper))
    ]
    coords_um = torch.stack(
        torch.meshgrid(*axes_um, indexing="ij"), dim=0
    )
    delta_um = coords_um - anchor_um[:, None, None, None]
    support = delta_um.square().sum(dim=0) <= radius_um.square()
    relative_xyz_dref = delta_um / dref_um.to(
        device=device, dtype=torch.float32
    ).clamp_min(1e-8)
    return PhysicalLocalCrop(slices, support, relative_xyz_dref)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class LocalNativeMaskDecoder(nn.Module):
    """Small query-conditioned native 3-D decoder for spatial proposals."""

    def __init__(
        self,
        cfg: LocalMaskConfig,
        *,
        d0_channels: int,
        spatial_input_channels: int,
        query_dim: int,
        background_logit: float,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.background_logit = float(background_logit)
        input_channels = d0_channels + spatial_input_channels + 3 + 3
        hidden = int(cfg.hidden_channels)
        query_channels = int(cfg.query_channels)
        groups = _group_count(hidden)
        self.input_channels = input_channels
        self.spatial = nn.Sequential(
            nn.Conv3d(input_channels, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.query = nn.Sequential(
            nn.Linear(query_dim, query_channels),
            nn.SiLU(),
            nn.Linear(query_channels, query_channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(hidden + query_channels, hidden, 1),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, local_spatial: Tensor, query_embedding: Tensor) -> Tensor:
        """Decode ``[N,C,Z,Y,X]`` evidence into ``[N,Z,Y,X]`` logits."""
        if local_spatial.ndim != 5 or local_spatial.shape[1] != self.input_channels:
            raise ValueError(
                "local_spatial must have shape [N,"
                f"{self.input_channels},Z,Y,X], got {tuple(local_spatial.shape)}"
            )
        feature = self.spatial(local_spatial)
        query_feature = self.query(query_embedding)[..., None, None, None]
        query_feature = query_feature.expand(
            -1, -1, *feature.shape[-3:]
        )
        return self.fuse(torch.cat([feature, query_feature], dim=1))[:, 0]

    def decode_one(
        self,
        d0_features: Tensor,
        spatial_inputs: Tensor,
        dense_outputs: dict[str, Tensor],
        query_embedding: Tensor,
        anchor_cellscale: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        batch_index: int,
    ) -> PhysicalLocalCrop:
        """Decode one native crop; no query-by-whole-volume tensor is built."""
        geometry = physical_local_crop(
            tuple(int(value) for value in d0_features.shape[-3:]),
            spacing_um,
            anchor_cellscale,
            dref_um,
            self.cfg.support_radius_dref,
        )
        zyx = geometry.slices
        d0_crop = d0_features[batch_index, :, zyx[0], zyx[1], zyx[2]]
        input_crop = spatial_inputs[batch_index, :, zyx[0], zyx[1], zyx[2]].to(
            dtype=d0_crop.dtype
        )
        probability_crops = []
        for key in (
            "foreground_logits",
            "center_heatmap_logits",
            "boundary_logits",
        ):
            probability = dense_outputs[key][batch_index, :, zyx[0], zyx[1], zyx[2]].sigmoid()
            if self.cfg.detach_dense_evidence:
                probability = probability.detach()
            probability_crops.append(probability.to(dtype=d0_crop.dtype))
        local_spatial = torch.cat(
            [
                d0_crop,
                input_crop,
                *probability_crops,
                geometry.relative_xyz_dref.to(dtype=d0_crop.dtype),
            ],
            dim=0,
        )[None]
        logits = self(
            local_spatial, query_embedding[None].to(dtype=d0_crop.dtype)
        )[0]
        logits = logits.masked_fill(
            ~geometry.support, self.background_logit
        )
        geometry.logits = logits
        return geometry

    def decode_requests(
        self,
        d0_features: Tensor,
        spatial_inputs: Tensor,
        dense_outputs: dict[str, Tensor],
        query_embeddings: Tensor,
        anchors_cellscale: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        requests: list[tuple[int, int]],
    ) -> list[PhysicalLocalCrop]:
        """Decode sparse requests in configurable, initially singleton chunks."""
        predictions: list[PhysicalLocalCrop] = []
        chunk_size = max(1, int(self.cfg.query_chunk_size))
        for chunk_start in range(0, len(requests), chunk_size):
            for batch_index, query_index in requests[
                chunk_start : chunk_start + chunk_size
            ]:
                predictions.append(
                    self.decode_one(
                        d0_features,
                        spatial_inputs,
                        dense_outputs,
                        query_embeddings[batch_index, query_index],
                        anchors_cellscale[batch_index, query_index],
                        spacing_um[batch_index],
                        dref_um[batch_index],
                        batch_index=batch_index,
                    )
                )
        return predictions
