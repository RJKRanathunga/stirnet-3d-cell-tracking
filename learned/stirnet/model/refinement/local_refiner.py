from __future__ import annotations

from dataclasses import replace
from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import RefinementConfig, SpatialConfig
from ..spatial.blocks import groups_for
from ..types import GeometryState, RefinementRequest, RefinementState
from ..utils.physical import physical_crop_slices


class LocalGeometryRefiner(nn.Module):
    """Native-resolution residual geometry refiner for selected ambiguous ROIs.

    Unlike the old local mask decoder, this module does not emit an independent
    instance mask. It corrects the shared geometry fields; watershed/RAG
    partitioning is rerun afterward, preserving connectivity and exclusivity.
    """

    def __init__(
        self,
        cfg: RefinementConfig,
        spatial_cfg: SpatialConfig,
        geometry_feature_channels: int,
        d_model: int,
    ):
        super().__init__()
        self.cfg = cfg
        # D0 + original inputs + probabilities/geometry:
        # fg,surface,sep,sdf,flow3,offset3,seed = 11 channels, plus
        # request-relative physical dz/dy/dx/distance = 4 channels.
        in_channels = spatial_cfg.channels[0] + spatial_cfg.in_channels + 11 + 4
        h = cfg.hidden_channels
        q = cfg.query_channels
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, h, 3, padding=1, bias=False),
            nn.GroupNorm(groups_for(h), h),
            nn.SiLU(),
            nn.Conv3d(h, h, 3, padding=1, bias=False),
            nn.GroupNorm(groups_for(h), h),
            nn.SiLU(),
        )
        self.query = nn.Sequential(
            nn.Linear(d_model, q), nn.SiLU(), nn.Linear(q, 2 * h)
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(h, h, 1),
            nn.SiLU(),
            nn.Conv3d(h, 11, 1),
        )

    def _decode_crop(self, local: Tensor, token: Tensor) -> Tensor:
        spatial = self.spatial(local[None])
        scale, bias = self.query(token[None]).chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale)[..., None, None, None]
        bias = bias[..., None, None, None]
        return self.fuse(spatial * (1.0 + scale) + bias)[0]

    @staticmethod
    def _relative_coordinates(
        shape: tuple[int, int, int],
        crop: tuple[slice, slice, slice],
        spacing_um: Tensor,
        center_um: Tensor,
        dref_um: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        patch_center_voxel = 0.5 * (
            torch.as_tensor(shape, device=device, dtype=torch.float32) - 1
        )
        axes = []
        for axis, axis_slice in enumerate(crop):
            voxel = torch.arange(
                int(axis_slice.start),
                int(axis_slice.stop),
                device=device,
                dtype=torch.float32,
            )
            axes.append((voxel - patch_center_voxel[axis]) * spacing_um[axis].float())
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        coordinates = torch.stack([zz, yy, xx], dim=0)
        delta = (coordinates - center_um.float()[:, None, None, None]) / dref_um.float().clamp_min(1e-6)
        distance = torch.linalg.vector_norm(delta, dim=0, keepdim=True)
        return torch.cat([delta, distance], dim=0).to(dtype=dtype)

    def _bounded_crop(
        self,
        shape: tuple[int, int, int],
        spacing_um: Tensor,
        center_um: Tensor,
        radius_um: Tensor,
    ) -> tuple[slice, slice, slice]:
        radius = radius_um.float()
        for _ in range(24):
            slices = physical_crop_slices(shape, spacing_um, center_um, radius)
            voxels = 1
            for axis_slice in slices:
                voxels *= max(int(axis_slice.stop) - int(axis_slice.start), 0)
            if voxels <= self.cfg.max_roi_voxels:
                return slices
            scale = max(
                0.10,
                0.95 * (self.cfg.max_roi_voxels / max(voxels, 1)) ** (1.0 / 3.0),
            )
            radius = radius * scale
        raise RuntimeError("Unable to bound local refinement ROI")

    def forward(
        self,
        d0: Tensor,
        spatial_inputs: Tensor,
        geometry: GeometryState,
        spacing_um: Tensor,
        dref_um: Tensor,
        requests: List[RefinementRequest],
    ) -> RefinementState:
        if not self.cfg.enabled or not requests:
            return RefinementState(geometry=geometry, requests=requests, applied_count=0)
        probs = geometry.probabilities()
        planned: list[tuple[RefinementRequest, tuple[slice, slice, slice]]] = []
        for request in requests:
            batch_index = request.batch_index
            planned.append(
                (
                    request,
                    self._bounded_crop(
                        tuple(geometry.sdf.shape[-3:]),
                        spacing_um[batch_index],
                        request.center_um,
                        self.cfg.roi_radius_dref * dref_um[batch_index],
                    ),
                )
            )
        fields = [
            geometry.foreground_logits.clone(),
            geometry.surface_logits.clone(),
            geometry.separator_logits.clone(),
            geometry.sdf.clone(),
            geometry.flow.clone(),
            geometry.centroid_offset.clone(),
            geometry.seed_logits.clone(),
        ]
        applied = 0
        for request, zyx in planned:
            b = request.batch_index
            dense_crop = torch.cat(
                [
                    probs["foreground"][b, :, zyx[0], zyx[1], zyx[2]],
                    probs["surface"][b, :, zyx[0], zyx[1], zyx[2]],
                    probs["separator"][b, :, zyx[0], zyx[1], zyx[2]],
                    geometry.sdf[b, :, zyx[0], zyx[1], zyx[2]],
                    geometry.flow[b, :, zyx[0], zyx[1], zyx[2]],
                    geometry.centroid_offset[b, :, zyx[0], zyx[1], zyx[2]],
                    probs["seed"][b, :, zyx[0], zyx[1], zyx[2]],
                ],
                dim=0,
            ).to(dtype=d0[b].dtype)
            local = torch.cat(
                [
                    d0[b, :, zyx[0], zyx[1], zyx[2]],
                    spatial_inputs[b, :, zyx[0], zyx[1], zyx[2]].to(
                        dtype=d0[b].dtype
                    ),
                    dense_crop,
                    self._relative_coordinates(
                        tuple(geometry.sdf.shape[-3:]),
                        zyx,
                        spacing_um[b],
                        request.center_um,
                        dref_um[b],
                        device=d0.device,
                        dtype=d0.dtype,
                    ),
                ],
                dim=0,
            )
            residual = self._decode_crop(local, request.query_token.to(local.dtype))
            residual = self.cfg.residual_scale * torch.tanh(residual)
            crop_shape = tuple(axis.stop - axis.start for axis in zyx)
            overlap = residual.new_zeros((1, *crop_shape))
            for other_request, other in planned:
                if other_request.batch_index != b:
                    continue
                lower = [max(zyx[axis].start, other[axis].start) for axis in range(3)]
                upper = [min(zyx[axis].stop, other[axis].stop) for axis in range(3)]
                if any(stop <= start for start, stop in zip(lower, upper)):
                    continue
                local_slices = tuple(
                    slice(lower[axis] - zyx[axis].start, upper[axis] - zyx[axis].start)
                    for axis in range(3)
                )
                overlap[(slice(None), *local_slices)] += 1
            residual = residual / overlap.clamp_min(1)
            deltas = torch.split(residual, (1, 1, 1, 1, 3, 3, 1), dim=0)
            for field, delta in zip(fields, deltas):
                field[b, :, zyx[0], zyx[1], zyx[2]] = (
                    field[b, :, zyx[0], zyx[1], zyx[2]] + delta.to(field.dtype)
                )
            applied += 1
        refined = replace(
            geometry,
            foreground_logits=fields[0],
            surface_logits=fields[1],
            separator_logits=fields[2],
            sdf=fields[3],
            flow=fields[4],
            centroid_offset=fields[5],
            seed_logits=fields[6],
        )
        return RefinementState(
            geometry=refined, requests=requests, applied_count=applied
        )
