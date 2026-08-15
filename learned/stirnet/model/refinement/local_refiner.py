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
        # fg,surface,sep,sdf,flow3,offset3,seed = 11 channels.
        in_channels = spatial_cfg.channels[0] + spatial_cfg.in_channels + 11
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
            nn.Linear(d_model, q), nn.SiLU(), nn.Linear(q, q)
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(h + q, h, 1),
            nn.SiLU(),
            nn.Conv3d(h, 11, 1),
        )

    def _decode_crop(self, local: Tensor, token: Tensor) -> Tensor:
        spatial = self.spatial(local[None])
        q = self.query(token[None])[..., None, None, None]
        q = q.expand(-1, -1, *spatial.shape[-3:])
        return self.fuse(torch.cat([spatial, q], dim=1))[0]

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
        base_fields = [
            geometry.foreground_logits,
            geometry.surface_logits,
            geometry.separator_logits,
            geometry.sdf,
            geometry.flow,
            geometry.centroid_offset,
            geometry.seed_logits,
        ]
        # 11 residual channels correspond to 1+1+1+1+3+3+1.
        accum = geometry.sdf.new_zeros(
            (geometry.sdf.shape[0], 11, *geometry.sdf.shape[-3:])
        )
        weight = geometry.sdf.new_zeros(
            (geometry.sdf.shape[0], 1, *geometry.sdf.shape[-3:])
        )
        applied = 0
        for request in requests:
            b = request.batch_index
            radius_um = self.cfg.roi_radius_dref * dref_um[b]
            zyx = physical_crop_slices(
                tuple(geometry.sdf.shape[-3:]),
                spacing_um[b],
                request.center_um,
                radius_um,
            )
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
            )
            local = torch.cat(
                [
                    d0[b, :, zyx[0], zyx[1], zyx[2]],
                    spatial_inputs[b, :, zyx[0], zyx[1], zyx[2]],
                    dense_crop,
                ],
                dim=0,
            )
            residual = self._decode_crop(local, request.query_token.to(local.dtype))
            residual = self.cfg.residual_scale * torch.tanh(residual)
            accum[b, :, zyx[0], zyx[1], zyx[2]] = (
                accum[b, :, zyx[0], zyx[1], zyx[2]] + residual
            )
            weight[b, :, zyx[0], zyx[1], zyx[2]] = (
                weight[b, :, zyx[0], zyx[1], zyx[2]] + 1.0
            )
            applied += 1
        correction = accum / weight.clamp_min(1)
        correction = correction * (weight > 0).to(correction.dtype)
        i = 0
        fg = geometry.foreground_logits + correction[:, i : i + 1]; i += 1
        surface = geometry.surface_logits + correction[:, i : i + 1]; i += 1
        separator = geometry.separator_logits + correction[:, i : i + 1]; i += 1
        sdf = geometry.sdf + correction[:, i : i + 1]; i += 1
        flow = geometry.flow + correction[:, i : i + 3]; i += 3
        offset = geometry.centroid_offset + correction[:, i : i + 3]; i += 3
        seed = geometry.seed_logits + correction[:, i : i + 1]
        refined = replace(
            geometry,
            foreground_logits=fg,
            surface_logits=surface,
            separator_logits=separator,
            sdf=sdf,
            flow=flow,
            centroid_offset=offset,
            seed_logits=seed,
        )
        return RefinementState(
            geometry=refined, requests=requests, applied_count=applied
        )
