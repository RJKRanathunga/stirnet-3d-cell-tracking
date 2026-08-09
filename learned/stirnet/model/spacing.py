from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


def build_acquisition_features(spacing_um: Tensor, dref_um: Tensor) -> Tensor:
    if spacing_um.ndim == 1:
        spacing_um = spacing_um.unsqueeze(0)
    if dref_um.ndim == 0:
        dref_um = dref_um.unsqueeze(0)
    log_s = torch.log(spacing_um.clamp_min(1e-8))
    rel = torch.log((spacing_um / dref_um[:, None].clamp_min(1e-8)).clamp_min(1e-8))
    return torch.cat([log_s, rel, torch.log(dref_um[:, None].clamp_min(1e-8))], dim=-1)


class AcquisitionEmbedding(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, out_dim))

    def forward(self, spacing_um: Tensor, dref_um: Tensor) -> Tensor:
        return self.net(build_acquisition_features(spacing_um, dref_um))


def choose_downsample_stride(spacing_um: Tensor, threshold: float = 1.5) -> tuple[int, int, int]:
    """Choose one shared stride for a spacing-compatible batch."""
    if spacing_um.ndim == 2:
        s = spacing_um.median(dim=0).values
    else:
        s = spacing_um
    max_s = float(s.max().item())
    stride = tuple(2 if float(v.item()) < max_s / threshold else 1 for v in s)
    if stride == (1, 1, 1):
        stride = (2, 2, 2)
    return stride


def propagate_spacing(spacing_um: Tensor, stride: Iterable[int]) -> Tensor:
    scale = torch.tensor(tuple(stride), device=spacing_um.device, dtype=spacing_um.dtype)
    return spacing_um * scale


def physical_radius_to_voxels(radius_um: Tensor | float, spacing_um: Tensor) -> Tensor:
    radius = torch.as_tensor(radius_um, device=spacing_um.device, dtype=spacing_um.dtype)
    while radius.ndim < spacing_um.ndim:
        radius = radius.unsqueeze(-1)
    return torch.ceil(radius / spacing_um.clamp_min(1e-8)).long()
