from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from ..utils.physical import build_acquisition_features


class AcquisitionEmbedding(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, out_dim)
        )

    def forward(self, spacing_um: Tensor, dref_um: Tensor) -> Tensor:
        return self.net(build_acquisition_features(spacing_um, dref_um))


def choose_downsample_stride(
    spacing_um: Tensor, threshold: float = 1.5
) -> tuple[int, int, int]:
    s = spacing_um.median(dim=0).values if spacing_um.ndim == 2 else spacing_um
    max_s = float(s.max().item())
    stride = tuple(2 if float(v.item()) < max_s / threshold else 1 for v in s)
    if stride == (1, 1, 1):
        stride = (2, 2, 2)
    return stride


def propagate_spacing(spacing_um: Tensor, stride: Iterable[int]) -> Tensor:
    scale = torch.tensor(tuple(stride), device=spacing_um.device, dtype=spacing_um.dtype)
    return spacing_um * scale
