"""U-Net decoder for the vector instance CNN."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualAnisotropicBlock, ResidualIsotropicBlock, Upsample3D
from .encoder import EncoderFeatures


class VectorCNNDecoder(nn.Module):
    def __init__(
        self,
        *,
        channels: tuple[int, int, int, int, int],
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3, cb = channels
        self.up3 = Upsample3D(cb, c3, groups=groups)
        self.fuse3 = ResidualIsotropicBlock(c3 + c3, c3, groups=groups, dropout=dropout)
        self.up2 = Upsample3D(c3, c2, groups=groups)
        self.fuse2 = ResidualIsotropicBlock(c2 + c2, c2, groups=groups, dropout=dropout)
        self.up1 = Upsample3D(c2, c1, groups=groups)
        self.fuse1 = ResidualAnisotropicBlock(c1 + c1, c1, groups=groups, dropout=dropout)
        self.up0 = Upsample3D(c1, c0, groups=groups)
        self.fuse0 = ResidualAnisotropicBlock(c0 + c0, c0, groups=groups, dropout=dropout)

    @staticmethod
    def _shape(x: torch.Tensor) -> tuple[int, int, int]:
        return tuple(int(v) for v in x.shape[-3:])

    def forward(self, features: EncoderFeatures) -> torch.Tensor:
        x = self.up3(features.bottleneck, target_shape=self._shape(features.level3))
        x = self.fuse3(torch.cat((x, features.level3), dim=1))
        x = self.up2(x, target_shape=self._shape(features.level2))
        x = self.fuse2(torch.cat((x, features.level2), dim=1))
        x = self.up1(x, target_shape=self._shape(features.level1))
        x = self.fuse1(torch.cat((x, features.level1), dim=1))
        x = self.up0(x, target_shape=self._shape(features.level0))
        return self.fuse0(torch.cat((x, features.level0), dim=1))
