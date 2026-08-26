"""Small reusable neural blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        layers: int = 2,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(in_dim, out_dim)
        modules: list[nn.Module] = []
        current = in_dim
        for _ in range(max(layers - 1, 0)):
            modules += [nn.Linear(current, hidden), nn.GELU(), nn.Dropout(dropout)]
            current = hidden
        modules.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    """Reliability-conditioned fusion rather than unconditional concatenation."""

    def __init__(self, a_dim: int, b_dim: int, reliability_dim: int, out_dim: int) -> None:
        super().__init__()
        self.a = nn.Linear(a_dim, out_dim)
        self.b = nn.Linear(b_dim, out_dim)
        self.gate = MLP(reliability_dim, 2 * out_dim, hidden_dim=out_dim, layers=2)
        self.out = nn.Sequential(nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, a: Tensor, b: Tensor, reliability: Tensor) -> Tensor:
        gates = torch.sigmoid(self.gate(reliability))
        ga, gb = gates.chunk(2, dim=-1)
        return self.out(ga * self.a(a) + gb * self.b(b))


class BranchDropout(nn.Module):
    """Drop an entire evidence branch without changing validity metadata.

    `shared_dims` lets temporal streams share one dropout decision across every
    observation of a tracklet, which is closer to true modality dropout than
    independently erasing random frames.
    """

    def __init__(self, probability: float, *, shared_dims: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.probability = float(probability)
        self.shared_dims = tuple(shared_dims)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability <= 0.0:
            return x
        shape = list(x.shape)
        shape[-1] = 1
        for dim in self.shared_dims:
            shape[dim] = 1
        keep = torch.rand(tuple(shape), device=x.device, dtype=x.dtype) >= self.probability
        return x * keep / (1.0 - self.probability)
