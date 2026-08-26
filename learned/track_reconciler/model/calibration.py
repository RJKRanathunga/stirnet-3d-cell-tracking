"""Post-training probability calibration helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemperatureScaler(nn.Module):
    """Learn one positive scalar temperature on held-out logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        initial = torch.tensor(float(initial_temperature)).log()
        self.log_temperature = nn.Parameter(initial)

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp_min(1e-4)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature
