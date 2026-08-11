from __future__ import annotations

from typing import Any
import json

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def tensor_stats(
    tensor: Tensor,
    *,
    name: str = "",
    channel_stats: bool = False,
) -> dict[str, Any]:
    """Compute bounded-size numerical diagnostics without copying full tensors."""

    t = tensor.detach()
    row: dict[str, Any] = {
        "name": name,
        "shape": "x".join(str(int(v)) for v in t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "numel": int(t.numel()),
    }

    if t.numel() == 0:
        row.update(
            finite_fraction=1.0,
            zero_fraction=1.0,
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            rms=0.0,
            abs_max=0.0,
        )
        return row

    work = t.float()
    finite = torch.isfinite(work)
    finite_count = int(finite.sum().item())
    row["finite_fraction"] = finite_count / max(work.numel(), 1)

    if finite_count == 0:
        row.update(
            zero_fraction=0.0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            rms=float("nan"),
            abs_max=float("nan"),
        )
        return row

    values = work[finite]
    row.update(
        zero_fraction=float((values == 0).float().mean().item()),
        mean=float(values.mean().cpu()),
        std=float(values.std(unbiased=False).cpu()),
        min=float(values.min().cpu()),
        max=float(values.max().cpu()),
        rms=float(values.square().mean().sqrt().cpu()),
        abs_max=float(values.abs().max().cpu()),
    )

    if channel_stats and t.ndim >= 2:
        dims = tuple(i for i in range(t.ndim) if i != 1)
        row["channel_mean"] = work.mean(dim=dims).cpu().numpy().astype(np.float32)
        row["channel_std"] = work.std(dim=dims, unbiased=False).cpu().numpy().astype(np.float32)

    return row


def flatten_stats_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            result[key] = json.dumps(value.tolist(), separators=(",", ":"))
        else:
            result[key] = value
    return result


@torch.no_grad()
def feature_norm_volume(tensor: Tensor) -> np.ndarray:
    if tensor.ndim != 5:
        raise ValueError(f"Expected [B,C,Z,Y,X], got {tuple(tensor.shape)}")
    norm = torch.linalg.vector_norm(tensor.detach().float(), dim=1)
    return norm[0].cpu().numpy().astype(np.float32, copy=False)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-8)
