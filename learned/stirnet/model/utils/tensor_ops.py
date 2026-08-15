from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def relabel_contiguous(labels: Tensor) -> Tensor:
    """Relabel positive integer labels to 1..N without changing connectivity."""
    unique = torch.unique(labels)
    unique = unique[unique > 0]
    if unique.numel() == 0:
        return torch.zeros_like(labels)
    out = torch.zeros_like(labels)
    for new_id, old_id in enumerate(unique.tolist(), 1):
        out[labels == int(old_id)] = new_id
    return out


def pool_labeled_features(
    feature: Tensor,
    labels: Tensor,
    *,
    include_max: bool = True,
) -> tuple[Tensor, Tensor]:
    """Pool [C,Z,Y,X] features over contiguous labels [Z,Y,X].

    Returns [N,C] mean (or [N,2C] mean+max) and counts [N].
    Label zero is ignored. Empty IDs are retained if labels are not contiguous.
    """
    if feature.ndim != 4 or labels.ndim != 3:
        raise ValueError("feature must be [C,Z,Y,X] and labels [Z,Y,X]")
    max_id = int(labels.max().item())
    if max_id <= 0:
        width = feature.shape[0] * (2 if include_max else 1)
        return feature.new_zeros((0, width)), feature.new_zeros((0,))
    flat_labels = labels.reshape(-1).long()
    flat_feat = feature.flatten(1).transpose(0, 1)
    valid = flat_labels > 0
    ids = flat_labels[valid] - 1
    values = flat_feat[valid]
    counts = feature.new_zeros((max_id,))
    counts.index_add_(0, ids, torch.ones_like(ids, dtype=feature.dtype))
    sums = feature.new_zeros((max_id, feature.shape[0]))
    sums.index_add_(0, ids, values)
    means = sums / counts.clamp_min(1)[:, None]
    if not include_max:
        return means, counts
    maxima = feature.new_full((max_id, feature.shape[0]), -torch.inf)
    maxima.scatter_reduce_(
        0,
        ids[:, None].expand(-1, feature.shape[0]),
        values,
        reduce="amax",
        include_self=True,
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return torch.cat([means, maxima], dim=-1), counts


def resize_labels_nearest(labels: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
    if tuple(labels.shape[-3:]) == target_shape:
        return labels
    x = labels.float()[None, None]
    return F.interpolate(x, size=target_shape, mode="nearest")[0, 0].long()
