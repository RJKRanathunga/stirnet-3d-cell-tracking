from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class LabelContingency:
    row_ids: Tensor
    column_ids: Tensor
    intersections: Tensor
    row_counts: Tensor
    column_counts: Tensor


def label_contingency(row_labels: Tensor, column_labels: Tensor, *, valid_mask: Tensor | None = None) -> LabelContingency:
    """Positive-label contingency; ignored voxels count as neither GT nor background."""
    if row_labels.shape != column_labels.shape: raise ValueError('contingency partitions must have identical shapes')
    rows = row_labels.reshape(-1).long(); cols = column_labels.to(row_labels.device).reshape(-1).long()
    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=row_labels.device).bool()
        if valid.ndim == row_labels.ndim + 1 and valid.shape[0] == 1: valid = valid[0]
        if valid.shape != row_labels.shape: raise ValueError('valid_mask must match contingency label shape')
        keep = valid.reshape(-1); rows = rows[keep]; cols = cols[keep]
    row_ids, _, row_counts = torch.unique(rows[rows > 0], sorted=True, return_inverse=True, return_counts=True)
    col_ids, _, col_counts = torch.unique(cols[cols > 0], sorted=True, return_inverse=True, return_counts=True)
    intersections = torch.zeros((row_ids.numel(), col_ids.numel()), device=rows.device, dtype=torch.long)
    positive = (rows > 0) & (cols > 0)
    if positive.any() and row_ids.numel() and col_ids.numel():
        ri = torch.searchsorted(row_ids, rows[positive]); ci = torch.searchsorted(col_ids, cols[positive]); packed = ri * col_ids.numel() + ci
        intersections = torch.bincount(packed, minlength=row_ids.numel() * col_ids.numel()).reshape(row_ids.numel(), col_ids.numel())
    return LabelContingency(row_ids, col_ids, intersections, row_counts, col_counts)

__all__ = ['LabelContingency', 'label_contingency']
