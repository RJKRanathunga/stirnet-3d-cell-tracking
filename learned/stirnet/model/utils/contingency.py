from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LabelContingency:
    """Compact positive-label overlap table for two aligned integer partitions."""

    row_ids: Tensor
    column_ids: Tensor
    intersections: Tensor
    row_counts: Tensor
    column_counts: Tensor


def label_contingency(row_labels: Tensor, column_labels: Tensor) -> LabelContingency:
    """Build a positive-label contingency table with one flattened volume scan.

    Counts for each partition include overlap with background in the other
    partition, while ``intersections`` contains positive-positive overlap only.
    Arbitrary non-contiguous label IDs are supported without allocating by the
    maximum raw label value.
    """
    if row_labels.shape != column_labels.shape:
        raise ValueError("contingency partitions must have identical shapes")
    rows = row_labels.reshape(-1).long()
    columns = column_labels.to(row_labels.device).reshape(-1).long()
    row_ids, row_inverse, row_counts = torch.unique(
        rows[rows > 0], sorted=True, return_inverse=True, return_counts=True
    )
    column_ids, column_inverse, column_counts = torch.unique(
        columns[columns > 0], sorted=True, return_inverse=True, return_counts=True
    )
    intersections = torch.zeros(
        (row_ids.numel(), column_ids.numel()),
        device=rows.device,
        dtype=torch.long,
    )
    positive = (rows > 0) & (columns > 0)
    if positive.any() and row_ids.numel() and column_ids.numel():
        row_index = torch.searchsorted(row_ids, rows[positive])
        column_index = torch.searchsorted(column_ids, columns[positive])
        packed = row_index * column_ids.numel() + column_index
        intersections = torch.bincount(
            packed, minlength=row_ids.numel() * column_ids.numel()
        ).reshape(row_ids.numel(), column_ids.numel())
    return LabelContingency(
        row_ids=row_ids,
        column_ids=column_ids,
        intersections=intersections,
        row_counts=row_counts,
        column_counts=column_counts,
    )


__all__ = ["LabelContingency", "label_contingency"]
