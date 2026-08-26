"""Hard-negative selection utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def hard_negative_indices(
    target: Tensor,
    hardness: Tensor,
    *,
    negatives_per_positive: int = 4,
    minimum_negatives: int = 8,
) -> Tensor:
    """Keep all positives and the most confusing negatives.

    `hardness` should increase for plausible/confusing negatives (for example
    old Stage-11 score, Trackastra probability, or inverse motion residual).
    Random unrelated cells are intentionally not the dominant negatives.
    """

    target = target.bool().flatten()
    hardness = hardness.flatten()
    positives = torch.nonzero(target, as_tuple=False).flatten()
    negatives = torch.nonzero(~target, as_tuple=False).flatten()
    count = max(minimum_negatives, negatives_per_positive * max(int(positives.numel()), 1))
    count = min(count, int(negatives.numel()))
    if count:
        selected_neg = negatives[torch.topk(hardness[negatives], k=count).indices]
        return torch.cat((positives, selected_neg)).sort().values
    return positives
