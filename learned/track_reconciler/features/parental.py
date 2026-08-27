"""Target- and temporal-gap-aware parental normalization."""

from __future__ import annotations

import torch
from torch import Tensor


def parental_softmax(
    logits: Tensor,
    target_index: Tensor,
    gap_frames: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """HOCT/Trackastra-style parental softmax with an implicit no-parent state.

    Candidate parents of the same target compete only when their temporal gap
    is equal.  The constant unit logit for the no-parent state prevents a weak
    candidate set from being normalized into a false high-confidence edge.

    Returns
    -------
    edge_probability:
        `[B,E]`, zero for masked edges.
    no_parent_probability:
        `[B,E]`; each active edge receives the no-parent probability of its
        `(target, gap)` group.  Repetition keeps downstream tensor APIs simple.
    """

    if logits.ndim != 2 or target_index.shape != logits.shape or gap_frames.shape != logits.shape:
        raise ValueError("logits, target_index and gap_frames must be [B,E]")
    if mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B,E]")

    # Probability normalization is deliberately performed in float32 for
    # low-precision logits. CUDA autocast promotes logsumexp/exp to float32
    # anyway; matching the output dtype avoids BF16/FP16 indexed-assignment
    # errors and improves numerical precision for the downstream focal loss.
    probability_dtype = (
        torch.float32
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits.dtype
    )
    probs = torch.zeros(
        logits.shape,
        device=logits.device,
        dtype=probability_dtype,
    )
    no_parent = torch.ones(
        logits.shape,
        device=logits.device,
        dtype=probability_dtype,
    )

    for batch in range(logits.shape[0]):
        active = torch.nonzero(mask[batch], as_tuple=False).flatten()
        if active.numel() == 0:
            continue

        # Python grouping is fine because local reconciliation graphs are small;
        # all probability computations themselves remain differentiable tensors.
        groups: dict[tuple[int, int], list[int]] = {}
        for idx in active.tolist():
            key = (int(target_index[batch, idx]), int(gap_frames[batch, idx]))
            groups.setdefault(key, []).append(idx)

        for indices in groups.values():
            ids = torch.tensor(
                indices,
                device=logits.device,
                dtype=torch.long,
            )
            local = logits[batch, ids].to(probability_dtype)
            zero = torch.zeros(
                1,
                device=logits.device,
                dtype=probability_dtype,
            )
            log_denom = torch.logsumexp(
                torch.cat((zero, local)),
                dim=0,
            )
            local_p = torch.exp(local - log_denom)
            q = torch.exp(-log_denom)

            probs[batch, ids] = local_p
            no_parent[batch, ids] = q

    return probs, no_parent
