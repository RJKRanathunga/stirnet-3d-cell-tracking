from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


T = TypeVar("T")


def checkpoint_if_enabled(
    function: Callable[..., T],
    *args: Tensor,
    enabled: bool,
) -> T:
    """Run a tensor-only function through non-reentrant checkpointing when useful."""
    if enabled and torch.is_grad_enabled() and any(arg.requires_grad for arg in args):
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)
