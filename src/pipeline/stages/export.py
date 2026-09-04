"""Stage 5: final output/export boundary."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any


def require_exporter(exporter: Callable[..., Any] | None) -> Callable[..., Any]:
    if exporter is None:
        raise RuntimeError("No final-output exporter was supplied to the pipeline.")
    return exporter


__all__ = ["require_exporter"]
