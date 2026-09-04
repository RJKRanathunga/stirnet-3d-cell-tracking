"""Stage 4: learned track-stitching/reconciliation boundary."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any


class LearnedTrackStitchingNotFinalized(RuntimeError):
    """Raised when no production learned stitcher has been supplied."""


def require_track_stitcher(stitcher: Callable[..., Any] | None) -> Callable[..., Any]:
    if stitcher is None:
        raise LearnedTrackStitchingNotFinalized(
            "learned.track_reconciler exists, but its production inference "
            "adapter/checkpoint contract is not finalized. Supply an explicit "
            "learned stitcher; do not silently fall back to the classical stitcher."
        )
    return stitcher


__all__ = ["LearnedTrackStitchingNotFinalized", "require_track_stitcher"]
