"""Reusable Stage 3 instance-segmentation diagnostic package."""

from __future__ import annotations

from .models import (
    CanonicalMismatchError,
    DisplayScope,
    Stage3ComponentRun,
    Stage3FrameSelection,
    TargetComponentResolution,
)
from .runner import run_stage3_component
from .source import (
    Stage3AnalysisSource,
    build_display_scope,
    resolve_target_components,
)


def add_stage3_analysis_widget(*args, **kwargs):
    """Lazily import and install the Qt/Napari Stage 3 widget."""

    from .napari_widget import add_stage3_analysis_widget as install

    return install(*args, **kwargs)


def __getattr__(name: str):
    if name == "Stage3AnalysisWidget":
        from .napari_widget import Stage3AnalysisWidget

        return Stage3AnalysisWidget
    raise AttributeError(name)


__all__ = [
    "CanonicalMismatchError",
    "DisplayScope",
    "Stage3AnalysisSource",
    "Stage3AnalysisWidget",
    "Stage3ComponentRun",
    "Stage3FrameSelection",
    "TargetComponentResolution",
    "add_stage3_analysis_widget",
    "build_display_scope",
    "resolve_target_components",
    "run_stage3_component",
]
