"""Canonical end-to-end orchestration for the current pipeline."""

from __future__ import annotations

from .contracts import PipelineRequest, PipelineStages, PipelineState


def run_pipeline(request: PipelineRequest, stages: PipelineStages) -> PipelineState:
    """Run source -> STIR-Net -> Trackastra -> learned stitching -> export."""
    state = PipelineState()
    state.source_instances = stages.source_instances(request)
    state.spatial_refinement = stages.spatial_refinement(
        request, state.source_instances
    )
    state.primary_tracking = stages.primary_tracking(
        request, state.spatial_refinement
    )
    state.track_stitching = stages.track_stitching(
        request, state.spatial_refinement, state.primary_tracking
    )
    state.exported = stages.export(
        request, state.spatial_refinement, state.track_stitching
    )
    return state


__all__ = ["run_pipeline"]
