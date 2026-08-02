"""Small multi-stage diagnostic runner."""

from __future__ import annotations

from typing import Iterable

from src.api import (
    create_binary_mask,
    detect_cells,
    extract_cell_features,
    preprocess_volume,
    segment_instances,
)
from src.io import PipelinePaths, load_timepoint

from .models import DiagnosticTrace


def run_frame_diagnostics(
    sample_id: str,
    frame: int,
    *,
    stages: Iterable[str] = ("preprocessing", "masking", "segmentation", "detection", "features"),
    paths: PipelinePaths | None = None,
) -> DiagnosticTrace:
    """Run selected frame-local stages and combine their native traces."""

    requested = tuple(stages)
    resolved = paths or PipelinePaths.discover()
    raw = load_timepoint(resolved.sample_zarr(sample_id), frame)
    trace = DiagnosticTrace(scene_id=sample_id, frame_range=(int(frame), int(frame)))
    processed = raw
    if "preprocessing" in requested:
        processed, stage_trace = preprocess_volume(raw, return_diagnostics=True)
        trace.stages[stage_trace.stage_name] = stage_trace
    mask = None
    if "masking" in requested:
        mask, stage_trace = create_binary_mask(processed, return_diagnostics=True)
        trace.stages[stage_trace.stage_name] = stage_trace
    labels = None
    if "segmentation" in requested:
        if mask is None:
            raise ValueError("segmentation diagnostics require the masking stage")
        labels, stage_trace = segment_instances(mask, return_diagnostics=True)
        trace.stages[stage_trace.stage_name] = stage_trace
    cells = None
    if "detection" in requested:
        if labels is None:
            raise ValueError("detection diagnostics require the segmentation stage")
        cells, stage_trace = detect_cells(labels, return_diagnostics=True)
        trace.stages[stage_trace.stage_name] = stage_trace
    final_result = cells
    if "features" in requested:
        if cells is None or labels is None:
            raise ValueError("feature diagnostics require detection and segmentation")
        final_result, stage_trace = extract_cell_features(
            cells, labels, processed, return_diagnostics=True
        )
        trace.stages[stage_trace.stage_name] = stage_trace
    trace.final_result = final_result
    return trace
