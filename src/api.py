"""Deprecated compatibility facade.

New code should import src.source_instances, src.tracking, or src.pipeline.
"""

from __future__ import annotations
from importlib import import_module

from src.source_instances import (
    create_binary_mask,
    detect_cells,
    extract_cell_features,
    preprocess_volume,
    segment_instances,
)

_LEGACY_EXPORTS = {
    "run_cell_tracking": ("legacy.classical_pipeline.tracking", "run_cell_tracking"),
    "FourDGraphConfig": ("legacy.classical_pipeline.tracking", "FourDGraphConfig"),
    "GraphTrackingConfig": ("legacy.classical_pipeline.tracking", "GraphTrackingConfig"),
    "run_track_stitching": (
        "legacy.classical_pipeline.stitching.step01_pipeline", "run_track_stitching"
    ),
    "prepare_visualization_data": (
        "legacy.classical_pipeline.visualization.step03_pipeline",
        "prepare_visualization_data",
    ),
    "run_cell_lineage": (
        "legacy.classical_pipeline.lineage.step06_pipeline", "run_cell_lineage"
    ),
    "run_track_reconciliation": (
        "legacy.classical_pipeline.reconciliation.step11_pipeline",
        "run_track_reconciliation",
    ),
}


def __getattr__(name: str):
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "FourDGraphConfig",
    "GraphTrackingConfig",
    "create_binary_mask",
    "detect_cells",
    "extract_cell_features",
    "prepare_visualization_data",
    "preprocess_volume",
    "run_cell_lineage",
    "run_cell_tracking",
    "run_track_reconciliation",
    "run_track_stitching",
    "segment_instances",
]
