"""Narrow public facade for stable stage-level pipeline entry points."""

from __future__ import annotations

from importlib import import_module


preprocess_volume = import_module(
    "src.01_preprocessing.pipeline"
).preprocess_volume
create_binary_mask = import_module(
    "src.02_masking.pipeline"
).create_binary_mask
segment_instances = import_module(
    "src.03_segmentation.pipeline"
).segment_instances
detect_cells = import_module(
    "src.04_detection.pipeline"
).detect_cells
extract_cell_features = import_module(
    "src.05_feature_extraction.pipeline"
).extract_cell_features


def _not_available(name: str):
    def missing(*args, **kwargs):
        raise RuntimeError(f"Stage entry point {name} has not been installed")
    return missing


try:
    tracking_module = import_module("src.07_cell_tracking")

    run_cell_tracking = tracking_module.run_cell_tracking
    FourDGraphConfig = tracking_module.FourDGraphConfig
    GraphTrackingConfig = tracking_module.GraphTrackingConfig
except ModuleNotFoundError:
    run_cell_tracking = _not_available("run_cell_tracking")
    FourDGraphConfig = _not_available("FourDGraphConfig")
    GraphTrackingConfig = _not_available("GraphTrackingConfig")

try:
    run_track_stitching = import_module(
        "src.08_track_stitching.step01_pipeline"
    ).run_track_stitching
except ModuleNotFoundError:
    run_track_stitching = _not_available("run_track_stitching")

try:
    prepare_visualization_data = import_module(
        "src.09_visualization.step03_pipeline"
    ).prepare_visualization_data
except ModuleNotFoundError:
    prepare_visualization_data = _not_available("prepare_visualization_data")

try:
    run_cell_lineage = import_module(
        "src.10_cell_lineage.step06_pipeline"
    ).run_cell_lineage
except ModuleNotFoundError:
    run_cell_lineage = _not_available("run_cell_lineage")

try:
    run_track_reconciliation = import_module(
        "src.11_track_reconciliation.step11_pipeline"
    ).run_track_reconciliation
except ModuleNotFoundError:
    run_track_reconciliation = _not_available("run_track_reconciliation")


__all__ = [
    "FourDGraphConfig",
    "GraphTrackingConfig",
    "create_binary_mask",
    "detect_cells",
    "extract_cell_features",
    "prepare_visualization_data",
    "preprocess_volume",
    "run_cell_tracking",
    "run_cell_lineage",
    "run_track_reconciliation",
    "run_track_stitching",
    "segment_instances",
]
