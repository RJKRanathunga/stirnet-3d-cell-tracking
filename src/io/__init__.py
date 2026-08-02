"""Shared, contract-preserving pipeline I/O."""

from .arrays import load_npy, load_npy_time_series, load_timepoint, open_sample, save_npy
from .paths import (
    PipelinePaths,
    find_project_root,
    get_sample_output_dir,
    get_stage_dir,
)
from .scene_io import TrackingSceneData, load_tracking_scene
from .stage_io import (
    ProcessedDatasetInputs,
    Stage8Outputs,
    load_processed_dataset_inputs,
    load_stage7_detections,
    load_stage8_outputs,
    save_processed_frame,
    save_stitching_result,
    save_tracking_result,
)
from .tables import load_csv, load_json, load_optional_csv, save_csv, save_json

__all__ = [
    "PipelinePaths",
    "ProcessedDatasetInputs",
    "Stage8Outputs",
    "TrackingSceneData",
    "find_project_root",
    "get_sample_output_dir",
    "get_stage_dir",
    "load_csv",
    "load_json",
    "load_npy",
    "load_npy_time_series",
    "load_optional_csv",
    "load_processed_dataset_inputs",
    "load_stage7_detections",
    "load_stage8_outputs",
    "load_tracking_scene",
    "load_timepoint",
    "open_sample",
    "save_csv",
    "save_json",
    "save_npy",
    "save_processed_frame",
    "save_stitching_result",
    "save_tracking_result",
]
