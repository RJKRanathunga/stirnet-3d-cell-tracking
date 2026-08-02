"""Compatibility constants backed by the canonical shared path model."""

from src.io.paths import PipelinePaths

_PATHS = PipelinePaths.discover()
PROJECT_ROOT = _PATHS.project_root
DATA_ROOT = _PATHS.data_root
TRAIN_ROOT = DATA_ROOT / "biohub_5samples_20timepoints" / "train"
PROCESSED_ROOT = _PATHS.processed_root
