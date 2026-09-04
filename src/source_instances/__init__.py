"""Production source-instance front-end for the current pipeline.

These algorithms are classical, but they are active: STIR-Net consumes
their normalized image, foreground mask, and source instances.
"""

from .preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    preprocess_volume,
)
from .foreground import (
    DEFAULT_MASKING_CONFIG,
    MaskingConfig,
    create_binary_mask,
)
from .segmentation import (
    DEFAULT_SEGMENTATION_CONFIG,
    SegmentationConfig,
    segment_instances,
    segment_instances_detailed,
)
from .detection.pipeline import detect_cells
from .features.pipeline import extract_cell_features
from .volume_pipeline import process_dataset

__all__ = [
    "DEFAULT_MASKING_CONFIG",
    "DEFAULT_PREPROCESSING_CONFIG",
    "DEFAULT_SEGMENTATION_CONFIG",
    "MaskingConfig",
    "PreprocessingConfig",
    "SegmentationConfig",
    "create_binary_mask",
    "detect_cells",
    "extract_cell_features",
    "preprocess_volume",
    "process_dataset",
    "segment_instances",
    "segment_instances_detailed",
]
