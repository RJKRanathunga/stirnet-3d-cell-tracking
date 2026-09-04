"""Stage 1: production source-instance generation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.source_instances import (
    create_binary_mask,
    detect_cells,
    extract_cell_features,
    preprocess_volume,
    segment_instances,
)


@dataclass
class SourceFrameResult:
    raw: np.ndarray
    preprocessed: np.ndarray
    foreground_mask: np.ndarray
    source_instances: np.ndarray
    cells: pd.DataFrame


def prepare_source_frame(raw: np.ndarray, *, segmentation_config=None) -> SourceFrameResult:
    """Create the classical source representation consumed by STIR-Net."""
    preprocessed = preprocess_volume(raw)
    foreground = create_binary_mask(preprocessed)
    if segmentation_config is None:
        labels = segment_instances(foreground)
    else:
        labels = segment_instances(foreground, config=segmentation_config)
    cells = detect_cells(labels)
    cells = extract_cell_features(cells, labels, preprocessed)
    return SourceFrameResult(
        raw=np.asarray(raw),
        preprocessed=np.asarray(preprocessed),
        foreground_mask=np.asarray(foreground),
        source_instances=np.asarray(labels),
        cells=cells,
    )


__all__ = ["SourceFrameResult", "prepare_source_frame"]
