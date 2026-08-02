"""Stage 6 orchestration over the stable stage APIs and shared artifact I/O."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.api import (
    create_binary_mask,
    detect_cells,
    extract_cell_features,
    preprocess_volume,
    segment_instances,
)
from src.io import load_timepoint, open_sample, save_processed_frame


def process_dataset(
    sample_path: str | Path,
    output_directory: str | Path,
    *,
    progress: Callable[[range], object] | None = None,
) -> int:
    """Process every timepoint and preserve the existing Stage 6 layout."""

    sample = open_sample(sample_path)
    frames = range(int(sample.shape[0]))
    iterable = progress(frames) if progress is not None else frames

    for frame in iterable:
        raw = load_timepoint(sample_path, frame)
        preprocessed = preprocess_volume(raw)
        binary_mask = create_binary_mask(preprocessed)
        labels = segment_instances(binary_mask)
        cells = detect_cells(labels)
        cells = extract_cell_features(cells, labels, preprocessed)
        save_processed_frame(
            output_directory,
            frame,
            preprocessed=preprocessed,
            binary_mask=binary_mask,
            instance_labels=labels,
            cells=cells,
        )

    return int(sample.shape[0])


__all__ = ["process_dataset"]
