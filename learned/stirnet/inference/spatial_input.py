from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
import time
from typing import Callable

import numpy as np

from learned.stirnet.inference.runtime import (
    SpatialInferenceConfig,
)


@dataclass
class PreparedSpatialFrame:
    frame: int
    raw: np.ndarray
    preprocessed: np.ndarray
    source_mask: np.ndarray
    source_labels: np.ndarray
    spatial: np.ndarray
    dref_um: float
    preparation_seconds: float


def build_spatial_input(
    preprocessed: np.ndarray,
    source_labels: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Build canonical full-volume STIR-Net channels before CNN tiling."""
    from learned.stirnet.data.sample_builder import (
        build_spatial_channels,
    )
    from learned.stirnet.data.targets import (
        estimate_model_dref_um,
    )

    raw = np.asarray(
        preprocessed,
        dtype=np.float32,
    )
    labels = np.asarray(source_labels)

    if raw.shape != labels.shape or raw.ndim != 3:
        raise ValueError(
            "Preprocessing and source segmentation must align in 3-D; "
            f"got {raw.shape} and {labels.shape}"
        )

    if not np.isfinite(raw).all():
        raise ValueError(
            "Canonical preprocessing contains NaN/Inf values"
        )

    raw_min = float(raw.min()) if raw.size else 0.0
    raw_max = float(raw.max()) if raw.size else 0.0
    if raw_min < -1e-5 or raw_max > 1.0001:
        raise ValueError(
            "Canonical preprocessing is expected in [0,1], "
            f"observed [{raw_min:.6g}, {raw_max:.6g}]"
        )

    spacing = tuple(
        float(v) for v in spacing_zyx_um
    )
    dref_um = float(
        estimate_model_dref_um(
            labels,
            spacing,
        )
    )
    spatial = build_spatial_channels(
        raw,
        labels,
        spacing,
        dref_um,
        derive_marker=True,
    )

    return (
        np.ascontiguousarray(
            spatial,
            dtype=np.float32,
        ),
        dref_um,
    )


def canonical_source_segmentation_config():
    """
    Production source-instance configuration used by the validated Stage-6 path.

    Geometric completion remains disabled exactly as in the promoted Kaggle
    parallel pipeline.
    """
    module = import_module(
        "src.03_segmentation.config"
    )
    return replace(
        module.DEFAULT_SEGMENTATION_CONFIG,
        enable_geometric_completion=False,
    )


def prepare_spatial_frame(
    sample_zarr: str | Path,
    frame: int,
    *,
    config: SpatialInferenceConfig,
    segmentation_config,
    source_mask_validator: Callable[[int, np.ndarray], None] | None = None,
) -> PreparedSpatialFrame:
    """CPU-only preparation. This function must never touch CUDA."""
    from src.api import (
        create_binary_mask,
        preprocess_volume,
        segment_instances,
    )
    from src.io import load_timepoint

    started = time.perf_counter()

    raw = load_timepoint(
        Path(sample_zarr),
        int(frame),
    )
    preprocessed = preprocess_volume(raw)
    source_mask = create_binary_mask(
        preprocessed
    )
    if source_mask_validator is not None:
        source_mask_validator(
            int(frame),
            np.asarray(source_mask),
        )
    source_labels = segment_instances(
        source_mask,
        config=segmentation_config,
    )
    spatial, dref_um = build_spatial_input(
        preprocessed,
        source_labels,
        config.spacing_zyx_um,
    )

    return PreparedSpatialFrame(
        frame=int(frame),
        raw=np.asarray(raw),
        preprocessed=np.asarray(preprocessed),
        source_mask=np.asarray(source_mask),
        source_labels=np.asarray(
            source_labels,
            dtype=np.int32,
        ),
        spatial=spatial,
        dref_um=float(dref_um),
        preparation_seconds=float(
            time.perf_counter() - started
        ),
    )
