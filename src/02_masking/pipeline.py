from .threshold import (
    compute_otsu_threshold,
    apply_threshold,
)
from .connected_components import (
    label_connected_components,
)
import numpy as np

from src.diagnostics import StageTrace
from .config import DEFAULT_MASKING_CONFIG, MaskingConfig


def create_binary_mask(
    volume: np.ndarray,
    *,
    config: MaskingConfig = DEFAULT_MASKING_CONFIG,
    return_diagnostics: bool = False,
):

    base_threshold = compute_otsu_threshold(volume)
    threshold = (
        base_threshold * config.threshold_multiplier + config.threshold_offset
    )

    binary_mask = apply_threshold(
        volume,
        threshold,
    )

    if not return_diagnostics:
        return binary_mask
    trace = StageTrace(
        stage_name="02_masking",
        inputs={"volume": volume, "config": config},
        outputs={"binary_mask": binary_mask},
        intermediates={
            "base_otsu_threshold": float(base_threshold),
            "effective_threshold": float(threshold),
            "binary_mask": binary_mask,
        },
        metrics={
            "base_otsu_threshold": float(base_threshold),
            "effective_threshold": float(threshold),
            "foreground_voxels": int(np.count_nonzero(binary_mask)),
        },
    )
    return binary_mask, trace
