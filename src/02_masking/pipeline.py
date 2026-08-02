from .threshold import (
    compute_otsu_threshold,
    apply_threshold,
)
from .connected_components import (
    label_connected_components,
)
import numpy as np

from src.diagnostics import StageTrace


def create_binary_mask(
        volume: np.ndarray,
        *,
        return_diagnostics: bool = False,
):

    threshold = compute_otsu_threshold(volume)

    binary_mask = apply_threshold(
        volume,
        threshold,
    )

    if not return_diagnostics:
        return binary_mask
    trace = StageTrace(
        stage_name="02_masking",
        inputs={"volume": volume},
        outputs={"binary_mask": binary_mask},
        intermediates={"threshold": threshold},
        metrics={"foreground_voxels": int(np.count_nonzero(binary_mask))},
    )
    return binary_mask, trace
