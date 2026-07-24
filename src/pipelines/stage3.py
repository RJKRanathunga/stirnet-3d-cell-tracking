from pathlib import Path

import numpy as np

from src.io.zarr_loader import load_timepoint
from src.preprocessing.pipeline import preprocess_volume
from src.segmentation.thresholding import create_binary_mask


def run_stage3(
        zarr_path,
        timepoint=0
):
    """
    Run Stage 3 semantic segmentation on one timepoint.

    Pipeline:
        1. Load raw 3D volume
        2. Preprocess volume
        3. Create binary foreground/background mask

    Parameters
    ----------
    zarr_path : str or Path
        Path to the Zarr dataset.

    timepoint : int, default=0
        Timepoint to process.

    Returns
    -------
    binary_mask : np.ndarray
        3D binary semantic segmentation mask
        with shape (z, y, x).
    """

    zarr_path = Path(zarr_path)

    # ---------------------------------------------------------
    # 1. Load volume
    # ---------------------------------------------------------
    volume = load_timepoint(
        zarr_path,
        timepoint
    )

    # ---------------------------------------------------------
    # 2. Preprocess volume
    # ---------------------------------------------------------
    preprocessing_results = preprocess_volume(
        volume
    )

    # Select the volume that should be segmented
    corrected = preprocessing_results["normalized"]

    # ---------------------------------------------------------
    # 3. Semantic segmentation
    # ---------------------------------------------------------
    binary_mask = create_binary_mask(
        corrected
    )

    return binary_mask