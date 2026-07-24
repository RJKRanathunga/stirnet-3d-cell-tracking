from pathlib import Path

import numpy as np

from src.io.zarr_loader import load_timepoint
from src.preprocessing.pipeline import preprocess_volume

from src.segmentation.thresholding import (
    create_binary_mask
)

from src.segmentation.labeling import (
    label_objects
)

from src.segmentation.statistics import (
    calculate_object_statistics
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZARR_PATH = (
        PROJECT_ROOT
        / "data/sample"
        / "biohub_5samples_20timepoints"
        / "train"
        / "44b6_0113de3b"
        / "44b6_0113de3b.zarr"
)

TIMEPOINT = 0


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("CELL TRACKING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load raw 3D volume
    # --------------------------------------------------------

    print("\nLoading volume...")

    volume = load_timepoint(
        ZARR_PATH,
        TIMEPOINT
    )

    print(
        "Raw volume shape:",
        volume.shape
    )

    print(
        "Raw dtype:",
        volume.dtype
    )

    # --------------------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------------------

    print("\nRunning preprocessing...")

    preprocessing_results = preprocess_volume(
        volume
    )

    corrected = preprocessing_results[
        "corrected"
    ]

    print(
        "Preprocessed volume shape:",
        corrected.shape
    )

    # --------------------------------------------------------
    # 3. Segmentation
    # --------------------------------------------------------

    print("\nRunning Otsu segmentation...")

    binary_mask = create_binary_mask(
        corrected
    )

    print(
        "Binary mask shape:",
        binary_mask.shape
    )

    # --------------------------------------------------------
    # 4. Connected-component labeling
    # --------------------------------------------------------

    print("\nLabeling objects...")

    labeled_volume, num_objects = label_objects(
        binary_mask
    )

    print(
        "Labeled volume shape:",
        labeled_volume.shape
    )

    print(
        "Number of objects:",
        num_objects
    )

    # --------------------------------------------------------
    # 5. Object statistics
    # --------------------------------------------------------

    print("\nCalculating object statistics...")

    objects_df = calculate_object_statistics(
        labeled_volume
    )

    print(
        objects_df.head()
    )

    print("\nObject statistics:")
    print(
        objects_df["volume"].describe()
    )


if __name__ == "__main__":
    main()