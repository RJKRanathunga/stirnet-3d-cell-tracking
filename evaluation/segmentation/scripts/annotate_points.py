from pathlib import Path

import napari
import numpy as np
import pandas as pd
from scipy import ndimage

from src.io import load_timepoint


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ID = "44b6_0113de3b"

ZARR_PATH = Path(
    "../../../data/sample/biohub_5samples_20timepoints/train/"
    f"{SAMPLE_ID}/{SAMPLE_ID}.zarr"
)

TIMEPOINT = 0

# Stage 3 segmentation output
STAGE3_DIR = Path(
    f"../../../data/sample/processed/"
    f"stage3_segmentation/{SAMPLE_ID}"
)

BINARY_MASK_PATH = (
        STAGE3_DIR /
        "binary_mask.npy"
)

# Ground-truth annotation output
OUTPUT_DIR = Path(
    f"evaluation/segmentation/annotations/{SAMPLE_ID}"
)

OUTPUT_CSV = (
        OUTPUT_DIR /
        f"ground_truth_points_t{TIMEPOINT:03d}.csv"
)


# ============================================================
# LOAD RAW TIMEPOINT
# ============================================================

print("=" * 60)
print("Loading 3D timepoint")
print("=" * 60)

print(f"Zarr: {ZARR_PATH}")
print(f"Timepoint: {TIMEPOINT}")

volume = load_timepoint(
    ZARR_PATH,
    TIMEPOINT,
)

print(
    f"Raw volume shape: {volume.shape}"
)

print(
    f"Raw volume dtype: {volume.dtype}"
)


# ============================================================
# LOAD STAGE 3 BINARY MASK
# ============================================================

print()
print("=" * 60)
print("Loading Stage 3 binary mask")
print("=" * 60)

print(
    f"Binary mask: {BINARY_MASK_PATH}"
)

binary_mask = np.load(
    BINARY_MASK_PATH
)

print(
    f"Binary mask shape: {binary_mask.shape}"
)

print(
    f"Binary mask dtype: {binary_mask.dtype}"
)


# ============================================================
# VALIDATE VOLUME SHAPES
# ============================================================

if volume.shape != binary_mask.shape:

    raise ValueError(
        "Raw volume and binary mask shapes do not match!\n"
        f"Raw volume shape: {volume.shape}\n"
        f"Binary mask shape: {binary_mask.shape}"
    )


# ============================================================
# CONVERT BINARY MASK TO BOOLEAN
# ============================================================

binary_mask = (
        binary_mask > 0
)


# ============================================================
# EXTRACT 2D BOUNDARIES
# ============================================================

print()
print("=" * 60)
print("Extracting binary mask boundaries")
print("=" * 60)

boundaries = np.zeros(
    binary_mask.shape,
    dtype=bool,
)


for z in range(
        binary_mask.shape[0]
):

    mask_slice = (
        binary_mask[z]
    )

    # Erode the mask by one pixel.
    # The difference between the original
    # and eroded mask gives the boundary.

    eroded = ndimage.binary_erosion(
        mask_slice,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
    )

    boundaries[z] = (
            mask_slice &
            ~eroded
    )


print(
    f"Boundary voxels: "
    f"{boundaries.sum()}"
)


# ============================================================
# CREATE RGBA BOUNDARY OVERLAY
# ============================================================

# Napari expects:
#
# (z, y, x, RGBA)
#
# We create a transparent image where
# only boundary pixels are visible.

boundary_overlay = np.zeros(
    (
        *boundaries.shape,
        4,
    ),
    dtype=np.float32,
)


# Green channel
boundary_overlay[..., 1] = (
    boundaries.astype(
        np.float32
    )
)


# Alpha channel
boundary_overlay[..., 3] = (
    boundaries.astype(
        np.float32
    )
)


# ============================================================
# CREATE NAPARI VIEWER
# ============================================================

viewer = napari.Viewer(
    ndisplay=2
)


# ============================================================
# LAYER 1 — RAW VOLUME
# ============================================================

viewer.add_image(
    volume,
    name="Raw Volume",
    colormap="gray",
)


# ============================================================
# LAYER 2 — BINARY MASK BOUNDARY
# ============================================================

boundary_layer = (
    viewer.add_image(
        boundary_overlay,
        name="Segmentation Boundary",
        rgb=True,
        blending="additive",
        opacity=1.0,
    )
)


# ============================================================
# LAYER 3 — GROUND TRUTH POINTS
# ============================================================

points_layer = viewer.add_points(
    ndim=3,
    name="Ground Truth Points",
    size=8,
    face_color="red",
)


# ============================================================
# SAVE POINTS
# ============================================================

def save_points():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    points = (
        points_layer.data
    )

    if len(points) == 0:

        print(
            "No points have been annotated."
        )

        return

    # Napari coordinates:
    #
    # (z, y, x)

    df = pd.DataFrame(
        points,
        columns=[
            "z",
            "y",
            "x",
        ],
    )

    # Convert coordinates to
    # integer voxel indices.

    df = (
        df
        .round()
        .astype(int)
    )

    # Save CSV

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 60)
    print("Ground truth points saved")
    print("=" * 60)

    print(
        f"Output: {OUTPUT_CSV}"
    )

    print(
        f"Number of points: "
        f"{len(df)}"
    )

    print()

    print(df)


# ============================================================
# KEYBOARD SHORTCUT
# ============================================================

@viewer.bind_key(
    "Ctrl+S"
)
def save_annotation(
        viewer
):

    save_points()


# ============================================================
# ANNOTATION INSTRUCTIONS
# ============================================================

print()
print("=" * 60)
print("ANNOTATION INSTRUCTIONS")
print("=" * 60)

print()

print(
    "Raw image is displayed in grayscale."
)

print(
    "Stage 3 segmentation boundaries "
    "are displayed in green."
)

print(
    "Ground-truth points are displayed in red."
)

print()

print(
    "Click once at the center of every "
    "cell you can confidently identify."
)

print(
    "Navigate through all 65 z-slices."
)

print(
    "Do NOT draw cell boundaries."
)

print(
    "If a cell appears in multiple z-slices, "
    "click it on each visible slice."
)

print()

print(
    "Press Ctrl+S to save annotations."
)

print()

print(
    f"Output: {OUTPUT_CSV}"
)

print()


# ============================================================
# START NAPARI
# ============================================================

napari.run()
