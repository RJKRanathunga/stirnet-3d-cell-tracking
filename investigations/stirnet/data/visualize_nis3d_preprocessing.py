"""
Audit the CURRENT production preprocessing + masking pipeline on NIS3D.

This script intentionally calls the existing repository implementation:

    src.api.preprocess_volume
    src.api.create_binary_mask

so the visualization represents the real production pipeline rather than
a duplicated approximation.

Run from the project root:

    python investigations/stirnet/data/visualize_nis3d_preprocessing.py

Default sample:
    Zebrafish_2

Choose another sample:

    python investigations/stirnet/data/visualize_nis3d_preprocessing.py \
        --sample Zebrafish_1

By default the same production preprocessing parameters are used, but
with the physical voxel spacing read from NIS3D Info.txt.

To deliberately reproduce the exact Biohub default voxel spacing:

    python investigations/stirnet/data/visualize_nis3d_preprocessing.py \
        --sample Zebrafish_2 \
        --spacing-mode biohub
"""

from __future__ import annotations

import argparse
import re
from importlib import import_module
from pathlib import Path

import napari
import numpy as np
import tifffile


# ---------------------------------------------------------------------
# Project / dataset paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NIS3D_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "NIS3D"
    / "NIS3D"
)

SAMPLES = (
    "Drosophila_1",
    "Drosophila_2",
    "MusMusculus_1",
    "MusMusculus_2",
    "Zebrafish_1",
    "Zebrafish_2",
)


# ---------------------------------------------------------------------
# Import the ACTUAL production pipeline.
# ---------------------------------------------------------------------

from src.api import (
    preprocess_volume,
    create_binary_mask,
)

preprocessing_config_module = import_module(
    "src.01_preprocessing.config"
)

PreprocessingConfig = (
    preprocessing_config_module.PreprocessingConfig
)

DEFAULT_PREPROCESSING_CONFIG = (
    preprocessing_config_module.DEFAULT_PREPROCESSING_CONFIG
)


# ---------------------------------------------------------------------
# NIS3D filename variants
# ---------------------------------------------------------------------

RAW_NAMES = (
    "data.tif",
    "Data.tif",
)

GT_NAMES = (
    "GroundTruth.tif",
    "groundtruth.tif",
    "gt.tif",
    "GT.tif",
)

CONFIDENCE_NAMES = (
    "ConfidenceScore.tif",
    "confidencescore.tif",
    "scoreOfConfidence.tif",
    "ScoreOfConfidence.tif",
)

INFO_NAMES = (
    "Info.txt",
    "info.txt",
)


def find_file(
    directory: Path,
    candidates: tuple[str, ...],
) -> Path:
    """
    Resolve NIS3D's slightly inconsistent filenames.
    """

    for name in candidates:
        path = directory / name

        if path.exists():
            return path

    files = {
        path.name.lower(): path
        for path in directory.iterdir()
        if path.is_file()
    }

    for name in candidates:
        path = files.get(name.lower())

        if path is not None:
            return path

    available = "\n".join(
        f"    {path.name}"
        for path in directory.iterdir()
        if path.is_file()
    )

    raise FileNotFoundError(
        f"\nCould not find any of:\n"
        f"    {candidates}\n\n"
        f"in:\n"
        f"    {directory}\n\n"
        f"Available files:\n"
        f"{available}"
    )


# ---------------------------------------------------------------------
# Info.txt / physical spacing
# ---------------------------------------------------------------------

def read_info(
    sample_dir: Path,
) -> str:

    try:
        path = find_file(
            sample_dir,
            INFO_NAMES,
        )
    except FileNotFoundError:
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def parse_voxel_spacing(
    info: str,
) -> tuple[float, float, float] | None:
    """
    Parse NIS3D physical resolution.

    Examples:

        Resolution:
        1 um x 1 um x 1 um

        0.43 um x 0.43 um x 2.5 um

    NIS3D textual ordering is interpreted as:

        X × Y × Z

    Production functions expect:

        (Z, Y, X)
    """

    if not info:
        return None

    text = (
        info
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("×", "x")
    )

    number = r"([0-9]+(?:\.[0-9]+)?)"

    pattern = (
        number
        + r"\s*(?:um|micrometer(?:s)?)?\s*x\s*"
        + number
        + r"\s*(?:um|micrometer(?:s)?)?\s*x\s*"
        + number
        + r"\s*(?:um|micrometer(?:s)?)?"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    x, y, z = map(
        float,
        match.groups(),
    )

    return (
        z,
        y,
        x,
    )


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_tiff(
    path: Path,
) -> np.ndarray:
    """
    Load TIFF.

    For the preprocessing test we need a real ndarray because the
    production normalization immediately converts the volume to
    float32 anyway.
    """

    print(
        f"Loading {path.name} ..."
    )

    return tifffile.imread(
        path
    )


# ---------------------------------------------------------------------
# Mask evaluation
# ---------------------------------------------------------------------

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_mask(
    predicted_mask: np.ndarray,
    gt: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float]:
    """
    Compare production binary foreground mask against NIS3D GT.

    Confidence == 1 corresponds to undefined/unreliable regions, so
    those voxels are excluded from evaluation.
    """

    predicted = predicted_mask.astype(
        bool,
        copy=False,
    )

    target = gt > 0

    # NIS3D:
    #   0 = background
    #   1 = undefined / unreliable
    #   2 = 1/3 confidence
    #   3 = 2/3 confidence
    #   4 = 3/3 confidence
    valid = confidence != 1

    pred = predicted[valid]
    truth = target[valid]

    tp = int(
        np.count_nonzero(
            pred & truth
        )
    )

    fp = int(
        np.count_nonzero(
            pred & ~truth
        )
    )

    fn = int(
        np.count_nonzero(
            ~pred & truth
        )
    )

    tn = int(
        np.count_nonzero(
            ~pred & ~truth
        )
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    dice = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    iou = safe_divide(
        tp,
        tp + fp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    predicted_fraction = float(
        np.mean(pred)
    )

    gt_fraction = float(
        np.mean(truth)
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "specificity": specificity,
        "predicted_foreground_fraction": predicted_fraction,
        "gt_foreground_fraction": gt_fraction,
    }


def evaluate_cell_coverage(
    predicted_mask: np.ndarray,
    gt: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float]:
    """
    Determine how much of every GT nucleus survives the binary mask.

    This is especially useful here because a reasonable voxel Dice can
    still hide nuclei that disappear almost completely during
    preprocessing/masking.
    """

    predicted = predicted_mask.astype(
        bool,
        copy=False,
    )

    valid = confidence != 1

    max_label = int(
        np.max(gt)
    )

    valid_gt = gt[valid].ravel()

    valid_pred = (
        predicted[valid]
        .astype(np.uint8)
        .ravel()
    )

    total_voxels = np.bincount(
        valid_gt,
        minlength=max_label + 1,
    )

    covered_voxels = np.bincount(
        valid_gt,
        weights=valid_pred,
        minlength=max_label + 1,
    )

    total_voxels = total_voxels[1:]
    covered_voxels = covered_voxels[1:]

    exists = total_voxels > 0

    total_voxels = total_voxels[
        exists
    ]

    covered_voxels = covered_voxels[
        exists
    ]

    if total_voxels.size == 0:
        return {
            "n_cells": 0,
            "median_coverage": 0.0,
            "cells_ge_95": 0.0,
            "cells_ge_80": 0.0,
            "cells_ge_50": 0.0,
            "cells_zero": 0.0,
        }

    coverage = (
        covered_voxels
        / total_voxels
    )

    return {
        "n_cells": int(
            coverage.size
        ),
        "median_coverage": float(
            np.median(coverage)
        ),
        "cells_ge_95": float(
            np.mean(coverage >= 0.95)
        ),
        "cells_ge_80": float(
            np.mean(coverage >= 0.80)
        ),
        "cells_ge_50": float(
            np.mean(coverage >= 0.50)
        ),
        "cells_zero": float(
            np.mean(coverage == 0.0)
        ),
    }


def create_error_map(
    predicted_mask: np.ndarray,
    gt: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """
    Labels:

        0 = ignored / correct background
        1 = true positive foreground
        2 = false negative: GT missed by mask
        3 = false positive: mask outside GT
    """

    predicted = predicted_mask.astype(
        bool,
        copy=False,
    )

    target = gt > 0

    valid = confidence != 1

    error = np.zeros(
        gt.shape,
        dtype=np.uint8,
    )

    error[
        predicted
        & target
        & valid
    ] = 1

    error[
        (~predicted)
        & target
        & valid
    ] = 2

    error[
        predicted
        & (~target)
        & valid
    ] = 3

    return error


# ---------------------------------------------------------------------
# Contrast helper
# ---------------------------------------------------------------------

def raw_contrast_limits(
    image: np.ndarray,
) -> tuple[float, float]:

    z_step = max(
        1,
        image.shape[0] // 32,
    )

    y_step = max(
        1,
        image.shape[1] // 256,
    )

    x_step = max(
        1,
        image.shape[2] // 256,
    )

    sample = image[
        ::z_step,
        ::y_step,
        ::x_step,
    ]

    sample = np.asarray(
        sample
    )

    low, high = np.percentile(
        sample,
        (
            0.5,
            99.8,
        ),
    )

    if high <= low:
        low = float(
            np.min(sample)
        )

        high = float(
            np.max(sample)
        )

    return (
        float(low),
        float(high),
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the production preprocessing and binary masking "
            "pipeline on one NIS3D volume and inspect every stage."
        )
    )

    parser.add_argument(
        "--sample",
        choices=SAMPLES,
        default="Zebrafish_2",
        help=(
            "NIS3D sample to inspect. "
            "Default: Zebrafish_2"
        ),
    )

    parser.add_argument(
        "--spacing-mode",
        choices=(
            "dataset",
            "biohub",
        ),
        default="dataset",
        help=(
            "'dataset': use NIS3D physical voxel spacing while keeping "
            "all existing preprocessing parameters. "
            "'biohub': use the current hard-coded production voxel "
            "spacing exactly as-is."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    sample_dir = (
        NIS3D_ROOT
        / args.sample
    )

    if not sample_dir.exists():
        raise FileNotFoundError(
            f"NIS3D sample not found:\n"
            f"    {sample_dir}"
        )

    raw_path = find_file(
        sample_dir,
        RAW_NAMES,
    )

    gt_path = find_file(
        sample_dir,
        GT_NAMES,
    )

    confidence_path = find_file(
        sample_dir,
        CONFIDENCE_NAMES,
    )

    info = read_info(
        sample_dir
    )

    dataset_spacing = parse_voxel_spacing(
        info
    )

    if dataset_spacing is None:
        raise RuntimeError(
            "Could not determine NIS3D physical voxel spacing "
            "from Info.txt. Refusing to guess."
        )

    print()
    print("=" * 80)
    print(
        f"NIS3D preprocessing audit: {args.sample}"
    )
    print("=" * 80)

    print(
        f"Dataset directory:\n"
        f"    {sample_dir}"
    )

    print()
    print(
        "NIS3D physical spacing (Z,Y,X): "
        f"{dataset_spacing} um"
    )

    print(
        "Production default spacing (Z,Y,X): "
        f"{DEFAULT_PREPROCESSING_CONFIG.voxel_size_zyx_um} um"
    )

    # ---------------------------------------------------------
    # Choose spacing used by preprocessing.
    # ---------------------------------------------------------

    if args.spacing_mode == "dataset":

        processing_spacing = (
            dataset_spacing
        )

    else:

        processing_spacing = (
            DEFAULT_PREPROCESSING_CONFIG
            .voxel_size_zyx_um
        )

    print(
        f"Spacing USED by preprocessing: "
        f"{processing_spacing} um"
    )

    print()
    print(
        "Production preprocessing parameters:"
    )

    print(
        f"  percentile normalization : "
        f"{DEFAULT_PREPROCESSING_CONFIG.low_percentile}"
        f"–"
        f"{DEFAULT_PREPROCESSING_CONFIG.high_percentile}"
    )

    print(
        f"  denoise sigma            : "
        f"{DEFAULT_PREPROCESSING_CONFIG.denoise_sigma_um} um"
    )

    print(
        f"  background sigma         : "
        f"{DEFAULT_PREPROCESSING_CONFIG.background_sigma_um} um"
    )

    print(
        "  binary mask              : global Otsu"
    )

    # ---------------------------------------------------------
    # Load data.
    # ---------------------------------------------------------

    print()
    print("Loading volume...")

    raw = load_tiff(
        raw_path
    )

    gt = load_tiff(
        gt_path
    )

    confidence = load_tiff(
        confidence_path
    )

    if (
        raw.shape != gt.shape
        or raw.shape != confidence.shape
    ):
        raise RuntimeError(
            "Raw, GT, and confidence shapes do not match:\n"
            f"raw        = {raw.shape}\n"
            f"GT         = {gt.shape}\n"
            f"confidence = {confidence.shape}"
        )

    n_voxels = int(
        np.prod(raw.shape)
    )

    print()
    print(
        f"Shape        : {raw.shape}"
    )

    print(
        f"Raw dtype    : {raw.dtype}"
    )

    print(
        f"GT dtype     : {gt.dtype}"
    )

    print(
        f"Voxels       : {n_voxels:,}"
    )

    print(
        f"Max GT ID    : {int(np.max(gt)):,}"
    )

    approx_float32_gib = (
        n_voxels
        * 4
        / 1024**3
    )

    print(
        f"One float32 volume: "
        f"~{approx_float32_gib:.2f} GiB"
    )

    # ---------------------------------------------------------
    # Construct production config.
    #
    # SAME current preprocessing values.
    # Only physical spacing changes for NIS3D when requested.
    # ---------------------------------------------------------

    preprocessing_config = (
        PreprocessingConfig(
            low_percentile=(
                DEFAULT_PREPROCESSING_CONFIG
                .low_percentile
            ),
            high_percentile=(
                DEFAULT_PREPROCESSING_CONFIG
                .high_percentile
            ),
            denoise_sigma_um=(
                DEFAULT_PREPROCESSING_CONFIG
                .denoise_sigma_um
            ),
            background_sigma_um=(
                DEFAULT_PREPROCESSING_CONFIG
                .background_sigma_um
            ),
            voxel_size_zyx_um=(
                processing_spacing
            ),
        )
    )

    # ---------------------------------------------------------
    # Run CURRENT production preprocessing.
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "Running CURRENT production preprocessing..."
    )
    print("=" * 80)

    corrected, preprocessing_trace = (
        preprocess_volume(
            raw,
            config=preprocessing_config,
            return_diagnostics=True,
        )
    )

    normalized = (
        preprocessing_trace
        .intermediates["normalized"]
    )

    denoised = (
        preprocessing_trace
        .intermediates["denoised"]
    )

    background = (
        preprocessing_trace
        .intermediates["background"]
    )

    low_value = (
        preprocessing_trace
        .metrics["low_percentile_value"]
    )

    high_value = (
        preprocessing_trace
        .metrics["high_percentile_value"]
    )

    print(
        f"Raw percentile values:"
    )

    print(
        f"  low  = {low_value:.6g}"
    )

    print(
        f"  high = {high_value:.6g}"
    )

    # ---------------------------------------------------------
    # Run CURRENT production masking.
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "Running CURRENT production binary masking..."
    )
    print("=" * 80)

    binary_mask, masking_trace = (
        create_binary_mask(
            corrected,
            return_diagnostics=True,
        )
    )

    otsu = (
        masking_trace
        .metrics["base_otsu_threshold"]
    )

    effective_threshold = (
        masking_trace
        .metrics["effective_threshold"]
    )

    foreground_voxels = (
        masking_trace
        .metrics["foreground_voxels"]
    )

    print(
        f"Otsu threshold      : {otsu:.6f}"
    )

    print(
        f"Effective threshold : {effective_threshold:.6f}"
    )

    print(
        f"Foreground voxels   : {foreground_voxels:,}"
    )

    print(
        f"Foreground fraction : "
        f"{foreground_voxels / n_voxels:.4%}"
    )

    # ---------------------------------------------------------
    # Compare binary mask against NIS3D GT.
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "Binary-mask vs NIS3D ground truth"
    )
    print("=" * 80)

    metrics = evaluate_mask(
        binary_mask,
        gt,
        confidence,
    )

    print(
        f"GT foreground fraction        : "
        f"{metrics['gt_foreground_fraction']:.3%}"
    )

    print(
        f"Predicted foreground fraction : "
        f"{metrics['predicted_foreground_fraction']:.3%}"
    )

    print()

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"Dice      : {metrics['dice']:.4f}"
    )

    print(
        f"IoU       : {metrics['iou']:.4f}"
    )

    print(
        f"Specificity: "
        f"{metrics['specificity']:.4f}"
    )

    # ---------------------------------------------------------
    # Per-cell foreground retention.
    # ---------------------------------------------------------

    cell_metrics = evaluate_cell_coverage(
        binary_mask,
        gt,
        confidence,
    )

    print()
    print("=" * 80)
    print(
        "GT nucleus coverage by binary mask"
    )
    print("=" * 80)

    print(
        f"Evaluated nuclei       : "
        f"{cell_metrics['n_cells']:,}"
    )

    print(
        f"Median voxel coverage  : "
        f"{cell_metrics['median_coverage']:.2%}"
    )

    print(
        f"Cells >=95% retained   : "
        f"{cell_metrics['cells_ge_95']:.2%}"
    )

    print(
        f"Cells >=80% retained   : "
        f"{cell_metrics['cells_ge_80']:.2%}"
    )

    print(
        f"Cells >=50% retained   : "
        f"{cell_metrics['cells_ge_50']:.2%}"
    )

    print(
        f"Completely missed cells: "
        f"{cell_metrics['cells_zero']:.2%}"
    )

    # ---------------------------------------------------------
    # Error visualization.
    # ---------------------------------------------------------

    error_map = create_error_map(
        binary_mask,
        gt,
        confidence,
    )

    gt_foreground = (
        gt > 0
    )

    # ---------------------------------------------------------
    # Napari
    # ---------------------------------------------------------

    print()
    print(
        "Opening Napari..."
    )

    viewer = napari.Viewer(
        title=(
            f"NIS3D preprocessing audit — "
            f"{args.sample}"
        ),
        ndisplay=3,
    )

    # IMPORTANT:
    # Display geometry always uses the TRUE NIS3D spacing,
    # regardless of the spacing-mode used as a processing test.

    viewer.add_image(
        raw,
        name="01 | RAW",
        scale=dataset_spacing,
        colormap="gray",
        contrast_limits=raw_contrast_limits(
            raw
        ),
        visible=False,
    )

    viewer.add_image(
        normalized,
        name="02 | NORMALIZED",
        scale=dataset_spacing,
        colormap="gray",
        contrast_limits=(
            0.0,
            1.0,
        ),
        visible=False,
    )

    viewer.add_image(
        denoised,
        name="03 | DENOISED",
        scale=dataset_spacing,
        colormap="gray",
        contrast_limits=(
            0.0,
            1.0,
        ),
        visible=False,
    )

    viewer.add_image(
        background,
        name="04 | ESTIMATED BACKGROUND",
        scale=dataset_spacing,
        colormap="gray",
        visible=False,
    )

    viewer.add_image(
        corrected,
        name="05 | BACKGROUND CORRECTED",
        scale=dataset_spacing,
        colormap="gray",
        contrast_limits=(
            0.0,
            1.0,
        ),
        visible=True,
    )

    viewer.add_labels(
        binary_mask.astype(
            np.uint8
        ),
        name="06 | PRODUCTION BINARY MASK",
        scale=dataset_spacing,
        opacity=0.40,
        visible=True,
    )

    viewer.add_labels(
        gt,
        name="07 | GT INSTANCES",
        scale=dataset_spacing,
        opacity=0.45,
        visible=False,
    )

    viewer.add_labels(
        gt_foreground.astype(
            np.uint8
        ),
        name="08 | GT FOREGROUND",
        scale=dataset_spacing,
        opacity=0.40,
        visible=False,
    )

    viewer.add_image(
        confidence,
        name="09 | GT CONFIDENCE",
        scale=dataset_spacing,
        colormap="turbo",
        contrast_limits=(
            0,
            4,
        ),
        opacity=0.65,
        visible=False,
    )

    viewer.add_labels(
        error_map,
        name=(
            "10 | MASK ERROR "
            "[1=TP, 2=MISS, 3=FALSE+]"
        ),
        scale=dataset_spacing,
        opacity=0.75,
        visible=False,
    )

    viewer.reset_view()

    print()
    print("=" * 80)
    print("What to inspect")
    print("=" * 80)

    print(
        """
01 RAW
    Original NIS3D fluorescence.

02 NORMALIZED
    Result of current 1–99.5 percentile normalization.

03 DENOISED
    Current 0.8 µm Gaussian denoising.

04 ESTIMATED BACKGROUND
    What the current 4.0 µm Gaussian believes is background.

05 BACKGROUND CORRECTED
    The actual image given to the masking stage.

06 PRODUCTION BINARY MASK
    Current global-Otsu foreground result.

07 GT INSTANCES
    NIS3D instance ground truth.

08 GT FOREGROUND
    Ground-truth binary foreground.

09 GT CONFIDENCE
    NIS3D annotation confidence.

10 MASK ERROR
    Green   = correctly retained GT foreground
    Red     = GT foreground LOST by preprocessing/masking
    Magenta = foreground produced outside GT
"""
    )

    print(
        "Most important comparison:"
    )

    print(
        "    05 BACKGROUND CORRECTED"
    )

    print(
        "    06 PRODUCTION BINARY MASK"
    )

    print(
        "    07 GT INSTANCES"
    )

    print(
        "    10 MASK ERROR"
    )

    print()

    napari.run()


if __name__ == "__main__":
    main()