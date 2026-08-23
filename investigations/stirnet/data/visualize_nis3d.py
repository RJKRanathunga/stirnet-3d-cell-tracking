"""
Visualize the six canonical NIS3D datasets in Napari.

Run from project root:

    python investigations/stirnet/data/visualize_nis3d.py

One dataset only:

    python investigations/stirnet/data/visualize_nis3d.py \
        --sample Zebrafish_2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import napari
import numpy as np
import tifffile


# ---------------------------------------------------------------------
# Paths
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
# NIS3D filename discovery
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
    Find one file while tolerating inconsistent capitalization/naming.
    """

    # First try exact names.
    for name in candidates:
        path = directory / name

        if path.exists():
            return path

    # Then do a case-insensitive lookup.
    files = {
        p.name.lower(): p
        for p in directory.iterdir()
        if p.is_file()
    }

    for name in candidates:
        result = files.get(name.lower())

        if result is not None:
            return result

    available = "\n".join(
        f"    {p.name}"
        for p in directory.iterdir()
        if p.is_file()
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
# TIFF loading
# ---------------------------------------------------------------------

def open_tiff(path: Path) -> np.ndarray:
    """
    Prefer direct TIFF memory mapping.

    If the TIFF is compressed/tiled in a way that prevents direct
    mapping, tifffile decompresses it into a temporary disk-backed
    memmap instead of keeping the entire array permanently in RAM.
    """

    try:
        array = tifffile.memmap(path)

        print(
            f"  [mmap]      {path.name}"
        )

        return array

    except Exception:
        print(
            f"  [disk mmap] {path.name}"
        )

        return tifffile.imread(
            path,
            out="memmap",
        )


# ---------------------------------------------------------------------
# Info / physical spacing
# ---------------------------------------------------------------------

def read_info(sample_dir: Path) -> str:
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
    Extract physical spacing from NIS3D Info.txt.

    NIS3D commonly uses strings such as:

        Resolution:
        1 um x 1 um x 1 um

    or:

        voxel size is 0.43 um x 0.43 um x 2.5 um

    The textual order is assumed to be X, Y, Z.

    Napari expects:
        Z, Y, X
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
# Display helpers
# ---------------------------------------------------------------------

def robust_contrast_limits(
    image: np.ndarray,
) -> tuple[float, float]:

    if image.ndim != 3:
        values = np.asarray(image)

    else:
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

        values = np.asarray(
            image[
                ::z_step,
                ::y_step,
                ::x_step,
            ]
        )

    nonzero = values[
        values > 0
    ]

    if nonzero.size > 100:
        values = nonzero

    low, high = np.percentile(
        values,
        [0.5, 99.8],
    )

    if high <= low:
        low = float(
            np.min(values)
        )
        high = float(
            np.max(values)
        )

    return (
        float(low),
        float(high),
    )


def add_sample(
    viewer: napari.Viewer,
    sample_name: str,
    visible: bool,
) -> None:

    sample_dir = (
        NIS3D_ROOT
        / sample_name
    )

    if not sample_dir.exists():
        raise FileNotFoundError(
            sample_dir
        )

    print()
    print("=" * 72)
    print(sample_name)
    print("=" * 72)

    # ---------------------------------------------------------
    # Resolve inconsistent NIS3D filenames.
    # ---------------------------------------------------------

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

    print(
        f"Raw file        : {raw_path.name}"
    )
    print(
        f"GT file         : {gt_path.name}"
    )
    print(
        f"Confidence file : {confidence_path.name}"
    )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    raw = open_tiff(
        raw_path
    )

    gt = open_tiff(
        gt_path
    )

    confidence = open_tiff(
        confidence_path
    )

    print(
        f"Raw          shape={raw.shape}, "
        f"dtype={raw.dtype}"
    )

    print(
        f"Ground truth shape={gt.shape}, "
        f"dtype={gt.dtype}"
    )

    print(
        f"Confidence   shape={confidence.shape}, "
        f"dtype={confidence.dtype}"
    )

    if raw.shape != gt.shape:
        raise ValueError(
            f"{sample_name}: "
            f"raw shape {raw.shape} != "
            f"GT shape {gt.shape}"
        )

    if raw.shape != confidence.shape:
        raise ValueError(
            f"{sample_name}: "
            f"raw shape {raw.shape} != "
            f"confidence shape {confidence.shape}"
        )

    # ---------------------------------------------------------
    # Physical spacing
    # ---------------------------------------------------------

    info = read_info(
        sample_dir
    )

    spacing = parse_voxel_spacing(
        info
    )

    if spacing is None:

        spacing = (
            1.0,
            1.0,
            1.0,
        )

        print(
            "Voxel spacing : could not parse "
            "Info.txt; using (1,1,1)"
        )

    else:

        print(
            "Voxel spacing : "
            f"z={spacing[0]:g}, "
            f"y={spacing[1]:g}, "
            f"x={spacing[2]:g} um"
        )

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    # GT labels are expected to be contiguous-ish integer instance IDs.
    # max() is cheap enough and avoids np.unique over enormous volumes.
    max_label = int(
        np.max(gt)
    )

    print(
        f"Maximum GT instance ID: "
        f"{max_label:,}"
    )

    # ---------------------------------------------------------
    # Add Napari layers
    # ---------------------------------------------------------

    viewer.add_image(
        raw,
        name=f"{sample_name} | RAW",
        scale=spacing,
        colormap="gray",
        contrast_limits=robust_contrast_limits(
            raw
        ),
        visible=visible,
    )

    viewer.add_labels(
        gt,
        name=f"{sample_name} | GT INSTANCES",
        scale=spacing,
        opacity=0.55,
        visible=visible,
    )

    viewer.add_image(
        confidence,
        name=f"{sample_name} | CONFIDENCE",
        scale=spacing,
        colormap="turbo",
        contrast_limits=(
            0,
            4,
        ),
        opacity=0.65,
        blending="translucent",
        visible=False,
    )

    # ---------------------------------------------------------
    # Console metadata
    # ---------------------------------------------------------

    if info:

        print()
        print("Info.txt:")
        print("-" * 72)
        print(
            info.strip()
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        choices=SAMPLES,
        default=None,
        help=(
            "Load one dataset only. "
            "Default: load all six."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    print(
        f"NIS3D root:\n"
        f"    {NIS3D_ROOT}"
    )

    if not NIS3D_ROOT.exists():

        raise FileNotFoundError(
            f"NIS3D directory not found:\n"
            f"{NIS3D_ROOT}"
        )

    if args.sample is None:

        samples = SAMPLES

    else:

        samples = (
            args.sample,
        )

    viewer = napari.Viewer(
        title="NIS3D Dataset Inspection",
        ndisplay=3,
    )

    # Only the first dataset is visible initially.
    # Otherwise six unrelated coordinate systems would overlap.
    for index, sample in enumerate(
        samples
    ):

        add_sample(
            viewer,
            sample,
            visible=(index == 0),
        )

    viewer.reset_view()

    print()
    print("=" * 72)
    print("NIS3D loaded")
    print("=" * 72)

    print(
        "\nConfidence map:"
        "\n  0 = background"
        "\n  1 = undefined/unreliable"
        "\n  2 = 1/3 annotators"
        "\n  3 = 2/3 annotators"
        "\n  4 = 3/3 annotators"
    )

    print(
        "\nNapari:"
        "\n  2 → 2D view"
        "\n  3 → 3D view"
        "\n\nEnable only the RAW/GT/CONFIDENCE layers "
        "for the dataset you currently want to inspect."
    )

    napari.run()


if __name__ == "__main__":
    main()