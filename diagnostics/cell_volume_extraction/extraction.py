"""Reusable 3D cell-volume extraction utilities.

This module supports both:

1. Standalone batch extraction by editing the configuration at the top and
   running this file directly.
2. Interactive extraction from Napari through ``napari_extractor.py``.

Saved outputs are written to ``data/extracted/merged_cells`` by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ============================================================
# Standalone configuration
# ============================================================

SAMPLE_ID = "44b6_0113de3b"
FRAME = 0
CELL_IDS = [1, 2, 3]
BOX_SIZE = (17, 65, 65)  # (Z, Y, X), in voxels
VOXEL_SIZE = (1.625, 0.40625, 0.40625)  # (Z, Y, X)
PAD_VALUE = 0
OVERWRITE = False


# ============================================================
# Project paths
# ============================================================

# Expected location:
#   <project-root>/diagnostics/cell_volume_extraction/extraction.py
def _detect_project_root() -> Path:
    module_path = Path(__file__).resolve()

    for parent in module_path.parents:
        if (parent / "data").exists() and (parent / "diagnostics").exists():
            return parent

    # Normal repository layout, even before the data directory exists.
    if len(module_path.parents) >= 3:
        return module_path.parents[2]

    return Path.cwd().resolve()


PROJECT_ROOT = _detect_project_root()
DATA_ROOT = PROJECT_ROOT / "data"

CELLS_DIR = (
    DATA_ROOT
    / "sample"
    / "processed"
    / "stage_7_processed_dataset"
    / SAMPLE_ID
    / "cells"
)

ZARR_ARRAY_PATH = (
    DATA_ROOT
    / "sample"
    / "biohub_5samples_20timepoints"
    / "train"
    / SAMPLE_ID
    / f"{SAMPLE_ID}.zarr"
    / "0"
)

OUTPUT_DIR = DATA_ROOT / "extracted" / "merged_cells"


# ============================================================
# Validation and lookup
# ============================================================

def validate_box_size(box_size: Sequence[int]) -> np.ndarray:
    """Validate a ``(Z, Y, X)`` box size and return an integer array."""

    size = np.asarray(box_size, dtype=int)

    if size.shape != (3,):
        raise ValueError(
            "box_size must contain exactly three values in (Z, Y, X) order; "
            f"received {tuple(box_size)}."
        )

    if np.any(size <= 0):
        raise ValueError(
            f"All box dimensions must be positive; received {tuple(size)}."
        )

    return size


def validate_image_volume(image_volume: Any) -> None:
    """Validate that the image volume uses ``(T, Z, Y, X)`` ordering."""

    if not hasattr(image_volume, "shape"):
        raise TypeError("image_volume must expose a shape attribute.")

    if len(image_volume.shape) != 4:
        raise ValueError(
            "Expected image_volume with shape (T, Z, Y, X), "
            f"but found {image_volume.shape}."
        )


def validate_cell_table(cells: pd.DataFrame) -> None:
    """Validate columns needed for frame/cell lookup and extraction."""

    required = {
        "cell_id",
        "centroid_z",
        "centroid_y",
        "centroid_x",
    }
    missing = required.difference(cells.columns)

    if missing:
        raise KeyError(f"Cell table is missing required columns: {sorted(missing)}")


def find_cell(cells: pd.DataFrame, frame: int, cell_id: int) -> pd.Series:
    """Return exactly one cell identified by ``frame`` and ``cell_id``.

    If the table contains a ``frame`` column, both fields are used. If it does
    not, the table is assumed to already contain only the requested frame.
    """

    validate_cell_table(cells)

    matches = cells.loc[cells["cell_id"].astype(int) == int(cell_id)]

    if "frame" in cells.columns:
        matches = matches.loc[matches["frame"].astype(int) == int(frame)]
        frame_cells = cells.loc[cells["frame"].astype(int) == int(frame)]
    else:
        frame_cells = cells

    if matches.empty:
        available_ids = (
            frame_cells["cell_id"].astype(int).sort_values().drop_duplicates().tolist()
        )
        preview = available_ids[:30]
        suffix = " ..." if len(available_ids) > 30 else ""
        raise KeyError(
            f"Cell ID {cell_id} was not found in frame {frame}. "
            f"Available IDs: {preview}{suffix}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Found {len(matches)} rows for cell ID {cell_id} in frame {frame}; "
            "expected exactly one."
        )

    return matches.iloc[0]


# ============================================================
# Bounds and extraction
# ============================================================

def calculate_box_bounds(
    centroid_zyx: Sequence[float],
    box_size: Sequence[int],
    spatial_shape_zyx: Sequence[int],
) -> dict[str, list[int]]:
    """Calculate requested, clipped, and padded bounds for one crop."""

    size = validate_box_size(box_size)
    centroid = np.asarray(centroid_zyx, dtype=float)
    spatial_shape = np.asarray(spatial_shape_zyx, dtype=int)

    if centroid.shape != (3,):
        raise ValueError("centroid_zyx must contain exactly three values.")

    if spatial_shape.shape != (3,) or np.any(spatial_shape <= 0):
        raise ValueError("spatial_shape_zyx must contain three positive values.")

    centroid_index = np.rint(centroid).astype(int)
    requested_start = centroid_index - size // 2
    requested_stop = requested_start + size

    clipped_start = np.maximum(requested_start, 0)
    clipped_stop = np.minimum(requested_stop, spatial_shape)

    if np.any(clipped_start >= clipped_stop):
        raise ValueError(
            "The requested box does not overlap the image volume. "
            f"Centroid index: {centroid_index.tolist()}"
        )

    padding_before = np.maximum(-requested_start, 0)
    padding_after = np.maximum(requested_stop - spatial_shape, 0)

    return {
        "centroid_index_zyx": centroid_index.tolist(),
        "requested_start_zyx": requested_start.tolist(),
        "requested_stop_zyx": requested_stop.tolist(),
        "clipped_start_zyx": clipped_start.tolist(),
        "clipped_stop_zyx": clipped_stop.tolist(),
        "padding_before_zyx": padding_before.tolist(),
        "padding_after_zyx": padding_after.tolist(),
    }


def extract_fixed_box(
    image_volume: Any,
    frame: int,
    centroid_zyx: Sequence[float],
    box_size: Sequence[int],
    pad_value: int | float = 0,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Extract a fixed-size box from a ``(T, Z, Y, X)`` image volume.

    Any part outside the image is padded so the returned array always has the
    requested shape.
    """

    validate_image_volume(image_volume)
    frame = int(frame)

    if not 0 <= frame < image_volume.shape[0]:
        raise IndexError(
            f"Frame {frame} is outside the valid range "
            f"0 to {image_volume.shape[0] - 1}."
        )

    size = validate_box_size(box_size)
    bounds = calculate_box_bounds(
        centroid_zyx=centroid_zyx,
        box_size=size,
        spatial_shape_zyx=image_volume.shape[1:],
    )

    requested_start = np.asarray(bounds["requested_start_zyx"], dtype=int)
    clipped_start = np.asarray(bounds["clipped_start_zyx"], dtype=int)
    clipped_stop = np.asarray(bounds["clipped_stop_zyx"], dtype=int)

    source_slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(clipped_start, clipped_stop)
    )

    destination_start = clipped_start - requested_start
    destination_stop = destination_start + clipped_stop - clipped_start
    destination_slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(destination_start, destination_stop)
    )

    crop = np.full(tuple(size), pad_value, dtype=image_volume.dtype)
    crop[destination_slices] = np.asarray(image_volume[(frame, *source_slices)])

    return crop, bounds


# ============================================================
# Metadata and saving
# ============================================================

def to_json_value(value: Any) -> Any:
    """Convert pandas/NumPy/path values into JSON-compatible values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)

    return value


def _display_path(path: Path | str | None) -> str | None:
    if path is None:
        return None

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def build_output_paths(
    sample_id: str,
    frame: int,
    cell_id: int,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Build the ``.npy`` and ``.json`` paths for one extraction."""

    directory = Path(output_dir)
    stem = f"{sample_id}_t{int(frame):04d}_cell{int(cell_id):05d}"
    return directory / f"{stem}.npy", directory / f"{stem}.json"


def make_metadata(
    *,
    sample_id: str,
    row: pd.Series,
    frame: int,
    cell_id: int,
    centroid_zyx: np.ndarray,
    box_size: np.ndarray,
    bounds: dict[str, list[int]],
    voxel_size_zyx: Sequence[float],
    output_file: Path,
    source_cell_file: Path | str | None = None,
    source_zarr_array: Path | str | None = None,
) -> dict[str, Any]:
    """Create metadata for one saved cell volume."""

    cell_features = {
        str(column): to_json_value(value)
        for column, value in row.items()
    }

    metadata: dict[str, Any] = {
        "sample_id": str(sample_id),
        "frame": int(frame),
        "cell_id": int(cell_id),
        "output_volume": _display_path(output_file),
        "centroid_zyx": centroid_zyx.tolist(),
        "centroid_index_zyx": bounds["centroid_index_zyx"],
        "box_size_zyx": box_size.tolist(),
        "voxel_size_zyx": [float(v) for v in voxel_size_zyx],
        "requested_start_zyx": bounds["requested_start_zyx"],
        "requested_stop_zyx": bounds["requested_stop_zyx"],
        "clipped_start_zyx": bounds["clipped_start_zyx"],
        "clipped_stop_zyx": bounds["clipped_stop_zyx"],
        "padding_before_zyx": bounds["padding_before_zyx"],
        "padding_after_zyx": bounds["padding_after_zyx"],
        "cell_features": cell_features,
    }

    if source_cell_file is not None:
        metadata["source_cell_file"] = _display_path(source_cell_file)
    if source_zarr_array is not None:
        metadata["source_zarr_array"] = _display_path(source_zarr_array)

    return metadata


def update_manifest(
    record: dict[str, Any],
    output_dir: Path | str = OUTPUT_DIR,
) -> Path:
    """Insert or replace one row in ``manifest.csv``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.csv"
    new_row = pd.DataFrame([record])

    if manifest_path.exists():
        manifest = pd.concat([pd.read_csv(manifest_path), new_row], ignore_index=True)
    else:
        manifest = new_row

    manifest = manifest.drop_duplicates(
        subset=["sample_id", "frame", "cell_id"],
        keep="last",
    ).sort_values(["sample_id", "frame", "cell_id"])

    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def save_cell_extraction(
    *,
    image_volume: Any,
    cells: pd.DataFrame,
    sample_id: str,
    frame: int,
    cell_id: int,
    box_size: Sequence[int] = BOX_SIZE,
    output_dir: Path | str = OUTPUT_DIR,
    voxel_size_zyx: Sequence[float] = VOXEL_SIZE,
    pad_value: int | float = PAD_VALUE,
    overwrite: bool = False,
    source_cell_file: Path | str | None = None,
    source_zarr_array: Path | str | None = None,
) -> tuple[Path, Path, np.ndarray, dict[str, Any]]:
    """Extract and save one cell volume plus metadata and manifest entry."""

    validate_image_volume(image_volume)
    size = validate_box_size(box_size)
    frame = int(frame)
    cell_id = int(cell_id)
    row = find_cell(cells=cells, frame=frame, cell_id=cell_id)

    centroid_zyx = row[
        ["centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=float)

    volume_path, metadata_path = build_output_paths(
        sample_id=sample_id,
        frame=frame,
        cell_id=cell_id,
        output_dir=output_dir,
    )

    if (volume_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Extraction already exists for frame {frame}, cell {cell_id}: "
            f"{volume_path.name}. Enable overwrite to replace it."
        )

    crop, bounds = extract_fixed_box(
        image_volume=image_volume,
        frame=frame,
        centroid_zyx=centroid_zyx,
        box_size=size,
        pad_value=pad_value,
    )

    volume_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(volume_path, crop, allow_pickle=False)

    metadata = make_metadata(
        sample_id=sample_id,
        row=row,
        frame=frame,
        cell_id=cell_id,
        centroid_zyx=centroid_zyx,
        box_size=size,
        bounds=bounds,
        voxel_size_zyx=voxel_size_zyx,
        output_file=volume_path,
        source_cell_file=source_cell_file,
        source_zarr_array=source_zarr_array,
    )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    update_manifest(
        {
            "sample_id": str(sample_id),
            "frame": frame,
            "cell_id": cell_id,
            "centroid_z": float(centroid_zyx[0]),
            "centroid_y": float(centroid_zyx[1]),
            "centroid_x": float(centroid_zyx[2]),
            "box_depth": int(size[0]),
            "box_height": int(size[1]),
            "box_width": int(size[2]),
            "volume_file": volume_path.name,
            "metadata_file": metadata_path.name,
        },
        output_dir=output_dir,
    )

    return volume_path, metadata_path, crop, metadata


# ============================================================
# Standalone loading and batch execution
# ============================================================

def get_cell_file(frame: int, cells_dir: Path | str = CELLS_DIR) -> Path:
    """Return the per-frame cell CSV using sorted ``t*.csv`` ordering."""

    cell_files = sorted(Path(cells_dir).glob("t*.csv"))

    if not cell_files:
        raise FileNotFoundError(f"No cell CSV files were found in {cells_dir}")
    if not 0 <= int(frame) < len(cell_files):
        raise IndexError(
            f"Frame {frame} is outside the available range 0 to {len(cell_files) - 1}."
        )

    return cell_files[int(frame)]


def load_frame_cells(
    frame: int,
    cells_dir: Path | str = CELLS_DIR,
) -> tuple[pd.DataFrame, Path]:
    """Load one frame's cell CSV and add its zero-based frame column."""

    cell_file = get_cell_file(frame=frame, cells_dir=cells_dir)
    cells = pd.read_csv(cell_file)
    cells["frame"] = int(frame)
    validate_cell_table(cells)
    return cells, cell_file


def load_original_volume(zarr_array_path: Path | str = ZARR_ARRAY_PATH):
    """Open the original Zarr array without loading it fully into RAM."""

    path = Path(zarr_array_path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr array was not found at {path}")

    import zarr

    volume = zarr.open_array(str(path), mode="r")
    validate_image_volume(volume)
    return volume


def extract_selected_cells() -> None:
    """Standalone batch extraction using the configuration above."""

    if not CELL_IDS:
        raise ValueError("CELL_IDS is empty.")
    if len(CELL_IDS) != len(set(CELL_IDS)):
        raise ValueError("CELL_IDS contains duplicate IDs.")

    frame = int(FRAME)
    cells, cell_file = load_frame_cells(frame)
    image_volume = load_original_volume()

    print(f"Sample:     {SAMPLE_ID}")
    print(f"Frame:      {frame}")
    print(f"Box size:   {tuple(validate_box_size(BOX_SIZE))}")
    print(f"Output:     {OUTPUT_DIR}")
    print()

    saved = skipped = failed = 0

    for cell_id in CELL_IDS:
        try:
            volume_path, _, crop, _ = save_cell_extraction(
                image_volume=image_volume,
                cells=cells,
                sample_id=SAMPLE_ID,
                frame=frame,
                cell_id=int(cell_id),
                box_size=BOX_SIZE,
                output_dir=OUTPUT_DIR,
                voxel_size_zyx=VOXEL_SIZE,
                pad_value=PAD_VALUE,
                overwrite=OVERWRITE,
                source_cell_file=cell_file,
                source_zarr_array=ZARR_ARRAY_PATH,
            )
            print(f"[SAVED]   Cell {cell_id}: {volume_path.name} | shape={crop.shape}")
            saved += 1
        except FileExistsError as error:
            print(f"[SKIPPED] Cell {cell_id}: {error}")
            skipped += 1
        except Exception as error:
            print(f"[FAILED]  Cell {cell_id}: {type(error).__name__}: {error}")
            failed += 1

    print()
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    extract_selected_cells()
