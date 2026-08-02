"""Reusable aligned 3D cell-volume extraction utilities.

The extractor saves a fixed-size crop around a detected cell centroid.  The
same spatial bounds are applied to every supplied pipeline representation, so
raw intensity, preprocessed intensity, the binary mask, and watershed instance
labels remain voxel-aligned.

The raw crop keeps the original backward-compatible filename::

    <sample>_t####_cell#####.npy

Optional diagnostic crops use explicit suffixes::

    <sample>_t####_cell#####_preprocessed.npy
    <sample>_t####_cell#####_binary_mask.npy
    <sample>_t####_cell#####_instance_labels.npy
    <sample>_t####_cell#####.json

This module supports both standalone batch extraction and interactive use from
``napari_extractor.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.io import PipelinePaths


# ============================================================
# Standalone configuration
# ============================================================

SAMPLE_ID = "44b6_0113de3b"
FRAME = 0
CELL_IDS = [1, 2, 3]
BOX_SIZE = (12, 50, 50)  # (Z, Y, X), in voxels
VOXEL_SIZE = (1.625, 0.40625, 0.40625)  # (Z, Y, X)
PAD_VALUE = 0
OVERWRITE = False

# Optional standalone diagnostic sources.  Set these to a 3D array for FRAME
# or a 4D (T, Z, Y, X) array.  Leave as None when using the Napari widget.
PREPROCESSED_ARRAY_PATH: Path | None = None
BINARY_MASK_ARRAY_PATH: Path | None = None
INSTANCE_LABELS_ARRAY_PATH: Path | None = None


# ============================================================
# Project paths
# ============================================================


PROJECT_ROOT = PipelinePaths.discover().project_root
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
    """Validate a ``(Z, Y, X)`` crop size and return an integer array."""

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


def validate_image_volume(image_volume: Any, *, name: str = "image_volume") -> None:
    """Validate a primary image source with ``(T, Z, Y, X)`` ordering."""

    if not hasattr(image_volume, "shape"):
        raise TypeError(f"{name} must expose a shape attribute.")

    if len(image_volume.shape) != 4:
        raise ValueError(
            f"Expected {name} with shape (T, Z, Y, X), "
            f"but found {image_volume.shape}."
        )


def validate_aligned_volume(
    volume: Any,
    *,
    reference_volume: Any,
    name: str,
) -> None:
    """Validate an optional aligned 3D or 4D diagnostic representation.

    A 3D source is interpreted as the data for the selected frame.  A 4D
    source must be aligned with the primary ``(T, Z, Y, X)`` image volume.
    """

    if not hasattr(volume, "shape"):
        raise TypeError(f"{name} must expose a shape attribute.")

    shape = tuple(int(value) for value in volume.shape)
    reference_shape = tuple(int(value) for value in reference_volume.shape)

    if len(shape) == 3:
        if shape != reference_shape[1:]:
            raise ValueError(
                f"{name} has spatial shape {shape}, but the raw volume has "
                f"spatial shape {reference_shape[1:]}."
            )
        return

    if len(shape) == 4:
        if shape != reference_shape:
            raise ValueError(
                f"{name} has shape {shape}, but the raw volume has shape "
                f"{reference_shape}."
            )
        return

    raise ValueError(
        f"{name} must be either 3D (Z, Y, X) or 4D (T, Z, Y, X); "
        f"found shape {shape}."
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
    """Return exactly one cell identified by ``frame`` and ``cell_id``."""

    validate_cell_table(cells)

    matches = cells.loc[cells["cell_id"].astype(int) == int(cell_id)]

    if "frame" in cells.columns:
        matches = matches.loc[matches["frame"].astype(int) == int(frame)]
        frame_cells = cells.loc[cells["frame"].astype(int) == int(frame)]
    else:
        frame_cells = cells

    if matches.empty:
        available_ids = (
            frame_cells["cell_id"]
            .astype(int)
            .sort_values()
            .drop_duplicates()
            .tolist()
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
# Bounds and aligned extraction
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


def extract_box_with_bounds(
    volume: Any,
    *,
    frame: int,
    box_size: Sequence[int],
    bounds: Mapping[str, Sequence[int]],
    pad_value: int | float | bool = 0,
) -> np.ndarray:
    """Extract one aligned 3D crop using already-calculated bounds.

    ``volume`` may be either a 4D ``(T, Z, Y, X)`` source or a 3D source for
    the selected frame.
    """

    if not hasattr(volume, "shape"):
        raise TypeError("volume must expose a shape attribute.")

    size = validate_box_size(box_size)
    ndim = len(volume.shape)

    if ndim not in (3, 4):
        raise ValueError(
            "volume must have shape (Z, Y, X) or (T, Z, Y, X); "
            f"found {volume.shape}."
        )

    frame = int(frame)
    if ndim == 4 and not 0 <= frame < volume.shape[0]:
        raise IndexError(
            f"Frame {frame} is outside the valid range 0 to {volume.shape[0] - 1}."
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

    dtype = getattr(volume, "dtype", np.asarray(volume).dtype)
    crop = np.full(tuple(size), pad_value, dtype=dtype)

    if ndim == 4:
        source_data = np.asarray(volume[(frame, *source_slices)])
    else:
        source_data = np.asarray(volume[source_slices])

    crop[destination_slices] = source_data
    return crop


def extract_fixed_box(
    image_volume: Any,
    frame: int,
    centroid_zyx: Sequence[float],
    box_size: Sequence[int],
    pad_value: int | float = 0,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Extract a fixed-size box from a ``(T, Z, Y, X)`` image volume."""

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

    crop = extract_box_with_bounds(
        image_volume,
        frame=frame,
        box_size=size,
        bounds=bounds,
        pad_value=pad_value,
    )

    return crop, bounds


# ============================================================
# Metadata and output paths
# ============================================================


def to_json_value(value: Any) -> Any:
    """Convert pandas, NumPy, and path values into JSON-compatible values."""

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


def build_artifact_paths(
    sample_id: str,
    frame: int,
    cell_id: int,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict[str, Path]:
    """Build all possible output paths for one diagnostic case."""

    directory = Path(output_dir)
    stem = f"{sample_id}_t{int(frame):04d}_cell{int(cell_id):05d}"

    return {
        # Backward-compatible raw filename used by the existing notebook.
        "raw": directory / f"{stem}.npy",
        "preprocessed": directory / f"{stem}_preprocessed.npy",
        "binary_mask": directory / f"{stem}_binary_mask.npy",
        "instance_labels": directory / f"{stem}_instance_labels.npy",
        "metadata": directory / f"{stem}.json",
    }


def build_output_paths(
    sample_id: str,
    frame: int,
    cell_id: int,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Backward-compatible helper returning raw and metadata paths."""

    paths = build_artifact_paths(sample_id, frame, cell_id, output_dir)
    return paths["raw"], paths["metadata"]


def _summarize_crop(name: str, crop: np.ndarray, *, cell_id: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": [int(value) for value in crop.shape],
        "dtype": str(crop.dtype),
    }

    if crop.size == 0:
        return summary

    if name == "binary_mask":
        summary["foreground_voxels"] = int(np.count_nonzero(crop))
        summary["foreground_fraction"] = float(np.count_nonzero(crop) / crop.size)
    elif name == "instance_labels":
        labels = np.unique(crop)
        nonzero = labels[labels != 0]
        summary["nonzero_label_count"] = int(len(nonzero))
        summary["nonzero_labels"] = [int(value) for value in nonzero.tolist()]
        summary["selected_cell_id_present_as_label"] = bool(np.any(crop == cell_id))
        summary["selected_cell_id_voxels"] = int(np.count_nonzero(crop == cell_id))
    else:
        summary["minimum"] = float(np.min(crop))
        summary["maximum"] = float(np.max(crop))
        summary["mean"] = float(np.mean(crop))

    return summary


def make_metadata(
    *,
    sample_id: str,
    row: pd.Series,
    frame: int,
    cell_id: int,
    centroid_zyx: np.ndarray,
    box_size: np.ndarray,
    bounds: Mapping[str, Sequence[int]],
    voxel_size_zyx: Sequence[float],
    output_files: Mapping[str, Path],
    artifact_summaries: Mapping[str, Mapping[str, Any]],
    source_cell_file: Path | str | None = None,
    source_zarr_array: Path | str | None = None,
    source_preprocessed: Path | str | None = None,
    source_binary_mask: Path | str | None = None,
    source_instance_labels: Path | str | None = None,
) -> dict[str, Any]:
    """Create metadata for one aligned diagnostic extraction."""

    cell_features = {
        str(column): to_json_value(value)
        for column, value in row.items()
    }

    displayed_outputs = {
        name: _display_path(path)
        for name, path in output_files.items()
    }

    metadata: dict[str, Any] = {
        "sample_id": str(sample_id),
        "frame": int(frame),
        "cell_id": int(cell_id),
        # Retained for backward compatibility.
        "output_volume": displayed_outputs.get("raw"),
        "output_files": displayed_outputs,
        "available_artifacts": sorted(displayed_outputs),
        "artifact_summaries": dict(artifact_summaries),
        "centroid_zyx": centroid_zyx.tolist(),
        "centroid_index_zyx": list(bounds["centroid_index_zyx"]),
        "box_size_zyx": box_size.tolist(),
        "voxel_size_zyx": [float(value) for value in voxel_size_zyx],
        "requested_start_zyx": list(bounds["requested_start_zyx"]),
        "requested_stop_zyx": list(bounds["requested_stop_zyx"]),
        "clipped_start_zyx": list(bounds["clipped_start_zyx"]),
        "clipped_stop_zyx": list(bounds["clipped_stop_zyx"]),
        "padding_before_zyx": list(bounds["padding_before_zyx"]),
        "padding_after_zyx": list(bounds["padding_after_zyx"]),
        "cell_features": cell_features,
    }

    source_paths = {
        "cell_table": source_cell_file,
        "raw_zarr_array": source_zarr_array,
        "preprocessed": source_preprocessed,
        "binary_mask": source_binary_mask,
        "instance_labels": source_instance_labels,
    }
    metadata["source_files"] = {
        name: _display_path(path)
        for name, path in source_paths.items()
        if path is not None
    }

    return metadata


# ============================================================
# Saving
# ============================================================


def update_manifest(
    record: Mapping[str, Any],
    output_dir: Path | str = OUTPUT_DIR,
) -> Path:
    """Insert or replace one row in ``manifest.csv``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.csv"
    new_row = pd.DataFrame([dict(record)])

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
    # Optional aligned pipeline representations.
    preprocessed_volume: Any | None = None,
    binary_mask_volume: Any | None = None,
    instance_labels_volume: Any | None = None,
    # Optional source-path provenance.
    source_cell_file: Path | str | None = None,
    source_zarr_array: Path | str | None = None,
    source_preprocessed: Path | str | None = None,
    source_binary_mask: Path | str | None = None,
    source_instance_labels: Path | str | None = None,
) -> tuple[Path, Path, np.ndarray, dict[str, Any]]:
    """Extract and save one aligned diagnostic cell case.

    The same requested/clipped bounds are used for every supplied volume.  The
    returned tuple is intentionally backward compatible:

    ``(raw_path, metadata_path, raw_crop, metadata)``.
    """

    validate_image_volume(image_volume)
    size = validate_box_size(box_size)
    frame = int(frame)
    cell_id = int(cell_id)

    optional_sources = {
        "preprocessed": preprocessed_volume,
        "binary_mask": binary_mask_volume,
        "instance_labels": instance_labels_volume,
    }
    for name, source in optional_sources.items():
        if source is not None:
            validate_aligned_volume(
                source,
                reference_volume=image_volume,
                name=f"{name}_volume",
            )

    row = find_cell(cells=cells, frame=frame, cell_id=cell_id)
    centroid_zyx = row[
        ["centroid_z", "centroid_y", "centroid_x"]
    ].to_numpy(dtype=float)

    bounds = calculate_box_bounds(
        centroid_zyx=centroid_zyx,
        box_size=size,
        spatial_shape_zyx=image_volume.shape[1:],
    )

    all_paths = build_artifact_paths(
        sample_id=sample_id,
        frame=frame,
        cell_id=cell_id,
        output_dir=output_dir,
    )

    requested_artifacts = ["raw"]
    requested_artifacts.extend(
        name for name, source in optional_sources.items() if source is not None
    )
    requested_paths = [all_paths[name] for name in requested_artifacts]
    requested_paths.append(all_paths["metadata"])

    existing = [path for path in requested_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Extraction already contains existing files for frame {frame}, "
            f"cell {cell_id}: {names}. Enable overwrite to regenerate the case."
        )

    crops: dict[str, np.ndarray] = {
        "raw": extract_box_with_bounds(
            image_volume,
            frame=frame,
            box_size=size,
            bounds=bounds,
            pad_value=pad_value,
        )
    }

    if preprocessed_volume is not None:
        crops["preprocessed"] = extract_box_with_bounds(
            preprocessed_volume,
            frame=frame,
            box_size=size,
            bounds=bounds,
            pad_value=pad_value,
        )

    if binary_mask_volume is not None:
        crops["binary_mask"] = extract_box_with_bounds(
            binary_mask_volume,
            frame=frame,
            box_size=size,
            bounds=bounds,
            pad_value=False,
        ).astype(bool, copy=False)

    if instance_labels_volume is not None:
        crops["instance_labels"] = extract_box_with_bounds(
            instance_labels_volume,
            frame=frame,
            box_size=size,
            bounds=bounds,
            pad_value=0,
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}
    for name, crop in crops.items():
        path = all_paths[name]
        np.save(path, crop, allow_pickle=False)
        saved_files[name] = path

    artifact_summaries = {
        name: _summarize_crop(name, crop, cell_id=cell_id)
        for name, crop in crops.items()
    }

    metadata = make_metadata(
        sample_id=sample_id,
        row=row,
        frame=frame,
        cell_id=cell_id,
        centroid_zyx=centroid_zyx,
        box_size=size,
        bounds=bounds,
        voxel_size_zyx=voxel_size_zyx,
        output_files=saved_files,
        artifact_summaries=artifact_summaries,
        source_cell_file=source_cell_file,
        source_zarr_array=source_zarr_array,
        source_preprocessed=source_preprocessed,
        source_binary_mask=source_binary_mask,
        source_instance_labels=source_instance_labels,
    )

    metadata_path = all_paths["metadata"]
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
            # Retained for compatibility with the existing manifest.
            "volume_file": saved_files["raw"].name,
            "raw_file": saved_files["raw"].name,
            "preprocessed_file": (
                saved_files["preprocessed"].name
                if "preprocessed" in saved_files
                else None
            ),
            "binary_mask_file": (
                saved_files["binary_mask"].name
                if "binary_mask" in saved_files
                else None
            ),
            "instance_labels_file": (
                saved_files["instance_labels"].name
                if "instance_labels" in saved_files
                else None
            ),
            "metadata_file": metadata_path.name,
        },
        output_dir=output_dir,
    )

    return saved_files["raw"], metadata_path, crops["raw"], metadata


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


def _load_optional_npy(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(f"Optional diagnostic array was not found at {path}")
    return np.load(path, mmap_mode="r")


def extract_selected_cells() -> None:
    """Standalone batch extraction using the configuration above."""

    if not CELL_IDS:
        raise ValueError("CELL_IDS is empty.")
    if len(CELL_IDS) != len(set(CELL_IDS)):
        raise ValueError("CELL_IDS contains duplicate IDs.")

    frame = int(FRAME)
    cells, cell_file = load_frame_cells(frame)
    image_volume = load_original_volume()

    preprocessed_volume = _load_optional_npy(PREPROCESSED_ARRAY_PATH)
    binary_mask_volume = _load_optional_npy(BINARY_MASK_ARRAY_PATH)
    instance_labels_volume = _load_optional_npy(INSTANCE_LABELS_ARRAY_PATH)

    print(f"Sample:     {SAMPLE_ID}")
    print(f"Frame:      {frame}")
    print(f"Box size:   {tuple(validate_box_size(BOX_SIZE))}")
    print(f"Output:     {OUTPUT_DIR}")
    print(
        "Artifacts:  raw"
        + (", preprocessed" if preprocessed_volume is not None else "")
        + (", binary mask" if binary_mask_volume is not None else "")
        + (", instance labels" if instance_labels_volume is not None else "")
    )
    print()

    saved = skipped = failed = 0

    for cell_id in CELL_IDS:
        try:
            volume_path, _, crop, metadata = save_cell_extraction(
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
                preprocessed_volume=preprocessed_volume,
                binary_mask_volume=binary_mask_volume,
                instance_labels_volume=instance_labels_volume,
                source_cell_file=cell_file,
                source_zarr_array=ZARR_ARRAY_PATH,
                source_preprocessed=PREPROCESSED_ARRAY_PATH,
                source_binary_mask=BINARY_MASK_ARRAY_PATH,
                source_instance_labels=INSTANCE_LABELS_ARRAY_PATH,
            )
            artifact_names = ", ".join(metadata["available_artifacts"])
            print(
                f"[SAVED]   Cell {cell_id}: {volume_path.name} | "
                f"shape={crop.shape} | {artifact_names}"
            )
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
