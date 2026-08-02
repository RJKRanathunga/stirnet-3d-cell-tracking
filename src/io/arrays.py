"""NumPy and Zarr loading without implicit dtype conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr


def save_npy(array: np.ndarray, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)


def load_npy(
    path: str | Path,
    *,
    expected_ndim: int | None = None,
    expected_dtype: np.dtype | type | None = None,
    mmap_mode: str | None = None,
) -> np.ndarray:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Expected NumPy array does not exist: {target}")
    array = np.load(target, mmap_mode=mmap_mode, allow_pickle=False)
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(f"{target}: expected {expected_ndim} dimensions, found {array.shape}")
    if expected_dtype is not None and array.dtype != np.dtype(expected_dtype):
        raise TypeError(f"{target}: expected dtype {np.dtype(expected_dtype)}, found {array.dtype}")
    return array


def open_sample(sample_path: str | Path) -> Any:
    path = Path(sample_path)
    array_path = path if path.name == "0" else path / "0"
    if not array_path.exists():
        raise FileNotFoundError(f"Expected Zarr array does not exist: {array_path}")
    return zarr.open_array(str(array_path), mode="r")


def load_timepoint(sample_path: str | Path, t: int) -> np.ndarray:
    array = open_sample(sample_path)
    if not 0 <= int(t) < array.shape[0]:
        raise IndexError(f"Timepoint {t} is outside 0..{array.shape[0] - 1}")
    return np.asarray(array[int(t)])


def _timepoint_files(directory: str | Path, suffix: str) -> list[Path]:
    root = Path(directory)
    files = sorted(root.glob(f"t*.{suffix}"))
    if not files:
        raise FileNotFoundError(f"No t*.{suffix} files found in: {root}")
    expected = [f"t{index:03d}.{suffix}" for index in range(len(files))]
    actual = [path.name for path in files]
    if actual != expected:
        raise ValueError(f"Non-contiguous or unexpected timepoint files in {root}: {actual}")
    return files


def load_npy_time_series(
    directory: str | Path,
    expected_frames: int | None = None,
):
    """Return the notebook's lazy `(T, Z, Y, X)` Dask stack and file list."""

    import dask.array as da

    files = _timepoint_files(directory, "npy")
    if expected_frames is not None and len(files) != int(expected_frames):
        raise ValueError(
            f"{Path(directory).name}: expected {expected_frames} frames, found {len(files)}"
        )
    first = load_npy(files[0], expected_ndim=3, mmap_mode="r")
    spatial_shape = first.shape
    chunks = tuple(min(limit, size) for limit, size in zip((8, 64, 64), spatial_shape))
    arrays = []
    for path in files:
        array = load_npy(path, expected_ndim=3, mmap_mode="r")
        if array.shape != spatial_shape:
            raise ValueError(
                f"Inconsistent shape in {path.name}: expected {spatial_shape}, found {array.shape}"
            )
        arrays.append(da.from_array(array, chunks=chunks))
    return da.stack(arrays, axis=0), files


def list_timepoint_files(directory: str | Path, suffix: str) -> list[Path]:
    return _timepoint_files(directory, suffix)
