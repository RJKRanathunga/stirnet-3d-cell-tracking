from pathlib import Path

import numpy as np
import zarr


def open_sample(sample_path: str | Path):
    """Open a Zarr sample."""

    return zarr.open_array(Path(sample_path) / "0")


def load_timepoint(sample_path: str | Path, t: int) -> np.ndarray:
    """Load a single 3D volume."""

    array = open_sample(sample_path)

    return array[t]