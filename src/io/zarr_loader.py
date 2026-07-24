from pathlib import Path

import zarr


def load_zarr_array(zarr_path: str | Path):
    """
    Open a Zarr array stored at <zarr_path>/0.

    Parameters
    ----------
    zarr_path:
        Path to the .zarr directory.

    Returns
    -------
    zarr.Array
        Zarr array with shape:
        (time, z, y, x)
    """

    zarr_path = Path(zarr_path)
    array_path = zarr_path / "0"

    if not array_path.exists():
        raise FileNotFoundError(
            f"Zarr array not found:\n{array_path}"
        )

    volume = zarr.open_array(
        str(array_path),
        mode="r"
    )

    return volume


def load_timepoint(
        zarr_path: str | Path,
        timepoint: int
):
    """
    Load one 3D timepoint from a Zarr dataset.

    Parameters
    ----------
    zarr_path:
        Path to the .zarr directory.

    timepoint:
        Time index.

    Returns
    -------
    numpy.ndarray
        3D volume with shape (z, y, x).
    """

    volume = load_zarr_array(zarr_path)

    if timepoint < 0 or timepoint >= volume.shape[0]:
        raise IndexError(
            f"Timepoint {timepoint} is outside valid range "
            f"[0, {volume.shape[0] - 1}]"
        )

    return volume[timepoint]