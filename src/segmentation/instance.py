import numpy as np

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def compute_distance_transform(
        binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute the 3D Euclidean distance transform.

    Parameters
    ----------
    binary_mask : np.ndarray
        3D binary foreground mask with shape (Z, Y, X).

    Returns
    -------
    np.ndarray
        3D distance transform.
    """

    if binary_mask.ndim != 3:
        raise ValueError(
            "binary_mask must be a 3D array "
            "with shape (Z, Y, X)."
        )

    distance = ndimage.distance_transform_edt(
        binary_mask
    )

    return distance


def detect_cell_centers(
        distance: np.ndarray,
        min_distance: int = 5,
        threshold_abs: float = 2,
) -> np.ndarray:
    """
    Detect candidate cell centers from a 3D
    distance transform.

    Parameters
    ----------
    distance : np.ndarray
        3D distance transform.

    min_distance : int
        Minimum distance between detected peaks.

    threshold_abs : float
        Minimum distance-transform value
        required for a peak.

    Returns
    -------
    np.ndarray
        Peak coordinates with shape (N, 3),
        where each coordinate is (z, y, x).
    """

    if distance.ndim != 3:
        raise ValueError(
            "distance must be a 3D array."
        )

    coordinates = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False,
    )

    return coordinates


def create_watershed_markers(
        coordinates: np.ndarray,
        shape: tuple,
) -> np.ndarray:
    """
    Create a marker volume from detected cell centers.

    Parameters
    ----------
    coordinates : np.ndarray
        Peak coordinates with shape (N, 3).

    shape : tuple
        Shape of the target 3D volume.

    Returns
    -------
    np.ndarray
        Integer marker volume.
    """

    markers = np.zeros(
        shape,
        dtype=np.int32,
    )

    for marker_id, coord in enumerate(
            coordinates,
            start=1,
    ):
        z, y, x = coord

        markers[
            z,
            y,
            x,
        ] = marker_id

    return markers


def apply_watershed(
        distance: np.ndarray,
        markers: np.ndarray,
        binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Perform 3D marker-controlled watershed
    instance segmentation.

    Parameters
    ----------
    distance : np.ndarray
        3D distance transform.

    markers : np.ndarray
        3D marker volume.

    binary_mask : np.ndarray
        3D binary foreground mask.

    Returns
    -------
    np.ndarray
        3D instance-label volume.
    """

    instance_labels = watershed(
        -distance,
        markers,
        mask=binary_mask,
    )

    return instance_labels