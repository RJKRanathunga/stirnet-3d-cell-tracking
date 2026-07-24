import numpy as np
import pandas as pd

from scipy.ndimage import center_of_mass


def calculate_object_statistics(
        labeled_volume: np.ndarray
) -> pd.DataFrame:
    """
    Calculate statistics for each segmented object.

    Parameters
    ----------
    labeled_volume:
        3D labeled volume.

    Returns
    -------
    pd.DataFrame
        Object statistics containing:

        object_id
        volume
        centroid_z
        centroid_y
        centroid_x
    """

    num_objects = int(
        labeled_volume.max()
    )

    if num_objects == 0:
        return pd.DataFrame(
            columns=[
                "object_id",
                "volume",
                "centroid_z",
                "centroid_y",
                "centroid_x"
            ]
        )

    object_ids = np.arange(
        1,
        num_objects + 1
    )

    volumes = np.bincount(
        labeled_volume.ravel()
    )[1:]

    centroids = center_of_mass(
        np.ones_like(labeled_volume),
        labeled_volume,
        object_ids
    )

    rows = []

    for object_id, volume, centroid in zip(
            object_ids,
            volumes,
            centroids
    ):
        z, y, x = centroid

        rows.append({
            "object_id": int(object_id),
            "volume": int(volume),
            "centroid_z": float(z),
            "centroid_y": float(y),
            "centroid_x": float(x)
        })

    return pd.DataFrame(
        rows
    )