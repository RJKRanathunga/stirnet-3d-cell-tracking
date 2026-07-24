import numpy as np

from scipy.ndimage import label


def label_objects(
        binary_mask: np.ndarray,
        connectivity: int = 3
) -> tuple[np.ndarray, int]:
    """
    Label connected components in a 3D binary mask.

    Parameters
    ----------
    binary_mask:
        3D binary segmentation mask.

    connectivity:
        Connectivity structure.

        3 = full 26-connectivity in 3D.

    Returns
    -------
    labeled_volume:
        3D array where each object has a unique integer label.

    num_objects:
        Number of detected objects.
    """

    if connectivity == 1:
        structure = np.zeros(
            (3, 3, 3),
            dtype=np.uint8
        )

        structure[1, 1, 1] = 1

        structure[0, 1, 1] = 1
        structure[2, 1, 1] = 1

        structure[1, 0, 1] = 1
        structure[1, 2, 1] = 1

        structure[1, 1, 0] = 1
        structure[1, 1, 2] = 1

    else:
        structure = np.ones(
            (3, 3, 3),
            dtype=np.uint8
        )

    labeled_volume, num_objects = label(
        binary_mask,
        structure=structure
    )

    return labeled_volume, num_objects