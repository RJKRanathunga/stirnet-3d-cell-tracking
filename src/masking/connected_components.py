from scipy import ndimage
import numpy as np


def label_connected_components(
        binary_mask: np.ndarray,
        connectivity: int = 2,
) -> tuple[np.ndarray, int]:
    """
    Label 3D connected components.
    """

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=connectivity,
    )

    return ndimage.label(
        binary_mask,
        structure=structure,
    )