import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima


def label_connected_components(
        binary_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Label connected foreground components.
    """

    return ndimage.label(binary_mask)


def detect_hmaxima(
        distance: np.ndarray,
        h: float = 0.5,
) -> np.ndarray:
    """
    Detect regional maxima in the distance transform.
    """

    return h_maxima(distance, h=h)

def create_markers(
        hmax: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Label h-maxima regions to create watershed markers.
    """

    return ndimage.label(hmax)

def ensure_component_markers(
        markers: np.ndarray,
        component_labels: np.ndarray,
        num_components: int,
        distance: np.ndarray,
) -> np.ndarray:
    """
    Ensure every connected component has at least one marker.
    """

    marker_components = component_labels * (markers > 0)

    counts = np.bincount(
        marker_components.ravel(),
        minlength=num_components + 1,
    )

    next_marker = markers.max() + 1

    for component_id in range(1, num_components + 1):

        if counts[component_id] == 0:

            component = component_labels == component_id

            peak = np.unravel_index(
                np.argmax(distance * component),
                component.shape,
            )

            markers[peak] = next_marker

            next_marker += 1

    return markers