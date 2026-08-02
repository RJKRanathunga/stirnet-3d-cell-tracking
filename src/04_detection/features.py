import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def extract_cell_features(
        instance_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Extract geometric features from each segmented cell.
    """

    properties = regionprops_table(
        instance_labels,
        properties=[
            "label",
            "centroid",
            "area",
            "bbox",
        ],
    )

    return pd.DataFrame(properties)