from .features import extract_cell_features
from .dataframe import format_detection_dataframe

import numpy as np
import pandas as pd


def detect_cells(
        instance_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Detect cells from an instance segmentation.
    """

    cells = extract_cell_features(instance_labels)

    cells = format_detection_dataframe(cells)

    return cells