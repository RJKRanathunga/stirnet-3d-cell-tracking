from .features import extract_cell_features
from .dataframe import format_detection_dataframe

import numpy as np
import pandas as pd

from src.diagnostics import Provenance, StageTrace


def detect_cells(
        instance_labels: np.ndarray,
        *,
        return_diagnostics: bool = False,
):
    """
    Detect cells from an instance segmentation.
    """

    cells = extract_cell_features(instance_labels)

    cells = format_detection_dataframe(cells)

    if not return_diagnostics:
        return cells
    trace = StageTrace(
        stage_name="04_detection",
        inputs={"instance_labels": instance_labels},
        outputs={"cells": cells},
        metrics={"detections": len(cells)},
        provenance={
            f"cell:{int(cell_id)}": Provenance(
                source_type="observed_instance",
                source_stage="03_segmentation",
                source_instance_id=int(cell_id),
                source_cell_id=int(cell_id),
            )
            for cell_id in cells["cell_id"]
        },
    )
    return cells, trace
