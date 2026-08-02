"""Stable entry point for the canonical feature extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics import StageTrace

from .features import extract_cell_features as _extract_cell_features


def extract_cell_features(
    cells_df: pd.DataFrame,
    labels: np.ndarray,
    volume: np.ndarray,
    *,
    return_diagnostics: bool = False,
):
    result = _extract_cell_features(cells_df, labels, volume)
    if not return_diagnostics:
        return result
    trace = StageTrace(
        stage_name="05_feature_extraction",
        inputs={"cells": cells_df, "labels": labels, "volume": volume},
        outputs={"features": result},
        metrics={
            "input_cells": len(cells_df),
            "output_cells": len(result),
            "feature_columns": len(result.columns),
        },
    )
    return result, trace
