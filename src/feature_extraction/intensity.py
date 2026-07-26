import numpy as np
import pandas as pd


def extract_intensity_features(
        cells_df: pd.DataFrame,
        labels: np.ndarray,
        volume: np.ndarray,
) -> pd.DataFrame:
    """
    Extract intensity statistics for each detected cell.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell detections.
    labels : np.ndarray
        3D labeled image.
    volume : np.ndarray
        Original 3D fluorescence image.

    Returns
    -------
    pd.DataFrame
        Cell detections with intensity features.
    """

    feature_rows = []

    for cell_id in cells_df["cell_id"]:

        mask = labels == cell_id
        intensities = volume[mask]

        q25 = np.percentile(intensities, 25)
        q75 = np.percentile(intensities, 75)

        feature_rows.append({
            "cell_id": cell_id,
            "intensity_mean": intensities.mean(),
            "intensity_median": np.median(intensities),
            "intensity_std": intensities.std(),
            "intensity_min": intensities.min(),
            "intensity_max": intensities.max(),
            "intensity_q25": q25,
            "intensity_q75": q75,
            "intensity_sum": intensities.sum(),
            "intensity_range": intensities.max() - intensities.min(),
            "intensity_iqr": q75 - q25,
            "intensity_cv": intensities.std() / intensities.mean(),
        })

    features_df = pd.DataFrame(feature_rows)

    return cells_df.merge(features_df, on="cell_id")