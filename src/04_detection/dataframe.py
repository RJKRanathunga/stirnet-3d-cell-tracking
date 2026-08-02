import pandas as pd


COLUMN_MAPPING = {
    "label": "cell_id",
    "centroid-0": "centroid_z",
    "centroid-1": "centroid_y",
    "centroid-2": "centroid_x",
    "area": "volume_voxels",
    "bbox-0": "z_min",
    "bbox-1": "y_min",
    "bbox-2": "x_min",
    "bbox-3": "z_max",
    "bbox-4": "y_max",
    "bbox-5": "x_max",
}


def format_detection_dataframe(
        cells: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rename columns to the project's naming convention.
    """

    return cells.rename(columns=COLUMN_MAPPING)