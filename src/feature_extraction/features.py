import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import regionprops


def extract_cell_features(
        cells_df: pd.DataFrame,
        labels: np.ndarray,
        volume: np.ndarray,
) -> pd.DataFrame:
    """
    Extract intensity, morphology, geometry and spatial features for each cell.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell detections.

    labels : np.ndarray
        3D instance label image.

    volume : np.ndarray
        Original fluorescence image.

    Returns
    -------
    pd.DataFrame
        Cell detections with extracted features.
    """

    feature_rows = []

    regions = {
        region.label: region
        for region in regionprops(labels, intensity_image=volume)
    }

    for cell_id in cells_df["cell_id"]:

        mask = labels == cell_id
        coords = np.argwhere(mask)
        intensities = volume[mask]

        if len(coords) == 0:
            continue

        region = regions[cell_id]

        # ==========================================================
        # Intensity Features
        # ==========================================================

        intensity_mean = intensities.mean()
        intensity_std = intensities.std()

        q25 = np.percentile(intensities, 25)
        q75 = np.percentile(intensities, 75)

        # ==========================================================
        # Size Features
        # ==========================================================

        cell_volume = mask.sum()

        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)

        bbox_depth = zmax - zmin + 1
        bbox_height = ymax - ymin + 1
        bbox_width = xmax - xmin + 1

        bbox_volume = (
                bbox_depth *
                bbox_height *
                bbox_width
        )

        extent = (
            cell_volume / bbox_volume
            if bbox_volume > 0 else 0
        )

        equivalent_radius = (
                                    (3 * cell_volume) / (4 * np.pi)
                            ) ** (1 / 3)

        # ==========================================================
        # PCA Shape Features
        # ==========================================================

        centered = coords - coords.mean(axis=0)

        if len(coords) >= 3:

            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

        else:

            eigvals = np.zeros(3)

        axis_major = np.sqrt(max(eigvals[0], 0))
        axis_middle = np.sqrt(max(eigvals[1], 0))
        axis_minor = np.sqrt(max(eigvals[2], 0))

        elongation = (
            axis_major / axis_middle
            if axis_middle > 0 else 0
        )

        flatness = (
            axis_middle / axis_minor
            if axis_minor > 0 else 0
        )

        anisotropy = (
            axis_major / axis_minor
            if axis_minor > 0 else 0
        )

        # ==========================================================
        # Convex Hull Features
        # ==========================================================

        if len(coords) >= 4:

            try:

                hull = ConvexHull(coords)

                convex_volume = hull.volume

                solidity = (
                    cell_volume / convex_volume
                    if convex_volume > 0 else np.nan
                )

            except Exception:

                convex_volume = np.nan
                solidity = np.nan

        else:

            convex_volume = np.nan
            solidity = np.nan

        # ==========================================================
        # Surface Approximation
        # ==========================================================

        surface_area = region.area_bbox

        compactness = (
            cell_volume / (surface_area ** 1.5)
            if surface_area > 0 else 0
        )

        # ==========================================================
        # Centroid
        # ==========================================================

        cz, cy, cx = region.centroid

        # ==========================================================
        # Store Features
        # ==========================================================

        feature_rows.append({

            "cell_id": cell_id,

            # ------------------------------------------------------
            # Position
            # ------------------------------------------------------

            "centroid_z": cz,
            "centroid_y": cy,
            "centroid_x": cx,

            # ------------------------------------------------------
            # Intensity
            # ------------------------------------------------------

            "intensity_mean": intensity_mean,
            "intensity_median": np.median(intensities),
            "intensity_std": intensity_std,
            "intensity_min": intensities.min(),
            "intensity_max": intensities.max(),
            "intensity_q25": q25,
            "intensity_q75": q75,
            "intensity_sum": intensities.sum(),
            "intensity_range": intensities.max() - intensities.min(),
            "intensity_iqr": q75 - q25,
            "intensity_cv": (
                intensity_std / intensity_mean
                if intensity_mean > 0 else 0
            ),

            # ------------------------------------------------------
            # Size
            # ------------------------------------------------------

            "volume": cell_volume,
            "bbox_depth": bbox_depth,
            "bbox_height": bbox_height,
            "bbox_width": bbox_width,
            "bbox_volume": bbox_volume,
            "extent": extent,
            "equivalent_radius": equivalent_radius,

            # ------------------------------------------------------
            # Shape
            # ------------------------------------------------------

            "axis_major": axis_major,
            "axis_middle": axis_middle,
            "axis_minor": axis_minor,

            "elongation": elongation,
            "flatness": flatness,
            "anisotropy": anisotropy,

            "convex_volume": convex_volume,
            "solidity": solidity,
            "compactness": compactness,

        })

    features_df = pd.DataFrame(feature_rows)

    overlap = features_df.columns.intersection(cells_df.columns).difference(["cell_id"])

    return cells_df.drop(columns=overlap).merge(
                features_df,
                on="cell_id",
                how="left"
            )