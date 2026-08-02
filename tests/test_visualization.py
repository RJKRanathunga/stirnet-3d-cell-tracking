"""Stage 9 preparation contract migrated from the visualization notebook."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.api import prepare_visualization_data


class VisualizationPreparationTests(unittest.TestCase):
    def test_nearest_cell_mapping_and_napari_array_order(self) -> None:
        tracks = pd.DataFrame({
            "track_id": [8, 7],
            "frame": [1, 0],
            "z": [2.0, 1.0],
            "y": [4.1, 3.0],
            "x": [6.0, 5.0],
        })
        cells = pd.DataFrame({
            "frame": [0, 1],
            "cell_id": [11, 12],
            "centroid_z": [1.0, 2.0],
            "centroid_y": [3.0, 4.0],
            "centroid_x": [5.0, 6.0],
        })

        result, trace = prepare_visualization_data(
            tracks, cells, return_diagnostics=True
        )
        self.assertEqual(result.tracks["cell_id"].tolist(), [12, 11])
        np.testing.assert_array_equal(
            result.tracks_array,
            tracks[["track_id", "frame", "z", "y", "x"]].to_numpy(float),
        )
        np.testing.assert_array_equal(
            result.points_array,
            tracks[["frame", "z", "y", "x"]].to_numpy(float),
        )
        self.assertEqual(trace.stage_name, "09_visualization")


if __name__ == "__main__":
    unittest.main()
