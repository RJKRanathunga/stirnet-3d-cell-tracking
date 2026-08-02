"""Compact Stage 7 regression covering matches, misses, births, and boundaries."""

from __future__ import annotations

import unittest

import pandas as pd

from src.api import run_cell_tracking


def detection(cell_id: int, z: float, y: float, x: float) -> dict[str, object]:
    boundary = z < 3
    return {
        "cell_id": cell_id,
        "centroid_z": z,
        "centroid_y": y,
        "centroid_x": x,
        "volume_voxels": 100.0,
        "z_min": 0.0 if boundary else z - 2,
        "y_min": y - 3,
        "x_min": x - 3,
        "z_max": z + 2,
        "y_max": y + 3,
        "x_max": x + 3,
        "extent": 0.8,
        "equivalent_radius": 3.0,
        "elongation": 1.2,
        "flatness": 1.1,
        "anisotropy": 1.3,
        "solidity": 0.9,
        "compactness": 0.8,
        "intensity_mean": 0.5,
        "intensity_std": 0.1,
        "intensity_cv": 0.2,
        "bbox_depth": 5.0,
        "bbox_height": 7.0,
        "bbox_width": 7.0,
    }


class CellTrackingTests(unittest.TestCase):
    def test_matches_misses_births_and_boundary_reacquisition(self) -> None:
        frames = [
            pd.DataFrame([detection(1, 20, 100, 100), detection(2, 1, 50, 50)]),
            pd.DataFrame([detection(1, 20, 101, 100), detection(3, 25, 150, 150)]),
            pd.DataFrame([
                detection(1, 20, 102, 100),
                detection(2, 1, 50, 51),
                detection(3, 25, 151, 150),
            ]),
        ]

        result = run_cell_tracking(frames, sample_id="synthetic")
        decisions = set(result.association_events["decision_type"])
        self.assertTrue({"match", "miss", "birth"}.issubset(decisions))
        self.assertIn("boundary_missing", set(result.boundary_events["event_type"]))
        self.assertIn("boundary_reacquired", set(result.boundary_events["event_type"]))
        self.assertEqual(result.tracks["frame"].tolist(), sorted(result.tracks["frame"]))

        diagnosed, trace = run_cell_tracking(
            frames, sample_id="synthetic", return_diagnostics=True
        )
        pd.testing.assert_frame_equal(diagnosed.tracks, result.tracks)
        self.assertEqual(trace.stage_name, "07_cell_tracking")
        self.assertTrue(trace.decisions)
        self.assertTrue(all(decision.provenance is not None for decision in trace.decisions))


if __name__ == "__main__":
    unittest.main()
