from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from investigations.stage_07_cell_tracking.small_cell_statistics.scene_io import (
    build_selected_observations,
    scan_scenes,
)
from investigations.stage_07_cell_tracking.small_cell_statistics.statistics import (
    classify_control_tracks,
    link_manual_trajectories,
    summarize_tracks,
    threshold_candidates,
)


class SmallCellStatisticsTests(unittest.TestCase):
    def test_scene_scan_and_selected_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "001"
            scene.mkdir()
            with (scene / "scene.json").open("w", encoding="utf-8") as file:
                json.dump({
                    "sample_id": "sample",
                    "selected_cells": {"1": [4], "2": [7]},
                    "source": {"tracks_csv": "missing/tracks.csv"},
                }, file)
            cases, validation = scan_scenes(root)
            selected = build_selected_observations(cases)
            self.assertEqual(len(cases), 1)
            self.assertTrue(bool(validation.iloc[0]["valid"]))
            self.assertEqual(selected["cell_id"].tolist(), [4, 7])

    def test_manual_linking_uses_physical_distance(self) -> None:
        observations = pd.DataFrame([
            {"case_id": "case", "sample_id": "sample", "frame": 0, "cell_id": 1,
             "centroid_z": 10.0, "centroid_y": 10.0, "centroid_x": 10.0},
            {"case_id": "case", "sample_id": "sample", "frame": 0, "cell_id": 2,
             "centroid_z": 10.0, "centroid_y": 30.0, "centroid_x": 10.0},
            {"case_id": "case", "sample_id": "sample", "frame": 1, "cell_id": 3,
             "centroid_z": 10.0, "centroid_y": 11.0, "centroid_x": 10.0},
            {"case_id": "case", "sample_id": "sample", "frame": 1, "cell_id": 4,
             "centroid_z": 10.0, "centroid_y": 29.0, "centroid_x": 10.0},
        ])
        linked = link_manual_trajectories(
            observations,
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
            maximum_distance_um=5.0,
        )
        by_cell = linked.set_index("cell_id")["manual_track_id"]
        self.assertEqual(by_cell[1], by_cell[3])
        self.assertEqual(by_cell[2], by_cell[4])
        self.assertNotEqual(by_cell[1], by_cell[2])

    def test_track_summary_measures_adjacent_relative_change(self) -> None:
        observations = pd.DataFrame([
            {"sample_id": "sample", "manual_track_id": "a", "frame": 0, "cell_id": 1, "analysis_volume_voxels": 100.0},
            {"sample_id": "sample", "manual_track_id": "a", "frame": 1, "cell_id": 2, "analysis_volume_voxels": 120.0},
            {"sample_id": "sample", "manual_track_id": "a", "frame": 2, "cell_id": 3, "analysis_volume_voxels": 80.0},
        ])
        summary = summarize_tracks(
            observations, track_column="manual_track_id", cohort="target"
        )
        row = summary.iloc[0]
        self.assertTrue(bool(row["contiguous"]))
        self.assertEqual(int(row["adjacent_transition_count"]), 2)
        self.assertGreater(float(row["adjacent_relative_change_p90"]), 0.2)

    def test_stable_controls_exclude_gaps_boundaries_events_and_targets(self) -> None:
        rows = []
        for track_id, frames in ((1, [0, 1, 2, 3, 4]), (2, [0, 2, 3, 4, 5]), (3, [0, 1, 2, 3, 4]), (4, [0, 1, 2, 3, 4])):
            for frame in frames:
                rows.append({
                    "sample_id": "sample", "track_id": track_id, "frame": frame,
                    "cell_id": 100 * track_id + frame,
                    "touches_boundary": track_id == 3 and frame == 2,
                    "is_virtual_merge": False,
                })
        manifest = classify_control_tracks(
            pd.DataFrame(rows),
            target_track_ids={4},
            event_track_ids={3},
            minimum_observations=5,
            require_contiguous=True,
            exclude_boundary=True,
            exclude_virtual=True,
            exclude_events=True,
        ).set_index("track_id")
        self.assertTrue(bool(manifest.loc[1, "stable_control"]))
        self.assertFalse(bool(manifest.loc[2, "stable_control"]))
        self.assertFalse(bool(manifest.loc[3, "stable_control"]))
        self.assertFalse(bool(manifest.loc[4, "stable_control"]))

    def test_threshold_candidates_report_recall_and_prevalence(self) -> None:
        table = threshold_candidates(
            np.asarray([40.0, 50.0, 60.0]),
            np.asarray([30.0, 60.0, 100.0, 200.0, 300.0]),
            np.asarray([80.0, 100.0, 120.0]),
        )
        self.assertFalse(table.empty)
        self.assertTrue(table["selected_target_recall"].between(0, 1).all())
        self.assertTrue(table["all_other_cell_prevalence"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
