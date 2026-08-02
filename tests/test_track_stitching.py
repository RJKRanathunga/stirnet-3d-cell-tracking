"""Positive and negative merge-repair regressions for Stage 8."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.api import run_track_stitching


def detection(cell_id: int, volume: float, bbox: tuple[int, ...]) -> dict[str, object]:
    z0, y0, x0, z1, y1, x1 = bbox
    return {
        "cell_id": cell_id,
        "intensity_sum": volume * 10,
        "intensity_mean": 10.0,
        "intensity_std": 1.0,
        "equivalent_radius": 3.0,
        "axis_major": 8.0,
        "axis_middle": 6.0,
        "axis_minor": 4.0,
        "elongation": 1.0,
        "flatness": 1.0,
        "anisotropy": 1.0,
        "solidity": 0.9,
        "compactness": 0.8,
        "bbox_depth": z1 - z0 + 1,
        "bbox_height": y1 - y0 + 1,
        "bbox_width": x1 - x0 + 1,
        "z_min": z0,
        "y_min": y0,
        "x_min": x0,
        "z_max": z1,
        "y_max": y1,
        "x_max": x1,
        "touches_boundary": False,
        "boundary_faces": "",
    }


def merge_case(merged_volume: float = 200.0):
    frames = [
        pd.DataFrame([
            detection(1, 100, (2, 16, 11, 6, 24, 17)),
            detection(2, 100, (2, 16, 23, 6, 24, 29)),
        ]),
        pd.DataFrame([
            detection(1, 100, (2, 16, 11, 6, 24, 17)),
            detection(2, 100, (2, 16, 23, 6, 24, 29)),
        ]),
        pd.DataFrame([detection(1, merged_volume, (1, 13, 10, 7, 27, 30))]),
    ]
    tracks = pd.DataFrame([
        {"track_id": 0, "frame": 0, "cell": 0, "z": 4., "y": 20., "x": 14., "volume": 100.},
        {"track_id": 1, "frame": 0, "cell": 1, "z": 4., "y": 20., "x": 26., "volume": 100.},
        {"track_id": 0, "frame": 1, "cell": 0, "z": 4., "y": 20., "x": 14., "volume": 100.},
        {"track_id": 1, "frame": 1, "cell": 1, "z": 4., "y": 20., "x": 26., "volume": 100.},
        {"track_id": 2, "frame": 2, "cell": 0, "z": 4., "y": 20., "x": 20., "volume": merged_volume},
    ])
    return tracks, frames


class TrackStitchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.segmentations = []
        for frame in range(3):
            labels = np.zeros((9, 41, 41), dtype=np.int32)
            if frame < 2:
                labels[2:7, 16:25, 11:18] = 1
                labels[2:7, 16:25, 23:30] = 2
            else:
                z, y, x = np.ogrid[:9, :41, :41]
                labels[((z - 4) / 3) ** 2 + ((y - 20) / 7) ** 2 + ((x - 20) / 10) ** 2 <= 1] = 1
            path = root / f"t{frame:03d}.npy"
            np.save(path, labels)
            self.segmentations.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_merge_is_detected_and_corrected(self) -> None:
        tracks, frames = merge_case()
        result = run_track_stitching(tracks, frames, self.segmentations, sample_id="synthetic")
        self.assertEqual(len(result.merge_onsets), 1)
        self.assertEqual(len(result.segmentation_events), 1)
        self.assertEqual(len(result.merge_center_trajectories), 2)
        self.assertEqual(len(result.merge_track_repairs), 0)
        self.assertEqual(len(result.tracks), 6)
        self.assertEqual(set(result.tracks.loc[result.tracks.frame == 2, "track_id"]), {0, 1})
        diagnosed, trace = run_track_stitching(
            tracks,
            frames,
            self.segmentations,
            sample_id="synthetic",
            return_diagnostics=True,
        )
        pd.testing.assert_frame_equal(diagnosed.tracks, result.tracks)
        self.assertEqual(trace.decisions[0].outcome, "accepted")
        self.assertEqual(trace.decisions[0].provenance.source_type, "merge_candidate")

    def test_implausible_volume_keeps_merge_rejected(self) -> None:
        tracks, frames = merge_case(300.0)
        result = run_track_stitching(tracks, frames, self.segmentations, sample_id="synthetic")
        self.assertTrue(result.merge_onsets.empty)
        self.assertTrue(result.segmentation_events.empty)
        pd.testing.assert_frame_equal(
            result.tracks[tracks.columns].reset_index(drop=True),
            tracks.sort_values(["track_id", "frame"]).reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
