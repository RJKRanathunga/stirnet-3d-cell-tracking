"""Focused synthetic tests for the Stage 05 intensity investigation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from investigations.stage_05_feature_extraction.intensity_statistics.comparisons import (
    summarize_association_pairs,
)
from investigations.stage_05_feature_extraction.intensity_statistics.frame_analysis import (
    analyze_frame,
)
from investigations.stage_05_feature_extraction.intensity_statistics.methods import (
    IntensityImage,
    weak_gaussian,
)
from investigations.stage_05_feature_extraction.intensity_statistics.statistics import (
    summarize_values,
)
from investigations.stage_05_feature_extraction.intensity_statistics.tracks import (
    select_track_candidates,
)


class IntensityStatisticsTests(unittest.TestCase):
    def test_summarize_values(self) -> None:
        result = summarize_values(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(result["voxel_count"], 4)
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["range"], 3.0)
        self.assertAlmostEqual(result["iqr"], 1.5)

    def test_zero_sigma_preserves_raw(self) -> None:
        raw = np.arange(27, dtype=np.uint16).reshape(3, 3, 3)
        filtered = weak_gaussian(
            raw,
            sigma_um=0.0,
            voxel_size_zyx_um=(1.625, 0.40625, 0.40625),
        )
        np.testing.assert_array_equal(filtered, raw.astype(np.float32))

    def test_fixed_instance_masks_preserve_geometry(self) -> None:
        labels = np.zeros((2, 3, 3), dtype=np.int32)
        labels[:, 1:, 1:] = 1
        binary = labels > 0
        images = {
            "raw": IntensityImage(
                "raw", np.ones_like(labels, dtype=np.float32), None, ""
            ),
            "scaled": IntensityImage(
                "scaled", np.full_like(labels, 5, dtype=np.float32), None, ""
            ),
        }
        cells, foreground = analyze_frame(
            sample_id="sample",
            frame=0,
            binary_mask=binary,
            instance_labels=labels,
            images=images,
        )
        self.assertEqual(len(cells), 2)
        self.assertEqual(cells["volume_voxels"].nunique(), 1)
        self.assertEqual(cells["centroid_z"].nunique(), 1)
        self.assertEqual(
            cells.set_index("method").loc["raw", "mean"], 1.0
        )
        self.assertEqual(
            cells.set_index("method").loc["scaled", "mean"], 5.0
        )
        self.assertEqual(len(foreground), 2)

    def test_complete_track_selection(self) -> None:
        tracks = pd.DataFrame(
            {
                "track_id": [1, 1, 1, 2, 2],
                "frame": [0, 1, 2, 0, 1],
                "cell_id": [10, 11, 12, 20, 21],
                "touches_boundary": [False] * 5,
                "match_type": ["observed"] * 5,
            }
        )
        candidates = select_track_candidates(tracks, (0, 1, 2))
        selected = candidates.loc[candidates["selected"], "track_id"].tolist()
        self.assertEqual(selected, [1])

    def test_association_auc_prefers_larger_wrong_cost(self) -> None:
        pairs = pd.DataFrame(
            {
                "method": ["raw"] * 6,
                "pair_type": ["correct"] * 3 + ["wrong_nearby"] * 3,
                "feature_cost": [0.1, 0.2, 0.15, 0.7, 0.8, 0.9],
            }
        )
        summary = summarize_association_pairs(pairs)
        separation = summary[summary["pair_type"] == "separation"].iloc[0]
        self.assertAlmostEqual(
            separation["wrong_greater_than_correct_auc"], 1.0
        )
        self.assertGreater(separation["median_separation"], 0)


if __name__ == "__main__":
    unittest.main()
