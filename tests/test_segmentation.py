"""Focused tests for spatial probabilistic instance segmentation."""

from __future__ import annotations

import unittest
from importlib import import_module
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from src.api import detect_cells, segment_instances

DEFAULT_SEGMENTATION_CONFIG = import_module(
    "src.03_segmentation.config"
).DEFAULT_SEGMENTATION_CONFIG
segment_instances_detailed = import_module(
    "src.03_segmentation.pipeline"
).segment_instances_detailed


class SyntheticComponents:
    """Small physical-space masks with reproducible EDT topology."""

    def __init__(self, shape: tuple[int, int, int] = (11, 61, 61)) -> None:
        self.shape = shape
        self.voxel_size = np.asarray(
            DEFAULT_SEGMENTATION_CONFIG.voxel_size_zyx_um, dtype=float
        )
        self.grid = np.indices(shape).transpose(1, 2, 3, 0) * self.voxel_size
        self.center = np.asarray(
            (
                5 * self.voxel_size[0],
                (shape[1] // 2) * self.voxel_size[1],
                (shape[2] // 2) * self.voxel_size[2],
            )
        )

    def ball(self, center_zyx_um: np.ndarray, radius_um: float) -> np.ndarray:
        return np.linalg.norm(self.grid - center_zyx_um, axis=-1) <= radius_um

    def pair(self, separation_um: float, radius_um: float = 2.8) -> np.ndarray:
        offset = np.asarray((0.0, separation_um / 2.0, 0.0))
        return self.ball(self.center - offset, radius_um) | self.ball(
            self.center + offset, radius_um
        )

    def triple(self, separation_um: float, radius_um: float = 2.8) -> np.ndarray:
        offset = np.asarray((0.0, separation_um, 0.0))
        return (
            self.ball(self.center - offset, radius_um)
            | self.ball(self.center, radius_um)
            | self.ball(self.center + offset, radius_um)
        )


class ProbabilisticSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.synthetic = SyntheticComponents()

    def test_normal_single_cell_remains_one_instance(self) -> None:
        mask = self.synthetic.ball(self.synthetic.center, 2.8)

        result = segment_instances_detailed(mask)
        diagnostic = result.component_diagnostics[0]

        self.assertEqual(int(result.instance_labels.max()), 1)
        self.assertEqual(diagnostic.selected_cell_count, 1)
        self.assertEqual(diagnostic.decision_status, "accepted_single")
        self.assertIsNone(diagnostic.error)
        self.assertEqual(result.hypothesis_diagnostics, ())

    def test_irregular_single_cell_collapses_multiple_edt_maxima(self) -> None:
        mask = self.synthetic.pair(separation_um=2.5)

        result = segment_instances_detailed(mask)
        diagnostic = result.component_diagnostics[0]

        self.assertGreaterEqual(diagnostic.raw_peak_count, 2)
        self.assertEqual(diagnostic.effective_lobe_count, 1)
        self.assertEqual(diagnostic.selected_cell_count, 1)
        self.assertEqual(diagnostic.decision_status, "accepted_single")

    def test_two_lobed_merge_splits_into_two(self) -> None:
        mask = self.synthetic.pair(separation_um=4.4)

        result = segment_instances_detailed(
            mask, include_hypothesis_diagnostics=True
        )
        diagnostic = result.component_diagnostics[0]

        self.assertEqual(diagnostic.selected_cell_count, 2)
        self.assertEqual(diagnostic.decision_status, "accepted_two_cell_split")
        self.assertGreater(diagnostic.posterior_h2, diagnostic.posterior_h1)
        self.assertEqual(
            {record.k for record in result.hypothesis_diagnostics}, {1, 2}
        )

    def test_three_lobed_merge_requires_sufficient_evidence(self) -> None:
        weak = segment_instances_detailed(self.synthetic.triple(separation_um=4.0))
        strong = segment_instances_detailed(self.synthetic.triple(separation_um=4.8))

        weak_diagnostic = weak.component_diagnostics[0]
        strong_diagnostic = strong.component_diagnostics[0]
        self.assertNotEqual(weak_diagnostic.selected_cell_count, 3)
        self.assertEqual(strong_diagnostic.selected_cell_count, 3)
        self.assertEqual(
            strong_diagnostic.decision_status, "accepted_three_cell_split"
        )
        self.assertGreater(
            strong_diagnostic.posterior_h3, strong_diagnostic.posterior_h2
        )

    def test_ambiguous_component_returns_uncertain_no_split(self) -> None:
        # This lobe pair has split evidence but does not meet a deliberately
        # conservative acceptance policy, exercising the explicit uncertainty
        # outcome independently of the feature-generation model.
        config = replace(
            DEFAULT_SEGMENTATION_CONFIG,
            h2_min_conditional_probability=0.95,
            h2_min_odds_vs_h1=10.0,
        )

        result = segment_instances_detailed(
            self.synthetic.pair(separation_um=3.3), config
        )
        diagnostic = result.component_diagnostics[0]

        self.assertGreater(diagnostic.posterior_h2, 0.0)
        self.assertEqual(diagnostic.decision_status, "uncertain_no_split")
        self.assertEqual(diagnostic.selected_cell_count, 1)

    def test_failed_component_processing_falls_back_to_one_instance(self) -> None:
        mask = self.synthetic.pair(separation_um=4.4)

        with patch.object(
            import_module("src.03_segmentation.pipeline"),
            "analyze_component_crop",
            side_effect=RuntimeError("injected failure"),
        ):
            result = segment_instances_detailed(mask)

        diagnostic = result.component_diagnostics[0]
        self.assertEqual(int(result.instance_labels.max()), 1)
        self.assertEqual(diagnostic.decision_status, "fallback_single")
        self.assertIn("injected failure", diagnostic.error or "")
        np.testing.assert_array_equal(result.instance_labels > 0, mask)

    def test_labels_and_marker_ids_are_aligned_and_deterministic(self) -> None:
        mask = self.synthetic.pair(separation_um=4.4)

        first = segment_instances_detailed(mask)
        second = segment_instances_detailed(mask)

        np.testing.assert_array_equal(first.instance_labels, second.instance_labels)
        np.testing.assert_array_equal(first.markers, second.markers)
        marker_ids = np.sort(first.markers[first.markers > 0])
        np.testing.assert_array_equal(
            marker_ids, np.arange(1, int(first.instance_labels.max()) + 1)
        )
        for coordinate in np.argwhere(first.markers > 0):
            position = tuple(int(value) for value in coordinate)
            self.assertEqual(
                int(first.markers[position]), int(first.instance_labels[position])
            )

    def test_full_entry_point_preserves_downstream_output_contract(self) -> None:
        mask = self.synthetic.ball(self.synthetic.center, 2.8)

        labels = segment_instances(mask)
        detected = detect_cells(labels)

        self.assertEqual(labels.shape, mask.shape)
        self.assertEqual(labels.dtype, np.int32)
        np.testing.assert_array_equal(labels > 0, mask)
        self.assertEqual(detected["cell_id"].tolist(), [1])
        self.assertTrue(
            {
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "volume_voxels",
            }.issubset(detected.columns)
        )


if __name__ == "__main__":
    unittest.main()
