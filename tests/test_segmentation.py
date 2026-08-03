"""Focused tests for all-effective-peak Stage 3 segmentation."""

from __future__ import annotations

import inspect
import sys
import unittest
from importlib import import_module
from unittest.mock import patch

import numpy as np

from src.api import detect_cells, segment_instances


segmentation_config_module = import_module("src.03_segmentation.config")
pipeline_module = import_module("src.03_segmentation.pipeline")
DEFAULT_SEGMENTATION_CONFIG = segmentation_config_module.DEFAULT_SEGMENTATION_CONFIG
segment_instances_detailed = pipeline_module.segment_instances_detailed


class SyntheticComponents:
    """Small physical-space masks with reproducible EDT topology."""

    def __init__(self, shape: tuple[int, int, int] = (11, 81, 61)) -> None:
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

    def lobes(
        self,
        count: int,
        separation_um: float,
        radius_um: float = 2.8,
    ) -> np.ndarray:
        offsets = (
            np.arange(count, dtype=float) - (count - 1) / 2.0
        ) * separation_um
        mask = np.zeros(self.shape, dtype=bool)
        for offset in offsets:
            mask |= self.ball(
                self.center + np.asarray((0.0, offset, 0.0)), radius_um
            )
        return mask

    def pair(self, separation_um: float, radius_um: float = 2.8) -> np.ndarray:
        return self.lobes(2, separation_um, radius_um)

    def triple(self, separation_um: float, radius_um: float = 2.8) -> np.ndarray:
        return self.lobes(3, separation_um, radius_um)


class AllEffectivePeakSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.synthetic = SyntheticComponents()

    def assert_success_uses_every_effective_peak(self, result) -> None:
        for diagnostic in result.component_diagnostics:
            if diagnostic.processing_status != "processed":
                continue
            self.assertIsNone(diagnostic.error)
            self.assertEqual(
                diagnostic.instance_count, diagnostic.effective_peak_count
            )
            self.assertEqual(
                diagnostic.marker_count, diagnostic.effective_peak_count
            )
            self.assertEqual(
                len(diagnostic.marker_positions_zyx),
                diagnostic.effective_peak_count,
            )

    def test_normal_single_cell_remains_one_instance(self) -> None:
        mask = self.synthetic.ball(self.synthetic.center, 2.8)

        result = segment_instances_detailed(mask)
        diagnostic = result.component_diagnostics[0]

        self.assertEqual(int(result.final_labels.max()), 1)
        self.assertEqual(diagnostic.effective_peak_count, 1)
        self.assertEqual(diagnostic.instance_count, 1)
        self.assertEqual(diagnostic.processing_status, "processed")
        self.assert_success_uses_every_effective_peak(result)

    def test_irregular_single_cell_collapses_multiple_edt_maxima(self) -> None:
        mask = self.synthetic.pair(separation_um=2.5)

        result = segment_instances_detailed(mask)
        diagnostic = result.component_diagnostics[0]

        self.assertGreaterEqual(diagnostic.raw_peak_count, 2)
        self.assertEqual(diagnostic.effective_peak_count, 1)
        self.assertEqual(diagnostic.instance_count, 1)
        self.assert_success_uses_every_effective_peak(result)

    def test_two_effective_peaks_produce_two_instances(self) -> None:
        result = segment_instances_detailed(self.synthetic.pair(separation_um=4.4))
        diagnostic = result.component_diagnostics[0]

        self.assertEqual(diagnostic.effective_peak_count, 2)
        self.assertEqual(diagnostic.instance_count, 2)
        self.assert_success_uses_every_effective_peak(result)

    def test_more_than_three_effective_peaks_produce_more_than_three_instances(
        self,
    ) -> None:
        mask = self.synthetic.lobes(4, separation_um=4.8)

        result = segment_instances_detailed(mask, retain_debug_artifacts=True)
        diagnostic = result.component_diagnostics[0]
        artifact = result.component_debug_artifacts[0]

        self.assertGreater(diagnostic.effective_peak_count, 3)
        self.assertEqual(diagnostic.instance_count, diagnostic.effective_peak_count)
        self.assertEqual(int(result.final_labels.max()), diagnostic.instance_count)
        self.assertEqual(
            artifact.marker_positions_zyx,
            tuple(
                tuple(
                    coordinate - DEFAULT_SEGMENTATION_CONFIG.component_padding_voxels
                    for coordinate in peak.position_zyx
                )
                for peak in artifact.effective_peaks
            ),
        )
        self.assert_success_uses_every_effective_peak(result)

    def test_failed_component_processing_falls_back_to_one_instance(self) -> None:
        mask = self.synthetic.pair(separation_um=4.4)

        with patch.object(
            pipeline_module,
            "analyze_component_crop",
            side_effect=RuntimeError("injected failure"),
        ):
            result = segment_instances_detailed(mask)

        diagnostic = result.component_diagnostics[0]
        self.assertEqual(int(result.final_labels.max()), 1)
        self.assertEqual(diagnostic.processing_status, "fallback_single")
        self.assertIn("injected failure", diagnostic.error or "")
        np.testing.assert_array_equal(result.final_labels > 0, mask)

    def test_labels_and_marker_ids_are_aligned_and_deterministic(self) -> None:
        mask = self.synthetic.lobes(4, separation_um=4.8)

        first = segment_instances_detailed(mask)
        second = segment_instances_detailed(mask)

        np.testing.assert_array_equal(first.final_labels, second.final_labels)
        np.testing.assert_array_equal(first.markers, second.markers)
        marker_ids = np.sort(first.markers[first.markers > 0])
        np.testing.assert_array_equal(
            marker_ids, np.arange(1, int(first.final_labels.max()) + 1)
        )
        for coordinate in np.argwhere(first.markers > 0):
            position = tuple(int(value) for value in coordinate)
            self.assertEqual(
                int(first.markers[position]), int(first.final_labels[position])
            )
        self.assert_success_uses_every_effective_peak(first)

    def test_complete_stage2_mask_coverage_is_preserved(self) -> None:
        mask = self.synthetic.lobes(4, separation_um=4.8)
        second_component = np.zeros_like(mask)
        second_component[2:5, 2:7, 2:7] = True
        mask |= second_component

        result = segment_instances_detailed(mask)

        np.testing.assert_array_equal(result.final_labels > 0, mask)
        self.assertFalse(np.any(result.final_labels[~mask]))
        self.assert_success_uses_every_effective_peak(result)

    def test_production_has_no_hypothesis_import_or_cell_count_limit(self) -> None:
        source = inspect.getsource(pipeline_module)
        config_fields = segmentation_config_module.SegmentationConfig.__dataclass_fields__

        self.assertNotIn("hypotheses", source)
        self.assertNotIn("Hypothesis", source)
        self.assertNotIn("max_cells", config_fields)
        self.assertNotIn("max_candidate_peaks", config_fields)
        self.assertNotIn("max_combinations_per_k", config_fields)
        self.assertNotIn("src.03_segmentation.hypotheses", sys.modules)

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
