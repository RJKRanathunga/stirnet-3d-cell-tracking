"""Conservative production marker-completion behavior."""

from __future__ import annotations

import unittest
from importlib import import_module
from unittest.mock import patch

import numpy as np


config_module = import_module("src.03_segmentation.config")
body_module = import_module("src.03_segmentation.geometric_bodies")
completion_module = import_module("src.03_segmentation.marker_completion")
models_module = import_module("src.03_segmentation.models")
pipeline_module = import_module("src.03_segmentation.pipeline")
DEFAULT_CONFIG = config_module.DEFAULT_SEGMENTATION_CONFIG


class EllipsoidFixtures:
    def __init__(self, shape: tuple[int, int, int] = (11, 121, 121)) -> None:
        self.shape = shape
        self.spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        self.grid = np.indices(shape).transpose(1, 2, 3, 0) * self.spacing
        self.center = np.asarray((5, shape[1] // 2, shape[2] // 2)) * self.spacing

    def ellipsoid(
        self,
        center_um: np.ndarray,
        semi_axes_um: tuple[float, float, float],
        rotation: np.ndarray,
    ) -> np.ndarray:
        local = (self.grid - center_um) @ rotation
        return sum(
            (local[..., axis] / semi_axes_um[axis]) ** 2
            for axis in range(3)
        ) <= 1.0

    def thin_recovery(self) -> np.ndarray:
        strong_rotation = np.column_stack(
            (np.asarray((0, 1, 0)), np.asarray((1, 0, 0)), np.asarray((0, 0, 1)))
        )
        thin_rotation = np.column_stack(
            (np.asarray((0, 0, 1)), np.asarray((1, 0, 0)), np.asarray((0, 1, 0)))
        )
        strong = self.ellipsoid(self.center, (4.0, 2.8, 2.8), strong_rotation)
        thin = self.ellipsoid(
            self.center + np.asarray((0.0, 4.0, 0.0)),
            (5.0, 2.0, 1.2),
            thin_rotation,
        )
        return strong | thin


class GeometricCompletionProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = EllipsoidFixtures()

    def test_thin_unrepresented_body_adds_one_production_marker(self) -> None:
        mask = self.fixture.thin_recovery()

        result = pipeline_module.segment_instances_detailed(
            mask, retain_debug_artifacts=True
        )
        diagnostic = result.component_diagnostics[0]
        artifact = result.component_debug_artifacts[0]

        self.assertEqual(diagnostic.effective_peak_count, 1)
        self.assertEqual(diagnostic.supplemental_marker_count, 1)
        self.assertEqual(diagnostic.final_marker_count, 2)
        self.assertEqual(diagnostic.instance_count, 2)
        self.assertEqual(artifact.final_markers[0].source, "effective_edt")
        self.assertEqual(artifact.final_markers[1].source, "geometric_completion")
        self.assertEqual(
            artifact.final_markers[0].source_reference_id,
            artifact.effective_peaks[0].peak_id,
        )
        np.testing.assert_array_equal(result.final_labels > 0, mask)
        self.assertFalse(np.any(result.final_labels[~mask]))
        self.assertEqual(
            diagnostic.final_marker_count,
            diagnostic.effective_peak_count + diagnostic.supplemental_marker_count,
        )

    def test_elongated_single_ellipsoid_does_not_gain_a_marker(self) -> None:
        rotation = np.column_stack(
            (np.asarray((0, 1, 0)), np.asarray((1, 0, 0)), np.asarray((0, 0, 1)))
        )
        mask = self.fixture.ellipsoid(
            self.fixture.center, (5.0, 2.3, 2.3), rotation
        )
        diagnostic = pipeline_module.segment_instances_detailed(mask).component_diagnostics[0]
        self.assertEqual(diagnostic.effective_peak_count, 1)
        self.assertEqual(diagnostic.supplemental_marker_count, 0)
        self.assertEqual(diagnostic.instance_count, 1)

    def test_geometry_failure_preserves_edt_processing(self) -> None:
        rotation = np.eye(3)
        mask = self.fixture.ellipsoid(
            self.fixture.center, (2.8, 2.8, 2.8), rotation
        )
        with patch.object(
            completion_module,
            "analyze_surface_geometry",
            side_effect=RuntimeError("injected geometry failure"),
        ):
            result = pipeline_module.segment_instances_detailed(
                mask, force_geometric_analysis=True
            )
        diagnostic = result.component_diagnostics[0]
        self.assertEqual(diagnostic.processing_status, "processed")
        self.assertEqual(diagnostic.geometry_processing_status, "failed")
        self.assertIn("injected geometry failure", diagnostic.geometry_error or "")
        self.assertEqual(diagnostic.final_marker_count, diagnostic.effective_peak_count)
        self.assertEqual(diagnostic.supplemental_marker_count, 0)
        np.testing.assert_array_equal(result.final_labels > 0, mask)

    def test_three_caps_cannot_select_two_bodies(self) -> None:
        evidence = models_module.BodyEvidence(*(1.0 for _ in range(13)))

        def body(body_id: int, cap_ids: tuple[int, int], score: float):
            return models_module.GeometricBody(
                body_id,
                cap_ids,
                (2.0, float(body_id * 3), 2.0),
                (2.0, float(body_id * 3), 2.0),
                np.eye(3),
                (2.0, 1.0, 1.0),
                ((1.0, float(body_id * 3), 2.0), (3.0, float(body_id * 3), 2.0)),
                evidence,
                score,
                True,
                (),
            )

        bodies = (body(1, (1, 2), 0.9), body(2, (1, 3), 0.8), body(3, (2, 3), 0.7))
        masks = {
            item.body_id: np.pad(
                np.ones((1, 1, 1), dtype=bool),
                ((1, 1), (item.body_id * 2, 8 - item.body_id * 2), (1, 1)),
            )
            for item in bodies
        }
        selected = body_module.select_geometric_bodies(
            bodies,
            masks,
            DEFAULT_CONFIG.voxel_size_zyx_um,
            DEFAULT_CONFIG.geometric_completion,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].cap_ids, (1, 2))


if __name__ == "__main__":
    unittest.main()
