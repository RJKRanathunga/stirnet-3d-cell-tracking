"""Physical-surface and neutral-marker contracts for Stage 3."""

from __future__ import annotations

import unittest
from dataclasses import replace
from importlib import import_module

import numpy as np


config_module = import_module("src.source_instances.segmentation.config")
models_module = import_module("src.source_instances.segmentation.models")
surface_module = import_module("src.source_instances.segmentation.surface_geometry")
watershed_module = import_module("src.source_instances.segmentation.watershed")
completion_module = import_module("src.source_instances.segmentation.marker_completion")
peaks_module = import_module("src.source_instances.segmentation.peaks")
DEFAULT_CONFIG = config_module.DEFAULT_SEGMENTATION_CONFIG
InstanceMarker = models_module.InstanceMarker


class NeutralMarkerTests(unittest.TestCase):
    def test_effective_order_is_preserved_and_supplements_are_sorted(self) -> None:
        def peak(peak_id: int, position: tuple[int, int, int]):
            return peaks_module.PeakCandidate(
                peak_id, position, 2.0, 1.8, 1.0, 1.0, 1.0, 3, 0.8
            )

        effective = completion_module.convert_effective_peaks_to_markers(
            (peak(9, (3, 4, 5)), peak(2, (1, 4, 5)))
        )
        supplemental = (
            InstanceMarker((4, 4, 4), "geometric_completion", 7, 0.7),
            InstanceMarker((2, 4, 4), "geometric_completion", 8, 0.8),
        )
        combined = completion_module.combine_markers(effective, supplemental)
        self.assertEqual(
            [marker.source_reference_id for marker in combined], [9, 2, 8, 7]
        )

    def test_arbitrary_marker_count_is_sequential_and_covers_mask(self) -> None:
        mask = np.zeros((5, 7, 25), dtype=bool)
        mask[2, 2:5, 2:23] = True
        distance = import_module("scipy.ndimage").distance_transform_edt(mask)
        markers = tuple(
            InstanceMarker((2, 3, x), "effective_edt", index, 0.9)
            for index, x in enumerate((3, 9, 15, 21), start=1)
        )

        labels = watershed_module.build_marker_watershed(mask, distance, markers)

        self.assertEqual(labels.dtype, np.int32)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2, 3, 4})
        np.testing.assert_array_equal(labels > 0, mask)
        self.assertEqual(tuple(int(labels[m.position_zyx]) for m in markers), (1, 2, 3, 4))

    def test_duplicate_marker_position_is_rejected(self) -> None:
        mask = np.ones((3, 3, 3), dtype=bool)
        markers = (
            InstanceMarker((1, 1, 1), "effective_edt", 1, 1.0),
            InstanceMarker((1, 1, 1), "geometric_completion", 2, 0.8),
        )
        with self.assertRaisesRegex(ValueError, "duplicate marker"):
            watershed_module.build_marker_watershed(mask, mask.astype(float), markers)


class PhysicalSurfaceTests(unittest.TestCase):
    @staticmethod
    def sphere_mask(shift_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
        shape = (9, 41, 41)
        spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        grid = np.indices(shape).transpose(1, 2, 3, 0) * spacing
        center = np.asarray((4, 20 + shift_y, 20)) * spacing
        return np.linalg.norm(grid - center, axis=-1) <= 3.0, center

    def test_signed_distance_is_physical_and_inside_positive(self) -> None:
        mask, _ = self.sphere_mask()
        result = surface_module.physical_signed_distance(
            mask,
            DEFAULT_CONFIG.voxel_size_zyx_um,
            DEFAULT_CONFIG.geometric_completion.surface_padding_um,
        )
        padding = result.padding_zyx
        center = tuple(np.asarray((4, 20, 20)) + np.asarray(padding))
        self.assertGreater(float(result.signed_distance_um[center]), 2.0)
        self.assertLess(float(result.signed_distance_um[(0, 0, 0)]), 0.0)
        self.assertNotEqual(padding[0], padding[1])

    def test_sphere_normals_point_outward_and_caps_are_deterministic(self) -> None:
        mask, center_um = self.sphere_mask()
        first = surface_module.analyze_surface_geometry(
            mask,
            DEFAULT_CONFIG.voxel_size_zyx_um,
            DEFAULT_CONFIG.geometric_completion,
        )
        second = surface_module.analyze_surface_geometry(
            mask,
            DEFAULT_CONFIG.voxel_size_zyx_um,
            DEFAULT_CONFIG.geometric_completion,
        )
        radial = first.boundary_positions_um - center_um
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-8)
        outward = np.sum(radial * first.boundary_normals_zyx, axis=1)
        self.assertGreater(float(np.median(outward)), 0.75)
        self.assertEqual(first.caps, second.caps)
        self.assertLess(len(first.caps), len(first.boundary_positions_zyx) // 2)

    def test_geometric_config_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                DEFAULT_CONFIG.geometric_completion,
                min_cap_separation_um=9.0,
                max_cap_separation_um=2.0,
            )
        with self.assertRaises(ValueError):
            replace(
                DEFAULT_CONFIG.geometric_completion,
                min_axis_occupancy=1.1,
            )


if __name__ == "__main__":
    unittest.main()
