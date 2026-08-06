from __future__ import annotations

import unittest

import numpy as np

from graph_tracking.geometry import boundary_coverage_score, classify_point_against_volume


class GeometryTests(unittest.TestCase):
    def test_inside_and_outside_classification(self) -> None:
        shape = np.asarray([11, 11, 11])
        spacing = np.ones(3)
        inside = classify_point_against_volume(np.asarray([5, 5, 0.5]), volume_shape_zyx=shape, voxel_size_zyx_um=spacing)
        self.assertTrue(inside.inside)
        self.assertEqual(inside.nearest_face, "x_min")
        outside = classify_point_against_volume(np.asarray([5, 5, 12]), volume_shape_zyx=shape, voxel_size_zyx_um=spacing)
        self.assertFalse(outside.inside)
        self.assertEqual(outside.nearest_face, "x_max")
        self.assertAlmostEqual(outside.outside_distance_um, 2.0)

    def test_coverage_decreases_near_boundary(self) -> None:
        shape = np.asarray([21, 21, 21])
        spacing = np.ones(3)
        center = boundary_coverage_score(np.asarray([10, 10, 10]), search_radius_um=5, volume_shape_zyx=shape, voxel_size_zyx_um=spacing)
        edge = boundary_coverage_score(np.asarray([10, 10, 0]), search_radius_um=5, volume_shape_zyx=shape, voxel_size_zyx_um=spacing)
        self.assertGreater(center, edge)


if __name__ == "__main__":
    unittest.main()
