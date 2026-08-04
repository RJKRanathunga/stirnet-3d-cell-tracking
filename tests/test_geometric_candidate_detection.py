"""Peak-based geometric-candidate detection contracts."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from importlib import import_module
from unittest.mock import patch

import numpy as np
from scipy import ndimage


candidate_module = import_module("src.03_segmentation.candidate_detection")
config_module = import_module("src.03_segmentation.config")
models_module = import_module("src.03_segmentation.models")
peaks_module = import_module("src.03_segmentation.peaks")
pipeline_module = import_module("src.03_segmentation.pipeline")
DEFAULT_CONFIG = config_module.DEFAULT_SEGMENTATION_CONFIG
DEFAULT_CANDIDATE = DEFAULT_CONFIG.geometric_completion.candidate_detection


def raw_peak(
    peak_id: int,
    position: tuple[int, int, int],
    *,
    depth: float = 2.5,
    persistence: float = 0.9,
    setting_support: float = 0.8,
):
    return peaks_module.PeakCandidate(
        peak_id,
        position,
        depth,
        depth * 0.9,
        0.8,
        0.8,
        setting_support,
        8,
        persistence,
    )


def shape_peak(
    peak_id: int,
    position: tuple[int, int, int],
    *,
    sigma: float = 1.2,
    relative: float = 0.95,
    support: float = 0.8,
):
    spacing = np.asarray((1.0, 1.0, 1.0))
    return models_module.ShapePeakCandidate(
        peak_id,
        position,
        tuple(float(value) for value in np.asarray(position) * spacing),
        sigma,
        relative,
        relative,
        support,
        4,
        2.0,
        1.0,
    )


class CandidateArchitectureTests(unittest.TestCase):
    def test_old_size_gate_fields_are_removed(self) -> None:
        fields = config_module.GeometricCompletionConfig.__dataclass_fields__
        self.assertNotIn("eligibility_min_volume_per_marker_um3", fields)
        self.assertNotIn("eligibility_min_extent_per_marker_um", fields)
        self.assertNotIn("eligibility_max_unrepresented_distance_um", fields)

    def test_detector_has_no_size_or_farthest_point_gate(self) -> None:
        source = inspect.getsource(candidate_module).lower()
        for prohibited in (
            "volume_per_marker",
            "component_volume",
            "component_extent",
            "farthest_point",
            "max_unrepresented_distance",
            "raw_peak_count > effective",
        ):
            self.assertNotIn(prohibited, source)
        self.assertNotIn("h1", source)
        self.assertNotIn("h2", source)
        self.assertNotIn("h3", source)
        self.assertNotIn("max_candidate", source)

    def test_candidate_config_validation_is_physical_and_normalized(self) -> None:
        with self.assertRaises(ValueError):
            replace(DEFAULT_CANDIDATE, shape_sigma_levels_um=())
        with self.assertRaises(ValueError):
            replace(DEFAULT_CANDIDATE, cross_transform_match_radius_um=0.0)
        with self.assertRaises(ValueError):
            replace(DEFAULT_CANDIDATE, shape_only_min_relative_response=1.1)


class PhysicalLogTests(unittest.TestCase):
    @staticmethod
    def sphere(
        radius_um: float = 2.8,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        shape = (11, 41, 41)
        center = (5, 20, 20)
        grid = np.indices(shape).transpose(1, 2, 3, 0) * spacing
        center_um = np.asarray(center) * spacing
        return np.linalg.norm(grid - center_um, axis=-1) <= radius_um, center

    def test_anisotropic_physical_log_matches_axis_corrected_derivatives(self) -> None:
        mask, _ = self.sphere()
        sigma = 1.2
        spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        actual = candidate_module.physical_binary_log_response(
            mask, sigma, spacing, padding_sigma_multiplier=4.0
        )
        padding = np.maximum(np.ceil(4.0 * sigma / spacing).astype(int), 1)
        padded = np.pad(mask, tuple((int(v), int(v)) for v in padding))
        sigma_vox = sigma / spacing
        expected = np.zeros(padded.shape, dtype=float)
        for axis, order in enumerate(((2, 0, 0), (0, 2, 0), (0, 0, 2))):
            expected += ndimage.gaussian_filter(
                padded.astype(float),
                sigma=sigma_vox,
                order=order,
                mode="constant",
                cval=0.0,
            ) / spacing[axis] ** 2
        expected = -(sigma**2) * expected
        inner = tuple(slice(int(v), -int(v)) for v in padding)
        expected = np.where(mask, np.maximum(expected[inner], 0.0), 0.0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertFalse(np.any(actual[~mask]))

    def test_sphere_shape_peaks_are_deterministic_and_crop_aligned(self) -> None:
        mask, center = self.sphere()
        first, responses, maxima = candidate_module.detect_multiscale_shape_peaks(
            mask,
            DEFAULT_CANDIDATE,
            DEFAULT_CONFIG.voxel_size_zyx_um,
            retain_debug_artifacts=True,
        )
        second, _, _ = candidate_module.detect_multiscale_shape_peaks(
            mask, DEFAULT_CANDIDATE, DEFAULT_CONFIG.voxel_size_zyx_um
        )
        self.assertEqual(first, second)
        self.assertEqual(len(responses), len(DEFAULT_CANDIDATE.shape_sigma_levels_um))
        self.assertTrue(all(response.shape == mask.shape for response in responses))
        self.assertTrue(
            all(mask[peak.position_zyx] for peak in first)
            and all(mask[tuple(point)] for points in maxima for point in points)
        )
        spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        nearest = min(
            np.linalg.norm((np.asarray(peak.position_zyx) - center) * spacing)
            for peak in first
        )
        self.assertLessEqual(nearest, max(spacing))

    def test_complete_link_consolidation_does_not_chain(self) -> None:
        config = replace(
            DEFAULT_CANDIDATE,
            shape_sigma_levels_um=(0.8, 1.2, 1.6),
            shape_peak_cluster_radius_um=1.1,
            shape_peak_min_scale_support=0.0,
        )
        detections = tuple(
            candidate_module.ShapePeakDetection((0, 0, x), sigma, 1.0, 1.0)
            for x, sigma in zip((0, 1, 2), (0.8, 1.2, 1.6))
        )
        consolidated = candidate_module.consolidate_shape_peak_detections(
            detections, config, (1.0, 1.0, 1.0)
        )
        self.assertEqual(len(consolidated), 2)
        self.assertEqual(
            [peak.peak_id for peak in consolidated],
            [1, 2],
        )

    def test_ellipsoid_shape_peaks_are_stable(self) -> None:
        spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        shape = (11, 61, 61)
        center = np.asarray((5, 30, 30))
        grid_um = np.indices(shape).transpose(1, 2, 3, 0) * spacing
        local = grid_um - center * spacing
        mask = (
            (local[..., 0] / 2.5) ** 2
            + (local[..., 1] / 4.5) ** 2
            + (local[..., 2] / 2.2) ** 2
            <= 1.0
        )
        first, _, _ = candidate_module.detect_multiscale_shape_peaks(
            mask, DEFAULT_CANDIDATE, spacing
        )
        second, _, _ = candidate_module.detect_multiscale_shape_peaks(
            mask, DEFAULT_CANDIDATE, spacing
        )
        self.assertEqual(first, second)
        self.assertTrue(any(peak.scale_support >= 0.4 for peak in first))

    def test_large_sphere_and_elongated_ellipsoid_are_not_candidates(self) -> None:
        spacing = np.asarray(DEFAULT_CONFIG.voxel_size_zyx_um)
        shape = (15, 81, 81)
        center = np.asarray((7, 40, 40))
        grid_um = np.indices(shape).transpose(1, 2, 3, 0) * spacing
        local = grid_um - center * spacing
        sphere = np.linalg.norm(local, axis=-1) <= 5.0
        elongated = (
            (local[..., 0] / 2.3) ** 2
            + (local[..., 1] / 5.0) ** 2
            + (local[..., 2] / 2.3) ** 2
            <= 1.0
        )
        for mask in (sphere, elongated):
            diagnostic = pipeline_module.segment_instances_detailed(
                mask
            ).component_diagnostics[0]
            self.assertFalse(diagnostic.merge_candidate)
            self.assertFalse(diagnostic.geometry_executed)

    def test_plateau_tie_break_is_lexicographic(self) -> None:
        response = np.zeros((3, 3, 3), dtype=float)
        response[1, 1, 1] = response[1, 1, 2] = 1.0
        maxima = response > 0
        self.assertEqual(
            candidate_module._plateau_representatives(response, maxima),
            ((1, 1, 1),),
        )


class ProposalRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spacing = (1.0, 1.0, 1.0)
        self.effective = (raw_peak(1, (0, 0, 0), depth=2.5),)
        self.config = replace(
            DEFAULT_CANDIDATE,
            proposal_min_absolute_separation_um=2.0,
            proposal_min_normalized_separation=0.5,
        )

    def proposals(self, raw=(), shape=(), pairs=()):
        return candidate_module.build_center_proposals(
            tuple(raw), tuple(shape), self.effective, tuple(pairs), self.config, self.spacing
        )

    def test_representation_uses_absolute_and_local_scale_separation(self) -> None:
        represented = self.proposals(shape=(shape_peak(1, (0, 0, 1)),))[0]
        separated = self.proposals(shape=(shape_peak(1, (0, 0, 6)),))[0]
        broad_scale = self.proposals(
            shape=(shape_peak(1, (0, 0, 6), sigma=4.0),)
        )[0]
        self.assertTrue(represented.represented)
        self.assertFalse(separated.represented)
        self.assertGreater(
            separated.normalized_effective_separation,
            broad_scale.normalized_effective_separation,
        )

    def test_cross_transform_route(self) -> None:
        raw = raw_peak(2, (0, 0, 6))
        proposal = self.proposals(
            raw=(raw,), shape=(shape_peak(1, (0, 0, 6)),)
        )[0]
        self.assertTrue(proposal.candidate)
        self.assertEqual(proposal.route, "cross_transform")

    def test_shape_only_route_uses_stricter_support(self) -> None:
        proposal = self.proposals(shape=(shape_peak(1, (0, 0, 6)),))[0]
        self.assertTrue(proposal.candidate)
        self.assertEqual(proposal.route, "shape_only")
        weak = self.proposals(
            shape=(shape_peak(1, (0, 0, 6), support=0.2),)
        )[0]
        self.assertFalse(weak.candidate)
        off_center = self.proposals(
            shape=(
                replace(
                    shape_peak(1, (0, 0, 6)),
                    local_depth_ratio=0.2,
                ),
            )
        )[0]
        self.assertFalse(off_center.candidate)

    def test_suppressed_edt_route_does_not_veto_low_lobe_probability(self) -> None:
        raw = raw_peak(2, (0, 0, 6))
        pair = peaks_module.PairEvidence(
            1, 2, 6.0, 0.8, 0.5, 0.2, 0.7, 0.8, -8.0, 0.001
        )
        proposal = self.proposals(raw=(raw,), pairs=(pair,))[0]
        self.assertTrue(proposal.candidate)
        self.assertEqual(proposal.route, "suppressed_edt")
        self.assertLess(proposal.distinct_lobe_probability, 0.01)

    def test_raw_peak_count_alone_cannot_activate_candidate(self) -> None:
        weak = raw_peak(
            2,
            (0, 0, 6),
            persistence=0.05,
            setting_support=0.01,
        )
        proposal = self.proposals(raw=(weak,))[0]
        self.assertFalse(proposal.candidate)


class PipelineCandidateGateTests(unittest.TestCase):
    @staticmethod
    def mask() -> np.ndarray:
        mask = np.zeros((7, 19, 19), dtype=bool)
        grid = np.indices(mask.shape).transpose(1, 2, 3, 0)
        mask[np.linalg.norm(grid - np.asarray((3, 9, 9)), axis=-1) <= 3] = True
        return mask

    def test_no_candidate_skips_geometry(self) -> None:
        result = models_module.GeometricCandidateResult.no_candidate()
        with patch.object(
            pipeline_module, "safely_detect_geometric_candidate", return_value=result
        ), patch.object(
            pipeline_module, "safely_complete_geometric_markers"
        ) as geometry:
            segmented = pipeline_module.segment_instances_detailed(self.mask())
        geometry.assert_not_called()
        self.assertFalse(segmented.component_diagnostics[0].geometry_executed)

    def test_candidate_and_force_each_execute_geometry(self) -> None:
        candidate = models_module.GeometricCandidateResult(
            (), (), (1,), True, "processed", None
        )
        completion = models_module.GeometricCompletionResult.not_candidate(candidate)
        with patch.object(
            pipeline_module,
            "safely_detect_geometric_candidate",
            return_value=candidate,
        ), patch.object(
            pipeline_module,
            "safely_complete_geometric_markers",
            return_value=completion,
        ) as geometry:
            pipeline_module.segment_instances_detailed(self.mask())
        geometry.assert_called_once()

        no_candidate = models_module.GeometricCandidateResult.no_candidate()
        forced_completion = models_module.GeometricCompletionResult(
            (), (), (), (), "forced", None, None, no_candidate
        )
        with patch.object(
            pipeline_module,
            "safely_detect_geometric_candidate",
            return_value=no_candidate,
        ), patch.object(
            pipeline_module,
            "safely_complete_geometric_markers",
            return_value=forced_completion,
        ) as geometry:
            pipeline_module.segment_instances_detailed(
                self.mask(), force_geometric_analysis=True
            )
        geometry.assert_called_once()

    def test_candidate_failure_preserves_edt_segmentation(self) -> None:
        failure = models_module.GeometricCandidateResult.failed(
            RuntimeError("injected candidate failure")
        )
        with patch.object(
            pipeline_module,
            "safely_detect_geometric_candidate",
            return_value=failure,
        ), patch.object(
            pipeline_module, "safely_complete_geometric_markers"
        ) as geometry:
            result = pipeline_module.segment_instances_detailed(self.mask())
        diagnostic = result.component_diagnostics[0]
        geometry.assert_not_called()
        self.assertEqual(diagnostic.processing_status, "processed")
        self.assertEqual(diagnostic.candidate_processing_status, "failed")
        self.assertIn("injected candidate failure", diagnostic.candidate_error or "")
        np.testing.assert_array_equal(result.final_labels > 0, self.mask())


if __name__ == "__main__":
    unittest.main()
