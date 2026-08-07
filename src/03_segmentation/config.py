"""Configuration for effective-EDT plus geometric marker segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class PairFeatureModel:
    """Beta likelihoods for one pairwise lobe-evidence feature."""

    name: str
    distinct: tuple[float, float]
    same: tuple[float, float]
    weight: float


def _default_pair_feature_models() -> tuple[PairFeatureModel, ...]:
    return (
        PairFeatureModel("branch_persistence", (4.0, 2.2), (1.5, 5.0), 1.35),
        PairFeatureModel("branch_balance", (3.0, 2.7), (1.2, 6.0), 1.15),
        PairFeatureModel("separation_support", (4.0, 2.2), (2.0, 4.2), 0.75),
        PairFeatureModel("peak_support", (3.5, 1.9), (2.0, 2.8), 0.45),
    )


@dataclass(frozen=True)
class CenterCandidateConfig:
    """Provisional thresholds for cheap center-proposal detection.

    These values control transform agreement and proposal representation. They
    are candidate-detection thresholds, not biological cell measurements.
    Distances and LoG scales are expressed in micrometres.
    """

    shape_sigma_levels_um: tuple[float, ...] = (0.8, 1.2, 1.25, 1.6, 2.2)
    shape_padding_sigma_multiplier: float = 4.0
    shape_peak_h_fraction: float = 0.08
    shape_peak_min_relative_response: float = 0.18
    shape_peak_cluster_radius_um: float = 1.8
    shape_peak_min_scale_support: float = 0.2
    shape_center_neighborhood_radius_um: float = 1.8

    cross_transform_match_radius_um: float = 1.4

    proposal_min_absolute_separation_um: float = 2.8
    proposal_min_normalized_separation: float = 0.62
    proposal_radius_from_sigma_factor: float = 3.0**0.5

    cross_min_raw_persistence: float = 0.32
    cross_min_shape_relative_response: float = 0.45
    cross_min_shape_scale_support: float = 0.2

    shape_only_min_relative_response: float = 0.82
    shape_only_min_scale_support: float = 0.4
    shape_only_min_local_depth_ratio: float = 0.78

    raw_only_min_persistence: float = 0.55
    raw_only_min_setting_support: float = 0.2
    raw_only_min_depth_ratio: float = 0.55
    raw_only_min_branch_persistence: float = 0.24
    raw_only_min_separation_support: float = 0.45

    def __post_init__(self) -> None:
        if not self.shape_sigma_levels_um or any(
            not isfinite(float(value)) or float(value) <= 0
            for value in self.shape_sigma_levels_um
        ):
            raise ValueError("shape_sigma_levels_um must be nonempty and positive")
        positive = (
            "shape_padding_sigma_multiplier",
            "shape_peak_cluster_radius_um",
            "shape_center_neighborhood_radius_um",
            "cross_transform_match_radius_um",
            "proposal_min_absolute_separation_um",
            "proposal_radius_from_sigma_factor",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        normalized = (
            "shape_peak_h_fraction",
            "shape_peak_min_relative_response",
            "shape_peak_min_scale_support",
            "proposal_min_normalized_separation",
            "cross_min_raw_persistence",
            "cross_min_shape_relative_response",
            "cross_min_shape_scale_support",
            "shape_only_min_relative_response",
            "shape_only_min_scale_support",
            "shape_only_min_local_depth_ratio",
            "raw_only_min_persistence",
            "raw_only_min_setting_support",
            "raw_only_min_depth_ratio",
            "raw_only_min_branch_persistence",
            "raw_only_min_separation_support",
        )
        for name in normalized:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.shape_peak_h_fraction <= 0:
            raise ValueError("shape_peak_h_fraction must be positive")
        if self.shape_only_min_relative_response < self.cross_min_shape_relative_response:
            raise ValueError("shape-only response threshold must be at least cross-transform threshold")
        if self.shape_only_min_scale_support < self.cross_min_shape_scale_support:
            raise ValueError("shape-only scale support must be at least cross-transform support")


@dataclass(frozen=True)
class GeometricCompletionConfig:
    """Conservative physical thresholds for supplemental body markers.

    These initial thresholds are intentionally strict and provisional. They
    must be calibrated against curated real merge scenes before being treated
    as biological measurements.
    """

    surface_padding_um: float = 2.0
    surface_sigma_levels_um: tuple[float, ...] = (0.35, 0.65, 1.0)

    boundary_neighborhood_radius_um: float = 2.0
    cap_patch_radius_um: float = 1.8
    cap_cluster_radius_um: float = 1.2
    cap_min_area_um2: float = 0.8
    cap_min_prominence_um: float = 0.08
    cap_min_normal_coherence: float = 0.55
    cap_min_scale_support: float = 1.0 / 3.0

    min_normal_opposition: float = 0.58
    min_axis_alignment: float = 0.52
    min_cap_separation_um: float = 2.0
    max_cap_separation_um: float = 16.0

    axis_sample_spacing_um: float = 0.35
    min_axis_occupancy: float = 0.82
    min_consecutive_axis_occupancy: float = 0.72

    cross_section_spacing_um: float = 0.65
    cross_section_thickness_um: float = 0.65
    min_valid_cross_sections: int = 4
    min_valid_cross_section_fraction: float = 0.55
    min_median_ellipse_iou: float = 0.48
    max_centerline_deviation_um: float = 1.25
    max_area_profile_error: float = 0.48

    min_ellipsoid_occupancy: float = 0.58
    min_ellipsoid_surface_support: float = 0.24
    min_unique_volume_fraction: float = 0.10
    min_unique_surface_fraction: float = 0.08

    representation_ellipsoid_radius: float = 0.85
    marker_min_separation_um: float = 1.2
    marker_search_radius_um: float = 2.4
    min_body_score: float = 0.58

    candidate_detection: CenterCandidateConfig = field(
        default_factory=CenterCandidateConfig
    )

    def __post_init__(self) -> None:
        positive = (
            "surface_padding_um",
            "boundary_neighborhood_radius_um",
            "cap_patch_radius_um",
            "cap_cluster_radius_um",
            "cap_min_area_um2",
            "min_cap_separation_um",
            "max_cap_separation_um",
            "axis_sample_spacing_um",
            "cross_section_spacing_um",
            "cross_section_thickness_um",
            "max_centerline_deviation_um",
            "representation_ellipsoid_radius",
            "marker_min_separation_um",
            "marker_search_radius_um",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.surface_sigma_levels_um or any(
            not isfinite(float(value)) or float(value) <= 0
            for value in self.surface_sigma_levels_um
        ):
            raise ValueError("surface_sigma_levels_um must be nonempty and positive")
        if self.min_cap_separation_um > self.max_cap_separation_um:
            raise ValueError("minimum cap separation cannot exceed maximum")
        if self.min_valid_cross_sections < 3:
            raise ValueError("min_valid_cross_sections must be at least three")
        normalized = (
            "cap_min_normal_coherence",
            "cap_min_scale_support",
            "min_normal_opposition",
            "min_axis_alignment",
            "min_axis_occupancy",
            "min_consecutive_axis_occupancy",
            "min_valid_cross_section_fraction",
            "min_median_ellipse_iou",
            "max_area_profile_error",
            "min_ellipsoid_occupancy",
            "min_ellipsoid_surface_support",
            "min_unique_volume_fraction",
            "min_unique_surface_fraction",
            "min_body_score",
        )
        for name in normalized:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isfinite(self.cap_min_prominence_um) or self.cap_min_prominence_um < 0:
            raise ValueError("cap_min_prominence_um cannot be negative")


@dataclass(frozen=True)
class SegmentationConfig:
    """Tunable peak, pair-evidence, and distance-processing parameters.

    Defaults are ported from ``03_instance_segmentation.ipynb``. Distances and
    smoothing widths are expressed in micrometres unless explicitly named as
    voxels.
    """

    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    sigma_levels_um: tuple[float, ...] = (0.25, 0.40, 0.60, 0.85, 1.10)
    h_levels_um: tuple[float, ...] = (0.10, 0.18, 0.28, 0.42, 0.60)
    peak_cluster_radius_um: float = 1.10
    merge_tree_sigma_um: float = 0.20
    watershed_sigma_um: float = 0.50

    same_lobe_collapse_probability: float = 0.76
    pair_prior_distinct: float = 0.35
    pair_likelihood_temperature: float = 1.60
    pair_feature_models: tuple[PairFeatureModel, ...] = field(
        default_factory=_default_pair_feature_models
    )

    enable_geometric_completion: bool = True

    geometric_completion: GeometricCompletionConfig = field(
        default_factory=GeometricCompletionConfig
    )

    probability_epsilon: float = 1e-6
    component_padding_voxels: int = 2

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(
            value <= 0 for value in self.voxel_size_zyx_um
        ):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if self.component_padding_voxels < 0:
            raise ValueError("component_padding_voxels cannot be negative")
        if not self.sigma_levels_um or not self.h_levels_um:
            raise ValueError("at least one sigma and h level are required")

DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
