"""Configuration for spatial probabilistic instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class SegmentationConfig:
    """Tunable priors and numerical safeguards for component splitting.

    Defaults are ported from ``03_instance_segmentation.ipynb``. Distances and
    smoothing widths are expressed in micrometres unless explicitly named as
    voxels.
    """

    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    sigma_levels_um: tuple[float, ...] = (0.25, 0.40, 0.60, 0.85, 1.10)
    h_levels_um: tuple[float, ...] = (0.10, 0.18, 0.28, 0.42, 0.60)
    peak_cluster_radius_um: float = 1.10
    max_candidate_peaks: int = 10
    max_combinations_per_k: int = 60

    merge_tree_sigma_um: float = 0.20
    watershed_sigma_um: float = 0.50
    max_cells: int = 3

    hard_min_marker_separation_um: float = 0.85
    hard_min_child_voxels: int = 24
    hard_min_child_fraction_k2: float = 0.025
    hard_min_child_fraction_k3: float = 0.020
    hard_min_equivalent_radius_um: float = 0.55

    same_lobe_collapse_probability: float = 0.76
    pair_prior_distinct: float = 0.35
    pair_likelihood_temperature: float = 1.60
    pair_feature_models: tuple[PairFeatureModel, ...] = field(
        default_factory=_default_pair_feature_models
    )

    hypothesis_prior_h1: float = 0.24
    hypothesis_prior_h2: float = 0.66
    hypothesis_prior_h3: float = 0.10

    weight_lobe_support: float = 1.35
    weight_coverage_support: float = 2.50
    weight_marker_quality: float = 0.35
    weight_neck_support: float = 1.05
    weight_child_shape: float = 0.85
    weight_shape_improvement: float = 0.75
    weight_child_volume: float = 0.55
    weight_fragment_safety: float = 1.10

    k3_generation_min_pair_probability: float = 0.22
    k3_generation_min_geometric_probability: float = 0.40

    h2_min_conditional_probability: float = 0.62
    h2_min_odds_vs_h1: float = 1.65
    h3_min_conditional_probability: float = 0.78
    h3_min_odds_vs_h2: float = 3.50
    h1_confident_conditional_probability: float = 0.38

    probability_epsilon: float = 1e-6
    component_padding_voxels: int = 2

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx_um) != 3 or any(
            value <= 0 for value in self.voxel_size_zyx_um
        ):
            raise ValueError("voxel_size_zyx_um must contain three positive values")
        if not 1 <= self.max_cells <= 3:
            raise ValueError("max_cells must be between one and three")
        if self.component_padding_voxels < 0:
            raise ValueError("component_padding_voxels cannot be negative")
        if not self.sigma_levels_um or not self.h_levels_um:
            raise ValueError("at least one sigma and h level are required")

    def hard_min_child_fraction(self, k: int) -> float:
        """Return the broad fragment-size gate for a hypothesis size."""

        return {
            2: self.hard_min_child_fraction_k2,
            3: self.hard_min_child_fraction_k3,
        }.get(k, 0.015)

    def hypothesis_prior(self, k: int) -> float:
        """Return the prior probability assigned to an H-k hypothesis."""

        return {
            1: self.hypothesis_prior_h1,
            2: self.hypothesis_prior_h2,
            3: self.hypothesis_prior_h3,
        }.get(k, self.probability_epsilon)

    def evidence_weights(self) -> dict[str, float]:
        """Return evidence weights keyed by hypothesis feature name."""

        return {
            "lobe_support": self.weight_lobe_support,
            "coverage_support": self.weight_coverage_support,
            "marker_quality": self.weight_marker_quality,
            "neck_support": self.weight_neck_support,
            "child_shape": self.weight_child_shape,
            "shape_improvement": self.weight_shape_improvement,
            "child_volume": self.weight_child_volume,
            "fragment_safety": self.weight_fragment_safety,
        }


DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()
