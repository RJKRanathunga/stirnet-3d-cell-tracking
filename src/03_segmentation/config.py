"""Configuration for all-effective-peak instance segmentation."""

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
