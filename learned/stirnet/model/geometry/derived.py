from __future__ import annotations

from contextlib import nullcontext
import torch
from torch import Tensor

from ..config import PartitionConfig
from ..types import GeometryDerivedCache, GeometryLike, geometry_field


def _profile(profiler, name: str):
    return nullcontext() if profiler is None else profiler.profile(name)


@torch.no_grad()
def build_geometry_derived_cache(
    geometry: GeometryLike,
    config: PartitionConfig,
    *,
    padding_mask: Tensor | None = None,
    stage_profiler=None,
    profile_prefix: str = "watershed",
) -> GeometryDerivedCache:
    """Materialize each partition probability exactly once for one forward."""
    with _profile(stage_profiler, f"{profile_prefix}_geometry_probability_prepare"):
        foreground = geometry_field(geometry, "foreground_logits").detach().sigmoid()
        surface = geometry_field(geometry, "surface_logits").detach().sigmoid()
        separator = geometry_field(geometry, "separator_logits").detach().sigmoid()
        seed = geometry_field(geometry, "seed_logits").detach().sigmoid()
        sdf = geometry_field(geometry, "sdf").detach()
    with _profile(stage_profiler, f"{profile_prefix}_foreground_threshold"):
        foreground_mask = foreground[:, 0] >= config.foreground_threshold
        if padding_mask is not None:
            foreground_mask = foreground_mask & ~padding_mask.bool()
    with _profile(stage_profiler, f"{profile_prefix}_sdf_normalization"):
        sdf_positive = sdf.clamp_min(0)
        masked_max = torch.where(
            foreground_mask[:, None], sdf_positive, torch.zeros_like(sdf_positive)
        ).flatten(1).amax(dim=1).clamp_min(1e-6)
        sdf_normalized = sdf_positive / masked_max[:, None, None, None, None]
    with _profile(stage_profiler, f"{profile_prefix}_seed_score"):
        seed_score = (
            config.seed_sdf_weight * sdf_normalized
            + config.seed_head_weight * seed
        ) * (1.0 - separator)
    with _profile(stage_profiler, f"{profile_prefix}_watershed_energy"):
        energy = (
            config.watershed_separator_weight * separator
            + config.watershed_surface_weight * surface
            + config.watershed_sdf_weight * (1.0 - sdf_normalized)
        )
    return GeometryDerivedCache(
        foreground_prob=foreground,
        surface_prob=surface,
        separator_prob=separator,
        seed_prob=seed,
        sdf=sdf,
        sdf_normalized=sdf_normalized,
        foreground_mask=foreground_mask,
        seed_score=seed_score,
        watershed_energy=energy,
    )


__all__ = ["build_geometry_derived_cache"]
