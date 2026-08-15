"""Explicit STIR-Net execution/memory profiles.

Model configuration controls mathematical architecture and parameterization.
Runtime profiles control memory/compute tradeoffs while executing that model.
Apply a profile before constructing :class:`StirNet` or
:class:`RefinementCriterion`, because those objects copy some runtime settings
at construction time.

Permanent memory-safe algorithms (integer label maps, streamed losses, local
mask rendering, exact online/chunked attention, CPU-resident target maps, and
pre-decode local-mask sampling) are intentionally not selectable here.
``decoder.max_spatial_tokens`` is also excluded: it controls adaptive spatial
pooling and therefore changes model-visible information. The local-mask train
cap is included, but unlike the other fields it changes training supervision
density; it still leaves parameter shapes and checkpoint compatibility intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import StirNetConfig


class RuntimeProfile(str, Enum):
    """Supported explicit STIR-Net execution profiles."""

    LOCAL_6GB = "local_6gb"
    CLOUD_48GB = "cloud_48gb"


@dataclass(frozen=True)
class _RuntimeSettings:
    activation_checkpointing: bool
    checkpoint_spatial: bool
    checkpoint_coreasoning: bool
    checkpoint_history: bool
    checkpoint_losses: bool
    temporal_query_chunk_size: int
    spatial_query_chunk_size: int
    spatial_key_chunk_size: int
    history_node_chunk_size: int
    native_chunk_voxels: int
    dense_chunk_voxels: int
    local_mask_train_cap: int


_RUNTIME_SETTINGS = {
    # These values reproduce the current RTX 4050 memory-constrained defaults.
    RuntimeProfile.LOCAL_6GB: _RuntimeSettings(
        activation_checkpointing=True,
        checkpoint_spatial=True,
        checkpoint_coreasoning=True,
        checkpoint_history=True,
        checkpoint_losses=True,
        temporal_query_chunk_size=8,
        spatial_query_chunk_size=8_192,
        spatial_key_chunk_size=65_536,
        history_node_chunk_size=128,
        native_chunk_voxels=262_144,
        dense_chunk_voxels=524_288,
        local_mask_train_cap=1,
    ),
    RuntimeProfile.CLOUD_48GB: _RuntimeSettings(
        # Use the larger VRAM budget to retain spatial activations, while
        # co-reasoning remains checkpointed because its retained attention
        # graph exceeds the practical 48 GB budget on the tested workload.
        # History and streamed losses avoid checkpoint recomputation.
        activation_checkpointing=True,
        checkpoint_spatial=False,
        checkpoint_coreasoning=True,
        checkpoint_history=False,
        checkpoint_losses=False,
        temporal_query_chunk_size=64,
        spatial_query_chunk_size=32_768,
        spatial_key_chunk_size=131_072,
        history_node_chunk_size=512,
        native_chunk_voxels=1_048_576,
        dense_chunk_voxels=1_048_576,
        local_mask_train_cap=8,
    ),
}


def _coerce_profile(profile: RuntimeProfile | str) -> RuntimeProfile:
    if isinstance(profile, RuntimeProfile):
        return profile
    try:
        return RuntimeProfile(profile)
    except (TypeError, ValueError):
        choices = ", ".join(item.value for item in RuntimeProfile)
        raise ValueError(
            f"Unknown STIR-Net runtime profile {profile!r}; expected one of: "
            f"{choices}"
        ) from None


def apply_runtime_profile(
    cfg: StirNetConfig,
    profile: RuntimeProfile | str,
) -> StirNetConfig:
    """Apply one complete runtime profile to ``cfg`` in place.

    Every controlled value is assigned absolutely, making repeated application
    idempotent and allowing the same base model/experiment config (including a
    reduced config) to use either execution profile.
    """
    if not isinstance(cfg, StirNetConfig):
        raise TypeError("cfg must be a StirNetConfig")
    selected = _coerce_profile(profile)
    settings = _RUNTIME_SETTINGS[selected]

    cfg.training.activation_checkpointing = settings.activation_checkpointing
    cfg.training.checkpoint_spatial = settings.checkpoint_spatial
    cfg.training.checkpoint_coreasoning = settings.checkpoint_coreasoning
    cfg.training.checkpoint_history = settings.checkpoint_history
    cfg.training.checkpoint_losses = settings.checkpoint_losses
    cfg.coreasoning.temporal_query_chunk_size = (
        settings.temporal_query_chunk_size
    )
    cfg.coreasoning.spatial_query_chunk_size = settings.spatial_query_chunk_size
    cfg.coreasoning.spatial_key_chunk_size = settings.spatial_key_chunk_size
    cfg.history.node_chunk_size = settings.history_node_chunk_size
    cfg.losses.native_chunk_voxels = settings.native_chunk_voxels
    cfg.losses.dense_chunk_voxels = settings.dense_chunk_voxels
    cfg.local_masks.train_max_queries_per_batch = settings.local_mask_train_cap
    cfg._runtime_profile = selected.value
    return cfg


def describe_runtime_profile(cfg: StirNetConfig) -> dict[str, Any]:
    """Return the active profile and every effective profile-controlled value."""
    if not isinstance(cfg, StirNetConfig):
        raise TypeError("cfg must be a StirNetConfig")
    return {
        "runtime_profile": cfg.runtime_profile,
        "training.activation_checkpointing": (
            cfg.training.activation_checkpointing
        ),
        "training.checkpoint_spatial": cfg.training.checkpoint_spatial,
        "training.checkpoint_coreasoning": (
            cfg.training.checkpoint_coreasoning
        ),
        "training.checkpoint_history": cfg.training.checkpoint_history,
        "training.checkpoint_losses": cfg.training.checkpoint_losses,
        "coreasoning.temporal_query_chunk_size": (
            cfg.coreasoning.temporal_query_chunk_size
        ),
        "coreasoning.spatial_query_chunk_size": (
            cfg.coreasoning.spatial_query_chunk_size
        ),
        "coreasoning.spatial_key_chunk_size": (
            cfg.coreasoning.spatial_key_chunk_size
        ),
        "history.node_chunk_size": cfg.history.node_chunk_size,
        "losses.native_chunk_voxels": cfg.losses.native_chunk_voxels,
        "losses.dense_chunk_voxels": cfg.losses.dense_chunk_voxels,
        "local_masks.train_max_queries_per_batch": (
            cfg.local_masks.train_max_queries_per_batch
        ),
    }


__all__ = [
    "RuntimeProfile",
    "apply_runtime_profile",
    "describe_runtime_profile",
]
