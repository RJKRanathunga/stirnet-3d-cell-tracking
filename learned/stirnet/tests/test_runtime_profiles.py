from __future__ import annotations

import copy
from dataclasses import fields

import pytest
import torch

from learned.stirnet import (
    RuntimeProfile,
    StirNet,
    StirNetConfig,
    apply_runtime_profile,
    describe_runtime_profile,
)
from learned.stirnet.debugging.acceptance.first_overfit import _reduced_config
from learned.stirnet.model.config import LossConfig
from learned.stirnet.training.checkpoint import load_checkpoint, save_checkpoint


EXPECTED_RUNTIME_SETTINGS = {
    RuntimeProfile.LOCAL_6GB: {
        "runtime_profile": "local_6gb",
        "training.activation_checkpointing": True,
        "training.checkpoint_spatial": True,
        "training.checkpoint_coreasoning": True,
        "training.checkpoint_history": True,
        "training.checkpoint_losses": True,
        "coreasoning.temporal_query_chunk_size": 8,
        "coreasoning.spatial_query_chunk_size": 8_192,
        "coreasoning.spatial_key_chunk_size": 65_536,
        "history.node_chunk_size": 128,
        "losses.native_chunk_voxels": 262_144,
        "losses.dense_chunk_voxels": 524_288,
        "local_masks.train_max_queries_per_batch": 2,
    },
    RuntimeProfile.CLOUD_48GB: {
        "runtime_profile": "cloud_48gb",
        "training.activation_checkpointing": True,
        "training.checkpoint_spatial": True,
        "training.checkpoint_coreasoning": False,
        "training.checkpoint_history": False,
        "training.checkpoint_losses": False,
        "coreasoning.temporal_query_chunk_size": 64,
        "coreasoning.spatial_query_chunk_size": 32_768,
        "coreasoning.spatial_key_chunk_size": 131_072,
        "history.node_chunk_size": 512,
        "losses.native_chunk_voxels": 1_048_576,
        "losses.dense_chunk_voxels": 1_048_576,
        "local_masks.train_max_queries_per_batch": 8,
    },
}


def _small_model_config() -> StirNetConfig:
    """Small model architecture for fast state-dict compatibility tests."""
    cfg = StirNetConfig()
    cfg.spatial.channels = (4, 4, 4, 8)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.mask_dim = 1
    cfg.temporal.d_model = 8
    cfg.temporal.graph_ffn_dim = 16
    cfg.temporal.memory_ffn_dim = 16
    cfg.coreasoning.d_model = 8
    cfg.queries.d_model = 8
    cfg.queries.discovery_queries = 0
    cfg.decoder.d_model = 8
    cfg.decoder.heads = 2
    cfg.decoder.ffn_dim = 16
    cfg.decoder.mask_dim = 1
    cfg.decoder.max_spatial_tokens = 137
    cfg.proposals.local_dim = 8
    cfg.local_masks.hidden_channels = 4
    cfg.local_masks.query_channels = 4
    return cfg


@pytest.mark.parametrize(
    "profile",
    [RuntimeProfile.LOCAL_6GB, RuntimeProfile.CLOUD_48GB],
)
def test_runtime_profile_selection_and_introspection(
    profile: RuntimeProfile,
) -> None:
    cfg = StirNetConfig()
    returned = apply_runtime_profile(cfg, profile.value)
    assert returned is cfg
    assert describe_runtime_profile(cfg) == EXPECTED_RUNTIME_SETTINGS[profile]


def test_local_profile_matches_current_memory_constrained_defaults() -> None:
    defaults = StirNetConfig()
    local = apply_runtime_profile(StirNetConfig(), RuntimeProfile.LOCAL_6GB)
    assert defaults.to_dict() == local.to_dict()


@pytest.mark.parametrize("profile", list(RuntimeProfile))
def test_runtime_profiles_are_orthogonal_to_reduced_config(
    profile: RuntimeProfile,
) -> None:
    cfg = _reduced_config()
    model_settings = (
        cfg.spatial.channels,
        cfg.spatial.blocks_per_level,
        cfg.decoder.d_model,
        cfg.decoder.max_spatial_tokens,
    )
    apply_runtime_profile(cfg, profile)
    assert (
        cfg.spatial.channels,
        cfg.spatial.blocks_per_level,
        cfg.decoder.d_model,
        cfg.decoder.max_spatial_tokens,
    ) == model_settings


def test_profile_application_preserves_semantic_model_settings() -> None:
    base = _small_model_config()
    semantic_settings = {
        "spatial_channels": base.spatial.channels,
        "spatial_blocks": base.spatial.blocks_per_level,
        "spatial_mask_dim": base.spatial.mask_dim,
        "temporal_d_model": base.temporal.d_model,
        "temporal_graph_layers": base.temporal.graph_layers,
        "temporal_graph_ffn_dim": base.temporal.graph_ffn_dim,
        "coreasoning_d_model": base.coreasoning.d_model,
        "coreasoning_blocks": base.coreasoning.blocks,
        "query_d_model": base.queries.d_model,
        "decoder_d_model": base.decoder.d_model,
        "decoder_layers": base.decoder.layers,
        "decoder_ffn_dim": base.decoder.ffn_dim,
        "decoder_mask_dim": base.decoder.mask_dim,
        "decoder_max_spatial_tokens": base.decoder.max_spatial_tokens,
        "proposal_local_grid_size": base.proposals.local_grid_size,
        "proposal_local_extent_dref": base.proposals.local_extent_dref,
        "proposal_local_dim": base.proposals.local_dim,
        "proposal_native_support_radius_dref": (
            base.proposals.native_support_radius_dref
        ),
        "history_grid_size": base.history.grid_size,
        "history_extent_dref": base.history.extent_dref,
        "local_mask_support_radius_dref": base.local_masks.support_radius_dref,
        "local_mask_hidden_channels": base.local_masks.hidden_channels,
        "local_mask_query_channels": base.local_masks.query_channels,
    }
    loss_weights = {
        item.name: getattr(base.losses, item.name)
        for item in fields(LossConfig)
        if item.name not in {"native_chunk_voxels", "dense_chunk_voxels"}
    }

    for profile in RuntimeProfile:
        cfg = copy.deepcopy(base)
        apply_runtime_profile(cfg, profile)
        assert {
            "spatial_channels": cfg.spatial.channels,
            "spatial_blocks": cfg.spatial.blocks_per_level,
            "spatial_mask_dim": cfg.spatial.mask_dim,
            "temporal_d_model": cfg.temporal.d_model,
            "temporal_graph_layers": cfg.temporal.graph_layers,
            "temporal_graph_ffn_dim": cfg.temporal.graph_ffn_dim,
            "coreasoning_d_model": cfg.coreasoning.d_model,
            "coreasoning_blocks": cfg.coreasoning.blocks,
            "query_d_model": cfg.queries.d_model,
            "decoder_d_model": cfg.decoder.d_model,
            "decoder_layers": cfg.decoder.layers,
            "decoder_ffn_dim": cfg.decoder.ffn_dim,
            "decoder_mask_dim": cfg.decoder.mask_dim,
            "decoder_max_spatial_tokens": cfg.decoder.max_spatial_tokens,
            "proposal_local_grid_size": cfg.proposals.local_grid_size,
            "proposal_local_extent_dref": cfg.proposals.local_extent_dref,
            "proposal_local_dim": cfg.proposals.local_dim,
            "proposal_native_support_radius_dref": (
                cfg.proposals.native_support_radius_dref
            ),
            "history_grid_size": cfg.history.grid_size,
            "history_extent_dref": cfg.history.extent_dref,
            "local_mask_support_radius_dref": (
                cfg.local_masks.support_radius_dref
            ),
            "local_mask_hidden_channels": cfg.local_masks.hidden_channels,
            "local_mask_query_channels": cfg.local_masks.query_channels,
        } == semantic_settings
        assert {
            item.name: getattr(cfg.losses, item.name)
            for item in fields(LossConfig)
            if item.name not in {"native_chunk_voxels", "dense_chunk_voxels"}
        } == loss_weights


def test_models_have_identical_shapes_and_strict_cross_profile_loading() -> None:
    base = _small_model_config()
    local_cfg = apply_runtime_profile(
        copy.deepcopy(base), RuntimeProfile.LOCAL_6GB
    )
    cloud_cfg = apply_runtime_profile(
        copy.deepcopy(base), RuntimeProfile.CLOUD_48GB
    )
    local = StirNet(local_cfg)
    cloud = StirNet(cloud_cfg)

    local_state = local.state_dict()
    cloud_state = cloud.state_dict()
    assert local_state.keys() == cloud_state.keys()
    for key in local_state:
        assert local_state[key].shape == cloud_state[key].shape, key

    cloud.load_state_dict(local_state, strict=True)
    local.load_state_dict(cloud.state_dict(), strict=True)


def test_runtime_profile_application_is_idempotent() -> None:
    cfg = StirNetConfig()
    cfg.coreasoning.temporal_query_chunk_size = 3
    cfg.losses.native_chunk_voxels = 17
    cfg.local_masks.train_max_queries_per_batch = 5

    apply_runtime_profile(cfg, RuntimeProfile.CLOUD_48GB)
    first_config = copy.deepcopy(cfg.to_dict())
    first_description = describe_runtime_profile(cfg)
    apply_runtime_profile(cfg, "cloud_48gb")
    assert cfg.to_dict() == first_config
    assert describe_runtime_profile(cfg) == first_description


def test_unknown_runtime_profile_fails_clearly() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Unknown STIR-Net runtime profile.*local_6gb.*cloud_48gb"
        ),
    ):
        apply_runtime_profile(StirNetConfig(), "large_gpu")


def test_checkpoint_load_is_strict_and_does_not_force_saved_profile(
    tmp_path,
) -> None:
    base = _small_model_config()
    local_cfg = apply_runtime_profile(
        copy.deepcopy(base), RuntimeProfile.LOCAL_6GB
    )
    cloud_cfg = apply_runtime_profile(
        copy.deepcopy(base), RuntimeProfile.CLOUD_48GB
    )
    local = StirNet(local_cfg)
    cloud = StirNet(cloud_cfg)
    path = tmp_path / "local_profile.pt"

    save_checkpoint(path, model=local, config=local_cfg)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert raw["extra"]["runtime_profile"] == "local_6gb"
    assert "runtime_profile" not in raw["config"]

    load_checkpoint(path, cloud, map_location="cpu", strict=True)
    assert cloud_cfg.runtime_profile == "cloud_48gb"
    for key, value in local.state_dict().items():
        assert torch.equal(value, cloud.state_dict()[key]), key

    # Legacy checkpoints have no profile metadata and remain valid. Loading
    # one likewise leaves the current process's selected profile untouched.
    old_path = tmp_path / "checkpoint_without_profile_metadata.pt"
    del raw["extra"]["runtime_profile"]
    torch.save(raw, old_path)
    load_checkpoint(old_path, cloud, map_location="cpu", strict=True)
    assert cloud_cfg.runtime_profile == "cloud_48gb"
