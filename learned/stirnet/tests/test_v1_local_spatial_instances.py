from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from learned.stirnet.inference.postprocess import postprocess_batch
from learned.stirnet.model.config import (
    DecoderConfig,
    LocalMaskConfig,
    LossConfig,
    ProposalConfig,
    QueryConfig,
    StirNetConfig,
)
from learned.stirnet.model.local_masks import (
    LocalNativeMaskDecoder,
    physical_local_crop,
)
from learned.stirnet.model.losses import RefinementCriterion
from learned.stirnet.model.matcher import (
    HungarianMatcher3D,
    MatchResult,
    target_masks_at_shape,
)
from learned.stirnet.model.query_builder import (
    QUERY_DISCOVERY,
    QUERY_SPATIAL_PROPOSAL,
    QUERY_TEMPORAL,
)
from learned.stirnet.model.query_decoder import QueryDecoderLayer
from learned.stirnet.model.stir_net import StirNet
from learned.stirnet.model.types import QueryState, StirNetOutput
from learned.stirnet.training.checkpoint import (
    load_checkpoint,
    migrate_stirnet_checkpoint_config,
)
from learned.stirnet.training.curriculum import (
    curriculum_stage,
    model_parameter_groups,
)
from learned.stirnet.debugging.cli.inspect_checkpoint import config_from_dict


def _small_config() -> StirNetConfig:
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
    cfg.proposals.local_dim = 8
    cfg.local_masks.hidden_channels = 4
    cfg.local_masks.query_channels = 4
    cfg.training.activation_checkpointing = False
    return cfg


def _local_decoder(
    *,
    detach_dense: bool = True,
    train_cap: int = 2,
) -> LocalNativeMaskDecoder:
    return LocalNativeMaskDecoder(
        LocalMaskConfig(
            support_radius_dref=1.5,
            hidden_channels=4,
            query_channels=4,
            query_chunk_size=1,
            train_max_queries_per_batch=train_cap,
            detach_dense_evidence=detach_dense,
        ),
        d0_channels=2,
        spatial_input_channels=5,
        query_dim=8,
        background_logit=-20.0,
    )


def test_occupancy_targets_preserve_instances_missed_by_nearest() -> None:
    labels = torch.zeros(1, 1, 8, dtype=torch.long)
    labels[..., 1] = 11
    labels[..., 5] = 29
    target = {"ids": torch.tensor([11, 29]), "label_map": labels}
    nearest = target_masks_at_shape(
        target, (1, 1, 2), torch.device("cpu"), mode="nearest"
    )
    occupancy = target_masks_at_shape(
        target, (1, 1, 2), torch.device("cpu"), mode="occupancy"
    )
    assert not nearest.flatten(1).any(dim=1).any()
    assert occupancy.flatten(1).any(dim=1).tolist() == [True, True]


def test_occupancy_target_does_not_allocate_target_by_native_volume() -> None:
    labels = torch.zeros(3, 5, 7, dtype=torch.long)
    labels[0, 0, 0] = 1
    labels[-1, -1, -1] = 2
    target = {"ids": torch.tensor([1, 2]), "label_map": labels}
    forbidden = (2, *labels.shape)
    with patch("learned.stirnet.model.matcher.torch.zeros", wraps=torch.zeros) as zeros:
        result = target_masks_at_shape(
            target, (1, 2, 3), torch.device("cpu"), mode="occupancy"
        )
    assert result.shape == (2, 1, 2, 3)
    assert all(tuple(call.args[0]) != forbidden for call in zeros.call_args_list)


def test_proposal_matching_identity_uses_immutable_anchor() -> None:
    matcher = HungarianMatcher3D(proposal_match_radius_dref=2.0)
    rows, columns = matcher._structured_assignment(
        torch.tensor([[20.0, 0.0], [0.0, 20.0]]),
        torch.arange(2),
        torch.arange(2),
        torch.full((2,), QUERY_SPATIAL_PROPOSAL),
        torch.tensor([9, 9]),
        torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
        torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]),
        torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
        {
            "ids": torch.tensor([1, 2]),
            "source_ids": torch.tensor([9]),
            "source_gt_overlap": torch.ones(1, 2, dtype=torch.long),
        },
    )
    assert dict(zip(rows.tolist(), columns.tolist())) == {0: 0, 1: 1}


def _decoder_state(query_types: list[int]) -> QueryState:
    count = len(query_types)
    references = torch.zeros(1, count, 3)
    return QueryState(
        embeddings=torch.randn(1, count, 8),
        references_cellscale=references.clone(),
        query_types=torch.tensor([query_types]),
        padding_mask=torch.zeros(1, count, dtype=torch.bool),
        source_instance_ids=torch.full((1, count), -1, dtype=torch.long),
        temporal_salience=torch.zeros(1, count, 1),
        temporal_reliability=torch.zeros(1, count, 1),
        initial_references_cellscale=references.clone(),
    )


def test_proposal_center_bound_is_total_across_three_layers() -> None:
    cfg = DecoderConfig(
        d_model=8,
        heads=2,
        layers=3,
        ffn_dim=16,
        mask_dim=2,
        dropout=0.0,
        proposal_center_max_offset_dref=0.5,
    )
    layers = [QueryDecoderLayer(cfg).eval() for _ in range(3)]
    state = _decoder_state([QUERY_SPATIAL_PROPOSAL, QUERY_TEMPORAL])
    for layer in layers:
        with torch.no_grad():
            layer.center.net[-1].weight.zero_()
            layer.center.net[-1].bias.fill_(100.0)
        state, _ = layer(
            state,
            torch.randn(1, 1, 8),
            torch.zeros(1, 1, 3),
            torch.ones(1, 2, 1, dtype=torch.bool),
            torch.ones(1),
            torch.randn(1, 2, 1, 1, 1),
        )
    proposal_distance = torch.linalg.vector_norm(
        state.references_cellscale[0, 0]
        - state.initial_references_cellscale[0, 0]
    )
    assert proposal_distance <= cfg.proposal_center_max_offset_dref + 1e-6
    # Temporal behavior remains cumulative and therefore exceeds one step.
    assert state.references_cellscale[0, 1].abs().max() > cfg.temporal_center_step_dref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_proposal_center_bound_is_amp_safe_across_three_cuda_layers() -> None:
    cfg = DecoderConfig(
        d_model=8,
        heads=2,
        layers=3,
        ffn_dim=16,
        mask_dim=2,
        dropout=0.0,
        proposal_center_max_offset_dref=0.5,
    )
    layers = [QueryDecoderLayer(cfg).cuda().eval() for _ in range(3)]
    state = _decoder_state([QUERY_SPATIAL_PROPOSAL, QUERY_TEMPORAL])
    for name, value in vars(state).items():
        if isinstance(value, torch.Tensor):
            value = value.cuda()
            if name == "embeddings":
                value = value.half()
            setattr(state, name, value)
    spatial_tokens = torch.randn(1, 1, 8, device="cuda", dtype=torch.float16)
    spatial_positions = torch.zeros(1, 1, 3, device="cuda")
    support = torch.ones(1, 2, 1, device="cuda", dtype=torch.bool)
    dref_um = torch.ones(1, device="cuda")
    mask_features = torch.randn(
        1, 2, 1, 1, 1, device="cuda", dtype=torch.float16
    )
    for layer in layers:
        with torch.no_grad():
            layer.center.net[-1].weight.zero_()
            layer.center.net[-1].bias.fill_(100.0)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                state, _ = layer(
                    state,
                    spatial_tokens,
                    spatial_positions,
                    support,
                    dref_um,
                    mask_features,
                )
    assert torch.isfinite(state.references_cellscale).all()
    proposal_distance = torch.linalg.vector_norm(
        state.references_cellscale[0, 0]
        - state.initial_references_cellscale[0, 0]
    )
    assert proposal_distance <= cfg.proposal_center_max_offset_dref + 1e-6
    assert state.references_cellscale[0, 1].abs().max() > cfg.temporal_center_step_dref


def test_physical_crop_uses_anisotropic_native_spacing() -> None:
    crop = physical_local_crop(
        (9, 15, 31),
        torch.tensor([2.0, 1.0, 0.5]),
        torch.zeros(3),
        torch.tensor(4.0),
        1.5,
    )
    assert crop.slices == (slice(1, 8), slice(1, 14), slice(3, 28))
    assert crop.support.shape == (7, 13, 25)
    center = tuple(size // 2 for size in crop.support.shape)
    assert crop.support[center]
    assert crop.relative_xyz_dref[0, -1, center[1], center[2]] == 1.5
    assert not crop.support[-1, -1, -1]


def test_off_mask_local_decoder_and_exact_support_background() -> None:
    assert ProposalConfig().native_support_radius_dref == 1.5
    decoder = _local_decoder()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.fuse[-1].bias.fill_(2.0)
    shape = (7, 7, 7)
    dense = {
        key: torch.zeros(1, 1, *shape)
        for key in ("foreground_logits", "center_heatmap_logits", "boundary_logits")
    }
    prediction = decoder.decode_one(
        torch.zeros(1, 2, *shape),
        torch.zeros(1, 5, *shape),
        dense,
        torch.zeros(8),
        torch.zeros(3),
        torch.ones(3),
        torch.tensor(2.0),
        batch_index=0,
    )
    assert prediction.logits is not None
    assert torch.all(prediction.logits[prediction.support] == 2.0)
    assert torch.all(prediction.logits[~prediction.support] == -20.0)
    # No source/current-mask ID is consumed by the local decoder, so an
    # off-mask proposal (source_instance_id=-1) follows this same path.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp16_cached_local_decoder_runs_outside_autocast_with_fp32_parameters() -> None:
    decoder = _local_decoder().cuda().train()
    shape = (7, 7, 7)
    d0 = torch.randn(
        1, 2, *shape, device="cuda", dtype=torch.float16, requires_grad=True
    )
    spatial = torch.randn(1, 5, *shape, device="cuda", dtype=torch.float16)
    dense = {
        key: torch.randn(1, 1, *shape, device="cuda", dtype=torch.float16)
        for key in ("foreground_logits", "center_heatmap_logits", "boundary_logits")
    }
    prediction = decoder.decode_one(
        d0,
        spatial,
        dense,
        torch.randn(8, device="cuda", dtype=torch.float16),
        torch.zeros(3, device="cuda"),
        torch.ones(3, device="cuda"),
        torch.tensor(2.0, device="cuda"),
        batch_index=0,
    )
    assert prediction.logits is not None
    assert prediction.logits.dtype == torch.float16
    assert torch.isfinite(prediction.logits).all()
    assert all(parameter.dtype == torch.float32 for parameter in decoder.parameters())
    prediction.logits.float().mean().backward()
    assert d0.grad is not None and torch.isfinite(d0.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


def _proposal_loss_output(
    d0: torch.Tensor,
    dense: dict[str, torch.Tensor],
    *,
    query_count: int = 1,
    query_types: list[int] | None = None,
) -> StirNetOutput:
    shape = tuple(int(value) for value in d0.shape[-3:])
    if query_types is None:
        query_types = [QUERY_SPATIAL_PROPOSAL] * query_count
    if len(query_types) != query_count:
        raise ValueError("query_types must have query_count entries")
    return StirNetOutput(
        exist_logits=torch.zeros(1, query_count, requires_grad=True),
        centers_cellscale=torch.zeros(1, query_count, 3),
        coarse_mask_logits=torch.zeros(1, query_count, 1, 1, 1),
        coarse_spacing_um=torch.ones(1, 3),
        query_embeddings=torch.randn(1, query_count, 8, requires_grad=True),
        native_mask_embeddings=torch.zeros(1, query_count, 1),
        query_types=torch.tensor([query_types]),
        query_padding_mask=torch.zeros(1, query_count, dtype=torch.bool),
        source_instance_ids=torch.full((1, query_count), -1, dtype=torch.long),
        query_initial_references_cellscale=torch.zeros(1, query_count, 3),
        temporal_salience=torch.zeros(1, query_count, 1),
        temporal_reliability=torch.zeros(1, query_count, 1),
        aux_outputs=[],
        dense_outputs=dense,
        mask_features=torch.zeros(1, 1, *shape),
        spacing_um=torch.ones(1, 3),
        dref_um=torch.tensor([2.0]),
        instance_labels=torch.zeros(1, *shape, dtype=torch.long),
        d0_features=d0,
        spatial_inputs=torch.zeros(1, 5, *shape),
    )


def _many_proposal_loss_case(
    query_count: int,
) -> tuple[StirNetOutput, list[dict], list[MatchResult]]:
    shape = (7, 7, 7)
    d0 = torch.randn(1, 2, *shape, requires_grad=True)
    dense = {
        key: torch.randn(1, 1, *shape, requires_grad=True)
        for key in ("foreground_logits", "center_heatmap_logits", "boundary_logits")
    }
    output = _proposal_loss_output(d0, dense, query_count=query_count)
    labels = torch.zeros(shape, dtype=torch.long)
    labels[3, 3, 3] = 1
    target = {
        "ids": torch.arange(1, query_count + 1),
        "label_map": labels,
        "centers_cellscale": torch.zeros(query_count, 3),
    }
    indices = torch.arange(query_count)
    return output, [target], [MatchResult(indices, indices)]


def test_training_samples_before_decoding_and_caps_each_batch_item() -> None:
    output, targets, matches = _many_proposal_loss_case(7)
    decoder = _local_decoder(train_cap=2)
    criterion = RefinementCriterion(LossConfig(), QueryConfig()).train()
    with patch.object(decoder, "decode_one", wraps=decoder.decode_one) as decode:
        dice, focal = criterion._proposal_local_mask_losses(
            output, targets, matches, decoder
        )
    assert decode.call_count == 2
    assert criterion.last_local_original_request_counts == [7]
    assert len(criterion.last_local_sampled_requests) == 2
    assert len(set(criterion.last_local_sampled_requests)) == 2
    assert torch.isfinite(dice + focal)


def test_training_sampling_is_seeded_and_not_always_the_first_queries() -> None:
    subsets = []
    for seed in (41, 41, 42):
        torch.manual_seed(seed)
        output, targets, matches = _many_proposal_loss_case(8)
        decoder = _local_decoder(train_cap=2)
        criterion = RefinementCriterion(LossConfig(), QueryConfig()).train()
        criterion._proposal_local_mask_losses(output, targets, matches, decoder)
        subsets.append(criterion.last_local_sampled_requests)
    assert subsets[0] == subsets[1]
    assert subsets[0] != subsets[2]
    assert [query for _, query in subsets[0]] != [0, 1]


def test_evaluation_streams_every_matched_proposal_despite_training_cap() -> None:
    output, targets, matches = _many_proposal_loss_case(6)
    decoder = _local_decoder(train_cap=2)
    criterion = RefinementCriterion(LossConfig(), QueryConfig()).eval()
    with patch.object(decoder, "decode_one", wraps=decoder.decode_one) as decode:
        dice, focal = criterion._proposal_local_mask_losses(
            output, targets, matches, decoder
        )
    assert decode.call_count == 6
    assert criterion.last_local_original_request_counts == [6]
    assert criterion.last_local_sampled_requests == [(0, index) for index in range(6)]
    assert torch.isfinite(dice + focal)


def test_only_sampled_queries_receive_local_query_gradients() -> None:
    torch.manual_seed(53)
    output, targets, matches = _many_proposal_loss_case(5)
    decoder = _local_decoder(train_cap=2)
    criterion = RefinementCriterion(LossConfig(), QueryConfig()).train()
    dice, focal = criterion._proposal_local_mask_losses(
        output, targets, matches, decoder
    )
    (dice + focal).backward()
    selected = {
        query_index
        for batch_index, query_index in criterion.last_local_sampled_requests
        if batch_index == 0
    }
    gradient = output.query_embeddings.grad
    assert gradient is not None
    assert len(selected) == 2
    assert all(gradient[0, index].abs().sum() > 0 for index in selected)
    assert all(
        gradient[0, index].abs().sum() == 0
        for index in range(5)
        if index not in selected
    )


def test_hybrid_mask_weighting_uses_original_match_counts_after_sampling() -> None:
    proposal_count = 8
    legacy_count = 2
    query_types = [QUERY_SPATIAL_PROPOSAL] * proposal_count + [
        QUERY_DISCOVERY
    ] * legacy_count
    shape = (3, 3, 3)
    output = _proposal_loss_output(
        torch.zeros(1, 2, *shape),
        {
            key: torch.zeros(1, 1, *shape)
            for key in (
                "foreground_logits",
                "center_heatmap_logits",
                "boundary_logits",
            )
        },
        query_count=len(query_types),
        query_types=query_types,
    )
    indices = torch.arange(len(query_types))
    matches = [MatchResult(indices, indices)]
    targets = [
        {
            "ids": torch.arange(1, len(query_types) + 1),
            "label_map": torch.zeros(shape, dtype=torch.long),
        }
    ]
    criterion = RefinementCriterion(LossConfig(), QueryConfig()).train()
    with (
        patch.object(
            criterion,
            "_proposal_local_mask_losses",
            return_value=(torch.tensor(2.0), torch.tensor(4.0)),
        ),
        patch.object(
            criterion,
            "_legacy_native_mask_losses",
            return_value=(torch.tensor(10.0), torch.tensor(20.0)),
        ),
    ):
        dice, focal = criterion._native_mask_losses(
            output, targets, matches, _local_decoder(train_cap=2)
        )
    torch.testing.assert_close(dice, torch.tensor(3.6))
    torch.testing.assert_close(focal, torch.tensor(7.2))


def test_local_loss_routes_gradients_and_detaches_dense_side_evidence() -> None:
    shape = (7, 7, 7)
    d0 = torch.randn(1, 2, *shape, requires_grad=True)
    dense = {
        key: torch.randn(1, 1, *shape, requires_grad=True)
        for key in ("foreground_logits", "center_heatmap_logits", "boundary_logits")
    }
    output = _proposal_loss_output(d0, dense)
    labels = torch.zeros(shape, dtype=torch.long)
    labels[3, 3, 3] = 7
    target = {
        "ids": torch.tensor([7]),
        "label_map": labels,
        "centers_cellscale": torch.zeros(1, 3),
    }
    decoder = _local_decoder(detach_dense=True)
    criterion = RefinementCriterion(LossConfig(), QueryConfig())
    dice, focal = criterion._proposal_local_mask_losses(
        output,
        [target],
        [MatchResult(torch.tensor([0]), torch.tensor([0]))],
        decoder,
    )
    (dice + focal).backward()
    assert d0.grad is not None and d0.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in decoder.parameters()
    )
    assert all(value.grad is None for value in dense.values())


def test_local_loss_reaches_spatial_decoder_through_d0() -> None:
    torch.manual_seed(19)
    cfg = _small_config()
    cfg.proposals.max_proposals = 4
    cfg.proposals.candidate_pool_size = 8
    model = StirNet(cfg).train()
    labels = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    labels[:, 4:12, 4:12, 4:12] = 1
    spatial = torch.randn(1, 5, 16, 16, 16)
    spatial[:, 1] = (labels > 0).float()
    output = model(
        spatial,
        labels,
        torch.ones(1, 3),
        torch.tensor([4.0]),
        torch.zeros(1, 14),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.zeros(1, 3),
        torch.zeros(0, cfg.temporal.node_dim),
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, cfg.temporal.edge_dim),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, 3),
        torch.zeros(0, cfg.temporal.status_dim),
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, cfg.temporal.hypothesis_edge_dim),
        torch.zeros(0, dtype=torch.long),
        bypass_coreasoning=True,
    )
    proposal = torch.nonzero(
        output.query_types[0] == QUERY_SPATIAL_PROPOSAL, as_tuple=False
    ).flatten()[0]
    # Isolate the direct D0 evidence route from the separate final-query route.
    output.query_embeddings = output.query_embeddings.detach()
    target = {
        "ids": torch.tensor([7]),
        "label_map": torch.where(labels[0] > 0, 7, 0),
        "centers_cellscale": torch.zeros(1, 3),
    }
    criterion = RefinementCriterion(cfg.losses, cfg.queries)
    model.zero_grad(set_to_none=True)
    dice, focal = criterion._proposal_local_mask_losses(
        output,
        [target],
        [MatchResult(proposal[None], torch.tensor([0]))],
        model.local_mask_decoder,
    )
    (dice + focal).backward()
    spatial_gradient = model.decoder.stage_e0.fuse.weight.grad
    assert spatial_gradient is not None and spatial_gradient.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.local_mask_decoder.parameters()
    )


def test_local_loss_supervises_the_complete_allowed_sphere() -> None:
    shape = (7, 7, 7)
    d0 = torch.zeros(1, 2, *shape, requires_grad=True)
    dense = {
        key: torch.zeros(1, 1, *shape, requires_grad=True)
        for key in ("foreground_logits", "center_heatmap_logits", "boundary_logits")
    }
    output = _proposal_loss_output(d0, dense)
    labels = torch.zeros(shape, dtype=torch.long)
    labels[3, 3, 3] = 7
    target = {
        "ids": torch.tensor([7]),
        "label_map": labels,
        "centers_cellscale": torch.zeros(1, 3),
    }
    decoder = _local_decoder()
    captured: list[torch.Tensor] = []

    def record(logits, targets, support, **kwargs):
        captured.append(support.detach().clone())
        zero = logits.sum() * 0
        return zero, zero

    criterion = RefinementCriterion(LossConfig(), QueryConfig())
    with patch(
        "learned.stirnet.model.losses.local_matched_mask_losses",
        side_effect=record,
    ):
        criterion._proposal_local_mask_losses(
            output,
            [target],
            [MatchResult(torch.tensor([0]), torch.tensor([0]))],
            decoder,
        )
    expected = physical_local_crop(
        shape, torch.ones(3), torch.zeros(3), torch.tensor(2.0), 1.5
    ).support
    assert len(captured) == 1
    assert torch.equal(captured[0][0], expected)


def _render_output() -> StirNetOutput:
    shape = (1, 1, 3)
    query_types = torch.tensor(
        [[QUERY_DISCOVERY, QUERY_SPATIAL_PROPOSAL, QUERY_DISCOVERY]]
    )
    return StirNetOutput(
        exist_logits=torch.full((1, 3), 10.0),
        centers_cellscale=torch.zeros(1, 3, 3),
        coarse_mask_logits=torch.zeros(1, 3, *shape),
        coarse_spacing_um=torch.ones(1, 3),
        query_embeddings=torch.zeros(1, 3, 8),
        native_mask_embeddings=torch.tensor([[[1.0], [2.0], [3.0]]]),
        query_types=query_types,
        query_padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        source_instance_ids=torch.full((1, 3), -1, dtype=torch.long),
        query_initial_references_cellscale=torch.zeros(1, 3, 3),
        temporal_salience=torch.zeros(1, 3, 1),
        temporal_reliability=torch.zeros(1, 3, 1),
        aux_outputs=[],
        dense_outputs={
            key: torch.zeros(1, 1, *shape)
            for key in ("foreground_logits", "center_heatmap_logits", "boundary_logits")
        },
        mask_features=torch.ones(1, 1, *shape),
        spacing_um=torch.ones(1, 3),
        dref_um=torch.tensor([2.0]),
        instance_labels=torch.zeros(1, *shape, dtype=torch.long),
        d0_features=torch.zeros(1, 4, *shape),
        spatial_inputs=torch.zeros(1, 5, *shape),
    )


def test_legacy_renderer_and_hybrid_selection_order_are_preserved() -> None:
    model = StirNet(_small_config()).eval()
    with torch.no_grad():
        for parameter in model.local_mask_decoder.parameters():
            parameter.zero_()
        model.local_mask_decoder.fuse[-1].bias.fill_(2.0)
        rendered = model.render_masks(
            _render_output(), [torch.tensor([2, 1, 0])]
        )[0]
    torch.testing.assert_close(rendered[:, 0, 0, 1], torch.tensor([3.0, 2.0, 1.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp16_cached_proposal_renders_outside_autocast() -> None:
    cfg = _small_config()
    cfg.proposals.max_proposals = 4
    cfg.proposals.candidate_pool_size = 8
    model = StirNet(cfg).cuda().eval()
    labels = torch.zeros(1, 16, 16, 16, device="cuda", dtype=torch.long)
    labels[:, 4:12, 4:12, 4:12] = 1
    spatial = torch.randn(1, 5, 16, 16, 16, device="cuda")
    spatial[:, 1] = (labels > 0).float()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            spatial,
            labels,
            torch.ones(1, 3, device="cuda"),
            torch.tensor([4.0], device="cuda"),
            torch.zeros(1, 14, device="cuda"),
            torch.tensor([1], device="cuda"),
            torch.tensor([0], device="cuda"),
            torch.zeros(1, 3, device="cuda"),
            torch.zeros(0, cfg.temporal.node_dim, device="cuda"),
            torch.zeros(2, 0, device="cuda", dtype=torch.long),
            torch.zeros(0, cfg.temporal.edge_dim, device="cuda"),
            torch.zeros(0, device="cuda", dtype=torch.long),
            torch.zeros(0, 3, device="cuda"),
            torch.zeros(0, cfg.temporal.status_dim, device="cuda"),
            torch.zeros(2, 0, device="cuda", dtype=torch.long),
            torch.zeros(0, cfg.temporal.hypothesis_edge_dim, device="cuda"),
            torch.zeros(0, device="cuda", dtype=torch.long),
            bypass_coreasoning=True,
        )
    assert output.d0_features is not None
    assert output.d0_features.dtype == torch.float16
    proposal = torch.nonzero(
        output.query_types[0] == QUERY_SPATIAL_PROPOSAL, as_tuple=False
    ).flatten()[0]
    with torch.no_grad():
        rendered = model.render_masks(
            output, [proposal[None]]
        )[0]
    assert rendered.shape == (1, 16, 16, 16)
    assert rendered.dtype == torch.float16
    assert torch.isfinite(rendered).all()
    assert all(
        parameter.dtype == torch.float32
        for parameter in model.local_mask_decoder.parameters()
    )


def test_postprocess_uses_anchor_seed_only_for_spatial_proposals() -> None:
    output = _render_output()
    output.query_types = torch.tensor(
        [[QUERY_SPATIAL_PROPOSAL, QUERY_TEMPORAL, QUERY_DISCOVERY]]
    )
    output.query_initial_references_cellscale[0, :2, 2] = -0.5
    output.centers_cellscale[0, :2, 2] = 0.5
    masks = torch.full((3, 1, 1, 3), -20.0)
    masks[:, 0, 0, [0, 2]] = 20.0

    class FakeModel:
        def render_masks(self, _outputs, _selected):
            return [masks]

    seeds: list[np.ndarray] = []

    def record(mask, center, _minimum):
        seeds.append(center.copy())
        return mask

    with patch(
        "learned.stirnet.inference.postprocess._component_near_center",
        side_effect=record,
    ):
        postprocess_batch(
            FakeModel(),
            output,
            render_threshold=0.1,
            final_exist_threshold=0.1,
            min_mask_voxels=1,
        )
    assert seeds[0].tolist() == [0.0, 0.0, 0.0]
    assert seeds[1].tolist() == [0.0, 0.0, 2.0]


def test_old_checkpoint_initializes_local_decoder_and_migrates_config(tmp_path) -> None:
    model = StirNet(_small_config())
    initialized = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.startswith("local_mask_decoder.")
    }
    legacy = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("local_mask_decoder.")
    }
    path = tmp_path / "legacy_v1.pt"
    torch.save(
        {
            "model": legacy,
            "config": {"decoder": {"proposal_center_step_dref": 0.5}},
        },
        path,
    )
    checkpoint = load_checkpoint(path, model, strict=True)
    for name, expected in initialized.items():
        torch.testing.assert_close(model.state_dict()[name], expected)
    assert (
        checkpoint["config"]["decoder"]["proposal_center_max_offset_dref"]
        == 0.5
    )
    assert any(
        "local_mask_decoder" in note
        for note in checkpoint["model_migration"]
    )
    migrated, notes = migrate_stirnet_checkpoint_config(
        {"decoder": {"proposal_center_step_dref": 0.25}}
    )
    assert migrated["decoder"]["proposal_center_max_offset_dref"] == 0.25
    assert notes


def test_local_mask_training_cap_config_loads_and_defaults_for_old_configs() -> None:
    assert config_from_dict({"local_masks": {}}).local_masks.train_max_queries_per_batch == 2
    assert (
        config_from_dict(
            {"local_masks": {"train_max_queries_per_batch": 1}}
        ).local_masks.train_max_queries_per_batch
        == 1
    )


def test_curriculum_bootstraps_only_local_mask_and_joint_unfreezes_backbone() -> None:
    cfg = _small_config()
    cfg.curriculum.enabled = True
    cfg.curriculum.native_bootstrap_steps = 1
    native = curriculum_stage(cfg.curriculum, 0)
    assert native.trainable_groups == frozenset({"local_mask"})
    assert native.lr_scales["local_mask"] == 1.0
    assert native.lr_scales["spatial"] == 0.0
    assert native.loss_weight_overrides["dice_coarse"] == 0.0
    assert "dice_hi" not in native.loss_weight_overrides
    joint = curriculum_stage(cfg.curriculum, 1)
    assert joint.lr_scales["local_mask"] == 1.0
    assert joint.lr_scales["spatial"] == cfg.curriculum.joint_spatial_lr_scale
    model = StirNet(cfg)
    groups = model_parameter_groups(model)
    assert groups["local_mask"]
    local_ids = {id(value) for value in groups["local_mask"]}
    assert all(
        name.startswith("local_mask_decoder.")
        for name, parameter in model.named_parameters()
        if id(parameter) in local_ids
    )
