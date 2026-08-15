from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from learned.stirnet.data.targets import build_gt_targets
from learned.stirnet.model.config import (
    DecoderConfig,
    ProposalConfig,
    QueryConfig,
    StirNetConfig,
)
from learned.stirnet.model.losses import RefinementCriterion
from learned.stirnet.model.matcher import HungarianMatcher3D
from learned.stirnet.model.native_masks import native_query_prior_and_support
from learned.stirnet.model.query_builder import (
    InstanceQueryBuilder,
    QUERY_PRIMARY,
    QUERY_SPATIAL_PROPOSAL,
    QUERY_SPLIT,
)
from learned.stirnet.model.query_decoder import InstanceQueryDecoder
from learned.stirnet.model.spatial_proposals import SpatialProposalGenerator
from learned.stirnet.model.stir_net import StirNet
from learned.stirnet.model.types import QueryState, TemporalState
from learned.stirnet.training.checkpoint import migrate_history_checkpoint_state_dict


def _empty_temporal(d_model: int) -> TemporalState:
    return TemporalState(
        tokens=torch.zeros(0, d_model),
        ref_um=torch.zeros(0, 3),
        ref_cellscale=torch.zeros(0, 3),
        salience=torch.zeros(0, 1),
        reliability=torch.zeros(0, 1),
        status=torch.zeros(0, 10),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 22),
        batch_index=torch.zeros(0, dtype=torch.long),
    )


def _proposal_fixture(*, fallback_only: bool = False):
    cfg = ProposalConfig(
        max_proposals=4,
        candidate_pool_size=8,
        nms_radius_dref=0.4,
        inference_score_threshold=0.999 if fallback_only else 0.9,
        local_grid_size=3,
        local_extent_dref=1.0,
        local_dim=8,
    )
    generator = SpatialProposalGenerator(
        cfg, d0_channels=2, e2_channels=3
    ).eval()
    shape = (3, 5, 7)
    labels = torch.full((1, *shape), 9, dtype=torch.long)
    d0 = torch.randn(1, 2, *shape)
    e2 = torch.randn(1, 3, *shape)
    inputs = torch.randn(1, 5, *shape)
    center = torch.full((1, 1, *shape), -12.0)
    if not fallback_only:
        center[0, 0, 1, 2, 2] = 12.0
        center[0, 0, 1, 2, 4] = 11.0
    dense = {
        "foreground_logits": torch.zeros_like(center),
        "center_heatmap_logits": center,
        "boundary_logits": torch.zeros_like(center),
    }
    state, score_logits = generator(
        d0,
        e2,
        inputs,
        dense,
        labels,
        torch.ones(1, 3),
        torch.ones(1, 3),
        torch.ones(1),
        torch.tensor([9]),
        torch.tensor([0]),
        torch.zeros(1, 3),
    )
    return generator, state, score_logits


def test_two_proposals_in_one_source_have_distinct_spatial_identity() -> None:
    torch.manual_seed(3)
    _, state, _ = _proposal_fixture()
    valid = torch.nonzero(~state.padding_mask[0], as_tuple=False).flatten()
    learned = valid[~state.fallback_mask[0, valid]]
    assert len(learned) == 2
    assert state.source_instance_ids[0, learned].tolist() == [9, 9]
    assert not torch.equal(
        state.references_cellscale[0, learned[0]],
        state.references_cellscale[0, learned[1]],
    )
    assert not torch.allclose(
        state.embeddings[0, learned[0]], state.embeddings[0, learned[1]]
    )


def test_nms_uses_physical_anisotropic_distance() -> None:
    cfg = ProposalConfig(
        max_proposals=4,
        candidate_pool_size=8,
        nms_radius_dref=1.0,
        inference_score_threshold=0.9,
        local_dim=8,
    )
    generator = SpatialProposalGenerator(
        cfg, d0_channels=2, e2_channels=3
    ).eval()
    logits = torch.full((3, 3, 3), -12.0)
    logits[1, 1, 1] = 12.0
    logits[2, 1, 1] = 11.0  # 3 um away in Z: outside the 2 um radius.
    logits[1, 2, 1] = 10.0  # 1 um away in Y: suppressed.
    selected = generator._learned_centers(
        logits,
        torch.tensor([3.0, 1.0, 1.0]),
        torch.tensor(2.0),
        None,
        4,
    )
    points = {tuple(point) for point in selected.tolist()}
    assert (1, 1, 1) in points
    assert (2, 1, 1) in points
    assert (1, 2, 1) not in points


def test_proposal_decoder_support_does_not_union_whole_source() -> None:
    decoder = InstanceQueryDecoder(
        (2, 2, 2),
        DecoderConfig(d_model=8, heads=2, ffn_dim=16, mask_dim=4),
        QueryConfig(d_model=8),
        proposal_cfg=ProposalConfig(
            local_dim=8,
            attention_radius_layer0_dref=1.0,
            attention_radius_layer1_dref=1.5,
            attention_radius_layer2_dref=2.0,
        ),
    )
    references = torch.zeros(1, 1, 3)
    query = QueryState(
        embeddings=torch.zeros(1, 1, 8),
        references_cellscale=references,
        query_types=torch.tensor([[QUERY_SPATIAL_PROPOSAL]]),
        padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        source_instance_ids=torch.tensor([[9]]),
        temporal_salience=torch.zeros(1, 1, 1),
        temporal_reliability=torch.zeros(1, 1, 1),
        initial_references_cellscale=references.clone(),
    )
    positions = torch.tensor(
        [[[0.0, 0.0, -2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]]
    )
    radial = decoder._reference_support(query, positions, torch.ones(1), 0)
    whole_source = decoder._source_instance_support(
        query, torch.full((1, 1, 1, 5), 9), (1, 1, 5)
    ).flatten(2)
    assert not whole_source.any()
    assert radial[0, 0].tolist() == [False, True, True, True, False]


def test_off_mask_proposal_can_render_without_current_foreground() -> None:
    coords = torch.tensor(
        [[0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    )
    prior, support = native_query_prior_and_support(
        torch.tensor([QUERY_SPATIAL_PROPOSAL]),
        torch.tensor([-1]),
        torch.zeros(1, 3),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(1, 3, dtype=torch.bool),
        coords,
        torch.tensor(1.0),
        support_radius_dref=1.0,
        proposal_support_radius_dref=1.5,
        temporal_sigma_dref=0.75,
        prior_inside_logit=1.5,
        prior_outside_logit=-1.5,
    )
    assert support.tolist() == [[False, True, False]]
    torch.testing.assert_close(prior, torch.zeros_like(prior))


def _structured_proposal_assignment(
    query_sources: list[int],
    initial_references: torch.Tensor,
    gt_centers: torch.Tensor,
    source_ids: list[int],
    overlap: torch.Tensor,
):
    query_count = len(query_sources)
    target_count = len(gt_centers)
    matcher = HungarianMatcher3D(proposal_match_radius_dref=1.0)
    return matcher._structured_assignment(
        torch.eye(query_count, target_count),
        torch.arange(query_count),
        torch.arange(target_count),
        torch.full((query_count,), QUERY_SPATIAL_PROPOSAL),
        torch.tensor(query_sources),
        initial_references,
        initial_references,
        gt_centers,
        {
            "ids": torch.arange(1, target_count + 1),
            "source_ids": torch.tensor(source_ids),
            "source_gt_overlap": overlap.reshape(len(source_ids), target_count),
        },
    )


def test_off_mask_proposal_matches_near_initial_anchor() -> None:
    rows, columns = _structured_proposal_assignment(
        [-1],
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.8]]),
        [],
        torch.zeros(0, 1, dtype=torch.long),
    )
    assert rows.tolist() == [0]
    assert columns.tolist() == [0]


def test_false_positive_source_proposal_can_remain_unmatched() -> None:
    rows, columns = _structured_proposal_assignment(
        [5],
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        [5],
        torch.zeros(1, 1, dtype=torch.long),
    )
    assert rows.numel() == columns.numel() == 0


def test_nine_proposals_from_source_nine_match_nine_cells() -> None:
    centers = torch.stack(
        [torch.tensor([0.0, 0.0, float(index)]) for index in range(9)]
    )
    rows, columns = _structured_proposal_assignment(
        [9] * 9,
        centers.clone(),
        centers,
        [9],
        torch.ones(1, 9, dtype=torch.long),
    )
    assert len(rows) == len(columns) == 9
    assert rows.unique().numel() == columns.unique().numel() == 9


def test_uncovered_component_gets_exactly_one_fallback() -> None:
    _, state, _ = _proposal_fixture(fallback_only=True)
    valid = ~state.padding_mask[0]
    assert int(valid.sum()) == 1
    assert int(state.fallback_mask[0, valid].sum()) == 1
    assert state.source_instance_ids[0, valid].tolist() == [9]


def _small_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.spatial.channels = (4, 8, 8, 16)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.mask_dim = 4
    cfg.temporal.d_model = 16
    cfg.temporal.graph_ffn_dim = 32
    cfg.temporal.memory_ffn_dim = 32
    cfg.coreasoning.d_model = 16
    cfg.coreasoning.dropout = 0.0
    cfg.queries.d_model = 16
    cfg.queries.discovery_queries = 0
    cfg.decoder.d_model = 16
    cfg.decoder.ffn_dim = 32
    cfg.decoder.mask_dim = 4
    cfg.decoder.dropout = 0.0
    cfg.decoder.max_spatial_tokens = 64
    cfg.proposals.local_dim = 16
    cfg.proposals.max_proposals = 4
    cfg.proposals.candidate_pool_size = 16
    cfg.proposals.nms_radius_dref = 0.75
    cfg.training.activation_checkpointing = False
    return cfg


def test_old_four_row_query_embedding_checkpoint_migrates() -> None:
    model = StirNet(_small_config())
    current = model.state_dict()
    old = {
        key: value.clone()
        for key, value in current.items()
        if not key.startswith("spatial_proposal_generator.")
        and not key.startswith("query_builder.proposal_proj.")
        and not key.startswith("query_builder.proposal_score_proj.")
        and not key.startswith("query_builder.component_context_gate.")
    }
    old_rows = current["query_builder.type_embedding.weight"][:4].clone()
    old["query_builder.type_embedding.weight"] = old_rows
    migrated, notes = migrate_history_checkpoint_state_dict(model, old)
    model.load_state_dict(migrated, strict=True)
    loaded = model.query_builder.type_embedding.weight.detach()
    torch.testing.assert_close(loaded[:4], old_rows)
    torch.testing.assert_close(loaded[4], 0.5 * (old_rows[0] + old_rows[1]))
    assert any("4 -> 5 query types" in note for note in notes)
    assert any("spatial_proposal_generator" in note for note in notes)


def test_legacy_query_mode_preserves_primary_and_split_builder() -> None:
    cfg = QueryConfig(
        d_model=8,
        instance_feature_dim=5,
        discovery_queries=0,
        split_companions_per_instance=1,
        max_split_companions_per_instance=1,
    )
    builder = InstanceQueryBuilder(cfg, feature_channels=2)
    labels = torch.ones(1, 1, 1, 3, dtype=torch.long)
    state = builder(
        torch.zeros(1, 2, 1, 1, 3),
        torch.ones(1, 3),
        labels,
        torch.zeros(1, 5),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.zeros(1, 3),
        torch.ones(1),
        _empty_temporal(8),
        query_mode="legacy",
    )
    assert state.query_types[0].tolist() == [QUERY_PRIMARY, QUERY_SPLIT]


def _small_forward(model: StirNet):
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[:, 2:6, 2:6, 2:6] = 1
    spatial = torch.randn(1, 5, 8, 8, 8)
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
        torch.zeros(0, model.cfg.temporal.node_dim),
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, model.cfg.temporal.edge_dim),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, 3),
        torch.zeros(0, model.cfg.temporal.status_dim),
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, model.cfg.temporal.hypothesis_edge_dim),
        torch.zeros(0, dtype=torch.long),
        bypass_coreasoning=True,
        return_debug=True,
    )
    return output, labels


def test_spatial_proposal_mode_runs_without_temporal_coreasoning() -> None:
    model = StirNet(_small_config()).eval()
    with torch.no_grad():
        output, _ = _small_forward(model)
    assert (output.query_types == QUERY_SPATIAL_PROPOSAL).any()
    assert output.proposals is not None
    assert output.debug is not None
    assert output.debug["query_mode"] == "spatial_proposals"
    assert torch.isfinite(output.exist_logits).all()


def test_forward_criterion_backward_gives_new_modules_finite_gradients() -> None:
    torch.manual_seed(11)
    cfg = _small_config()
    model = StirNet(cfg).train()
    output, labels = _small_forward(model)
    target = build_gt_targets(
        labels[0].numpy(), (1.0, 1.0, 1.0), 4.0, current_labels=labels[0].numpy()
    )
    criterion = RefinementCriterion(
        cfg.losses, cfg.queries, cfg.training, cfg.proposals
    )
    losses = criterion(
        output, [target], local_mask_decoder=model.local_mask_decoder
    )
    losses["loss"].backward()
    for parameter in (
        model.spatial_proposal_generator.from_d0.weight,
        model.spatial_proposal_generator.local_encoder[0].weight,
        model.query_builder.proposal_proj.weight,
        model.local_mask_decoder.spatial[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.isfinite(losses["proposal_center"])
    assert torch.isfinite(losses["internal_boundary"])


def test_proposal_gathering_never_concatenates_native_feature_tensors() -> None:
    with patch("torch.cat", wraps=torch.cat) as concatenate:
        _, state, score_logits = _proposal_fixture()
    assert score_logits.shape[1] == 1
    assert state.embeddings.ndim == 3
    for call in concatenate.call_args_list:
        tensors = call.args[0]
        assert not any(torch.is_tensor(value) and value.ndim == 5 for value in tensors)
