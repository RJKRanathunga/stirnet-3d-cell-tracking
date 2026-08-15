from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F

from learned.stirnet import StirNet, StirNetConfig
from learned.stirnet.model.config import (
    CurriculumConfig,
    LossConfig,
    QueryConfig,
)
from learned.stirnet.model.losses import RefinementCriterion
from learned.stirnet.model.matcher import HungarianMatcher3D, MatchResult
from learned.stirnet.model.query_builder import (
    InstanceQueryBuilder,
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
)
from learned.stirnet.model.types import TemporalState
from learned.stirnet.training.curriculum import (
    CurriculumController,
    curriculum_stage,
    model_parameter_groups,
    optimizer_parameter_groups,
    stage_name_for_step,
)


def _empty_temporal(d_model: int) -> TemporalState:
    return TemporalState(
        tokens=torch.zeros(0, d_model),
        ref_um=torch.zeros(0, 3),
        ref_cellscale=torch.zeros(0, 3),
        salience=torch.zeros(0, 1),
        reliability=torch.zeros(0, 1),
        status=torch.zeros(0, 10),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 8),
        batch_index=torch.zeros(0, dtype=torch.long),
    )


def _query_fixture(*, max_queries: int | None = None):
    cfg = QueryConfig(
        d_model=8,
        discovery_queries=0,
        instance_feature_dim=5,
        split_companions_per_instance=1,
        max_split_companions_per_instance=4,
        max_queries=max_queries,
    )
    builder = InstanceQueryBuilder(cfg, feature_channels=2)
    labels = torch.zeros(2, 1, 1, 12, dtype=torch.long)
    labels[0, 0, 0, :2] = 1
    labels[0, 0, 0, 2:4] = 2
    labels[0, 0, 0, 4:12] = 3
    labels[1, 0, 0, 0] = 4
    labels[1, 0, 0, 1] = 5
    instance_ids = torch.tensor([1, 2, 3, 4, 5])
    instance_batch = torch.tensor([0, 0, 0, 1, 1])
    state = builder(
        torch.zeros(2, 2, 1, 1, 12),
        torch.ones(2, 3),
        labels,
        torch.zeros(5, 5),
        instance_ids,
        instance_batch,
        torch.zeros(5, 3),
        torch.ones(2),
        _empty_temporal(8),
    )
    return builder, labels, instance_ids, instance_batch, state


def test_split_capacity_uses_robust_volume_ratio_and_honors_baseline() -> None:
    builder, labels, instance_ids, instance_batch, state = _query_fixture()
    counts = builder.split_companion_counts(labels, instance_ids, instance_batch)
    assert counts.tolist() == [1, 1, 3, 1, 1]

    # Batch 0 has 3 primary + 5 split queries; batch 1 has 2 + 2 and is padded.
    assert state.embeddings.shape[:2] == (2, 8)
    assert state.padding_mask[0].tolist() == [False] * 8
    assert state.padding_mask[1].tolist() == [False] * 4 + [True] * 4
    assert state.query_types[0].tolist() == [
        QUERY_PRIMARY,
        QUERY_PRIMARY,
        QUERY_PRIMARY,
        QUERY_SPLIT,
        QUERY_SPLIT,
        QUERY_SPLIT,
        QUERY_SPLIT,
        QUERY_SPLIT,
    ]
    assert state.source_instance_ids[0].tolist() == [1, 2, 3, 1, 2, 3, 3, 3]

    # Companions for one source receive distinct learned slot identities.
    source_three_splits = state.embeddings[0, [5, 6, 7]]
    assert not torch.allclose(source_three_splits[0], source_three_splits[1])
    assert not torch.allclose(source_three_splits[1], source_three_splits[2])


def test_split_capacity_respects_explicit_maximum_and_query_safety_limit() -> None:
    cfg = QueryConfig(
        d_model=8,
        discovery_queries=0,
        instance_feature_dim=5,
        split_companions_per_instance=2,
        max_split_companions_per_instance=3,
    )
    builder = InstanceQueryBuilder(cfg, feature_channels=2)
    labels = torch.zeros(1, 1, 1, 24, dtype=torch.long)
    labels[0, 0, 0, :2] = 1
    labels[0, 0, 0, 2:4] = 2
    labels[0, 0, 0, 4:] = 3
    counts = builder.split_companion_counts(
        labels, torch.tensor([1, 2, 3]), torch.zeros(3, dtype=torch.long)
    )
    assert counts.tolist() == [2, 2, 3]

    try:
        _query_fixture(max_queries=7)
    except RuntimeError as error:
        assert "explicit safety limit" in str(error)
    else:
        raise AssertionError("max_queries did not reject an undersized limit")


def test_eligible_assignment_maximizes_cardinality_before_cost() -> None:
    cost = torch.tensor([[0.0, 100.0], [1.0, 0.0]])
    eligible = torch.tensor([[True, True], [True, False]])
    rows, columns = HungarianMatcher3D._eligible_assignment(cost, eligible)
    assert dict(zip(rows.tolist(), columns.tolist())) == {0: 1, 1: 0}


def _structured_assignment(
    query_types: list[int],
    source_ids: list[int],
    initial_references: list[list[float]],
    final_centers: list[list[float]],
    gt_centers: list[list[float]],
    cost: list[list[float]],
    *,
    target_source_ids: list[int] | None = None,
    source_gt_overlap: list[list[int]] | None = None,
):
    matcher = HungarianMatcher3D(
        temporal_match_radius_dref=1.0,
        discovery_match_radius_dref=1.5,
    )
    query_count = len(query_types)
    target_count = len(gt_centers)
    target_source_ids = target_source_ids or []
    source_gt_overlap = source_gt_overlap or []
    return matcher._structured_assignment(
        torch.tensor(cost, dtype=torch.float32),
        torch.arange(query_count),
        torch.arange(target_count),
        torch.tensor(query_types),
        torch.tensor(source_ids),
        torch.tensor(initial_references, dtype=torch.float32),
        torch.tensor(final_centers, dtype=torch.float32),
        torch.tensor(gt_centers, dtype=torch.float32),
        {
            "ids": torch.arange(1, target_count + 1),
            "source_ids": torch.tensor(target_source_ids),
            "source_gt_overlap": torch.tensor(
                source_gt_overlap, dtype=torch.long
            ).reshape(len(target_source_ids), target_count),
        },
    )


def test_temporal_gating_uses_initial_reference_and_allows_no_match() -> None:
    rows, columns = _structured_assignment(
        [QUERY_TEMPORAL, QUERY_TEMPORAL],
        [-1, -1],
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        [[20.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.8, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[10.0, 0.0], [0.0, 0.0]],
    )
    assert rows.tolist() == [0]
    assert columns.tolist() == [0]


def test_discovery_gating_uses_final_center_and_only_temporal_leftovers() -> None:
    rows, columns = _structured_assignment(
        [QUERY_TEMPORAL, QUERY_DISCOVERY, QUERY_DISCOVERY],
        [-1, -1, -1],
        [[0.2, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        [[20.0, 0.0, 0.0], [0.1, 0.0, 0.0], [3.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [[100.0, 100.0], [0.0, 100.0], [100.0, 0.0]],
    )
    # B1 owns target 0 despite its high cost; B2 sees only target 1.
    assert dict(zip(rows.tolist(), columns.tolist())) == {0: 0, 2: 1}


def test_discovery_outside_radius_is_ineligible_and_gt_can_stay_unmatched() -> None:
    rows, columns = _structured_assignment(
        [QUERY_DISCOVERY, QUERY_DISCOVERY],
        [-1, -1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.4, 0.0, 0.0], [10.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        [[10.0, 0.0], [0.0, 0.0]],
    )
    assert rows.tolist() == [0]
    assert columns.tolist() == [0]


def test_all_matching_stages_are_one_to_one_and_recover_all_candidates() -> None:
    rows, columns = _structured_assignment(
        [QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL, QUERY_DISCOVERY],
        [9, 9, -1, -1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0], [6.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        [[0.0, 10.0, 10.0, 10.0], [10.0, 0.0, 10.0, 10.0], [10.0, 10.0, 0.0, 10.0], [10.0, 10.0, 10.0, 0.0]],
        target_source_ids=[9],
        source_gt_overlap=[[1, 1, 0, 0]],
    )
    assert rows.tolist() == [0, 1, 2, 3]
    assert columns.tolist() == [0, 1, 2, 3]


def test_count_loss_targets_matched_positive_queries_not_raw_gt_count() -> None:
    criterion = RefinementCriterion(LossConfig(), QueryConfig())
    logits = torch.full((1, 5), torch.logit(torch.tensor(0.6)))
    padding = torch.zeros_like(logits, dtype=torch.bool)
    three_matches = [MatchResult(torch.arange(3), torch.arange(3))]
    loss = criterion._count_loss(logits, padding, three_matches)
    expected = F.smooth_l1_loss(
        logits.sigmoid().sum(-1), torch.tensor([3.0]), reduction="none"
    ).mean() / 3.0
    torch.testing.assert_close(loss, expected)
    assert float(loss) < 1e-6

    five_matches = [MatchResult(torch.arange(5), torch.arange(5))]
    forced_five_loss = criterion._count_loss(logits, padding, five_matches)
    assert float(forced_five_loss) > float(loss)


def test_count_loss_is_unchanged_when_every_gt_is_matched() -> None:
    criterion = RefinementCriterion(LossConfig(), QueryConfig())
    logits = torch.tensor([[0.2, -0.4, 0.8, 1.1, -0.1]])
    padding = torch.zeros_like(logits, dtype=torch.bool)
    matches = [MatchResult(torch.arange(5), torch.arange(5))]
    actual = criterion._count_loss(logits, padding, matches)
    old_gt_count_result = (
        F.smooth_l1_loss(
            logits.sigmoid().sum(-1), torch.tensor([5.0]), reduction="none"
        )
        / 5.0
    ).mean()
    torch.testing.assert_close(actual, old_gt_count_result)


def _reduced_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.spatial.channels = (4, 8, 8, 16)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.mask_dim = 4
    cfg.temporal.d_model = 16
    cfg.temporal.graph_ffn_dim = 32
    cfg.coreasoning.d_model = 16
    cfg.coreasoning.dropout = 0.0
    cfg.queries.d_model = 16
    cfg.queries.discovery_queries = 1
    cfg.decoder.d_model = 16
    cfg.decoder.ffn_dim = 32
    cfg.decoder.dropout = 0.0
    cfg.decoder.mask_dim = 4
    cfg.decoder.max_spatial_tokens = 64
    cfg.training.activation_checkpointing = False
    return cfg


def test_curriculum_transitions_preserve_model_optimizer_and_named_state() -> None:
    cfg = _reduced_config()
    cfg.curriculum = CurriculumConfig(
        enabled=True,
        spatial_dense_steps=1,
        temporal_dense_steps=1,
        query_bootstrap_steps=1,
        native_bootstrap_steps=1,
        joint_spatial_lr_scale=0.1,
        joint_dense_lr_scale=0.5,
    )
    model = StirNet(cfg)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, cfg.training.lr),
        lr=cfg.training.lr,
    )
    controller = CurriculumController(
        model, optimizer, cfg.curriculum, cfg.training.lr
    )
    model_identity = id(model)
    optimizer_identity = id(optimizer)
    group_identities = [id(group) for group in optimizer.param_groups]
    expected_names = [
        "spatial_dense",
        "temporal_dense",
        "query_bootstrap",
        "native_bootstrap",
        "joint",
    ]
    expected_trainable = [
        {"spatial", "dense", "proposal"},
        {"spatial", "dense", "proposal", "temporal"},
        {"spatial", "dense", "proposal", "temporal", "query"},
        {"local_mask"},
        {
            "spatial", "dense", "proposal", "temporal", "query", "native",
            "local_mask",
        },
    ]
    parameter_groups = model_parameter_groups(model)
    for step, (name, trainable) in enumerate(
        zip(expected_names, expected_trainable)
    ):
        stage = controller.apply(step)
        assert stage.name == name
        assert id(model) == model_identity
        assert id(optimizer) == optimizer_identity
        assert [id(group) for group in optimizer.param_groups] == group_identities
        for group_name, parameters in parameter_groups.items():
            assert all(
                parameter.requires_grad == (group_name in trainable)
                for parameter in parameters
            )
        learning_rates = {
            group["name"]: group["lr"] for group in optimizer.param_groups
        }
        assert learning_rates == {
            group: cfg.training.lr * stage.lr_scales[group]
            for group in learning_rates
        }

    final_stage = controller.current
    assert final_stage is not None
    assert final_stage.lr_scales["spatial"] == 0.1
    assert final_stage.lr_scales["dense"] == 0.5
    assert final_stage.lr_scales["proposal"] == 0.5
    assert final_stage.lr_scales["local_mask"] == 1.0


def test_curriculum_loss_gates_and_disabled_mode_preserve_legacy_behavior() -> None:
    cfg = CurriculumConfig(
        enabled=True,
        spatial_dense_steps=1,
        temporal_dense_steps=1,
        query_bootstrap_steps=1,
        native_bootstrap_steps=1,
    )
    assert [stage_name_for_step(cfg, step) for step in range(5)] == [
        "spatial_dense",
        "temporal_dense",
        "query_bootstrap",
        "native_bootstrap",
        "joint",
    ]
    stage_one = curriculum_stage(cfg, 0)
    assert stage_one.bypass_coreasoning
    assert stage_one.loss_weight_overrides["exist"] == 0
    assert stage_one.loss_weight_overrides["aux_layer"] == 0
    stage_three = curriculum_stage(cfg, 2)
    assert stage_three.loss_weight_overrides["dice_hi"] == 0
    assert stage_three.loss_weight_overrides["count"] == 0
    stage_four = curriculum_stage(cfg, 3)
    assert "dice_hi" not in stage_four.loss_weight_overrides
    assert stage_four.loss_weight_overrides["count"] == 0

    criterion = RefinementCriterion(LossConfig(), QueryConfig())
    criterion.set_loss_weight_overrides(stage_one.loss_weight_overrides)
    stage_one_weights = criterion._effective_loss_weights()
    assert stage_one_weights["foreground"] > 0
    assert stage_one_weights["boundary"] > 0
    assert stage_one_weights["center_heatmap"] > 0
    assert all(
        stage_one_weights[name] == 0
        for name in (
            "exist", "dice_hi", "focal_hi", "dice_coarse",
            "focal_coarse", "center", "count", "overlap", "aux_layer",
        )
    )
    criterion.set_loss_weight_overrides(stage_three.loss_weight_overrides)
    stage_three_weights = criterion._effective_loss_weights()
    assert stage_three_weights["exist"] > 0
    assert stage_three_weights["center"] > 0
    assert stage_three_weights["dice_coarse"] > 0
    assert stage_three_weights["dice_hi"] == 0
    assert stage_three_weights["count"] == 0
    criterion.set_loss_weight_overrides(stage_four.loss_weight_overrides)
    stage_four_weights = criterion._effective_loss_weights()
    assert stage_four_weights["dice_hi"] > 0
    assert stage_four_weights["count"] == 0
    criterion.set_loss_weight_overrides(
        curriculum_stage(cfg, 4).loss_weight_overrides
    )
    assert criterion._effective_loss_weights()["count"] > 0

    legacy = curriculum_stage(CurriculumConfig(enabled=False), 100)
    assert legacy.name == "legacy"
    assert not legacy.bypass_coreasoning
    assert legacy.loss_weight_overrides == {}
    assert all(scale == 1.0 for scale in legacy.lr_scales.values())
    criterion.set_loss_weight_overrides(legacy.loss_weight_overrides)
    assert criterion._effective_loss_weights()["count"] == LossConfig().count


def test_stage_one_bypass_does_not_execute_coreasoning_blocks() -> None:
    cfg = _reduced_config()
    model = StirNet(cfg).eval()
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[:, 2:6, 2:6, 2:6] = 1
    inputs = (
        torch.zeros(1, 5, 8, 8, 8),
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
    )
    with (
        patch.object(
            model.cr1, "forward", side_effect=AssertionError("cr1 executed")
        ),
        patch.object(
            model.cr2, "forward", side_effect=AssertionError("cr2 executed")
        ),
        torch.no_grad(),
    ):
        output = model(*inputs, bypass_coreasoning=True)
    assert torch.isfinite(output.exist_logits).all()
