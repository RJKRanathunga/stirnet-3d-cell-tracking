from __future__ import annotations

import numpy as np
import pytest
import torch

from learned.stirnet.data.targets import build_gt_targets
from learned.stirnet.model.config import DecoderConfig, LossConfig, QueryConfig
from learned.stirnet.model.heads import CenterHead
from learned.stirnet.model.losses import RefinementCriterion, local_matched_mask_losses
from learned.stirnet.model.matcher import (
    HungarianMatcher3D,
    build_cost_matrix,
    build_local_support_masks,
)
from learned.stirnet.model.native_masks import (
    compose_native_query_logits,
    native_chunk_coordinates_um,
    native_query_prior_and_support,
    temporal_gaussian_prior_logits,
)
from learned.stirnet.model.query_builder import (
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
)
from learned.stirnet.model.query_decoder import QueryDecoderLayer
from learned.stirnet.model.types import QueryState, StirNetOutput


def _structured_match(
    current: np.ndarray,
    ground_truth: np.ndarray,
    query_types: list[int],
    source_ids: list[int],
):
    target = build_gt_targets(
        ground_truth,
        (1.0, 1.0, 1.0),
        1.0,
        current_labels=current,
    )
    query_count = len(query_types)
    coarse_shape = tuple(int(v) for v in ground_truth.shape)
    output = {
        "exist_logits": torch.zeros(1, query_count),
        "coarse_mask_logits": torch.zeros(1, query_count, *coarse_shape),
        "centers_cellscale": torch.zeros(1, query_count, 3),
        "coarse_spacing_um": torch.ones(1, 3),
        "dref_um": torch.ones(1),
        "query_types": torch.tensor([query_types]),
        "source_instance_ids": torch.tensor([source_ids]),
    }
    result = HungarianMatcher3D()(
        output,
        torch.zeros(1, query_count, dtype=torch.bool),
        [target],
    )[0]
    gt_ids = target["ids"][result.target_indices.cpu()]
    return result.pred_indices.cpu(), gt_ids.cpu(), target


def test_default_overlap_is_disabled_and_skips_overlap_computation() -> None:
    assert LossConfig().overlap == 0.0
    output, target = _criterion_fixture()
    criterion = _RecordingCriterion(LossConfig(), QueryConfig())

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("disabled overlap objective was evaluated")

    criterion._overlap_loss = fail_if_called  # type: ignore[method-assign]
    losses = criterion(output, [target])
    assert torch.isfinite(losses["overlap"])
    assert float(losses["overlap"]) == 0.0


def test_one_gt_source_activates_primary_but_not_split() -> None:
    current = np.zeros((1, 1, 4), np.int32)
    ground_truth = np.zeros_like(current)
    current[..., 0] = 10
    ground_truth[..., 0] = 3
    ground_truth[..., 3] = 8
    queries, gt_ids, _ = _structured_match(
        current,
        ground_truth,
        [QUERY_PRIMARY, QUERY_SPLIT],
        [10, 10],
    )
    assert queries.tolist() == [0]
    assert gt_ids.tolist() == [3]


def test_merged_source_allows_primary_and_split_to_cover_two_gt() -> None:
    current = np.zeros((1, 1, 3), np.int32)
    ground_truth = np.zeros_like(current)
    current[..., :2] = 20
    ground_truth[..., 0] = 4
    ground_truth[..., 1] = 5
    queries, gt_ids, _ = _structured_match(
        current,
        ground_truth,
        [QUERY_PRIMARY, QUERY_SPLIT],
        [20, 20],
    )
    assert set(queries.tolist()) == {0, 1}
    assert set(gt_ids.tolist()) == {4, 5}


def test_source_with_zero_gt_overlap_stays_unmatched() -> None:
    current = np.zeros((1, 1, 4), np.int32)
    ground_truth = np.zeros_like(current)
    current[..., 3] = 10
    ground_truth[..., 0] = 3
    queries, gt_ids, _ = _structured_match(
        current,
        ground_truth,
        [QUERY_PRIMARY, QUERY_SPLIT],
        [10, 10],
    )
    assert queries.numel() == 0
    assert gt_ids.numel() == 0


def test_oversegmented_sources_compete_for_one_gt() -> None:
    current = np.zeros((1, 1, 3), np.int32)
    ground_truth = np.zeros_like(current)
    current[..., 0] = 10
    current[..., 1] = 11
    ground_truth[..., :2] = 3
    queries, gt_ids, _ = _structured_match(
        current,
        ground_truth,
        [QUERY_PRIMARY, QUERY_PRIMARY],
        [10, 11],
    )
    assert queries.numel() == 1
    assert gt_ids.tolist() == [3]


def test_temporal_discovery_stage_matches_only_remaining_gt() -> None:
    current = np.zeros((1, 1, 4), np.int32)
    ground_truth = np.zeros_like(current)
    current[..., 0] = 10
    ground_truth[..., 0] = 3
    ground_truth[..., 3] = 8
    queries, gt_ids, _ = _structured_match(
        current,
        ground_truth,
        [QUERY_PRIMARY, QUERY_TEMPORAL, QUERY_DISCOVERY],
        [10, -1, -1],
    )
    mapping = dict(zip(queries.tolist(), gt_ids.tolist()))
    assert mapping[0] == 3
    assert set(mapping.values()) == {3, 8}
    assert len(mapping) == 2
    assert next(query for query, gt_id in mapping.items() if gt_id == 8) in {1, 2}


def test_every_seeded_assignment_is_source_compatible_in_mixed_case() -> None:
    current = np.array([[[20, 20, 30, 40, 0, 0]]], dtype=np.int32)
    ground_truth = np.array([[[4, 5, 6, 6, 7, 0]]], dtype=np.int32)
    query_types = [
        QUERY_PRIMARY,
        QUERY_SPLIT,
        QUERY_PRIMARY,
        QUERY_SPLIT,
        QUERY_PRIMARY,
        QUERY_SPLIT,
        QUERY_TEMPORAL,
        QUERY_DISCOVERY,
    ]
    source_ids = [20, 20, 30, 30, 40, 40, -1, -1]
    queries, gt_ids, target = _structured_match(
        current, ground_truth, query_types, source_ids
    )
    source_rows = {
        int(source): row for row, source in enumerate(target["source_ids"].tolist())
    }
    target_cols = {int(gt): col for col, gt in enumerate(target["ids"].tolist())}
    for query, gt_id in zip(queries.tolist(), gt_ids.tolist()):
        if query_types[query] in (QUERY_PRIMARY, QUERY_SPLIT):
            assert target["source_gt_overlap"][
                source_rows[source_ids[query]], target_cols[gt_id]
            ] > 0
    # Source 30 overlaps one GT, so its split companion cannot be positive.
    assert 3 not in queries.tolist()


class _CountingMatcher(HungarianMatcher3D):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return super().forward(*args, **kwargs)


class _RecordingCriterion(RefinementCriterion):
    def __init__(self, loss_cfg, query_cfg):
        super().__init__(loss_cfg, query_cfg)
        self.recorded_matches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _coarse_losses(self, out, matches, targets, coarse_targets):
        match = matches[0]
        self.recorded_matches.append(
            (match.pred_indices.detach().clone(), match.target_indices.detach().clone())
        )
        zero = out["exist_logits"].sum() * 0
        return zero, zero, zero

    def _native_mask_losses(self, outputs, targets, matches):
        zero = outputs.exist_logits.sum() * 0
        return zero, zero

    def _dense_losses(self, dense, targets):
        zero = next(iter(dense.values())).sum() * 0
        return zero, zero, zero


def _criterion_fixture() -> tuple[StirNetOutput, dict]:
    target = {
        "ids": torch.tensor([1, 2]),
        "label_map": torch.tensor([[[1, 2]]], dtype=torch.int32),
        "centers_cellscale": torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
    }
    final_masks = torch.tensor([[[[[8.0, -8.0]]], [[[-8.0, 8.0]]]]])
    final_centers = target["centers_cellscale"][None].clone()
    swapped_masks = final_masks.flip(1)
    swapped_centers = final_centers.flip(1)

    def decoder_output(masks, centers):
        return {
            "exist_logits": torch.zeros(1, 2),
            "centers_cellscale": centers,
            "coarse_mask_logits": masks,
            "query_embeddings": torch.zeros(1, 2, 4),
            "coarse_spacing_um": torch.ones(1, 3),
        }

    output = StirNetOutput(
        exist_logits=torch.zeros(1, 2),
        centers_cellscale=final_centers,
        coarse_mask_logits=final_masks,
        coarse_spacing_um=torch.ones(1, 3),
        query_embeddings=torch.zeros(1, 2, 4),
        native_mask_embeddings=torch.zeros(1, 2, 1),
        query_types=torch.tensor([[QUERY_TEMPORAL, QUERY_DISCOVERY]]),
        query_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        source_instance_ids=torch.full((1, 2), -1, dtype=torch.long),
        temporal_salience=torch.zeros(1, 2, 1),
        temporal_reliability=torch.zeros(1, 2, 1),
        aux_outputs=[
            decoder_output(swapped_masks, swapped_centers),
            decoder_output(swapped_masks, swapped_centers),
        ],
        dense_outputs={"dummy": torch.zeros(1)},
        mask_features=torch.zeros(1, 1, 1, 1, 2),
        spacing_um=torch.ones(1, 3),
        dref_um=torch.ones(1),
        instance_labels=torch.zeros(1, 1, 1, 2, dtype=torch.long),
    )
    return output, target


def test_final_assignment_is_reused_for_all_auxiliary_layers() -> None:
    output, target = _criterion_fixture()
    criterion = _RecordingCriterion(LossConfig(), QueryConfig())
    matcher = _CountingMatcher()
    criterion.matcher = matcher
    losses = criterion(output, [target])
    assert torch.isfinite(losses["loss"])
    assert matcher.calls == 1
    assert len(criterion.recorded_matches) == 3
    final_identity = criterion.recorded_matches[0]
    for identity in criterion.recorded_matches[1:]:
        torch.testing.assert_close(identity[0], final_identity[0])
        torch.testing.assert_close(identity[1], final_identity[1])


def test_local_mask_loss_ignores_remote_background_but_not_local_logits() -> None:
    target = torch.zeros(1, 1, 1, 9)
    target[..., 4] = 1
    centers = torch.tensor([[0.0, 0.0, 0.0]])
    support = build_local_support_masks(
        target, centers, torch.ones(3), torch.tensor(1.0), 1.0
    )
    logits = torch.zeros_like(target)
    baseline = local_matched_mask_losses(
        logits, target, support, alpha=0.75, gamma=2.0
    )
    remote = logits.clone()
    remote[..., 0] = 20
    remote_loss = local_matched_mask_losses(
        remote, target, support, alpha=0.75, gamma=2.0
    )
    local = logits.clone()
    local[..., 3] = 20
    local_loss = local_matched_mask_losses(
        local, target, support, alpha=0.75, gamma=2.0
    )
    torch.testing.assert_close(remote_loss[0], baseline[0])
    torch.testing.assert_close(remote_loss[1], baseline[1])
    assert not torch.isclose(local_loss[0], baseline[0])
    assert not torch.isclose(local_loss[1], baseline[1])


def test_local_support_never_crops_elongated_gt_positives() -> None:
    target = torch.zeros(1, 1, 1, 9)
    target[..., 0] = 1
    target[..., 4] = 1
    support = build_local_support_masks(
        target,
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.ones(3),
        torch.tensor(1.0),
        1.0,
    )
    assert support[target.bool()].all()
    assert support[..., 0]


def test_temporal_prior_has_exact_center_sigma_and_negative_far_field() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 100.0]])
    prior = temporal_gaussian_prior_logits(
        coords,
        torch.zeros(1, 3),
        torch.tensor(2.0),
        sigma_dref=1.0,
        inside_logit=1.5,
        outside_logit=-1.5,
    )[0]
    assert prior[0] == pytest.approx(1.5)
    assert prior[1] == pytest.approx(-1.5 + 3.0 * np.exp(-0.5), rel=1e-6)
    assert prior[2] == pytest.approx(-1.5)
    assert float(prior[2]) != 0.0


def test_native_shared_helper_is_chunk_equivalent_for_training_and_rendering() -> None:
    shape = (1, 1, 9)
    spacing = torch.ones(3)
    dref = torch.tensor(1.0)
    learned = torch.linspace(-1, 1, 9)[None]
    query_types = torch.tensor([QUERY_TEMPORAL])
    source_ids = torch.tensor([-1])
    refs = torch.zeros(1, 3)
    labels = torch.zeros(9, dtype=torch.long)
    full_coords = native_chunk_coordinates_um(shape, spacing, 0, 9)
    full, _, _ = compose_native_query_logits(
        learned,
        query_types,
        source_ids,
        refs,
        labels,
        torch.zeros(1, 9, dtype=torch.bool),
        full_coords,
        dref,
        support_radius_dref=1.5,
        temporal_sigma_dref=0.75,
        prior_inside_logit=1.5,
        prior_outside_logit=-1.5,
        background_logit=-20.0,
    )
    chunks = []
    for start, end in ((0, 4), (4, 9)):
        chunk, _, _ = compose_native_query_logits(
            learned[:, start:end],
            query_types,
            source_ids,
            refs,
            labels[start:end],
            torch.zeros(1, end - start, dtype=torch.bool),
            native_chunk_coordinates_um(shape, spacing, start, end),
            dref,
            support_radius_dref=1.5,
            temporal_sigma_dref=0.75,
            prior_inside_logit=1.5,
            prior_outside_logit=-1.5,
            background_logit=-20.0,
        )
        chunks.append(chunk)
    torch.testing.assert_close(torch.cat(chunks, dim=1), full)


def test_split_uses_source_support_without_whole_source_positive_prior() -> None:
    prior, support = native_query_prior_and_support(
        torch.tensor([QUERY_SPLIT]),
        torch.tensor([10]),
        torch.zeros(1, 3),
        torch.tensor([10, 10, 0]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        torch.tensor(1.0),
        support_radius_dref=1.5,
        temporal_sigma_dref=0.75,
        prior_inside_logit=1.5,
        prior_outside_logit=-1.5,
    )
    assert support.all()
    torch.testing.assert_close(prior, torch.zeros_like(prior))


def test_native_query_roles_use_bounded_role_specific_support_and_priors() -> None:
    query_types = torch.tensor(
        [QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL, QUERY_DISCOVERY]
    )
    source_ids = torch.tensor([10, 10, -1, -1])
    coords = torch.tensor(
        [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    source_support = torch.zeros(4, 5, dtype=torch.bool)
    source_support[:2, :2] = True
    prior, support = native_query_prior_and_support(
        query_types,
        source_ids,
        torch.zeros(4, 3),
        torch.tensor([0, 10, 0, 0, 0]),
        source_support,
        coords,
        torch.tensor(1.0),
        support_radius_dref=1.0,
        temporal_sigma_dref=0.75,
        prior_inside_logit=1.5,
        prior_outside_logit=-1.5,
    )
    assert support[QUERY_PRIMARY].tolist() == [True, True, True, True, False]
    assert support[QUERY_SPLIT].tolist() == [True, True, False, False, False]
    assert support[QUERY_TEMPORAL].tolist() == [False, True, True, True, False]
    assert support[QUERY_DISCOVERY].tolist() == [False, True, True, True, False]
    assert prior[QUERY_PRIMARY, 1] == pytest.approx(1.5)
    assert prior[QUERY_PRIMARY, 2] == pytest.approx(-1.5)
    torch.testing.assert_close(prior[QUERY_SPLIT], torch.zeros(5))
    torch.testing.assert_close(prior[QUERY_DISCOVERY], torch.zeros(5))


def _query_state(query_types: list[int], references: torch.Tensor) -> QueryState:
    count = len(query_types)
    return QueryState(
        embeddings=torch.randn(1, count, 8),
        references_cellscale=references.clone(),
        query_types=torch.tensor([query_types]),
        padding_mask=torch.zeros(1, count, dtype=torch.bool),
        source_instance_ids=torch.full((1, count), -1, dtype=torch.long),
        temporal_salience=torch.zeros(1, count, 1),
        temporal_reliability=torch.zeros(1, count, 1),
    )


def test_center_head_zero_initialization_preserves_fresh_decoder_references() -> None:
    head = CenterHead(8)
    torch.testing.assert_close(head(torch.randn(4, 8)), torch.zeros(4, 3))

    cfg = DecoderConfig(d_model=8, heads=2, ffn_dim=16, mask_dim=4, dropout=0.0)
    layer = QueryDecoderLayer(cfg).eval()
    references = torch.tensor([[[0.25, -0.5, 1.0]]])
    state = _query_state([QUERY_TEMPORAL], references)
    updated, _ = layer(
        state,
        torch.randn(1, 2, 8),
        torch.zeros(1, 2, 3),
        torch.ones(1, 1, 2, dtype=torch.bool),
        torch.ones(1),
        torch.randn(1, 4, 1, 1, 2),
    )
    torch.testing.assert_close(updated.references_cellscale, references)


def test_center_updates_are_bounded_per_query_type_and_padding_is_unchanged() -> None:
    cfg = DecoderConfig(d_model=8, heads=2, ffn_dim=16, mask_dim=4)
    layer = QueryDecoderLayer(cfg)
    references = torch.zeros(1, 5, 3)
    state = _query_state(
        [QUERY_PRIMARY, QUERY_SPLIT, QUERY_TEMPORAL, QUERY_DISCOVERY, QUERY_TEMPORAL],
        references,
    )
    state.padding_mask[0, 4] = True
    delta = layer._bounded_center_delta(torch.full((1, 5, 3), 1e6), state)
    limits = torch.tensor([0.50, 0.75, 0.25, 1.00])
    assert torch.all(delta[0, :4].abs().amax(dim=-1) <= limits + 1e-6)
    assert delta[0, 2].abs().max() <= cfg.temporal_center_step_dref + 1e-6
    torch.testing.assert_close(delta[0, 4], torch.zeros(3))


def test_corrected_local_cost_and_loss_are_fp32_finite_from_fp16_outputs() -> None:
    logits = torch.tensor(
        [[[[20.0, -20.0, 5.0]]], [[[-20.0, 20.0, -5.0]]]],
        dtype=torch.float16,
    )
    target = torch.tensor(
        [[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]],
    )
    centers = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    support = build_local_support_masks(
        target, centers, torch.ones(3), torch.tensor(1.0), 1.0
    )
    cost = build_cost_matrix(
        torch.tensor([10.0, -10.0], dtype=torch.float16),
        logits,
        torch.zeros(2, 3, dtype=torch.float16),
        target,
        centers,
        gt_support=support,
        mask_focal_alpha_pos=0.75,
        mask_focal_gamma=2.0,
    )
    dice, focal = local_matched_mask_losses(
        logits, target, support, alpha=0.75, gamma=2.0
    )
    assert cost.dtype == torch.float32
    assert torch.isfinite(cost).all()
    assert torch.isfinite(dice)
    assert torch.isfinite(focal)
