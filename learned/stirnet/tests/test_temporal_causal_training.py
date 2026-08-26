from __future__ import annotations

import torch

from learned.stirnet.model.types import ReasoningState, TemporalState
from learned.stirnet.training.config import CurriculumConfig, LossConfig
from learned.stirnet.training.curriculum import curriculum_stage
from learned.stirnet.training.temporal_causal import (
    causal_temporal_loss_terms,
    contentless_temporal_state,
    shuffled_temporal_state,
)


def _reasoning(logits, gate):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    gate = torch.as_tensor(gate, dtype=torch.float32)
    return ReasoningState(
        instance_tokens=torch.zeros((0, 4)),
        instance_exist_logits=torch.zeros((0,)),
        split_logits=torch.zeros((0,)),
        temporal_support=torch.zeros((0,)),
        temporal_attention_entropy=torch.zeros((0,)),
        edge_temporal_delta=torch.zeros_like(logits),
        edge_temporal_gate=gate,
        final_edge_logits=logits,
        recovery_track_indices=torch.zeros((0,), dtype=torch.long),
        recovery_logits=torch.zeros((0,)),
        recovery_scores=torch.zeros((0,)),
    )


def test_contentless_retains_support_geometry_but_erases_tokens():
    temporal = TemporalState(
        tokens=torch.randn(3, 8),
        ref_um=torch.randn(3, 3),
        batch_index=torch.tensor([0, 0, 0]),
        salience=torch.rand(3),
        reliability=torch.rand(3),
        status=torch.randn(3, 10),
        node_tokens=torch.randn(7, 8),
    )
    result = contentless_temporal_state(temporal)
    assert torch.equal(result.ref_um, temporal.ref_um)
    assert torch.equal(result.batch_index, temporal.batch_index)
    assert torch.equal(result.salience, temporal.salience)
    assert torch.equal(result.reliability, temporal.reliability)
    assert torch.equal(result.status, temporal.status)
    assert torch.count_nonzero(result.tokens) == 0
    assert result.node_tokens is not None
    assert torch.count_nonzero(result.node_tokens) == 0


def test_shuffle_never_crosses_batch_and_keeps_reference_positions_fixed():
    temporal = TemporalState(
        tokens=torch.tensor(
            [[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]]
        ),
        ref_um=torch.arange(18, dtype=torch.float32).reshape(6, 3),
        batch_index=torch.tensor([0, 0, 0, 1, 1, 1]),
        salience=torch.arange(6, dtype=torch.float32),
        reliability=torch.arange(6, dtype=torch.float32) + 10,
        status=torch.arange(60, dtype=torch.float32).reshape(6, 10),
        node_tokens=None,
    )
    result = shuffled_temporal_state(temporal, seed=31)
    assert torch.equal(result.ref_um, temporal.ref_um)
    assert torch.equal(result.batch_index, temporal.batch_index)
    assert sorted(result.tokens[:3, 0].tolist()) == [1.0, 2.0, 3.0]
    assert sorted(result.tokens[3:, 0].tolist()) == [10.0, 20.0, 30.0]


def test_causal_margin_penalizes_static_temporal_shortcut():
    spatial = torch.tensor([3.5])
    valid = torch.tensor([True])
    target = torch.tensor([0.0])
    spatial_same = torch.tensor([True])

    shortcut = causal_temporal_loss_terms(
        spatial_edge_logits=spatial,
        spatial_same_component=spatial_same,
        full_reasoning=_reasoning([3.5], [0.9]),
        corrupted_reasoning=_reasoning([3.5], [0.0]),
        target=target,
        valid=valid,
        noop_weight=0.5,
        corrupted_gate_weight=0.05,
        margin_weight=0.5,
        margin=1.0,
    )
    causal = causal_temporal_loss_terms(
        spatial_edge_logits=spatial,
        spatial_same_component=spatial_same,
        full_reasoning=_reasoning([-2.0], [0.9]),
        corrupted_reasoning=_reasoning([3.5], [0.0]),
        target=target,
        valid=valid,
        noop_weight=0.5,
        corrupted_gate_weight=0.05,
        margin_weight=0.5,
        margin=1.0,
    )
    assert shortcut.margin > 0
    assert causal.margin == 0
    assert causal.noop == 0
    assert causal.correction_fraction == 1


def test_causal_margin_is_directional_for_false_spatial_cut():
    terms = causal_temporal_loss_terms(
        spatial_edge_logits=torch.tensor([-3.0]),
        spatial_same_component=torch.tensor([False]),
        full_reasoning=_reasoning([2.0], [0.8]),
        corrupted_reasoning=_reasoning([-3.0], [0.0]),
        target=torch.tensor([1.0]),
        valid=torch.tensor([True]),
        noop_weight=0.5,
        corrupted_gate_weight=0.05,
        margin_weight=0.5,
        margin=1.0,
    )
    assert terms.margin == 0
    assert terms.correction_fraction == 1


def test_temporal_stage_freezes_mature_spatial_groups_by_default():
    cfg = CurriculumConfig(fixed_stage="instance_temporal")
    stage = curriculum_stage(cfg, 0)
    assert stage.trainable_groups == frozenset({"instances", "temporal"})


def test_causal_training_defaults_match_investigation31_objective():
    cfg = LossConfig()
    assert cfg.temporal_causal_enabled is True
    assert cfg.temporal_causal_noop_weight == 0.50
    assert cfg.temporal_causal_corrupted_gate_weight == 0.05
    assert cfg.temporal_causal_margin_weight == 0.50
    assert cfg.temporal_causal_margin == 1.0
    assert cfg.temporal_causal_corruptions == ("contentless", "shuffled")
