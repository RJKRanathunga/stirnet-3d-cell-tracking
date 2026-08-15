from __future__ import annotations

import torch

from learned.stirnet.model.config import TemporalConfig
from learned.stirnet.model.query_builder import (
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPATIAL_PROPOSAL,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
    build_competition_group_ids,
)
from learned.stirnet.model.temporal_memory import (
    HierarchicalTemporalFusion,
    TemporalEventRelevance,
    TemporalMemoryAttention,
)
from learned.stirnet.model.types import TemporalNodeMemory, TemporalState
from learned.stirnet.temporal_events import (
    TEMPORAL_NODE_EVENT_FEATURE_NAMES,
    event_features_from_legacy_graph_x,
)


def _config() -> TemporalConfig:
    return TemporalConfig(
        d_model=4,
        memory_heads=1,
        memory_ffn_dim=8,
        relation_bias_hidden=4,
        event_hidden_dim=6,
        memory_debug_topk=4,
    )


def _zero_relation(attention: TemporalMemoryAttention) -> None:
    with torch.no_grad():
        for parameter in attention.relation_bias.parameters():
            parameter.zero_()


def _attention_call(
    attention: TemporalMemoryAttention,
    queries: torch.Tensor,
    memory: torch.Tensor,
    *,
    query_batch: torch.Tensor | None = None,
    memory_batch: torch.Tensor | None = None,
    event_features: torch.Tensor | None = None,
    groups: torch.Tensor | None = None,
    routing_ablation: str = "full",
    debug: bool = True,
):
    query_batch = (
        query_batch
        if query_batch is not None
        else torch.zeros(len(queries), dtype=torch.long)
    )
    memory_batch = (
        memory_batch
        if memory_batch is not None
        else torch.zeros(len(memory), dtype=torch.long)
    )
    return attention(
        queries,
        torch.zeros(len(queries), 3),
        query_batch,
        memory,
        torch.zeros(len(memory), 3),
        torch.zeros(len(memory), 3),
        torch.zeros(len(memory)),
        memory_batch,
        torch.ones(len(memory), dtype=torch.bool),
        torch.ones(int(query_batch.max().item()) + 1 if len(query_batch) else 1),
        event_features=event_features,
        competition_group_ids=groups,
        routing_ablation=routing_ablation,
        return_debug=debug,
        full_attention=debug,
    )


def _temporal_state() -> TemporalState:
    node_tokens = torch.randn(3, 4)
    node_refs = torch.zeros(3, 3)
    node_memory = TemporalNodeMemory(
        tokens=node_tokens,
        observed_ref_um=node_refs,
        projected_ref_um=node_refs,
        time_offset=torch.tensor([-1.0, 0.0, 1.0]),
        tracklet_id=torch.zeros(3, dtype=torch.long),
        batch_index=torch.zeros(3, dtype=torch.long),
        history_valid=torch.ones(3, dtype=torch.bool),
        event_features=torch.zeros(3, 8),
    )
    return TemporalState(
        tokens=torch.randn(1, 4),
        ref_um=torch.zeros(1, 3),
        ref_cellscale=torch.zeros(1, 3),
        salience=torch.ones(1, 1),
        reliability=torch.ones(1, 1),
        status=torch.zeros(1, 10),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 22),
        batch_index=torch.zeros(1, dtype=torch.long),
        node_memory=node_memory,
    )


def test_explicit_event_contract_and_legacy_reconstruction() -> None:
    assert TEMPORAL_NODE_EVENT_FEATURE_NAMES == (
        "normalized_time",
        "normalized_length_before",
        "normalized_length_after",
        "is_current",
        "is_interior_start",
        "is_interior_end",
        "is_division",
        "is_boundary",
    )
    graph_x = torch.zeros(1, 32)
    graph_x[0, [0, 23, 24, 27, 28, 29, 30, 31]] = torch.tensor(
        [-0.5, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    )
    event = event_features_from_legacy_graph_x(graph_x, temporal_radius=2)
    torch.testing.assert_close(
        event,
        torch.tensor([[-0.5, 0.4, 0.6, 0.0, 1.0, 0.0, 1.0, 0.0]]),
    )


def test_initial_event_prior_orders_correction_and_boundary_events() -> None:
    scorer = TemporalEventRelevance(_config())
    features = torch.zeros(5, 8)
    features[1, 4] = 1.0
    features[2, 5] = 1.0
    features[3, 6] = 1.0
    features[4, 4] = 1.0
    features[4, 7] = 1.0
    logits, probabilities, residual = scorer(
        features, reference=torch.zeros(5, 4)
    )
    assert logits[1] > logits[0]
    assert logits[2] > logits[0]
    assert logits[3] > logits[0]
    assert logits[4] < logits[1]
    assert torch.count_nonzero(residual) == 0
    assert torch.all((probabilities > 0) & (probabilities < 1))


def test_disabled_routing_matches_the_pre_feature_reader() -> None:
    torch.manual_seed(2)
    cfg = _config()
    cfg.event_routing_enabled = False
    cfg.source_competition_enabled = False
    baseline = TemporalMemoryAttention(cfg)
    routed = TemporalMemoryAttention(cfg, fine_routing=True)
    result = routed.load_state_dict(baseline.state_dict(), strict=False)
    assert result.unexpected_keys == []
    queries = torch.randn(3, 4)
    memory = torch.randn(4, 4)
    event = torch.randn(4, 8)
    groups = torch.tensor([5, 5, -1])
    old_output, old_debug = _attention_call(baseline, queries, memory)
    new_output, new_debug = _attention_call(
        routed,
        queries,
        memory,
        event_features=event,
        groups=groups,
        routing_ablation="full",
    )
    torch.testing.assert_close(new_output, old_output, atol=1e-7, rtol=1e-6)
    assert old_debug is not None and new_debug is not None
    torch.testing.assert_close(
        new_debug["full_weights"], old_debug["full_weights"], atol=1e-7, rtol=1e-6
    )


def test_event_bias_increases_correction_node_attention() -> None:
    attention = TemporalMemoryAttention(_config(), fine_routing=True)
    _zero_relation(attention)
    with torch.no_grad():
        attention.q.weight.zero_()
        attention.k.weight.zero_()
    event = torch.zeros(2, 8)
    event[1, 4] = 1.0
    _, debug = _attention_call(
        attention,
        torch.zeros(1, 4),
        torch.randn(2, 4),
        event_features=event,
        routing_ablation="event_only",
    )
    assert debug is not None
    assert debug["full_weights"][0, 1] > debug["full_weights"][0, 0]
    assert debug["correction_event_mass"][0] > debug["continuous_track_mass"][0]


def test_competition_is_grouped_and_batch_isolated() -> None:
    attention = TemporalMemoryAttention(_config(), fine_routing=True)
    _zero_relation(attention)
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.out):
            projection.weight.copy_(torch.eye(4))
    queries = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ]
    )
    memory = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ]
    )
    query_batch = torch.tensor([0, 0, 0, 1])
    memory_batch = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([7, 7, 8, 7])
    _, baseline = _attention_call(
        attention,
        queries,
        memory,
        query_batch=query_batch,
        memory_batch=memory_batch,
        groups=groups,
        routing_ablation="baseline",
    )
    _, competitive = _attention_call(
        attention,
        queries,
        memory,
        query_batch=query_batch,
        memory_batch=memory_batch,
        groups=groups,
        routing_ablation="competition_only",
    )
    assert baseline is not None and competitive is not None
    assert competitive["competition_group_size"].tolist() == [2, 2, 1, 1]
    assert not torch.allclose(
        competitive["full_weights"][:2], baseline["full_weights"][:2]
    )
    torch.testing.assert_close(
        competitive["full_weights"][2:], baseline["full_weights"][2:]
    )
    assert torch.count_nonzero(competitive["full_weights"][0, 2:]) == 0
    assert torch.count_nonzero(competitive["full_weights"][3, :2]) == 0


def test_query_metadata_builds_current_and_legacy_competition_groups() -> None:
    query_types = torch.tensor(
        [
            QUERY_SPATIAL_PROPOSAL,
            QUERY_SPATIAL_PROPOSAL,
            QUERY_SPATIAL_PROPOSAL,
            QUERY_TEMPORAL,
            QUERY_DISCOVERY,
            QUERY_PRIMARY,
            QUERY_SPLIT,
        ]
    )
    source_ids = torch.tensor([5, 5, -1, 5, 5, 9, 9])
    groups = build_competition_group_ids(query_types, source_ids)
    assert groups.tolist() == [5, 5, -1, -1, -1, 9, 9]


def test_routing_parameters_receive_finite_gradients_without_inplace_errors() -> None:
    torch.manual_seed(9)
    attention = TemporalMemoryAttention(_config(), fine_routing=True)
    queries = torch.randn(2, 4, requires_grad=True)
    memory = torch.randn(3, 4, requires_grad=True)
    event = torch.zeros(3, 8)
    event[1, 4] = 1.0
    event[2, 5] = 1.0
    output, _ = _attention_call(
        attention,
        queries,
        memory,
        event_features=event,
        groups=torch.tensor([4, 4]),
        routing_ablation="full",
        debug=False,
    )
    loss = output[0].square().sum() + 0.37 * output[1].square().sum()
    loss.backward()
    gradients = {
        "event_residual": attention.event_relevance.residual[-1].weight.grad,
        "event_strength": attention.event_strength_raw.grad,
        "competition_temperature": attention.log_competition_temperature.grad,
    }
    for gradient in gradients.values():
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_empty_memory_and_existing_memory_ablations_remain_supported() -> None:
    cfg = _config()
    attention = TemporalMemoryAttention(cfg, fine_routing=True)
    output, debug = _attention_call(
        attention,
        torch.randn(2, 4),
        torch.zeros(0, 4),
        event_features=torch.zeros(0, 8),
        groups=torch.tensor([3, 3]),
    )
    assert output.shape == (2, 4) and torch.count_nonzero(output) == 0
    assert debug is not None and debug["full_weights"].shape == (2, 0)
    assert debug["event_logit"].shape == (0,)

    fusion = HierarchicalTemporalFusion(cfg)
    temporal = _temporal_state()
    for memory_ablation in (
        "full",
        "zero_node",
        "shuffle_node",
        "tracklet_only",
        "node_only",
    ):
        fused, diagnostics = fusion(
            torch.randn(2, 4),
            torch.zeros(2, 3),
            torch.zeros(2, dtype=torch.long),
            temporal,
            torch.ones(1),
            memory_ablation=memory_ablation,
            routing_ablation="baseline",
            return_debug=True,
            full_attention=True,
        )
        assert torch.isfinite(fused).all()
        assert diagnostics is not None
        if memory_ablation != "tracklet_only":
            assert diagnostics["node"]["full_weights"].shape == (2, 3)

    for routing_ablation in (
        "full",
        "event_only",
        "competition_only",
        "baseline",
        "shuffled_event",
    ):
        fused, diagnostics = fusion(
            torch.randn(2, 4),
            torch.zeros(2, 3),
            torch.zeros(2, dtype=torch.long),
            temporal,
            torch.ones(1),
            memory_ablation="node_only",
            routing_ablation=routing_ablation,
            competition_group_ids=torch.tensor([2, 2]),
            return_debug=True,
        )
        assert torch.isfinite(fused).all()
        assert diagnostics is not None
        assert diagnostics["routing_ablation"] == routing_ablation
