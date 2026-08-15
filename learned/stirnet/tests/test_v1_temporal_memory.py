from __future__ import annotations

import pytest
import torch

from learned.stirnet.data.collate import stirnet_collate
from learned.stirnet.data.graph_builder import (
    DETECTION_EDGE_ACCEPTED_COLUMN,
    DETECTION_EDGE_DIM,
    AssociationRecord,
    DetectionRecord,
    build_temporal_graph,
)
from learned.stirnet.data.historical_instances import TEMPORAL_CACHE_CONTRACT_VERSION
from learned.stirnet.data.trackastra_cache import (
    CACHE_CONTRACT_KEY,
    load_cache,
    save_cache,
)
from learned.stirnet.model.config import StirNetConfig, TemporalConfig
from learned.stirnet.model.graph_encoder import DetectionGraphEncoder
from learned.stirnet.model.query_builder import build_competition_group_ids
from learned.stirnet.model.stir_net import StirNet
from learned.stirnet.model.temporal_memory import TemporalMemoryAttention
from learned.stirnet.model.types import TemporalNodeMemory, TemporalState
from learned.stirnet.training.checkpoint import migrate_history_checkpoint_state_dict
from learned.stirnet.training.curriculum import model_parameter_groups


def _records() -> list[DetectionRecord]:
    return [
        DetectionRecord(101, -2, (0.0, 0.0, -3.0), 8.0, intensity_mean=0.2),
        DetectionRecord(102, -1, (0.0, 0.0, -1.0), 9.0, intensity_mean=0.3),
        DetectionRecord(201, -2, (0.0, 0.0, 3.0), 8.0, intensity_mean=0.7),
        DetectionRecord(202, 0, (0.0, 0.0, 0.0), 16.0, intensity_mean=0.5),
    ]


def _associations() -> list[AssociationRecord]:
    return [
        AssociationRecord(101, 102, 0.9),
        AssociationRecord(102, 202, 0.8),
    ]


def _small_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.spatial.channels = (4, 8, 8, 16)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.mask_dim = 4
    cfg.temporal.d_model = 16
    cfg.temporal.graph_ffn_dim = 32
    cfg.temporal.memory_heads = 4
    cfg.temporal.memory_ffn_dim = 32
    cfg.coreasoning.d_model = 16
    cfg.coreasoning.dropout = 0.0
    cfg.queries.d_model = 16
    cfg.queries.discovery_queries = 1
    cfg.decoder.d_model = 16
    cfg.decoder.ffn_dim = 32
    cfg.decoder.mask_dim = 4
    cfg.decoder.dropout = 0.0
    cfg.decoder.max_spatial_tokens = 64
    cfg.training.activation_checkpointing = False
    return cfg


def _temporal_state(
    node_tokens: torch.Tensor,
    node_batch: torch.Tensor,
    *,
    tracklet_tokens: torch.Tensor | None = None,
    tracklet_batch: torch.Tensor | None = None,
) -> TemporalState:
    n, d_model = node_tokens.shape
    tracklet_tokens = (
        tracklet_tokens
        if tracklet_tokens is not None
        else node_tokens.new_zeros((0, d_model))
    )
    tracklet_batch = (
        tracklet_batch
        if tracklet_batch is not None
        else node_batch.new_zeros((len(tracklet_tokens),))
    )
    refs = node_tokens.new_zeros((len(tracklet_tokens), 3))
    node_refs = node_tokens.new_zeros((n, 3))
    memory = TemporalNodeMemory(
        tokens=node_tokens,
        observed_ref_um=node_refs,
        projected_ref_um=node_refs,
        time_offset=node_tokens.new_zeros((n,)),
        tracklet_id=node_batch.new_zeros((n,)),
        batch_index=node_batch,
        history_valid=torch.ones(n, dtype=torch.bool, device=node_tokens.device),
        node_ids=torch.arange(n, device=node_tokens.device),
    )
    return TemporalState(
        tokens=tracklet_tokens,
        ref_um=refs,
        ref_cellscale=refs,
        salience=node_tokens.new_zeros((len(tracklet_tokens), 1)),
        reliability=node_tokens.new_ones((len(tracklet_tokens), 1)),
        status=node_tokens.new_zeros((len(tracklet_tokens), 10)),
        edge_index=torch.zeros(2, 0, dtype=torch.long, device=node_tokens.device),
        edge_attr=node_tokens.new_zeros((0, 22)),
        batch_index=tracklet_batch,
        node_memory=memory,
    )


def test_candidate_graph_is_complete_and_marks_only_accepted_relations() -> None:
    records = _records()
    graph = build_temporal_graph(records, _associations(), dref_um=4.0)
    n = len(records)
    assert graph["graph_edge_index"].shape == (2, n * (n - 1))
    assert graph["graph_edge_attr"].shape == (n * (n - 1), DETECTION_EDGE_DIM)
    pairs = graph["graph_edge_index"].t().tolist()
    assert len({tuple(pair) for pair in pairs}) == n * (n - 1)
    accepted = graph["graph_edge_attr"][:, DETECTION_EDGE_ACCEPTED_COLUMN].bool()
    marked_pairs = {tuple(pair) for pair in graph["graph_edge_index"][:, accepted].t().tolist()}
    assert marked_pairs == {(0, 1), (1, 0), (1, 3), (3, 1)}
    assert not graph["graph_edge_attr"][~accepted, 9].bool().any()


def test_candidate_graph_does_not_change_tracklet_identity() -> None:
    dense = build_temporal_graph(_records(), _associations(), dref_um=4.0)
    accepted = build_temporal_graph(
        _records(), _associations(), dref_um=4.0, candidate_graph_enabled=False
    )
    torch.testing.assert_close(dense["tracklet_id"], accepted["tracklet_id"])
    assert dense["graph_edge_index"].shape[1] > accepted["graph_edge_index"].shape[1]


def test_candidate_graph_has_no_cross_batch_edges_after_collation() -> None:
    graph_a = build_temporal_graph(_records()[:2], _associations()[:1], dref_um=4.0)
    graph_b = build_temporal_graph(_records()[2:], [], dref_um=4.0)

    def sample(graph: dict, value: int) -> dict:
        return {
            "spatial_inputs": torch.zeros(5, 2, 2, 2),
            "instance_labels": torch.zeros(2, 2, 2, dtype=torch.long),
            "spacing_um": torch.ones(3),
            "dref_um": torch.tensor(4.0),
            "target": {},
            "instance_features": torch.zeros(0, 14),
            "instance_ids": torch.zeros(0, dtype=torch.long),
            "instance_centroids_um": torch.zeros(0, 3),
            **graph,
            "sample": value,
        }

    batch = stirnet_collate([sample(graph_a, 0), sample(graph_b, 1)])
    node_batch = batch["temporal_batch"][batch["tracklet_id"]]
    source, destination = batch["graph_edge_index"]
    assert torch.equal(node_batch[source], node_batch[destination])
    assert batch["graph_edge_index"].shape[1] == 4
    torch.testing.assert_close(
        batch["node_event_features"],
        torch.cat(
            [graph_a["node_event_features"], graph_b["node_event_features"]]
        ),
    )


def test_candidate_graph_safety_limit_raises_without_truncation() -> None:
    with pytest.raises(RuntimeError, match="will not silently drop"):
        build_temporal_graph(
            _records(), _associations(), dref_um=4.0, max_candidate_edges=11
        )


def test_fine_node_memory_survives_tracklet_pooling() -> None:
    cfg = _small_config()
    model = StirNet(cfg).eval()
    graph = build_temporal_graph(_records(), _associations(), dref_um=4.0)
    temporal = model._build_temporal(
        graph["graph_x"],
        graph["graph_edge_index"],
        graph["graph_edge_attr"],
        graph["tracklet_id"],
        graph["temporal_ref_um"],
        graph["temporal_status"],
        graph["hypothesis_edge_index"],
        graph["hypothesis_edge_attr"],
        torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long),
        torch.tensor([4.0]),
        graph["node_instance_grid"],
        graph["node_history_valid"],
        node_observed_ref_um=graph["node_observed_ref_um"],
        node_time_offset=graph["node_time_offset"],
        node_ids=graph["node_ids"],
    )
    assert temporal.node_memory is not None
    assert temporal.node_memory.tokens.shape == (len(_records()), cfg.temporal.d_model)
    assert temporal.tokens.shape[0] == len(graph["temporal_ref_um"])
    torch.testing.assert_close(
        temporal.node_memory.projected_ref_um,
        temporal.ref_um[temporal.node_memory.tracklet_id],
    )
    torch.testing.assert_close(
        temporal.node_memory.event_features,
        graph["node_event_features"],
    )


def test_node_order_permutation_is_equivariant() -> None:
    torch.manual_seed(4)
    cfg = TemporalConfig(d_model=16, graph_ffn_dim=32, edge_dim=15)
    encoder = DetectionGraphEncoder(cfg).eval()
    original = build_temporal_graph(_records(), _associations(), dref_um=4.0)
    order = [2, 0, 3, 1]
    permuted_records = [_records()[index] for index in order]
    permuted = build_temporal_graph(permuted_records, _associations(), dref_um=4.0)
    with torch.no_grad():
        first = encoder(
            original["graph_x"], original["graph_edge_index"], original["graph_edge_attr"]
        )
        second = encoder(
            permuted["graph_x"], permuted["graph_edge_index"], permuted["graph_edge_attr"]
        )
    by_id = {int(node_id): row for row, node_id in enumerate(permuted["node_ids"])}
    restored = torch.stack([second[by_id[int(node_id)]] for node_id in original["node_ids"]])
    torch.testing.assert_close(first, restored, atol=2e-6, rtol=2e-5)


def test_temporal_memory_attention_is_batch_isolated_and_finite() -> None:
    torch.manual_seed(5)
    cfg = TemporalConfig(d_model=8, memory_heads=2, relation_bias_hidden=4)
    attention = TemporalMemoryAttention(cfg).eval()
    query = torch.randn(2, 8)
    memory = torch.randn(4, 8)
    refs = torch.zeros(2, 3)
    memory_refs = torch.zeros(4, 3)
    batch = torch.tensor([0, 1])
    memory_batch = torch.tensor([0, 0, 1, 1])
    time_offset = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    history_valid = torch.tensor([True, False, True, False])
    output, debug = attention(
        query,
        refs,
        batch,
        memory,
        memory_refs,
        memory_refs,
        time_offset,
        memory_batch,
        history_valid,
        torch.ones(2),
        return_debug=True,
        full_attention=True,
    )
    assert torch.isfinite(output).all()
    assert debug is not None
    assert torch.count_nonzero(debug["full_weights"][0, 2:]) == 0
    assert torch.count_nonzero(debug["full_weights"][1, :2]) == 0
    changed = memory.clone()
    changed[2:] += 1000
    changed_output, _ = attention(
        query,
        refs,
        batch,
        changed,
        memory_refs,
        memory_refs,
        time_offset,
        memory_batch,
        history_valid,
        torch.ones(2),
    )
    torch.testing.assert_close(output[0], changed_output[0])


def test_empty_temporal_memory_returns_valid_shapes() -> None:
    cfg = TemporalConfig(d_model=8, memory_heads=2)
    attention = TemporalMemoryAttention(cfg)
    output, debug = attention(
        torch.randn(3, 8),
        torch.zeros(3, 3),
        torch.tensor([0, 0, 1]),
        torch.zeros(0, 8),
        torch.zeros(0, 3),
        torch.zeros(0, 3),
        torch.zeros(0),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.bool),
        torch.ones(2),
        return_debug=True,
        full_attention=True,
    )
    assert output.shape == (3, 8) and torch.count_nonzero(output) == 0
    assert debug is not None and debug["full_weights"].shape == (3, 0)


def test_split_slots_can_produce_distinct_node_attention() -> None:
    cfg = TemporalConfig(
        d_model=4,
        memory_heads=1,
        relation_bias_hidden=2,
        memory_debug_topk=2,
    )
    attention = TemporalMemoryAttention(cfg).eval()
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.out):
            projection.weight.copy_(torch.eye(4))
        for parameter in attention.relation_bias.parameters():
            parameter.zero_()
    # These stand in for two learned split-slot offsets on the same component.
    split_slots = torch.tensor([[4.0, 0, 0, 0], [-4.0, 0, 0, 0]])
    memory = torch.tensor([[4.0, 0, 0, 0], [-4.0, 0, 0, 0]])
    _, debug = attention(
        split_slots,
        torch.zeros(2, 3),
        torch.zeros(2, dtype=torch.long),
        memory,
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.tensor([-2.0, -2.0]),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2, dtype=torch.bool),
        torch.ones(1),
        return_debug=True,
        full_attention=True,
    )
    assert debug is not None
    assert debug["full_weights"][0].argmax() == 0
    assert debug["full_weights"][1].argmax() == 1
    assert not torch.allclose(debug["full_weights"][0], debug["full_weights"][1])


def test_query_memory_gradient_reaches_detection_and_history_encoders() -> None:
    torch.manual_seed(6)
    cfg = _small_config()
    model = StirNet(cfg).train()
    graph = build_temporal_graph(_records(), _associations(), dref_um=4.0)
    grids = torch.randn(4, 4, 12, 12, 12, requires_grad=True)
    valid = torch.ones(4, dtype=torch.bool)
    temporal = model._build_temporal(
        graph["graph_x"], graph["graph_edge_index"], graph["graph_edge_attr"],
        graph["tracklet_id"], graph["temporal_ref_um"], graph["temporal_status"],
        graph["hypothesis_edge_index"], graph["hypothesis_edge_attr"],
        torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long), torch.tensor([4.0]),
        grids, valid,
        node_observed_ref_um=graph["node_observed_ref_um"],
        node_time_offset=graph["node_time_offset"],
        node_ids=graph["node_ids"],
    )
    queries = torch.randn(3, 16, requires_grad=True)
    fused, _ = model.query_decoder.layers[0].temporal_fusion(
        queries,
        torch.zeros(3, 3),
        torch.zeros(3, dtype=torch.long),
        temporal,
        torch.tensor([4.0]),
    )
    loss = fused.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert grids.grad is not None and torch.isfinite(grids.grad).all()
    assert grids.grad.abs().sum() > 0
    for module in (
        model.query_decoder.layers[0].temporal_fusion.node_attention,
        model.graph_encoder,
        model.history_encoder,
        model.history_fusion,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
            for gradient in gradients
        )


def test_detection_edge_checkpoint_migration_preserves_legacy_prefix() -> None:
    model = StirNet(_small_config())
    state = model.state_dict()
    legacy = {
        key: value.clone()
        for key, value in state.items()
        if ".temporal_fusion." not in key
    }
    edge_keys = [
        key
        for key in legacy
        if key.startswith("graph_encoder.layers.") and key.endswith("attn.edge.weight")
    ]
    for key in edge_keys:
        legacy[key] = legacy[key][:, :14].clone()
    migrated, notes = migrate_history_checkpoint_state_dict(model, legacy)
    assert any("14 -> 15" in note for note in notes)
    assert any("temporal_fusion" in note for note in notes)
    for key in edge_keys:
        torch.testing.assert_close(migrated[key][:, :14], legacy[key])
        assert torch.count_nonzero(migrated[key][:, 14:]) == 0
    model.load_state_dict(migrated, strict=True)


def test_temporal_memory_parameters_are_in_query_curriculum_group() -> None:
    model = StirNet(_small_config())
    groups = model_parameter_groups(model)
    query_ids = {id(parameter) for parameter in groups["query"]}
    assert all(
        id(parameter) in query_ids
        for parameter in model.query_builder.temporal_fusion.parameters()
    )
    assert all(
        id(parameter) in query_ids
        for layer in model.query_decoder.layers
        for parameter in layer.temporal_fusion.parameters()
    )


def test_temporal_cache_v3_is_explicit_and_legacy_cache_is_rejected(tmp_path) -> None:
    current = tmp_path / "temporal_v3" / "temporal_graph.pt"
    save_cache(current, {"graph_x": torch.zeros(0, 32)})
    loaded = load_cache(current)
    assert loaded[CACHE_CONTRACT_KEY] == TEMPORAL_CACHE_CONTRACT_VERSION

    legacy = tmp_path / "history_v2.pt"
    torch.save(
        {
            CACHE_CONTRACT_KEY: 2,
            "graph_x": torch.zeros(1, 32),
            "graph_edge_attr": torch.zeros(0, 14),
        },
        legacy,
    )
    with pytest.raises(ValueError, match="Rebuild under temporal_v3"):
        load_cache(legacy)


def test_missing_history_full_forward_debugs_every_memory_read() -> None:
    torch.manual_seed(8)
    cfg = _small_config()
    model = StirNet(cfg).eval()
    graph = build_temporal_graph(_records(), _associations(), dref_um=4.0)
    labels = torch.zeros(1, 8, 16, 16, dtype=torch.long)
    labels[:, 2:6, 5:11, 5:11] = 1
    with torch.no_grad():
        output = model(
            torch.randn(1, 5, 8, 16, 16),
            labels,
            torch.ones(1, 3),
            torch.tensor([4.0]),
            torch.zeros(1, 14),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.zeros(1, 3),
            graph["graph_x"],
            graph["graph_edge_index"],
            graph["graph_edge_attr"],
            graph["tracklet_id"],
            graph["temporal_ref_um"],
            graph["temporal_status"],
            graph["hypothesis_edge_index"],
            graph["hypothesis_edge_attr"],
            torch.zeros(len(graph["temporal_ref_um"]), dtype=torch.long),
            node_instance_grid=torch.zeros(4, 4, 12, 12, 12),
            node_history_valid=torch.zeros(4, dtype=torch.bool),
            node_observed_ref_um=graph["node_observed_ref_um"],
            node_time_offset=graph["node_time_offset"],
            node_ids=graph["node_ids"],
            return_debug=True,
            return_full_temporal_attention=True,
        )
    assert torch.isfinite(output.exist_logits).all()
    assert output.debug is not None
    assert output.debug["node_memory"]["tokens"].shape[0] == 4
    component = output.debug["component_temporal_attention"]["node"]
    assert component["full_weights"].shape == (1, 4)
    assert component["competition_group_id"].tolist() == [-1]
    layers = output.debug["query_temporal_attention"]
    assert len(layers) == cfg.decoder.layers
    assert all(layer["node"]["full_weights"].shape[1] == 4 for layer in layers)
    assert all(torch.isfinite(layer["node"]["entropy"]).all() for layer in layers)
    for layer in layers:
        expected_groups = build_competition_group_ids(
            layer["query_type"], layer["source_instance_id"]
        )
        torch.testing.assert_close(
            layer["node"]["competition_group_id"], expected_groups
        )
