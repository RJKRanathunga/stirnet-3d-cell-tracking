from __future__ import annotations

import math

import torch

from learned.stirnet.model.config import PartitionConfig
from learned.stirnet.model.partition.partitioner import GraphPartitioner
from learned.stirnet.model.types import RAGState


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _rag(node_count: int, edges: list[tuple[int, int]]) -> RAGState:
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T.contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_count = len(edges)
    labels = torch.arange(
        1,
        node_count + 1,
        dtype=torch.long,
    ).reshape(1, 1, node_count)

    return RAGState(
        node_features=torch.zeros(node_count, 1),
        node_embeddings=torch.zeros(node_count, 1),
        node_batch=torch.zeros(node_count, dtype=torch.long),
        node_supervoxel_id=torch.arange(
            1,
            node_count + 1,
            dtype=torch.long,
        ),
        node_centroid_um=torch.zeros(node_count, 3),
        node_volume_voxels=torch.ones(node_count),
        edge_index=edge_index,
        edge_features=torch.zeros(edge_count, 1),
        edge_embeddings=torch.zeros(edge_count, 1),
        spatial_edge_logits=torch.zeros(edge_count),
        edge_batch=torch.zeros(edge_count, dtype=torch.long),
        supervoxel_labels=[labels],
        node_offsets=torch.tensor([0, node_count], dtype=torch.long),
    )


def _logits(probabilities: list[float]) -> torch.Tensor:
    return torch.tensor(
        [_logit(value) for value in probabilities],
        dtype=torch.float32,
    )


def test_partition_config_defaults_to_multicut() -> None:
    cfg = PartitionConfig()
    assert cfg.spatial_partition_backend == "multicut"
    assert cfg.final_partition_backend == "union_find"
    assert cfg.spatial_merge_threshold == 0.845


def test_union_find_retains_historical_transitive_join() -> None:
    cfg = PartitionConfig(spatial_partition_backend="union_find")
    partitioner = GraphPartitioner(cfg)
    rag = _rag(3, [(0, 1), (1, 2), (0, 2)])

    result = partitioner(
        rag,
        _logits([0.97, 0.95, 0.05]),
        0.845,
        stage="spatial",
    )
    assert result.component_count_per_batch.tolist() == [1]
    assert result.node_component.tolist() == [0, 0, 0]


def test_multicut_respects_strong_repulsive_triangle_edge() -> None:
    cfg = PartitionConfig(spatial_partition_backend="multicut")
    partitioner = GraphPartitioner(cfg)
    rag = _rag(3, [(0, 1), (1, 2), (0, 2)])

    result = partitioner(
        rag,
        _logits([0.97, 0.95, 0.05]),
        0.845,
        stage="spatial",
    )

    assert result.component_count_per_batch.tolist() == [2]
    assert result.node_component[0] == result.node_component[1]
    assert result.node_component[0] != result.node_component[2]


def test_multicut_square_matches_expected_partition() -> None:
    cfg = PartitionConfig(spatial_partition_backend="multicut")
    partitioner = GraphPartitioner(cfg)
    rag = _rag(4, [(0, 1), (1, 2), (2, 3), (3, 0)])

    result = partitioner(
        rag,
        _logits([0.05, 0.97, 0.97, 0.05]),
        0.845,
        stage="spatial",
    )

    assert result.component_count_per_batch.tolist() == [2]
    a, b, c, d = result.node_component.tolist()
    assert b == c == d
    assert a != b


def test_multicut_handles_no_edges() -> None:
    cfg = PartitionConfig(spatial_partition_backend="multicut")
    result = GraphPartitioner(cfg)(
        _rag(3, []),
        torch.zeros(0),
        0.845,
        stage="spatial",
    )
    assert result.component_count_per_batch.tolist() == [3]
    assert result.node_component.tolist() == [0, 1, 2]


def test_final_stage_uses_final_backend() -> None:
    cfg = PartitionConfig(
        spatial_partition_backend="union_find",
        final_partition_backend="multicut",
    )
    partitioner = GraphPartitioner(cfg)
    rag = _rag(3, [(0, 1), (1, 2), (0, 2)])

    result = partitioner(
        rag,
        _logits([0.90, 0.80, 0.05]),
        0.50,
        stage="final",
    )
    assert result.component_count_per_batch.tolist() == [2]


def test_parent_component_constraint_blocks_external_union_find_bridge() -> None:
    cfg = PartitionConfig(final_partition_backend="union_find")
    partitioner = GraphPartitioner(cfg)
    rag = _rag(3, [(0, 1), (0, 2), (2, 1)])
    logits = _logits([0.05, 0.99, 0.99])

    unconstrained = partitioner(rag, logits, 0.50, stage="final")
    constrained = partitioner(
        rag,
        logits,
        0.50,
        stage="final",
        node_parent_component=torch.tensor([0, 0, 1]),
    )

    assert unconstrained.node_component.tolist() == [0, 0, 0]
    assert constrained.component_count_per_batch.tolist() == [3]
    assert constrained.node_component[0] != constrained.node_component[1]
    assert constrained.node_component[0] != constrained.node_component[2]
    assert constrained.node_component[1] != constrained.node_component[2]


def test_parent_component_constraint_applies_to_multicut_backend() -> None:
    cfg = PartitionConfig(final_partition_backend="multicut")
    partitioner = GraphPartitioner(cfg)
    rag = _rag(3, [(0, 1), (0, 2), (2, 1)])

    constrained = partitioner(
        rag,
        _logits([0.05, 0.99, 0.99]),
        0.50,
        stage="final",
        node_parent_component=torch.tensor([0, 0, 1]),
    )

    assert constrained.component_count_per_batch.tolist() == [3]


def test_parent_component_constraint_validates_shape_and_dtype() -> None:
    partitioner = GraphPartitioner(PartitionConfig())
    rag = _rag(2, [(0, 1)])
    logits = _logits([0.9])

    try:
        partitioner(
            rag,
            logits,
            0.50,
            stage="final",
            node_parent_component=torch.tensor([0]),
        )
    except ValueError as error:
        assert "align one-to-one" in str(error)
    else:
        raise AssertionError("Expected parent-component shape validation")

    try:
        partitioner(
            rag,
            logits,
            0.50,
            stage="final",
            node_parent_component=torch.tensor([0.0, 0.0]),
        )
    except TypeError as error:
        assert "integer dtype" in str(error)
    else:
        raise AssertionError("Expected parent-component dtype validation")
