from __future__ import annotations

"""Signed and legacy graph partitioning backends for STIR-Net."""

from collections import deque
import math
from typing import List, Literal

import numpy as np
import torch
from torch import Tensor, nn

from ..config import PartitionConfig
from ..types import PartitionState, RAGState


# STIRNET_SIGNED_MULTICUT_V1


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _components_from_uncut(
    node_count: int,
    edges: np.ndarray,
    cut: np.ndarray,
) -> np.ndarray:
    uf = _UnionFind(node_count)
    for edge_row in np.flatnonzero(~cut):
        uf.union(int(edges[0, edge_row]), int(edges[1, edge_row]))

    root_to_component: dict[int, int] = {}
    component = np.empty(node_count, dtype=np.int64)
    for node in range(node_count):
        root = uf.find(node)
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component)
        component[node] = root_to_component[root]
    return component


def _uncut_adjacency(
    node_count: int,
    edges: np.ndarray,
    cut: np.ndarray,
) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(node_count)
    ]
    for edge_row in np.flatnonzero(~cut):
        u = int(edges[0, edge_row])
        v = int(edges[1, edge_row])
        adjacency[u].append((v, int(edge_row)))
        adjacency[v].append((u, int(edge_row)))
    return adjacency


def _find_path_edges(
    adjacency: list[list[tuple[int, int]]],
    source: int,
    target: int,
) -> list[int] | None:
    if source == target:
        return []

    parent_node = {source: -1}
    parent_edge: dict[int, int] = {}
    queue: deque[int] = deque([source])

    while queue:
        u = queue.popleft()
        for v, edge_row in adjacency[u]:
            if v in parent_node:
                continue
            parent_node[v] = u
            parent_edge[v] = edge_row
            if v == target:
                path: list[int] = []
                current = target
                while current != source:
                    path.append(parent_edge[current])
                    current = parent_node[current]
                path.reverse()
                return path
            queue.append(v)
    return None


def _violated_cycle_constraints(
    node_count: int,
    edges: np.ndarray,
    cut: np.ndarray,
    *,
    max_constraints: int,
) -> list[tuple[int, tuple[int, ...]]]:
    """Find cut edges whose endpoints remain connected through uncut edges."""
    adjacency = _uncut_adjacency(node_count, edges, cut)
    component = _components_from_uncut(node_count, edges, cut)

    violations: list[tuple[int, tuple[int, ...]]] = []
    for edge_row in np.flatnonzero(cut):
        u = int(edges[0, edge_row])
        v = int(edges[1, edge_row])
        if component[u] != component[v]:
            continue

        path = _find_path_edges(adjacency, u, v)
        if path is None:
            continue
        violations.append(
            (int(edge_row), tuple(int(value) for value in path))
        )
        if len(violations) >= max_constraints:
            break
    return violations


def _signed_costs(
    probability: np.ndarray,
    *,
    neutral_probability: float,
    epsilon: float,
) -> np.ndarray:
    p = np.clip(
        np.asarray(probability, dtype=np.float64),
        epsilon,
        1.0 - epsilon,
    )
    q = float(
        np.clip(neutral_probability, epsilon, 1.0 - epsilon)
    )
    return np.log(p / (1.0 - p)) - math.log(q / (1.0 - q))


def _solve_multicut_graph(
    *,
    node_count: int,
    edges: np.ndarray,
    probability: np.ndarray,
    neutral_probability: float,
    cfg: PartitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cut, component)`` for one batch-local RAG."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    edge_count = int(edges.shape[1])
    if edge_count == 0:
        return (
            np.zeros(0, dtype=bool),
            np.arange(node_count, dtype=np.int64),
        )

    costs = _signed_costs(
        probability,
        neutral_probability=neutral_probability,
        epsilon=float(cfg.multicut_probability_epsilon),
    )

    rows: list[tuple[int, tuple[int, ...]]] = []
    row_keys: set[tuple[int, tuple[int, ...]]] = set()
    last_cut: np.ndarray | None = None
    last_status = None
    last_message = ""

    for _round_index in range(int(cfg.multicut_max_rounds) + 1):
        if rows:
            data: list[float] = []
            row_index: list[int] = []
            col_index: list[int] = []
            for constraint_row, (cut_edge, path_edges) in enumerate(rows):
                row_index.append(constraint_row)
                col_index.append(cut_edge)
                data.append(1.0)
                for path_edge in path_edges:
                    row_index.append(constraint_row)
                    col_index.append(path_edge)
                    data.append(-1.0)

            matrix = csr_matrix(
                (data, (row_index, col_index)),
                shape=(len(rows), edge_count),
                dtype=np.float64,
            )
            constraints = LinearConstraint(
                matrix,
                lb=np.full(len(rows), -np.inf, dtype=np.float64),
                ub=np.zeros(len(rows), dtype=np.float64),
            )
        else:
            constraints = None

        options = {
            "presolve": True,
            "disp": False,
            "mip_rel_gap": float(cfg.multicut_mip_rel_gap),
        }
        if float(cfg.multicut_time_limit_seconds) > 0:
            options["time_limit"] = float(cfg.multicut_time_limit_seconds)

        result = milp(
            c=costs,
            integrality=np.ones(edge_count, dtype=np.uint8),
            bounds=Bounds(
                np.zeros(edge_count, dtype=np.float64),
                np.ones(edge_count, dtype=np.float64),
            ),
            constraints=constraints,
            options=options,
        )
        last_status = result.status
        last_message = str(result.message)

        if result.x is None:
            raise RuntimeError(
                "Multicut MILP returned no incumbent solution: "
                f"status={result.status}, message={result.message}"
            )

        cut = np.asarray(result.x >= 0.5, dtype=bool)
        last_cut = cut
        violations = _violated_cycle_constraints(
            node_count,
            edges,
            cut,
            max_constraints=int(cfg.multicut_max_constraints_per_round),
        )
        if not violations:
            return (
                cut,
                _components_from_uncut(node_count, edges, cut),
            )

        new_rows = 0
        for cut_edge, path_edges in violations:
            key = (cut_edge, tuple(path_edges))
            if key in row_keys:
                continue
            row_keys.add(key)
            rows.append(key)
            new_rows += 1

        if new_rows == 0:
            raise RuntimeError(
                "Multicut cutting-plane separation stalled on repeated "
                "cycle constraints."
            )

    remaining = (
        []
        if last_cut is None
        else _violated_cycle_constraints(
            node_count,
            edges,
            last_cut,
            max_constraints=1,
        )
    )
    raise RuntimeError(
        "Multicut did not reach cycle consistency within "
        f"{cfg.multicut_max_rounds} cutting rounds. "
        f"last_status={last_status}, last_message={last_message!r}, "
        f"remaining_violation={bool(remaining)}"
    )


class GraphPartitioner(nn.Module):
    """Partition watershed supervoxels with union-find or signed multicut.

    For ``multicut``, the supplied threshold is the neutral probability q in
    ``logit(p) - logit(q)``.  Thus the existing spatial threshold 0.845 is used
    directly as the Investigation-21-validated multicut neutral point.
    """

    def __init__(self, cfg: PartitionConfig | None = None):
        super().__init__()
        self.cfg = cfg or PartitionConfig()

    def _backend(self, stage: Literal["spatial", "final"]) -> str:
        return (
            self.cfg.spatial_partition_backend
            if stage == "spatial"
            else self.cfg.final_partition_backend
        )

    @staticmethod
    def _union_find_components(
        *,
        node_count: int,
        edges: np.ndarray,
        probability: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        uf = _UnionFind(node_count)
        if edges.shape[1]:
            order = np.argsort(-probability, kind="stable")
            for edge_row in order:
                if float(probability[edge_row]) < threshold:
                    break
                uf.union(
                    int(edges[0, edge_row]),
                    int(edges[1, edge_row]),
                )

        root_to_component: dict[int, int] = {}
        component = np.empty(node_count, dtype=np.int64)
        for node in range(node_count):
            root = uf.find(node)
            if root not in root_to_component:
                root_to_component[root] = len(root_to_component)
            component[node] = root_to_component[root]
        return component

    def forward(
        self,
        rag: RAGState,
        edge_logits: Tensor,
        threshold: float,
        *,
        stage: Literal["spatial", "final"] = "spatial",
    ) -> PartitionState:
        if edge_logits.shape != (rag.edge_index.shape[1],):
            raise ValueError(
                "edge_logits must align one-to-one with RAG edges"
            )
        if stage not in {"spatial", "final"}:
            raise ValueError("stage must be 'spatial' or 'final'")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("partition threshold must lie in (0, 1)")

        backend = self._backend(stage)
        if backend not in {"union_find", "multicut"}:
            raise ValueError(
                f"Unsupported partition backend: {backend!r}"
            )

        labels_out: List[Tensor] = []
        node_component_local = torch.zeros(
            rag.node_features.shape[0],
            device=rag.node_features.device,
            dtype=torch.long,
        )
        node_component_global = torch.zeros_like(node_component_local)
        component_counts: list[int] = []
        global_component_offset = 0

        probabilities = (
            edge_logits.detach().float().sigmoid().cpu().numpy()
        )
        edge_index_cpu = rag.edge_index.detach().long().cpu().numpy()
        edge_batch_cpu = rag.edge_batch.detach().long().cpu().numpy()

        for batch_index, supervoxels in enumerate(rag.supervoxel_labels):
            start = int(rag.node_offsets[batch_index].item())
            stop = int(rag.node_offsets[batch_index + 1].item())
            node_count = stop - start

            if node_count == 0:
                labels_out.append(torch.zeros_like(supervoxels))
                component_counts.append(0)
                continue

            edge_rows = np.flatnonzero(edge_batch_cpu == batch_index)
            if edge_rows.size:
                local_edges = (
                    edge_index_cpu[:, edge_rows] - start
                ).astype(np.int64, copy=False)
                local_probability = probabilities[edge_rows]
                if (
                    local_edges.min() < 0
                    or local_edges.max() >= node_count
                ):
                    raise IndexError(
                        "RAG edge references a node outside its batch "
                        "node-offset interval."
                    )
            else:
                local_edges = np.zeros((2, 0), dtype=np.int64)
                local_probability = np.zeros(0, dtype=np.float64)

            if backend == "union_find":
                local_component_np = self._union_find_components(
                    node_count=node_count,
                    edges=local_edges,
                    probability=local_probability,
                    threshold=float(threshold),
                )
            else:
                _, local_component_np = _solve_multicut_graph(
                    node_count=node_count,
                    edges=local_edges,
                    probability=local_probability,
                    neutral_probability=float(threshold),
                    cfg=self.cfg,
                )

            local_tensor = torch.as_tensor(
                local_component_np,
                device=supervoxels.device,
                dtype=torch.long,
            )
            node_component_local[start:stop] = local_tensor
            node_component_global[start:stop] = (
                local_tensor + global_component_offset
            )

            component_count = (
                int(local_component_np.max()) + 1 if node_count else 0
            )
            component_counts.append(component_count)

            mapping = torch.zeros(
                node_count + 1,
                device=supervoxels.device,
                dtype=torch.long,
            )
            mapping[1:] = local_tensor + 1
            labels_out.append(mapping[supervoxels.long()])
            global_component_offset += component_count

        return PartitionState(
            labels=labels_out,
            node_component=node_component_local,
            node_component_global=node_component_global,
            component_count_per_batch=torch.tensor(
                component_counts,
                device=rag.node_features.device,
                dtype=torch.long,
            ),
            edge_logits=edge_logits,
        )
