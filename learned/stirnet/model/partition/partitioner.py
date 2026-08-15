from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from ..types import PartitionState, RAGState


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


class GraphPartitioner(nn.Module):
    """Connectivity-preserving learned agglomeration over watershed supervoxels.

    PlantSeg shows the benefit of partitioning a supervoxel RAG rather than
    treating independent masks as final objects. This built-in solver is a
    dependency-free GASP-like greedy affinity agglomeration. Its interface is
    intentionally replaceable by Multicut/GASP/Mutex-Watershed later.
    """

    def forward(
        self,
        rag: RAGState,
        edge_logits: Tensor,
        threshold: float,
    ) -> PartitionState:
        if edge_logits.shape != (rag.edge_index.shape[1],):
            raise ValueError("edge_logits must align one-to-one with RAG edges")
        labels_out: List[Tensor] = []
        node_component_local = torch.zeros(
            rag.node_features.shape[0], device=rag.node_features.device, dtype=torch.long
        )
        node_component_global = torch.zeros_like(node_component_local)
        component_counts = []
        global_component_offset = 0

        probs = edge_logits.detach().sigmoid()
        for b, sv in enumerate(rag.supervoxel_labels):
            start = int(rag.node_offsets[b].item())
            stop = int(rag.node_offsets[b + 1].item())
            n = stop - start
            if n == 0:
                labels_out.append(torch.zeros_like(sv))
                component_counts.append(0)
                continue
            uf = _UnionFind(n)
            edge_rows = torch.nonzero(rag.edge_batch == b, as_tuple=False).flatten()
            if edge_rows.numel():
                order = edge_rows[torch.argsort(probs[edge_rows], descending=True)]
                for edge_row in order.tolist():
                    if float(probs[edge_row].item()) < threshold:
                        break
                    ga = int(rag.edge_index[0, edge_row].item())
                    gb = int(rag.edge_index[1, edge_row].item())
                    uf.union(ga - start, gb - start)
            roots = [uf.find(i) for i in range(n)]
            unique_roots = {}
            local_components = []
            for root in roots:
                if root not in unique_roots:
                    unique_roots[root] = len(unique_roots)
                local_components.append(unique_roots[root])
            local_tensor = torch.tensor(
                local_components, device=sv.device, dtype=torch.long
            )
            node_component_local[start:stop] = local_tensor
            node_component_global[start:stop] = local_tensor + global_component_offset
            count = len(unique_roots)
            component_counts.append(count)

            mapping = torch.zeros(n + 1, device=sv.device, dtype=torch.long)
            mapping[1:] = local_tensor + 1
            labels_out.append(mapping[sv.long()])
            global_component_offset += count

        return PartitionState(
            labels=labels_out,
            node_component=node_component_local,
            node_component_global=node_component_global,
            component_count_per_batch=torch.tensor(
                component_counts, device=rag.node_features.device, dtype=torch.long
            ),
            edge_logits=edge_logits,
        )
