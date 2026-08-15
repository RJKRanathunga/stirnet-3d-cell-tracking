from __future__ import annotations

from collections import defaultdict
from typing import List

import torch

from ..config import RefinementConfig
from ..types import (
    InstanceState,
    RAGState,
    ReasoningState,
    RefinementRequest,
    TemporalState,
)


def build_refinement_requests(
    instances: InstanceState,
    rag: RAGState,
    temporal: TemporalState,
    reasoning: ReasoningState,
    cfg: RefinementConfig,
) -> List[RefinementRequest]:
    """Select only ambiguous/recovery ROIs; strong spatial geometry is untouched."""
    requests: List[RefinementRequest] = []

    # Spatial object likely contains multiple cells.
    if reasoning.split_logits.numel():
        split_prob = reasoning.split_logits.sigmoid()
        for idx in torch.nonzero(split_prob >= cfg.split_threshold, as_tuple=False).flatten().tolist():
            requests.append(
                RefinementRequest(
                    batch_index=int(instances.batch_index[idx].item()),
                    center_um=instances.ref_um[idx],
                    query_token=reasoning.instance_tokens[idx],
                    kind="split",
                    source_index=idx,
                    score=float(split_prob[idx].item()),
                )
            )

    # Reliable temporal hypothesis without a satisfactory current spatial cell.
    for row, tidx in enumerate(reasoning.recovery_track_indices.tolist()):
        score = float(reasoning.recovery_scores[row].item())
        if score < cfg.recovery_threshold:
            continue
        requests.append(
            RefinementRequest(
                batch_index=int(temporal.batch_index[tidx].item()),
                center_um=temporal.ref_um[tidx],
                query_token=temporal.tokens[tidx],
                kind="recovery",
                source_index=tidx,
                score=score,
            )
        )

    # Edges near the merge/separate decision boundary are exactly where a
    # native-resolution second look is justified. This is the controlled place
    # where temporal reasoning may influence voxels.
    if rag.edge_index.shape[1]:
        ambiguous = rag.spatial_edge_logits.abs() <= cfg.ambiguity_logit_abs_max
        useful_temporal = reasoning.edge_temporal_gate > 0.05
        rows = torch.nonzero(ambiguous & useful_temporal, as_tuple=False).flatten()
        for edge_row in rows.tolist():
            a = int(rag.edge_index[0, edge_row].item())
            b = int(rag.edge_index[1, edge_row].item())
            ia = int(instances.node_to_instance[a].item())
            ib = int(instances.node_to_instance[b].item())
            if reasoning.instance_tokens.shape[0] == 0:
                continue
            token = 0.5 * (
                reasoning.instance_tokens[ia] + reasoning.instance_tokens[ib]
            )
            center = 0.5 * (rag.node_centroid_um[a] + rag.node_centroid_um[b])
            score = float(reasoning.edge_temporal_gate[edge_row].item())
            requests.append(
                RefinementRequest(
                    batch_index=int(rag.edge_batch[edge_row].item()),
                    center_um=center,
                    query_token=token,
                    kind="edge",
                    source_index=edge_row,
                    score=score,
                )
            )

    # Bound work per batch while keeping the highest-value requests.
    grouped: dict[int, list[RefinementRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.batch_index].append(request)
    selected: List[RefinementRequest] = []
    for b in sorted(grouped):
        ranked = sorted(grouped[b], key=lambda r: r.score, reverse=True)
        selected.extend(ranked[: cfg.max_rois_per_batch])
    return selected
