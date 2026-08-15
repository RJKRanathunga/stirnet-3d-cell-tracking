from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class SpatialPyramid:
    features: List[Tensor]
    spacings_um: List[Tensor]
    strides: List[tuple[int, int, int]]
    padding_masks: Optional[List[Tensor]] = None


@dataclass
class TemporalNodeMemory:
    """Fine per-detection memory retained after detection-GNN message passing."""

    tokens: Tensor                    # [N,D]
    observed_ref_um: Tensor           # [N,3], physical observed zyx
    projected_ref_um: Tensor          # [N,3], tracklet target-frame zyx
    time_offset: Tensor               # [N], signed frames from target
    tracklet_id: Tensor               # [N], packed coarse-memory index
    batch_index: Tensor               # [N], logical sample index
    history_valid: Tensor             # [N]
    node_ids: Optional[Tensor] = None  # [N], preprocessing/debug identity

    @property
    def is_empty(self) -> bool:
        return self.tokens.shape[0] == 0


@dataclass
class TemporalState:
    tokens: Tensor
    ref_um: Tensor
    ref_cellscale: Tensor
    salience: Tensor
    reliability: Tensor
    status: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch_index: Tensor
    history_support: Optional[Tensor] = None
    history_support_valid: Optional[Tensor] = None
    history_support_dt: Optional[Tensor] = None
    history_support_center_um: Optional[Tensor] = None
    history_support_extent_um: Optional[Tensor] = None
    node_history_valid: Optional[Tensor] = None
    history_gate: Optional[Tensor] = None
    best_current_component_id: Optional[Tensor] = None
    best_component_overlap: Optional[Tensor] = None
    second_best_component_overlap: Optional[Tensor] = None
    node_memory: Optional[TemporalNodeMemory] = None

    @property
    def is_empty(self) -> bool:
        return self.tokens.numel() == 0 or self.tokens.shape[0] == 0


@dataclass
class QueryState:
    embeddings: Tensor
    references_cellscale: Tensor
    query_types: Tensor
    padding_mask: Tensor
    source_instance_ids: Tensor
    temporal_salience: Tensor
    temporal_reliability: Tensor
    initial_references_cellscale: Optional[Tensor] = None


@dataclass
class SpatialProposalState:
    embeddings: Tensor
    references_cellscale: Tensor
    scores: Tensor
    padding_mask: Tensor
    source_instance_ids: Tensor
    fallback_mask: Tensor


@dataclass
class StirNetOutput:
    exist_logits: Tensor
    centers_cellscale: Tensor
    coarse_mask_logits: Tensor
    coarse_spacing_um: Tensor
    query_embeddings: Tensor
    native_mask_embeddings: Tensor
    query_types: Tensor
    query_padding_mask: Tensor
    source_instance_ids: Tensor
    query_initial_references_cellscale: Tensor
    temporal_salience: Tensor
    temporal_reliability: Tensor
    aux_outputs: List[Dict[str, Tensor]]
    dense_outputs: Dict[str, Tensor]
    mask_features: Tensor
    spacing_um: Tensor
    dref_um: Tensor
    instance_labels: Tensor
    debug: Optional[Dict[str, Any]] = None
    proposals: Optional[SpatialProposalState] = None
    d0_features: Optional[Tensor] = None
    spatial_inputs: Optional[Tensor] = None
