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


@dataclass
class StirNetOutput:
    exist_logits: Tensor
    centers_cellscale: Tensor
    coarse_mask_logits: Tensor
    query_embeddings: Tensor
    native_mask_embeddings: Tensor
    query_types: Tensor
    query_padding_mask: Tensor
    source_instance_ids: Tensor
    temporal_salience: Tensor
    temporal_reliability: Tensor
    aux_outputs: List[Dict[str, Tensor]]
    dense_outputs: Dict[str, Tensor]
    mask_features: Tensor
    spacing_um: Tensor
    dref_um: Tensor
    instance_labels: Tensor
    debug: Optional[Dict[str, Any]] = None
