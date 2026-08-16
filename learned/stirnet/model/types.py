from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class SpatialPyramid:
    features: List[Tensor]
    spacings_um: List[Tensor]
    strides: List[tuple[int, int, int]]
    padding_masks: Optional[List[Tensor]] = None

    @property
    def e0(self) -> Tensor:
        return self.features[0]

    @property
    def e1(self) -> Tensor:
        return self.features[1]

    @property
    def e2(self) -> Tensor:
        return self.features[2]

    @property
    def e3(self) -> Tensor:
        return self.features[3]


@dataclass
class SpatialDecodeState:
    d2: Tensor
    d1: Tensor
    d0: Tensor


@dataclass
class GeometryState:
    foreground_logits: Tensor
    surface_logits: Tensor
    separator_logits: Tensor
    sdf: Tensor
    flow: Tensor
    centroid_offset: Tensor
    seed_logits: Tensor
    features: Tensor

    def probabilities(self) -> Dict[str, Tensor]:
        return {
            "foreground": self.foreground_logits.sigmoid(),
            "surface": self.surface_logits.sigmoid(),
            "separator": self.separator_logits.sigmoid(),
            "seed": self.seed_logits.sigmoid(),
        }


@dataclass
class RAGState:
    node_features: Tensor
    node_embeddings: Tensor
    node_batch: Tensor
    node_supervoxel_id: Tensor
    node_centroid_um: Tensor
    node_volume_voxels: Tensor
    edge_index: Tensor
    edge_features: Tensor
    edge_embeddings: Tensor
    spatial_edge_logits: Tensor
    edge_batch: Tensor
    supervoxel_labels: List[Tensor]
    node_offsets: Tensor

    @property
    def is_empty(self) -> bool:
        return self.node_features.shape[0] == 0


@dataclass
class PartitionState:
    labels: List[Tensor]
    node_component: Tensor
    node_component_global: Tensor
    component_count_per_batch: Tensor
    edge_logits: Tensor


@dataclass
class InstanceState:
    tokens: Tensor
    ref_um: Tensor
    batch_index: Tensor
    local_ids: Tensor
    quality_logits: Tensor
    labels: List[Tensor]
    token_offsets: Tensor
    node_to_instance: Tensor
    spatial_tokens: Optional[Tensor] = None

    @property
    def is_empty(self) -> bool:
        return self.tokens.shape[0] == 0


@dataclass
class TemporalInput:
    graph_x: Tensor
    graph_edge_index: Tensor
    graph_edge_attr: Tensor
    tracklet_id: Tensor
    temporal_ref_um: Tensor
    temporal_status: Tensor
    temporal_batch: Tensor
    node_history_embedding: Optional[Tensor] = None


@dataclass
class TemporalState:
    tokens: Tensor
    ref_um: Tensor
    batch_index: Tensor
    salience: Tensor
    reliability: Tensor
    status: Tensor
    node_tokens: Optional[Tensor] = None

    @property
    def is_empty(self) -> bool:
        return self.tokens.shape[0] == 0


@dataclass
class ReasoningState:
    instance_tokens: Tensor
    instance_exist_logits: Tensor
    split_logits: Tensor
    temporal_support: Tensor
    temporal_attention_entropy: Tensor
    edge_temporal_delta: Tensor
    edge_temporal_gate: Tensor
    final_edge_logits: Tensor
    recovery_track_indices: Tensor
    recovery_logits: Tensor
    recovery_scores: Tensor


@dataclass
class RefinementRequest:
    batch_index: int
    center_um: Tensor
    query_token: Tensor
    kind: str
    source_index: int
    score: float
    selection_source: str = "model"


@dataclass
class RefinementState:
    geometry: GeometryState
    requests: List[RefinementRequest] = field(default_factory=list)
    applied_count: int = 0
    model_request_count: int = 0
    teacher_request_count: int = 0


@dataclass
class GeometryForwardOutput:
    geometry: GeometryState
    spatial_pyramid: SpatialPyramid
    decoded_spatial: SpatialDecodeState


@dataclass
class SpatialForwardOutput(GeometryForwardOutput):
    rag: RAGState
    spatial_partition: PartitionState


@dataclass
class StirNetOutput:
    final_labels: List[Tensor]
    centers_um: List[Tensor]
    geometry: GeometryState
    spatial_pyramid: SpatialPyramid
    decoded_spatial: SpatialDecodeState
    rag: RAGState
    spatial_partition: PartitionState
    provisional_instances: InstanceState
    temporal: TemporalState
    reasoning: ReasoningState
    final_partition: PartitionState
    initial_geometry: GeometryState
    initial_rag: RAGState
    initial_spatial_partition: PartitionState
    initial_provisional_instances: InstanceState
    initial_reasoning: ReasoningState
    refinement: Optional[RefinementState] = None
    debug: Optional[Dict[str, Any]] = None

    @property
    def dense_outputs(self) -> Dict[str, Tensor]:
        # Compatibility helper for existing diagnostics/training code.
        return {
            "foreground_logits": self.geometry.foreground_logits,
            "surface_boundary_logits": self.geometry.surface_logits,
            "boundary_logits": self.geometry.separator_logits,
            "separator_logits": self.geometry.separator_logits,
            "sdf": self.geometry.sdf,
            "flow": self.geometry.flow,
            "centroid_offset": self.geometry.centroid_offset,
            "seed_logits": self.geometry.seed_logits,
        }
