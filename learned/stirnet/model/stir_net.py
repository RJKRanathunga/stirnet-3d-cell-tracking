from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .geometry.decoder import DenseGeometryDecoder
from .instances.tokenizer import InstanceTokenizer, centers_from_labels
from .partition.graph_net import SpatialRAGNetwork
from .partition.partitioner import GraphPartitioner
from .partition.rag import RAGBuilder
from .partition.watershed import LearnedGeometryWatershed
from .refinement.local_refiner import LocalGeometryRefiner
from .refinement.requests import build_refinement_requests
from .spatial.acquisition import AcquisitionEmbedding
from .spatial.backbone import AnisotropyAwareSpatialBackbone
from .spatial.evidence_stem import EvidenceFusionStem
from .temporal.fusion import InstanceTemporalReasoner
from .temporal.graph_encoder import TemporalGraphEncoder
from .temporal.history import HistoricalInstanceEncoder
from .temporal.observer import TemporalSpatialObserver
from .types import (
    GeometryState,
    InstanceState,
    PartitionState,
    RAGState,
    ReasoningState,
    RefinementState,
    StirNetOutput,
    TemporalInput,
    TemporalState,
)


class StirNet(nn.Module):
    """Spatial-first STIR-Net V2 architecture.

    Design invariant:
        dense current-frame geometry -> connected spatial hypotheses ->
        instance/temporal reasoning -> gated local corrections -> final partition.

    Temporal evidence is never allowed to globally overwrite D0/D1/D2. It may
    only (1) read the spatial representation, (2) alter object/RAG decisions,
    and (3) request bounded native-resolution geometry residuals.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.cfg.validate()

        self.acquisition = AcquisitionEmbedding(self.cfg.spatial.acquisition_dim)
        self.evidence_stem = EvidenceFusionStem(self.cfg.evidence, self.cfg.spatial)
        self.spatial_backbone = AnisotropyAwareSpatialBackbone(self.cfg.spatial)
        self.geometry_decoder = DenseGeometryDecoder(
            self.cfg.geometry, self.cfg.spatial
        )

        self.watershed = LearnedGeometryWatershed(self.cfg.partition)
        self.rag_builder = RAGBuilder(self.cfg.partition, self.cfg.spatial)
        self.rag_network = SpatialRAGNetwork(
            self.cfg.partition,
            self.rag_builder.node_feature_dim,
            self.rag_builder.edge_feature_dim,
        )
        self.partitioner = GraphPartitioner()
        self.instance_tokenizer = InstanceTokenizer(
            self.cfg.instances, self.cfg.spatial
        )

        self.history_encoder = HistoricalInstanceEncoder(
            self.cfg.history, self.cfg.temporal
        )
        self.temporal_encoder = TemporalGraphEncoder(self.cfg.temporal)
        self.temporal_observer = TemporalSpatialObserver(
            self.cfg.temporal, self.cfg.spatial, self.cfg.geometry
        )
        self.instance_temporal = InstanceTemporalReasoner(
            self.cfg.temporal,
            self.cfg.instances,
            self.cfg.partition,
            self.cfg.refinement,
        )
        self.local_refiner = LocalGeometryRefiner(
            self.cfg.refinement,
            self.cfg.spatial,
            self.cfg.geometry.hidden_channels,
            self.cfg.instances.d_model,
        )

    def _coerce_temporal_input(
        self,
        temporal_input: TemporalInput | None,
        *,
        graph_x: Tensor | None,
        graph_edge_index: Tensor | None,
        graph_edge_attr: Tensor | None,
        tracklet_id: Tensor | None,
        temporal_ref_um: Tensor | None,
        temporal_status: Tensor | None,
        temporal_batch: Tensor | None,
        node_instance_grid: Tensor | None,
        node_history_valid: Tensor | None,
    ) -> TemporalInput | None:
        if temporal_input is not None:
            if any(
                x is not None
                for x in (
                    graph_x,
                    graph_edge_index,
                    graph_edge_attr,
                    tracklet_id,
                    temporal_ref_um,
                    temporal_status,
                    temporal_batch,
                )
            ):
                raise ValueError(
                    "Pass either temporal_input or legacy temporal tensors, not both"
                )
            data = temporal_input
        elif graph_x is None:
            return None
        else:
            required = {
                "graph_edge_index": graph_edge_index,
                "graph_edge_attr": graph_edge_attr,
                "tracklet_id": tracklet_id,
                "temporal_ref_um": temporal_ref_um,
                "temporal_status": temporal_status,
                "temporal_batch": temporal_batch,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "Legacy temporal input is incomplete: " + ", ".join(missing)
                )
            data = TemporalInput(
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,  # type: ignore[arg-type]
                graph_edge_attr=graph_edge_attr,  # type: ignore[arg-type]
                tracklet_id=tracklet_id,  # type: ignore[arg-type]
                temporal_ref_um=temporal_ref_um,  # type: ignore[arg-type]
                temporal_status=temporal_status,  # type: ignore[arg-type]
                temporal_batch=temporal_batch,  # type: ignore[arg-type]
            )
        if node_instance_grid is not None and self.cfg.history.enabled:
            history = self.history_encoder(node_instance_grid, node_history_valid)
            data = replace(data, node_history_embedding=history)
        return data

    def _spatial_objects(
        self,
        geometry: GeometryState,
        decoded,
        spatial_inputs: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        spatial_padding_mask: Tensor | None,
    ) -> tuple[RAGState, PartitionState, InstanceState]:
        supervoxels = self.watershed(
            geometry, spacing_um, dref_um, spatial_padding_mask
        )
        rag = self.rag_builder(
            supervoxels,
            decoded.d0,
            spatial_inputs,
            geometry,
            spacing_um,
            dref_um,
        )
        rag = self.rag_network(rag)
        partition = self.partitioner(
            rag,
            rag.spatial_edge_logits,
            self.cfg.partition.spatial_merge_threshold,
        )
        instances = self.instance_tokenizer(
            partition, rag, decoded, geometry, spacing_um, dref_um
        )
        return rag, partition, instances

    def _temporal_state(
        self,
        temporal_data: TemporalInput | None,
        decoded,
        geometry: GeometryState,
        pyramid,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> TemporalState:
        temporal = self.temporal_encoder(temporal_data)
        return self.temporal_observer(
            temporal,
            decoded,
            geometry,
            pyramid.spacings_um,
            spacing_um,
            dref_um,
        )

    @staticmethod
    def _filter_by_existence(
        partition: PartitionState,
        rag: RAGState,
        instances: InstanceState,
        reasoning: ReasoningState,
        threshold: float,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Remove low-existence final components without breaking connectivity."""
        if reasoning.instance_exist_logits.numel() == 0:
            return partition.labels, [
                label.new_zeros((int(label.max().item()),), dtype=torch.float32)
                for label in partition.labels
            ]
        exist_prob = reasoning.instance_exist_logits.sigmoid()
        filtered: list[Tensor] = []
        scores_per_batch: list[Tensor] = []
        for b, labels in enumerate(partition.labels):
            n = int(labels.max().item())
            if n == 0:
                filtered.append(labels)
                scores_per_batch.append(exist_prob.new_zeros((0,)))
                continue
            start = int(rag.node_offsets[b].item())
            stop = int(rag.node_offsets[b + 1].item())
            local_components = partition.node_component[start:stop]
            score_rows = []
            keep = []
            for comp in range(n):
                node_local = torch.nonzero(
                    local_components == comp, as_tuple=False
                ).flatten()
                if node_local.numel() == 0:
                    score = exist_prob.new_tensor(0.0)
                else:
                    node_global = node_local + start
                    provisional = torch.unique(instances.node_to_instance[node_global])
                    score = exist_prob[provisional].mean()
                score_rows.append(score)
                keep.append(bool(score >= threshold))
            mapping = torch.zeros(n + 1, device=labels.device, dtype=torch.long)
            next_id = 1
            for old_id, is_kept in enumerate(keep, 1):
                if is_kept:
                    mapping[old_id] = next_id
                    next_id += 1
            filtered.append(mapping[labels.long()])
            scores_per_batch.append(torch.stack(score_rows))
        return filtered, scores_per_batch

    def forward(
        self,
        spatial_inputs: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        temporal_input: TemporalInput | None = None,
        spatial_padding_mask: Tensor | None = None,
        prior_keep_mask: Tensor | None = None,
        run_refinement: bool | None = None,
        apply_existence_filter: bool | None = None,
        return_debug: bool = False,
        # Compatibility bridge for the current repository's temporal tensors.
        graph_x: Tensor | None = None,
        graph_edge_index: Tensor | None = None,
        graph_edge_attr: Tensor | None = None,
        tracklet_id: Tensor | None = None,
        temporal_ref_um: Tensor | None = None,
        temporal_status: Tensor | None = None,
        temporal_batch: Tensor | None = None,
        node_instance_grid: Tensor | None = None,
        node_history_valid: Tensor | None = None,
    ) -> StirNetOutput:
        if spatial_inputs.ndim != 5:
            raise ValueError("spatial_inputs must have shape [B,C,Z,Y,X]")
        if spatial_inputs.shape[1] != self.cfg.spatial.in_channels:
            raise ValueError(
                f"Expected {self.cfg.spatial.in_channels} spatial input channels; "
                f"got {spatial_inputs.shape[1]}"
            )
        if spacing_um.ndim == 1:
            spacing_um = spacing_um[None]
        if dref_um.ndim == 0:
            dref_um = dref_um[None]
        if spacing_um.shape != (spatial_inputs.shape[0], 3):
            raise ValueError("spacing_um must have shape [B,3]")
        if dref_um.shape != (spatial_inputs.shape[0],):
            raise ValueError("dref_um must have shape [B]")

        temporal_data = self._coerce_temporal_input(
            temporal_input,
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_edge_attr=graph_edge_attr,
            tracklet_id=tracklet_id,
            temporal_ref_um=temporal_ref_um,
            temporal_status=temporal_status,
            temporal_batch=temporal_batch,
            node_instance_grid=node_instance_grid,
            node_history_valid=node_history_valid,
        )

        acquisition = self.acquisition(spacing_um, dref_um)
        stem = self.evidence_stem(
            spatial_inputs, acquisition, prior_keep_mask=prior_keep_mask
        )
        pyramid, decoded = self.spatial_backbone(
            stem, spacing_um, acquisition, spatial_padding_mask
        )
        initial_geometry = self.geometry_decoder(decoded.d0, acquisition)

        initial_rag, initial_partition, initial_instances = self._spatial_objects(
            initial_geometry,
            decoded,
            spatial_inputs,
            spacing_um,
            dref_um,
            spatial_padding_mask,
        )
        temporal = self._temporal_state(
            temporal_data,
            decoded,
            initial_geometry,
            pyramid,
            spacing_um,
            dref_um,
        )
        initial_reasoning = self.instance_temporal(
            initial_instances, initial_rag, temporal, dref_um
        )

        use_refinement = (
            self.cfg.refinement.enabled if run_refinement is None else run_refinement
        )
        refinement: RefinementState | None = None
        geometry = initial_geometry
        rag = initial_rag
        spatial_partition = initial_partition
        instances = initial_instances
        reasoning = initial_reasoning

        if use_refinement:
            requests = build_refinement_requests(
                initial_instances,
                initial_rag,
                temporal,
                initial_reasoning,
                dref_um,
                self.cfg.refinement,
            )
            refinement = self.local_refiner(
                decoded.d0,
                spatial_inputs,
                initial_geometry,
                spacing_um,
                dref_um,
                requests,
            )
            if refinement.applied_count:
                geometry = refinement.geometry
                # Re-run the complete partition after local geometry changes.
                # This is how recovery can create a new object and split requests
                # can create a new separator without independent mask painting.
                rag, spatial_partition, instances = self._spatial_objects(
                    geometry,
                    decoded,
                    spatial_inputs,
                    spacing_um,
                    dref_um,
                    spatial_padding_mask,
                )
                temporal = self._temporal_state(
                    temporal_data,
                    decoded,
                    geometry,
                    pyramid,
                    spacing_um,
                    dref_um,
                )
                reasoning = self.instance_temporal(
                    instances, rag, temporal, dref_um
                )

        final_partition = self.partitioner(
            rag,
            reasoning.final_edge_logits,
            self.cfg.partition.final_merge_threshold,
        )

        use_exist = (
            self.cfg.instances.apply_existence_filter
            if apply_existence_filter is None
            else apply_existence_filter
        )
        if use_exist:
            final_labels, existence_scores = self._filter_by_existence(
                final_partition,
                rag,
                instances,
                reasoning,
                self.cfg.instances.exist_threshold,
            )
        else:
            final_labels = final_partition.labels
            existence_scores = [
                reasoning.instance_exist_logits.new_ones(
                    (int(label.max().item()),)
                )
                for label in final_labels
            ]

        centers = centers_from_labels(final_labels, spacing_um, geometry.sdf)
        debug: dict[str, Any] | None = None
        if return_debug:
            debug = {
                "research_design": {
                    "dense_geometry": "Omnipose + NucMM + NISNet3D",
                    "supervoxel_partition": "PlantSeg-style learned geometry + RAG",
                    "temporal_fusion": "spatial-read-only then gated RAG/ROI residuals",
                },
                "initial_supervoxel_count": [
                    int(x.max().item()) for x in initial_rag.supervoxel_labels
                ],
                "working_supervoxel_count": [
                    int(x.max().item()) for x in rag.supervoxel_labels
                ],
                "spatial_instance_count": [
                    int(x.max().item()) for x in spatial_partition.labels
                ],
                "final_instance_count": [int(x.max().item()) for x in final_labels],
                "final_component_existence_scores": existence_scores,
                "temporal_edge_gate": reasoning.edge_temporal_gate.detach(),
                "temporal_edge_delta": reasoning.edge_temporal_delta.detach(),
                "refinement_requests": []
                if refinement is None
                else [
                    {
                        "batch": r.batch_index,
                        "kind": r.kind,
                        "source": r.source_index,
                        "score": r.score,
                    }
                    for r in refinement.requests
                ],
            }

        return StirNetOutput(
            final_labels=final_labels,
            centers_um=centers,
            geometry=geometry,
            spatial_pyramid=pyramid,
            decoded_spatial=decoded,
            rag=rag,
            spatial_partition=spatial_partition,
            provisional_instances=instances,
            temporal=temporal,
            reasoning=reasoning,
            final_partition=final_partition,
            initial_geometry=initial_geometry,
            initial_rag=initial_rag,
            initial_spatial_partition=initial_partition,
            initial_provisional_instances=initial_instances,
            initial_reasoning=initial_reasoning,
            refinement=refinement,
            debug=debug,
        )
