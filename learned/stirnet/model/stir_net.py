from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import StirNetConfig
from .coreasoning import CoReasoningBlock
from .graph_encoder import DetectionGraphEncoder
from .history_encoder import HistoricalInstanceEncoder, HistoryFusion
from .heads import DenseAuxiliaryHeads, MaskEmbeddingHead, render_native_masks
from .local_masks import LocalNativeMaskDecoder
from .query_builder import QUERY_SPATIAL_PROPOSAL
from .query_builder import InstanceQueryBuilder
from .query_decoder import InstanceQueryDecoder
from .spacing import AcquisitionEmbedding
from .spatial_decoder import SpatialDecoder
from .spatial_encoder import SpatialEncoder
from .spatial_proposals import SpatialProposalGenerator
from .temporal_hypotheses import TemporalStateBuilder, TrackletPooler
from .types import StirNetOutput, TemporalNodeMemory, TemporalState


class StirNet(nn.Module):
    def __init__(self, cfg: StirNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or StirNetConfig()
        self._validate_config()
        c0, c1, c2, c3 = self.cfg.spatial.channels
        checkpoint_spatial = (
            self.cfg.training.activation_checkpointing
            and self.cfg.training.checkpoint_spatial
        )
        checkpoint_coreasoning = (
            self.cfg.training.activation_checkpointing
            and self.cfg.training.checkpoint_coreasoning
        )
        self.acquisition = AcquisitionEmbedding(self.cfg.spatial.acquisition_dim)
        self.encoder = SpatialEncoder(
            self.cfg.spatial, activation_checkpointing=checkpoint_spatial
        )
        self.decoder = SpatialDecoder(
            self.cfg.spatial, activation_checkpointing=checkpoint_spatial
        )
        self.graph_encoder = DetectionGraphEncoder(self.cfg.temporal)
        if self.cfg.history.enabled:
            self.history_encoder = HistoricalInstanceEncoder(
                self.cfg.history,
                self.cfg.temporal.d_model,
                activation_checkpointing=(
                    self.cfg.training.activation_checkpointing
                    and self.cfg.training.checkpoint_history
                ),
            )
            self.history_fusion = HistoryFusion(
                self.cfg.temporal.d_model, self.cfg.history.gate_init_bias
            )
        else:
            self.history_encoder = nn.Identity()
            self.history_fusion = nn.Identity()
        self.tracklet_pooler = TrackletPooler(self.cfg.temporal.d_model)
        self.temporal_builder = TemporalStateBuilder(self.cfg.temporal)
        self.cr1 = CoReasoningBlock(
            c3, self.cfg.spatial, self.cfg.temporal, self.cfg.coreasoning,
            history_cfg=self.cfg.history,
            activation_checkpointing=checkpoint_coreasoning,
        )
        self.cr2 = CoReasoningBlock(
            c2, self.cfg.spatial, self.cfg.temporal, self.cfg.coreasoning,
            history_cfg=self.cfg.history,
            activation_checkpointing=checkpoint_coreasoning,
        )
        self.query_builder = InstanceQueryBuilder(
            self.cfg.queries,
            feature_channels=c2,
            temporal_cfg=self.cfg.temporal,
            proposal_cfg=self.cfg.proposals,
        )
        self.query_decoder = InstanceQueryDecoder(
            (c3,c2,c1),
            self.cfg.decoder,
            self.cfg.queries,
            self.cfg.temporal,
            self.cfg.proposals,
        )
        self.native_mask_head = MaskEmbeddingHead(self.cfg.decoder.d_model,self.cfg.spatial.mask_dim)
        self.local_mask_decoder = LocalNativeMaskDecoder(
            self.cfg.local_masks,
            d0_channels=c0,
            spatial_input_channels=self.cfg.spatial.in_channels,
            query_dim=self.cfg.decoder.d_model,
            background_logit=self.cfg.queries.native_background_logit,
        )
        self.dense_heads = DenseAuxiliaryHeads(c0)
        self.spatial_proposal_generator = SpatialProposalGenerator(
            self.cfg.proposals,
            d0_channels=c0,
            e2_channels=c2,
            spatial_input_channels=self.cfg.spatial.in_channels,
        )

    def _validate_config(self) -> None:
        channels = self.cfg.spatial.channels
        if len(channels) != 4 or any(c <= 0 for c in channels):
            raise ValueError(
                "STIR-Net V1 requires four positive spatial channel widths "
                f"(E0..E3); got {channels}."
            )
        dims = {
            "temporal.d_model": self.cfg.temporal.d_model,
            "coreasoning.d_model": self.cfg.coreasoning.d_model,
            "queries.d_model": self.cfg.queries.d_model,
            "decoder.d_model": self.cfg.decoder.d_model,
        }
        if len(set(dims.values())) != 1:
            values = ", ".join(f"{name}={value}" for name, value in dims.items())
            raise ValueError(
                "STIR-Net V1 shares one representation width across temporal, "
                f"co-reasoning, query, and decoder modules; got {values}."
            )
        if self.cfg.decoder.layers != 3:
            raise ValueError("STIR-Net V1 requires exactly three query decoder layers")
        for name, d_model, heads in (
            ("temporal", self.cfg.temporal.d_model, self.cfg.temporal.graph_heads),
            ("temporal memory", self.cfg.temporal.d_model, self.cfg.temporal.memory_heads),
            ("coreasoning", self.cfg.coreasoning.d_model, self.cfg.coreasoning.heads),
            ("decoder", self.cfg.decoder.d_model, self.cfg.decoder.heads),
        ):
            if heads <= 0 or d_model % heads:
                raise ValueError(f"{name} d_model={d_model} must be divisible by heads={heads}")
        if self.cfg.history.input_channels != 4:
            raise ValueError("STIR-Net history contract currently requires four input channels")
        if self.cfg.history.support_channels != 2:
            raise ValueError("STIR-Net history attention requires occupancy and SDF support")
        if self.cfg.history.grid_size <= 1 or self.cfg.history.extent_dref <= 0:
            raise ValueError("history grid size and physical extent must be positive")
        if self.cfg.temporal.edge_dim != 15:
            raise ValueError("STIR-Net candidate detection edge contract requires edge_dim=15")
        if self.cfg.temporal.memory_debug_topk < 0:
            raise ValueError("memory_debug_topk must be non-negative")
        if (
            self.cfg.temporal.max_candidate_edges is not None
            and self.cfg.temporal.max_candidate_edges < 0
        ):
            raise ValueError("max_candidate_edges must be non-negative or None")
        proposal = self.cfg.proposals
        if proposal.query_mode not in {"legacy", "spatial_proposals"}:
            raise ValueError(
                "proposals.query_mode must be 'legacy' or 'spatial_proposals'"
            )
        if proposal.max_proposals <= 0 or proposal.candidate_pool_size <= 0:
            raise ValueError("proposal candidate limits must be positive")
        if proposal.candidate_pool_size < proposal.max_proposals:
            raise ValueError(
                "proposals.candidate_pool_size must be at least max_proposals"
            )
        if proposal.local_grid_size <= 0 or proposal.local_grid_size % 2 == 0:
            raise ValueError("proposals.local_grid_size must be a positive odd integer")
        if proposal.local_dim <= 0:
            raise ValueError("proposals.local_dim must be positive")
        positive_proposal_values = {
            "nms_radius_dref": proposal.nms_radius_dref,
            "local_extent_dref": proposal.local_extent_dref,
            "source_fallback_match_radius_dref": proposal.source_fallback_match_radius_dref,
            "match_radius_dref": proposal.match_radius_dref,
            "attention_radius_layer0_dref": proposal.attention_radius_layer0_dref,
            "attention_radius_layer1_dref": proposal.attention_radius_layer1_dref,
            "attention_radius_layer2_dref": proposal.attention_radius_layer2_dref,
            "native_support_radius_dref": proposal.native_support_radius_dref,
        }
        invalid_proposal_values = [
            name for name, value in positive_proposal_values.items() if value <= 0
        ]
        if invalid_proposal_values:
            raise ValueError(
                "proposal physical radii/extents must be positive: "
                + ", ".join(invalid_proposal_values)
            )
        if not 0 <= proposal.inference_score_threshold <= 1:
            raise ValueError("proposal inference score threshold must be in [0, 1]")
        proposal_center_limit = (
            self.cfg.decoder.proposal_center_max_offset_dref
            if self.cfg.decoder.proposal_center_step_dref is None
            else self.cfg.decoder.proposal_center_step_dref
        )
        if proposal_center_limit <= 0:
            raise ValueError("proposal center maximum offset must be positive")
        local = self.cfg.local_masks
        if local.support_radius_dref <= 0:
            raise ValueError("local mask support radius must be positive")
        if local.hidden_channels <= 0 or local.query_channels <= 0:
            raise ValueError("local mask channel widths must be positive")
        if local.query_chunk_size <= 0:
            raise ValueError("local mask query chunk size must be positive")

    def _build_temporal(
        self, graph_x: Tensor, graph_edge_index: Tensor, graph_edge_attr: Tensor,
        tracklet_id: Tensor, temporal_ref_um: Tensor, temporal_status: Tensor,
        hypothesis_edge_index: Tensor, hypothesis_edge_attr: Tensor,
        temporal_batch: Tensor, dref_um: Tensor,
        node_instance_grid: Tensor | None = None,
        node_history_valid: Tensor | None = None,
        history_support: Tensor | None = None,
        history_support_valid: Tensor | None = None,
        history_support_dt: Tensor | None = None,
        history_support_center_um: Tensor | None = None,
        history_support_extent_um: Tensor | None = None,
        best_current_component_id: Tensor | None = None,
        best_component_overlap: Tensor | None = None,
        second_best_component_overlap: Tensor | None = None,
        node_observed_ref_um: Tensor | None = None,
        node_time_offset: Tensor | None = None,
        node_ids: Tensor | None = None,
        detection_graph_ablation: str = "full",
    ) -> TemporalState:
        M=temporal_ref_um.shape[0]
        if graph_edge_attr.shape[-1] != self.cfg.temporal.edge_dim:
            if graph_edge_attr.shape[-1] == 14 and self.cfg.temporal.edge_dim == 15:
                graph_edge_attr=torch.nn.functional.pad(graph_edge_attr,(0,1))
            else:
                raise ValueError(
                    "graph_edge_attr width must match temporal.edge_dim "
                    f"({self.cfg.temporal.edge_dim}); got {graph_edge_attr.shape[-1]}"
                )
        if detection_graph_ablation not in {"full","accepted_only"}:
            raise ValueError("detection_graph_ablation must be 'full' or 'accepted_only'")
        if (
            self.cfg.temporal.max_candidate_edges is not None
            and graph_edge_index.shape[1] > self.cfg.temporal.max_candidate_edges
        ):
            raise RuntimeError(
                f"Detection graph has {graph_edge_index.shape[1]} edges, exceeding "
                f"max_candidate_edges={self.cfg.temporal.max_candidate_edges}; "
                "STIR-Net will not silently truncate candidate evidence."
            )
        if detection_graph_ablation=="accepted_only" and graph_edge_attr.shape[0]:
            accepted=graph_edge_attr[:,14]>0.5
            graph_edge_index=graph_edge_index[:,accepted]
            graph_edge_attr=graph_edge_attr[accepted]
        if hypothesis_edge_attr.shape[-1] != self.cfg.temporal.hypothesis_edge_dim:
            if hypothesis_edge_attr.shape[-1] == 8 and self.cfg.temporal.hypothesis_edge_dim == 22:
                hypothesis_edge_attr=torch.nn.functional.pad(hypothesis_edge_attr,(0,14))
            else:
                raise ValueError(
                    "hypothesis_edge_attr width must match temporal.hypothesis_edge_dim "
                    f"({self.cfg.temporal.hypothesis_edge_dim}); got {hypothesis_edge_attr.shape[-1]}"
                )
        node_count=graph_x.shape[0]
        if node_count:
            if tracklet_id.shape != (node_count,):
                raise ValueError("tracklet_id must align one-to-one with graph_x")
            node_batch=temporal_batch[tracklet_id]
            if node_observed_ref_um is None:
                node_observed_ref_um=(
                    graph_x[:,1:4].float()*dref_um[node_batch,None].float()
                )
            if node_time_offset is None:
                node_time_offset=(
                    graph_x[:,0].float()*float(self.cfg.temporal.temporal_radius)
                )
            if node_history_valid is None:
                node_history_valid=torch.zeros(
                    node_count,device=graph_x.device,dtype=torch.bool
                )
        else:
            node_batch=tracklet_id.new_zeros((0,))
            node_observed_ref_um=graph_x.new_zeros((0,3))
            node_time_offset=graph_x.new_zeros((0,))
            if node_history_valid is None:
                node_history_valid=torch.zeros(
                    0,device=graph_x.device,dtype=torch.bool
                )
        if M==0:
            tokens=graph_x.new_zeros((0,self.cfg.temporal.d_model))
            z1=graph_x.new_zeros((0,1))
            node_memory=TemporalNodeMemory(
                tokens=graph_x.new_zeros((node_count,self.cfg.temporal.d_model)),
                observed_ref_um=node_observed_ref_um,
                projected_ref_um=graph_x.new_zeros((node_count,3)),
                time_offset=node_time_offset,
                tracklet_id=tracklet_id,
                batch_index=node_batch,
                history_valid=node_history_valid,
                node_ids=node_ids,
            )
            return TemporalState(
                tokens,temporal_ref_um,temporal_ref_um,z1,z1,temporal_status,
                hypothesis_edge_index,hypothesis_edge_attr,temporal_batch,
                history_support=history_support,
                history_support_valid=history_support_valid,
                history_support_dt=history_support_dt,
                history_support_center_um=history_support_center_um,
                history_support_extent_um=history_support_extent_um,
                node_history_valid=node_history_valid,
                history_gate=graph_x.new_zeros((graph_x.shape[0],1)),
                best_current_component_id=best_current_component_id,
                best_component_overlap=best_component_overlap,
                second_best_component_overlap=second_best_component_overlap,
                node_memory=node_memory,
            )
        if graph_x.shape[0]:
            scalar_embedding=self.graph_encoder.project_scalars(graph_x)
            if self.cfg.history.enabled:
                if node_instance_grid is None:
                    node_instance_grid=graph_x.new_zeros(
                        (graph_x.shape[0],self.cfg.history.input_channels,
                         self.cfg.history.grid_size,self.cfg.history.grid_size,self.cfg.history.grid_size)
                    )
                if node_history_valid is None:
                    node_history_valid=torch.zeros(graph_x.shape[0],device=graph_x.device,dtype=torch.bool)
                history_embedding=self.history_encoder(node_instance_grid,node_history_valid)
                fused,history_gate=self.history_fusion(
                    scalar_embedding,history_embedding,node_history_valid
                )
            else:
                fused=scalar_embedding
                history_gate=graph_x.new_zeros((graph_x.shape[0],1))
            node_emb=self.graph_encoder(graph_x,graph_edge_index,graph_edge_attr,fused)
        else:
            node_emb=graph_x.new_zeros((0,self.cfg.temporal.d_model))
            history_gate=graph_x.new_zeros((0,1))
        pooled=self.tracklet_pooler(node_emb,tracklet_id,graph_x[:,0] if graph_x.shape[0] else graph_x.new_zeros((0,)),n_tracklets=M)
        node_memory=TemporalNodeMemory(
            tokens=node_emb,
            observed_ref_um=node_observed_ref_um,
            projected_ref_um=temporal_ref_um[tracklet_id],
            time_offset=node_time_offset,
            tracklet_id=tracklet_id,
            batch_index=node_batch,
            history_valid=node_history_valid,
            node_ids=node_ids,
        )
        dref_h=dref_um[temporal_batch]
        return self.temporal_builder(pooled,temporal_ref_um,temporal_status,hypothesis_edge_index,
                                     hypothesis_edge_attr,temporal_batch,dref_h,
                                     history_support=history_support,
                                     history_support_valid=history_support_valid,
                                     history_support_dt=history_support_dt,
                                     history_support_center_um=history_support_center_um,
                                     history_support_extent_um=history_support_extent_um,
                                     node_history_valid=node_history_valid,
                                     history_gate=history_gate,
                                     best_current_component_id=best_current_component_id,
                                     best_component_overlap=best_component_overlap,
                                     second_best_component_overlap=second_best_component_overlap,
                                     node_memory=node_memory)

    def forward(
        self,
        spatial_inputs: Tensor,
        instance_labels: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        instance_features: Tensor,
        instance_ids: Tensor,
        instance_batch: Tensor,
        instance_centroids_um: Tensor,
        graph_x: Tensor,
        graph_edge_index: Tensor,
        graph_edge_attr: Tensor,
        tracklet_id: Tensor,
        temporal_ref_um: Tensor,
        temporal_status: Tensor,
        hypothesis_edge_index: Tensor,
        hypothesis_edge_attr: Tensor,
        temporal_batch: Tensor,
        spatial_padding_mask: Tensor | None = None,
        return_debug: bool = False,
        bypass_coreasoning: bool = False,
        node_instance_grid: Tensor | None = None,
        node_history_valid: Tensor | None = None,
        history_support: Tensor | None = None,
        history_support_valid: Tensor | None = None,
        history_support_dt: Tensor | None = None,
        history_support_center_um: Tensor | None = None,
        history_support_extent_um: Tensor | None = None,
        best_current_component_id: Tensor | None = None,
        best_component_overlap: Tensor | None = None,
        second_best_component_overlap: Tensor | None = None,
        node_observed_ref_um: Tensor | None = None,
        node_time_offset: Tensor | None = None,
        node_ids: Tensor | None = None,
        temporal_memory_ablation: str = "full",
        detection_graph_ablation: str = "full",
        return_full_temporal_attention: bool = False,
    ) -> StirNetOutput:
        acq=self.acquisition(spacing_um,dref_um)
        pyramid=self.encoder(spatial_inputs,spacing_um,acq,spatial_padding_mask)
        temporal=self._build_temporal(graph_x,graph_edge_index,graph_edge_attr,tracklet_id,
                                      temporal_ref_um,temporal_status,hypothesis_edge_index,
                                      hypothesis_edge_attr,temporal_batch,dref_um,
                                      node_instance_grid,node_history_valid,history_support,
                                      history_support_valid,history_support_dt,
                                      history_support_center_um,history_support_extent_um,
                                      best_current_component_id,best_component_overlap,
                                      second_best_component_overlap,
                                      node_observed_ref_um,node_time_offset,node_ids,
                                      detection_graph_ablation)

        if bypass_coreasoning:
            e3=pyramid.features[3]
        else:
            e3,temporal=self.cr1(pyramid.features[3],pyramid.spacings_um[3],temporal,dref_um,acq,
                                 pyramid.padding_masks[3] if pyramid.padding_masks else None)
        e2=self.decoder.decode_to_e2(e3,pyramid,acq)
        if not bypass_coreasoning:
            e2,temporal=self.cr2(e2,pyramid.spacings_um[2],temporal,dref_um,acq,
                                 pyramid.padding_masks[2] if pyramid.padding_masks else None)
        d1,d0,mask_features=self.decoder.decode_from_e2(e2,pyramid,acq)
        dense=self.dense_heads(d0)

        query_mode = (
            self.cfg.proposals.query_mode
            if self.cfg.proposals.enabled
            else "legacy"
        )
        proposal_state = None
        if query_mode == "spatial_proposals":
            proposal_state, proposal_score_logits = self.spatial_proposal_generator(
                d0,
                e2,
                spatial_inputs,
                dense,
                instance_labels,
                spacing_um,
                pyramid.spacings_um[2],
                dref_um,
                instance_ids,
                instance_batch,
                instance_centroids_um,
                spatial_padding_mask,
            )
        else:
            proposal_score_logits = (
                self.spatial_proposal_generator.proposal_score_logits(
                    d0, spatial_inputs, dense
                )
            )
        dense["proposal_score_logits"] = proposal_score_logits

        qstate=self.query_builder(
            e2,pyramid.spacings_um[2],instance_labels,instance_features,
            instance_ids,instance_batch,instance_centroids_um,dref_um,temporal,
            memory_ablation=temporal_memory_ablation,
            return_debug=return_debug,
            full_attention=return_full_temporal_attention,
            proposal_state=proposal_state,
            query_mode=query_mode,
        )
        initial_query_references=(
            qstate.initial_references_cellscale
            if qstate.initial_references_cellscale is not None
            else qstate.references_cellscale.clone()
        )
        qstate,dec_outputs=self.query_decoder(
            qstate,[e3,e2,d1],
            [pyramid.spacings_um[3],pyramid.spacings_um[2],pyramid.spacings_um[1]],
            instance_labels,dref_um,temporal,
            memory_ablation=temporal_memory_ablation,
            return_debug=return_debug,
            full_attention=return_full_temporal_attention,
        )
        final=dec_outputs[-1]
        native_emb=self.native_mask_head(qstate.embeddings)
        debug=None
        if return_debug:
            gate=temporal.history_gate
            node_valid=temporal.node_history_valid
            support_valid=temporal.history_support_valid
            best_overlap=temporal.best_component_overlap
            same_conf=temporal.edge_attr[:,4] if temporal.edge_attr.shape[-1]>4 else temporal.edge_attr.new_zeros((0,))
            closing=temporal.edge_attr[:,11] if temporal.edge_attr.shape[-1]>11 else temporal.edge_attr.new_zeros((0,))
            def safe_mean(value: Tensor) -> Tensor:
                return value.float().mean() if value.numel() else value.new_zeros((),dtype=torch.float32)
            valid_gate=gate[node_valid] if gate is not None and node_valid is not None else graph_x.new_zeros((0,1))
            invalid_gate=gate[~node_valid] if gate is not None and node_valid is not None else graph_x.new_zeros((0,1))
            same_pairs=same_conf>0
            node_memory=temporal.node_memory
            debug={
                "temporal_salience":temporal.salience.detach(),
                "temporal_reliability":temporal.reliability.detach(),
                "temporal_refs_um":temporal.ref_um.detach(),
                "history_gate":temporal.history_gate.detach() if temporal.history_gate is not None else None,
                "node_history_valid":temporal.node_history_valid.detach() if temporal.node_history_valid is not None else None,
                "history_support_valid":temporal.history_support_valid.detach() if temporal.history_support_valid is not None else None,
                "history_support_dt":temporal.history_support_dt.detach() if temporal.history_support_dt is not None else None,
                "best_current_component_id":temporal.best_current_component_id.detach() if temporal.best_current_component_id is not None else None,
                "best_component_overlap":temporal.best_component_overlap.detach() if temporal.best_component_overlap is not None else None,
                "pairwise_convergence_features":temporal.edge_attr[:,8:15].detach(),
                "same_current_component_confidence":temporal.edge_attr[:,4].detach(),
                "projected_support_overlap":temporal.edge_attr[:,15].detach(),
                "history_support_attention_bias_stats":{
                    "cr1":self.cr1.cross.last_history_bias_stats,
                    "cr2":self.cr2.cross.last_history_bias_stats,
                },
                "history_statistics":{
                    "mean_history_gate":safe_mean(gate if gate is not None else graph_x.new_zeros((0,1))),
                    "mean_valid_history_gate":safe_mean(valid_gate),
                    "mean_invalid_history_gate":safe_mean(invalid_gate),
                    "fraction_past_support":safe_mean(support_valid[:,0] if support_valid is not None else graph_x.new_zeros((0,))),
                    "fraction_future_support":safe_mean(support_valid[:,1] if support_valid is not None else graph_x.new_zeros((0,))),
                    "mean_best_component_overlap":safe_mean(best_overlap if best_overlap is not None else graph_x.new_zeros((0,))),
                    "fraction_same_component_pairs":safe_mean(same_pairs),
                    "mean_closing_speed_same_component":safe_mean(closing[same_pairs]),
                },
                "query_initial_references_cellscale":initial_query_references.detach(),
                "query_layer_references_cellscale":torch.stack(
                    [layer["centers_cellscale"] for layer in dec_outputs], dim=0
                ).detach(),
                "query_references_cellscale":qstate.references_cellscale.detach(),
                "node_memory":None if node_memory is None else {
                    "tokens":node_memory.tokens.detach(),
                    "node_ids":node_memory.node_ids.detach() if node_memory.node_ids is not None else None,
                    "time_offset":node_memory.time_offset.detach(),
                    "tracklet_id":node_memory.tracklet_id.detach(),
                    "batch_index":node_memory.batch_index.detach(),
                    "observed_ref_um":node_memory.observed_ref_um.detach(),
                    "projected_ref_um":node_memory.projected_ref_um.detach(),
                    "history_valid":node_memory.history_valid.detach(),
                },
                "component_temporal_attention":self.query_builder.last_temporal_debug,
                "query_temporal_attention":[
                    layer.last_temporal_debug for layer in self.query_decoder.layers
                ],
                "temporal_memory_ablation":temporal_memory_ablation,
                "detection_graph_ablation":detection_graph_ablation,
                "candidate_detection_edge_count":torch.tensor(
                    graph_edge_index.shape[1],device=graph_x.device
                ),
                "accepted_detection_edge_count":(
                    (graph_edge_attr[:,14]>0.5).sum().detach()
                    if graph_edge_attr.shape[-1]>14 else torch.zeros(
                        (),device=graph_x.device,dtype=torch.long
                    )
                ),
                "proposal_score_logits":proposal_score_logits.detach(),
                "proposal_references_cellscale":(
                    proposal_state.references_cellscale.detach()
                    if proposal_state is not None else None
                ),
                "proposal_scores":(
                    proposal_state.scores.detach()
                    if proposal_state is not None else None
                ),
                "proposal_source_instance_ids":(
                    proposal_state.source_instance_ids.detach()
                    if proposal_state is not None else None
                ),
                "proposal_fallback_mask":(
                    proposal_state.fallback_mask.detach()
                    if proposal_state is not None else None
                ),
                "proposal_padding_mask":(
                    proposal_state.padding_mask.detach()
                    if proposal_state is not None else None
                ),
                "proposal_embeddings":(
                    proposal_state.embeddings.detach()
                    if proposal_state is not None else None
                ),
                "learned_proposal_count":(
                    ((~proposal_state.padding_mask) & ~proposal_state.fallback_mask)
                    .sum(dim=1)
                    .detach()
                    if proposal_state is not None else None
                ),
                "fallback_proposal_count":(
                    (proposal_state.fallback_mask & ~proposal_state.padding_mask)
                    .sum(dim=1)
                    .detach()
                    if proposal_state is not None else None
                ),
                "query_mode":query_mode,
            }
        return StirNetOutput(
            exist_logits=final["exist_logits"],
            centers_cellscale=final["centers_cellscale"],
            coarse_mask_logits=final["coarse_mask_logits"],
            coarse_spacing_um=final["coarse_spacing_um"],
            query_embeddings=qstate.embeddings,
            native_mask_embeddings=native_emb,
            query_types=qstate.query_types,
            query_padding_mask=qstate.padding_mask,
            source_instance_ids=qstate.source_instance_ids,
            query_initial_references_cellscale=initial_query_references,
            temporal_salience=qstate.temporal_salience,
            temporal_reliability=qstate.temporal_reliability,
            aux_outputs=dec_outputs[:-1],
            dense_outputs=dense,
            mask_features=mask_features,
            spacing_um=spacing_um,
            dref_um=dref_um,
            instance_labels=instance_labels,
            debug=debug,
            proposals=proposal_state,
            d0_features=d0,
            spatial_inputs=spatial_inputs,
        )

    def render_masks(self, outputs: StirNetOutput, selected_indices: list[Tensor]) -> list[Tensor]:
        """Render selected queries through proposal-local and legacy branches."""
        if len(selected_indices) != outputs.exist_logits.shape[0]:
            raise ValueError("selected_indices must contain one tensor per batch item")
        use_local = self.cfg.local_masks.enabled
        legacy_indices: list[Tensor] = []
        legacy_destinations: list[Tensor] = []
        local_requests: list[tuple[int, int]] = []
        local_destinations: list[tuple[int, int]] = []
        shape = tuple(int(value) for value in outputs.instance_labels.shape[-3:])
        rendered = [
            outputs.mask_features.new_full(
                (len(indices), *shape),
                float(self.cfg.queries.native_background_logit),
            )
            for indices in selected_indices
        ]
        for batch_index, indices in enumerate(selected_indices):
            selected_types = outputs.query_types[batch_index, indices]
            local = (
                selected_types == QUERY_SPATIAL_PROPOSAL
                if use_local
                else torch.zeros_like(selected_types, dtype=torch.bool)
            )
            legacy_indices.append(indices[~local])
            legacy_destinations.append(
                torch.nonzero(~local, as_tuple=False).flatten()
            )
            for destination, query_index in zip(
                torch.nonzero(local, as_tuple=False).flatten().tolist(),
                indices[local].tolist(),
            ):
                local_requests.append((batch_index, int(query_index)))
                local_destinations.append((batch_index, int(destination)))

        legacy = render_native_masks(
            outputs.mask_features,outputs.native_mask_embeddings,legacy_indices,
            outputs.query_types,outputs.source_instance_ids,outputs.centers_cellscale,
            outputs.instance_labels,outputs.spacing_um,outputs.dref_um,
            prior_inside_logit=self.cfg.queries.prior_inside_logit,
            prior_outside_logit=self.cfg.queries.prior_outside_logit,
            temporal_sigma_dref=self.cfg.queries.temporal_gaussian_sigma_dref,
            native_support_radius_dref=self.cfg.queries.native_support_radius_dref,
            native_source_dilation_dref=self.cfg.queries.native_source_dilation_dref,
            native_background_logit=self.cfg.queries.native_background_logit,
            proposal_native_support_radius_dref=(
                self.cfg.proposals.native_support_radius_dref
            ),
        )
        for batch_index, destinations in enumerate(legacy_destinations):
            if destinations.numel():
                rendered[batch_index][destinations] = legacy[batch_index]

        if local_requests:
            if outputs.d0_features is None or outputs.spatial_inputs is None:
                raise ValueError(
                    "spatial-proposal rendering requires d0_features and spatial_inputs"
                )
            local_predictions = self.local_mask_decoder.decode_requests(
                outputs.d0_features,
                outputs.spatial_inputs,
                outputs.dense_outputs,
                outputs.query_embeddings,
                outputs.query_initial_references_cellscale,
                outputs.spacing_um,
                outputs.dref_um,
                local_requests,
            )
            for (batch_index, destination), prediction in zip(
                local_destinations, local_predictions
            ):
                if prediction.logits is None:
                    raise RuntimeError("local mask decoder returned no logits")
                rendered[batch_index][destination][prediction.slices] = (
                    prediction.logits.to(rendered[batch_index].dtype)
                )
        return rendered
