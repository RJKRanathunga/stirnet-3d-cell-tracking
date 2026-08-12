from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import StirNetConfig
from .coreasoning import CoReasoningBlock
from .graph_encoder import DetectionGraphEncoder
from .heads import DenseAuxiliaryHeads, MaskEmbeddingHead, render_native_masks
from .query_builder import InstanceQueryBuilder
from .query_decoder import InstanceQueryDecoder
from .spacing import AcquisitionEmbedding
from .spatial_decoder import SpatialDecoder
from .spatial_encoder import SpatialEncoder
from .temporal_hypotheses import TemporalStateBuilder, TrackletPooler
from .types import StirNetOutput, TemporalState


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
        self.tracklet_pooler = TrackletPooler(self.cfg.temporal.d_model)
        self.temporal_builder = TemporalStateBuilder(self.cfg.temporal)
        self.cr1 = CoReasoningBlock(
            c3, self.cfg.spatial, self.cfg.temporal, self.cfg.coreasoning,
            activation_checkpointing=checkpoint_coreasoning,
        )
        self.cr2 = CoReasoningBlock(
            c2, self.cfg.spatial, self.cfg.temporal, self.cfg.coreasoning,
            activation_checkpointing=checkpoint_coreasoning,
        )
        self.query_builder = InstanceQueryBuilder(self.cfg.queries, feature_channels=c2)
        self.query_decoder = InstanceQueryDecoder((c3,c2,c1),self.cfg.decoder,self.cfg.queries)
        self.native_mask_head = MaskEmbeddingHead(self.cfg.decoder.d_model,self.cfg.spatial.mask_dim)
        self.dense_heads = DenseAuxiliaryHeads(c0)

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
            ("coreasoning", self.cfg.coreasoning.d_model, self.cfg.coreasoning.heads),
            ("decoder", self.cfg.decoder.d_model, self.cfg.decoder.heads),
        ):
            if heads <= 0 or d_model % heads:
                raise ValueError(f"{name} d_model={d_model} must be divisible by heads={heads}")

    def _build_temporal(
        self, graph_x: Tensor, graph_edge_index: Tensor, graph_edge_attr: Tensor,
        tracklet_id: Tensor, temporal_ref_um: Tensor, temporal_status: Tensor,
        hypothesis_edge_index: Tensor, hypothesis_edge_attr: Tensor,
        temporal_batch: Tensor, dref_um: Tensor,
    ) -> TemporalState:
        M=temporal_ref_um.shape[0]
        if M==0:
            device=graph_x.device
            tokens=graph_x.new_zeros((0,self.cfg.temporal.d_model))
            z1=graph_x.new_zeros((0,1))
            return TemporalState(tokens,temporal_ref_um,temporal_ref_um,z1,z1,temporal_status,
                                 hypothesis_edge_index,hypothesis_edge_attr,temporal_batch)
        node_emb=self.graph_encoder(graph_x,graph_edge_index,graph_edge_attr) if graph_x.shape[0] else graph_x.new_zeros((0,self.cfg.temporal.d_model))
        pooled=self.tracklet_pooler(node_emb,tracklet_id,graph_x[:,0] if graph_x.shape[0] else graph_x.new_zeros((0,)),n_tracklets=M)
        dref_h=dref_um[temporal_batch]
        return self.temporal_builder(pooled,temporal_ref_um,temporal_status,hypothesis_edge_index,
                                     hypothesis_edge_attr,temporal_batch,dref_h)

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
    ) -> StirNetOutput:
        acq=self.acquisition(spacing_um,dref_um)
        pyramid=self.encoder(spatial_inputs,spacing_um,acq,spatial_padding_mask)
        temporal=self._build_temporal(graph_x,graph_edge_index,graph_edge_attr,tracklet_id,
                                      temporal_ref_um,temporal_status,hypothesis_edge_index,
                                      hypothesis_edge_attr,temporal_batch,dref_um)

        e3,temporal=self.cr1(pyramid.features[3],pyramid.spacings_um[3],temporal,dref_um,acq,
                             pyramid.padding_masks[3] if pyramid.padding_masks else None)
        e2=self.decoder.decode_to_e2(e3,pyramid,acq)
        e2,temporal=self.cr2(e2,pyramid.spacings_um[2],temporal,dref_um,acq,
                             pyramid.padding_masks[2] if pyramid.padding_masks else None)
        d1,d0,mask_features=self.decoder.decode_from_e2(e2,pyramid,acq)
        dense=self.dense_heads(d0)

        qstate=self.query_builder(e2,pyramid.spacings_um[2],instance_labels,instance_features,
                                  instance_ids,instance_batch,instance_centroids_um,dref_um,temporal)
        initial_query_references=qstate.references_cellscale
        qstate,dec_outputs=self.query_decoder(qstate,[e3,e2,d1],
                                              [pyramid.spacings_um[3],pyramid.spacings_um[2],pyramid.spacings_um[1]],
                                              instance_labels,dref_um)
        final=dec_outputs[-1]
        native_emb=self.native_mask_head(qstate.embeddings)
        debug=None
        if return_debug:
            debug={
                "temporal_salience":temporal.salience.detach(),
                "temporal_reliability":temporal.reliability.detach(),
                "temporal_refs_um":temporal.ref_um.detach(),
                "query_initial_references_cellscale":initial_query_references.detach(),
                "query_layer_references_cellscale":torch.stack(
                    [layer["centers_cellscale"] for layer in dec_outputs], dim=0
                ).detach(),
                "query_references_cellscale":qstate.references_cellscale.detach(),
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
            temporal_salience=qstate.temporal_salience,
            temporal_reliability=qstate.temporal_reliability,
            aux_outputs=dec_outputs[:-1],
            dense_outputs=dense,
            mask_features=mask_features,
            spacing_um=spacing_um,
            dref_um=dref_um,
            instance_labels=instance_labels,
            debug=debug,
        )

    def render_masks(self, outputs: StirNetOutput, selected_indices: list[Tensor]) -> list[Tensor]:
        return render_native_masks(
            outputs.mask_features,outputs.native_mask_embeddings,selected_indices,
            outputs.query_types,outputs.source_instance_ids,outputs.centers_cellscale,
            outputs.instance_labels,outputs.spacing_um,outputs.dref_um,
            prior_inside_logit=self.cfg.queries.prior_inside_logit,
            prior_outside_logit=self.cfg.queries.prior_outside_logit,
            temporal_sigma_dref=self.cfg.queries.temporal_gaussian_sigma_dref,
            native_support_radius_dref=self.cfg.queries.native_support_radius_dref,
            native_source_dilation_dref=self.cfg.queries.native_source_dilation_dref,
            native_background_logit=self.cfg.queries.native_background_logit,
        )
