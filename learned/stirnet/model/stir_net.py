from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .geometry.decoder import DenseGeometryDecoder
from .instances.tokenizer import (
    InstanceTokenizer,
    centers_from_labels,
    centers_from_partition_statistics,
)
from .geometry.derived import build_geometry_derived_cache
from .partition.graph_net import SpatialRAGNetwork
from .partition.local_update import LocalPartitionUpdater
from .partition.partitioner import GraphPartitioner
from .partition.rag import RAGBuilder
from .partition.statistics import build_supervoxel_statistics
from .partition.watershed import LearnedGeometryWatershed
from .refinement.local_refiner import LocalGeometryRefiner
from .refinement.requests import (
    build_refinement_requests,
    select_refinement_requests,
)
from .spatial.acquisition import AcquisitionEmbedding
from .spatial.backbone import AnisotropyAwareSpatialBackbone
from .spatial.evidence_stem import EvidenceFusionStem
from .temporal.fusion import InstanceTemporalReasoner
from .temporal.graph_encoder import TemporalGraphEncoder
from .temporal.history import HistoricalInstanceEncoder
from .temporal.observer import TemporalSpatialObserver
from .utils.physical import (
    canonical_resample_spec,
    resample_continuous_volume,
    resample_labels_volume,
)


def _profile_stage(stage_profiler, name: str):
    return (
        nullcontext()
        if stage_profiler is None
        else stage_profiler.profile(name)
    )
from .types import (
    GeometryState,
    GeometryDerivedCache,
    GeometryLike,
    GeometryForwardOutput,
    InstanceState,
    PartitionState,
    RAGState,
    ReasoningState,
    RefinementState,
    RefinementRequest,
    SpatialObservationCache,
    SpatialForwardOutput,
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
        self.local_partition_updater = LocalPartitionUpdater(
            self.watershed, self.cfg.refinement
        )
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

    def _spatial_rag(
        self,
        geometry: GeometryLike,
        decoded,
        spatial_inputs: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        spatial_padding_mask: Tensor | None,
        stage_profiler=None,
        profile_prefix: str = "initial",
    ) -> tuple[RAGState, PartitionState]:
        with _profile_stage(
            stage_profiler, f"{profile_prefix}_geometry_probability_prepare"
        ):
            derived_cache = build_geometry_derived_cache(
                geometry,
                self.cfg.partition,
                padding_mask=spatial_padding_mask,
                stage_profiler=stage_profiler,
                profile_prefix=f"{profile_prefix}_watershed",
            )
        with _profile_stage(stage_profiler, f"{profile_prefix}_watershed"):
            supervoxels = self.watershed(
                geometry,
                spacing_um,
                dref_um,
                spatial_padding_mask,
                derived_cache=derived_cache,
                stage_profiler=stage_profiler,
                profile_prefix=f"{profile_prefix}_watershed",
            )
        return self._rag_from_supervoxels(
            supervoxels,
            geometry,
            decoded,
            spatial_inputs,
            spacing_um,
            dref_um,
            stage_profiler=stage_profiler,
            profile_prefix=profile_prefix,
            derived_cache=derived_cache,
        )

    def _rag_from_supervoxels(
        self,
        supervoxels: list[Tensor],
        geometry: GeometryLike,
        decoded,
        spatial_inputs: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        stage_profiler=None,
        profile_prefix: str = "initial",
        derived_cache: GeometryDerivedCache | None = None,
    ) -> tuple[RAGState, PartitionState]:
        with _profile_stage(stage_profiler, f"{profile_prefix}_region_stats"):
            statistics = build_supervoxel_statistics(
                supervoxels,
                spatial_inputs,
                geometry,
                spacing_um,
                (decoded.d0, decoded.d1, decoded.d2),
                derived=derived_cache,
                stage_profiler=stage_profiler,
            )
        with _profile_stage(stage_profiler, f"{profile_prefix}_rag_build"):
            rag = self.rag_builder(
                supervoxels,
                decoded.d0,
                spatial_inputs,
                geometry,
                spacing_um,
                dref_um,
                statistics_by_batch=statistics,
                derived_cache=derived_cache,
                stage_profiler=stage_profiler,
                profile_prefix=f"{profile_prefix}_rag",
            )
        with _profile_stage(stage_profiler, f"{profile_prefix}_rag_network"):
            rag = self.rag_network(rag)
        partition = self.partitioner(
            rag,
            rag.spatial_edge_logits,
            self.cfg.partition.spatial_merge_threshold,
        )
        return rag, partition

    def _observe_temporal(
        self,
        temporal_base: TemporalState,
        decoded,
        geometry: GeometryLike,
        pyramid,
        spacing_um: Tensor,
        dref_um: Tensor,
        cache: SpatialObservationCache | None = None,
    ) -> TemporalState:
        return self.temporal_observer(
            temporal_base,
            decoded,
            geometry,
            pyramid.spacings_um,
            spacing_um,
            dref_um,
            cache=cache,
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
        execution_stage: Literal["geometry", "spatial", "temporal", "refinement"] | None = None,
        teacher_request_builder: Callable[
            [InstanceState, RAGState, TemporalState, ReasoningState, Tensor],
            list[RefinementRequest],
        ]
        | None = None,
        apply_existence_filter: bool | None = None,
        return_debug: bool = False,
        precomputed_geometry: GeometryForwardOutput | None = None,
        stage_profiler=None,
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
    ) -> GeometryForwardOutput | SpatialForwardOutput | StirNetOutput:
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

        if execution_stage is None:
            use_refinement = (
                self.cfg.refinement.enabled
                if run_refinement is None
                else bool(run_refinement)
            )
            execution_stage = "refinement" if use_refinement else "temporal"
        elif run_refinement is not None:
            raise ValueError(
                "execution_stage is explicit; do not also pass run_refinement"
            )
        if execution_stage not in {"geometry", "spatial", "temporal", "refinement"}:
            raise ValueError(f"Unknown STIR-Net execution stage: {execution_stage}")

        if precomputed_geometry is None:
            network_inputs = spatial_inputs
            network_spacing_um = spacing_um
            network_padding_mask = spatial_padding_mask
            canonical_spacing = self.cfg.spatial.canonical_spacing_um
            if canonical_spacing is not None:
                target_shape, network_spacing_um = canonical_resample_spec(
                    tuple(spatial_inputs.shape[-3:]),
                    spacing_um,
                    canonical_spacing,
                )
                network_inputs = resample_continuous_volume(
                    spatial_inputs, target_shape
                )
                if spatial_padding_mask is not None:
                    network_padding_mask = resample_labels_volume(
                        spatial_padding_mask.long(), target_shape
                    ).bool()
            acquisition = self.acquisition(network_spacing_um, dref_um)
            with _profile_stage(stage_profiler, "evidence_stem"):
                stem = self.evidence_stem(
                    network_inputs, acquisition, prior_keep_mask=prior_keep_mask
                )
            pyramid, decoded = self.spatial_backbone(
                stem,
                network_spacing_um,
                acquisition,
                network_padding_mask,
                stage_profiler=stage_profiler,
            )
            initial_geometry = self.geometry_decoder(
                decoded.d0, acquisition, stage_profiler=stage_profiler
            )
            initial_geometry = replace(
                initial_geometry,
                feature_spacing_um=network_spacing_um,
            )
            native_shape = tuple(spatial_inputs.shape[-3:])
            if tuple(initial_geometry.sdf.shape[-3:]) != native_shape:
                initial_geometry = replace(
                    initial_geometry,
                    foreground_logits=resample_continuous_volume(
                        initial_geometry.foreground_logits, native_shape
                    ),
                    surface_logits=resample_continuous_volume(
                        initial_geometry.surface_logits, native_shape
                    ),
                    separator_logits=resample_continuous_volume(
                        initial_geometry.separator_logits, native_shape
                    ),
                    sdf=resample_continuous_volume(initial_geometry.sdf, native_shape),
                    flow=resample_continuous_volume(initial_geometry.flow, native_shape),
                    centroid_offset=resample_continuous_volume(
                        initial_geometry.centroid_offset, native_shape
                    ),
                    seed_logits=resample_continuous_volume(
                        initial_geometry.seed_logits, native_shape
                    ),
                )
        else:
            initial_geometry = precomputed_geometry.geometry
            pyramid = precomputed_geometry.spatial_pyramid
            decoded = precomputed_geometry.decoded_spatial
            if decoded.d0.shape[0] != spatial_inputs.shape[0]:
                raise ValueError("precomputed geometry batch does not match spatial inputs")
            if initial_geometry.sdf.shape[-3:] != spatial_inputs.shape[-3:]:
                raise ValueError("precomputed explicit geometry shape does not match inputs")

        if execution_stage == "geometry":
            return GeometryForwardOutput(
                geometry=initial_geometry,
                spatial_pyramid=pyramid,
                decoded_spatial=decoded,
            )

        initial_rag, initial_partition = self._spatial_rag(
            initial_geometry,
            decoded,
            spatial_inputs,
            spacing_um,
            dref_um,
            spatial_padding_mask,
            stage_profiler=stage_profiler,
            profile_prefix="initial",
        )
        if execution_stage == "spatial":
            return SpatialForwardOutput(
                geometry=initial_geometry,
                spatial_pyramid=pyramid,
                decoded_spatial=decoded,
                rag=initial_rag,
                spatial_partition=initial_partition,
            )

        with _profile_stage(stage_profiler, "instance_tokenizer"):
            initial_instances = self.instance_tokenizer(
                initial_partition,
                initial_rag,
                decoded,
                initial_geometry,
                spacing_um,
                dref_um,
                stage_profiler=stage_profiler,
                profile_prefix="initial_tokenizer",
            )
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
        with _profile_stage(stage_profiler, "temporal_encoder"):
            temporal_base = self.temporal_encoder(temporal_data)
        with _profile_stage(stage_profiler, "temporal_observer"):
            observation_cache = self.temporal_observer.build_cache(
                temporal_base,
                decoded,
                initial_geometry,
                pyramid.spacings_um,
                spacing_um,
                dref_um,
            )
            initial_geometry = replace(initial_geometry, features=None)
            temporal = self._observe_temporal(
                temporal_base,
                decoded,
                initial_geometry,
                pyramid,
                spacing_um,
                dref_um,
                observation_cache,
            )
        with _profile_stage(stage_profiler, "temporal_reasoning"):
            initial_reasoning = self.instance_temporal(
                initial_instances, initial_rag, temporal, dref_um
            )

        use_refinement = (
            execution_stage == "refinement" and self.cfg.refinement.enabled
        )
        refinement: RefinementState | None = None
        geometry = initial_geometry
        rag = initial_rag
        spatial_partition = initial_partition
        instances = initial_instances
        reasoning = initial_reasoning

        if use_refinement:
            model_requests = build_refinement_requests(
                initial_instances,
                initial_rag,
                temporal,
                initial_reasoning,
                dref_um,
                self.cfg.refinement,
                select=teacher_request_builder is None,
            )
            teacher_requests: list[RefinementRequest] = []
            if teacher_request_builder is not None:
                teacher_requests = teacher_request_builder(
                    initial_instances,
                    initial_rag,
                    temporal,
                    initial_reasoning,
                    dref_um,
                )
                requests = select_refinement_requests(
                    [*teacher_requests, *model_requests],
                    dref_um,
                    self.cfg.refinement,
                )
            else:
                requests = model_requests
            with _profile_stage(stage_profiler, "local_refinement"):
                refinement = self.local_refiner(
                    decoded.d0,
                    spatial_inputs,
                    initial_geometry,
                    spacing_um,
                    dref_um,
                    requests,
                )
            refinement = replace(
                refinement,
                model_request_count=sum(
                    request.selection_source == "model" for request in requests
                ),
                teacher_request_count=sum(
                    request.selection_source == "teacher" for request in requests
                ),
            )
            if refinement.applied_count:
                geometry = refinement.geometry
                if self.cfg.refinement.partition_update == "local":
                    with _profile_stage(stage_profiler, "refined_partition_update"):
                        local_update = self.local_partition_updater(
                            initial_rag.supervoxel_labels,
                            geometry,
                            spacing_um,
                            dref_um,
                            spatial_padding_mask,
                            requests=refinement.requests,
                            rag=initial_rag,
                            instances=initial_instances,
                        )
                    if local_update.used_fallback:
                        rag, spatial_partition = self._rag_from_supervoxels(
                            local_update.supervoxel_labels,
                            geometry,
                            decoded,
                            spatial_inputs,
                            spacing_um,
                            dref_um,
                            stage_profiler=stage_profiler,
                            profile_prefix="refined",
                        )
                    else:
                        with _profile_stage(
                            stage_profiler, "refined_rag_local_update"
                        ):
                            rag = self.rag_builder.update_local(
                                initial_rag,
                                local_update.supervoxel_labels,
                                local_update.updated_boxes or [],
                                decoded,
                                spatial_inputs,
                                geometry,
                                spacing_um,
                                dref_um,
                            )
                        with _profile_stage(
                            stage_profiler, "refined_rag_network"
                        ):
                            rag = self.rag_network(rag)
                        spatial_partition = self.partitioner(
                            rag,
                            rag.spatial_edge_logits,
                            self.cfg.partition.spatial_merge_threshold,
                        )
                    refinement = replace(
                        refinement,
                        partition_update="local",
                        partition_fallback=local_update.used_fallback,
                        partition_fallback_reason=local_update.fallback_reason,
                        partition_fallback_reason_code=(
                            local_update.fallback_reason_code
                        ),
                        partition_fallback_batch_index=(
                            local_update.fallback_batch_index
                        ),
                        partition_fallback_box_index=(
                            local_update.fallback_box_index
                        ),
                        partition_fallback_box_shape_zyx=(
                            local_update.fallback_box_shape_zyx
                        ),
                        partition_fallback_box_voxel_count=(
                            local_update.fallback_box_voxel_count
                        ),
                        partition_fallback_core_voxel_count=(
                            local_update.fallback_core_voxel_count
                        ),
                        partition_fallback_local_component_count=(
                            local_update.fallback_local_component_count
                        ),
                        partition_fallback_old_core_label_count=(
                            local_update.fallback_old_core_label_count
                        ),
                        partition_fallback_old_shell_label_count=(
                            local_update.fallback_old_shell_label_count
                        ),
                        partition_fallback_conflicting_old_label_ids=(
                            local_update.fallback_conflicting_old_label_ids or []
                        ),
                        local_update_box_count=local_update.updated_box_count,
                        local_update_voxel_fraction=(
                            local_update.updated_voxel_count
                            / max(sum(labels.numel() for labels in initial_rag.supervoxel_labels), 1)
                        ),
                    )
                else:
                    with _profile_stage(
                        stage_profiler, "refined_partition_update"
                    ):
                        rag, spatial_partition = self._spatial_rag(
                            geometry,
                            decoded,
                            spatial_inputs,
                            spacing_um,
                            dref_um,
                            spatial_padding_mask,
                            stage_profiler=stage_profiler,
                            profile_prefix="refined",
                        )
                    refinement = replace(
                        refinement,
                        partition_update="full",
                    )
                with _profile_stage(stage_profiler, "refined_tokenizer"):
                    instances = self.instance_tokenizer(
                        spatial_partition,
                        rag,
                        decoded,
                        geometry,
                        spacing_um,
                        dref_um,
                        stage_profiler=stage_profiler,
                        profile_prefix="refined_tokenizer",
                    )
                with _profile_stage(
                    stage_profiler, "refined_temporal_observer"
                ):
                    temporal = self._observe_temporal(
                        temporal_base,
                        decoded,
                        geometry,
                        pyramid,
                        spacing_um,
                        dref_um,
                        observation_cache,
                    )
                with _profile_stage(
                    stage_profiler, "refined_temporal_reasoning"
                ):
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

        if rag.statistics is not None:
            all_centers = centers_from_partition_statistics(final_partition, rag)
            centers = (
                [
                    batch_centers[(scores >= self.cfg.instances.exist_threshold)]
                    for batch_centers, scores in zip(all_centers, existence_scores)
                ]
                if use_exist
                else all_centers
            )
        else:
            centers = centers_from_labels(final_labels, spacing_um, geometry.sdf)
        debug: dict[str, Any] | None = None
        if return_debug:
            debug = {
                "research_design": {
                    "dense_geometry": "Omnipose + NucMM + NISNet3D",
                    "supervoxel_partition": "PlantSeg-style learned geometry + RAG",
                    "temporal_fusion": "spatial-read-only then gated RAG/ROI residuals",
                },
                "watershed_backend": self.cfg.partition.watershed_backend,
                "region_stats_backend": "torch",
                "post_statistics_full_geometry_scan_count": 0,
                "refined_full_geometry_scan_count": int(
                    refinement is not None and refinement.partition_fallback
                ),
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
                "refinement_partition_update": (
                    "none" if refinement is None else refinement.partition_update
                ),
                "refinement_partition_fallback": bool(
                    refinement is not None and refinement.partition_fallback
                ),
                "refinement_partition_fallback_reason": (
                    "" if refinement is None else refinement.partition_fallback_reason
                ),
                "local_update_box_count": (
                    0 if refinement is None else refinement.local_update_box_count
                ),
                "local_update_voxel_fraction": (
                    0.0 if refinement is None else refinement.local_update_voxel_fraction
                ),
                "local_update_fallback_count": int(
                    refinement is not None and refinement.partition_fallback
                ),
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
