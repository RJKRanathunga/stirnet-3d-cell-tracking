from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Iterable, List

import torch
from torch import Tensor
import torch.nn.functional as F

from ..model.config import InferenceConfig
from ..model.types import (
    GeometryForwardOutput,
    GeometryState,
    InstanceState,
    PartitionState,
    RAGState,
    SpatialDecodeState,
    SpatialObservationCache,
    TemporalInput,
    TemporalState,
    ReasoningState,
    RefinementState,
    geometry_field,
)
from ..model.instances.tokenizer import centers_from_labels
from ..model.refinement.requests import build_refinement_requests


@dataclass(frozen=True)
class DenseTileSpec:
    batch_index: int
    slices_zyx: tuple[slice, slice, slice]


@dataclass(frozen=True)
class TiledDenseResult:
    geometry: GeometryState
    blend_weight_sum: Tensor
    tile_count: int


@dataclass(frozen=True)
class StreamedLabelFeatureStats:
    """Per-label raw mean/max feature statistics from a bounded second pass."""

    pooled_scales: tuple[List[Tensor], List[Tensor], List[Tensor]]
    counts_scales: tuple[List[Tensor], List[Tensor], List[Tensor]]


@dataclass(frozen=True)
class TiledSpatialResult:
    dense: TiledDenseResult
    supervoxel_labels: List[Tensor]
    feature_stats: StreamedLabelFeatureStats
    rag: RAGState
    spatial_partition: PartitionState
    provisional_instances: InstanceState


@dataclass(frozen=True)
class TiledTemporalResult:
    spatial: TiledSpatialResult
    geometry: object
    rag: RAGState
    spatial_partition: PartitionState
    provisional_instances: InstanceState
    temporal: TemporalState
    reasoning: ReasoningState
    final_partition: PartitionState
    final_labels: List[Tensor]
    centers_um: List[Tensor]
    refinement: RefinementState | None


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    tile = min(tile, length)
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def generate_dense_tiles(
    batch_size: int,
    shape: tuple[int, int, int],
    config: InferenceConfig,
) -> list[DenseTileSpec]:
    axes = [
        _tile_starts(length, tile, overlap)
        for length, tile, overlap in zip(
            shape, config.tile_shape_zyx, config.tile_overlap_zyx
        )
    ]
    tile_shape = tuple(min(size, length) for size, length in zip(config.tile_shape_zyx, shape))
    return [
        DenseTileSpec(
            batch_index=batch_index,
            slices_zyx=tuple(
                slice(start, start + tile_shape[axis])
                for axis, start in enumerate(starts)
            ),
        )
        for batch_index in range(batch_size)
        for starts in product(*axes)
    ]


def _axis_blend_weight(
    length: int,
    halo: int,
    *,
    touches_low: bool,
    touches_high: bool,
    device: torch.device,
) -> Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    halo = min(halo, max((length - 1) // 2, 0))
    if halo and not touches_low:
        weight[:halo] = torch.linspace(
            1.0 / (halo + 1), halo / (halo + 1), halo, device=device
        )
    if halo and not touches_high:
        weight[-halo:] = torch.linspace(
            halo / (halo + 1), 1.0 / (halo + 1), halo, device=device
        )
    return weight


def tile_blend_weight(
    spec: DenseTileSpec,
    volume_shape: tuple[int, int, int],
    halo_zyx: tuple[int, int, int],
    *,
    device: torch.device,
) -> Tensor:
    axes = []
    for axis, (axis_slice, full, halo) in enumerate(
        zip(spec.slices_zyx, volume_shape, halo_zyx)
    ):
        axes.append(
            _axis_blend_weight(
                int(axis_slice.stop) - int(axis_slice.start),
                halo,
                touches_low=int(axis_slice.start) == 0,
                touches_high=int(axis_slice.stop) == full,
                device=device,
            )
        )
    return (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )


def _pack_geometry(geometry: GeometryState) -> Tensor:
    return torch.cat(
        [
            geometry_field(geometry, "foreground_logits"),
            geometry_field(geometry, "surface_logits"),
            geometry_field(geometry, "separator_logits"),
            geometry_field(geometry, "sdf"),
            geometry_field(geometry, "flow"),
            geometry_field(geometry, "centroid_offset"),
            geometry_field(geometry, "seed_logits"),
        ],
        dim=1,
    )


def _unpack_geometry(packed: Tensor) -> GeometryState:
    return GeometryState(
        foreground_logits=packed[:, 0:1],
        surface_logits=packed[:, 1:2],
        separator_logits=packed[:, 2:3],
        sdf=packed[:, 3:4],
        flow=packed[:, 4:7],
        centroid_offset=packed[:, 7:10],
        seed_logits=packed[:, 10:11],
        features=None,
    )


def _chunks(rows: list[DenseTileSpec], size: int) -> Iterable[list[DenseTileSpec]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@torch.inference_mode()
def tiled_dense_geometry(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    config: InferenceConfig | None = None,
    spatial_padding_mask: Tensor | None = None,
) -> TiledDenseResult:
    """Blend continuous dense geometry tiles into one global prediction."""
    config = config or model.cfg.inference
    shape = tuple(spatial_inputs.shape[-3:])
    specs = generate_dense_tiles(spatial_inputs.shape[0], shape, config)
    accumulator = torch.zeros(
        (spatial_inputs.shape[0], 11, *shape),
        device=spatial_inputs.device,
        dtype=torch.float32,
    )
    weight_sum = torch.zeros(
        (spatial_inputs.shape[0], 1, *shape),
        device=spatial_inputs.device,
        dtype=torch.float32,
    )
    output_dtype = spatial_inputs.dtype
    for group in _chunks(specs, config.tile_batch_size):
        tiles = torch.cat(
            [
                spatial_inputs[
                    spec.batch_index : spec.batch_index + 1,
                    :,
                    spec.slices_zyx[0],
                    spec.slices_zyx[1],
                    spec.slices_zyx[2],
                ]
                for spec in group
            ],
            dim=0,
        )
        tile_spacing = torch.stack([spacing_um[spec.batch_index] for spec in group])
        tile_dref = torch.stack([dref_um[spec.batch_index] for spec in group])
        tile_padding = None
        if spatial_padding_mask is not None:
            tile_padding = torch.cat(
                [
                    spatial_padding_mask[
                        spec.batch_index : spec.batch_index + 1,
                        spec.slices_zyx[0],
                        spec.slices_zyx[1],
                        spec.slices_zyx[2],
                    ]
                    for spec in group
                ]
            )
        output = model(
            tiles,
            tile_spacing,
            tile_dref,
            spatial_padding_mask=tile_padding,
            execution_stage="geometry",
        )
        if not isinstance(output, GeometryForwardOutput):
            raise TypeError("geometry-stage tiled forward returned an invalid output")
        packed_native = _pack_geometry(output.geometry)
        output_dtype = packed_native.dtype
        packed = packed_native.float()
        for row, spec in enumerate(group):
            weight = tile_blend_weight(
                spec,
                shape,
                config.tile_halo_zyx,
                device=spatial_inputs.device,
            )
            target = (
                spec.batch_index,
                slice(None),
                spec.slices_zyx[0],
                spec.slices_zyx[1],
                spec.slices_zyx[2],
            )
            accumulator[target] += packed[row] * weight[None]
            weight_sum[target] += weight[None]
    packed_global = (accumulator / weight_sum.clamp_min(1e-8)).to(output_dtype)
    return TiledDenseResult(
        geometry=_unpack_geometry(packed_global),
        blend_weight_sum=weight_sum,
        tile_count=len(specs),
    )


@torch.inference_mode()
def stream_tiled_label_feature_stats(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    labels: List[Tensor],
    blend_weight_sum: Tensor,
    *,
    config: InferenceConfig | None = None,
) -> StreamedLabelFeatureStats:
    """Second pass: reduce D0/D1/D2 directly into global label rows."""
    config = config or model.cfg.inference
    shape = tuple(spatial_inputs.shape[-3:])
    specs = generate_dense_tiles(spatial_inputs.shape[0], shape, config)
    sums: list[list[Tensor | None]] = [[None] * len(labels) for _ in range(3)]
    maxima: list[list[Tensor | None]] = [[None] * len(labels) for _ in range(3)]
    counts: list[list[Tensor | None]] = [[None] * len(labels) for _ in range(3)]

    for spec in specs:
        b = spec.batch_index
        tile = spatial_inputs[
            b : b + 1,
            :,
            spec.slices_zyx[0],
            spec.slices_zyx[1],
            spec.slices_zyx[2],
        ]
        output = model(
            tile,
            spacing_um[b : b + 1],
            dref_um[b : b + 1],
            execution_stage="geometry",
        )
        weight = tile_blend_weight(
            spec, shape, config.tile_halo_zyx, device=tile.device
        )
        normalized_weight = weight / blend_weight_sum[
            b,
            0,
            spec.slices_zyx[0],
            spec.slices_zyx[1],
            spec.slices_zyx[2],
        ].clamp_min(1e-8)
        label_crop = labels[b][spec.slices_zyx]
        for scale, feature in enumerate(
            (
                output.decoded_spatial.d0[0],
                output.decoded_spatial.d1[0],
                output.decoded_spatial.d2[0],
            )
        ):
            scaled_labels = F.interpolate(
                label_crop.float()[None, None],
                size=feature.shape[-3:],
                mode="nearest",
            )[0, 0].long()
            scaled_weight = F.interpolate(
                normalized_weight[None, None],
                size=feature.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            max_id = int(labels[b].max().item())
            if sums[scale][b] is None:
                sums[scale][b] = feature.new_zeros((max_id, feature.shape[0]))
                maxima[scale][b] = feature.new_full(
                    (max_id, feature.shape[0]), -torch.inf
                )
                counts[scale][b] = feature.new_zeros((max_id,))
            valid = scaled_labels.reshape(-1) > 0
            ids = scaled_labels.reshape(-1)[valid] - 1
            values = feature.flatten(1).transpose(0, 1)[valid]
            weights = scaled_weight.reshape(-1)[valid].to(values.dtype)
            sums[scale][b].index_add_(0, ids, values * weights[:, None])
            counts[scale][b].index_add_(0, ids, weights)
            maxima[scale][b].scatter_reduce_(
                0,
                ids[:, None].expand_as(values),
                values,
                reduce="amax",
                include_self=True,
            )

    pooled_scales: list[List[Tensor]] = [[], [], []]
    count_scales: list[List[Tensor]] = [[], [], []]
    for scale in range(3):
        for b, label in enumerate(labels):
            if sums[scale][b] is None:
                channels = model.cfg.spatial.channels[scale]
                pooled_scales[scale].append(
                    spatial_inputs.new_zeros((0, 2 * channels))
                )
                count_scales[scale].append(spatial_inputs.new_zeros((0,)))
                continue
            count = counts[scale][b]
            mean = sums[scale][b] / count.clamp_min(1)[:, None]
            maximum = torch.where(
                torch.isfinite(maxima[scale][b]),
                maxima[scale][b],
                torch.zeros_like(maxima[scale][b]),
            )
            pooled_scales[scale].append(torch.cat([mean, maximum], dim=-1))
            count_scales[scale].append(count)
    return StreamedLabelFeatureStats(
        pooled_scales=tuple(pooled_scales),
        counts_scales=tuple(count_scales),
    )


@torch.inference_mode()
def infer_dense_geometry(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    config: InferenceConfig | None = None,
    spatial_padding_mask: Tensor | None = None,
) -> TiledDenseResult:
    config = config or model.cfg.inference
    use_tiled = config.mode == "tiled" or config.tiled_dense_enabled
    if use_tiled:
        return tiled_dense_geometry(
            model,
            spatial_inputs,
            spacing_um,
            dref_um,
            config=config,
            spatial_padding_mask=spatial_padding_mask,
        )
    output = model(
        spatial_inputs,
        spacing_um,
        dref_um,
        spatial_padding_mask=spatial_padding_mask,
        execution_stage="geometry",
    )
    return TiledDenseResult(
        geometry=GeometryState(
            foreground_logits=output.geometry.foreground_logits,
            surface_logits=output.geometry.surface_logits,
            separator_logits=output.geometry.separator_logits,
            sdf=output.geometry.sdf,
            flow=output.geometry.flow,
            centroid_offset=output.geometry.centroid_offset,
            seed_logits=output.geometry.seed_logits,
            features=None,
        ),
        blend_weight_sum=spatial_inputs.new_ones(
            (spatial_inputs.shape[0], 1, *spatial_inputs.shape[-3:])
        ),
        tile_count=1,
    )


@torch.inference_mode()
def tiled_spatial_inference(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    config: InferenceConfig | None = None,
    spatial_padding_mask: Tensor | None = None,
) -> TiledSpatialResult:
    """Run global watershed/RAG/tokenization without stitched feature volumes."""
    config = config or model.cfg.inference
    dense = tiled_dense_geometry(
        model,
        spatial_inputs,
        spacing_um,
        dref_um,
        config=config,
        spatial_padding_mask=spatial_padding_mask,
    )
    supervoxels = model.watershed(
        dense.geometry,
        spacing_um,
        dref_um,
        spatial_padding_mask,
    )
    stats = stream_tiled_label_feature_stats(
        model,
        spatial_inputs,
        spacing_um,
        dref_um,
        supervoxels,
        dense.blend_weight_sum,
        config=config,
    )
    batch_size = spatial_inputs.shape[0]
    channels = model.cfg.spatial.channels
    dummy = SpatialDecodeState(
        d0=spatial_inputs.new_zeros((batch_size, channels[0], 1, 1, 1)),
        d1=spatial_inputs.new_zeros((batch_size, channels[1], 1, 1, 1)),
        d2=spatial_inputs.new_zeros((batch_size, channels[2], 1, 1, 1)),
    )
    rag = model.rag_builder(
        supervoxels,
        dummy.d0,
        spatial_inputs,
        dense.geometry,
        spacing_um,
        dref_um,
        pooled_d0_by_batch=stats.pooled_scales[0],
    )
    rag = model.rag_network(rag)
    partition = model.partitioner(
        rag,
        rag.spatial_edge_logits,
        model.cfg.partition.spatial_merge_threshold,
    )
    instances = model.instance_tokenizer(
        partition,
        rag,
        dummy,
        dense.geometry,
        spacing_um,
        dref_um,
        pooled_supervoxel_scales=stats.pooled_scales,
        pooled_supervoxel_counts=stats.counts_scales,
    )
    return TiledSpatialResult(
        dense=dense,
        supervoxel_labels=supervoxels,
        feature_stats=stats,
        rag=rag,
        spatial_partition=partition,
        provisional_instances=instances,
    )


def _dummy_decoded(model, spatial_inputs: Tensor) -> SpatialDecodeState:
    batch_size = spatial_inputs.shape[0]
    channels = model.cfg.spatial.channels
    return SpatialDecodeState(
        d0=spatial_inputs.new_zeros((batch_size, channels[0], 1, 1, 1)),
        d1=spatial_inputs.new_zeros((batch_size, channels[1], 1, 1, 1)),
        d2=spatial_inputs.new_zeros((batch_size, channels[2], 1, 1, 1)),
    )


@torch.inference_mode()
def stream_tiled_observation_cache(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    temporal: TemporalState,
    *,
    config: InferenceConfig | None = None,
) -> SpatialObservationCache:
    """Sample D1/D2/hidden geometry at references without global feature maps."""
    config = config or model.cfg.inference
    count = temporal.tokens.shape[0]
    width = model.cfg.temporal.d_model
    cache = SpatialObservationCache(
        d1_projected=temporal.tokens.new_zeros((count, width)),
        d2_projected=temporal.tokens.new_zeros((count, width)),
        hidden_geometry_projected=temporal.tokens.new_zeros((count, width)),
    )
    if temporal.is_empty:
        return cache
    shape = tuple(spatial_inputs.shape[-3:])
    specs = generate_dense_tiles(spatial_inputs.shape[0], shape, config)
    assignments: dict[int, list[int]] = {}
    for row in range(count):
        b = int(temporal.batch_index[row].item())
        extent = (
            torch.as_tensor(shape, device=temporal.ref_um.device).float() - 1
        ) * spacing_um[b].float()
        voxel = torch.round(
            (temporal.ref_um[row].float() + 0.5 * extent)
            / spacing_um[b].float().clamp_min(1e-6)
        ).long()
        best_index = None
        best_weight = -1.0
        for spec_index, spec in enumerate(specs):
            if spec.batch_index != b:
                continue
            if not all(
                int(spec.slices_zyx[axis].start) <= int(voxel[axis]) < int(spec.slices_zyx[axis].stop)
                for axis in range(3)
            ):
                continue
            local = tuple(
                int(voxel[axis]) - int(spec.slices_zyx[axis].start)
                for axis in range(3)
            )
            weight = float(
                tile_blend_weight(
                    spec,
                    shape,
                    config.tile_halo_zyx,
                    device=spatial_inputs.device,
                )[local].item()
            )
            if weight > best_weight:
                best_weight = weight
                best_index = spec_index
        if best_index is None:
            raise RuntimeError("temporal reference is outside all dense tiles")
        assignments.setdefault(best_index, []).append(row)

    for spec_index, rows in assignments.items():
        spec = specs[spec_index]
        b = spec.batch_index
        tile = spatial_inputs[
            b : b + 1,
            :,
            spec.slices_zyx[0],
            spec.slices_zyx[1],
            spec.slices_zyx[2],
        ]
        output = model(
            tile,
            spacing_um[b : b + 1],
            dref_um[b : b + 1],
            execution_stage="geometry",
        )
        row_index = torch.tensor(rows, device=temporal.tokens.device, dtype=torch.long)
        global_center = 0.5 * (
            torch.as_tensor(shape, device=spacing_um.device).float() - 1
        )
        tile_center = torch.tensor(
            [
                0.5 * (int(axis.start) + int(axis.stop) - 1)
                for axis in spec.slices_zyx
            ],
            device=spacing_um.device,
        )
        shift_um = (tile_center - global_center) * spacing_um[b]
        subset = replace(
            temporal,
            tokens=temporal.tokens[row_index],
            ref_um=temporal.ref_um[row_index] - shift_um,
            batch_index=torch.zeros(len(rows), device=row_index.device, dtype=torch.long),
            salience=temporal.salience[row_index],
            reliability=temporal.reliability[row_index],
            status=temporal.status[row_index],
        )
        local_cache = model.temporal_observer.build_cache(
            subset,
            output.decoded_spatial,
            output.geometry,
            output.spatial_pyramid.spacings_um,
            spacing_um[b : b + 1],
            dref_um[b : b + 1],
        )
        cache.d1_projected[row_index] = local_cache.d1_projected
        cache.d2_projected[row_index] = local_cache.d2_projected
        cache.hidden_geometry_projected[row_index] = (
            local_cache.hidden_geometry_projected
        )
    return cache


def _spatial_from_streamed_stats(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    geometry,
    supervoxels: List[Tensor],
    stats: StreamedLabelFeatureStats,
) -> tuple[RAGState, PartitionState, InstanceState, SpatialDecodeState]:
    dummy = _dummy_decoded(model, spatial_inputs)
    rag = model.rag_builder(
        supervoxels,
        dummy.d0,
        spatial_inputs,
        geometry,
        spacing_um,
        dref_um,
        pooled_d0_by_batch=stats.pooled_scales[0],
    )
    rag = model.rag_network(rag)
    partition = model.partitioner(
        rag,
        rag.spatial_edge_logits,
        model.cfg.partition.spatial_merge_threshold,
    )
    instances = model.instance_tokenizer(
        partition,
        rag,
        dummy,
        geometry,
        spacing_um,
        dref_um,
        pooled_supervoxel_scales=stats.pooled_scales,
        pooled_supervoxel_counts=stats.counts_scales,
    )
    return rag, partition, instances, dummy


@torch.inference_mode()
def tiled_temporal_inference(
    model,
    spatial_inputs: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    temporal_input: TemporalInput | None = None,
    config: InferenceConfig | None = None,
    spatial_padding_mask: Tensor | None = None,
    run_refinement: bool = True,
    apply_existence_filter: bool = True,
) -> TiledTemporalResult:
    """Memory-bounded global tiled inference through temporal/refinement stages."""
    config = config or model.cfg.inference
    spatial = tiled_spatial_inference(
        model,
        spatial_inputs,
        spacing_um,
        dref_um,
        config=config,
        spatial_padding_mask=spatial_padding_mask,
    )
    temporal_base = model.temporal_encoder(temporal_input)
    cache = stream_tiled_observation_cache(
        model,
        spatial_inputs,
        spacing_um,
        dref_um,
        temporal_base,
        config=config,
    )
    dummy = _dummy_decoded(model, spatial_inputs)
    temporal = model.temporal_observer(
        temporal_base,
        dummy,
        spatial.dense.geometry,
        [spacing_um, spacing_um, spacing_um],
        spacing_um,
        dref_um,
        cache=cache,
    )
    initial_reasoning = model.instance_temporal(
        spatial.provisional_instances, spatial.rag, temporal, dref_um
    )
    geometry = spatial.dense.geometry
    rag = spatial.rag
    partition = spatial.spatial_partition
    instances = spatial.provisional_instances
    reasoning = initial_reasoning
    refinement = None

    if run_refinement and model.cfg.refinement.enabled:
        requests = build_refinement_requests(
            instances,
            rag,
            temporal,
            reasoning,
            dref_um,
            model.cfg.refinement,
        )
        specs = generate_dense_tiles(spatial_inputs.shape[0], tuple(spatial_inputs.shape[-3:]), config)

        def d0_crop_provider(
            batch_index: int, crop: tuple[slice, slice, slice]
        ) -> Tensor:
            candidates = [
                spec
                for spec in specs
                if spec.batch_index == batch_index
                and all(
                    int(spec.slices_zyx[a].start) <= int(crop[a].start)
                    and int(spec.slices_zyx[a].stop) >= int(crop[a].stop)
                    for a in range(3)
                )
            ]
            if candidates:
                box = candidates[0].slices_zyx
            else:
                full_shape = spatial_inputs.shape[-3:]
                box_rows = []
                for axis in range(3):
                    crop_size = int(crop[axis].stop) - int(crop[axis].start)
                    size = min(
                        full_shape[axis],
                        max(config.tile_shape_zyx[axis], crop_size),
                    )
                    center = 0.5 * (int(crop[axis].start) + int(crop[axis].stop) - 1)
                    start = max(0, min(int(round(center - 0.5 * (size - 1))), full_shape[axis] - size))
                    box_rows.append(slice(start, start + size))
                box = tuple(box_rows)
            tile = spatial_inputs[
                batch_index : batch_index + 1,
                :,
                box[0],
                box[1],
                box[2],
            ]
            output = model(
                tile,
                spacing_um[batch_index : batch_index + 1],
                dref_um[batch_index : batch_index + 1],
                execution_stage="geometry",
            )
            local_crop = tuple(
                slice(
                    int(crop[a].start) - int(box[a].start),
                    int(crop[a].stop) - int(box[a].start),
                )
                for a in range(3)
            )
            box_shape = tuple(int(axis.stop) - int(axis.start) for axis in box)
            return model.local_refiner._feature_crop(
                output.decoded_spatial.d0[0], box_shape, local_crop
            )

        refinement = model.local_refiner(
            dummy.d0,
            spatial_inputs,
            geometry,
            spacing_um,
            dref_um,
            requests,
            d0_crop_provider=d0_crop_provider,
        )
        if refinement.applied_count:
            geometry = refinement.geometry
            if model.cfg.refinement.partition_update == "local":
                update = model.local_partition_updater(
                    spatial.supervoxel_labels,
                    geometry,
                    spacing_um,
                    dref_um,
                    spatial_padding_mask,
                )
                supervoxels = update.supervoxel_labels
                refinement = replace(
                    refinement,
                    partition_update="local",
                    partition_fallback=update.used_fallback,
                    partition_fallback_reason=update.fallback_reason,
                )
            else:
                supervoxels = model.watershed(
                    geometry, spacing_um, dref_um, spatial_padding_mask
                )
                refinement = replace(refinement, partition_update="full")
            refined_stats = stream_tiled_label_feature_stats(
                model,
                spatial_inputs,
                spacing_um,
                dref_um,
                supervoxels,
                spatial.dense.blend_weight_sum,
                config=config,
            )
            rag, partition, instances, dummy = _spatial_from_streamed_stats(
                model,
                spatial_inputs,
                spacing_um,
                dref_um,
                geometry,
                supervoxels,
                refined_stats,
            )
            temporal = model.temporal_observer(
                temporal_base,
                dummy,
                geometry,
                [spacing_um, spacing_um, spacing_um],
                spacing_um,
                dref_um,
                cache=cache,
            )
            reasoning = model.instance_temporal(
                instances, rag, temporal, dref_um
            )

    final_partition = model.partitioner(
        rag,
        reasoning.final_edge_logits,
        model.cfg.partition.final_merge_threshold,
    )
    if apply_existence_filter:
        final_labels, _ = model._filter_by_existence(
            final_partition,
            rag,
            instances,
            reasoning,
            model.cfg.instances.exist_threshold,
        )
    else:
        final_labels = final_partition.labels
    centers = centers_from_labels(
        final_labels, spacing_um, geometry_field(geometry, "sdf")
    )
    return TiledTemporalResult(
        spatial=spatial,
        geometry=geometry,
        rag=rag,
        spatial_partition=partition,
        provisional_instances=instances,
        temporal=temporal,
        reasoning=reasoning,
        final_partition=final_partition,
        final_labels=final_labels,
        centers_um=centers,
        refinement=refinement,
    )


__all__ = [
    "DenseTileSpec",
    "StreamedLabelFeatureStats",
    "TiledDenseResult",
    "TiledSpatialResult",
    "TiledTemporalResult",
    "generate_dense_tiles",
    "infer_dense_geometry",
    "stream_tiled_label_feature_stats",
    "stream_tiled_observation_cache",
    "tile_blend_weight",
    "tiled_dense_geometry",
    "tiled_spatial_inference",
    "tiled_temporal_inference",
]
