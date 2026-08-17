from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..types import (
    AggregatedRegionStatistics,
    GeometryDerivedCache,
    GeometryLike,
    ScaleFeatureStatistics,
    SupervoxelStatistics,
    geometry_field,
    geometry_field_crop,
)
from ..utils.tensor_ops import pool_labeled_features, reduce_labeled_voxels, resize_labels_nearest


NATIVE_FIELD_ORDER = (
    "raw",
    "foreground",
    "surface",
    "separator",
    "sdf",
    "flow_z",
    "flow_y",
    "flow_x",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "seed",
)


def _profile(profiler, name: str):
    return nullcontext() if profiler is None else profiler.profile(name)


def native_geometry_fields(
    geometry: GeometryLike,
    batch_index: int,
    raw: Tensor,
    *,
    derived: GeometryDerivedCache | None = None,
    crop: tuple[slice, slice, slice] | None = None,
) -> dict[str, Tensor]:
    if crop is None:
        foreground = (
            derived.foreground_prob[batch_index, 0]
            if derived is not None
            else geometry_field(geometry, "foreground_logits")[batch_index, 0].sigmoid()
        )
        surface = (
            derived.surface_prob[batch_index, 0]
            if derived is not None
            else geometry_field(geometry, "surface_logits")[batch_index, 0].sigmoid()
        )
        separator = (
            derived.separator_prob[batch_index, 0]
            if derived is not None
            else geometry_field(geometry, "separator_logits")[batch_index, 0].sigmoid()
        )
        seed = (
            derived.seed_prob[batch_index, 0]
            if derived is not None
            else geometry_field(geometry, "seed_logits")[batch_index, 0].sigmoid()
        )
        sdf = derived.sdf[batch_index, 0] if derived is not None else geometry_field(geometry, "sdf")[batch_index, 0]
        flow = geometry_field(geometry, "flow")[batch_index]
        centroid = geometry_field(geometry, "centroid_offset")[batch_index]
        raw_scalar = raw[batch_index, 0]
    else:
        foreground = geometry_field_crop(geometry, "foreground_logits", batch_index, crop)[0].sigmoid()
        surface = geometry_field_crop(geometry, "surface_logits", batch_index, crop)[0].sigmoid()
        separator = geometry_field_crop(geometry, "separator_logits", batch_index, crop)[0].sigmoid()
        seed = geometry_field_crop(geometry, "seed_logits", batch_index, crop)[0].sigmoid()
        sdf = geometry_field_crop(geometry, "sdf", batch_index, crop)[0]
        flow = geometry_field_crop(geometry, "flow", batch_index, crop)
        centroid = geometry_field_crop(geometry, "centroid_offset", batch_index, crop)
        raw_scalar = raw[batch_index, 0, crop[0], crop[1], crop[2]]
    return {
        "raw": raw_scalar,
        "foreground": foreground,
        "surface": surface,
        "separator": separator,
        "sdf": sdf,
        "flow_z": flow[0],
        "flow_y": flow[1],
        "flow_x": flow[2],
        "centroid_z": centroid[0],
        "centroid_y": centroid[1],
        "centroid_x": centroid[2],
        "seed": seed,
    }


def _scale_statistics(feature: Tensor, labels: Tensor) -> ScaleFeatureStatistics:
    pooled, counts = pool_labeled_features(feature, labels)
    means, maxima = pooled.chunk(2, dim=-1)
    return ScaleFeatureStatistics(counts=counts, sums=means * counts[:, None], maxima=maxima)


def _pad_rows(value: Tensor, rows: int, fill: float = 0.0) -> Tensor:
    if value.shape[0] >= rows:
        return value[:rows]
    shape = (rows - value.shape[0], *value.shape[1:])
    return torch.cat([value, value.new_full(shape, fill)], dim=0)


# Adaptive exact region-local pooling for large feature maps.
# The existing dense scatter reducer is retained as a fallback.
_BBOX_POOL_MAX_REGIONS = 512
_BBOX_POOL_MAX_VOLUME_RATIO = 16.0


def _active_region_boxes(
    counts: Tensor,
    min_voxel: Tensor,
    max_voxel: Tensor,
) -> list[tuple[int, tuple[int, int, int], tuple[int, int, int]]]:
    if counts.numel() == 0:
        return []
    counts_cpu = counts.detach().cpu().tolist()
    bounds_cpu = torch.cat([min_voxel, max_voxel], dim=-1).detach().cpu().tolist()
    result = []
    for row, count in enumerate(counts_cpu):
        if float(count) <= 0.0:
            continue
        values = bounds_cpu[row]
        result.append(
            (
                row,
                (int(values[0]), int(values[1]), int(values[2])),
                (int(values[3]), int(values[4]), int(values[5])),
            )
        )
    return result


def _map_region_boxes_to_scale(
    native_boxes: Sequence[
        tuple[int, tuple[int, int, int], tuple[int, int, int]]
    ],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[
    list[tuple[int, tuple[int, int, int], tuple[int, int, int]]],
    float,
]:
    # PyTorch nearest resize maps target index j to floor(j*S/T).
    # For native source interval [lo, hi], the exact target half-open interval is
    # [ceil(lo*T/S), ceil((hi+1)*T/S)).
    mapped = []
    total_bbox_voxels = 0
    for row, native_lower, native_upper in native_boxes:
        starts = []
        stops = []
        valid = True
        for axis in range(3):
            source_size = int(source_shape[axis])
            target_size = int(target_shape[axis])
            lo = int(native_lower[axis])
            hi = int(native_upper[axis])
            start = (lo * target_size + source_size - 1) // source_size
            stop = ((hi + 1) * target_size + source_size - 1) // source_size
            start = max(0, min(start, target_size))
            stop = max(0, min(stop, target_size))
            if stop <= start:
                valid = False
                break
            starts.append(start)
            stops.append(stop)
        if not valid:
            continue
        lower = (starts[0], starts[1], starts[2])
        upper = (stops[0], stops[1], stops[2])
        total_bbox_voxels += (
            (upper[0] - lower[0])
            * (upper[1] - lower[1])
            * (upper[2] - lower[2])
        )
        mapped.append((row, lower, upper))

    target_voxels = max(
        int(target_shape[0]) * int(target_shape[1]) * int(target_shape[2]), 1
    )
    return mapped, float(total_bbox_voxels) / float(target_voxels)


def _scale_statistics_bounded(
    feature: Tensor,
    labels: Tensor,
    rows: int,
    mapped_boxes: Sequence[
        tuple[int, tuple[int, int, int], tuple[int, int, int]]
    ],
    *,
    known_counts: Tensor | None = None,
) -> ScaleFeatureStatistics:
    if feature.ndim != 4 or labels.ndim != 3:
        raise ValueError('feature must be [C,Z,Y,X] and labels [Z,Y,X]')
    if tuple(feature.shape[-3:]) != tuple(labels.shape):
        raise ValueError('feature and labels must have matching spatial shapes')

    channels = int(feature.shape[0])
    counts = feature.new_zeros((rows,))
    sums = feature.new_zeros((rows, channels))
    maxima = feature.new_full((rows, channels), -torch.inf)

    if known_counts is not None:
        copy_rows = min(rows, int(known_counts.shape[0]))
        counts[:copy_rows] = known_counts[:copy_rows].to(
            device=feature.device, dtype=feature.dtype
        )

    for row, lower, upper in mapped_boxes:
        z0, y0, x0 = lower
        z1, y1, x1 = upper
        label_crop = labels[z0:z1, y0:y1, x0:x1]
        feature_crop = feature[:, z0:z1, y0:y1, x0:x1]
        mask = label_crop == (row + 1)
        mask4 = mask.unsqueeze(0)

        if known_counts is None:
            counts[row] = mask.sum().to(feature.dtype)

        sums[row] = (
            feature_crop.masked_fill(~mask4, 0)
            .sum(dim=(1, 2, 3), dtype=torch.float32)
            .to(feature.dtype)
        )
        maxima[row] = feature_crop.masked_fill(~mask4, -torch.inf).amax(
            dim=(1, 2, 3)
        )

    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return ScaleFeatureStatistics(counts=counts, sums=sums, maxima=maxima)


def _scale_statistics_adaptive(
    feature: Tensor,
    labels: Tensor,
    rows: int,
    native_boxes: Sequence[
        tuple[int, tuple[int, int, int], tuple[int, int, int]]
    ],
    source_shape: tuple[int, int, int],
    *,
    native_counts: Tensor | None = None,
) -> ScaleFeatureStatistics:
    target_shape = tuple(int(v) for v in feature.shape[-3:])
    mapped_boxes, bbox_ratio = _map_region_boxes_to_scale(
        native_boxes, source_shape, target_shape
    )

    use_bounded = (
        len(native_boxes) <= _BBOX_POOL_MAX_REGIONS
        and bbox_ratio <= _BBOX_POOL_MAX_VOLUME_RATIO
    )
    if use_bounded:
        return _scale_statistics_bounded(
            feature,
            labels,
            rows,
            mapped_boxes,
            known_counts=(
                native_counts
                if target_shape == source_shape and native_counts is not None
                else None
            ),
        )

    dense = _scale_statistics(feature, labels)
    return ScaleFeatureStatistics(
        counts=_pad_rows(dense.counts, rows),
        sums=_pad_rows(dense.sums, rows),
        maxima=_pad_rows(dense.maxima, rows),
    )


def build_supervoxel_statistics(
    labels_by_batch: Sequence[Tensor],
    spatial_inputs: Tensor,
    geometry: GeometryLike,
    spacing_um: Tensor,
    scale_features: tuple[Tensor, Tensor, Tensor] | None,
    *,
    derived: GeometryDerivedCache | None = None,
    pooled_scales: tuple[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]] | None = None,
    pooled_counts: tuple[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]] | None = None,
    stage_profiler=None,
) -> list[SupervoxelStatistics]:
    """Make the single native-volume reduction shared by RAG and tokenizer."""
    result: list[SupervoxelStatistics] = []
    for batch_index, labels in enumerate(labels_by_batch):
        fields = native_geometry_fields(
            geometry, batch_index, spatial_inputs, derived=derived
        )
        with _profile(stage_profiler, "node_native_geometry_stats"):
            reduced = reduce_labeled_voxels(
                labels,
                spacing_um[batch_index],
                fields=fields,
                argmax_field=fields["sdf"],
                need_nearest_centroid=False,
            )
        n = reduced.counts.shape[0]
        native_boxes = _active_region_boxes(
            reduced.counts, reduced.min_voxel, reduced.max_voxel
        )
        source_shape = tuple(int(v) for v in labels.shape)
        scales: list[ScaleFeatureStatistics] = []
        for scale in range(3):
            if pooled_scales is not None:
                pooled = _pad_rows(pooled_scales[scale][batch_index], n)
                counts = (
                    _pad_rows(pooled_counts[scale][batch_index], n)
                    if pooled_counts is not None
                    else reduced.counts.to(pooled.dtype)
                )
                mean, maximum = pooled.chunk(2, dim=-1)
                scales.append(
                    ScaleFeatureStatistics(
                        counts=counts,
                        sums=mean * counts[:, None],
                        maxima=maximum,
                    )
                )
            elif scale_features is not None:
                feature = scale_features[scale][batch_index]
                scaled_labels = resize_labels_nearest(
                    labels, tuple(feature.shape[-3:])
                )
                with _profile(stage_profiler, f"node_D{scale}_stats"):
                    scales.append(
                        _scale_statistics_adaptive(
                            feature,
                            scaled_labels,
                            n,
                            native_boxes,
                            source_shape,
                            native_counts=reduced.counts,
                        )
                    )
            else:
                raise ValueError("scale_features or pooled_scales must be supplied")
        if reduced.argmax_flat_index is None:
            raise RuntimeError("SDF argmax statistics were not produced")
        result.append(
            SupervoxelStatistics(
                volume_shape_zyx=tuple(labels.shape),
                spacing_um=spacing_um[batch_index],
                counts=reduced.counts,
                coordinate_sums=reduced.coordinate_sums,
                coordinate_square_sums=reduced.coordinate_square_sums,
                min_voxel=reduced.min_voxel,
                max_voxel=reduced.max_voxel,
                field_sums=dict(reduced.field_sums),
                field_maxima=dict(reduced.field_maxima),
                sdf_argmax_flat_index=reduced.argmax_flat_index,
                scales=tuple(scales),  # type: ignore[arg-type]
            )
        )
    return result


def _aggregate_sum(values: Tensor, component: Tensor, count: int) -> Tensor:
    out = values.new_zeros((count, *values.shape[1:]))
    if values.numel():
        out.index_add_(0, component, values)
    return out


def _aggregate_extreme(
    values: Tensor, component: Tensor, count: int, *, reduce: str, fill: float
) -> Tensor:
    out = values.new_full((count, *values.shape[1:]), fill)
    if values.numel():
        index = component.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
        out.scatter_reduce_(0, index, values, reduce=reduce, include_self=True)
    return out


def aggregate_supervoxel_statistics(
    statistics: SupervoxelStatistics,
    component: Tensor,
    component_count: int,
) -> AggregatedRegionStatistics:
    """Exactly combine supervoxel rows into provisional/final instances."""
    rows = min(component.numel(), statistics.counts.numel())
    component = component[:rows].long()
    valid = statistics.counts[:rows] > 0
    component_valid = component[valid]
    counts = _aggregate_sum(statistics.counts[:rows][valid], component_valid, component_count)
    coordinate_sums = _aggregate_sum(
        statistics.coordinate_sums[:rows][valid], component_valid, component_count
    )
    square_sums = _aggregate_sum(
        statistics.coordinate_square_sums[:rows][valid], component_valid, component_count
    )
    minima = _aggregate_extreme(
        statistics.min_voxel[:rows][valid], component_valid, component_count,
        reduce="amin", fill=float(max(statistics.volume_shape_zyx)),
    )
    maxima = _aggregate_extreme(
        statistics.max_voxel[:rows][valid], component_valid, component_count,
        reduce="amax", fill=-1,
    )
    empty = counts <= 0
    minima[empty] = 0
    maxima[empty] = 0
    field_sums = {
        name: _aggregate_sum(value[:rows][valid], component_valid, component_count)
        for name, value in statistics.field_sums.items()
    }
    field_maxima = {
        name: _aggregate_extreme(
            value[:rows][valid], component_valid, component_count,
            reduce="amax", fill=-torch.inf,
        )
        for name, value in statistics.field_maxima.items()
    }
    for value in field_maxima.values():
        value[empty] = 0

    sdf_values = statistics.field_maxima["sdf"][:rows][valid]
    sdf_max = _aggregate_extreme(
        sdf_values, component_valid, component_count, reduce="amax", fill=-torch.inf
    )
    source_indices = statistics.sdf_argmax_flat_index[:rows][valid]
    total_voxels = int(torch.tensor(statistics.volume_shape_zyx).prod().item())
    candidates = torch.where(
        sdf_values == sdf_max[component_valid],
        source_indices,
        torch.full_like(source_indices, total_voxels),
    )
    sdf_argmax = _aggregate_extreme(
        candidates, component_valid, component_count, reduce="amin", fill=total_voxels
    )
    sdf_argmax[empty] = 0

    scales: list[ScaleFeatureStatistics] = []
    for scale in statistics.scales:
        scale_rows = min(rows, scale.counts.numel())
        scale_valid = scale.counts[:scale_rows] > 0
        scale_component = component[:scale_rows][scale_valid]
        scale_counts = _aggregate_sum(
            scale.counts[:scale_rows][scale_valid], scale_component, component_count
        )
        scale_sums = _aggregate_sum(
            scale.sums[:scale_rows][scale_valid], scale_component, component_count
        )
        scale_maxima = _aggregate_extreme(
            scale.maxima[:scale_rows][scale_valid], scale_component, component_count,
            reduce="amax", fill=-torch.inf,
        )
        scale_maxima[scale_counts <= 0] = 0
        scales.append(ScaleFeatureStatistics(scale_counts, scale_sums, scale_maxima))
    return AggregatedRegionStatistics(
        counts=counts,
        coordinate_sums=coordinate_sums,
        coordinate_square_sums=square_sums,
        min_voxel=minima,
        max_voxel=maxima,
        field_sums=field_sums,
        field_maxima=field_maxima,
        sdf_argmax_flat_index=sdf_argmax,
        scales=tuple(scales),  # type: ignore[arg-type]
        volume_shape_zyx=statistics.volume_shape_zyx,
        spacing_um=statistics.spacing_um,
    )


def _scaled_crop(
    labels: Tensor,
    feature: Tensor,
    source_box: tuple[slice, slice, slice],
) -> tuple[Tensor, Tensor]:
    source_shape = labels.shape
    target_shape = feature.shape[-3:]
    target_indices: list[Tensor] = []
    target_slices: list[slice] = []
    for axis, (source_size, target_size) in enumerate(zip(source_shape, target_shape)):
        destination = torch.arange(target_size, device=labels.device)
        source = torch.div(destination * source_size, target_size, rounding_mode="floor")
        keep = (source >= int(source_box[axis].start)) & (source < int(source_box[axis].stop))
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        if selected.numel() == 0:
            selected = torch.tensor(
                [min(target_size - 1, int(source_box[axis].start) * target_size // source_size)],
                device=labels.device,
            )
        start = int(selected[0].item())
        stop = int(selected[-1].item()) + 1
        target_slices.append(slice(start, stop))
        target_indices.append(source[start:stop])
    scaled_labels = labels.index_select(0, target_indices[0])
    scaled_labels = scaled_labels.index_select(1, target_indices[1])
    scaled_labels = scaled_labels.index_select(2, target_indices[2])
    feature_crop = feature[
        :, target_slices[0], target_slices[1], target_slices[2]
    ]
    return feature_crop, scaled_labels


def _replace_rows(
    base: Tensor,
    update: Tensor,
    rows: Tensor,
    total: int,
    *,
    fill: float = 0.0,
) -> Tensor:
    """Replace every affected row, explicitly clearing rows absent locally."""
    output = _pad_rows(base, total, fill=fill).clone()
    target = rows[(rows >= 0) & (rows < total)]
    if target.numel():
        output[target] = fill
        available = target[target < update.shape[0]]
        if available.numel():
            output[available] = update[available].to(output.dtype)
    return output


def local_refresh_groups(
    initial_labels: Sequence[Tensor],
    labels_by_batch: Sequence[Tensor],
    updated_boxes: Sequence[tuple[int, tuple[slice, slice, slice]]],
) -> list[tuple[int, tuple[slice, slice, slice], Tensor]]:
    """Preserve successful local-partition cluster boundaries for downstream refresh."""
    groups: list[tuple[int, tuple[slice, slice, slice], Tensor]] = []
    for batch_index, box in updated_boxes:
        if batch_index < 0 or batch_index >= len(labels_by_batch):
            raise IndexError("local refresh batch index out of range")
        current_ids = torch.unique(labels_by_batch[batch_index][box])
        previous_ids = torch.unique(initial_labels[batch_index][box])
        affected = torch.unique(torch.cat([current_ids, previous_ids])).long()
        affected = affected[affected > 0]
        groups.append((batch_index, box, affected))
    return groups


def local_dependency_box(
    statistics: SupervoxelStatistics,
    affected_ids: Tensor,
    edit_box: tuple[slice, slice, slice],
    volume_shape: tuple[int, int, int],
    *,
    halo: int = 0,
) -> tuple[slice, slice, slice]:
    """Bound one refresh cluster by its edit plus complete cached affected-object bounds."""
    device = statistics.counts.device
    lower = torch.tensor(
        [int(edit_box[axis].start) for axis in range(3)],
        device=device,
        dtype=torch.long,
    )
    upper = torch.tensor(
        [int(edit_box[axis].stop) for axis in range(3)],
        device=device,
        dtype=torch.long,
    )
    rows = affected_ids.to(device=device, dtype=torch.long) - 1
    rows = rows[(rows >= 0) & (rows < statistics.counts.shape[0])]
    if rows.numel():
        rows = rows[statistics.counts[rows] > 0]
    if rows.numel():
        lower = torch.minimum(lower, statistics.min_voxel[rows].amin(dim=0))
        upper = torch.maximum(
            upper,
            statistics.max_voxel[rows].amax(dim=0) + 1,
        )
    if halo:
        lower = lower - int(halo)
        upper = upper + int(halo)
    shape = torch.as_tensor(volume_shape, device=device, dtype=torch.long)
    lower = lower.clamp_min(0)
    upper = torch.minimum(upper, shape)
    return tuple(
        slice(int(lower[axis].item()), int(upper[axis].item()))
        for axis in range(3)
    )


def update_supervoxel_statistics_local(
    initial: Sequence[SupervoxelStatistics],
    initial_labels: Sequence[Tensor],
    labels_by_batch: Sequence[Tensor],
    updated_boxes: Sequence[tuple[int, tuple[slice, slice, slice]]],
    spatial_inputs: Tensor,
    geometry: GeometryLike,
    spacing_um: Tensor,
    scale_features: tuple[Tensor, Tensor, Tensor] | None,
    *,
    pooled_scales: tuple[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]] | None = None,
    pooled_counts: tuple[Sequence[Tensor], Sequence[Tensor], Sequence[Tensor]] | None = None,
) -> tuple[list[SupervoxelStatistics], list[Tensor]]:
    """Refresh cached rows sequentially per local-partition cluster.

    The previous implementation unioned every update box in a batch before
    reducing voxels. Distant local edits therefore recreated a near-global
    reduction. Here each successful local-partition cluster keeps its own
    bounded dependency region. If a label is touched by more than one cluster,
    the later cluster sees the already-updated cached bounds and remains exact.
    """
    result = list(initial)
    groups = local_refresh_groups(
        initial_labels,
        labels_by_batch,
        updated_boxes,
    )
    affected_chunks: list[list[Tensor]] = [
        [] for _ in labels_by_batch
    ]

    for batch_index, edit_box, affected in groups:
        if affected.numel() == 0:
            continue
        labels = labels_by_batch[batch_index]
        previous = result[batch_index]
        affected_chunks[batch_index].append(affected)

        dependency_box = local_dependency_box(
            previous,
            affected,
            edit_box,
            tuple(labels.shape),
        )
        lower = tuple(int(axis.start) for axis in dependency_box)
        label_crop = labels[dependency_box]
        fields = native_geometry_fields(
            geometry,
            batch_index,
            spatial_inputs,
            crop=dependency_box,
        )
        reduced = reduce_labeled_voxels(
            label_crop,
            spacing_um[batch_index],
            fields=fields,
            argmax_field=fields["sdf"],
            need_nearest_centroid=False,
            coordinate_offset_zyx=lower,
            coordinate_shape_zyx=tuple(labels.shape),
        )
        if reduced.argmax_flat_index is None:
            raise RuntimeError("local SDF argmax statistics were not produced")

        total = int(labels.max().item())
        rows = affected - 1
        scales: list[ScaleFeatureStatistics] = []
        for scale_index, old_scale in enumerate(previous.scales):
            if scale_features is not None:
                feature_crop, scaled_labels = _scaled_crop(
                    labels,
                    scale_features[scale_index][batch_index],
                    dependency_box,
                )
                local_scale = _scale_statistics(feature_crop, scaled_labels)
                scales.append(
                    ScaleFeatureStatistics(
                        counts=_replace_rows(
                            old_scale.counts,
                            local_scale.counts,
                            rows,
                            total,
                        ),
                        sums=_replace_rows(
                            old_scale.sums,
                            local_scale.sums,
                            rows,
                            total,
                        ),
                        maxima=_replace_rows(
                            old_scale.maxima,
                            local_scale.maxima,
                            rows,
                            total,
                        ),
                    )
                )
            elif pooled_scales is not None and pooled_counts is not None:
                pooled = _pad_rows(
                    pooled_scales[scale_index][batch_index],
                    total,
                )
                counts = _pad_rows(
                    pooled_counts[scale_index][batch_index],
                    total,
                )
                mean, maximum = pooled.chunk(2, dim=-1)
                scales.append(
                    ScaleFeatureStatistics(
                        counts,
                        mean * counts[:, None],
                        maximum,
                    )
                )
            else:
                raise ValueError(
                    "local statistics need scale features or pooled scale rows"
                )

        result[batch_index] = SupervoxelStatistics(
            volume_shape_zyx=tuple(labels.shape),
            spacing_um=spacing_um[batch_index],
            counts=_replace_rows(
                previous.counts,
                reduced.counts,
                rows,
                total,
            ),
            coordinate_sums=_replace_rows(
                previous.coordinate_sums,
                reduced.coordinate_sums,
                rows,
                total,
            ),
            coordinate_square_sums=_replace_rows(
                previous.coordinate_square_sums,
                reduced.coordinate_square_sums,
                rows,
                total,
            ),
            min_voxel=_replace_rows(
                previous.min_voxel,
                reduced.min_voxel,
                rows,
                total,
            ),
            max_voxel=_replace_rows(
                previous.max_voxel,
                reduced.max_voxel,
                rows,
                total,
            ),
            field_sums={
                name: _replace_rows(
                    previous.field_sums[name],
                    reduced.field_sums[name],
                    rows,
                    total,
                )
                for name in NATIVE_FIELD_ORDER
            },
            field_maxima={
                name: _replace_rows(
                    previous.field_maxima[name],
                    reduced.field_maxima[name],
                    rows,
                    total,
                )
                for name in NATIVE_FIELD_ORDER
            },
            sdf_argmax_flat_index=_replace_rows(
                previous.sdf_argmax_flat_index,
                reduced.argmax_flat_index,
                rows,
                total,
            ),
            scales=tuple(scales),  # type: ignore[arg-type]
        )

    affected_by_batch: list[Tensor] = []
    for batch_index, labels in enumerate(labels_by_batch):
        chunks = affected_chunks[batch_index]
        if chunks:
            affected = torch.unique(torch.cat(chunks)).long()
            affected = affected[affected > 0]
        else:
            affected = labels.new_zeros((0,), dtype=torch.long)
        affected_by_batch.append(affected)
    return result, affected_by_batch


__all__ = [
    "NATIVE_FIELD_ORDER",
    "aggregate_supervoxel_statistics",
    "build_supervoxel_statistics",
    "local_dependency_box",
    "local_refresh_groups",
    "native_geometry_fields",
    "update_supervoxel_statistics_local",
]
