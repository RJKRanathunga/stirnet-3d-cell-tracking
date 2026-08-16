from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class LabeledVoxelStats:
    """Vectorized per-label geometry and optional scalar-field statistics.

    Rows correspond to label IDs ``1..labels.max()``. Missing IDs are retained
    as zero rows so callers keep their existing ordering contract.
    """

    counts: Tensor
    centroid_voxel: Tensor
    centroid_um: Tensor
    variance_um2: Tensor
    min_voxel: Tensor
    max_voxel: Tensor
    field_means: Mapping[str, Tensor]
    field_maxima: Mapping[str, Tensor]
    argmax_flat_index: Tensor | None
    nearest_centroid_flat_index: Tensor


def relabel_contiguous(labels: Tensor) -> Tensor:
    """Relabel positive integer labels to 1..N without changing connectivity."""
    unique = torch.unique(labels)
    unique = unique[unique > 0]
    if unique.numel() == 0:
        return torch.zeros_like(labels)
    out = torch.zeros_like(labels)
    for new_id, old_id in enumerate(unique.tolist(), 1):
        out[labels == int(old_id)] = new_id
    return out


def pool_labeled_features(
    feature: Tensor,
    labels: Tensor,
    *,
    include_max: bool = True,
) -> tuple[Tensor, Tensor]:
    """Pool [C,Z,Y,X] features over contiguous labels [Z,Y,X].

    Returns [N,C] mean (or [N,2C] mean+max) and counts [N].
    Label zero is ignored. Empty IDs are retained if labels are not contiguous.
    """
    if feature.ndim != 4 or labels.ndim != 3:
        raise ValueError("feature must be [C,Z,Y,X] and labels [Z,Y,X]")
    max_id = int(labels.max().item())
    if max_id <= 0:
        width = feature.shape[0] * (2 if include_max else 1)
        return feature.new_zeros((0, width)), feature.new_zeros((0,))
    flat_labels = labels.reshape(-1).long()
    flat_feat = feature.flatten(1).transpose(0, 1)
    valid = flat_labels > 0
    ids = flat_labels[valid] - 1
    values = flat_feat[valid]
    counts = feature.new_zeros((max_id,))
    counts.index_add_(0, ids, torch.ones_like(ids, dtype=feature.dtype))
    sums = feature.new_zeros((max_id, feature.shape[0]))
    sums.index_add_(0, ids, values)
    means = sums / counts.clamp_min(1)[:, None]
    if not include_max:
        return means, counts
    maxima = feature.new_full((max_id, feature.shape[0]), -torch.inf)
    maxima.scatter_reduce_(
        0,
        ids[:, None].expand(-1, feature.shape[0]),
        values,
        reduce="amax",
        include_self=True,
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return torch.cat([means, maxima], dim=-1), counts


def pool_labeled_feature_fields(
    fields: Sequence[Tensor],
    labels: Tensor,
    *,
    include_max: bool = True,
) -> tuple[Tensor, Tensor]:
    """Pool several fields without first concatenating dense volumes.

    Each field is ``[C,Z,Y,X]``. The output channel order matches pooling a
    dense concatenation: all means followed by all maxima. Label indexing and
    the valid-voxel scan are shared across fields.
    """
    if labels.ndim != 3 or not fields:
        raise ValueError("labels must be [Z,Y,X] and fields cannot be empty")
    if any(field.ndim != 4 or field.shape[-3:] != labels.shape for field in fields):
        raise ValueError("every field must be [C,Z,Y,X] and align with labels")
    max_id = int(labels.max().item())
    total_channels = sum(field.shape[0] for field in fields)
    if max_id <= 0:
        width = total_channels * (2 if include_max else 1)
        return fields[0].new_zeros((0, width)), fields[0].new_zeros((0,))

    flat_labels = labels.reshape(-1).long()
    valid = flat_labels > 0
    ids = flat_labels[valid] - 1
    counts = fields[0].new_zeros((max_id,))
    counts.index_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
    means: list[Tensor] = []
    maxima: list[Tensor] = []
    for field in fields:
        values = field.flatten(1).transpose(0, 1)[valid]
        sums = field.new_zeros((max_id, field.shape[0]))
        sums.index_add_(0, ids, values)
        means.append(sums / counts.to(field.dtype).clamp_min(1)[:, None])
        if include_max:
            maximum = field.new_full((max_id, field.shape[0]), -torch.inf)
            maximum.scatter_reduce_(
                0,
                ids[:, None].expand(-1, field.shape[0]),
                values,
                reduce="amax",
                include_self=True,
            )
            maxima.append(
                torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            )
    pooled = torch.cat(means + maxima, dim=-1) if include_max else torch.cat(means, dim=-1)
    return pooled, counts


def project_pooled_mean_max(pooled: Tensor, projection) -> Tensor:
    """Apply a channel projection after mean/max pooling.

    The mean half is algebraically identical to mean-pooling a bias-free dense
    pointwise projection. The max half deliberately uses project(max(raw)).
    """
    if pooled.shape[-1] % 2:
        raise ValueError("pooled mean/max features must have an even width")
    mean, maximum = pooled.chunk(2, dim=-1)
    return torch.cat([projection(mean), projection(maximum)], dim=-1)


def reduce_labeled_voxels(
    labels: Tensor,
    spacing_um: Tensor,
    *,
    fields: Mapping[str, Tensor] | None = None,
    argmax_field: Tensor | None = None,
) -> LabeledVoxelStats:
    """Reduce a labeled voxel field in O(Nvoxels + Nlabels) work.

    Physical coordinates are relative to the volume centre and are derived
    from separable axes/linear indices; no ``[Z,Y,X,3]`` grid is constructed.
    Scalar fields must align with ``labels``. If ``argmax_field`` is supplied,
    the first maximum voxel in flattened order is returned for every label.
    """
    if labels.ndim != 3:
        raise ValueError("labels must have shape [Z,Y,X]")
    fields = {} if fields is None else fields
    if any(value.shape != labels.shape for value in fields.values()):
        raise ValueError("all scalar fields must align with labels")
    if argmax_field is not None and argmax_field.shape != labels.shape:
        raise ValueError("argmax_field must align with labels")

    max_id = int(labels.max().item())
    device = labels.device
    coordinate_dtype = torch.float32
    if max_id <= 0:
        empty_scalar = torch.zeros((0,), device=device, dtype=coordinate_dtype)
        empty_vector = torch.zeros((0, 3), device=device, dtype=coordinate_dtype)
        empty_long = torch.zeros((0,), device=device, dtype=torch.long)
        return LabeledVoxelStats(
            counts=empty_scalar,
            centroid_voxel=empty_vector,
            centroid_um=empty_vector,
            variance_um2=empty_vector,
            min_voxel=empty_vector.long(),
            max_voxel=empty_vector.long(),
            field_means={name: value.new_zeros((0,)) for name, value in fields.items()},
            field_maxima={name: value.new_zeros((0,)) for name, value in fields.items()},
            argmax_flat_index=empty_long if argmax_field is not None else None,
            nearest_centroid_flat_index=empty_long,
        )

    flat_labels = labels.reshape(-1).long()
    flat_index = torch.nonzero(flat_labels > 0, as_tuple=False).flatten()
    ids = flat_labels[flat_index] - 1
    counts_long = torch.bincount(ids, minlength=max_id)
    counts = counts_long.to(coordinate_dtype)
    z_size, y_size, x_size = labels.shape
    z_index = torch.div(flat_index, y_size * x_size, rounding_mode="floor")
    remainder = flat_index % (y_size * x_size)
    y_index = torch.div(remainder, x_size, rounding_mode="floor")
    x_index = remainder % x_size
    voxel_components = (z_index, y_index, x_index)

    sums = []
    square_sums = []
    minima = []
    maxima = []
    for component, size in zip(voxel_components, labels.shape):
        values = component.to(coordinate_dtype)
        component_sum = torch.zeros(max_id, device=device, dtype=coordinate_dtype)
        component_sum.index_add_(0, ids, values)
        component_square_sum = torch.zeros_like(component_sum)
        component_square_sum.index_add_(0, ids, values.square())
        minimum = torch.full((max_id,), int(size), device=device, dtype=torch.long)
        maximum = torch.full((max_id,), -1, device=device, dtype=torch.long)
        minimum.scatter_reduce_(0, ids, component, reduce="amin", include_self=True)
        maximum.scatter_reduce_(0, ids, component, reduce="amax", include_self=True)
        sums.append(component_sum)
        square_sums.append(component_square_sum)
        minima.append(minimum)
        maxima.append(maximum)

    safe_counts = counts.clamp_min(1)
    centroid_voxel = torch.stack(sums, dim=-1) / safe_counts[:, None]
    second_moment = torch.stack(square_sums, dim=-1) / safe_counts[:, None]
    variance_voxel = (second_moment - centroid_voxel.square()).clamp_min(0)
    spacing = spacing_um.to(device=device, dtype=coordinate_dtype)
    volume_center = 0.5 * (
        torch.as_tensor(labels.shape, device=device, dtype=coordinate_dtype) - 1
    )
    centroid_um = (centroid_voxel - volume_center) * spacing
    variance_um2 = variance_voxel * spacing.square()
    min_voxel = torch.stack(minima, dim=-1)
    max_voxel = torch.stack(maxima, dim=-1)
    empty = counts_long == 0
    min_voxel[empty] = 0
    max_voxel[empty] = 0
    centroid_voxel[empty] = 0
    centroid_um[empty] = 0
    variance_um2[empty] = 0

    field_means: dict[str, Tensor] = {}
    field_maxima: dict[str, Tensor] = {}
    for name, field in fields.items():
        values = field.reshape(-1)[flat_index]
        field_sum = field.new_zeros((max_id,))
        field_sum.index_add_(0, ids, values)
        field_mean = field_sum / counts.to(field.dtype).clamp_min(1)
        field_max = field.new_full((max_id,), -torch.inf)
        field_max.scatter_reduce_(0, ids, values, reduce="amax", include_self=True)
        field_means[name] = torch.where(empty, torch.zeros_like(field_mean), field_mean)
        field_maxima[name] = torch.where(empty, torch.zeros_like(field_max), field_max)

    delta_z = z_index.to(coordinate_dtype) - centroid_voxel[ids, 0]
    delta_y = y_index.to(coordinate_dtype) - centroid_voxel[ids, 1]
    delta_x = x_index.to(coordinate_dtype) - centroid_voxel[ids, 2]
    distance2 = delta_z.square() + delta_y.square() + delta_x.square()
    nearest_distance = torch.full(
        (max_id,), torch.inf, device=device, dtype=coordinate_dtype
    )
    nearest_distance.scatter_reduce_(
        0, ids, distance2, reduce="amin", include_self=True
    )
    nearest_candidates = torch.where(
        distance2 == nearest_distance[ids],
        flat_index,
        torch.full_like(flat_index, labels.numel()),
    )
    nearest_index = torch.full(
        (max_id,), labels.numel(), device=device, dtype=torch.long
    )
    nearest_index.scatter_reduce_(
        0, ids, nearest_candidates, reduce="amin", include_self=True
    )
    nearest_index[empty] = 0

    argmax_index: Tensor | None = None
    if argmax_field is not None:
        arg_values = argmax_field.reshape(-1)[flat_index]
        arg_max = argmax_field.new_full((max_id,), -torch.inf)
        arg_max.scatter_reduce_(0, ids, arg_values, reduce="amax", include_self=True)
        candidates = torch.where(
            arg_values == arg_max[ids],
            flat_index,
            torch.full_like(flat_index, labels.numel()),
        )
        argmax_index = torch.full(
            (max_id,), labels.numel(), device=device, dtype=torch.long
        )
        argmax_index.scatter_reduce_(
            0, ids, candidates, reduce="amin", include_self=True
        )
        argmax_index[empty] = 0

    return LabeledVoxelStats(
        counts=counts,
        centroid_voxel=centroid_voxel,
        centroid_um=centroid_um,
        variance_um2=variance_um2,
        min_voxel=min_voxel,
        max_voxel=max_voxel,
        field_means=field_means,
        field_maxima=field_maxima,
        argmax_flat_index=argmax_index,
        nearest_centroid_flat_index=nearest_index,
    )


def resize_labels_nearest(labels: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
    if tuple(labels.shape[-3:]) == target_shape:
        return labels
    x = labels.float()[None, None]
    return F.interpolate(x, size=target_shape, mode="nearest")[0, 0].long()
