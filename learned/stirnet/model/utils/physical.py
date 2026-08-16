from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def build_acquisition_features(spacing_um: Tensor, dref_um: Tensor) -> Tensor:
    if spacing_um.ndim == 1:
        spacing_um = spacing_um.unsqueeze(0)
    if dref_um.ndim == 0:
        dref_um = dref_um.unsqueeze(0)
    log_s = torch.log(spacing_um.clamp_min(1e-8))
    rel = torch.log((spacing_um / dref_um[:, None].clamp_min(1e-8)).clamp_min(1e-8))
    return torch.cat([log_s, rel, torch.log(dref_um[:, None].clamp_min(1e-8))], dim=-1)


def relative_grid_coordinates_um(
    shape: tuple[int, int, int], spacing_um: Tensor, *, device: torch.device | None = None
) -> Tensor:
    """Return [Z,Y,X,3] physical zyx coordinates relative to volume centre."""
    device = device or spacing_um.device
    spacing = spacing_um.to(device=device, dtype=torch.float32)
    axes = []
    for size, step in zip(shape, spacing):
        extent = (size - 1) * step
        axes.append(torch.arange(size, device=device, dtype=torch.float32) * step - 0.5 * extent)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def canonical_resample_spec(
    native_shape: tuple[int, int, int],
    spacing_um: Tensor,
    canonical_spacing_um: tuple[float, float, float],
) -> tuple[tuple[int, int, int], Tensor]:
    """Return an endpoint-preserving canonical grid and its actual spacing."""
    if spacing_um.ndim == 1:
        spacing_um = spacing_um[None]
    native_extent = (
        torch.as_tensor(native_shape, device=spacing_um.device, dtype=spacing_um.dtype)
        - 1
    ) * spacing_um
    requested = spacing_um.new_tensor(canonical_spacing_um)
    target_shapes = torch.round(native_extent / requested).long() + 1
    target_shapes = target_shapes.clamp_min(1)
    if not torch.equal(target_shapes, target_shapes[0:1].expand_as(target_shapes)):
        raise ValueError(
            "batched canonical resampling requires a common target voxel shape"
        )
    target_shape = tuple(int(value) for value in target_shapes[0].tolist())
    denominator = (target_shapes - 1).clamp_min(1).to(spacing_um.dtype)
    actual_spacing = native_extent / denominator
    actual_spacing = torch.where(
        target_shapes > 1, actual_spacing, spacing_um
    )
    return target_shape, actual_spacing


def resample_continuous_volume(
    value: Tensor,
    target_shape: tuple[int, int, int],
) -> Tensor:
    if tuple(value.shape[-3:]) == target_shape:
        return value
    return F.interpolate(
        value,
        size=target_shape,
        mode="trilinear",
        align_corners=True,
    )


def resample_labels_volume(
    labels: Tensor,
    target_shape: tuple[int, int, int],
) -> Tensor:
    """Nearest-neighbor-only label resampling with integer output."""
    original_ndim = labels.ndim
    if original_ndim == 3:
        labels = labels[None, None]
    elif original_ndim == 4:
        labels = labels[:, None]
    elif original_ndim != 5:
        raise ValueError("labels must be [Z,Y,X], [B,Z,Y,X], or [B,1,Z,Y,X]")
    resized = F.interpolate(labels.float(), size=target_shape, mode="nearest").long()
    if original_ndim == 3:
        return resized[0, 0]
    if original_ndim == 4:
        return resized[:, 0]
    return resized


def physical_crop_slices(
    shape: tuple[int, int, int],
    spacing_um: Tensor,
    center_um: Tensor,
    radius_um: Tensor | float,
) -> tuple[slice, slice, slice]:
    spacing = spacing_um.float()
    center = center_um.float()
    radius = torch.as_tensor(radius_um, device=center.device, dtype=torch.float32)
    if radius.ndim == 0:
        radius = radius.repeat(3)
    shape_t = torch.tensor(shape, device=center.device, dtype=torch.long)
    extent = (shape_t.float() - 1) * spacing
    center_vox = torch.round((center + 0.5 * extent) / spacing.clamp_min(1e-8)).long()
    radius_vox = torch.ceil(radius / spacing.clamp_min(1e-8)).long()
    lo = torch.maximum(center_vox - radius_vox, torch.zeros_like(center_vox))
    hi = torch.minimum(center_vox + radius_vox + 1, shape_t)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo.tolist(), hi.tolist()))  # type: ignore[return-value]


def _axis_gradient(x: Tensor, spacing: Tensor, dim: int) -> Tensor:
    """Stable first derivative with central interior and replicated edge differences."""
    if x.shape[dim] <= 1:
        return torch.zeros_like(x)
    fwd = torch.roll(x, shifts=-1, dims=dim)
    bwd = torch.roll(x, shifts=1, dims=dim)
    g = (fwd - bwd) / (2.0 * spacing)
    first = [slice(None)] * x.ndim
    second = [slice(None)] * x.ndim
    first[dim] = 0
    second[dim] = 1
    g[tuple(first)] = (x[tuple(second)] - x[tuple(first)]) / spacing
    last = [slice(None)] * x.ndim
    prev = [slice(None)] * x.ndim
    last[dim] = -1
    prev[dim] = -2
    g[tuple(last)] = (x[tuple(last)] - x[tuple(prev)]) / spacing
    return g


def physical_gradient3d(scalar: Tensor, spacing_um: Tensor) -> Tensor:
    """Gradient of [B,1,Z,Y,X] in physical z/y/x units -> [B,3,Z,Y,X]."""
    if scalar.ndim != 5 or scalar.shape[1] != 1:
        raise ValueError("scalar must have shape [B,1,Z,Y,X]")
    if spacing_um.ndim == 1:
        spacing_um = spacing_um[None]
    grads = []
    for axis, dim in enumerate((2, 3, 4)):
        per_batch = []
        for b in range(scalar.shape[0]):
            per_batch.append(_axis_gradient(scalar[b : b + 1], spacing_um[b, axis], dim))
        grads.append(torch.cat(per_batch, dim=0))
    return torch.cat(grads, dim=1)
