from __future__ import annotations

import torch
from torch import Tensor


def voxel_to_physical(coords_vox: Tensor, spacing_um: Tensor) -> Tensor:
    """Convert (..., 3) zyx voxel coordinates to physical micrometres."""
    while spacing_um.ndim < coords_vox.ndim:
        spacing_um = spacing_um.unsqueeze(-2)
    return coords_vox * spacing_um


def physical_to_voxel(coords_um: Tensor, spacing_um: Tensor) -> Tensor:
    while spacing_um.ndim < coords_um.ndim:
        spacing_um = spacing_um.unsqueeze(-2)
    return coords_um / spacing_um.clamp_min(1e-8)


def physical_to_cellscale(coords_um_relative: Tensor, dref_um: Tensor) -> Tensor:
    while dref_um.ndim < coords_um_relative.ndim:
        dref_um = dref_um.unsqueeze(-1)
    return coords_um_relative / dref_um.clamp_min(1e-8)


def cellscale_to_physical(coords_cellscale: Tensor, dref_um: Tensor) -> Tensor:
    while dref_um.ndim < coords_cellscale.ndim:
        dref_um = dref_um.unsqueeze(-1)
    return coords_cellscale * dref_um


def feature_grid_coordinates_um(
    spatial_shape: tuple[int, int, int],
    spacing_um: Tensor,
    *,
    relative_to_center: bool = True,
) -> Tensor:
    """Return [B, Z*Y*X, 3] feature-centre coordinates in zyx micrometres."""
    if spacing_um.ndim == 1:
        spacing_um = spacing_um.unsqueeze(0)
    b = spacing_um.shape[0]
    z, y, x = spatial_shape
    device, dtype = spacing_um.device, spacing_um.dtype
    zz = torch.arange(z, device=device, dtype=dtype)
    yy = torch.arange(y, device=device, dtype=dtype)
    xx = torch.arange(x, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=-1).reshape(-1, 3)
    grid = grid.unsqueeze(0) * spacing_um[:, None, :]
    if relative_to_center:
        extent = torch.tensor([z - 1, y - 1, x - 1], device=device, dtype=dtype)[None, :] * spacing_um
        grid = grid - 0.5 * extent[:, None, :]
    return grid


def resize_label_map_nearest(labels: Tensor, spatial_shape: tuple[int, int, int]) -> Tensor:
    """Nearest-neighbour resize for integer 3D label maps without float copies."""
    if labels.ndim not in (3, 4):
        raise ValueError(f"labels must have shape [Z,Y,X] or [B,Z,Y,X], got {tuple(labels.shape)}")
    out = labels
    for dim, target_size in zip(range(labels.ndim - 3, labels.ndim), spatial_shape):
        source_size = labels.shape[dim]
        if source_size == target_size:
            continue
        index = torch.div(
            torch.arange(target_size, device=labels.device) * source_size,
            target_size,
            rounding_mode="floor",
        ).clamp_max(source_size - 1)
        out = out.index_select(dim, index)
    return out


def normalize_reference_to_grid(coords_um_relative: Tensor, shape: tuple[int, int, int], spacing_um: Tensor) -> Tensor:
    """Map relative physical coordinates to grid_sample coordinates in xyz order [-1, 1]."""
    if spacing_um.ndim == 1:
        spacing_um = spacing_um.unsqueeze(0)
    if coords_um_relative.ndim == 2:
        coords_um_relative = coords_um_relative.unsqueeze(0)
    extent = torch.tensor([shape[0] - 1, shape[1] - 1, shape[2] - 1], device=coords_um_relative.device, dtype=coords_um_relative.dtype)
    extent = extent[None, :] * spacing_um
    zyx01 = (coords_um_relative + 0.5 * extent[:, None, :]) / extent[:, None, :].clamp_min(1e-8)
    zyx11 = zyx01 * 2 - 1
    return zyx11[..., [2, 1, 0]]
