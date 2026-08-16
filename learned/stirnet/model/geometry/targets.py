from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import Tensor


@dataclass
class GeometryTargets:
    foreground: Tensor
    surface: Tensor
    separator: Tensor
    sdf: Tensor
    sdf_valid: Tensor
    flow: Tensor
    centroid_offset: Tensor
    seed: Tensor

    def to(self, device: torch.device | str) -> "GeometryTargets":
        return GeometryTargets(**{k: v.to(device) for k, v in self.__dict__.items()})


def _boundaries(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    surface = np.zeros_like(labels, dtype=bool)
    separator = np.zeros_like(labels, dtype=bool)
    for axis in range(3):
        a_slice = [slice(None)] * 3
        b_slice = [slice(None)] * 3
        a_slice[axis] = slice(0, -1)
        b_slice[axis] = slice(1, None)
        a = labels[tuple(a_slice)]
        b = labels[tuple(b_slice)]
        diff = a != b
        surf = diff & ((a == 0) ^ (b == 0))
        sep = diff & (a > 0) & (b > 0)
        sa = [slice(None)] * 3
        sb = [slice(None)] * 3
        sa[axis] = slice(0, -1)
        sb[axis] = slice(1, None)
        surface[tuple(sa)] |= surf
        surface[tuple(sb)] |= surf
        separator[tuple(sa)] |= sep
        separator[tuple(sb)] |= sep
    return surface, separator


def _soft_interface_target(
    interface: np.ndarray,
    spacing_um: np.ndarray,
    sigma_um: float,
) -> np.ndarray:
    """Turn a one-voxel interface into a physically isotropic soft band."""
    if not interface.any():
        return np.zeros(interface.shape, dtype=np.float32)
    distance_um = ndi.distance_transform_edt(
        ~interface, sampling=spacing_um
    ).astype(np.float32)
    return np.exp(-0.5 * np.square(distance_um / max(sigma_um, 1e-6))).astype(
        np.float32
    )


def _face_centered_soft_interface_target(
    labels: np.ndarray,
    spacing_um: np.ndarray,
    sigma_um: float,
    *,
    cell_cell: bool,
) -> np.ndarray:
    """Build a soft interface band from physical voxel-face locations.

    Each axis is temporarily sampled at half its native pitch so the face
    between two differing labels lies on an actual grid point. Distances are
    evaluated there and sampled back only at native voxel centers. Processing
    one doubled axis at a time avoids an eightfold full half-grid volume.
    """
    labels = np.asarray(labels)
    spacing_um = np.asarray(spacing_um, dtype=np.float64)
    nearest_um = np.full(labels.shape, np.inf, dtype=np.float32)
    found_interface = False
    for axis in range(3):
        lower_index = [slice(None)] * 3
        upper_index = [slice(None)] * 3
        lower_index[axis] = slice(0, -1)
        upper_index[axis] = slice(1, None)
        lower = labels[tuple(lower_index)]
        upper = labels[tuple(upper_index)]
        if cell_cell:
            faces = (lower > 0) & (upper > 0) & (lower != upper)
        else:
            faces = (lower != upper) & ((lower == 0) ^ (upper == 0))
        if not faces.any():
            continue
        found_interface = True
        half_shape = list(labels.shape)
        half_shape[axis] = max(2 * labels.shape[axis] - 1, 1)
        face_grid = np.zeros(half_shape, dtype=bool)
        face_index = [slice(None)] * 3
        face_index[axis] = slice(1, None, 2)
        face_grid[tuple(face_index)] = faces
        half_spacing = spacing_um.copy()
        half_spacing[axis] *= 0.5
        distance_half = ndi.distance_transform_edt(
            ~face_grid, sampling=half_spacing
        ).astype(np.float32)
        center_index = [slice(None)] * 3
        center_index[axis] = slice(0, None, 2)
        np.minimum(nearest_um, distance_half[tuple(center_index)], out=nearest_um)
    if not found_interface:
        return np.zeros(labels.shape, dtype=np.float32)
    return np.exp(
        -0.5 * np.square(nearest_um / max(float(sigma_um), 1e-6))
    ).astype(np.float32)


def _single_volume_targets(
    labels: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    sdf_clip_dref: float,
    sdf_supervision_radius_dref: float,
    surface_target_sigma_um: float,
    separator_target_sigma_um: float,
) -> dict[str, np.ndarray]:
    labels = labels.astype(np.int64, copy=False)
    shape = labels.shape
    fg = labels > 0
    surface = _face_centered_soft_interface_target(
        labels,
        spacing_um,
        surface_target_sigma_um,
        cell_cell=False,
    )
    separator = _face_centered_soft_interface_target(
        labels,
        spacing_um,
        separator_target_sigma_um,
        cell_cell=True,
    )
    sdf_um = np.zeros(shape, np.float32)
    flow = np.zeros((3, *shape), np.float32)
    offsets = np.zeros((3, *shape), np.float32)
    seed = np.zeros(shape, np.float32)

    # Negative background distance gives the regression a meaningful sign and
    # makes zero a true object surface. Clip locally to avoid background scale
    # dominating the target.
    if (~fg).any():
        bg_dist = ndi.distance_transform_edt(~fg, sampling=spacing_um).astype(np.float32)
        sdf_um[~fg] = -bg_dist[~fg]

    # ``find_objects`` locates each cell once. All per-cell EDT, gradient, and
    # coordinate work then stays inside a small padded box instead of scanning
    # the native scene once per instance.
    object_slices = ndi.find_objects(labels)
    for instance_id, bbox in enumerate(object_slices, 1):
        if bbox is None:
            continue
        region = tuple(
            slice(max(int(axis.start) - 1, 0), min(int(axis.stop) + 1, shape[i]))
            for i, axis in enumerate(bbox)
        )
        local_mask = labels[region] == instance_id
        if not local_mask.any():
            continue
        # A false one-voxel halo gives EDT an explicit exterior even for a
        # tightly cropped object. The retained region includes an additional
        # native neighbor wherever the scene permits, so local gradients match
        # the full-volume field at object voxels.
        padded_mask = np.pad(local_mask, 1, mode="constant", constant_values=False)
        padded_dist = ndi.distance_transform_edt(
            padded_mask, sampling=spacing_um
        ).astype(np.float32)
        inner = tuple(slice(1, -1) for _ in range(3))
        dist = padded_dist[inner]
        sdf_local = sdf_um[region]
        sdf_local[local_mask] = dist[local_mask]

        # Omnipose-style local distance gradient, computed independently per
        # instance so touching labels cannot leak gradients across separators.
        grads = np.gradient(dist, *spacing_um, edge_order=1)
        norm = np.sqrt(sum(g.astype(np.float32) ** 2 for g in grads)) + 1e-6
        local_indices = np.nonzero(local_mask)
        for axis, grad in enumerate(grads):
            component = grad.astype(np.float32) / norm
            flow_region = flow[(axis, *region)]
            flow_region[local_mask] = component[local_mask]

        starts = np.asarray([axis.start for axis in region], dtype=np.float32)
        global_coords = np.stack(local_indices, axis=1).astype(np.float32) + starts
        centroid_vox = global_coords.mean(axis=0)
        centroid_um = centroid_vox * spacing_um
        for axis in range(3):
            offset_region = offsets[(axis, *region)]
            coordinate_um = local_indices[axis].astype(np.float32)
            coordinate_um = (coordinate_um + starts[axis]) * spacing_um[axis]
            offset_region[local_indices] = (
                centroid_um[axis] - coordinate_um
            ) / max(dref_um, 1e-6)

        max_dist = float(dist[local_mask].max())
        if max_dist > 0:
            # Smooth marker target: interior medial locations score highest,
            # rather than forcing a brittle single-voxel center class.
            seed_region = seed[region]
            seed_region[local_mask] = np.clip(
                dist[local_mask] / max_dist, 0.0, 1.0
            )

    sdf_unclipped = sdf_um / max(dref_um, 1e-6)
    # Preserve complete foreground/interior supervision, but exclude distant
    # background using the *unclipped* signed distance. Comparing the clipped
    # tensor to sdf_clip_dref makes every saturated voxel appear valid.
    sdf_valid = fg | (np.abs(sdf_unclipped) <= sdf_supervision_radius_dref)
    sdf = np.clip(sdf_unclipped, -sdf_clip_dref, sdf_clip_dref)
    return {
        "foreground": fg.astype(np.float32)[None],
        "surface": surface[None],
        "separator": separator[None],
        "sdf": sdf.astype(np.float32)[None],
        "sdf_valid": sdf_valid[None],
        "flow": flow,
        "centroid_offset": offsets,
        "seed": seed.astype(np.float32)[None],
    }


def build_geometry_targets(
    instance_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    sdf_clip_dref: float = 2.5,
    sdf_supervision_radius_dref: float = 2.5,
    surface_target_sigma_um: float = 0.75,
    separator_target_sigma_um: float = 0.50,
    device: torch.device | None = None,
) -> GeometryTargets:
    """Build dense research-backed geometry targets from GT instance labels.

    This function is deliberately target-generation code, not part of the
    differentiable model. It uses exact physical sampling in SciPy EDT.
    """
    if instance_labels.ndim == 3:
        instance_labels = instance_labels[None]
    if spacing_um.ndim == 1:
        spacing_um = spacing_um[None]
    if dref_um.ndim == 0:
        dref_um = dref_um[None]
    batches = []
    labels_cpu = instance_labels.detach().cpu().numpy()
    spacing_cpu = spacing_um.detach().cpu().numpy()
    dref_cpu = dref_um.detach().cpu().numpy()
    for b in range(instance_labels.shape[0]):
        batches.append(
            _single_volume_targets(
                labels_cpu[b],
                spacing_cpu[b],
                float(dref_cpu[b]),
                sdf_clip_dref,
                sdf_supervision_radius_dref,
                surface_target_sigma_um,
                separator_target_sigma_um,
            )
        )
    target_device = device or instance_labels.device
    fields = {}
    for key in batches[0].keys():
        dtype = torch.bool if key == "sdf_valid" else torch.float32
        fields[key] = torch.from_numpy(np.stack([x[key] for x in batches])).to(
            device=target_device, dtype=dtype
        )
    return GeometryTargets(**fields)
