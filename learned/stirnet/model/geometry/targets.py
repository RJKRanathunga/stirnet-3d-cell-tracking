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


def _single_volume_targets(
    labels: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    sdf_clip_dref: float,
) -> dict[str, np.ndarray]:
    labels = labels.astype(np.int64, copy=False)
    shape = labels.shape
    fg = labels > 0
    surface, separator = _boundaries(labels)
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

    coords = np.indices(shape, dtype=np.float32)
    for instance_id in np.unique(labels):
        if instance_id <= 0:
            continue
        mask = labels == instance_id
        if not mask.any():
            continue
        dist = ndi.distance_transform_edt(mask, sampling=spacing_um).astype(np.float32)
        sdf_um[mask] = dist[mask]

        # Omnipose-style local distance gradient, computed independently per
        # instance so touching labels cannot leak gradients across separators.
        grads = np.gradient(dist, *spacing_um, edge_order=1)
        norm = np.sqrt(sum(g.astype(np.float32) ** 2 for g in grads)) + 1e-6
        for axis, grad in enumerate(grads):
            component = grad.astype(np.float32) / norm
            flow[axis, mask] = component[mask]

        vox = np.argwhere(mask).astype(np.float32)
        centroid_vox = vox.mean(axis=0)
        centroid_um = centroid_vox * spacing_um
        for axis in range(3):
            coord_um = coords[axis] * spacing_um[axis]
            offsets[axis, mask] = (centroid_um[axis] - coord_um[mask]) / max(dref_um, 1e-6)

        max_dist = float(dist[mask].max())
        if max_dist > 0:
            # Smooth marker target: interior medial locations score highest,
            # rather than forcing a brittle single-voxel center class.
            seed[mask] = np.clip(dist[mask] / max_dist, 0.0, 1.0)

    sdf = np.clip(sdf_um / max(dref_um, 1e-6), -sdf_clip_dref, sdf_clip_dref)
    return {
        "foreground": fg.astype(np.float32)[None],
        "surface": surface.astype(np.float32)[None],
        "separator": separator.astype(np.float32)[None],
        "sdf": sdf.astype(np.float32)[None],
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
                labels_cpu[b], spacing_cpu[b], float(dref_cpu[b]), sdf_clip_dref
            )
        )
    target_device = device or instance_labels.device
    fields = {}
    for key in batches[0].keys():
        fields[key] = torch.from_numpy(np.stack([x[key] for x in batches])).to(
            device=target_device, dtype=torch.float32
        )
    return GeometryTargets(**fields)
