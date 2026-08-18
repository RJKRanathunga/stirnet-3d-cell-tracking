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


# ---------------------------------------------------------------------------
# STIRNET_SOURCE_CONDITIONED_SEPARATOR_V1
# ---------------------------------------------------------------------------

def _source_conditioned_territories(
    gt_labels: np.ndarray,
    current_labels: np.ndarray,
    spacing_um: np.ndarray,
    *,
    min_overlap_voxels: int,
    min_gt_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate GT identities through merge-prone current source components."""
    gt_labels = np.asarray(gt_labels)
    current_labels = np.asarray(current_labels)
    spacing_um = np.asarray(spacing_um, dtype=np.float64)

    if gt_labels.shape != current_labels.shape:
        raise ValueError(
            "GT and current labels must share one native shape; got "
            f"{gt_labels.shape} and {current_labels.shape}"
        )
    if gt_labels.ndim != 3:
        raise ValueError(
            "Source-conditioned separator construction expects one 3-D volume"
        )
    if min_overlap_voxels < 1:
        raise ValueError("min_overlap_voxels must be positive")
    if not 0.0 <= min_gt_fraction <= 1.0:
        raise ValueError("min_gt_fraction must be in [0, 1]")

    territory = np.zeros(gt_labels.shape, dtype=np.int64)
    owner = np.zeros(gt_labels.shape, dtype=np.int32)

    positive_gt = gt_labels[gt_labels > 0]
    if positive_gt.size == 0 or not np.any(current_labels > 0):
        return territory, owner

    gt_ids, gt_counts = np.unique(positive_gt, return_counts=True)
    gt_count_lookup = {
        int(instance_id): int(count)
        for instance_id, count in zip(gt_ids.tolist(), gt_counts.tolist())
    }

    gt_slices = ndi.find_objects(gt_labels)
    source_slices = ndi.find_objects(current_labels)
    connectivity = ndi.generate_binary_structure(3, 1)
    owner_id = 0
    shape = np.asarray(gt_labels.shape, dtype=np.int64)

    for source_id, source_bbox in enumerate(source_slices, 1):
        if source_bbox is None:
            continue

        source_local = current_labels[source_bbox] == source_id
        if not source_local.any():
            continue

        component_labels, component_count = ndi.label(
            source_local, structure=connectivity
        )
        component_slices = ndi.find_objects(component_labels)
        source_origin = np.asarray(
            [int(axis.start) for axis in source_bbox], dtype=np.int64
        )

        for component_id in range(1, component_count + 1):
            component_bbox_local = component_slices[component_id - 1]
            if component_bbox_local is None:
                continue

            component_mask = component_labels[component_bbox_local] == component_id
            component_low = source_origin + np.asarray(
                [int(axis.start) for axis in component_bbox_local], dtype=np.int64
            )
            component_high = source_origin + np.asarray(
                [int(axis.stop) for axis in component_bbox_local], dtype=np.int64
            )
            component_bbox_global = tuple(
                slice(int(lo), int(hi))
                for lo, hi in zip(component_low, component_high)
            )

            gt_component = gt_labels[component_bbox_global]
            overlaps = gt_component[component_mask]
            overlaps = overlaps[overlaps > 0]
            if overlaps.size == 0:
                continue

            candidate_ids, overlap_counts = np.unique(overlaps, return_counts=True)
            meaningful: list[int] = []
            for candidate_id, overlap_count in zip(
                candidate_ids.tolist(), overlap_counts.tolist()
            ):
                gt_count = gt_count_lookup.get(int(candidate_id), 0)
                if gt_count <= 0:
                    continue
                gt_fraction = float(overlap_count) / float(gt_count)
                if (
                    int(overlap_count) >= min_overlap_voxels
                    and gt_fraction >= min_gt_fraction
                ):
                    meaningful.append(int(candidate_id))

            if len(meaningful) < 2:
                continue

            # Include the complete candidate GT cells in the physical-distance ROI.
            roi_low = component_low.copy()
            roi_high = component_high.copy()
            for candidate_id in meaningful:
                index = candidate_id - 1
                if 0 <= index < len(gt_slices):
                    bbox = gt_slices[index]
                    if bbox is not None:
                        roi_low = np.minimum(
                            roi_low,
                            np.asarray(
                                [int(axis.start) for axis in bbox], dtype=np.int64
                            ),
                        )
                        roi_high = np.maximum(
                            roi_high,
                            np.asarray(
                                [int(axis.stop) for axis in bbox], dtype=np.int64
                            ),
                        )

            roi_low = np.maximum(roi_low, 0)
            roi_high = np.minimum(roi_high, shape)
            roi = tuple(
                slice(int(lo), int(hi)) for lo, hi in zip(roi_low, roi_high)
            )

            component_mask_roi = np.zeros(
                tuple(int(hi - lo) for lo, hi in zip(roi_low, roi_high)),
                dtype=bool,
            )
            placement = tuple(
                slice(
                    int(component_low[axis] - roi_low[axis]),
                    int(component_high[axis] - roi_low[axis]),
                )
                for axis in range(3)
            )
            component_mask_roi[placement] = component_mask

            gt_roi = gt_labels[roi]
            candidate_mask = np.isin(
                gt_roi, np.asarray(meaningful, dtype=gt_roi.dtype)
            )
            seed_labels = np.where(candidate_mask, gt_roi, 0)
            if not np.any(seed_labels > 0):
                continue

            _, nearest_indices = ndi.distance_transform_edt(
                seed_labels == 0,
                sampling=spacing_um,
                return_indices=True,
            )
            nearest_gt = seed_labels[tuple(nearest_indices)]

            owner_id += 1
            territory_view = territory[roi]
            owner_view = owner[roi]
            territory_view[component_mask_roi] = nearest_gt[component_mask_roi]
            owner_view[component_mask_roi] = owner_id

    return territory, owner


def _hard_partition_interface(
    territory: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    """Return voxels adjacent to a territory change in one source component."""
    interface = np.zeros(territory.shape, dtype=bool)
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)

        lower_owner = owner[tuple(lower_slice)]
        upper_owner = owner[tuple(upper_slice)]
        lower_territory = territory[tuple(lower_slice)]
        upper_territory = territory[tuple(upper_slice)]

        faces = (
            (lower_owner > 0)
            & (lower_owner == upper_owner)
            & (lower_territory > 0)
            & (upper_territory > 0)
            & (lower_territory != upper_territory)
        )
        interface[tuple(lower_slice)] |= faces
        interface[tuple(upper_slice)] |= faces
    return interface


def _face_centered_soft_partition_target(
    territory: np.ndarray,
    owner: np.ndarray,
    spacing_um: np.ndarray,
    sigma_um: float,
) -> np.ndarray:
    """Soft physical-distance band around territory faces inside merged sources."""
    spacing_um = np.asarray(spacing_um, dtype=np.float64)
    nearest_um = np.full(territory.shape, np.inf, dtype=np.float32)
    found_interface = False

    for axis in range(3):
        lower_index = [slice(None)] * 3
        upper_index = [slice(None)] * 3
        lower_index[axis] = slice(0, -1)
        upper_index[axis] = slice(1, None)

        lower_owner = owner[tuple(lower_index)]
        upper_owner = owner[tuple(upper_index)]
        lower_territory = territory[tuple(lower_index)]
        upper_territory = territory[tuple(upper_index)]

        faces = (
            (lower_owner > 0)
            & (lower_owner == upper_owner)
            & (lower_territory > 0)
            & (upper_territory > 0)
            & (lower_territory != upper_territory)
        )
        if not faces.any():
            continue

        found_interface = True
        half_shape = list(territory.shape)
        half_shape[axis] = max(2 * territory.shape[axis] - 1, 1)
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
        np.minimum(
            nearest_um,
            distance_half[tuple(center_index)],
            out=nearest_um,
        )

    if not found_interface:
        return np.zeros(territory.shape, dtype=np.float32)

    result = np.exp(
        -0.5 * np.square(nearest_um / max(float(sigma_um), 1e-6))
    ).astype(np.float32)
    result[owner <= 0] = 0.0
    return result


def build_source_conditioned_separator_interface(
    gt_labels: np.ndarray,
    current_labels: np.ndarray,
    spacing_um: np.ndarray | tuple[float, float, float],
    *,
    min_overlap_voxels: int = 8,
    min_gt_fraction: float = 0.05,
) -> np.ndarray:
    """Return hard bounded sheets for current components containing 2+ GT cells."""
    territory, owner = _source_conditioned_territories(
        gt_labels,
        current_labels,
        np.asarray(spacing_um, dtype=np.float64),
        min_overlap_voxels=min_overlap_voxels,
        min_gt_fraction=min_gt_fraction,
    )
    return _hard_partition_interface(territory, owner)


def build_source_conditioned_separator_target(
    gt_labels: np.ndarray,
    current_labels: np.ndarray,
    spacing_um: np.ndarray | tuple[float, float, float],
    sigma_um: float,
    *,
    min_overlap_voxels: int = 8,
    min_gt_fraction: float = 0.05,
) -> np.ndarray:
    """Return a soft bounded separator sheet inside under-segmented sources."""
    territory, owner = _source_conditioned_territories(
        gt_labels,
        current_labels,
        np.asarray(spacing_um, dtype=np.float64),
        min_overlap_voxels=min_overlap_voxels,
        min_gt_fraction=min_gt_fraction,
    )
    return _face_centered_soft_partition_target(
        territory,
        owner,
        np.asarray(spacing_um, dtype=np.float64),
        sigma_um,
    )


def build_separator_target(
    gt_labels: np.ndarray,
    spacing_um: np.ndarray,
    sigma_um: float,
    *,
    current_labels: np.ndarray | None = None,
    source_conditioned: bool = True,
    source_min_overlap_voxels: int = 8,
    source_min_gt_fraction: float = 0.05,
) -> np.ndarray:
    """Build direct GT interfaces plus source-conditioned corrective sheets."""
    direct = _face_centered_soft_interface_target(
        gt_labels,
        spacing_um,
        sigma_um,
        cell_cell=True,
    )
    if current_labels is None or not source_conditioned:
        return direct

    corrective = build_source_conditioned_separator_target(
        gt_labels,
        current_labels,
        spacing_um,
        sigma_um,
        min_overlap_voxels=source_min_overlap_voxels,
        min_gt_fraction=source_min_gt_fraction,
    )
    return np.maximum(direct, corrective).astype(np.float32, copy=False)


def _single_volume_targets(
    labels: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    sdf_clip_dref: float,
    sdf_supervision_radius_dref: float,
    surface_target_sigma_um: float,
    separator_target_sigma_um: float,
    current_labels: np.ndarray | None,
    separator_source_conditioned: bool,
    separator_source_min_overlap_voxels: int,
    separator_source_min_gt_fraction: float,
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
    separator = build_separator_target(
        labels,
        spacing_um,
        separator_target_sigma_um,
        current_labels=current_labels,
        source_conditioned=separator_source_conditioned,
        source_min_overlap_voxels=separator_source_min_overlap_voxels,
        source_min_gt_fraction=separator_source_min_gt_fraction,
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
    current_labels: Tensor | None = None,
    sdf_clip_dref: float = 2.5,
    sdf_supervision_radius_dref: float = 2.5,
    surface_target_sigma_um: float = 0.75,
    separator_target_sigma_um: float = 0.50,
    separator_source_conditioned: bool = True,
    separator_source_min_overlap_voxels: int = 8,
    separator_source_min_gt_fraction: float = 0.05,
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
    if current_labels is not None and current_labels.ndim == 3:
        current_labels = current_labels[None]
    if current_labels is not None and current_labels.shape != instance_labels.shape:
        raise ValueError(
            "current_labels must match instance_labels [B,Z,Y,X]; got "
            f"{tuple(current_labels.shape)} and {tuple(instance_labels.shape)}"
        )
    batches = []
    labels_cpu = instance_labels.detach().cpu().numpy()
    current_cpu = (
        None
        if current_labels is None
        else current_labels.detach().cpu().numpy()
    )
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
                None if current_cpu is None else current_cpu[b],
                separator_source_conditioned,
                separator_source_min_overlap_voxels,
                separator_source_min_gt_fraction,
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
