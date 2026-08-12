from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .query_builder import (
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
)


def native_chunk_coordinates_um(
    spatial_shape: tuple[int, int, int],
    spacing_um: Tensor,
    start: int,
    end: int,
) -> Tensor:
    """Return centered physical coordinates for one contiguous native chunk."""
    z_size, y_size, x_size = spatial_shape
    linear = torch.arange(start, end, device=spacing_um.device, dtype=torch.long)
    z_coord = torch.div(linear, y_size * x_size, rounding_mode="floor")
    remainder = linear.remainder(y_size * x_size)
    y_coord = torch.div(remainder, x_size, rounding_mode="floor")
    x_coord = remainder.remainder(x_size)
    coords = torch.stack([z_coord, y_coord, x_coord], dim=-1).float()
    spacing = spacing_um.float()
    extent = torch.tensor(
        [z_size - 1, y_size - 1, x_size - 1],
        device=spacing.device,
        dtype=torch.float32,
    ) * spacing
    return coords * spacing[None] - 0.5 * extent[None]


def temporal_gaussian_prior_logits(
    coords_um: Tensor,
    refs_um: Tensor,
    dref_um: Tensor,
    *,
    sigma_dref: float,
    inside_logit: float,
    outside_logit: float,
) -> Tensor:
    """Interpolated temporal prior with an explicitly negative far field."""
    delta = coords_um.float()[None] - refs_um.float()[:, None]
    sigma_um = float(sigma_dref) * dref_um.float()
    gaussian = torch.exp(
        -0.5
        * delta.square().sum(dim=-1)
        / sigma_um.clamp_min(1e-6).square()
    )
    return float(outside_logit) + (
        float(inside_logit) - float(outside_logit)
    ) * gaussian


def _dilate_source_masks(
    labels: Tensor,
    source_ids: Tensor,
    spacing_um: Tensor,
    radius_um: Tensor,
) -> Tensor:
    """Axis-wise physical source dilation for a bounded label subvolume."""
    result = torch.zeros(
        (len(source_ids), *labels.shape), device=labels.device, dtype=torch.bool
    )
    radii = torch.ceil(
        radius_um.float() / spacing_um.float().clamp_min(1e-8)
    ).long().clamp_min(0)
    for row, source_id in enumerate(source_ids.tolist()):
        if source_id < 0:
            continue
        mask = (labels == int(source_id)).float()[None, None]
        for axis, radius in enumerate(radii.tolist()):
            if radius <= 0:
                continue
            kernel = [1, 1, 1]
            padding = [0, 0, 0]
            kernel[axis] = 2 * radius + 1
            padding[axis] = radius
            mask = F.max_pool3d(
                mask,
                kernel_size=tuple(kernel),
                stride=1,
                padding=tuple(padding),
            )
        result[row] = mask[0, 0].bool()
    return result


def dilate_source_masks(
    instance_labels: Tensor,
    source_ids: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    radius_dref: float,
) -> Tensor:
    """Dilate selected sources without constructing masks for every source."""
    return _dilate_source_masks(
        instance_labels,
        source_ids,
        spacing_um,
        float(radius_dref) * dref_um,
    )


def source_dilation_support_chunk(
    instance_labels: Tensor,
    source_ids: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    radius_dref: float,
    start: int,
    end: int,
) -> Tensor:
    """Return source+dilation support for one flat chunk with a Z halo.

    Only the selected query sources and the chunk's native Z slab are
    materialized. This avoids a full query-by-native-volume support tensor in
    streamed training.
    """
    if len(source_ids) == 0 or not bool((source_ids >= 0).any()):
        return torch.zeros(
            (len(source_ids), end - start),
            device=instance_labels.device,
            dtype=torch.bool,
        )
    z_size, y_size, x_size = instance_labels.shape
    plane = y_size * x_size
    z0 = start // plane
    z1 = (end - 1) // plane
    radius_um = float(radius_dref) * dref_um.float()
    radius_z = int(
        torch.ceil(radius_um / spacing_um[0].float().clamp_min(1e-8)).item()
    )
    halo_start = max(0, z0 - radius_z)
    halo_end = min(z_size, z1 + radius_z + 1)
    slab = instance_labels[halo_start:halo_end]
    dilated = _dilate_source_masks(
        slab, source_ids, spacing_um, radius_um
    )
    core = dilated[:, z0 - halo_start : z1 - halo_start + 1].flatten(1)
    local_start = start - z0 * plane
    local_end = local_start + (end - start)
    return core[:, local_start:local_end]


def native_query_prior_and_support(
    query_types: Tensor,
    source_instance_ids: Tensor,
    refs_um: Tensor,
    current_label_chunk: Tensor,
    source_dilated_support: Tensor,
    coords_um: Tensor,
    dref_um: Tensor,
    *,
    support_radius_dref: float,
    temporal_sigma_dref: float,
    prior_inside_logit: float,
    prior_outside_logit: float,
) -> tuple[Tensor, Tensor]:
    """Shared native prior/support semantics for training and inference."""
    device = coords_um.device
    query_types = query_types.to(device=device)
    source_instance_ids = source_instance_ids.to(device=device)
    refs_um = refs_um.to(device=device, dtype=torch.float32)
    distance_squared = (coords_um.float()[None] - refs_um[:, None]).square().sum(-1)
    radius_um = float(support_radius_dref) * dref_um.float()
    radial_support = distance_squared <= radius_um.square()
    source_support = source_dilated_support.to(device=device, dtype=torch.bool)

    primary = query_types == QUERY_PRIMARY
    split = query_types == QUERY_SPLIT
    temporal = query_types == QUERY_TEMPORAL
    discovery = query_types == QUERY_DISCOVERY
    support = torch.zeros_like(radial_support)
    support[primary] = radial_support[primary] | source_support[primary]
    support[split] = source_support[split]
    support[temporal | discovery] = radial_support[temporal | discovery]

    prior = torch.zeros_like(distance_squared, dtype=torch.float32)
    if primary.any():
        inside_source = (
            current_label_chunk[None]
            == source_instance_ids[primary, None]
        )
        prior[primary] = torch.where(
            inside_source,
            prior.new_tensor(prior_inside_logit),
            prior.new_tensor(prior_outside_logit),
        )
    if temporal.any():
        prior[temporal] = temporal_gaussian_prior_logits(
            coords_um,
            refs_um[temporal],
            dref_um,
            sigma_dref=temporal_sigma_dref,
            inside_logit=prior_inside_logit,
            outside_logit=prior_outside_logit,
        )
    # Split and discovery priors remain zero: support localizes them, while the
    # learned mask embedding supplies their shape.
    return prior, support


def compose_native_query_logits(
    learned_logits: Tensor,
    query_types: Tensor,
    source_instance_ids: Tensor,
    refs_um: Tensor,
    current_label_chunk: Tensor,
    source_dilated_support: Tensor,
    coords_um: Tensor,
    dref_um: Tensor,
    *,
    support_radius_dref: float,
    temporal_sigma_dref: float,
    prior_inside_logit: float,
    prior_outside_logit: float,
    background_logit: float,
) -> tuple[Tensor, Tensor, Tensor]:
    prior, support = native_query_prior_and_support(
        query_types,
        source_instance_ids,
        refs_um,
        current_label_chunk,
        source_dilated_support,
        coords_um,
        dref_um,
        support_radius_dref=support_radius_dref,
        temporal_sigma_dref=temporal_sigma_dref,
        prior_inside_logit=prior_inside_logit,
        prior_outside_logit=prior_outside_logit,
    )
    combined = (learned_logits.float() + prior).masked_fill(
        ~support, float(background_logit)
    )
    return combined, prior, support
