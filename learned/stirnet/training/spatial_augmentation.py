from __future__ import annotations

"""Deterministic crop-level XY reflection augmentation for STIR-Net training."""

from typing import Any

import torch
from torch import Tensor

from ..model.geometry.targets import GeometryTargets
from .crops import CropBatch

XY_IDENTITY = 0
XY_FLIP_X = 1
XY_FLIP_Y = 2
XY_FLIP_XY = XY_FLIP_X | XY_FLIP_Y


def sample_xy_flip_codes(
    batch_size: int,
    *,
    probability: float,
    seed: int,
) -> Tensor:
    """Sample independent X/Y reflections deterministically on CPU."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if seed < 0:
        raise ValueError("seed cannot be negative")

    if probability <= 0:
        return torch.zeros(batch_size, dtype=torch.uint8)
    if probability >= 1:
        return torch.full((batch_size,), XY_FLIP_XY, dtype=torch.uint8)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    draws = torch.rand((batch_size, 2), generator=generator)

    flip_x = draws[:, 0] < float(probability)
    flip_y = draws[:, 1] < float(probability)
    return (
        flip_x.to(torch.uint8) * XY_FLIP_X
        + flip_y.to(torch.uint8) * XY_FLIP_Y
    )


def _validate_codes(codes: Tensor, batch_size: int) -> Tensor:
    values = torch.as_tensor(codes).detach().cpu().to(torch.uint8).reshape(-1)
    if values.numel() != batch_size:
        raise ValueError(
            f"Expected {batch_size} XY transform codes, got {values.numel()}"
        )
    if bool((values > XY_FLIP_XY).any()):
        raise ValueError("XY transform codes must be in {0,1,2,3}")
    return values


def _flip_scalar_rows(value: Tensor, codes: Tensor) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim < 3:
        raise ValueError("Spatial tensors must have at least [B,Y,X] dimensions")
    if tensor.shape[0] != codes.numel():
        raise ValueError("Tensor batch dimension does not match transform codes")

    rows = []
    for row, code_value in enumerate(codes.tolist()):
        item = tensor[row]
        dims = []
        if int(code_value) & XY_FLIP_Y:
            dims.append(-2)
        if int(code_value) & XY_FLIP_X:
            dims.append(-1)
        if dims:
            item = torch.flip(item, dims=dims)
        rows.append(item)
    return torch.stack(rows, dim=0)


def _flip_vector_rows(value: Tensor, codes: Tensor) -> Tensor:
    """Reflect [B,3,Z,Y,X] vectors with component order [z,y,x]."""
    tensor = torch.as_tensor(value)
    if tensor.ndim != 5 or tensor.shape[1] != 3:
        raise ValueError("Vector targets must have shape [B,3,Z,Y,X]")

    result = _flip_scalar_rows(tensor, codes)
    for row, code_value in enumerate(codes.tolist()):
        code = int(code_value)
        if code & XY_FLIP_X:
            result[row, 2] = -result[row, 2]
        if code & XY_FLIP_Y:
            result[row, 1] = -result[row, 1]
    return result


def _transform_geometry_targets(
    targets: GeometryTargets,
    codes: Tensor,
) -> GeometryTargets:
    return GeometryTargets(
        foreground=_flip_scalar_rows(targets.foreground, codes),
        surface=_flip_scalar_rows(targets.surface, codes),
        separator=_flip_scalar_rows(targets.separator, codes),
        sdf=_flip_scalar_rows(targets.sdf, codes),
        sdf_valid=_flip_scalar_rows(targets.sdf_valid, codes),
        flow=_flip_vector_rows(targets.flow, codes),
        centroid_offset=_flip_vector_rows(targets.centroid_offset, codes),
        seed=_flip_scalar_rows(targets.seed, codes),
    )


def apply_xy_flip_codes(
    crop: CropBatch,
    geometry_targets: GeometryTargets,
    codes: Tensor,
) -> tuple[CropBatch, GeometryTargets, dict[str, int]]:
    batch_size = int(crop.gt_labels.shape[0])
    values = _validate_codes(codes, batch_size)

    batch: dict[str, Any] = dict(crop.batch)
    for key in (
        "spatial_inputs",
        "instance_labels",
        "spatial_padding_mask",
        "supervision_valid_mask",
    ):
        if batch.get(key) is not None:
            batch[key] = _flip_scalar_rows(batch[key], values)

    batch["xy_flip_codes"] = values.clone()

    transformed_crop = CropBatch(
        batch=batch,
        gt_labels=_flip_scalar_rows(crop.gt_labels, values),
        geometry_targets=None,
        specs=crop.specs,
    )
    transformed_targets = _transform_geometry_targets(
        geometry_targets,
        values,
    )

    counts = {
        "identity": int((values == XY_IDENTITY).sum()),
        "flip_x": int((values == XY_FLIP_X).sum()),
        "flip_y": int((values == XY_FLIP_Y).sum()),
        "flip_xy": int((values == XY_FLIP_XY).sum()),
    }
    return transformed_crop, transformed_targets, counts


def apply_xy_flip_augmentation(
    crop: CropBatch,
    geometry_targets: GeometryTargets,
    *,
    probability: float,
    seed: int,
) -> tuple[CropBatch, GeometryTargets, dict[str, int]]:
    codes = sample_xy_flip_codes(
        int(crop.gt_labels.shape[0]),
        probability=float(probability),
        seed=int(seed),
    )
    return apply_xy_flip_codes(crop, geometry_targets, codes)


__all__ = [
    "XY_FLIP_X",
    "XY_FLIP_XY",
    "XY_FLIP_Y",
    "XY_IDENTITY",
    "apply_xy_flip_augmentation",
    "apply_xy_flip_codes",
    "sample_xy_flip_codes",
]
