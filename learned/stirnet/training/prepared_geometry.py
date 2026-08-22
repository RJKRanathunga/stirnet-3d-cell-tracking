from __future__ import annotations

"""Reusable static GT geometry plus dynamic source-conditioned separator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..model.geometry.edt_backend import geometry_edt_backend, release_cupy_memory
from ..model.geometry.targets import (
    GeometryTargets,
    build_geometry_targets,
    build_source_conditioned_separator_target,
)

PREPARED_STATIC_GEOMETRY_VERSION = 1


@dataclass(frozen=True)
class PreparedStaticGeometry:
    targets: GeometryTargets
    metadata: dict[str, Any]


def _geometry_kwargs(cfg) -> dict[str, Any]:
    return {
        "sdf_clip_dref": float(cfg.sdf_clip_dref),
        "sdf_supervision_radius_dref": float(cfg.sdf_supervision_radius_dref),
        "surface_target_sigma_um": float(cfg.surface_target_sigma_um),
        "separator_target_sigma_um": float(cfg.separator_target_sigma_um),
        "separator_source_min_overlap_voxels": int(
            cfg.separator_source_min_overlap_voxels
        ),
        "separator_source_min_gt_fraction": float(
            cfg.separator_source_min_gt_fraction
        ),
    }


def _labels(value: Tensor) -> Tensor:
    value = torch.as_tensor(value).detach().cpu().long()
    return value[None] if value.ndim == 3 else value


def _spacing(value: Tensor) -> Tensor:
    value = torch.as_tensor(value).detach().cpu().float()
    return value[None] if value.ndim == 1 else value


def _dref(value: Tensor) -> Tensor:
    value = torch.as_tensor(value).detach().cpu().float()
    return value.reshape(1) if value.ndim == 0 else value


def build_static_geometry_targets(
    gt_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    geometry_config,
    backend: str = "auto",
    gpu_min_voxels: int = 262_144,
    device: torch.device | str = "cpu",
) -> GeometryTargets:
    """Build immutable GT-only targets; separator is direct GT only."""
    try:
        with geometry_edt_backend(backend, gpu_min_voxels=gpu_min_voxels):
            return build_geometry_targets(
                _labels(gt_labels),
                _spacing(spacing_um),
                _dref(dref_um),
                current_labels=None,
                separator_source_conditioned=False,
                device=torch.device(device),
                **_geometry_kwargs(geometry_config),
            )
    finally:
        release_cupy_memory()


def build_corrective_separator_target(
    gt_labels: Tensor,
    current_labels: Tensor | None,
    spacing_um: Tensor,
    *,
    geometry_config,
    backend: str = "auto",
    gpu_min_voxels: int = 262_144,
    device: torch.device | str = "cpu",
) -> Tensor | None:
    """Build only the source-conditioned corrective separator sheet."""
    if current_labels is None or not bool(
        geometry_config.separator_source_conditioned
    ):
        return None
    gt = _labels(gt_labels)
    current = _labels(current_labels)
    spacing = _spacing(spacing_um)
    if current.shape != gt.shape:
        raise ValueError("current_labels must match gt_labels")
    rows: list[np.ndarray] = []
    try:
        with geometry_edt_backend(backend, gpu_min_voxels=gpu_min_voxels):
            for b in range(gt.shape[0]):
                rows.append(
                    build_source_conditioned_separator_target(
                        gt[b].numpy(),
                        current[b].numpy(),
                        spacing[b].numpy(),
                        float(geometry_config.separator_target_sigma_um),
                        min_overlap_voxels=int(
                            geometry_config.separator_source_min_overlap_voxels
                        ),
                        min_gt_fraction=float(
                            geometry_config.separator_source_min_gt_fraction
                        ),
                    )
                )
    finally:
        release_cupy_memory()
    return torch.from_numpy(np.stack(rows))[:, None].float().to(device)


def compose_geometry_targets(
    static_targets: GeometryTargets,
    corrective_separator: Tensor | None,
) -> GeometryTargets:
    if corrective_separator is None:
        return static_targets
    corrective = corrective_separator.to(
        static_targets.separator.device, dtype=static_targets.separator.dtype
    )
    if corrective.shape != static_targets.separator.shape:
        raise ValueError("corrective separator shape does not match static separator")
    values = dict(static_targets.__dict__)
    values["separator"] = torch.maximum(static_targets.separator, corrective)
    return GeometryTargets(**values)


def compose_source_conditioned_geometry_targets(
    static_targets: GeometryTargets,
    gt_labels: Tensor,
    current_labels: Tensor | None,
    spacing_um: Tensor,
    *,
    geometry_config,
    backend: str = "auto",
    gpu_min_voxels: int = 262_144,
) -> GeometryTargets:
    corrective = build_corrective_separator_target(
        gt_labels,
        current_labels,
        spacing_um,
        geometry_config=geometry_config,
        backend=backend,
        gpu_min_voxels=gpu_min_voxels,
        device=static_targets.separator.device,
    )
    return compose_geometry_targets(static_targets, corrective)


def build_prepared_geometry_targets(
    gt_labels: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
    *,
    current_labels: Tensor | None,
    geometry_config,
    backend: str = "auto",
    gpu_min_voxels: int = 262_144,
    device: torch.device | str = "cpu",
) -> GeometryTargets:
    static = build_static_geometry_targets(
        gt_labels,
        spacing_um,
        dref_um,
        geometry_config=geometry_config,
        backend=backend,
        gpu_min_voxels=gpu_min_voxels,
        device=device,
    )
    return compose_source_conditioned_geometry_targets(
        static,
        gt_labels,
        current_labels,
        spacing_um,
        geometry_config=geometry_config,
        backend=backend,
        gpu_min_voxels=gpu_min_voxels,
    )


def geometry_targets_to_mapping(
    targets: GeometryTargets, *, squeeze_batch: bool = False
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in targets.__dict__.items():
        tensor = value.detach().cpu()
        if squeeze_batch:
            if tensor.shape[0] != 1:
                raise ValueError("squeeze_batch requires exactly one batch row")
            tensor = tensor[0]
        result[name] = tensor
    return result


def geometry_targets_from_mapping(fields: dict[str, Tensor]) -> GeometryTargets:
    required = {
        "foreground", "surface", "separator", "sdf", "sdf_valid",
        "flow", "centroid_offset", "seed",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"geometry target mapping missing fields: {sorted(missing)}")
    return GeometryTargets(**{name: torch.as_tensor(fields[name]) for name in required})


def save_static_geometry_targets(
    path: str | Path,
    targets: GeometryTargets,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": PREPARED_STATIC_GEOMETRY_VERSION,
        "metadata": dict(metadata or {}),
        "targets": {k: v.detach().cpu() for k, v in targets.__dict__.items()},
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_static_geometry_targets(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> PreparedStaticGeometry:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=map_location)
    version = int(payload.get("format_version", -1))
    if version != PREPARED_STATIC_GEOMETRY_VERSION:
        raise ValueError(f"Unsupported static geometry format version: {version}")
    return PreparedStaticGeometry(
        targets=GeometryTargets(**payload["targets"]),
        metadata=dict(payload.get("metadata", {})),
    )


__all__ = [
    "PREPARED_STATIC_GEOMETRY_VERSION",
    "PreparedStaticGeometry",
    "build_corrective_separator_target",
    "build_prepared_geometry_targets",
    "build_static_geometry_targets",
    "compose_geometry_targets",
    "compose_source_conditioned_geometry_targets",
    "geometry_targets_from_mapping",
    "geometry_targets_to_mapping",
    "load_static_geometry_targets",
    "save_static_geometry_targets",
]
