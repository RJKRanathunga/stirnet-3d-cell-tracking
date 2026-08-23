from __future__ import annotations

"""Exact halo-aware static GT target cache for focused crop training."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..model.geometry.targets import GeometryTargets
from .crops import CropSpec
from .prepared_geometry import build_static_geometry_targets, load_static_geometry_targets, save_static_geometry_targets

STATIC_CROP_CACHE_VERSION = 1


def _cat_targets(rows: list[GeometryTargets]) -> GeometryTargets:
    if not rows:
        raise ValueError("cannot concatenate empty geometry targets")
    return GeometryTargets(**{
        name: torch.cat([getattr(row, name) for row in rows], dim=0)
        for name in rows[0].__dict__
    })


def _crop_targets(targets: GeometryTargets, core_relative):
    return GeometryTargets(**{
        name: value[:, :, core_relative[0], core_relative[1], core_relative[2]]
        for name, value in targets.__dict__.items()
    })


def _expanded_slices(spec: CropSpec, spacing_um: Tensor, halo_um: float):
    spacing = torch.as_tensor(spacing_um).detach().cpu().float().clamp_min(1e-6)
    radius = torch.ceil(float(halo_um) / spacing).long()
    halo, relative = [], []
    for axis in range(3):
        start = max(0, int(spec.slices_zyx[axis].start) - int(radius[axis]))
        stop = min(int(spec.full_shape_zyx[axis]), int(spec.slices_zyx[axis].stop) + int(radius[axis]))
        halo.append(slice(start, stop))
        relative.append(slice(int(spec.slices_zyx[axis].start) - start, int(spec.slices_zyx[axis].stop) - start))
    return tuple(halo), tuple(relative)


def _geometry_signature(cfg):
    names = (
        "sdf_clip_dref", "sdf_supervision_radius_dref", "surface_target_sigma_um",
        "separator_target_sigma_um", "separator_source_min_overlap_voxels",
        "separator_source_min_gt_fraction",
    )
    return {name: getattr(cfg, name) for name in names}


def _source_key(source_batch: dict, gt_labels: Tensor, spec: CropSpec):
    source_ids = source_batch.get("source_ids")
    if source_ids is not None and len(source_ids) > spec.batch_index:
        return str(source_ids[spec.batch_index]), True
    return f"tensor:{tuple(gt_labels.shape)}:{int(gt_labels.data_ptr())}:{spec.batch_index}", False


def _key_payload(source_batch, gt_labels, spec, *, spacing_um, dref_um, geometry_config, halo_um):
    source, stable = _source_key(source_batch, gt_labels, spec)
    payload = {
        "version": STATIC_CROP_CACHE_VERSION,
        "source": source,
        "crop": [[int(axis.start), int(axis.stop)] for axis in spec.slices_zyx],
        "spacing_um": [float(v) for v in torch.as_tensor(spacing_um).detach().cpu()],
        "dref_um": float(torch.as_tensor(dref_um).detach().cpu()),
        "halo_um": float(halo_um),
        "geometry": _geometry_signature(geometry_config),
    }
    return payload, stable


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class StaticCropTargetCache:
    """Small exact CPU LRU with optional persistent exact disk backing."""

    def __init__(self, *, max_memory_entries: int = 4, disk_dir: str | Path | None = None):
        if max_memory_entries < 0:
            raise ValueError("max_memory_entries cannot be negative")
        self.max_memory_entries = int(max_memory_entries)
        self.disk_dir = None if disk_dir is None else Path(disk_dir)
        self._memory: OrderedDict[str, GeometryTargets] = OrderedDict()

    def _remember(self, key: str, value: GeometryTargets) -> None:
        if self.max_memory_entries <= 0:
            return
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)

    def clear_memory(self) -> None:
        self._memory.clear()

    def _disk_path(self, digest: str) -> Path:
        assert self.disk_dir is not None
        return self.disk_dir / digest[:2] / f"{digest}.pt"

    def _build_one(self, gt_labels, spec, *, spacing_um, dref_um, geometry_config, backend, gpu_min_voxels, halo_um):
        halo, relative = _expanded_slices(spec, spacing_um, halo_um)
        static_halo = build_static_geometry_targets(
            gt_labels[spec.batch_index][halo][None],
            torch.as_tensor(spacing_um).reshape(1, 3),
            torch.as_tensor(dref_um).reshape(1),
            geometry_config=geometry_config,
            backend=backend,
            gpu_min_voxels=gpu_min_voxels,
            device=torch.device("cpu"),
        )
        return _crop_targets(static_halo, relative)

    def get_or_build_batch(
        self,
        source_batch: dict,
        gt_labels: Tensor,
        specs: list[CropSpec],
        *,
        spacing_um: Tensor,
        dref_um: Tensor,
        geometry_config,
        backend: str,
        gpu_min_voxels: int,
        halo_um: float,
    ):
        rows = []
        stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0}
        for spec in specs:
            payload, stable = _key_payload(
                source_batch,
                gt_labels,
                spec,
                spacing_um=spacing_um[spec.batch_index],
                dref_um=dref_um[spec.batch_index],
                geometry_config=geometry_config,
                halo_um=halo_um,
            )
            key = _digest(payload)
            cached = self._memory.get(key)
            if cached is not None:
                self._memory.move_to_end(key)
                rows.append(cached)
                stats["memory_hits"] += 1
                continue
            disk_path = self._disk_path(key) if stable and self.disk_dir is not None else None
            if disk_path is not None and disk_path.exists():
                prepared = load_static_geometry_targets(disk_path)
                if prepared.metadata.get("cache_key") == key:
                    cached = prepared.targets
                    self._remember(key, cached)
                    rows.append(cached)
                    stats["disk_hits"] += 1
                    continue
            cached = self._build_one(
                gt_labels,
                spec,
                spacing_um=spacing_um[spec.batch_index],
                dref_um=dref_um[spec.batch_index],
                geometry_config=geometry_config,
                backend=backend,
                gpu_min_voxels=gpu_min_voxels,
                halo_um=halo_um,
            )
            stats["misses"] += 1
            if disk_path is not None:
                save_static_geometry_targets(
                    disk_path,
                    cached,
                    metadata={"cache_key": key, "key_payload": payload},
                )
            self._remember(key, cached)
            rows.append(cached)
        return _cat_targets(rows), stats


__all__ = ["STATIC_CROP_CACHE_VERSION", "StaticCropTargetCache"]
