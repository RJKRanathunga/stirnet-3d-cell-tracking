from __future__ import annotations

"""Raw-volume source preparation for production STIR-Net spatial training."""

from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..data.sample_builder import build_spatial_channels, normalize_with_percentiles, robust_percentiles
from ..data.targets import estimate_model_dref_um
from .crops import CropBatch
from .source_corruption import apply_selected_source_dropout_labels, select_source_instance_dropout_ids

RAW_SOURCE_CACHE_VERSION = 1


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _native_contiguous_array(value: np.ndarray) -> np.ndarray:
    """Return a writable C-contiguous NumPy array with native byte order.

    PyTorch ``torch.from_numpy`` does not accept arrays whose dtype byte order
    differs from the host. TIFF/memmap inputs may be big-endian (for example
    ``>u2``) even on little-endian hosts.

    The dtype conversion below performs the required byte swap while preserving
    numeric values. Native writable C-contiguous arrays are returned without an
    unnecessary copy.
    """
    array = np.asarray(value)
    if array.dtype.isnative and array.flags.c_contiguous and array.flags.writeable:
        return array

    native_dtype = array.dtype.newbyteorder("=")
    return np.array(
        array,
        dtype=native_dtype,
        order="C",
        copy=True,
    )


def _source_id(raw: np.ndarray, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    filename = getattr(raw, "filename", None)
    if filename:
        return str(Path(filename).resolve())
    return f"memory:{id(raw)}:{tuple(raw.shape)}:{raw.dtype}"


def _signature(preprocessing_config, masking_config, *, connectivity: int, raw_low: float, raw_high: float):
    return {
        "preprocessing": asdict(preprocessing_config),
        "masking": asdict(masking_config),
        "connected_component_connectivity": int(connectivity),
        "raw_channel_low_percentile": float(raw_low),
        "raw_channel_high_percentile": float(raw_high),
    }


def _cache_matches(payload, *, source_id: str, raw: np.ndarray, spacing, signature) -> bool:
    return (
        int(payload.get("format_version", -1)) == RAW_SOURCE_CACHE_VERSION
        and payload.get("source_id") == source_id
        and tuple(payload.get("raw_shape", ())) == tuple(raw.shape)
        and str(payload.get("raw_dtype")) == str(raw.dtype)
        and tuple(float(v) for v in payload.get("spacing_um", ())) == tuple(spacing)
        and payload.get("preprocessing_signature") == signature
    )


def prepare_raw_training_batch(
    raw_volume: np.ndarray,
    gt_labels: np.ndarray,
    spacing_um,
    *,
    source_id: str | None = None,
    source_cache_path: str | Path | None = None,
    preprocessing_config=None,
    masking_config=None,
    connected_component_connectivity: int = 1,
    raw_channel_low_percentile: float = 1.0,
    raw_channel_high_percentile: float = 99.8,
) -> dict:
    """Prepare one CPU training frame from raw + GT without full spatial_inputs."""
    raw = np.asarray(raw_volume)
    gt = np.asarray(gt_labels)
    if raw.ndim != 3 or gt.ndim != 3 or raw.shape != gt.shape:
        raise ValueError("raw_volume and gt_labels must align as one [Z,Y,X] frame")
    spacing = tuple(float(v) for v in spacing_um)
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("spacing_um must contain three positive values")
    if connected_component_connectivity not in {1, 2, 3}:
        raise ValueError("connected_component_connectivity must be 1, 2, or 3")
    if not 0 <= raw_channel_low_percentile < raw_channel_high_percentile <= 100:
        raise ValueError("raw-channel percentiles are invalid")

    prep_module = import_module("src.01_preprocessing.pipeline")
    prep_config_module = import_module("src.01_preprocessing.config")
    mask_module = import_module("src.02_masking.pipeline")
    mask_config_module = import_module("src.02_masking.config")
    if preprocessing_config is None:
        preprocessing_config = prep_config_module.PreprocessingConfig(voxel_size_zyx_um=spacing)
    else:
        preprocessing_config = replace(preprocessing_config, voxel_size_zyx_um=spacing)
    if masking_config is None:
        masking_config = mask_config_module.MaskingConfig()

    sid = _source_id(raw, source_id)
    signature = _signature(
        preprocessing_config,
        masking_config,
        connectivity=connected_component_connectivity,
        raw_low=raw_channel_low_percentile,
        raw_high=raw_channel_high_percentile,
    )
    cache_path = None if source_cache_path is None else Path(source_cache_path)
    cached = None
    cache_hit = False
    if cache_path is not None and cache_path.exists():
        candidate = _torch_load(cache_path)
        if _cache_matches(candidate, source_id=sid, raw=raw, spacing=spacing, signature=signature):
            cached = candidate
            cache_hit = True

    timings: dict[str, float] = {}
    if cached is None:
        started = time.perf_counter()
        processed = prep_module.preprocess_volume(raw, config=preprocessing_config, return_diagnostics=False)
        timings["source_preprocess_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        binary = mask_module.create_binary_mask(processed, config=masking_config, return_diagnostics=False)
        timings["source_mask_seconds"] = time.perf_counter() - started
        from scipy import ndimage as ndi
        started = time.perf_counter()
        current, component_count = ndi.label(
            binary,
            structure=ndi.generate_binary_structure(3, connected_component_connectivity),
        )
        current = current.astype(np.int32, copy=False)
        timings["source_connected_components_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        dref_um = float(estimate_model_dref_um(current, spacing))
        timings["source_dref_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        low, high = robust_percentiles(raw, raw_channel_low_percentile, raw_channel_high_percentile)
        timings["raw_channel_percentiles_seconds"] = time.perf_counter() - started
        cached = {
            "format_version": RAW_SOURCE_CACHE_VERSION,
            "source_id": sid,
            "raw_shape": tuple(raw.shape),
            "raw_dtype": str(raw.dtype),
            "spacing_um": spacing,
            "preprocessing_signature": signature,
            "current_labels": torch.from_numpy(current.copy()),
            "raw_normalization_bounds": torch.tensor([low, high], dtype=torch.float32),
            "dref_um": float(dref_um),
            "current_instance_count": int(component_count),
        }
        if cache_path is not None:
            _atomic_torch_save(cache_path, cached)
        del processed, binary
    else:
        bounds = torch.as_tensor(cached["raw_normalization_bounds"])
        low, high = float(bounds[0]), float(bounds[1])
        dref_um = float(cached["dref_um"])
        component_count = int(cached.get("current_instance_count", int(torch.as_tensor(cached["current_labels"]).max())))

    raw_tensor = torch.from_numpy(_native_contiguous_array(raw))
    gt_tensor = torch.from_numpy(np.asarray(gt, dtype=np.int64).copy())
    current_tensor = torch.as_tensor(cached["current_labels"]).to(torch.int32)
    metadata = {
        "raw_source_format_version": RAW_SOURCE_CACHE_VERSION,
        "source_id": sid,
        "source_cache_hit": bool(cache_hit),
        "source_cache_path": None if cache_path is None else str(cache_path),
        "source_current_instance_count": int(component_count),
        "gt_instance_count": int(np.unique(gt[gt > 0]).size),
        "raw_channel_percentiles": [float(low), float(high)],
        "model_dref_um": float(dref_um),
        "model_dref_source": "current_segmentation",
        "preprocessing_signature": signature,
        "timings": timings,
    }
    return {
        "raw_volume": raw_tensor[None],
        "raw_normalization_bounds": torch.tensor([[low, high]], dtype=torch.float32),
        "instance_labels": current_tensor[None],
        "gt_labels": gt_tensor[None],
        "targets": [{"label_map": gt_tensor}],
        "spacing_um": torch.tensor([spacing], dtype=torch.float32),
        "dref_um": torch.tensor([dref_um], dtype=torch.float32),
        "source_ids": (sid,),
        "source_preprocessing_metadata": (metadata,),
    }


def _expanded_slices(core, full_shape, spacing_um, halo_um: float):
    spacing = torch.as_tensor(spacing_um).detach().cpu().float().clamp_min(1e-6)
    radius = torch.ceil(float(halo_um) / spacing).long()
    halo, relative = [], []
    for axis in range(3):
        start = max(0, int(core[axis].start) - int(radius[axis]))
        stop = min(int(full_shape[axis]), int(core[axis].stop) + int(radius[axis]))
        halo.append(slice(start, stop))
        relative.append(slice(int(core[axis].start) - start, int(core[axis].stop) - start))
    return tuple(halo), tuple(relative)


def materialize_raw_source_crop_batch(
    source_batch: dict,
    crop: CropBatch,
    *,
    source_halo_um: float,
    dropout_probability: float,
    dropout_max_instances: int,
    dropout_seed: int,
    dropout_min_purity: float,
    dropout_min_gt_coverage: float,
) -> CropBatch:
    """Apply missing-cell corruption, then build all five model channels."""
    if source_batch.get("raw_volume") is None:
        raise ValueError("raw-source materialization requires raw_volume")
    if source_batch.get("raw_normalization_bounds") is None:
        raise ValueError("raw-source materialization requires normalization bounds")
    selected = select_source_instance_dropout_ids(
        crop,
        probability=dropout_probability,
        max_instances=dropout_max_instances,
        seed=dropout_seed,
        min_purity=dropout_min_purity,
        min_gt_coverage=dropout_min_gt_coverage,
    )
    changed = apply_selected_source_dropout_labels(crop, selected)
    full_raw = torch.as_tensor(source_batch["raw_volume"])
    full_current = torch.as_tensor(source_batch["instance_labels"])
    bounds = torch.as_tensor(source_batch["raw_normalization_bounds"]).float()
    spatial_rows, current_rows = [], []
    for row, spec in enumerate(changed.specs):
        batch_index = int(spec.batch_index)
        halo, core_relative = _expanded_slices(
            spec.slices_zyx,
            spec.full_shape_zyx,
            source_batch["spacing_um"][batch_index],
            source_halo_um,
        )
        raw_halo = full_raw[batch_index][halo].detach().cpu().numpy()
        current_halo = full_current[batch_index][halo].detach().cpu().numpy().copy()
        for source_id in selected[row]:
            current_halo[current_halo == int(source_id)] = 0
        raw_norm = normalize_with_percentiles(
            raw_halo,
            float(bounds[batch_index, 0]),
            float(bounds[batch_index, 1]),
        )
        spatial_halo = build_spatial_channels(
            raw_norm,
            current_halo,
            source_batch["spacing_um"][batch_index].detach().cpu().numpy(),
            float(source_batch["dref_um"][batch_index].detach().cpu()),
            derive_marker=True,
        )
        spatial_core = spatial_halo[:, core_relative[0], core_relative[1], core_relative[2]]
        current_core = current_halo[core_relative]
        spatial_rows.append(torch.from_numpy(np.ascontiguousarray(spatial_core)))
        current_rows.append(torch.from_numpy(np.ascontiguousarray(current_core)))
    batch = dict(changed.batch)
    batch["spatial_inputs"] = torch.stack(spatial_rows).float()
    batch["instance_labels"] = torch.stack(current_rows).to(full_current.dtype)
    batch["source_dropout_ids"] = selected
    return CropBatch(batch=batch, gt_labels=changed.gt_labels, geometry_targets=changed.geometry_targets, specs=changed.specs)


__all__ = ["RAW_SOURCE_CACHE_VERSION", "materialize_raw_source_crop_batch", "prepare_raw_training_batch"]
