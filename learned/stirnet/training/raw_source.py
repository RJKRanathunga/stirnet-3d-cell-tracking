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
from ..data.targets import estimate_model_dref_um, make_instance_boundary
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



# ======================================================================================
# Full-volume RAM acceleration cache
# ======================================================================================

SOURCE_RAM_CACHE_VERSION = 1


def _build_source_edt_prior_and_bboxes(
    instance_labels: np.ndarray,
    spacing_um,
    dref_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact uncorrupted source EDT once for the full source volume.

    The EDT definition intentionally matches ``build_source_prior_channels``:
    each source instance is transformed independently inside its own padded
    bounding box, then normalized by dref.
    """
    from scipy import ndimage as ndi

    labels = np.asarray(instance_labels)
    spacing = np.asarray(spacing_um, dtype=np.float64)
    if labels.ndim != 3:
        raise ValueError("instance_labels must be one [Z,Y,X] volume")
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_um must contain three positive values")

    max_label = int(labels.max()) if labels.size else 0
    edt = np.zeros(labels.shape, dtype=np.float32)
    # [label, z0, y0, x0, z1, y1, x1]. -1 means absent.
    bboxes = np.full((max_label + 1, 6), -1, dtype=np.int32)

    for label, bbox in enumerate(ndi.find_objects(labels), 1):
        if bbox is None:
            continue
        local = labels[bbox] == label
        if not local.any():
            continue

        bboxes[label] = np.asarray(
            [
                int(bbox[0].start),
                int(bbox[1].start),
                int(bbox[2].start),
                int(bbox[0].stop),
                int(bbox[1].stop),
                int(bbox[2].stop),
            ],
            dtype=np.int32,
        )

        padded = np.pad(local, 1, mode="constant", constant_values=False)
        distance = ndi.distance_transform_edt(
            padded,
            sampling=spacing,
        )[tuple(slice(1, -1) for _ in range(3))].astype(np.float32)

        view = edt[bbox]
        view[local] = distance[local] / max(float(dref_um), 1e-6)

    return edt, bboxes


def prepare_raw_source_volume_cache(
    source_batch: dict,
    *,
    release_raw_volume: bool = True,
) -> dict:
    """Cache only expensive reusable full-volume source fields in host RAM.

    Cached:
      * normalized raw, float32
      * per-instance source EDT prior, float32
      * tiny per-instance bounding-box table

    Deliberately not cached:
      * foreground (cheap ``labels > 0``)
      * boundary (cheap local differencing)
      * marker (cheap argmax once EDT exists)

    The original uint16 raw tensor may be released after normalization to reduce
    the net RAM increase. The source cache on disk remains unchanged and small.
    """
    full_raw = source_batch.get("raw_volume")
    if full_raw is None:
        raise ValueError("RAM source cache preparation requires raw_volume")
    if source_batch.get("raw_normalization_bounds") is None:
        raise ValueError("RAM source cache preparation requires normalization bounds")
    if source_batch.get("instance_labels") is None:
        raise ValueError("RAM source cache preparation requires instance_labels")

    raw_tensor = torch.as_tensor(full_raw).detach().cpu()
    current_tensor = torch.as_tensor(source_batch["instance_labels"]).detach().cpu()
    bounds = torch.as_tensor(source_batch["raw_normalization_bounds"]).detach().cpu().float()
    spacing = torch.as_tensor(source_batch["spacing_um"]).detach().cpu().float()
    dref = torch.as_tensor(source_batch["dref_um"]).detach().cpu().float()

    if raw_tensor.ndim != 4 or current_tensor.shape != raw_tensor.shape:
        raise ValueError("raw_volume and instance_labels must align as [B,Z,Y,X]")
    if bounds.shape != (raw_tensor.shape[0], 2):
        raise ValueError("raw_normalization_bounds must be [B,2]")

    started = time.perf_counter()
    normalized_rows: list[torch.Tensor] = []
    edt_rows: list[torch.Tensor] = []
    bbox_rows: list[torch.Tensor] = []

    for batch_index in range(raw_tensor.shape[0]):
        raw_np = raw_tensor[batch_index].numpy()
        current_np = current_tensor[batch_index].numpy()

        normalized = normalize_with_percentiles(
            raw_np,
            float(bounds[batch_index, 0]),
            float(bounds[batch_index, 1]),
        )
        edt, bboxes = _build_source_edt_prior_and_bboxes(
            current_np,
            spacing[batch_index].numpy(),
            float(dref[batch_index]),
        )

        normalized_rows.append(
            torch.from_numpy(np.ascontiguousarray(normalized))
        )
        edt_rows.append(torch.from_numpy(np.ascontiguousarray(edt)))
        bbox_rows.append(torch.from_numpy(np.ascontiguousarray(bboxes)))

    normalized_volume = torch.stack(normalized_rows).float()
    edt_volume = torch.stack(edt_rows).float()
    elapsed = time.perf_counter() - started

    gross_cache_bytes = (
        normalized_volume.numel() * normalized_volume.element_size()
        + edt_volume.numel() * edt_volume.element_size()
        + sum(row.numel() * row.element_size() for row in bbox_rows)
    )
    released_raw_bytes = (
        raw_tensor.numel() * raw_tensor.element_size()
        if release_raw_volume
        else 0
    )

    result = dict(source_batch)
    result["raw_normalized_volume"] = normalized_volume
    result["source_edt_prior_volume"] = edt_volume
    result["source_instance_bboxes_zyx"] = tuple(bbox_rows)
    result["source_ram_cache_metadata"] = {
        "format_version": SOURCE_RAM_CACHE_VERSION,
        "enabled": True,
        "prepare_seconds": float(elapsed),
        "gross_cache_bytes": int(gross_cache_bytes),
        "released_raw_bytes": int(released_raw_bytes),
        "net_added_bytes": int(gross_cache_bytes - released_raw_bytes),
        "cached_fields": (
            "raw_normalized_volume",
            "source_edt_prior_volume",
            "source_instance_bboxes_zyx",
        ),
    }
    if release_raw_volume:
        result["raw_volume"] = None
    return result


def _bbox_is_inside_halo(
    bbox_row: np.ndarray,
    halo: tuple[slice, slice, slice],
) -> bool:
    if bbox_row.shape != (6,) or int(bbox_row[0]) < 0:
        return False
    starts = bbox_row[:3]
    stops = bbox_row[3:]
    return all(
        int(starts[axis]) >= int(halo[axis].start)
        and int(stops[axis]) <= int(halo[axis].stop)
        for axis in range(3)
    )


def _recompute_one_label_edt_in_halo(
    labels_halo: np.ndarray,
    source_id: int,
    spacing_um,
    dref_um: float,
) -> tuple[tuple[slice, slice, slice], np.ndarray] | None:
    """Recompute one clipped source instance exactly as the legacy path."""
    from scipy import ndimage as ndi

    mask = labels_halo == int(source_id)
    if not mask.any():
        return None

    coords = np.argwhere(mask)
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    bbox = tuple(
        slice(int(low[axis]), int(high[axis]))
        for axis in range(3)
    )
    local = mask[bbox]
    padded = np.pad(local, 1, mode="constant", constant_values=False)
    distance = ndi.distance_transform_edt(
        padded,
        sampling=np.asarray(spacing_um, dtype=np.float64),
    )[tuple(slice(1, -1) for _ in range(3))].astype(np.float32)
    normalized = distance / max(float(dref_um), 1e-6)
    return bbox, normalized


def _materialize_cached_source_core(
    *,
    raw_norm_halo: np.ndarray,
    current_halo: np.ndarray,
    cached_edt_halo: np.ndarray,
    bbox_table: np.ndarray,
    halo: tuple[slice, slice, slice],
    core_relative: tuple[slice, slice, slice],
    spacing_um,
    dref_um: float,
) -> tuple[np.ndarray, int]:
    """Build exact five-channel core using cached EDT plus rare label repair.

    A source object fully contained by the physical halo has exactly the same
    EDT as the historical per-crop computation, so its cached values are reused.

    If a core-visible source object extends beyond the halo, only that object's
    EDT is recomputed with the historical cropped-halo definition. This keeps
    the model input exact while avoiding a full per-crop EDT rebuild.
    """
    current_core = current_halo[core_relative]
    core_ids = np.unique(current_core)
    core_ids = core_ids[core_ids > 0]

    edt_work = np.asarray(cached_edt_halo, dtype=np.float32)
    owns_edt_copy = False
    recomputed_labels = 0

    for source_id_value in core_ids.tolist():
        source_id = int(source_id_value)
        bbox_row = (
            bbox_table[source_id]
            if 0 <= source_id < len(bbox_table)
            else np.full((6,), -1, dtype=np.int32)
        )
        if _bbox_is_inside_halo(np.asarray(bbox_row), halo):
            continue

        repaired = _recompute_one_label_edt_in_halo(
            current_halo,
            source_id,
            spacing_um,
            dref_um,
        )
        if repaired is None:
            continue
        if not owns_edt_copy:
            edt_work = np.array(edt_work, dtype=np.float32, copy=True)
            owns_edt_copy = True

        local_bbox, normalized = repaired
        local_mask = current_halo[local_bbox] == source_id
        view = edt_work[local_bbox]
        view[local_mask] = normalized[local_mask]
        recomputed_labels += 1

    # Marker semantics remain the same as the legacy halo implementation:
    # one EDT maximum per source object, with NumPy's deterministic first-argmax
    # tie-breaking. We only need labels that appear in the returned core.
    marker_halo = np.zeros(current_halo.shape, dtype=np.float32)
    for source_id_value in core_ids.tolist():
        source_id = int(source_id_value)
        local = current_halo == source_id
        if not local.any():
            continue
        score = np.where(local, edt_work, -np.inf)
        pos = np.unravel_index(int(np.argmax(score)), score.shape)
        marker_halo[pos] = 1.0

    edt_core = np.array(
        edt_work[core_relative],
        dtype=np.float32,
        copy=True,
    )
    # Synthetic source dropout leaves cached EDT values behind at deleted
    # voxels. Masking by the post-dropout labels is exact because source EDTs
    # are instance-local and independent.
    edt_core[current_core <= 0] = 0.0

    boundary_halo = make_instance_boundary(current_halo)

    spatial_core = np.stack(
        [
            np.asarray(raw_norm_halo[core_relative], dtype=np.float32),
            (current_core > 0).astype(np.float32),
            edt_core,
            boundary_halo[core_relative].astype(np.float32),
            marker_halo[core_relative],
        ],
        axis=0,
    )
    return spatial_core, recomputed_labels


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
    """Apply missing-cell corruption, then build all five model channels.

    If ``prepare_raw_source_volume_cache`` was called, normalized raw and source
    EDT are sliced from host RAM. Cheap priors are derived on demand. Rare source
    objects clipped by the physical halo get an exact per-label EDT repair, so
    the accelerated path preserves the historical model inputs.
    """
    has_ram_cache = (
        source_batch.get("raw_normalized_volume") is not None
        and source_batch.get("source_edt_prior_volume") is not None
        and source_batch.get("source_instance_bboxes_zyx") is not None
    )
    if not has_ram_cache and source_batch.get("raw_volume") is None:
        raise ValueError(
            "raw-source materialization requires raw_volume or the full-volume RAM cache"
        )
    if not has_ram_cache and source_batch.get("raw_normalization_bounds") is None:
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

    full_current = torch.as_tensor(source_batch["instance_labels"])
    bounds = (
        None
        if source_batch.get("raw_normalization_bounds") is None
        else torch.as_tensor(source_batch["raw_normalization_bounds"]).float()
    )
    full_raw = (
        None
        if source_batch.get("raw_volume") is None
        else torch.as_tensor(source_batch["raw_volume"])
    )
    full_normalized = (
        None
        if not has_ram_cache
        else torch.as_tensor(source_batch["raw_normalized_volume"])
    )
    full_edt = (
        None
        if not has_ram_cache
        else torch.as_tensor(source_batch["source_edt_prior_volume"])
    )
    bbox_rows = (
        None
        if not has_ram_cache
        else source_batch["source_instance_bboxes_zyx"]
    )

    spatial_rows, current_rows = [], []
    materialization_modes: list[str] = []
    recomputed_edt_label_counts: list[int] = []

    for row, spec in enumerate(changed.specs):
        batch_index = int(spec.batch_index)
        halo, core_relative = _expanded_slices(
            spec.slices_zyx,
            spec.full_shape_zyx,
            source_batch["spacing_um"][batch_index],
            source_halo_um,
        )

        current_halo = (
            full_current[batch_index][halo]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        for source_id in selected[row]:
            current_halo[current_halo == int(source_id)] = 0

        if has_ram_cache:
            raw_norm_halo = (
                full_normalized[batch_index][halo]
                .detach()
                .cpu()
                .numpy()
            )
            cached_edt_halo = (
                full_edt[batch_index][halo]
                .detach()
                .cpu()
                .numpy()
            )
            bbox_table = (
                torch.as_tensor(bbox_rows[batch_index])
                .detach()
                .cpu()
                .numpy()
            )

            spatial_core, recomputed_count = _materialize_cached_source_core(
                raw_norm_halo=raw_norm_halo,
                current_halo=current_halo,
                cached_edt_halo=cached_edt_halo,
                bbox_table=bbox_table,
                halo=halo,
                core_relative=core_relative,
                spacing_um=source_batch["spacing_um"][batch_index]
                .detach()
                .cpu()
                .numpy(),
                dref_um=float(
                    source_batch["dref_um"][batch_index].detach().cpu()
                ),
            )
            materialization_modes.append("ram_cache")
            recomputed_edt_label_counts.append(int(recomputed_count))
        else:
            assert full_raw is not None and bounds is not None
            raw_halo = (
                full_raw[batch_index][halo]
                .detach()
                .cpu()
                .numpy()
            )
            raw_norm = normalize_with_percentiles(
                raw_halo,
                float(bounds[batch_index, 0]),
                float(bounds[batch_index, 1]),
            )
            spatial_halo = build_spatial_channels(
                raw_norm,
                current_halo,
                source_batch["spacing_um"][batch_index]
                .detach()
                .cpu()
                .numpy(),
                float(source_batch["dref_um"][batch_index].detach().cpu()),
                derive_marker=True,
            )
            spatial_core = spatial_halo[
                :,
                core_relative[0],
                core_relative[1],
                core_relative[2],
            ]
            materialization_modes.append("legacy")
            recomputed_edt_label_counts.append(0)

        current_core = current_halo[core_relative]
        spatial_rows.append(
            torch.from_numpy(np.ascontiguousarray(spatial_core))
        )
        current_rows.append(
            torch.from_numpy(np.ascontiguousarray(current_core))
        )

    batch = dict(changed.batch)
    batch["spatial_inputs"] = torch.stack(spatial_rows).float()
    batch["instance_labels"] = torch.stack(current_rows).to(full_current.dtype)
    batch["source_dropout_ids"] = selected
    batch["source_materialization_modes"] = tuple(materialization_modes)
    batch["source_ram_cache_recomputed_edt_label_counts"] = tuple(
        recomputed_edt_label_counts
    )
    return CropBatch(
        batch=batch,
        gt_labels=changed.gt_labels,
        geometry_targets=changed.geometry_targets,
        specs=changed.specs,
    )


__all__ = [
    "RAW_SOURCE_CACHE_VERSION",
    "SOURCE_RAM_CACHE_VERSION",
    "materialize_raw_source_crop_batch",
    "prepare_raw_source_volume_cache",
    "prepare_raw_training_batch",
]
