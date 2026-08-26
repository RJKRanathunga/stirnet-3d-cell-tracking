from __future__ import annotations

r"""
STIR-Net Investigation 28 — FULL-VOLUME separator reliability architecture audit.

This replaces the earlier crop-based Investigation 28.

Question
--------
Should separator evidence remain merely one feature in the normal RAG fusion,
or should STIR-Net use a two-stage asymmetric architecture:

    normal full RAG reasoning
            |
            v
    frozen/base p_merge
            |
            + exact separator interface evidence
            |
            v
    separator reconsideration / veto

where:

    MERGE -> SEPARATE is allowed
    SEPARATE -> MERGE is forbidden

The decisive statistics are:

    P(strong separator | GT says separate)
    P(strong separator | GT says same cell)
    P(GT says separate | strong separator)

and, even more directly:

    P(
        GT says separate
        |
        frozen h100 says MERGE
        AND strong separator
    )

Why full volume
---------------
Training crops are useful for optimization but are not ideal for this
architectural audit:

* overlapping crops can count the same biological interface repeatedly;
* the merge-aware crop manifest is not the natural whole-volume edge
  distribution;
* crop truncation removes global cell/RAG context.

This script therefore evaluates each NIS3D volume ONCE and constructs ONE
global watershed/RAG. Each undirected touching supervoxel pair occurs once in
the graph.

GPU memory
----------
Drosophila_1 / Drosophila_2 are too large to place the complete 5-channel input
and complete 11-channel dense geometry on a 6 GiB GPU.

Dense CNN evaluation is therefore tiled, but the ANALYSIS is full-volume:

    CPU full source priors
        -> copy one tile to GPU
        -> geometry forward
        -> blend into temporary full-volume CPU/disk-backed geometry
        -> global production watershed
        -> memory-bounded per-supervoxel production safety guard
        -> second tiled D0 feature reduction
        -> one global RAG
        -> one row per unique valid RAG edge

The resumable geometry store uses float16 and one float32 blend-weight volume.
For Drosophila-sized data it is roughly 5 GiB. It is retained after failures so
a rerun can skip the expensive dense CNN pass, then deleted after a successful
sample unless --keep-dense-cache is supplied. Use --work-dir to point it at
another drive when needed.

The float16 store is ONLY an out-of-core representation between tiled CNN
inference and global graph construction. CNN computation, blend multiplication,
threshold statistics, and final RAG logits use float32 where practical. The
script reports this storage choice in summary.json.

No training is performed.

Default checkpoint
------------------
The frozen morphology-v2 h100 checkpoint:

    runs/stirnet/investigations/
        19_morphology_rag_v2_headroom_training/
        recovery/
        drosophila_12_morphology_rag_v2_headroom_h100/
        checkpoint_step_000600.pt

Typical run
-----------
python .\investigations\stirnet\28_separator_reliability_full_volume_audit.py

One sample first
----------------
python .\investigations\stirnet\28_separator_reliability_full_volume_audit.py `
    --samples Drosophila_1

Outputs
-------
summary.json
architecture_report.txt
edge_observations.csv
threshold_sweep.csv
base_probability_joint.csv
feature_distributions.csv
dangerous_same_cell_strong_separator.csv
veto_candidates.csv
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from tqdm import tqdm


EXPERIMENT_NAME = "28_separator_reliability_full_volume_audit"

DEFAULT_CHECKPOINT = (
    "runs/stirnet/investigations/"
    "19_morphology_rag_v2_headroom_training/"
    "recovery/"
    "drosophila_12_morphology_rag_v2_headroom_h100/"
    "checkpoint_step_000600.pt"
)
DEFAULT_SAMPLES = "Drosophila_1,Drosophila_2"
DEFAULT_SPACING_XYZ = "0.20312639,0.20312639,0.79099447"

DEFAULT_TILE_SHAPE_ZYX = (32, 192, 192)
DEFAULT_TILE_OVERLAP_ZYX = (8, 48, 48)
DEFAULT_TILE_HALO_ZYX = (4, 24, 24)

DEFAULT_MERGE_THRESHOLD = 0.845

DEFAULT_SEPARATOR_MEAN_MIN = 0.55
DEFAULT_SEPARATOR_MAX_MIN = 0.85
DEFAULT_SEPARATOR_COVERAGE70_MIN = 0.25

FEATURE_QUANTILES = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)

BASE_PROBABILITY_BINS = (
    (0.00, 0.10),
    (0.10, 0.50),
    (0.50, 0.70),
    (0.70, 0.845),
    (0.845, 0.95),
    (0.95, 1.000001),
)


# ======================================================================================
# Repository / imports
# ======================================================================================


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
        ):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate repository root. "
        "Run from the cell-tracking repository."
    )


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_support():
    inv26 = _load_module(
        "_stirnet_inv26_for_inv28_full",
        ROOT
        / "investigations"
        / "stirnet"
        / "26_separator_aware_rag_barrier_training.py",
    )
    inv17 = inv26._load_inv17_support()
    return inv26, inv17


# ======================================================================================
# Generic helpers
# ======================================================================================


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return _jsonable(tensor.item())
        return tensor.tolist()
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            text,
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
        ),
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        _atomic_text(path, "")
        return

    fields = list(rows[0].keys())
    known = set(fields)
    extra = sorted(
        {
            key
            for row in rows[1:]
            for key in row
            if key not in known
        }
    )
    fields.extend(extra)

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: _jsonable(value)
                        for key, value in row.items()
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _duration(seconds: float) -> str:
    seconds = max(
        0,
        int(round(seconds)),
    )
    hours, remainder = divmod(
        seconds,
        3600,
    )
    minutes, seconds = divmod(
        remainder,
        60,
    )
    if hours:
        return (
            f"{hours}h {minutes:02d}m "
            f"{seconds:02d}s"
        )
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0
    return float(
        numerator / denominator
    )


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)

    n = float(total)
    p = float(successes) / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (
        p + z2 / (2.0 * n)
    ) / denominator
    radius = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z2 / (4.0 * n * n)
        )
        / denominator
    )
    return (
        max(0.0, center - radius),
        min(1.0, center + radius),
    )


def _parse_zyx(
    value: str,
    *,
    name: str,
) -> tuple[int, int, int]:
    parts = tuple(
        int(token.strip())
        for token in value.split(",")
    )
    if len(parts) != 3:
        raise ValueError(
            f"{name} must be Z,Y,X"
        )
    return parts


def _amp_context():
    if torch.cuda.is_bf16_supported():
        return (
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ),
            "bf16",
        )
    return (
        torch.autocast(
            "cuda",
            dtype=torch.float16,
        ),
        "fp16",
    )


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _close_memmap(value: Any) -> None:
    """Flush and explicitly close a NumPy memmap (important on Windows)."""
    if value is None:
        return
    try:
        value.flush()
    except Exception:
        pass
    mmap_obj = getattr(value, "_mmap", None)
    if mmap_obj is not None:
        try:
            mmap_obj.close()
        except Exception:
            pass


def _safe_rmtree(path: Path) -> None:
    """Best-effort recursive cleanup after all memmap views have been released."""
    if not path.exists():
        return
    gc.collect()
    try:
        shutil.rmtree(path)
    except PermissionError:
        # Windows can hold a just-released mmap briefly.
        time.sleep(0.25)
        gc.collect()
        shutil.rmtree(path)


def _dense_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "geometry": cache_dir / "geometry_f16.dat",
        "weight": cache_dir / "blend_weight_f32.dat",
        "marker": cache_dir / "_DENSE_SUCCESS.json",
        "preliminary": cache_dir / "preliminary_supervoxels_i32.dat",
        "preliminary_marker": cache_dir / "_PRELIMINARY_WATERSHED_SUCCESS.json",
        "final": cache_dir / "supervoxels_i32.dat",
        "final_marker": cache_dir / "_SUPERVOXEL_GUARD_SUCCESS.json",
    }


def _expected_raw_file_size(
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> int:
    return int(np.prod(shape)) * int(np.dtype(dtype).itemsize)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _dense_cache_complete(
    cache_dir: Path,
    *,
    shape_zyx: tuple[int, int, int],
    cache_key: dict[str, Any],
) -> bool:
    paths = _dense_cache_paths(cache_dir)
    marker = _load_json_if_exists(paths["marker"])
    if marker is None or marker.get("cache_key") != _jsonable(cache_key):
        return False

    geometry_bytes = _expected_raw_file_size(
        (11, *shape_zyx), np.dtype(np.float16)
    )
    weight_bytes = _expected_raw_file_size(
        shape_zyx, np.dtype(np.float32)
    )

    return (
        paths["geometry"].is_file()
        and paths["weight"].is_file()
        and paths["geometry"].stat().st_size == geometry_bytes
        and paths["weight"].stat().st_size == weight_bytes
    )


def _label_cache_complete(
    path: Path,
    marker_path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    cache_key: dict[str, Any],
) -> bool:
    marker = _load_json_if_exists(marker_path)
    if marker is None or marker.get("cache_key") != _jsonable(cache_key):
        return False
    expected = _expected_raw_file_size(shape_zyx, np.dtype(np.int32))
    return path.is_file() and path.stat().st_size == expected


# ======================================================================================
# Full source access without a giant 5-channel tensor
# ======================================================================================


def _full_spatial_tile(
    source_batch: dict,
    slices_zyx: tuple[slice, slice, slice],
) -> torch.Tensor:
    """Materialize exactly one full-source five-channel tile in host RAM."""
    current = torch.as_tensor(
        source_batch["instance_labels"]
    )[0][slices_zyx]

    foreground = current > 0

    raw = torch.as_tensor(
        source_batch["raw_normalized_volume"]
    )[0][slices_zyx]

    edt = (
        torch.as_tensor(
            source_batch[
                "source_edt_prior_volume"
            ]
        )[0][slices_zyx]
        * foreground
    )

    boundary = torch.as_tensor(
        source_batch[
            "source_boundary_prior_volume"
        ]
    )[0][slices_zyx].bool()

    marker = (
        torch.as_tensor(
            source_batch[
                "source_marker_prior_volume"
            ]
        )[0][slices_zyx].bool()
        & foreground
    )

    return torch.stack(
        [
            raw.float(),
            foreground.float(),
            edt.float(),
            boundary.float(),
            marker.float(),
        ],
        dim=0,
    )


def _raw_only_full_spatial(
    source_batch: dict,
) -> torch.Tensor:
    """
    RAG morphology and native-field pooling only use spatial_inputs[:, :1].
    Avoid allocating another four full-volume channels.
    """
    raw = torch.as_tensor(
        source_batch[
            "raw_normalized_volume"
        ]
    ).detach().cpu()
    return raw[:, None]


# ======================================================================================
# Out-of-core global dense geometry
# ======================================================================================


def _geometry_storage_bytes(
    shape_zyx: tuple[int, int, int],
) -> int:
    voxels = int(np.prod(shape_zyx))
    # packed 11 x float16 + blend weights float32
    return (
        11 * voxels * np.dtype(np.float16).itemsize
        + voxels * np.dtype(np.float32).itemsize
    )


def _check_work_disk(
    work_root: Path,
    required_bytes: int,
) -> dict[str, float]:
    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    usage = shutil.disk_usage(work_root)

    # Keep 10% headroom above the two temporary files.
    required_with_headroom = int(
        required_bytes * 1.10
    )

    if usage.free < required_with_headroom:
        raise RuntimeError(
            "Insufficient free space for full-volume temporary geometry. "
            f"Need about {required_with_headroom / 2**30:.2f} GiB, "
            f"but only {usage.free / 2**30:.2f} GiB is free at "
            f"{work_root}. Use --work-dir on another drive."
        )

    return {
        "required_gib": (
            required_bytes / 2**30
        ),
        "required_with_headroom_gib": (
            required_with_headroom / 2**30
        ),
        "free_gib_before": (
            usage.free / 2**30
        ),
    }


def _pack_geometry(
    output,
) -> torch.Tensor:
    geometry = output.geometry
    return torch.cat(
        [
            geometry.foreground_logits,
            geometry.surface_logits,
            geometry.separator_logits,
            geometry.sdf,
            geometry.flow,
            geometry.centroid_offset,
            geometry.seed_logits,
        ],
        dim=1,
    )


def _build_inference_config(
    model_cfg,
    *,
    tile_shape: tuple[int, int, int],
    tile_overlap: tuple[int, int, int],
    tile_halo: tuple[int, int, int],
):
    if any(
        value <= 0
        for value in tile_shape
    ):
        raise ValueError(
            "Tile dimensions must be positive"
        )
    if any(
        overlap < 0
        or overlap >= tile
        for overlap, tile in zip(
            tile_overlap,
            tile_shape,
        )
    ):
        raise ValueError(
            "Every overlap must be >=0 and < tile size"
        )
    if any(
        halo < 0
        or 2 * halo >= tile
        for halo, tile in zip(
            tile_halo,
            tile_shape,
        )
    ):
        raise ValueError(
            "Every halo must be >=0 and < half tile size"
        )

    return replace(
        model_cfg.inference,
        mode="tiled",
        tiled_dense_enabled=True,
        tile_shape_zyx=tile_shape,
        tile_overlap_zyx=tile_overlap,
        tile_halo_zyx=tile_halo,
        tile_batch_size=1,
    )


def _compute_global_geometry_to_memmap(
    *,
    model,
    source_batch: dict,
    spacing_cuda: torch.Tensor,
    dref_cuda: torch.Tensor,
    shape_zyx: tuple[int, int, int],
    inference_cfg,
    cache_dir: Path,
    cache_key: dict[str, Any],
    amp_name: str,
) -> tuple[
    np.memmap,
    np.memmap,
    dict[str, Any],
]:
    """Compute or reuse a resumable full-volume dense geometry cache."""
    from learned.stirnet.inference.tiled_dense import (
        generate_dense_tiles,
        tile_blend_weight,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _dense_cache_paths(cache_dir)

    if _dense_cache_complete(
        cache_dir,
        shape_zyx=shape_zyx,
        cache_key=cache_key,
    ):
        print(
            f"[dense cache] reusing completed geometry: {cache_dir}",
            flush=True,
        )
        geometry_mm = np.memmap(
            paths["geometry"],
            mode="r+",
            dtype=np.float16,
            shape=(11, *shape_zyx),
        )
        weight_mm = np.memmap(
            paths["weight"],
            mode="r+",
            dtype=np.float32,
            shape=shape_zyx,
        )
        marker = _load_json_if_exists(paths["marker"]) or {}
        report = dict(marker.get("report", {}))
        report["reused_cache"] = True
        return geometry_mm, weight_mm, report

    # A partial/incompatible dense cache cannot safely be resumed tile-wise.
    # Remove it before starting a clean dense pass.
    for key in (
        "geometry",
        "weight",
        "marker",
        "preliminary",
        "preliminary_marker",
        "final",
        "final_marker",
    ):
        try:
            paths[key].unlink(missing_ok=True)
        except PermissionError:
            gc.collect()
            time.sleep(0.2)
            paths[key].unlink(missing_ok=True)

    geometry_mm = np.memmap(
        paths["geometry"],
        mode="w+",
        dtype=np.float16,
        shape=(11, *shape_zyx),
    )
    weight_mm = np.memmap(
        paths["weight"],
        mode="w+",
        dtype=np.float32,
        shape=shape_zyx,
    )

    geometry_mm[:] = 0
    weight_mm[:] = 0
    geometry_mm.flush()
    weight_mm.flush()

    specs = generate_dense_tiles(
        1,
        shape_zyx,
        inference_cfg,
    )

    started = time.perf_counter()

    progress = tqdm(
        total=len(specs),
        desc="Dense geometry",
        unit="tile",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    try:
        for spec in specs:
            zyx = spec.slices_zyx

            tile_cpu = _full_spatial_tile(
                source_batch,
                zyx,
            )
            tile_cuda = (
                tile_cpu[None]
                .to(
                    "cuda",
                    non_blocking=True,
                )
            )

            amp, _ = _amp_context()
            with torch.inference_mode(), amp:
                output = model(
                    tile_cuda,
                    spacing_cuda,
                    dref_cuda,
                    execution_stage="geometry",
                )

            packed_cuda = (
                _pack_geometry(output)[0]
                .detach()
                .float()
            )

            blend_cuda = tile_blend_weight(
                spec,
                shape_zyx,
                inference_cfg.tile_halo_zyx,
                device=packed_cuda.device,
            ).float()

            weighted = (
                packed_cuda
                * blend_cuda[None]
            ).cpu().numpy()

            blend_np = (
                blend_cuda
                .cpu()
                .numpy()
            )

            target = geometry_mm[
                :,
                zyx[0],
                zyx[1],
                zyx[2],
            ]

            # Arithmetic is float32; only the out-of-core representation is f16.
            target[:] = (
                target.astype(
                    np.float32,
                    copy=False,
                )
                + weighted
            ).astype(
                np.float16,
                copy=False,
            )

            weight_target = weight_mm[
                zyx[0],
                zyx[1],
                zyx[2],
            ]
            weight_target[:] = (
                weight_target
                + blend_np
            )

            progress.set_postfix(
                {
                    "shape": (
                        f"{tile_cpu.shape[-3]}x"
                        f"{tile_cpu.shape[-2]}x"
                        f"{tile_cpu.shape[-1]}"
                    ),
                    "VRAM": (
                        f"{torch.cuda.max_memory_allocated() / 2**30:.2f}G"
                    ),
                },
                refresh=False,
            )
            progress.update(1)

            del (
                output,
                packed_cuda,
                blend_cuda,
                weighted,
                blend_np,
                tile_cuda,
                tile_cpu,
                target,
                weight_target,
            )
            torch.cuda.empty_cache()
    finally:
        progress.close()

    geometry_mm.flush()
    weight_mm.flush()

    # Normalize in small Z slabs so no second full-volume geometry exists.
    slab_depth = max(
        1,
        min(8, shape_zyx[0]),
    )

    normalize_progress = tqdm(
        total=math.ceil(
            shape_zyx[0] / slab_depth
        ),
        desc="Normalize geometry",
        unit="slab",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    try:
        for z0 in range(
            0,
            shape_zyx[0],
            slab_depth,
        ):
            z1 = min(
                shape_zyx[0],
                z0 + slab_depth,
            )

            denominator = np.asarray(
                weight_mm[z0:z1],
                dtype=np.float32,
            )
            denominator = np.maximum(
                denominator,
                1e-8,
            )

            numerator = np.asarray(
                geometry_mm[:, z0:z1],
                dtype=np.float32,
            )

            geometry_mm[:, z0:z1] = (
                numerator
                / denominator[None]
            ).astype(
                np.float16,
                copy=False,
            )

            normalize_progress.update(1)

            del numerator, denominator
    finally:
        normalize_progress.close()

    geometry_mm.flush()
    weight_mm.flush()

    elapsed = (
        time.perf_counter()
        - started
    )

    report = {
        "tile_count": len(specs),
        "seconds": elapsed,
        "amp": amp_name,
        "geometry_storage_dtype": "float16",
        "geometry_file_gib": (
            paths["geometry"].stat().st_size
            / 2**30
        ),
        "weight_file_gib": (
            paths["weight"].stat().st_size
            / 2**30
        ),
        "reused_cache": False,
        "quantization_note": (
            "Tile CNN output and blend multiplication are float32; the "
            "out-of-core accumulated geometry is stored as float16."
        ),
    }

    _atomic_json(
        paths["marker"],
        {
            "cache_key": cache_key,
            "report": report,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    return (
        geometry_mm,
        weight_mm,
        report,
    )

def _geometry_state_from_memmap(
    geometry_mm: np.memmap,
):
    from learned.stirnet.model.types import (
        GeometryState,
    )

    packed = torch.from_numpy(
        geometry_mm
    )

    return GeometryState(
        foreground_logits=(
            packed[0:1][None]
        ),
        surface_logits=(
            packed[1:2][None]
        ),
        separator_logits=(
            packed[2:3][None]
        ),
        sdf=(
            packed[3:4][None]
        ),
        flow=(
            packed[4:7][None]
        ),
        centroid_offset=(
            packed[7:10][None]
        ),
        seed_logits=(
            packed[10:11][None]
        ),
        features=None,
    )


# ======================================================================================
# Global production watershed
# ======================================================================================


class _NoOpSupervoxelSafetyGuard(torch.nn.Module):
    """Temporarily bypass the global-memory guard; local guard is applied next."""

    def forward(
        self,
        labels,
        geometry,
        derived_cache,
        batch_index,
        spacing_um,
        dref_um,
    ):
        return labels


def _global_sdf_positive_max(
    geometry,
    *,
    foreground_threshold: float,
    slab_depth: int = 8,
) -> float:
    """Exact derived-cache SDF normalization denominator using bounded RAM."""
    z_size = int(geometry.sdf.shape[-3])
    maximum = 0.0

    for z0 in range(0, z_size, slab_depth):
        z1 = min(z_size, z0 + slab_depth)

        fg_logits = (
            geometry.foreground_logits[
                0,
                0,
                z0:z1,
            ]
            .float()
        )
        sdf = (
            geometry.sdf[
                0,
                0,
                z0:z1,
            ]
            .float()
        )

        foreground = (
            fg_logits.sigmoid()
            >= float(foreground_threshold)
        )

        if bool(foreground.any()):
            local_max = float(
                sdf.clamp_min(0)[foreground]
                .max()
                .item()
            )
            maximum = max(maximum, local_max)

        del fg_logits, sdf, foreground

    return max(maximum, 1e-6)


def _geometry_crop_numpy(
    tensor: torch.Tensor,
    crop: tuple[slice, slice, slice],
    *,
    sigmoid: bool = False,
) -> np.ndarray:
    value = tensor[
        0,
        ...,
        crop[0],
        crop[1],
        crop[2],
    ].float()
    if sigmoid:
        value = value.sigmoid()
    return np.asarray(
        value.numpy(),
        dtype=np.float32,
    )


def _expanded_box(
    box: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    *,
    halo: int,
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(
            max(0, int(axis.start) - halo),
            min(shape[i], int(axis.stop) + halo),
        )
        for i, axis in enumerate(box)
    )


def _target_face_slice_in_expanded(
    box: tuple[slice, slice, slice],
    expanded: tuple[slice, slice, slice],
    axis: int,
) -> tuple[slice, slice, slice]:
    result: list[slice] = []
    for dim in range(3):
        lo = (
            int(box[dim].start)
            - int(expanded[dim].start)
        )
        hi = (
            int(box[dim].stop)
            - int(expanded[dim].start)
        )
        if dim == axis:
            hi = max(lo, hi - 1)
        result.append(slice(lo, hi))
    return tuple(result)  # type: ignore[return-value]


def _memory_bounded_supervoxel_guard(
    *,
    preliminary: np.memmap,
    output_path: Path,
    geometry,
    spacing_cpu: torch.Tensor,
    dref_cpu: torch.Tensor,
    cfg,
    separator_sigma_um: float,
) -> tuple[np.memmap, dict[str, Any]]:
    """
    Exact per-supervoxel guard semantics with local face-evidence allocation.

    The production guard's decisions are independent per preliminary
    supervoxel. The original implementation first materializes all face
    evidence for the whole volume, then consumes only the faces inside one
    supervoxel at a time. Here we compute that same evidence only in a
    one-voxel halo around each supervoxel bounding box.
    """
    from learned.stirnet.model.partition.supervoxel_guard import (
        build_supervoxel_face_cuts,
        _local_components_with_face_cuts,
    )

    shape = tuple(int(v) for v in preliminary.shape)
    output = np.memmap(
        output_path,
        mode="w+",
        dtype=np.int32,
        shape=shape,
    )
    output[:] = 0
    output.flush()

    preliminary_count = int(preliminary.max())
    if (
        not cfg.supervoxel_guard_enabled
        or preliminary_count <= 0
    ):
        # Chunked copy avoids materializing another full-volume array.
        for z0 in range(0, shape[0], 8):
            z1 = min(shape[0], z0 + 8)
            output[z0:z1] = preliminary[z0:z1]
        output.flush()
        return output, {
            "preliminary_count": preliminary_count,
            "final_count": preliminary_count,
            "split_supervoxel_count": 0,
            "added_supervoxel_count": 0,
            "cut_face_count": 0,
            "suppressed_pathological_split_count": 0,
            "mode": "disabled_copy",
        }

    spacing = (
        spacing_cpu[0]
        .detach()
        .float()
        .numpy()
        .astype(np.float32)
    )
    dref = float(
        dref_cpu[0]
        .detach()
        .float()
        .item()
    )

    print(
        "[watershed] computing global SDF normalization scalar ...",
        flush=True,
    )
    sdf_positive_max = _global_sdf_positive_max(
        geometry,
        foreground_threshold=cfg.foreground_threshold,
    )

    # find_objects scans once and then lets us allocate only local evidence.
    print(
        "[watershed] locating preliminary supervoxel bounding boxes ...",
        flush=True,
    )
    objects = ndi.find_objects(preliminary)

    next_id = 1
    split_count = 0
    suppressed_count = 0
    internal_cut_face_count = 0
    largest_local_voxels = 0

    progress = tqdm(
        total=len(objects),
        desc="Local SV safety guard",
        unit="sv",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    try:
        for old_id, box in enumerate(objects, 1):
            if box is None:
                progress.update(1)
                continue

            local_labels = preliminary[box]
            local_mask = (
                local_labels == old_id
            )
            if not bool(local_mask.any()):
                progress.update(1)
                continue

            expanded = _expanded_box(
                box,
                shape,
                halo=1,
            )

            local_voxels = int(
                np.prod(
                    [
                        int(axis.stop)
                        - int(axis.start)
                        for axis in expanded
                    ]
                )
            )
            largest_local_voxels = max(
                largest_local_voxels,
                local_voxels,
            )

            separator = _geometry_crop_numpy(
                geometry.separator_logits,
                expanded,
                sigmoid=True,
            )[0]
            centroid_offset = _geometry_crop_numpy(
                geometry.centroid_offset,
                expanded,
                sigmoid=False,
            )
            flow = _geometry_crop_numpy(
                geometry.flow,
                expanded,
                sigmoid=False,
            )
            seed = _geometry_crop_numpy(
                geometry.seed_logits,
                expanded,
                sigmoid=True,
            )[0]
            sdf = _geometry_crop_numpy(
                geometry.sdf,
                expanded,
                sigmoid=False,
            )[0]
            sdf_normalized = (
                np.maximum(sdf, 0.0)
                / sdf_positive_max
            ).astype(
                np.float32,
                copy=False,
            )

            cut_faces_expanded, _ = (
                build_supervoxel_face_cuts(
                    separator,
                    centroid_offset,
                    flow,
                    seed,
                    sdf_normalized,
                    spacing,
                    dref,
                    cfg,
                    separator_sigma_um=separator_sigma_um,
                )
            )

            local_cuts: list[np.ndarray] = []
            has_internal_cut = False

            for axis in range(3):
                face_slice = (
                    _target_face_slice_in_expanded(
                        box,
                        expanded,
                        axis,
                    )
                )
                cuts = np.asarray(
                    cut_faces_expanded[axis][face_slice],
                    dtype=bool,
                )
                local_cuts.append(cuts)

                lower = [slice(None)] * 3
                upper = [slice(None)] * 3
                lower[axis] = slice(0, -1)
                upper[axis] = slice(1, None)

                internal = (
                    cuts
                    & local_mask[tuple(lower)]
                    & local_mask[tuple(upper)]
                )
                if bool(internal.any()):
                    has_internal_cut = True
                    internal_cut_face_count += int(
                        internal.sum()
                    )

            target = output[box]

            if not has_internal_cut:
                target[local_mask] = next_id
                output[box] = target
                next_id += 1
            else:
                (
                    components,
                    component_count,
                ) = _local_components_with_face_cuts(
                    local_mask,
                    tuple(local_cuts),  # type: ignore[arg-type]
                )

                if component_count < 2:
                    target[local_mask] = next_id
                    output[box] = target
                    next_id += 1
                else:
                    sizes = np.bincount(
                        components[local_mask].ravel(),
                        minlength=component_count + 1,
                    )[1:]

                    minimum = max(
                        int(
                            cfg.supervoxel_guard_min_fragment_voxels
                        ),
                        int(
                            np.ceil(
                                float(local_mask.sum())
                                * float(
                                    cfg.supervoxel_guard_min_fragment_fraction
                                )
                            )
                        ),
                    )
                    meaningful_count = int(
                        np.sum(
                            sizes >= minimum
                        )
                    )

                    if meaningful_count < 2:
                        target[local_mask] = next_id
                        output[box] = target
                        next_id += 1
                    elif (
                        component_count
                        > cfg.supervoxel_guard_max_fragments
                    ):
                        suppressed_count += 1
                        target[local_mask] = next_id
                        output[box] = target
                        next_id += 1
                    else:
                        split_count += 1
                        for component_id in range(
                            1,
                            component_count + 1,
                        ):
                            target[
                                components == component_id
                            ] = next_id
                            next_id += 1
                        output[box] = target

            if old_id % 32 == 0:
                output.flush()

            progress.set_postfix(
                {
                    "split": split_count,
                    "out": next_id - 1,
                    "bboxM": f"{local_voxels / 1e6:.2f}",
                },
                refresh=False,
            )
            progress.update(1)

            del (
                separator,
                centroid_offset,
                flow,
                seed,
                sdf,
                sdf_normalized,
                cut_faces_expanded,
                local_cuts,
                local_labels,
                local_mask,
                target,
            )

    finally:
        progress.close()

    output.flush()

    final_count = next_id - 1
    return output, {
        "preliminary_count": preliminary_count,
        "final_count": final_count,
        "split_supervoxel_count": split_count,
        "added_supervoxel_count": max(
            final_count - preliminary_count,
            0,
        ),
        "cut_face_count": internal_cut_face_count,
        "suppressed_pathological_split_count": suppressed_count,
        "largest_local_evidence_bbox_voxels": largest_local_voxels,
        "sdf_positive_max": sdf_positive_max,
        "mode": "memory_bounded_per_supervoxel_exact_local_dependency",
        "local_face_halo_voxels": 1,
    }


def _copy_tensor_labels_to_memmap(
    tensor: torch.Tensor,
    path: Path,
) -> np.memmap:
    shape = tuple(int(v) for v in tensor.shape)
    mm = np.memmap(
        path,
        mode="w+",
        dtype=np.int32,
        shape=shape,
    )
    slab = max(1, min(8, shape[0]))
    for z0 in range(0, shape[0], slab):
        z1 = min(shape[0], z0 + slab)
        mm[z0:z1] = (
            tensor[z0:z1]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int32,
                copy=False,
            )
        )
    mm.flush()
    return mm


def _global_watershed(
    *,
    model,
    geometry,
    spacing_cpu: torch.Tensor,
    dref_cpu: torch.Tensor,
    shape_zyx: tuple[int, int, int],
    cache_dir: Path,
    cache_key: dict[str, Any],
) -> tuple[
    torch.Tensor,
    np.memmap,
    dict[str, Any],
]:
    from learned.stirnet.model.geometry.derived import (
        build_geometry_derived_cache,
    )

    paths = _dense_cache_paths(cache_dir)
    started = time.perf_counter()

    # Fastest resume path: guarded supervoxels already completed.
    if _label_cache_complete(
        paths["final"],
        paths["final_marker"],
        shape_zyx=shape_zyx,
        cache_key=cache_key,
    ):
        print(
            "[watershed cache] reusing completed guarded supervoxels",
            flush=True,
        )
        final_mm = np.memmap(
            paths["final"],
            mode="r+",
            dtype=np.int32,
            shape=shape_zyx,
        )
        marker = _load_json_if_exists(
            paths["final_marker"]
        ) or {}
        report = dict(
            marker.get("report", {})
        )
        report["reused_final_cache"] = True
        return (
            torch.from_numpy(final_mm),
            final_mm,
            report,
        )

    preliminary_mm: np.memmap | None = None

    try:
        if _label_cache_complete(
            paths["preliminary"],
            paths["preliminary_marker"],
            shape_zyx=shape_zyx,
            cache_key=cache_key,
        ):
            print(
                "[watershed cache] reusing preliminary global watershed",
                flush=True,
            )
            preliminary_mm = np.memmap(
                paths["preliminary"],
                mode="r+",
                dtype=np.int32,
                shape=shape_zyx,
            )
            preliminary_report = (
                _load_json_if_exists(
                    paths["preliminary_marker"]
                )
                or {}
            ).get("report", {})
        else:
            print(
                "[watershed] building full-volume derived geometry on CPU ...",
                flush=True,
            )
            with torch.inference_mode():
                derived = build_geometry_derived_cache(
                    geometry,
                    model.cfg.partition,
                    padding_mask=None,
                )

            print(
                "[watershed] running global production watershed "
                "(safety guard deferred to bounded local pass) ...",
                flush=True,
            )

            original_guard = (
                model.watershed.safety_guard
            )
            model.watershed.safety_guard = (
                _NoOpSupervoxelSafetyGuard()
            )
            try:
                with torch.inference_mode():
                    preliminary_tensor = model.watershed(
                        geometry,
                        spacing_cpu,
                        dref_cpu,
                        None,
                        derived_cache=derived,
                    )[0].detach().cpu()
            finally:
                model.watershed.safety_guard = (
                    original_guard
                )

            # Derived geometry is the largest RAM consumer. Release it before
            # converting the int64 watershed output to an int32 disk map.
            del derived
            gc.collect()

            preliminary_mm = (
                _copy_tensor_labels_to_memmap(
                    preliminary_tensor,
                    paths["preliminary"],
                )
            )
            preliminary_count = int(
                preliminary_tensor.max().item()
            )
            del preliminary_tensor
            gc.collect()

            preliminary_report = {
                "preliminary_supervoxel_count": preliminary_count,
                "mode": "global_production_watershed_without_guard",
            }
            _atomic_json(
                paths["preliminary_marker"],
                {
                    "cache_key": cache_key,
                    "report": preliminary_report,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        print(
            "[watershed] applying memory-bounded production safety guard ...",
            flush=True,
        )

        final_mm, guard_report = (
            _memory_bounded_supervoxel_guard(
                preliminary=preliminary_mm,
                output_path=paths["final"],
                geometry=geometry,
                spacing_cpu=spacing_cpu,
                dref_cpu=dref_cpu,
                cfg=model.cfg.partition,
                separator_sigma_um=float(
                    model.cfg.geometry.separator_target_sigma_um
                ),
            )
        )

        report = {
            "seconds": (
                time.perf_counter()
                - started
            ),
            "preliminary": preliminary_report,
            "guard": guard_report,
            "supervoxel_count": int(
                guard_report["final_count"]
            ),
            "reused_final_cache": False,
        }

        _atomic_json(
            paths["final_marker"],
            {
                "cache_key": cache_key,
                "report": report,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Final guarded labels supersede the preliminary cache.
        _close_memmap(preliminary_mm)
        preliminary_mm = None
        paths["preliminary"].unlink(missing_ok=True)
        paths["preliminary_marker"].unlink(missing_ok=True)

        return (
            torch.from_numpy(final_mm),
            final_mm,
            report,
        )

    except BaseException:
        # Keep any completed dense/preliminary cache for the next run.
        if preliminary_mm is not None:
            _close_memmap(preliminary_mm)
        raise

# ======================================================================================
# Stream D0 statistics without any global feature pyramid
# ======================================================================================


def _stream_pooled_d0(
    *,
    model,
    source_batch: dict,
    supervoxels: torch.Tensor,
    weight_mm: np.memmap,
    spacing_cuda: torch.Tensor,
    dref_cuda: torch.Tensor,
    inference_cfg,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    from learned.stirnet.inference.tiled_dense import (
        generate_dense_tiles,
        tile_blend_weight,
    )

    shape_zyx = tuple(
        int(value)
        for value in supervoxels.shape
    )
    specs = generate_dense_tiles(
        1,
        shape_zyx,
        inference_cfg,
    )

    max_id = int(
        supervoxels.max().item()
    )
    if max_id <= 0:
        channels = int(
            model.cfg.spatial.channels[0]
        )
        return (
            torch.zeros(
                (0, 2 * channels),
                dtype=torch.float32,
            ),
            {
                "tile_count": len(specs),
                "seconds": 0.0,
            },
        )

    sums = None
    maxima = None
    counts = None

    started = time.perf_counter()

    progress = tqdm(
        total=len(specs),
        desc="Stream D0 stats",
        unit="tile",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    for spec in specs:
        zyx = spec.slices_zyx

        tile_cpu = _full_spatial_tile(
            source_batch,
            zyx,
        )
        tile_cuda = (
            tile_cpu[None]
            .to(
                "cuda",
                non_blocking=True,
            )
        )

        amp, _ = _amp_context()
        with torch.inference_mode(), amp:
            output = model(
                tile_cuda,
                spacing_cuda,
                dref_cuda,
                execution_stage="geometry",
            )

        feature = (
            output.decoded_spatial.d0[0]
        )

        label_crop = (
            supervoxels[zyx]
            .to(
                "cuda",
                non_blocking=True,
            )
        )

        scaled_labels = F.interpolate(
            label_crop.float()[
                None,
                None,
            ],
            size=feature.shape[-3:],
            mode="nearest",
        )[0, 0].long()

        blend_cuda = tile_blend_weight(
            spec,
            shape_zyx,
            inference_cfg.tile_halo_zyx,
            device=feature.device,
        ).float()

        local_weight_sum_np = np.array(
            weight_mm[
                zyx[0],
                zyx[1],
                zyx[2],
            ],
            dtype=np.float32,
            copy=True,
        )

        local_weight_sum = (
            torch.from_numpy(
                local_weight_sum_np
            )
            .to(
                feature.device,
                non_blocking=True,
            )
        )

        normalized_weight = (
            blend_cuda
            / local_weight_sum.clamp_min(
                1e-8
            )
        )

        scaled_weight = F.interpolate(
            normalized_weight[
                None,
                None,
            ],
            size=feature.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )[0, 0]

        if sums is None:
            sums = feature.new_zeros(
                (
                    max_id,
                    feature.shape[0],
                )
            )
            maxima = feature.new_full(
                (
                    max_id,
                    feature.shape[0],
                ),
                -torch.inf,
            )
            counts = feature.new_zeros(
                (max_id,)
            )

        valid = (
            scaled_labels.reshape(-1)
            > 0
        )

        if bool(valid.any()):
            ids = (
                scaled_labels
                .reshape(-1)[valid]
                - 1
            )
            values = (
                feature
                .flatten(1)
                .transpose(0, 1)[valid]
            )
            weights = (
                scaled_weight
                .reshape(-1)[valid]
                .to(values.dtype)
            )

            sums.index_add_(
                0,
                ids,
                values
                * weights[:, None],
            )
            counts.index_add_(
                0,
                ids,
                weights,
            )
            maxima.scatter_reduce_(
                0,
                ids[:, None].expand_as(
                    values
                ),
                values,
                reduce="amax",
                include_self=True,
            )

        progress.update(1)

        del (
            output,
            feature,
            label_crop,
            scaled_labels,
            blend_cuda,
            local_weight_sum_np,
            local_weight_sum,
            normalized_weight,
            scaled_weight,
            tile_cuda,
            tile_cpu,
        )
        torch.cuda.empty_cache()

    progress.close()

    assert sums is not None
    assert maxima is not None
    assert counts is not None

    mean = (
        sums
        / counts.clamp_min(1)[
            :, None
        ]
    )

    maximum = torch.where(
        torch.isfinite(maxima),
        maxima,
        torch.zeros_like(maxima),
    )

    pooled = torch.cat(
        [
            mean,
            maximum,
        ],
        dim=-1,
    ).detach().float().cpu()

    elapsed = (
        time.perf_counter()
        - started
    )

    del (
        sums,
        maxima,
        counts,
        mean,
        maximum,
    )
    torch.cuda.empty_cache()

    return (
        pooled,
        {
            "tile_count": len(specs),
            "seconds": elapsed,
        },
    )


# ======================================================================================
# Global RAG without CPU morphology CNN bottleneck
# ======================================================================================


def _encode_morphology_gpu_streaming(
    *,
    morphology,
    rag_cpu,
    full_raw_spatial_cpu: torch.Tensor,
    geometry,
    spacing_cpu: torch.Tensor,
    dref_cpu: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    """
    Reuse the exact production morphology patch extraction, but keep the huge
    full volume on CPU and send only bounded node/edge patches to the GPU CNN.
    """
    labels = rag_cpu.supervoxel_labels[0]
    start = int(
        rag_cpu.node_offsets[0].item()
    )
    stop = int(
        rag_cpu.node_offsets[1].item()
    )
    node_count = stop - start

    lower, upper = morphology._node_bounds(
        rag_cpu,
        0,
        labels,
        spacing_cpu[0],
    )

    chunk = int(
        morphology.cfg.rag_morphology_chunk_size
    )
    node_outputs: list[torch.Tensor] = []
    edge_outputs: list[torch.Tensor] = []

    started = time.perf_counter()

    node_progress = tqdm(
        total=node_count,
        desc="Morphology nodes",
        unit="node",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    for chunk_start in range(
        0,
        node_count,
        chunk,
    ):
        chunk_stop = min(
            node_count,
            chunk_start + chunk,
        )

        rows = [
            morphology._node_patch(
                labels,
                row + 1,
                lower[row],
                upper[row],
                full_raw_spatial_cpu,
                geometry,
                0,
                spacing_cpu[0],
                dref_cpu[0],
            )
            for row in range(
                chunk_start,
                chunk_stop,
            )
        ]

        stacked = (
            morphology._stack_node_rows(
                rows
            )
        )

        cuda_inputs = tuple(
            value.to(
                "cuda",
                non_blocking=True,
            )
            for value in stacked
        )

        amp, _ = _amp_context()
        with torch.inference_mode(), amp:
            encoded = (
                morphology.node_encoder(
                    *cuda_inputs
                )
            )

        node_outputs.append(
            encoded.detach().float().cpu()
        )

        node_progress.update(
            chunk_stop - chunk_start
        )

        del (
            rows,
            stacked,
            cuda_inputs,
            encoded,
        )
        torch.cuda.empty_cache()

    node_progress.close()

    edge_rows = torch.nonzero(
        rag_cpu.edge_batch == 0,
        as_tuple=False,
    ).flatten()

    local_edges = (
        rag_cpu.edge_index[
            :,
            edge_rows,
        ]
        - start
    )

    edge_count = int(
        local_edges.shape[1]
    )

    edge_progress = tqdm(
        total=edge_count,
        desc="Morphology edges",
        unit="edge",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    for chunk_start in range(
        0,
        edge_count,
        chunk,
    ):
        chunk_stop = min(
            edge_count,
            chunk_start + chunk,
        )

        pair_rows = [
            morphology._edge_pair_patch(
                labels,
                int(
                    local_edges[
                        0,
                        row,
                    ].item()
                ),
                int(
                    local_edges[
                        1,
                        row,
                    ].item()
                ),
                lower,
                upper,
                full_raw_spatial_cpu,
                geometry,
                0,
                spacing_cpu[0],
                dref_cpu[0],
            )
            for row in range(
                chunk_start,
                chunk_stop,
            )
        ]

        broad_cpu = (
            morphology._stack_edge_scale(
                [
                    row.broad
                    for row in pair_rows
                ]
            )
        )
        local_cpu = (
            morphology._stack_edge_scale(
                [
                    row.local
                    for row in pair_rows
                ]
            )
        )

        broad_cuda = tuple(
            value.to(
                "cuda",
                non_blocking=True,
            )
            for value in broad_cpu
        )
        local_cuda = tuple(
            value.to(
                "cuda",
                non_blocking=True,
            )
            for value in local_cpu
        )

        metadata = torch.stack(
            [
                row.metadata
                for row in pair_rows
            ],
            dim=0,
        ).to(
            "cuda",
            dtype=torch.float32,
            non_blocking=True,
        )

        amp, _ = _amp_context()
        with torch.inference_mode(), amp:
            broad_embedding = (
                morphology.edge_encoder(
                    *broad_cuda
                )
            )
            local_embedding = (
                morphology.edge_encoder(
                    *local_cuda
                )
            )
            fused = (
                morphology.edge_scale_fusion(
                    torch.cat(
                        [
                            broad_embedding,
                            local_embedding,
                            metadata.to(
                                dtype=(
                                    broad_embedding
                                    .dtype
                                )
                            ),
                        ],
                        dim=-1,
                    )
                )
            )

        edge_outputs.append(
            fused.detach().float().cpu()
        )

        edge_progress.update(
            chunk_stop - chunk_start
        )

        del (
            pair_rows,
            broad_cpu,
            local_cpu,
            broad_cuda,
            local_cuda,
            metadata,
            broad_embedding,
            local_embedding,
            fused,
        )
        torch.cuda.empty_cache()

    edge_progress.close()

    node_embedding = (
        torch.cat(
            node_outputs,
            dim=0,
        )
        if node_outputs
        else torch.zeros(
            (
                0,
                morphology.cfg.rag_node_morphology_dim,
            ),
            dtype=torch.float32,
        )
    )

    edge_embedding = (
        torch.cat(
            edge_outputs,
            dim=0,
        )
        if edge_outputs
        else torch.zeros(
            (
                0,
                morphology.cfg.rag_edge_morphology_dim,
            ),
            dtype=torch.float32,
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    return (
        node_embedding,
        edge_embedding,
        {
            "seconds": elapsed,
            "node_count": node_count,
            "edge_count": edge_count,
            "chunk_size": chunk,
            "execution": (
                "CPU patch extraction + GPU morphology CNN"
            ),
        },
    )


def _compact_rag_to_cuda(
    rag_cpu,
):
    return replace(
        rag_cpu,
        node_features=(
            rag_cpu.node_features
            .detach()
            .float()
            .to("cuda")
        ),
        node_embeddings=(
            rag_cpu.node_embeddings
            .detach()
            .float()
            .to("cuda")
        ),
        node_batch=(
            rag_cpu.node_batch
            .detach()
            .to("cuda")
        ),
        node_supervoxel_id=(
            rag_cpu.node_supervoxel_id
            .detach()
            .to("cuda")
        ),
        node_centroid_um=(
            rag_cpu.node_centroid_um
            .detach()
            .float()
            .to("cuda")
        ),
        node_volume_voxels=(
            rag_cpu.node_volume_voxels
            .detach()
            .float()
            .to("cuda")
        ),
        edge_index=(
            rag_cpu.edge_index
            .detach()
            .to("cuda")
        ),
        edge_features=(
            rag_cpu.edge_features
            .detach()
            .float()
            .to("cuda")
        ),
        edge_embeddings=(
            rag_cpu.edge_embeddings
            .detach()
            .float()
            .to("cuda")
        ),
        spatial_edge_logits=(
            rag_cpu.spatial_edge_logits
            .detach()
            .float()
            .to("cuda")
        ),
        edge_batch=(
            rag_cpu.edge_batch
            .detach()
            .to("cuda")
        ),
        node_offsets=(
            rag_cpu.node_offsets
            .detach()
            .to("cuda")
        ),
        statistics=None,
        node_morphology_embeddings=(
            None
            if (
                rag_cpu
                .node_morphology_embeddings
                is None
            )
            else (
                rag_cpu
                .node_morphology_embeddings
                .detach()
                .float()
                .to("cuda")
            )
        ),
        edge_morphology_embeddings=(
            None
            if (
                rag_cpu
                .edge_morphology_embeddings
                is None
            )
            else (
                rag_cpu
                .edge_morphology_embeddings
                .detach()
                .float()
                .to("cuda")
            )
        ),
        separator_barrier_features=(
            None
            if (
                rag_cpu
                .separator_barrier_features
                is None
            )
            else (
                rag_cpu
                .separator_barrier_features
                .detach()
                .float()
                .to("cuda")
            )
        ),
        base_spatial_edge_logits=None,
        separator_barrier_score=None,
        separator_barrier_correction=None,
    )


def _build_global_rag_and_base_logits(
    *,
    model,
    model_cfg,
    supervoxels: torch.Tensor,
    pooled_d0_cpu: torch.Tensor,
    full_raw_spatial_cpu: torch.Tensor,
    geometry,
    spacing_cpu: torch.Tensor,
    dref_cpu: torch.Tensor,
    gt_labels_cpu: torch.Tensor,
    valid_mask_cpu: torch.Tensor,
) -> tuple[
    Any,
    Any,
    dict[str, Any],
]:
    from learned.stirnet.model.partition.rag import (
        RAGCriterion,
    )

    builder = model.rag_builder
    morphology = (
        builder.morphology_builder
    )
    if morphology is None:
        raise RuntimeError(
            "h100 audit requires morphology-v2 "
            "to be enabled"
        )

    started = time.perf_counter()

    # Builder's scalar graph construction runs on CPU. Morphology is removed
    # temporarily so the enormous volume is never sent through CPU Conv3d.
    builder.node_projection.cpu()
    builder.morphology_builder = None

    dummy_d0_cpu = torch.zeros(
        (
            1,
            int(
                model.cfg.spatial.channels[0]
            ),
            1,
            1,
            1,
        ),
        dtype=torch.float32,
    )

    print(
        "[rag] building one global CPU RAG and exact separator interface features ...",
        flush=True,
    )

    with torch.inference_mode():
        rag_cpu = builder(
            [supervoxels],
            dummy_d0_cpu,
            full_raw_spatial_cpu,
            geometry,
            spacing_cpu,
            dref_cpu,
            pooled_d0_by_batch=[
                pooled_d0_cpu.float()
            ],
            statistics_by_batch=None,
            derived_cache=None,
            profile_prefix="inv28_full_rag",
        )

    # Restore production module hierarchy.
    builder.morphology_builder = morphology
    builder.node_projection.to("cuda")

    print(
        "[rag] encoding production morphology-v2 patches on GPU ...",
        flush=True,
    )

    (
        node_morphology,
        edge_morphology,
        morphology_report,
    ) = _encode_morphology_gpu_streaming(
        morphology=morphology,
        rag_cpu=rag_cpu,
        full_raw_spatial_cpu=(
            full_raw_spatial_cpu
        ),
        geometry=geometry,
        spacing_cpu=spacing_cpu,
        dref_cpu=dref_cpu,
    )

    rag_cpu = replace(
        rag_cpu,
        node_morphology_embeddings=(
            node_morphology
        ),
        edge_morphology_embeddings=(
            edge_morphology
        ),
    )

    print(
        "[rag] constructing full-volume GT edge targets ...",
        flush=True,
    )

    criterion = RAGCriterion(
        model_cfg.partition
    )

    with torch.inference_mode():
        targets_cpu = (
            criterion.build_targets(
                rag_cpu,
                gt_labels_cpu,
                valid_mask=valid_mask_cpu,
            )
        )

    print(
        "[rag] running frozen h100 graph network on compact global graph ...",
        flush=True,
    )

    rag_cuda = _compact_rag_to_cuda(
        rag_cpu
    )

    amp, amp_name = _amp_context()
    with torch.inference_mode(), amp:
        rag_scored_cuda = (
            model.rag_network(
                rag_cuda
            )
        )

    if (
        rag_scored_cuda
        .base_spatial_edge_logits
        is None
    ):
        raise RuntimeError(
            "Expected base_spatial_edge_logits "
            "from production graph network"
        )

    # Preserve CPU topology/metadata; only copy compact scores back.
    scored_cpu = replace(
        rag_cpu,
        base_spatial_edge_logits=(
            rag_scored_cuda
            .base_spatial_edge_logits
            .detach()
            .float()
            .cpu()
        ),
        spatial_edge_logits=(
            rag_scored_cuda
            .spatial_edge_logits
            .detach()
            .float()
            .cpu()
        ),
        separator_barrier_score=(
            None
            if (
                rag_scored_cuda
                .separator_barrier_score
                is None
            )
            else (
                rag_scored_cuda
                .separator_barrier_score
                .detach()
                .float()
                .cpu()
            )
        ),
        separator_barrier_correction=(
            None
            if (
                rag_scored_cuda
                .separator_barrier_correction
                is None
            )
            else (
                rag_scored_cuda
                .separator_barrier_correction
                .detach()
                .float()
                .cpu()
            )
        ),
    )

    del (
        rag_cuda,
        rag_scored_cuda,
    )
    torch.cuda.empty_cache()

    elapsed = (
        time.perf_counter()
        - started
    )

    return (
        scored_cpu,
        targets_cpu,
        {
            "seconds": elapsed,
            "amp": amp_name,
            "node_count": int(
                rag_cpu.node_features.shape[0]
            ),
            "edge_count": int(
                rag_cpu.edge_index.shape[1]
            ),
            "morphology": morphology_report,
        },
    )


# ======================================================================================
# Edge extraction / statistics
# ======================================================================================


def _extract_edge_rows(
    *,
    sample: str,
    rag,
    targets,
    feature_names: tuple[str, ...],
    merge_threshold: float,
    separator_mean_min: float,
    separator_max_min: float,
    separator_coverage70_min: float,
) -> list[dict[str, Any]]:
    if (
        rag.separator_barrier_features
        is None
    ):
        raise RuntimeError(
            "Global RAG did not retain separator barrier features"
        )
    if (
        rag.base_spatial_edge_logits
        is None
    ):
        raise RuntimeError(
            "Global RAG did not retain base logits"
        )

    valid_rows = torch.nonzero(
        targets.valid.bool(),
        as_tuple=False,
    ).flatten()

    features = (
        rag.separator_barrier_features
        .detach()
        .float()
        .cpu()
    )
    base_p = (
        rag.base_spatial_edge_logits
        .detach()
        .float()
        .sigmoid()
        .cpu()
    )

    edges = (
        rag.edge_index
        .detach()
        .long()
        .cpu()
    )
    node_ids = (
        rag.node_supervoxel_id
        .detach()
        .long()
        .cpu()
    )
    centroids = (
        rag.node_centroid_um
        .detach()
        .float()
        .cpu()
    )
    target = (
        targets.target
        .detach()
        .float()
        .cpu()
    )
    purity = (
        targets.node_purity
        .detach()
        .float()
        .cpu()
    )
    support = (
        targets.node_gt_support
        .detach()
        .float()
        .cpu()
    )
    dominant = (
        targets.dominant_gt
        .detach()
        .long()
        .cpu()
    )

    legacy_edge = (
        rag.edge_features
        .detach()
        .float()
        .cpu()
    )

    rows: list[
        dict[str, Any]
    ] = []

    for edge_row_tensor in valid_rows:
        edge_row = int(
            edge_row_tensor.item()
        )
        src = int(
            edges[
                0,
                edge_row,
            ].item()
        )
        dst = int(
            edges[
                1,
                edge_row,
            ].item()
        )

        values = (
            features[
                edge_row
            ].numpy()
        )
        feature_map = {
            name: float(
                values[index]
            )
            for index, name in enumerate(
                feature_names
            )
        }

        sep_mean = feature_map[
            "separator_mean"
        ]
        sep_max = feature_map[
            "separator_max"
        ]
        coverage70 = feature_map[
            "separator_coverage_070"
        ]

        strong = bool(
            sep_mean
            >= separator_mean_min
            or (
                sep_max
                >= separator_max_min
                and coverage70
                >= separator_coverage70_min
            )
        )

        gt_same = bool(
            float(
                target[
                    edge_row
                ].item()
            )
            >= 0.5
        )

        p_merge = float(
            base_p[
                edge_row
            ].item()
        )

        base_merge = bool(
            p_merge
            >= merge_threshold
        )

        row = {
            "sample": sample,
            "edge_row": edge_row,
            "src_node_index": src,
            "dst_node_index": dst,
            "src_supervoxel_id": int(
                node_ids[src].item()
            ),
            "dst_supervoxel_id": int(
                node_ids[dst].item()
            ),
            "src_dominant_gt": int(
                dominant[src].item()
            ),
            "dst_dominant_gt": int(
                dominant[dst].item()
            ),
            "src_node_purity": float(
                purity[src].item()
            ),
            "dst_node_purity": float(
                purity[dst].item()
            ),
            "src_node_gt_support": float(
                support[src].item()
            ),
            "dst_node_gt_support": float(
                support[dst].item()
            ),
            "src_centroid_z_um": float(
                centroids[
                    src,
                    0,
                ].item()
            ),
            "src_centroid_y_um": float(
                centroids[
                    src,
                    1,
                ].item()
            ),
            "src_centroid_x_um": float(
                centroids[
                    src,
                    2,
                ].item()
            ),
            "dst_centroid_z_um": float(
                centroids[
                    dst,
                    0,
                ].item()
            ),
            "dst_centroid_y_um": float(
                centroids[
                    dst,
                    1,
                ].item()
            ),
            "dst_centroid_x_um": float(
                centroids[
                    dst,
                    2,
                ].item()
            ),
            "gt_same_cell": gt_same,
            "gt_relation": (
                "same_cell_merge_legitimate"
                if gt_same
                else (
                    "different_cells_should_separate"
                )
            ),
            "base_merge_probability": p_merge,
            "base_merge_at_q": base_merge,
            "strong_separator_current_rule": strong,
            "veto_candidate": bool(
                base_merge
                and strong
            ),
            "veto_outcome": (
                "WRONG_SPLIT"
                if (
                    base_merge
                    and strong
                    and gt_same
                )
                else (
                    "CORRECT_SPLIT"
                    if (
                        base_merge
                        and strong
                        and not gt_same
                    )
                    else "NOT_APPLIED"
                )
            ),
            **feature_map,
        }

        for index in range(
            legacy_edge.shape[1]
        ):
            row[
                f"legacy_edge_f{index}"
            ] = float(
                legacy_edge[
                    edge_row,
                    index,
                ].item()
            )

        rows.append(row)

    return rows


def _rule_mask(
    rows: list[dict[str, Any]],
    *,
    rule_type: str,
    mean_min: float | None,
    max_min: float | None,
    coverage70_min: float | None,
) -> np.ndarray:
    sep_mean = np.asarray(
        [
            float(
                row[
                    "separator_mean"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    sep_max = np.asarray(
        [
            float(
                row[
                    "separator_max"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    coverage70 = np.asarray(
        [
            float(
                row[
                    "separator_coverage_070"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    if rule_type == "mean_only":
        assert mean_min is not None
        return (
            sep_mean
            >= mean_min
        )

    if rule_type == "max_and_cov70":
        assert max_min is not None
        assert (
            coverage70_min
            is not None
        )
        return (
            (sep_max >= max_min)
            & (
                coverage70
                >= coverage70_min
            )
        )

    if (
        rule_type
        == "mean_or_max_cov70"
    ):
        assert mean_min is not None
        assert max_min is not None
        assert (
            coverage70_min
            is not None
        )
        return (
            sep_mean >= mean_min
        ) | (
            (sep_max >= max_min)
            & (
                coverage70
                >= coverage70_min
            )
        )

    raise ValueError(
        f"Unknown rule {rule_type}"
    )


def _core_statistics(
    rows: list[dict[str, Any]],
    strong: np.ndarray,
    *,
    merge_threshold: float,
) -> dict[str, Any]:
    gt_same = np.asarray(
        [
            bool(
                row["gt_same_cell"]
            )
            for row in rows
        ],
        dtype=bool,
    )
    gt_separate = ~gt_same

    base_p = np.asarray(
        [
            float(
                row[
                    "base_merge_probability"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    tp = int(
        np.sum(
            strong
            & gt_separate
        )
    )
    fp = int(
        np.sum(
            strong
            & gt_same
        )
    )
    fn = int(
        np.sum(
            (~strong)
            & gt_separate
        )
    )
    tn = int(
        np.sum(
            (~strong)
            & gt_same
        )
    )

    separate_count = int(
        np.sum(gt_separate)
    )
    same_count = int(
        np.sum(gt_same)
    )

    base_merge = (
        base_p
        >= merge_threshold
    )

    base_false_merge = (
        base_merge
        & gt_separate
    )
    base_legitimate_merge = (
        base_merge
        & gt_same
    )

    veto = (
        base_merge
        & strong
    )
    veto_correct = (
        veto
        & gt_separate
    )
    veto_wrong = (
        veto
        & gt_same
    )

    precision = _safe_ratio(
        tp,
        tp + fp,
    )
    recall = _safe_ratio(
        tp,
        separate_count,
    )
    fpr = _safe_ratio(
        fp,
        same_count,
    )

    veto_precision = _safe_ratio(
        int(
            np.sum(veto_correct)
        ),
        int(np.sum(veto)),
    )

    false_merge_recall = (
        _safe_ratio(
            int(
                np.sum(
                    veto_correct
                )
            ),
            int(
                np.sum(
                    base_false_merge
                )
            ),
        )
    )

    damage_rate = _safe_ratio(
        int(
            np.sum(veto_wrong)
        ),
        int(
            np.sum(
                base_legitimate_merge
            )
        ),
    )

    return {
        "edge_count": len(rows),
        "gt_separate_edge_count": (
            separate_count
        ),
        "gt_same_cell_edge_count": (
            same_count
        ),
        "strong_separator_count": int(
            np.sum(strong)
        ),
        "true_separator_positive_count": tp,
        "dangerous_same_cell_strong_separator_count": fp,
        "missed_true_boundary_count": fn,
        "true_separator_negative_count": tn,

        "separator_recall_p_strong_given_gt_separate": (
            recall
        ),
        "separator_recall_95ci": list(
            _wilson_interval(
                tp,
                separate_count,
            )
        ),

        "same_cell_separator_fpr_p_strong_given_gt_same": (
            fpr
        ),
        "same_cell_separator_fpr_95ci": list(
            _wilson_interval(
                fp,
                same_count,
            )
        ),

        "separator_precision_p_gt_separate_given_strong": (
            precision
        ),
        "separator_precision_95ci": list(
            _wilson_interval(
                tp,
                tp + fp,
            )
        ),

        "base_merge_threshold": (
            merge_threshold
        ),
        "base_proposed_merge_count": int(
            np.sum(base_merge)
        ),
        "base_false_merge_count": int(
            np.sum(
                base_false_merge
            )
        ),
        "base_legitimate_merge_count": int(
            np.sum(
                base_legitimate_merge
            )
        ),

        "veto_candidate_count": int(
            np.sum(veto)
        ),
        "veto_correct_split_count": int(
            np.sum(
                veto_correct
            )
        ),
        "veto_wrong_split_count": int(
            np.sum(
                veto_wrong
            )
        ),

        "veto_precision_p_gt_separate_given_base_merge_and_strong": (
            veto_precision
        ),
        "veto_precision_95ci": list(
            _wilson_interval(
                int(
                    np.sum(
                        veto_correct
                    )
                ),
                int(
                    np.sum(veto)
                ),
            )
        ),

        "veto_recall_of_current_false_merges": (
            false_merge_recall
        ),
        "veto_recall_95ci": list(
            _wilson_interval(
                int(
                    np.sum(
                        veto_correct
                    )
                ),
                int(
                    np.sum(
                        base_false_merge
                    )
                ),
            )
        ),

        "veto_damage_rate_among_legitimate_base_merges": (
            damage_rate
        ),
        "veto_damage_rate_95ci": list(
            _wilson_interval(
                int(
                    np.sum(
                        veto_wrong
                    )
                ),
                int(
                    np.sum(
                        base_legitimate_merge
                    )
                ),
            )
        ),
    }


def _threshold_sweep(
    rows: list[dict[str, Any]],
    *,
    merge_threshold: float,
    current_mean: float,
    current_max: float,
    current_coverage70: float,
) -> list[dict[str, Any]]:
    candidates = [
        {
            "rule_name": "CURRENT_PRODUCTION",
            "rule_type": "mean_or_max_cov70",
            "mean_min": current_mean,
            "max_min": current_max,
            "coverage70_min": current_coverage70,
        }
    ]

    for mean_min in (
        0.40,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ):
        candidates.append(
            {
                "rule_name": (
                    f"mean_ge_{mean_min:.2f}"
                ),
                "rule_type": "mean_only",
                "mean_min": mean_min,
                "max_min": None,
                "coverage70_min": None,
            }
        )

    for max_min in (
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
    ):
        for coverage70_min in (
            0.10,
            0.25,
            0.40,
            0.60,
            0.80,
        ):
            candidates.append(
                {
                    "rule_name": (
                        f"max_ge_{max_min:.2f}"
                        f"_cov70_ge_{coverage70_min:.2f}"
                    ),
                    "rule_type": (
                        "max_and_cov70"
                    ),
                    "mean_min": None,
                    "max_min": max_min,
                    "coverage70_min": coverage70_min,
                }
            )

    for mean_min in (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ):
        for max_min in (
            0.80,
            0.85,
            0.90,
            0.95,
        ):
            for coverage70_min in (
                0.10,
                0.25,
                0.40,
                0.60,
            ):
                candidates.append(
                    {
                        "rule_name": (
                            f"mean_ge_{mean_min:.2f}"
                            "_OR_"
                            f"max_ge_{max_min:.2f}"
                            "_cov70_ge_"
                            f"{coverage70_min:.2f}"
                        ),
                        "rule_type": (
                            "mean_or_max_cov70"
                        ),
                        "mean_min": mean_min,
                        "max_min": max_min,
                        "coverage70_min": (
                            coverage70_min
                        ),
                    }
                )

    output = []
    seen = set()

    for candidate in candidates:
        key = (
            candidate["rule_type"],
            candidate["mean_min"],
            candidate["max_min"],
            candidate[
                "coverage70_min"
            ],
        )
        if key in seen:
            continue
        seen.add(key)

        strong = _rule_mask(
            rows,
            rule_type=(
                candidate["rule_type"]
            ),
            mean_min=(
                candidate["mean_min"]
            ),
            max_min=(
                candidate["max_min"]
            ),
            coverage70_min=(
                candidate[
                    "coverage70_min"
                ]
            ),
        )

        stats = _core_statistics(
            rows,
            strong,
            merge_threshold=(
                merge_threshold
            ),
        )

        output.append(
            {
                **candidate,
                "strong_separator_count": (
                    stats[
                        "strong_separator_count"
                    ]
                ),
                "separator_recall": (
                    stats[
                        "separator_recall_p_strong_given_gt_separate"
                    ]
                ),
                "separator_precision": (
                    stats[
                        "separator_precision_p_gt_separate_given_strong"
                    ]
                ),
                "same_cell_separator_fpr": (
                    stats[
                        "same_cell_separator_fpr_p_strong_given_gt_same"
                    ]
                ),
                "same_cell_strong_count": (
                    stats[
                        "dangerous_same_cell_strong_separator_count"
                    ]
                ),
                "base_false_merge_count": (
                    stats[
                        "base_false_merge_count"
                    ]
                ),
                "veto_candidate_count": (
                    stats[
                        "veto_candidate_count"
                    ]
                ),
                "veto_correct_count": (
                    stats[
                        "veto_correct_split_count"
                    ]
                ),
                "veto_wrong_count": (
                    stats[
                        "veto_wrong_split_count"
                    ]
                ),
                "veto_precision": (
                    stats[
                        "veto_precision_p_gt_separate_given_base_merge_and_strong"
                    ]
                ),
                "veto_false_merge_recall": (
                    stats[
                        "veto_recall_of_current_false_merges"
                    ]
                ),
                "veto_legitimate_merge_damage_rate": (
                    stats[
                        "veto_damage_rate_among_legitimate_base_merges"
                    ]
                ),
            }
        )

    return output


def _feature_distributions(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    output = []

    populations = {
        "gt_different_should_separate": [
            row
            for row in rows
            if not row["gt_same_cell"]
        ],
        "gt_same_merge_legitimate": [
            row
            for row in rows
            if row["gt_same_cell"]
        ],
    }

    for population_name, subset in populations.items():
        if not subset:
            continue

        for feature in feature_names:
            values = np.asarray(
                [
                    float(row[feature])
                    for row in subset
                ],
                dtype=np.float64,
            )

            record = {
                "population": population_name,
                "feature": feature,
                "count": int(
                    values.size
                ),
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std()
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                ),
            }

            for quantile in FEATURE_QUANTILES:
                record[
                    f"q{int(round(100 * quantile)):02d}"
                ] = float(
                    np.quantile(
                        values,
                        quantile,
                    )
                )

            output.append(record)

    return output


def _joint_probability_table(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for low, high in BASE_PROBABILITY_BINS:
        for strong in (
            False,
            True,
        ):
            subset = [
                row
                for row in rows
                if (
                    float(
                        row[
                            "base_merge_probability"
                        ]
                    )
                    >= low
                    and float(
                        row[
                            "base_merge_probability"
                        ]
                    )
                    < high
                    and bool(
                        row[
                            "strong_separator_current_rule"
                        ]
                    )
                    == strong
                )
            ]

            separate = sum(
                int(
                    not row[
                        "gt_same_cell"
                    ]
                )
                for row in subset
            )
            same = (
                len(subset)
                - separate
            )

            output.append(
                {
                    "base_probability_low": low,
                    "base_probability_high": (
                        1.0
                        if high > 1.0
                        else high
                    ),
                    "strong_separator": strong,
                    "edge_count": len(
                        subset
                    ),
                    "gt_separate_count": (
                        separate
                    ),
                    "gt_same_count": same,
                    "p_gt_separate": (
                        _safe_ratio(
                            separate,
                            len(subset),
                        )
                    ),
                }
            )

    return output


def _architecture_interpretation(
    stats: dict[str, Any],
) -> dict[str, str]:
    precision = stats[
        "separator_precision_p_gt_separate_given_strong"
    ]
    fpr = stats[
        "same_cell_separator_fpr_p_strong_given_gt_same"
    ]
    veto_precision = stats[
        "veto_precision_p_gt_separate_given_base_merge_and_strong"
    ]
    veto_recall = stats[
        "veto_recall_of_current_false_merges"
    ]

    if (
        precision >= 0.995
        and fpr <= 0.005
        and veto_precision >= 0.995
        and veto_recall >= 0.90
    ):
        return {
            "category": (
                "strong_support_for_privileged_asymmetric_veto"
            ),
            "text": (
                "The current full-volume edge audit strongly supports normal "
                "RAG reasoning first, followed by a separation-only separator "
                "reconsideration/veto stage. A learned exception gate is still "
                "safer than an unconditional hard rule unless dangerous "
                "same-cell separator cases are literally absent with adequate "
                "sample size."
            ),
        }

    if (
        precision >= 0.98
        and fpr <= 0.02
        and veto_precision >= 0.98
    ):
        return {
            "category": (
                "support_for_learned_asymmetric_veto"
            ),
            "text": (
                "Separator evidence is highly predictive, but exceptions are "
                "large enough that a learned separation-only veto is preferable "
                "to a deterministic hard split."
            ),
        }

    return {
        "category": (
            "separator_should_remain_contextual"
        ),
        "text": (
            "The current full-volume audit does not justify near-hard separator "
            "authority. Inspect the dangerous same-cell separator examples and "
            "keep separator evidence contextual."
        ),
    }


def _report_text(
    *,
    checkpoint: Path,
    overall: dict[str, Any],
    per_sample: dict[str, Any],
    rule: dict[str, float],
    interpretation: dict[str, str],
) -> str:
    lines = [
        "=" * 122,
        "STIR-Net Investigation 28 — FULL-VOLUME separator reliability audit",
        "=" * 122,
        f"checkpoint : {checkpoint}",
        (
            "rule       : "
            f"mean>={rule['mean_min']:.3f} OR "
            f"(max>={rule['max_min']:.3f} AND "
            f"coverage70>={rule['coverage70_min']:.3f})"
        ),
        "",
        "THE THREE CRITICAL PROBABILITIES",
        "-" * 122,
        (
            "P(strong separator | GT separate) = "
            f"{overall['separator_recall_p_strong_given_gt_separate']:.8f}"
        ),
        (
            "P(strong separator | GT same)     = "
            f"{overall['same_cell_separator_fpr_p_strong_given_gt_same']:.8f}"
        ),
        (
            "P(GT separate | strong separator) = "
            f"{overall['separator_precision_p_gt_separate_given_strong']:.8f}"
        ),
        "",
        (
            "Dangerous same-cell strong-separator edges: "
            f"{overall['dangerous_same_cell_strong_separator_count']}/"
            f"{overall['gt_same_cell_edge_count']}"
        ),
        (
            "same-cell separator FPR 95% CI             : "
            f"{overall['same_cell_separator_fpr_95ci']}"
        ),
        "",
        "DIRECT TEST OF THE PROPOSED POST-RAG VETO",
        "-" * 122,
        (
            "Frozen h100 false merges                    : "
            f"{overall['base_false_merge_count']}"
        ),
        (
            "Those false merges with strong separator   : "
            f"{overall['veto_correct_split_count']}"
        ),
        (
            "False-merge recall of separator veto       : "
            f"{overall['veto_recall_of_current_false_merges']:.8f}"
        ),
        (
            "Legitimate base merges that veto would cut : "
            f"{overall['veto_wrong_split_count']}"
        ),
        (
            "P(GT separate | base MERGE + strong sep)   : "
            f"{overall['veto_precision_p_gt_separate_given_base_merge_and_strong']:.8f}"
        ),
        (
            "Damage rate among legitimate base merges   : "
            f"{overall['veto_damage_rate_among_legitimate_base_merges']:.8f}"
        ),
        "",
        "PER DATASET",
        "-" * 122,
    ]

    for sample, stats in per_sample.items():
        lines.extend(
            [
                (
                    f"{sample}: edges={stats['edge_count']} "
                    f"GT-separate={stats['gt_separate_edge_count']} "
                    f"GT-same={stats['gt_same_cell_edge_count']}"
                ),
                (
                    "  separator recall="
                    f"{stats['separator_recall_p_strong_given_gt_separate']:.6f} "
                    "precision="
                    f"{stats['separator_precision_p_gt_separate_given_strong']:.6f} "
                    "same-FPR="
                    f"{stats['same_cell_separator_fpr_p_strong_given_gt_same']:.6f}"
                ),
                (
                    "  base-FM="
                    f"{stats['base_false_merge_count']} "
                    "veto-correct="
                    f"{stats['veto_correct_split_count']} "
                    "veto-wrong="
                    f"{stats['veto_wrong_split_count']} "
                    "veto-precision="
                    f"{stats['veto_precision_p_gt_separate_given_base_merge_and_strong']:.6f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "ARCHITECTURAL INTERPRETATION",
            "-" * 122,
            interpretation["category"],
            interpretation["text"],
            "",
            (
                "Important: unlike the previous crop audit, each sample here "
                "contains one global RAG, so each undirected touching "
                "supervoxel pair is counted once."
            ),
            "=" * 122,
        ]
    )

    return "\n".join(lines) + "\n"


# ======================================================================================
# One sample
# ======================================================================================


def _run_sample(
    *,
    sample: str,
    inv26,
    support,
    model,
    model_cfg,
    nis3d_root: Path,
    spacing_override_zyx_um,
    cache_root: Path,
    temp_parent: Path,
    inference_cfg,
    amp_name: str,
    dense_cache_key_base: dict[str, Any],
    args,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    from learned.stirnet.model.partition.separator_barrier import (
        SEPARATOR_BARRIER_FEATURE_NAMES,
    )

    sample_started = (
        time.perf_counter()
    )

    print(
        "=" * 122,
        flush=True,
    )
    print(
        f"[sample] {sample}",
        flush=True,
    )
    print(
        "=" * 122,
        flush=True,
    )

    signature = support._data_signature(
        nis3d_root=nis3d_root,
        sample=sample,
        spacing_override_zyx_um=(
            spacing_override_zyx_um
        ),
        confidence_ignore_margin_um=(
            args.confidence_ignore_margin_um
        ),
    )

    source_batch, data_report = (
        support._prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=(
                spacing_override_zyx_um
            ),
            confidence_ignore_margin_um=(
                args.confidence_ignore_margin_um
            ),
            cache_root=cache_root,
            cache_namespace=signature,
        )
    )

    shape_zyx = tuple(
        int(value)
        for value in data_report[
            "shape_zyx"
        ]
    )

    sample_cache_dir = (
        temp_parent
        / f"{sample}_dense_cache"
    )
    sample_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_key = {
        **dense_cache_key_base,
        "sample": sample,
        "shape_zyx": list(shape_zyx),
        "spacing_um_zyx": _jsonable(
            source_batch["spacing_um"]
        ),
        "dref_um": _jsonable(
            source_batch["dref_um"]
        ),
    }

    required_bytes = (
        _geometry_storage_bytes(
            shape_zyx
        )
    )

    if _dense_cache_complete(
        sample_cache_dir,
        shape_zyx=shape_zyx,
        cache_key=cache_key,
    ):
        usage = shutil.disk_usage(
            sample_cache_dir
        )
        disk_report = {
            "required_gib": (
                required_bytes / 2**30
            ),
            "required_with_headroom_gib": 0.0,
            "free_gib_before": (
                usage.free / 2**30
            ),
            "dense_cache_reused": True,
        }
    else:
        disk_report = _check_work_disk(
            temp_parent,
            required_bytes,
        )
        disk_report[
            "dense_cache_reused"
        ] = False

    print(
        f"[sample] shape                  : {shape_zyx}",
        flush=True,
    )
    print(
        f"[sample] GT instances           : {data_report['gt_ids_kept']}",
        flush=True,
    )
    print(
        f"[sample] source instances       : {data_report['source_instance_count']}",
        flush=True,
    )
    print(
        f"[sample] dref                   : {data_report['model_dref_um']:.4f} um",
        flush=True,
    )
    print(
        f"[sample] dense cache            : {sample_cache_dir}",
        flush=True,
    )
    print(
        f"[sample] temporary geometry     : ~{disk_report['required_gib']:.2f} GiB",
        flush=True,
    )
    print(
        f"[sample] free work-disk before  : {disk_report['free_gib_before']:.2f} GiB",
        flush=True,
    )

    spacing_cpu = torch.as_tensor(
        source_batch["spacing_um"]
    ).detach().cpu().float()

    dref_cpu = torch.as_tensor(
        source_batch["dref_um"]
    ).detach().cpu().float()

    spacing_cuda = (
        spacing_cpu.to("cuda")
    )
    dref_cuda = (
        dref_cpu.to("cuda")
    )

    gt_labels_cpu = torch.as_tensor(
        source_batch["gt_labels"]
    ).detach().cpu()

    valid_mask_cpu = torch.as_tensor(
        source_batch[
            "supervision_valid_mask"
        ]
    ).detach().cpu().bool()

    full_raw_spatial_cpu = (
        _raw_only_full_spatial(
            source_batch
        )
    )

    geometry_mm: np.memmap | None = None
    weight_mm: np.memmap | None = None
    supervoxel_mm: np.memmap | None = None
    geometry = None
    supervoxels = None
    pooled_d0_cpu = None
    rag_cpu = None
    targets_cpu = None
    sample_success = False

    try:
        (
            geometry_mm,
            weight_mm,
            dense_report,
        ) = (
            _compute_global_geometry_to_memmap(
                model=model,
                source_batch=source_batch,
                spacing_cuda=spacing_cuda,
                dref_cuda=dref_cuda,
                shape_zyx=shape_zyx,
                inference_cfg=inference_cfg,
                cache_dir=sample_cache_dir,
                cache_key=cache_key,
                amp_name=amp_name,
            )
        )

        geometry = (
            _geometry_state_from_memmap(
                geometry_mm
            )
        )

        (
            supervoxels,
            supervoxel_mm,
            watershed_report,
        ) = _global_watershed(
            model=model,
            geometry=geometry,
            spacing_cpu=spacing_cpu,
            dref_cpu=dref_cpu,
            shape_zyx=shape_zyx,
            cache_dir=sample_cache_dir,
            cache_key=cache_key,
        )

        (
            pooled_d0_cpu,
            d0_report,
        ) = _stream_pooled_d0(
            model=model,
            source_batch=source_batch,
            supervoxels=supervoxels,
            weight_mm=weight_mm,
            spacing_cuda=spacing_cuda,
            dref_cuda=dref_cuda,
            inference_cfg=inference_cfg,
        )

        (
            rag_cpu,
            targets_cpu,
            rag_report,
        ) = (
            _build_global_rag_and_base_logits(
                model=model,
                model_cfg=model_cfg,
                supervoxels=supervoxels,
                pooled_d0_cpu=pooled_d0_cpu,
                full_raw_spatial_cpu=(
                    full_raw_spatial_cpu
                ),
                geometry=geometry,
                spacing_cpu=spacing_cpu,
                dref_cpu=dref_cpu,
                gt_labels_cpu=(
                    gt_labels_cpu
                ),
                valid_mask_cpu=(
                    valid_mask_cpu
                ),
            )
        )

        rows = _extract_edge_rows(
            sample=sample,
            rag=rag_cpu,
            targets=targets_cpu,
            feature_names=tuple(
                SEPARATOR_BARRIER_FEATURE_NAMES
            ),
            merge_threshold=(
                args.merge_threshold
            ),
            separator_mean_min=(
                args.separator_mean_min
            ),
            separator_max_min=(
                args.separator_max_min
            ),
            separator_coverage70_min=(
                args.separator_coverage70_min
            ),
        )

        current_strong = np.asarray(
            [
                bool(
                    row[
                        "strong_separator_current_rule"
                    ]
                )
                for row in rows
            ],
            dtype=bool,
        )

        sample_stats = (
            _core_statistics(
                rows,
                current_strong,
                merge_threshold=(
                    args.merge_threshold
                ),
            )
        )

        sample_report = {
            "data": data_report,
            "disk": disk_report,
            "dense": dense_report,
            "watershed": (
                watershed_report
            ),
            "d0_stream": d0_report,
            "rag": rag_report,
            "statistics": (
                sample_stats
            ),
            "valid_unique_edge_count": (
                len(rows)
            ),
            "sample_elapsed_seconds": (
                time.perf_counter()
                - sample_started
            ),
            "cache_directory": str(
                sample_cache_dir
            ),
        }

        sample_success = True
        return rows, sample_report

    finally:
        # Drop all torch views before closing the backing mappings.
        try:
            del rag_cpu
        except Exception:
            pass
        try:
            del targets_cpu
        except Exception:
            pass
        try:
            del pooled_d0_cpu
        except Exception:
            pass
        try:
            del supervoxels
        except Exception:
            pass
        try:
            del geometry
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()

        _close_memmap(supervoxel_mm)
        _close_memmap(weight_mm)
        _close_memmap(geometry_mm)

        del (
            source_batch,
            full_raw_spatial_cpu,
            gt_labels_cpu,
            valid_mask_cpu,
            spacing_cuda,
            dref_cuda,
            spacing_cpu,
            dref_cpu,
        )

        gc.collect()
        torch.cuda.empty_cache()

        if (
            sample_success
            and not args.keep_dense_cache
        ):
            print(
                f"[cache] sample completed; removing resumable cache {sample_cache_dir}",
                flush=True,
            )
            _safe_rmtree(
                sample_cache_dir
            )
        elif not sample_success:
            print(
                f"[cache] sample failed; keeping resumable cache at {sample_cache_dir}",
                flush=True,
            )

# ======================================================================================
# Main
# ======================================================================================


def run(
    args,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Investigation 28 requires CUDA for tiled dense inference "
            "and morphology-v2 patch encoding."
        )

    inv26, support = (
        _load_support()
    )

    checkpoint = (
        inv26._resolve_checkpoint(
            args.checkpoint
        )
    )

    (
        model,
        model_cfg,
        checkpoint_payload,
        transfer_report,
        _,
    ) = (
        inv26._build_model_from_checkpoint(
            checkpoint,
            device="cuda",
        )
    )

    # Evaluation only.
    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )
    model.eval()

    samples = tuple(
        token.strip()
        for token in args.samples.split(",")
        if token.strip()
    )
    if not samples:
        raise ValueError(
            "No samples requested"
        )

    spacing_override_zyx_um = (
        support._parse_spacing_xyz_override(
            args.spacing_xyz
        )
    )

    nis3d_root = (
        support._discover_nis3d_root(
            samples,
            data_dir=args.data_dir,
            execution_mode="local",
        )
    )

    tile_shape = _parse_zyx(
        args.tile_shape_zyx,
        name="--tile-shape-zyx",
    )
    tile_overlap = _parse_zyx(
        args.tile_overlap_zyx,
        name="--tile-overlap-zyx",
    )
    tile_halo = _parse_zyx(
        args.tile_halo_zyx,
        name="--tile-halo-zyx",
    )

    inference_cfg = (
        _build_inference_config(
            model_cfg,
            tile_shape=tile_shape,
            tile_overlap=tile_overlap,
            tile_halo=tile_halo,
        )
    )

    _, amp_name = _amp_context()

    output_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / EXPERIMENT_NAME
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        output_root
        / "attempts"
        / (
            f"{timestamp}_"
            f"{args.run_name}"
        )
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if args.work_dir:
        work_root = Path(
            args.work_dir
        ).expanduser()
        if not work_root.is_absolute():
            work_root = (
                ROOT
                / work_root
            )
        work_root = (
            work_root.resolve()
        )
    else:
        work_root = (
            output_root
            / "_temporary_geometry"
        )

    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "17_morphology_rag_multicrop_training"
        / "cache"
    )

    checkpoint_stat = checkpoint.stat()
    dense_cache_key_base = {
        "checkpoint": str(checkpoint),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "checkpoint_global_step": checkpoint_payload.get("global_step"),
        "tile_shape_zyx": list(tile_shape),
        "tile_overlap_zyx": list(tile_overlap),
        "tile_halo_zyx": list(tile_halo),
        "amp": amp_name,
        "geometry_storage_dtype": "float16",
        "blend_weight_storage_dtype": "float32",
        "cache_version": 2,
    }

    props = torch.cuda.get_device_properties(
        0
    )

    print(
        "=" * 122,
        flush=True,
    )
    print(
        "STIR-Net Investigation 28 — FULL-VOLUME separator reliability audit",
        flush=True,
    )
    print(
        "=" * 122,
        flush=True,
    )
    print(
        f"GPU                       : {props.name}",
        flush=True,
    )
    print(
        f"GPU VRAM                  : {props.total_memory / 2**30:.2f} GiB",
        flush=True,
    )
    print(
        f"Checkpoint                : {checkpoint}",
        flush=True,
    )
    print(
        f"Checkpoint step           : {checkpoint_payload.get('global_step', 'unknown')}",
        flush=True,
    )
    print(
        f"Samples                   : {list(samples)}",
        flush=True,
    )
    print(
        f"NIS3D root                : {nis3d_root}",
        flush=True,
    )
    print(
        f"Effective spacing XYZ um  : {args.spacing_xyz}",
        flush=True,
    )
    print(
        f"Tile shape ZYX            : {tile_shape}",
        flush=True,
    )
    print(
        f"Tile overlap ZYX          : {tile_overlap}",
        flush=True,
    )
    print(
        f"Tile halo ZYX             : {tile_halo}",
        flush=True,
    )
    print(
        f"AMP                       : {amp_name}",
        flush=True,
    )
    print(
        "Geometry temp storage     : float16 packed + float32 blend weights",
        flush=True,
    )
    print(
        f"Work directory            : {work_root}",
        flush=True,
    )
    print(
        f"Base merge threshold      : {args.merge_threshold:.3f}",
        flush=True,
    )
    print(
        "Separator rule            : "
        f"mean>={args.separator_mean_min:.3f} OR "
        f"(max>={args.separator_max_min:.3f} AND "
        f"coverage70>={args.separator_coverage70_min:.3f})",
        flush=True,
    )
    print(
        "Training                  : NONE",
        flush=True,
    )
    print(
        "=" * 122,
        flush=True,
    )

    run_started = (
        time.perf_counter()
    )

    all_rows: list[
        dict[str, Any]
    ] = []
    sample_reports: dict[
        str,
        Any,
    ] = {}

    for sample in samples:
        rows, report = _run_sample(
            sample=sample,
            inv26=inv26,
            support=support,
            model=model,
            model_cfg=model_cfg,
            nis3d_root=nis3d_root,
            spacing_override_zyx_um=(
                spacing_override_zyx_um
            ),
            cache_root=cache_root,
            temp_parent=work_root,
            inference_cfg=inference_cfg,
            amp_name=amp_name,
            dense_cache_key_base=dense_cache_key_base,
            args=args,
        )

        all_rows.extend(rows)
        sample_reports[
            sample
        ] = report

        _write_csv(
            run_dir
            / f"{sample}_edge_observations.csv",
            rows,
        )
        _atomic_json(
            run_dir
            / f"{sample}_summary.json",
            report,
        )

        print(
            "[sample done] "
            f"{sample}: "
            f"valid unique edges={len(rows)} "
            f"same+strong="
            f"{report['statistics']['dangerous_same_cell_strong_separator_count']} "
            f"base-FM="
            f"{report['statistics']['base_false_merge_count']} "
            f"veto-correct="
            f"{report['statistics']['veto_correct_split_count']} "
            f"veto-wrong="
            f"{report['statistics']['veto_wrong_split_count']}",
            flush=True,
        )

    if not all_rows:
        raise RuntimeError(
            "No valid supervised global RAG edges were found."
        )

    current_strong = np.asarray(
        [
            bool(
                row[
                    "strong_separator_current_rule"
                ]
            )
            for row in all_rows
        ],
        dtype=bool,
    )

    overall = (
        _core_statistics(
            all_rows,
            current_strong,
            merge_threshold=(
                args.merge_threshold
            ),
        )
    )

    per_sample = {
        sample: (
            sample_reports[
                sample
            ][
                "statistics"
            ]
        )
        for sample in sample_reports
    }

    threshold_rows = (
        _threshold_sweep(
            all_rows,
            merge_threshold=(
                args.merge_threshold
            ),
            current_mean=(
                args.separator_mean_min
            ),
            current_max=(
                args.separator_max_min
            ),
            current_coverage70=(
                args.separator_coverage70_min
            ),
        )
    )

    from learned.stirnet.model.partition.separator_barrier import (
        SEPARATOR_BARRIER_FEATURE_NAMES,
    )

    feature_rows = (
        _feature_distributions(
            all_rows,
            tuple(
                SEPARATOR_BARRIER_FEATURE_NAMES
            ),
        )
    )

    joint_rows = (
        _joint_probability_table(
            all_rows
        )
    )

    dangerous = [
        row
        for row in all_rows
        if (
            row["gt_same_cell"]
            and row[
                "strong_separator_current_rule"
            ]
        )
    ]
    dangerous.sort(
        key=lambda row: (
            float(
                row[
                    "separator_mean"
                ]
            ),
            float(
                row[
                    "separator_coverage_070"
                ]
            ),
            float(
                row[
                    "separator_max"
                ]
            ),
            float(
                row[
                    "base_merge_probability"
                ]
            ),
        ),
        reverse=True,
    )

    veto_candidates = [
        row
        for row in all_rows
        if row["veto_candidate"]
    ]
    veto_candidates.sort(
        key=lambda row: (
            row["veto_outcome"]
            == "WRONG_SPLIT",
            float(
                row[
                    "base_merge_probability"
                ]
            ),
            float(
                row[
                    "separator_mean"
                ]
            ),
        ),
        reverse=True,
    )

    interpretation = (
        _architecture_interpretation(
            overall
        )
    )

    current_rule = {
        "mean_min": (
            args.separator_mean_min
        ),
        "max_min": (
            args.separator_max_min
        ),
        "coverage70_min": (
            args.separator_coverage70_min
        ),
    }

    elapsed = (
        time.perf_counter()
        - run_started
    )

    summary = {
        "status": "success",
        "experiment": EXPERIMENT_NAME,
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "checkpoint": str(
            checkpoint
        ),
        "checkpoint_global_step": (
            checkpoint_payload.get(
                "global_step"
            )
        ),
        "checkpoint_transfer_report": (
            transfer_report
        ),
        "samples": list(
            samples
        ),
        "global_graph_semantics": (
            "one full-volume RAG per sample; each undirected touching "
            "supervoxel pair occurs once per sample"
        ),
        "dense_inference_semantics": (
            "GPU tiled dense CNN; full-volume out-of-core blended geometry; "
            "global production watershed/RAG"
        ),
        "geometry_temporary_storage_dtype": (
            "float16"
        ),
        "blend_weight_storage_dtype": (
            "float32"
        ),
        "tile_shape_zyx": list(
            tile_shape
        ),
        "tile_overlap_zyx": list(
            tile_overlap
        ),
        "tile_halo_zyx": list(
            tile_halo
        ),
        "amp": amp_name,
        "merge_threshold": (
            args.merge_threshold
        ),
        "current_separator_rule": (
            current_rule
        ),
        "overall": overall,
        "per_sample": per_sample,
        "sample_execution_reports": (
            sample_reports
        ),
        "architecture_interpretation": (
            interpretation
        ),
        "elapsed_seconds": elapsed,
        "elapsed_human": (
            _duration(elapsed)
        ),
        "output_directory": str(
            run_dir
        ),
    }

    report = _report_text(
        checkpoint=checkpoint,
        overall=overall,
        per_sample=per_sample,
        rule=current_rule,
        interpretation=(
            interpretation
        ),
    )

    _atomic_json(
        run_dir / "summary.json",
        summary,
    )
    _atomic_text(
        run_dir
        / "architecture_report.txt",
        report,
    )
    _write_csv(
        run_dir
        / "edge_observations.csv",
        all_rows,
    )
    _write_csv(
        run_dir
        / "threshold_sweep.csv",
        threshold_rows,
    )
    _write_csv(
        run_dir
        / "feature_distributions.csv",
        feature_rows,
    )
    _write_csv(
        run_dir
        / "base_probability_joint.csv",
        joint_rows,
    )
    _write_csv(
        run_dir
        / "dangerous_same_cell_strong_separator.csv",
        dangerous[
            : args.top_examples
        ],
    )
    _write_csv(
        run_dir
        / "veto_candidates.csv",
        veto_candidates,
    )

    print(
        "",
        flush=True,
    )
    print(
        report,
        flush=True,
    )
    print(
        f"[done] elapsed : {_duration(elapsed)}",
        flush=True,
    )
    print(
        f"[done] output  : {run_dir}",
        flush=True,
    )

    return summary


# ======================================================================================
# CLI
# ======================================================================================


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Full-volume NIS3D separator reliability and post-RAG veto audit."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--samples",
        default=DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--data-dir",
        default="external/NIS3D/NIS3D",
    )
    parser.add_argument(
        "--spacing-xyz",
        default=DEFAULT_SPACING_XYZ,
    )

    parser.add_argument(
        "--tile-shape-zyx",
        default=",".join(
            str(value)
            for value in DEFAULT_TILE_SHAPE_ZYX
        ),
    )
    parser.add_argument(
        "--tile-overlap-zyx",
        default=",".join(
            str(value)
            for value in DEFAULT_TILE_OVERLAP_ZYX
        ),
    )
    parser.add_argument(
        "--tile-halo-zyx",
        default=",".join(
            str(value)
            for value in DEFAULT_TILE_HALO_ZYX
        ),
    )

    parser.add_argument(
        "--confidence-ignore-margin-um",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=DEFAULT_MERGE_THRESHOLD,
    )
    parser.add_argument(
        "--separator-mean-min",
        type=float,
        default=DEFAULT_SEPARATOR_MEAN_MIN,
    )
    parser.add_argument(
        "--separator-max-min",
        type=float,
        default=DEFAULT_SEPARATOR_MAX_MIN,
    )
    parser.add_argument(
        "--separator-coverage70-min",
        type=float,
        default=DEFAULT_SEPARATOR_COVERAGE70_MIN,
    )

    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Out-of-core/resumable geometry directory. A completed dense pass "
            "is kept after failures and reused on the next run."
        ),
    )
    parser.add_argument(
        "--keep-dense-cache",
        action="store_true",
        help=(
            "Keep the ~5 GiB per-sample dense/supervoxel cache even after that "
            "sample completes successfully. By default it is deleted only after "
            "the sample audit succeeds."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="drosophila_12_full_volume_separator_reliability_h100",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=200,
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.top_examples < 1:
        raise ValueError(
            "--top-examples must be positive"
        )

    if not (
        0.0
        < args.merge_threshold
        < 1.0
    ):
        raise ValueError(
            "--merge-threshold must be in (0,1)"
        )

    summary = run(args)

    compact = {
        "status": (
            summary["status"]
        ),
        "output_directory": (
            summary[
                "output_directory"
            ]
        ),
        "overall": (
            summary["overall"]
        ),
        "architecture_interpretation": (
            summary[
                "architecture_interpretation"
            ]
        ),
    }

    print(
        json.dumps(
            _jsonable(compact),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
