from __future__ import annotations

r"""
Investigation 03 — end-to-end continuation overfit on the ambiguous synthetic
cuts built by Investigation 01.

The objective is deliberately narrow:

    several nearby clean Trackastra tracks
        -> cut all of them at the same t -> t+1 boundary
        -> give the reconciler the true and cross-track candidate edges
        -> memorize the correct continuation assignment

Division, appearance and termination losses are OFF in this investigation.
The cell-fingerprint CNN, temporal encoders, primitive relation encoder,
geometry-biased edge reasoner and continuation head are trained end-to-end.

The run is considered successful only when the complete local assignments are
recovered, not merely when average edge loss decreases.

Inputs
------
runs/track_reconciler/investigations/
  01_build_overfit_dataset/<sample_id>/
    manifest.json
    examples.csv
    tracklets.csv
    observations.csv
    edges.csv
    global_motion.csv

Investigation-36 source assets referenced by manifest.json:
    raw.npy
    final_instances.npy

Default run
-----------
    python .\investigations\track_reconciler\03_continuation_overfit.py

Useful first run on the RTX 4050:
    python .\investigations\track_reconciler\03_continuation_overfit.py ^
        --steps 800 --batch-size 1 --overwrite

If VRAM allows:
    python .\investigations\track_reconciler\03_continuation_overfit.py ^
        --steps 800 --batch-size 2 --amp bf16 --overwrite

Ablation with no image fingerprint evidence:
    python .\investigations\track_reconciler\03_continuation_overfit.py ^
        --appearance-mode zeros --overwrite

Expected success signal
-----------------------
The strongest metric is `exact`, the fraction of examples for which a global
one-to-one linear assignment over continuation logits recovers every original
identity.  `src_top1` and `tgt_top1` should also approach 100%.

This is a memorization/debug experiment.  Normalization statistics are fitted
on the same tiny dataset by design and saved with the checkpoint.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from torch import nn

SCRIPT_NAME = "03_continuation_overfit"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"

DEFAULT_STEPS = 800
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 3.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_EVAL_EVERY = 25
DEFAULT_LOG_EVERY = 10
DEFAULT_SEED = 1337
DEFAULT_GRAD_CLIP = 5.0

# Native BioHub crop.  With spacing (1.625, 0.40625, 0.40625) um this covers
# roughly 21 x 16.7 x 16.7 um, close enough physically while retaining native
# voxels and keeping the first overfit small on a 6-GB GPU.
DEFAULT_CROP_SHAPE_ZYX = (13, 41, 41)
DEFAULT_DISTANCE_CLIP_UM = 5.0

STRICT_POSITIVE_PROBABILITY = 0.90
STRICT_NEGATIVE_PROBABILITY = 0.10
STRICT_SUCCESS_EVALS = 3


# =============================================================================
# Repository / paths
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.parent, *here.parents, Path.cwd().resolve())
    for candidate in candidates:
        if (
            (candidate / "learned" / "track_reconciler").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the repository package only after the repo root is guaranteed to be
# importable. This matters for direct Windows execution via:
#   python .\investigations\track_reconciler\03_continuation_overfit.py
from learned.track_reconciler import (
    CandidateEdgeBatch,
    ReconciliationBatch,
    ReconcilerConfig,
    TrackletBatch,
    TrackletReconciliationNetwork,
)
from learned.track_reconciler.integration.manifest import (
    TRACKLET_RELIABILITY_FEATURES,
    TRACKLET_STRUCTURED_PRIMITIVES,
)
from learned.track_reconciler.integration.stage11 import (
    STAGE11_PAIR_FEATURES,
    tensorize_stage11_pair_features,
)
from learned.track_reconciler.training.losses import (
    focal_binary_probability_loss,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def default_dataset(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "investigations"
        / "01_build_overfit_dataset"
        / sample_id
    ).resolve()


def default_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "investigations"
        / SCRIPT_NAME
        / sample_id
    ).resolve()


def parse_triplet_int(text: str, *, name: str) -> tuple[int, int, int]:
    values = tuple(int(token.strip()) for token in str(text).split(","))
    if len(values) != 3 or not all(value > 0 for value in values):
        raise ValueError(f"{name} must contain three positive integers")
    if not all(value % 2 == 1 for value in values):
        raise ValueError(
            f"{name} should use odd dimensions so the cell centroid is centered"
        )
    return values  # type: ignore[return-value]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def root_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


# =============================================================================
# Source dataset
# =============================================================================


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    manifest: dict[str, Any]
    examples: pd.DataFrame
    tracklets: pd.DataFrame
    observations: pd.DataFrame
    edges: pd.DataFrame
    global_motion: pd.DataFrame

    @classmethod
    def load(cls, root: Path) -> "SourceDataset":
        required = (
            "manifest.json",
            "examples.csv",
            "tracklets.csv",
            "observations.csv",
            "edges.csv",
            "global_motion.csv",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Overfit dataset is incomplete at {root}; missing {missing}"
            )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        examples = pd.read_csv(root / "examples.csv")
        tracklets = pd.read_csv(root / "tracklets.csv")
        observations = pd.read_csv(root / "observations.csv").reset_index(drop=True)
        observations.insert(0, "_observation_row", np.arange(len(observations)))
        return cls(
            root=root,
            manifest=manifest,
            examples=examples,
            tracklets=tracklets,
            observations=observations,
            edges=pd.read_csv(root / "edges.csv"),
            global_motion=pd.read_csv(root / "global_motion.csv"),
        )

    def source_path(self, key: str) -> Path:
        value = self.manifest.get("source", {}).get(key)
        if value is None:
            raise KeyError(f"manifest source does not contain {key!r}")
        return resolve(value)


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Crop cache / five-channel fingerprint input
# =============================================================================


@dataclass(frozen=True)
class CropCache:
    crops: np.ndarray                 # [M,5,D,H,W], float16
    valid_fraction: np.ndarray        # [M], float32


def _crop_bounds(center: float, size: int, limit: int) -> tuple[slice, slice, int]:
    radius = size // 2
    c = int(round(float(center)))
    start = c - radius
    stop = start + size

    source_start = max(start, 0)
    source_stop = min(stop, limit)

    target_start = source_start - start
    target_stop = target_start + (source_stop - source_start)
    return (
        slice(source_start, source_stop),
        slice(target_start, target_stop),
        max(source_stop - source_start, 0),
    )


def _extract_padded(
    volume: np.ndarray,
    center_zyx: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
    *,
    fill_value: float | int = 0,
) -> tuple[np.ndarray, float]:
    output = np.full(shape_zyx, fill_value, dtype=volume.dtype)

    zsrc, zdst, zn = _crop_bounds(center_zyx[0], shape_zyx[0], volume.shape[0])
    ysrc, ydst, yn = _crop_bounds(center_zyx[1], shape_zyx[1], volume.shape[1])
    xsrc, xdst, xn = _crop_bounds(center_zyx[2], shape_zyx[2], volume.shape[2])

    if zn > 0 and yn > 0 and xn > 0:
        output[zdst, ydst, xdst] = volume[zsrc, ysrc, xsrc]

    valid_fraction = float(
        (zn * yn * xn) / np.prod(shape_zyx)
    )
    return output, valid_fraction


def build_crop_cache(
    data: SourceDataset,
    *,
    cache_dir: Path,
    shape_zyx: tuple[int, int, int],
    distance_clip_um: float,
    rebuild: bool,
) -> CropCache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    crop_path = cache_dir / "fingerprint_crops_float16.npy"
    valid_path = cache_dir / "crop_valid_fraction.npy"
    meta_path = cache_dir / "crop_cache.json"

    expected_meta = {
        "schema_version": 1,
        "dataset": root_relative(data.root),
        "observation_rows": int(len(data.observations)),
        "shape_zyx": list(shape_zyx),
        "distance_clip_um": float(distance_clip_um),
        "cell_channels": [
            "normalized_raw",
            "target_mask",
            "target_edt",
            "signed_distance",
            "other_instance_occupancy",
        ],
    }

    if (
        not rebuild
        and crop_path.is_file()
        and valid_path.is_file()
        and meta_path.is_file()
    ):
        try:
            actual = json.loads(meta_path.read_text(encoding="utf-8"))
            if actual == expected_meta:
                crops = np.load(crop_path, mmap_mode="r", allow_pickle=False)
                valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
                if (
                    crops.shape[0] == len(data.observations)
                    and tuple(crops.shape[1:]) == (5, *shape_zyx)
                    and valid.shape == (len(data.observations),)
                ):
                    print(
                        f"[crops] reusing cache {crop_path} "
                        f"shape={tuple(crops.shape)}",
                        flush=True,
                    )
                    return CropCache(crops=crops, valid_fraction=valid)
        except Exception:
            pass

    raw_path = data.source_path("raw")
    labels_path = data.source_path("final_instances")
    raw_movie = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    labels_movie = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if raw_movie.shape != labels_movie.shape:
        raise ValueError(
            f"raw/label movie shape mismatch: {raw_movie.shape} vs "
            f"{labels_movie.shape}"
        )

    spacing = tuple(float(v) for v in data.manifest["spacing_zyx_um"])
    m = len(data.observations)
    crops = np.lib.format.open_memmap(
        crop_path,
        mode="w+",
        dtype=np.float16,
        shape=(m, 5, *shape_zyx),
    )
    valid = np.lib.format.open_memmap(
        valid_path,
        mode="w+",
        dtype=np.float32,
        shape=(m,),
    )

    contrast_cache: dict[int, tuple[float, float]] = {}

    print(
        f"[crops] building {m} five-channel crops "
        f"shape={shape_zyx} from Investigation 36 ...",
        flush=True,
    )
    for row_number, row in enumerate(data.observations.itertuples(index=False)):
        frame = int(row.frame)
        cell_id = int(row.cell_id)
        center = (float(row.z), float(row.y), float(row.x))

        raw_frame = np.asarray(raw_movie[frame])
        labels_frame = np.asarray(labels_movie[frame])

        if frame not in contrast_cache:
            low, high = np.percentile(raw_frame, [1.0, 99.8])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low = float(np.min(raw_frame))
                high = float(np.max(raw_frame))
                if high <= low:
                    high = low + 1.0
            contrast_cache[frame] = (float(low), float(high))
        low, high = contrast_cache[frame]

        raw_crop, valid_fraction = _extract_padded(
            raw_frame, center, shape_zyx, fill_value=0
        )
        label_crop, _ = _extract_padded(
            labels_frame, center, shape_zyx, fill_value=0
        )

        target = label_crop == cell_id
        if not target.any():
            raise RuntimeError(
                "Target instance is absent from its fingerprint crop: "
                f"row={row_number}, example={row.example_id}, "
                f"frame={frame}, cell_id={cell_id}, center={center}"
            )

        normalized_raw = np.clip(
            (raw_crop.astype(np.float32) - low) / max(high - low, 1e-6),
            0.0,
            1.0,
        )
        target_f = target.astype(np.float32)
        inside = distance_transform_edt(target, sampling=spacing).astype(np.float32)
        outside = distance_transform_edt(~target, sampling=spacing).astype(np.float32)
        edt = np.clip(
            inside / float(distance_clip_um),
            0.0,
            1.0,
        )
        sdf = np.clip(
            (inside - outside) / float(distance_clip_um),
            -1.0,
            1.0,
        )
        occupancy = ((label_crop > 0) & (~target)).astype(np.float32)

        crops[row_number] = np.stack(
            (normalized_raw, target_f, edt, sdf, occupancy),
            axis=0,
        ).astype(np.float16)
        valid[row_number] = float(valid_fraction)

        if (row_number + 1) % 100 == 0 or row_number + 1 == m:
            print(
                f"[crops] {row_number + 1:4d}/{m:4d}",
                flush=True,
            )

    crops.flush()
    valid.flush()
    atomic_json(meta_path, expected_meta)

    del crops, valid, raw_movie, labels_movie
    return CropCache(
        crops=np.load(crop_path, mmap_mode="r", allow_pickle=False),
        valid_fraction=np.load(valid_path, mmap_mode="r", allow_pickle=False),
    )


# =============================================================================
# Feature construction
# =============================================================================


def _motion_lookup(global_motion: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    lookup: dict[tuple[int, int], np.ndarray] = {}
    required = {
        "frame_from",
        "frame_to",
        "shift_z_um",
        "shift_y_um",
        "shift_x_um",
    }
    if not required.issubset(global_motion.columns):
        raise ValueError(
            f"global_motion.csv is missing {sorted(required - set(global_motion.columns))}"
        )
    for row in global_motion.itertuples(index=False):
        shift = np.asarray(
            [row.shift_z_um, row.shift_y_um, row.shift_x_um],
            dtype=np.float32,
        )
        if np.isfinite(shift).all():
            lookup[(int(row.frame_from), int(row.frame_to))] = shift
    return lookup


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _column_array(rows: pd.DataFrame, name: str) -> np.ndarray:
    if name not in rows:
        return np.full(len(rows), np.nan, dtype=np.float32)
    return pd.to_numeric(rows[name], errors="coerce").to_numpy(
        dtype=np.float32, copy=True
    )


def build_raw_structured(
    rows: pd.DataFrame,
    *,
    global_motion: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Return primitive [K,24], validity [K,24], and reliability diagnostics.

    Relative position is local to the synthetic tracklet.  Velocity and
    acceleration are physical um/frame.  Global shift is the robust frame
    translation entering the current observation.
    """
    rows = rows.sort_values("sequence_index")
    k = len(rows)
    primitives = np.zeros(
        (k, len(TRACKLET_STRUCTURED_PRIMITIVES)), dtype=np.float32
    )
    validity = np.zeros_like(primitives, dtype=np.float32)
    index = {
        name: i for i, name in enumerate(TRACKLET_STRUCTURED_PRIMITIVES)
    }

    xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float32)
    frames = rows["frame"].to_numpy(dtype=np.int64)
    relative_position = xyz - xyz[0:1]
    for axis, axis_name in enumerate(("z", "y", "x")):
        name = f"relative_position_{axis_name}_um"
        primitives[:, index[name]] = relative_position[:, axis]
        validity[:, index[name]] = np.isfinite(relative_position[:, axis])

    incoming_global = np.full((k, 3), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        shift = global_motion.get((int(frame) - 1, int(frame)))
        if shift is not None:
            incoming_global[i] = shift
    for axis, axis_name in enumerate(("z", "y", "x")):
        name = f"global_shift_{axis_name}_um"
        finite = np.isfinite(incoming_global[:, axis])
        primitives[finite, index[name]] = incoming_global[finite, axis]
        validity[finite, index[name]] = 1.0

    velocity = np.full((k, 3), np.nan, dtype=np.float32)
    if k > 1:
        dt = np.diff(frames).astype(np.float32)
        delta = np.diff(xyz, axis=0)
        okay = dt > 0
        velocity[1:][okay] = delta[okay] / dt[okay, None]

    relative_velocity = np.full((k, 3), np.nan, dtype=np.float32)
    for i in range(1, k):
        if not np.isfinite(velocity[i]).all():
            continue
        shift = global_motion.get((int(frames[i - 1]), int(frames[i])))
        if shift is not None:
            relative_velocity[i] = velocity[i] - shift

    acceleration = np.full((k, 3), np.nan, dtype=np.float32)
    for i in range(2, k):
        if np.isfinite(velocity[i]).all() and np.isfinite(velocity[i - 1]).all():
            dt = max(float(frames[i] - frames[i - 1]), 1.0)
            acceleration[i] = (velocity[i] - velocity[i - 1]) / dt

    for array, prefix in (
        (velocity, "velocity"),
        (relative_velocity, "relative_velocity"),
        (acceleration, "acceleration"),
    ):
        suffix = (
            "um_per_frame2" if prefix == "acceleration" else "um_per_frame"
        )
        for axis, axis_name in enumerate(("z", "y", "x")):
            name = f"{prefix}_{axis_name}_{suffix}"
            finite = np.isfinite(array[:, axis])
            primitives[finite, index[name]] = array[finite, axis]
            validity[finite, index[name]] = 1.0

    volume = _column_array(rows, "volume")
    valid_volume = np.isfinite(volume) & (volume > 0)
    log_volume = np.full(k, np.nan, dtype=np.float32)
    log_volume[valid_volume] = np.log(volume[valid_volume])
    finite = np.isfinite(log_volume)
    primitives[finite, index["log_volume"]] = log_volume[finite]
    validity[finite, index["log_volume"]] = 1.0

    delta_log = np.full(k, np.nan, dtype=np.float32)
    for i in range(1, k):
        if np.isfinite(log_volume[i]) and np.isfinite(log_volume[i - 1]):
            delta_log[i] = log_volume[i] - log_volume[i - 1]
    finite = np.isfinite(delta_log)
    primitives[finite, index["delta_log_volume"]] = delta_log[finite]
    validity[finite, index["delta_log_volume"]] = 1.0

    direct_columns = (
        "axis_major",
        "axis_middle",
        "axis_minor",
        "elongation",
        "flatness",
        "anisotropy",
        "intensity_mean",
    )
    for name in direct_columns:
        values = _column_array(rows, name)
        finite = np.isfinite(values)
        primitives[finite, index[name]] = values[finite]
        validity[finite, index[name]] = 1.0

    rv_valid = np.isfinite(relative_velocity).all(axis=1)
    rv_samples = relative_velocity[rv_valid]
    if len(rv_samples):
        rv_center = rv_samples.mean(axis=0)
        rv_error = float(
            np.linalg.norm(rv_samples - rv_center[None, :], axis=1).mean()
        )
    else:
        rv_error = 0.0

    acceleration_valid = np.isfinite(acceleration).all(axis=1)
    position_residual = (
        float(np.linalg.norm(acceleration[acceleration_valid], axis=1).mean())
        if acceleration_valid.any()
        else 0.0
    )

    diagnostics = {
        "relative_velocity_sample_count": float(len(rv_samples)),
        "relative_velocity_error_ema_um": rv_error,
        "position_residual_ema_norm_um": position_residual,
    }
    return primitives, validity, diagnostics


@dataclass(frozen=True)
class MaskedStats:
    mean: np.ndarray
    std: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def fit_masked_stats(
    values: np.ndarray,
    validity: np.ndarray,
) -> MaskedStats:
    if values.shape != validity.shape:
        raise ValueError("values/validity shape mismatch")
    dim = values.shape[-1]
    flat_values = values.reshape(-1, dim).astype(np.float64)
    flat_valid = validity.reshape(-1, dim).astype(bool)
    mean = np.zeros(dim, dtype=np.float32)
    std = np.ones(dim, dtype=np.float32)
    for feature in range(dim):
        selected = flat_values[flat_valid[:, feature], feature]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            mean[feature] = float(selected.mean())
            sigma = float(selected.std())
            std[feature] = sigma if sigma > 1e-6 else 1.0
    return MaskedStats(mean=mean, std=std)


def apply_masked_stats(
    values: np.ndarray,
    validity: np.ndarray,
    stats: MaskedStats,
) -> np.ndarray:
    normalized = (
        values - stats.mean.reshape((1,) * (values.ndim - 1) + (-1,))
    ) / stats.std.reshape((1,) * (values.ndim - 1) + (-1,))
    normalized = np.where(validity > 0.5, normalized, 0.0)
    return np.concatenate(
        (normalized.astype(np.float32), validity.astype(np.float32)),
        axis=-1,
    )


def build_reliability_raw(
    rows: pd.DataFrame,
    *,
    diagnostics: dict[str, float],
    crop_valid_fraction: np.ndarray,
    global_median_volume: float,
) -> np.ndarray:
    values = np.zeros(len(TRACKLET_RELIABILITY_FEATURES), dtype=np.float32)
    index = {
        name: i for i, name in enumerate(TRACKLET_RELIABILITY_FEATURES)
    }
    k = len(rows)
    frames = rows["frame"].to_numpy(dtype=np.int64)
    volume = _column_array(rows, "volume")
    valid_volume = volume[np.isfinite(volume) & (volume > 0)]

    values[index["log_observation_count"]] = math.log1p(k)
    values[index["temporal_span_frames"]] = (
        float(frames.max() - frames.min()) if len(frames) else 0.0
    )

    # Investigation 36 does not persist Trackastra per-edge confidence in its
    # public visualization CSV.  Zero means "not supplied" for this first
    # overfit; these two channels will be constant and therefore harmless.
    values[index["association_probability_mean"]] = 0.0
    values[index["association_probability_min"]] = 0.0

    values[index["relative_velocity_sample_count"]] = float(
        diagnostics["relative_velocity_sample_count"]
    )
    values[index["relative_velocity_error_ema_um"]] = float(
        diagnostics["relative_velocity_error_ema_um"]
    )
    values[index["position_residual_ema_norm_um"]] = float(
        diagnostics["position_residual_ema_norm_um"]
    )

    boundary = _column_array(rows, "distance_to_boundary_um")
    finite_boundary = boundary[np.isfinite(boundary)]
    boundary_min = (
        float(finite_boundary.min()) if finite_boundary.size else 0.0
    )
    values[index["distance_to_boundary_um"]] = boundary_min
    values[index["touches_boundary"]] = float(boundary_min < 4.0)

    median_volume = (
        float(np.median(valid_volume))
        if valid_volume.size
        else max(global_median_volume, 1.0)
    )
    values[index["log_median_volume"]] = math.log(max(median_volume, 1.0))
    values[index["small_cell_indicator"]] = float(
        median_volume < 0.5 * global_median_volume
    )
    values[index["crop_valid_fraction"]] = float(
        np.mean(crop_valid_fraction)
    )
    return values


def fit_reliability_stats(raw: np.ndarray) -> MaskedStats:
    """
    Z-score continuous reliability channels while keeping binary flags raw.
    """
    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0

    binary_names = {"touches_boundary", "small_cell_indicator"}
    for name in binary_names:
        idx = TRACKLET_RELIABILITY_FEATURES.index(name)
        mean[idx] = 0.0
        std[idx] = 1.0
    return MaskedStats(mean=mean, std=std)


# =============================================================================
# Pair feature enrichment
# =============================================================================


def _endpoint_rows(
    observations: pd.DataFrame,
) -> tuple[dict[tuple[int, int], pd.Series], dict[tuple[int, int], pd.Series]]:
    source: dict[tuple[int, int], pd.Series] = {}
    target: dict[tuple[int, int], pd.Series] = {}
    for (example_id, tracklet_index), group in observations.groupby(
        ["example_id", "tracklet_index"]
    ):
        group = group.sort_values("sequence_index")
        role = str(group["role"].iloc[0])
        key = (int(example_id), int(tracklet_index))
        if role == "source":
            source[key] = group.iloc[-1]
        elif role == "target":
            target[key] = group.iloc[0]
    return source, target


def _relative_error(a: Any, b: Any) -> float:
    if not _finite(a) or not _finite(b):
        return float("nan")
    av = float(a)
    bv = float(b)
    return abs(av - bv) / max(0.5 * (abs(av) + abs(bv)), 1e-6)


def _log_ratio_error(a: Any, b: Any) -> float:
    if not _finite(a) or not _finite(b):
        return float("nan")
    av = float(a)
    bv = float(b)
    if av <= 0 or bv <= 0:
        return float("nan")
    return abs(math.log(bv / av))


def enrich_pair_features(
    data: SourceDataset,
    *,
    tracklet_diagnostics: dict[tuple[int, int], dict[str, float]],
) -> pd.DataFrame:
    edges = data.edges.copy()
    source_endpoint, target_endpoint = _endpoint_rows(data.observations)
    tracklet_counts = data.tracklets.set_index(
        ["example_id", "tracklet_index"]
    )["observation_count"].to_dict()

    radius = float(
        data.manifest.get("parameters", {}).get("candidate_radius_um", 12.0)
    )

    rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        example_id = int(edge.example_id)
        source_index = int(edge.source_tracklet_index)
        target_index = int(edge.target_tracklet_index)
        s = source_endpoint[(example_id, source_index)]
        t = target_endpoint[(example_id, target_index)]

        row = dict(edge._asdict())
        row["direct_endpoint_distance_um"] = float(edge.direct_distance_um)
        row["hard_search_radius_um"] = radius

        if _finite(getattr(edge, "global_relative_error_um", np.nan)):
            row["forward_error_um"] = float(edge.global_relative_error_um)
        else:
            row["forward_error_um"] = np.nan
        row["forward_history_count"] = int(
            tracklet_counts[(example_id, source_index)]
        )
        row["forward_used_global_motion"] = bool(
            int(getattr(edge, "global_prediction_valid", 0))
        )
        row["forward_used_relative_velocity"] = bool(
            int(getattr(edge, "global_relative_prediction_valid", 0))
        )
        row["forward_uncertainty_um"] = float(
            tracklet_diagnostics[(example_id, source_index)][
                "relative_velocity_error_ema_um"
            ]
        )

        # Not available in the first synthetic-cut builder. Missingness is
        # intentional and becomes explicit via the Stage-11 validity channels.
        row["backward_error_um"] = np.nan
        row["backward_history_count"] = np.nan
        row["backward_prediction_available"] = False
        row["bidirectional_disagreement_um"] = np.nan
        row["anchor_count"] = np.nan
        row["anchor_prediction_error_um"] = np.nan
        row["neighborhood_distance_error_um"] = np.nan
        row["local_survival_ratio"] = np.nan

        row["volume_log_error"] = _log_ratio_error(
            s.get("volume", np.nan), t.get("volume", np.nan)
        )

        shape_terms = [
            _log_ratio_error(s.get(name, np.nan), t.get(name, np.nan))
            for name in ("axis_major", "axis_middle", "axis_minor")
        ]
        shape_terms = [value for value in shape_terms if np.isfinite(value)]
        row["shape_error"] = (
            float(np.mean(shape_terms)) if shape_terms else np.nan
        )

        intensity_names = (
            "intensity_mean",
            "intensity_median",
            "intensity_std",
            "intensity_iqr",
            "intensity_cv",
        )
        intensity_errors: list[float] = []
        for name in intensity_names:
            value = _relative_error(
                s.get(name, np.nan), t.get(name, np.nan)
            )
            row[f"{name}_error"] = value
            if np.isfinite(value):
                intensity_errors.append(value)
        row["intensity_error"] = (
            float(np.mean(intensity_errors))
            if intensity_errors
            else np.nan
        )

        row["target_real_observation_count"] = int(
            tracklet_counts[(example_id, target_index)]
        )

        row["stage7_candidate_available"] = False
        row["stage7_candidate_distance_um"] = np.nan
        row["stage7_candidate_pair_cost"] = np.nan
        row["stage7_candidate_probability"] = np.nan
        row["stage7_candidate_rank"] = np.nan

        row["candidate_quality_score"] = math.exp(
            -float(edge.direct_distance_um) / max(radius, 1e-6)
        )
        row["source_rank"] = int(edge.source_distance_rank)
        row["target_rank"] = int(edge.target_distance_rank)

        source_volume = float(s.get("volume", np.nan))
        target_volume = float(t.get("volume", np.nan))
        row["effective_pair_volume"] = (
            math.sqrt(source_volume * target_volume)
            if np.isfinite(source_volume)
            and np.isfinite(target_volume)
            and source_volume > 0
            and target_volume > 0
            else np.nan
        )
        row["small_cell_history_exception"] = False
        rows.append(row)

    enriched = pd.DataFrame(rows)

    enriched["source_candidate_count"] = enriched.groupby(
        ["example_id", "source_tracklet_index"]
    )["edge_index"].transform("size")
    enriched["target_predecessor_count"] = enriched.groupby(
        ["example_id", "target_tracklet_index"]
    )["edge_index"].transform("size")

    enriched["source_score_margin"] = np.nan
    enriched["target_score_margin"] = np.nan

    # Distance-based margin: nearest alternative distance minus this distance.
    # Positive means this edge is geometrically preferred.
    for _, group in enriched.groupby(
        ["example_id", "source_tracklet_index"]
    ):
        distances = group["direct_endpoint_distance_um"].to_numpy(dtype=float)
        for local_index, dataframe_index in enumerate(group.index):
            others = np.delete(distances, local_index)
            if others.size:
                enriched.loc[dataframe_index, "source_score_margin"] = (
                    float(others.min() - distances[local_index])
                )

    for _, group in enriched.groupby(
        ["example_id", "target_tracklet_index"]
    ):
        distances = group["direct_endpoint_distance_um"].to_numpy(dtype=float)
        for local_index, dataframe_index in enumerate(group.index):
            others = np.delete(distances, local_index)
            if others.size:
                enriched.loc[dataframe_index, "target_score_margin"] = (
                    float(others.min() - distances[local_index])
                )

    enriched["mutual_best"] = (
        (enriched["source_rank"].astype(int) == 1)
        & (enriched["target_rank"].astype(int) == 1)
    )
    return enriched


def fit_pair_stats(enriched: pd.DataFrame) -> MaskedStats:
    raw, validity = tensorize_stage11_pair_features(enriched, device="cpu")
    values = raw[:, : len(STAGE11_PAIR_FEATURES)].numpy()
    valid = validity.numpy()
    return fit_masked_stats(values, valid)


# =============================================================================
# Prepared examples
# =============================================================================


@dataclass
class PreparedExample:
    example_id: int
    tracklet_indices: np.ndarray             # [N]
    structured: np.ndarray                   # [N,K,48]
    observation_mask: np.ndarray             # [N,K]
    times: np.ndarray                        # [N,K]
    start_xyz_um: np.ndarray                 # [N,3]
    end_xyz_um: np.ndarray                   # [N,3]
    reliability_raw: np.ndarray              # [N,12]
    crops: np.ndarray | None                 # [N,K,5,D,H,W], float16
    fingerprints_zero: np.ndarray | None     # [N,K,96]
    edge_index: np.ndarray                   # [E,2], local N indices
    gap_frames: np.ndarray                   # [E]
    pair_features: np.ndarray                # [E,80]
    expected_global: np.ndarray              # [E,3]
    expected_global_relative: np.ndarray     # [E,3]
    expected_local: np.ndarray               # [E,3]
    expected_backward: np.ndarray            # [E,3]
    prediction_valid: np.ndarray             # [E,4]
    edge_target: np.ndarray                  # [E]


@dataclass(frozen=True)
class FeatureStats:
    structured: MaskedStats
    reliability: MaskedStats
    pair: MaskedStats

    def to_json(self) -> dict[str, Any]:
        return {
            "structured_primitives": list(TRACKLET_STRUCTURED_PRIMITIVES),
            "structured": self.structured.to_json(),
            "reliability_features": list(TRACKLET_RELIABILITY_FEATURES),
            "reliability": self.reliability.to_json(),
            "stage11_pair_primitives": list(STAGE11_PAIR_FEATURES),
            "pair": self.pair.to_json(),
        }


def prepare_examples(
    data: SourceDataset,
    *,
    crop_cache: CropCache | None,
    appearance_mode: str,
) -> tuple[list[PreparedExample], FeatureStats]:
    motion = _motion_lookup(data.global_motion)

    volume_all = pd.to_numeric(
        data.observations.get("volume", pd.Series(dtype=float)),
        errors="coerce",
    ).to_numpy(dtype=float)
    volume_all = volume_all[np.isfinite(volume_all) & (volume_all > 0)]
    global_median_volume = (
        float(np.median(volume_all)) if volume_all.size else 1.0
    )

    # Build raw structured streams and crop valid fractions tracklet by tracklet.
    raw_by_tracklet: dict[tuple[int, int], np.ndarray] = {}
    valid_by_tracklet: dict[tuple[int, int], np.ndarray] = {}
    diagnostics_by_tracklet: dict[tuple[int, int], dict[str, float]] = {}
    crop_fraction_by_tracklet: dict[tuple[int, int], np.ndarray] = {}

    all_raw: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []

    for (example_id, tracklet_index), rows in data.observations.groupby(
        ["example_id", "tracklet_index"], sort=False
    ):
        rows = rows.sort_values("sequence_index")
        key = (int(example_id), int(tracklet_index))
        raw, valid, diagnostics = build_raw_structured(
            rows, global_motion=motion
        )
        raw_by_tracklet[key] = raw
        valid_by_tracklet[key] = valid
        diagnostics_by_tracklet[key] = diagnostics
        all_raw.append(raw)
        all_valid.append(valid)

        if crop_cache is not None:
            row_ids = rows["_observation_row"].to_numpy(dtype=np.int64)
            crop_fraction_by_tracklet[key] = np.asarray(
                crop_cache.valid_fraction[row_ids], dtype=np.float32
            )
        else:
            crop_fraction_by_tracklet[key] = np.ones(
                len(rows), dtype=np.float32
            )

    structured_stats = fit_masked_stats(
        np.concatenate(all_raw, axis=0),
        np.concatenate(all_valid, axis=0),
    )

    reliability_raw_by_tracklet: dict[tuple[int, int], np.ndarray] = {}
    reliability_all: list[np.ndarray] = []
    for (example_id, tracklet_index), rows in data.observations.groupby(
        ["example_id", "tracklet_index"], sort=False
    ):
        rows = rows.sort_values("sequence_index")
        key = (int(example_id), int(tracklet_index))
        reliability = build_reliability_raw(
            rows,
            diagnostics=diagnostics_by_tracklet[key],
            crop_valid_fraction=crop_fraction_by_tracklet[key],
            global_median_volume=global_median_volume,
        )
        reliability_raw_by_tracklet[key] = reliability
        reliability_all.append(reliability)

    reliability_stats = fit_reliability_stats(
        np.stack(reliability_all, axis=0)
    )

    enriched_edges = enrich_pair_features(
        data,
        tracklet_diagnostics=diagnostics_by_tracklet,
    )
    pair_stats = fit_pair_stats(enriched_edges)

    pair_tensor, _ = tensorize_stage11_pair_features(
        enriched_edges,
        mean=pair_stats.mean,
        std=pair_stats.std,
        device="cpu",
    )
    enriched_edges = enriched_edges.copy()
    enriched_edges["_pair_row"] = np.arange(len(enriched_edges), dtype=np.int64)
    pair_matrix = pair_tensor.numpy().astype(np.float32, copy=False)

    stats = FeatureStats(
        structured=structured_stats,
        reliability=reliability_stats,
        pair=pair_stats,
    )

    prepared: list[PreparedExample] = []
    fingerprint_dim = ReconcilerConfig().fingerprint.embedding_dim

    for example_id in sorted(data.examples["example_id"].astype(int).unique()):
        tracklets = data.tracklets[
            data.tracklets["example_id"].astype(int) == example_id
        ].sort_values("tracklet_index")
        edges = enriched_edges[
            enriched_edges["example_id"].astype(int) == example_id
        ].sort_values("edge_index")

        tracklet_indices = tracklets["tracklet_index"].astype(int).to_numpy()
        local_index = {
            int(tracklet_index): local
            for local, tracklet_index in enumerate(tracklet_indices)
        }
        n = len(tracklets)

        groups = {
            int(tracklet_index): group.sort_values("sequence_index")
            for tracklet_index, group in data.observations[
                data.observations["example_id"].astype(int) == example_id
            ].groupby("tracklet_index")
        }
        k_max = max(len(group) for group in groups.values())

        structured = np.zeros((n, k_max, 48), dtype=np.float32)
        observation_mask = np.zeros((n, k_max), dtype=bool)
        times = np.zeros((n, k_max), dtype=np.float32)
        start_xyz = np.zeros((n, 3), dtype=np.float32)
        end_xyz = np.zeros((n, 3), dtype=np.float32)
        reliability_raw = np.zeros(
            (n, len(TRACKLET_RELIABILITY_FEATURES)), dtype=np.float32
        )

        crops = None
        zero_fingerprints = None
        if appearance_mode == "crops":
            assert crop_cache is not None
            crop_shape = tuple(crop_cache.crops.shape[1:])
            crops = np.zeros((n, k_max, *crop_shape), dtype=np.float16)
        else:
            zero_fingerprints = np.zeros(
                (n, k_max, fingerprint_dim), dtype=np.float32
            )

        for local, tracklet in enumerate(tracklets.itertuples(index=False)):
            tracklet_index = int(tracklet.tracklet_index)
            key = (example_id, tracklet_index)
            rows = groups[tracklet_index]
            k = len(rows)

            normalized_structured = apply_masked_stats(
                raw_by_tracklet[key],
                valid_by_tracklet[key],
                structured_stats,
            )
            structured[local, :k] = normalized_structured
            observation_mask[local, :k] = True
            times[local, :k] = rows["frame"].to_numpy(dtype=np.float32)

            xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float32)
            start_xyz[local] = xyz[0]
            end_xyz[local] = xyz[-1]
            reliability_raw[local] = reliability_raw_by_tracklet[key]

            if crops is not None:
                row_ids = rows["_observation_row"].to_numpy(dtype=np.int64)
                crops[local, :k] = np.asarray(
                    crop_cache.crops[row_ids], dtype=np.float16
                )

        reliability = (
            reliability_raw - reliability_stats.mean[None, :]
        ) / reliability_stats.std[None, :]

        e = len(edges)
        edge_index = np.zeros((e, 2), dtype=np.int64)
        for row_number, edge in enumerate(edges.itertuples(index=False)):
            edge_index[row_number, 0] = local_index[
                int(edge.source_tracklet_index)
            ]
            edge_index[row_number, 1] = local_index[
                int(edge.target_tracklet_index)
            ]

        def xyz_from_columns(prefix: str) -> tuple[np.ndarray, np.ndarray]:
            values = np.zeros((e, 3), dtype=np.float32)
            valid = np.ones(e, dtype=bool)
            for axis, axis_name in enumerate(("z", "y", "x")):
                name = f"{prefix}_{axis_name}_um"
                if name not in edges:
                    valid[:] = False
                    continue
                column = pd.to_numeric(
                    edges[name], errors="coerce"
                ).to_numpy(dtype=np.float32, copy=True)
                valid &= np.isfinite(column)
                values[:, axis] = np.nan_to_num(column, nan=0.0)
            values[~valid] = 0.0
            return values, valid

        expected_global, global_valid = xyz_from_columns("expected_global")
        expected_global_relative, relative_valid = xyz_from_columns(
            "expected_global_relative"
        )
        expected_local = np.zeros((e, 3), dtype=np.float32)
        expected_backward = np.zeros((e, 3), dtype=np.float32)
        local_valid = np.zeros(e, dtype=bool)
        backward_valid = np.zeros(e, dtype=bool)

        pair_rows = edges["_pair_row"].to_numpy(dtype=np.int64)
        prepared.append(
            PreparedExample(
                example_id=example_id,
                tracklet_indices=tracklet_indices.astype(np.int64),
                structured=structured,
                observation_mask=observation_mask,
                times=times,
                start_xyz_um=start_xyz,
                end_xyz_um=end_xyz,
                reliability_raw=reliability.astype(np.float32),
                crops=crops,
                fingerprints_zero=zero_fingerprints,
                edge_index=edge_index,
                gap_frames=edges["gap_frames"].to_numpy(dtype=np.int64),
                pair_features=pair_matrix[pair_rows],
                expected_global=expected_global,
                expected_global_relative=expected_global_relative,
                expected_local=expected_local,
                expected_backward=expected_backward,
                prediction_valid=np.stack(
                    (
                        global_valid,
                        relative_valid,
                        local_valid,
                        backward_valid,
                    ),
                    axis=-1,
                ),
                edge_target=edges["continuation_target"].to_numpy(
                    dtype=np.float32
                ),
            )
        )

    return prepared, stats


# =============================================================================
# Collation
# =============================================================================


@dataclass
class TrainingBatch:
    reconciliation: ReconciliationBatch
    edge_target: torch.Tensor
    example_ids: list[int]


def collate(
    examples: list[PreparedExample],
    *,
    device: torch.device,
    appearance_mode: str,
) -> TrainingBatch:
    b = len(examples)
    n_max = max(example.structured.shape[0] for example in examples)
    k_max = max(example.structured.shape[1] for example in examples)
    e_max = max(example.edge_index.shape[0] for example in examples)

    structured = torch.zeros(
        b, n_max, k_max, 48, dtype=torch.float32, device=device
    )
    observation_mask = torch.zeros(
        b, n_max, k_max, dtype=torch.bool, device=device
    )
    tracklet_mask = torch.zeros(
        b, n_max, dtype=torch.bool, device=device
    )
    times = torch.zeros(
        b, n_max, k_max, dtype=torch.float32, device=device
    )
    start_xyz = torch.zeros(
        b, n_max, 3, dtype=torch.float32, device=device
    )
    end_xyz = torch.zeros(
        b, n_max, 3, dtype=torch.float32, device=device
    )
    reliability = torch.zeros(
        b, n_max, 12, dtype=torch.float32, device=device
    )

    crops_tensor = None
    fingerprint_tensor = None
    if appearance_mode == "crops":
        crop_shape = examples[0].crops.shape[2:]  # type: ignore[union-attr]
        crops_tensor = torch.zeros(
            b,
            n_max,
            k_max,
            *crop_shape,
            dtype=torch.float32,
            device=device,
        )
    else:
        fingerprint_tensor = torch.zeros(
            b,
            n_max,
            k_max,
            ReconcilerConfig().fingerprint.embedding_dim,
            dtype=torch.float32,
            device=device,
        )

    edge_index = torch.zeros(
        b, e_max, 2, dtype=torch.long, device=device
    )
    edge_mask = torch.zeros(
        b, e_max, dtype=torch.bool, device=device
    )
    gap_frames = torch.zeros(
        b, e_max, dtype=torch.long, device=device
    )
    pair_features = torch.zeros(
        b, e_max, 80, dtype=torch.float32, device=device
    )

    expected_global = torch.zeros(
        b, e_max, 3, dtype=torch.float32, device=device
    )
    expected_relative = torch.zeros_like(expected_global)
    expected_local = torch.zeros_like(expected_global)
    expected_backward = torch.zeros_like(expected_global)
    prediction_valid = torch.zeros(
        b, e_max, 4, dtype=torch.bool, device=device
    )
    edge_target = torch.zeros(
        b, e_max, dtype=torch.float32, device=device
    )

    for batch_index, example in enumerate(examples):
        n, k = example.structured.shape[:2]
        e = len(example.edge_index)

        structured[batch_index, :n, :k] = torch.from_numpy(
            example.structured
        ).to(device)
        observation_mask[batch_index, :n, :k] = torch.from_numpy(
            example.observation_mask
        ).to(device)
        tracklet_mask[batch_index, :n] = True
        times[batch_index, :n, :k] = torch.from_numpy(
            example.times
        ).to(device)
        start_xyz[batch_index, :n] = torch.from_numpy(
            example.start_xyz_um
        ).to(device)
        end_xyz[batch_index, :n] = torch.from_numpy(
            example.end_xyz_um
        ).to(device)
        reliability[batch_index, :n] = torch.from_numpy(
            example.reliability_raw
        ).to(device)

        if crops_tensor is not None:
            assert example.crops is not None
            crops_tensor[batch_index, :n, :k] = torch.from_numpy(
                np.asarray(example.crops, dtype=np.float32)
            ).to(device)
        else:
            assert fingerprint_tensor is not None
            assert example.fingerprints_zero is not None
            fingerprint_tensor[batch_index, :n, :k] = torch.from_numpy(
                example.fingerprints_zero
            ).to(device)

        edge_index[batch_index, :e] = torch.from_numpy(
            example.edge_index
        ).to(device)
        edge_mask[batch_index, :e] = True
        gap_frames[batch_index, :e] = torch.from_numpy(
            np.array(example.gap_frames, copy=True)
        ).to(device)
        pair_features[batch_index, :e] = torch.from_numpy(
            example.pair_features
        ).to(device)
        expected_global[batch_index, :e] = torch.from_numpy(
            example.expected_global
        ).to(device)
        expected_relative[batch_index, :e] = torch.from_numpy(
            example.expected_global_relative
        ).to(device)
        expected_local[batch_index, :e] = torch.from_numpy(
            example.expected_local
        ).to(device)
        expected_backward[batch_index, :e] = torch.from_numpy(
            example.expected_backward
        ).to(device)
        prediction_valid[batch_index, :e] = torch.from_numpy(
            example.prediction_valid
        ).to(device)
        edge_target[batch_index, :e] = torch.from_numpy(
            example.edge_target
        ).to(device)

    tracklet_batch = TrackletBatch(
        structured=structured,
        observation_mask=observation_mask,
        tracklet_mask=tracklet_mask,
        times=times,
        start_xyz_um=start_xyz,
        end_xyz_um=end_xyz,
        reliability=reliability,
        crops=crops_tensor,
        fingerprints=fingerprint_tensor,
    )
    edge_batch = CandidateEdgeBatch(
        edge_index=edge_index,
        edge_mask=edge_mask,
        gap_frames=gap_frames,
        pair_features=pair_features,
        expected_global_xyz_um=expected_global,
        expected_global_relative_xyz_um=expected_relative,
        expected_local_xyz_um=expected_local,
        expected_backward_source_xyz_um=expected_backward,
        prediction_valid=prediction_valid,
    )
    return TrainingBatch(
        reconciliation=ReconciliationBatch(
            tracklets=tracklet_batch,
            edges=edge_batch,
            divisions=None,
        ),
        edge_target=edge_target,
        example_ids=[example.example_id for example in examples],
    )


# =============================================================================
# Metrics
# =============================================================================


@dataclass(frozen=True)
class Metrics:
    loss: float
    edge_accuracy: float
    source_top1: float
    target_top1: float
    exact_assignment: float
    positive_probability_mean: float
    negative_probability_mean: float
    negative_probability_max: float
    source_margin_mean: float
    target_margin_mean: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _top1_and_margin(
    logits: np.ndarray,
    edge_index: np.ndarray,
    target: np.ndarray,
    *,
    group_column: int,
) -> tuple[int, int, list[float]]:
    correct = 0
    total = 0
    margins: list[float] = []
    for group_id in np.unique(edge_index[:, group_column]):
        mask = edge_index[:, group_column] == group_id
        indices = np.flatnonzero(mask)
        positive = indices[target[indices] > 0.5]
        if len(positive) != 1:
            continue
        total += 1
        best = indices[np.argmax(logits[indices])]
        correct += int(best == positive[0])

        negatives = indices[target[indices] <= 0.5]
        if len(negatives):
            margins.append(
                float(logits[positive[0]] - np.max(logits[negatives]))
            )
    return correct, total, margins


def _exact_assignment(
    logits: np.ndarray,
    edge_index: np.ndarray,
    target: np.ndarray,
) -> bool:
    sources = sorted(np.unique(edge_index[:, 0]).tolist())
    targets = sorted(np.unique(edge_index[:, 1]).tolist())
    if len(sources) != len(targets):
        return False
    source_map = {value: index for index, value in enumerate(sources)}
    target_map = {value: index for index, value in enumerate(targets)}
    utility = np.full(
        (len(sources), len(targets)),
        -1e6,
        dtype=np.float64,
    )
    positive = np.zeros_like(utility, dtype=bool)
    for score, edge, label in zip(logits, edge_index, target):
        i = source_map[int(edge[0])]
        j = target_map[int(edge[1])]
        utility[i, j] = float(score)
        positive[i, j] = bool(label > 0.5)

    rows, cols = linear_sum_assignment(-utility)
    return bool(
        len(rows) == len(sources)
        and all(positive[row, col] for row, col in zip(rows, cols))
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    examples: list[PreparedExample],
    *,
    device: torch.device,
    appearance_mode: str,
    amp_mode: str,
) -> Metrics:
    model.eval()

    losses: list[float] = []
    edge_correct = 0
    edge_total = 0
    source_correct = 0
    source_total = 0
    target_correct = 0
    target_total = 0
    exact = 0
    positive_probabilities: list[float] = []
    negative_probabilities: list[float] = []
    source_margins: list[float] = []
    target_margins: list[float] = []

    for example in examples:
        batch = collate(
            [example],
            device=device,
            appearance_mode=appearance_mode,
        )
        with autocast_context(device, amp_mode):
            output = model(batch.reconciliation)
            loss = focal_binary_probability_loss(
                output.parental_probabilities,
                batch.edge_target,
                batch.reconciliation.edges.edge_mask,
                gamma=2.0,
            )
        losses.append(float(loss.detach().float().cpu()))

        e = len(example.edge_index)
        logits = (
            output.continuation_logits[0, :e]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        probabilities = (
            output.parental_probabilities[0, :e]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        target = example.edge_target.astype(np.float32)

        prediction = probabilities >= 0.5
        edge_correct += int(
            np.sum(prediction == (target > 0.5))
        )
        edge_total += e

        sc, st, sm = _top1_and_margin(
            logits, example.edge_index, target, group_column=0
        )
        tc, tt, tm = _top1_and_margin(
            logits, example.edge_index, target, group_column=1
        )
        source_correct += sc
        source_total += st
        target_correct += tc
        target_total += tt
        source_margins.extend(sm)
        target_margins.extend(tm)

        exact += int(
            _exact_assignment(logits, example.edge_index, target)
        )
        positive_probabilities.extend(
            probabilities[target > 0.5].tolist()
        )
        negative_probabilities.extend(
            probabilities[target <= 0.5].tolist()
        )

    neg = np.asarray(negative_probabilities, dtype=float)
    pos = np.asarray(positive_probabilities, dtype=float)

    return Metrics(
        loss=float(np.mean(losses)),
        edge_accuracy=edge_correct / max(edge_total, 1),
        source_top1=source_correct / max(source_total, 1),
        target_top1=target_correct / max(target_total, 1),
        exact_assignment=exact / max(len(examples), 1),
        positive_probability_mean=float(pos.mean()) if pos.size else float("nan"),
        negative_probability_mean=float(neg.mean()) if neg.size else float("nan"),
        negative_probability_max=float(neg.max()) if neg.size else float("nan"),
        source_margin_mean=(
            float(np.mean(source_margins))
            if source_margins
            else float("nan")
        ),
        target_margin_mean=(
            float(np.mean(target_margins))
            if target_margins
            else float("nan")
        ),
    )


# =============================================================================
# AMP / training utilities
# =============================================================================


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def resolved_amp_mode(device: torch.device, requested: str) -> str:
    requested = requested.lower()
    if device.type != "cuda":
        return "off"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return requested


def autocast_context(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "off":
        return _NullContext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def disable_all_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0
        if hasattr(module, "probability") and module.__class__.__name__ == "BranchDropout":
            module.probability = 0.0


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Metrics,
    stats: FeatureStats,
    args: argparse.Namespace,
    data: SourceDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics.as_dict(),
        "feature_stats": stats.to_json(),
        "args": vars(args),
        "dataset": root_relative(data.root),
        "dataset_manifest": data.manifest,
    }
    torch.save(payload, path)


def print_metrics(step: int, metrics: Metrics, *, prefix: str = "eval") -> None:
    print(
        f"[{prefix} step={step:04d}] "
        f"loss={metrics.loss:.5f} "
        f"exact={100.0 * metrics.exact_assignment:6.2f}% "
        f"src_top1={100.0 * metrics.source_top1:6.2f}% "
        f"tgt_top1={100.0 * metrics.target_top1:6.2f}% "
        f"edge={100.0 * metrics.edge_accuracy:6.2f}% "
        f"p+={metrics.positive_probability_mean:.4f} "
        f"p-mean={metrics.negative_probability_mean:.4f} "
        f"p-max={metrics.negative_probability_max:.4f} "
        f"src_margin={metrics.source_margin_mean:+.3f} "
        f"tgt_margin={metrics.target_margin_mean:+.3f}",
        flush=True,
    )


def success(metrics: Metrics) -> bool:
    return bool(
        metrics.exact_assignment >= 1.0
        and metrics.source_top1 >= 1.0
        and metrics.target_top1 >= 1.0
        and metrics.positive_probability_mean >= STRICT_POSITIVE_PROBABILITY
        and metrics.negative_probability_max <= STRICT_NEGATIVE_PROBABILITY
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end continuation memorization on the ambiguous synthetic "
            "Trackastra cut dataset."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda, cpu, or auto",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "off", "fp16", "bf16"),
        default="off",
        help=(
            "Autocast mode. Default off for the first numerical-debug overfit; "
            "bf16 is recommended if VRAM/speed requires AMP."
        ),
    )
    parser.add_argument(
        "--appearance-mode",
        choices=("crops", "zeros"),
        default="crops",
        help=(
            "crops trains the 3-D fingerprint CNN end-to-end; zeros is a "
            "structured/motion-only debugging ablation."
        ),
    )
    parser.add_argument(
        "--crop-shape",
        default="13,41,41",
        help="Odd native Z,Y,X crop shape.",
    )
    parser.add_argument(
        "--distance-clip-um",
        type=float,
        default=DEFAULT_DISTANCE_CLIP_UM,
    )
    parser.add_argument("--rebuild-crops", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "batch_size", "eval_every", "log_every"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    for name in ("lr", "grad_clip", "distance_clip_um"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if float(args.weight_decay) < 0:
        raise ValueError("--weight-decay must be >= 0")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(int(args.seed))

    dataset_root = (
        resolve(args.dataset)
        if args.dataset is not None
        else default_dataset(args.sample_id)
    )
    output = (
        resolve(args.output)
        if args.output is not None
        else default_output(args.sample_id)
    )
    crop_shape = parse_triplet_int(args.crop_shape, name="crop-shape")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    amp_mode = resolved_amp_mode(device, args.amp)
    if amp_mode == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 AMP requested but the CUDA device does not support bf16")

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if (checkpoint_dir / "best.pt").is_file() and not args.overwrite:
        raise FileExistsError(
            f"Existing overfit checkpoint found at {checkpoint_dir / 'best.pt'}. "
            "Pass --overwrite to start a fresh experiment."
        )

    data = SourceDataset.load(dataset_root)
    example_count = int(data.examples["example_id"].nunique())

    print("=" * 122, flush=True)
    print("TRACK RECONCILER — INVESTIGATION 03: CONTINUATION OVERFIT", flush=True)
    print("=" * 122, flush=True)
    print(f"dataset          : {data.root}", flush=True)
    print(f"examples         : {example_count}", flush=True)
    print(f"edges            : {len(data.edges)}", flush=True)
    print(f"appearance       : {args.appearance_mode}", flush=True)
    print(f"crop shape       : {crop_shape}", flush=True)
    print(f"device           : {device}", flush=True)
    print(f"AMP              : {amp_mode}", flush=True)
    print(f"steps            : {args.steps}", flush=True)
    print(f"batch size       : {args.batch_size}", flush=True)
    print(f"learning rate    : {args.lr:g}", flush=True)
    print("=" * 122, flush=True)

    crop_cache = None
    if args.appearance_mode == "crops":
        crop_cache = build_crop_cache(
            data,
            cache_dir=output / "cache",
            shape_zyx=crop_shape,
            distance_clip_um=float(args.distance_clip_um),
            rebuild=bool(args.rebuild_crops),
        )

    print("[features] tensorizing tracklet/pair evidence ...", flush=True)
    examples, stats = prepare_examples(
        data,
        crop_cache=crop_cache,
        appearance_mode=args.appearance_mode,
    )
    if not examples:
        raise RuntimeError("No training examples were prepared")
    atomic_json(output / "feature_stats.json", stats.to_json())

    config = ReconcilerConfig()
    model = TrackletReconciliationNetwork(config).to(device)
    disable_all_dropout(model)

    if args.appearance_mode == "zeros":
        for parameter in model.tracklets.fingerprint.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # GradScaler is useful only for fp16. bf16 has enough exponent range.
    scaler = None
    if device.type == "cuda" and amp_mode == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[model] trainable parameters={trainable_parameter_count(model):,}",
        flush=True,
    )

    initial = evaluate(
        model,
        examples,
        device=device,
        appearance_mode=args.appearance_mode,
        amp_mode=amp_mode,
    )
    print_metrics(0, initial, prefix="initial")

    rng = np.random.default_rng(int(args.seed))
    order = np.arange(len(examples), dtype=np.int64)
    cursor = len(order)
    running_loss = 0.0
    running_count = 0
    best_exact = -1.0
    best_loss = float("inf")
    strict_streak = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    def next_indices() -> np.ndarray:
        nonlocal cursor, order
        batch_size = min(int(args.batch_size), len(examples))
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        result = order[cursor: cursor + batch_size]
        cursor += batch_size
        return result

    for step in range(1, int(args.steps) + 1):
        model.train()
        indices = next_indices()
        selected = [examples[int(index)] for index in indices]
        batch = collate(
            selected,
            device=device,
            appearance_mode=args.appearance_mode,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_mode):
            output_pred = model(batch.reconciliation)
            loss = focal_binary_probability_loss(
                output_pred.parental_probabilities,
                batch.edge_target,
                batch.reconciliation.edges.edge_mask,
                gamma=2.0,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip)
            )
            optimizer.step()

        loss_value = float(loss.detach().float().cpu())
        running_loss += loss_value
        running_count += 1

        if step % int(args.log_every) == 0:
            mean_running = running_loss / max(running_count, 1)
            elapsed = time.perf_counter() - started
            if device.type == "cuda":
                peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                memory_text = f" peakVRAM={peak_gib:.2f}GiB"
            else:
                memory_text = ""
            print(
                f"[train step={step:04d}] "
                f"loss={mean_running:.5f} "
                f"time={elapsed:.1f}s{memory_text}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

        if (
            step % int(args.eval_every) == 0
            or step == int(args.steps)
        ):
            metrics = evaluate(
                model,
                examples,
                device=device,
                appearance_mode=args.appearance_mode,
                amp_mode=amp_mode,
            )
            print_metrics(step, metrics)
            history.append(
                {
                    "step": int(step),
                    **metrics.as_dict(),
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
            )
            atomic_json(output / "history.json", history)

            better = (
                metrics.exact_assignment > best_exact
                or (
                    metrics.exact_assignment == best_exact
                    and metrics.loss < best_loss
                )
            )
            if better:
                best_exact = metrics.exact_assignment
                best_loss = metrics.loss
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    metrics=metrics,
                    stats=stats,
                    args=args,
                    data=data,
                )

            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                metrics=metrics,
                stats=stats,
                args=args,
                data=data,
            )

            if success(metrics):
                strict_streak += 1
                print(
                    f"[success] strict memorization check "
                    f"{strict_streak}/{STRICT_SUCCESS_EVALS}",
                    flush=True,
                )
            else:
                strict_streak = 0

            if strict_streak >= STRICT_SUCCESS_EVALS:
                print(
                    "[success] continuation overfit reached exact assignment "
                    "and strict probability separation.",
                    flush=True,
                )
                break

    final = evaluate(
        model,
        examples,
        device=device,
        appearance_mode=args.appearance_mode,
        amp_mode=amp_mode,
    )
    elapsed = time.perf_counter() - started
    print("", flush=True)
    print("=" * 122, flush=True)
    print("FINAL CONTINUATION OVERFIT", flush=True)
    print("=" * 122, flush=True)
    print_metrics(step, final, prefix="final")
    print(f"elapsed          : {elapsed:.1f}s", flush=True)
    print(f"best checkpoint  : {checkpoint_dir / 'best.pt'}", flush=True)
    print(f"latest checkpoint: {checkpoint_dir / 'latest.pt'}", flush=True)

    summary = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "dataset": root_relative(data.root),
        "output": root_relative(output),
        "device": str(device),
        "amp": amp_mode,
        "appearance_mode": args.appearance_mode,
        "crop_shape_zyx": list(crop_shape),
        "steps_completed": int(step),
        "elapsed_seconds": float(elapsed),
        "final_metrics": final.as_dict(),
        "strict_success": bool(success(final)),
        "best_checkpoint": "checkpoints/best.pt",
        "latest_checkpoint": "checkpoints/latest.pt",
    }
    atomic_json(output / "summary.json", summary)

    if final.exact_assignment < 1.0:
        print(
            "[diagnostic] The model has not yet memorized every component. "
            "Do not move to the full 20-frame overfit yet.",
            flush=True,
        )
    elif (
        final.positive_probability_mean < STRICT_POSITIVE_PROBABILITY
        or final.negative_probability_max > STRICT_NEGATIVE_PROBABILITY
    ):
        print(
            "[diagnostic] Assignment is exact, but probability separation is "
            "not yet strict. Continue training before treating the pipeline as "
            "fully memorized.",
            flush=True,
        )
    else:
        print(
            "[diagnostic] Exact local assignment and strong probability "
            "separation achieved. This validates the first continuation path.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
