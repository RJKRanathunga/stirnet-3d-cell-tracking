from __future__ import annotations

r"""
Investigation 08 — Fluo-N3DH-CE online synthetic continuation training.

Purpose
-------
Train the current learned/track_reconciler continuation core from CLEAN CTC
tracking ground truth only.

The CE volume is never duplicated.  We first extract one compact observation
table from CTC TRA ground truth, then synthesize fresh reconciliation scenes
online by arbitrarily breaking true tracks.  Thousands/millions of different
training variants can therefore be generated without copying the 3-D movie.

Both CE sequences 01 and 02 participate in training.  A fixed fraction of
TRACK IDENTITIES is reserved independently from each sequence for validation;
reserved identities are never used in a training scene.

Important scientific scope
--------------------------
This first CE trainer intentionally trains:

    structured temporal stream
    reliability fusion
    explicit motion relations
    Stage-11-style primitive pair evidence
    edge-centric geometric Transformer
    continuation head / parental softmax

It intentionally DOES NOT train:

    3-D image fingerprint CNN
    appearance temporal stream
    division head / division prior
    appearance / termination event heads

Why:
- CE appearance/mask geometry is not yet audited for transfer to BioHub.
- CE physical sampling differs strongly from BioHub.
- The useful supervision we trust today is TRA identity + lineage continuity.
- Disabling image features makes arbitrary track-break synthesis extremely
  cheap and lets us isolate whether the continuation core learns real CE motion.

The disabled appearance stream is zeroed and frozen in the saved checkpoint,
so loading this checkpoint in the normal model still yields a zero appearance
branch.  A later BioHub fine-tuning stage can re-initialize/unfreeze appearance.

Input layout
------------
Default:

data/external/Fluo-N3DH-CE/Fluo-N3DH-CE_train/
    01/
    01_GT/TRA/
        man_track.txt
        man_track000.tif
        ...
    02/
    02_GT/TRA/
        man_track.txt
        man_track000.tif
        ...

Only *_GT/TRA is required by this investigation.  Raw frames are not read.

The script automatically ignores raw-only tail frames with no TRA ground truth.
Thus if sequence 01 has raw frames after the last available TRA frame, those
frames simply do not enter training.

Synthetic scene
---------------
For one cut frame t and gap g:

    original GT
      A A A A A A A A
      B B B B B B B B
      C C C C C C C C

becomes a local reconciliation component such as

    source tracklets                 target tracklets

      A0 A0 A0 |                    | A1 A1 A1
      B0 B0    |   artificial gap   | B1 B1 B1 B1
      C0 C0 C0 |                    | C1 C1

Candidate A0->A1 is positive.
Nearby cross-identity candidates A0->B1, A0->C1, ... are negatives.

Source and target ordering is independently shuffled so "diagonal edge" is
never a label shortcut.  Hard scenes preferentially use spatially close peers.

Typical run
-----------
From repository root:

    python .\investigations\track_reconciler\08_ce_synthetic_continuation_training.py --overwrite

Quick smoke run:

    python .\investigations\track_reconciler\08_ce_synthetic_continuation_training.py ^
        --steps 100 --stats-scenes 64 --validation-scenes 32 ^
        --train-eval-scenes 32 --overwrite

Prepare/audit only:

    python .\investigations\track_reconciler\08_ce_synthetic_continuation_training.py ^
        --prepare-only --overwrite

Rebuild the cached CE TRA observation table:

    python .\investigations\track_reconciler\08_ce_synthetic_continuation_training.py ^
        --rebuild-base --overwrite
"""

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_NAME = "08_ce_synthetic_continuation_training"

DEFAULT_SPACING_ZYX_UM = (1.0, 0.09, 0.09)

DEFAULT_MAX_GAP = 4
DEFAULT_MIN_HISTORY = 2
DEFAULT_MAX_HISTORY = 4
DEFAULT_MIN_FUTURE = 2
DEFAULT_MAX_FUTURE = 4
DEFAULT_TRACKS_PER_SCENE = 4
DEFAULT_SCENE_RADIUS_UM = 20.0
DEFAULT_HARD_FRACTION = 0.70
DEFAULT_HARD_PROPOSALS = 8

DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_STATS_SCENES = 512
DEFAULT_VALIDATION_SCENES = 256
DEFAULT_TRAIN_EVAL_SCENES = 128

DEFAULT_STEPS = 5000
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_LOG_EVERY = 25
DEFAULT_EVAL_EVERY = 250
DEFAULT_SEED = 20260827

PAIR_FEATURE_NAMES = (
    "gap_frames",
    "direct_endpoint_distance_um",
    "hard_search_radius_um",
    "forward_error_um",
    "forward_history_count",
    "forward_used_global_motion",
    "forward_used_relative_velocity",
    "forward_uncertainty_um",
    "volume_log_error",
    "target_real_observation_count",
    "candidate_quality_score",
    "source_candidate_count",
    "target_predecessor_count",
    "source_rank",
    "target_rank",
    "source_score_margin",
    "target_score_margin",
    "mutual_best",
    "effective_pair_volume",
    "small_cell_history_exception",
)


# =============================================================================
# Repository helpers
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.parent, *here.parents, Path.cwd().resolve())
    for candidate in candidates:
        if (
            (candidate / "learned" / "track_reconciler").is_dir()
            and (candidate / "investigations" / "track_reconciler").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def root_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def default_dataset_root() -> Path:
    nested = (
        ROOT
        / "data"
        / "external"
        / "Fluo-N3DH-CE"
        / "Fluo-N3DH-CE_train"
    ).resolve()
    if nested.is_dir():
        return nested

    flat = (ROOT / "data" / "external" / "Fluo-N3DH-CE").resolve()
    return flat


def default_output() -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "investigations"
        / SCRIPT_NAME
    ).resolve()


def default_base_cache() -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "cache"
        / "fluo_n3dh_ce_tra"
    ).resolve()


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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_trainer_modules():
    inv03 = load_module(
        ROOT / "investigations" / "track_reconciler" / "03_continuation_overfit.py",
        "_inv08_inv03",
    )
    inv04 = load_module(
        ROOT / "investigations" / "track_reconciler" / "04_hard_continuation_overfit.py",
        "_inv08_inv04",
    )
    inv04.patch_inv03(inv03)
    return inv03, inv04


# =============================================================================
# CTC discovery / base observation extraction
# =============================================================================


@dataclass(frozen=True)
class SequencePaths:
    sequence: str
    root: Path
    tra_dir: Path
    track_txt: Path


def sequence_paths(dataset_root: Path, sequence: str) -> SequencePaths:
    tra_dir = dataset_root / f"{sequence}_GT" / "TRA"
    track_txt = tra_dir / "man_track.txt"
    if not tra_dir.is_dir():
        raise FileNotFoundError(f"TRA GT directory not found: {tra_dir}")
    if not track_txt.is_file():
        raise FileNotFoundError(f"CTC lineage file not found: {track_txt}")
    return SequencePaths(
        sequence=sequence,
        root=dataset_root,
        tra_dir=tra_dir,
        track_txt=track_txt,
    )


def discover_tra_frames(paths: SequencePaths) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in paths.tra_dir.glob("man_track*.tif*"):
        stem = path.stem
        suffix = stem.replace("man_track", "", 1)
        if not suffix.isdigit():
            continue
        frame = int(suffix)
        if frame in files:
            raise RuntimeError(
                f"Duplicate TRA frame {frame}: {files[frame]} and {path}"
            )
        files[frame] = path
    if not files:
        raise FileNotFoundError(f"No man_track*.tif files in {paths.tra_dir}")
    return dict(sorted(files.items()))


def load_track_metadata(paths: SequencePaths) -> pd.DataFrame:
    table = np.loadtxt(paths.track_txt, dtype=np.int64, ndmin=2)
    if table.ndim != 2 or table.shape[1] < 4:
        raise ValueError(
            f"{paths.track_txt} must contain CTC columns "
            "[track_id, start_frame, end_frame, parent_track_id]"
        )
    frame = pd.DataFrame(
        table[:, :4],
        columns=(
            "track_id",
            "start_frame",
            "end_frame",
            "parent_track_id",
        ),
    )
    frame.insert(0, "sequence", paths.sequence)
    for column in (
        "track_id",
        "start_frame",
        "end_frame",
        "parent_track_id",
    ):
        frame[column] = frame[column].astype(np.int64)
    if frame["track_id"].duplicated().any():
        duplicated = frame.loc[
            frame["track_id"].duplicated(), "track_id"
        ].tolist()
        raise ValueError(
            f"Duplicate track IDs in {paths.track_txt}: {duplicated[:10]}"
        )
    return frame


def read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("tifffile is required for Investigation 08") from exc

    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D TRA TIFF at {path}, got {array.shape}")
    return array


def frame_observations(
    *,
    sequence: str,
    frame: int,
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    valid_track_ids: set[int],
) -> list[dict[str, Any]]:
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise RuntimeError("scipy is required for CE GT centroid extraction") from exc

    ids = np.unique(labels)
    ids = ids[ids > 0]
    if not len(ids):
        return []

    ids = np.asarray(
        [int(value) for value in ids if int(value) in valid_track_ids],
        dtype=np.int64,
    )
    if not len(ids):
        return []

    # Geometric centers of the labelled TRA regions.
    support = labels > 0
    centers = ndi.center_of_mass(
        support,
        labels=labels,
        index=ids.tolist(),
    )

    counts = np.bincount(labels.reshape(-1).astype(np.int64, copy=False))
    shape = np.asarray(labels.shape, dtype=np.float64)
    spacing_arr = np.asarray(spacing, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for track_id, center in zip(ids.tolist(), centers):
        xyz = np.asarray(center, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        volume = (
            float(counts[int(track_id)])
            if int(track_id) < len(counts)
            else float("nan")
        )
        physical = xyz * spacing_arr
        lower = xyz * spacing_arr
        upper = (shape - 1.0 - xyz) * spacing_arr
        boundary = float(np.min(np.concatenate((lower, upper))))
        rows.append(
            {
                "sequence": str(sequence),
                "frame": int(frame),
                "track_id": int(track_id),
                "cell_id": int(track_id),
                "z": float(xyz[0]),
                "y": float(xyz[1]),
                "x": float(xyz[2]),
                "z_um": float(physical[0]),
                "y_um": float(physical[1]),
                "x_um": float(physical[2]),
                "volume": float(volume),
                "distance_to_boundary_um": float(boundary),
                # Intentionally unavailable in this first TRA-only phase.
                "axis_major": np.nan,
                "axis_middle": np.nan,
                "axis_minor": np.nan,
                "elongation": np.nan,
                "flatness": np.nan,
                "anisotropy": np.nan,
                "intensity_mean": np.nan,
            }
        )
    return rows


def build_base_cache(
    dataset_root: Path,
    cache_root: Path,
    *,
    spacing: tuple[float, float, float],
    rebuild: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[int, Path]]]:
    manifest_path = cache_root / "manifest.json"
    observations_path = cache_root / "observations.csv"
    tracks_path = cache_root / "tracks.csv"

    expected_manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "spacing_zyx_um": list(spacing),
        "sequences": ["01", "02"],
        "source": "CTC *_GT/TRA only",
    }

    tra_files: dict[str, dict[int, Path]] = {}
    for sequence in ("01", "02"):
        paths = sequence_paths(dataset_root, sequence)
        tra_files[sequence] = discover_tra_frames(paths)

    if (
        not rebuild
        and manifest_path.is_file()
        and observations_path.is_file()
        and tracks_path.is_file()
    ):
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual == expected_manifest:
                observations = pd.read_csv(observations_path)
                tracks = pd.read_csv(tracks_path)
                print(
                    f"[base] reusing CE TRA cache: {cache_root} | "
                    f"observations={len(observations):,}",
                    flush=True,
                )
                return observations, tracks, tra_files
        except Exception:
            pass

    cache_root.mkdir(parents=True, exist_ok=True)

    all_tracks: list[pd.DataFrame] = []
    all_observations: list[dict[str, Any]] = []

    for sequence in ("01", "02"):
        paths = sequence_paths(dataset_root, sequence)
        tracks = load_track_metadata(paths)
        all_tracks.append(tracks)
        valid_ids = set(tracks["track_id"].astype(int).tolist())
        files = tra_files[sequence]

        print(
            f"[base] sequence {sequence}: TRA frames "
            f"{min(files)}..{max(files)} ({len(files)}) | "
            f"declared tracks={len(tracks):,}",
            flush=True,
        )

        for index, (frame, path) in enumerate(files.items(), start=1):
            labels = read_tiff(path)
            all_observations.extend(
                frame_observations(
                    sequence=sequence,
                    frame=frame,
                    labels=labels,
                    spacing=spacing,
                    valid_track_ids=valid_ids,
                )
            )
            if index % 25 == 0 or index == len(files):
                print(
                    f"[base] sequence {sequence}: "
                    f"{index}/{len(files)} TRA frames parsed",
                    flush=True,
                )

    observations = pd.DataFrame(all_observations)
    tracks = pd.concat(all_tracks, ignore_index=True)

    if observations.empty:
        raise RuntimeError("No CE tracking observations were extracted.")

    observations = observations.sort_values(
        ["sequence", "track_id", "frame"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Keep only labels inside the declared CTC lifespan for that identity.
    bounds = tracks[
        ["sequence", "track_id", "start_frame", "end_frame"]
    ].copy()
    observations = observations.merge(
        bounds,
        on=["sequence", "track_id"],
        how="left",
        validate="many_to_one",
    )
    bad_span = (
        observations["start_frame"].isna()
        | (observations["frame"] < observations["start_frame"])
        | (observations["frame"] > observations["end_frame"])
    )
    if bad_span.any():
        print(
            f"[base warning] discarding {int(bad_span.sum())} TRA labels "
            "outside man_track.txt declared spans.",
            flush=True,
        )
        observations = observations.loc[~bad_span].copy()

    observations = observations.drop(
        columns=["start_frame", "end_frame"]
    ).reset_index(drop=True)

    atomic_csv(observations_path, observations)
    atomic_csv(tracks_path, tracks)
    atomic_json(manifest_path, expected_manifest)

    return observations, tracks, tra_files


# =============================================================================
# Compact track store
# =============================================================================


@dataclass
class TrackSeries:
    sequence: str
    track_id: int
    frames: np.ndarray
    xyz_um: np.ndarray
    xyz_vox: np.ndarray
    volume: np.ndarray
    boundary_um: np.ndarray

    def index_of(self, frame: int) -> int | None:
        i = int(np.searchsorted(self.frames, int(frame)))
        if i >= len(self.frames) or int(self.frames[i]) != int(frame):
            return None
        return i

    def has_range(self, first: int, last: int) -> bool:
        if first > last:
            return False
        i = self.index_of(first)
        j = self.index_of(last)
        if i is None or j is None:
            return False
        expected = last - first + 1
        if j - i + 1 != expected:
            return False
        return bool(
            np.array_equal(
                self.frames[i : j + 1],
                np.arange(first, last + 1, dtype=np.int64),
            )
        )

    def rows(self, first: int, last: int, *, role: str) -> pd.DataFrame:
        i = self.index_of(first)
        j = self.index_of(last)
        if i is None or j is None or not self.has_range(first, last):
            raise KeyError(
                f"Track {self.sequence}:{self.track_id} has no contiguous "
                f"range {first}..{last}"
            )
        indices = np.arange(i, j + 1)
        n = len(indices)
        return pd.DataFrame(
            {
                "sequence_index": np.arange(n, dtype=np.int64),
                "sequence": self.sequence,
                "frame": self.frames[indices],
                "track_id": self.track_id,
                "cell_id": self.track_id,
                "role": role,
                "z_um": self.xyz_um[indices, 0],
                "y_um": self.xyz_um[indices, 1],
                "x_um": self.xyz_um[indices, 2],
                "z": self.xyz_vox[indices, 0],
                "y": self.xyz_vox[indices, 1],
                "x": self.xyz_vox[indices, 2],
                "volume": self.volume[indices],
                "distance_to_boundary_um": self.boundary_um[indices],
                "axis_major": np.nan,
                "axis_middle": np.nan,
                "axis_minor": np.nan,
                "elongation": np.nan,
                "flatness": np.nan,
                "anisotropy": np.nan,
                "intensity_mean": np.nan,
            }
        )

    def xyz_at(self, frame: int) -> np.ndarray:
        i = self.index_of(frame)
        if i is None:
            raise KeyError((self.sequence, self.track_id, frame))
        return self.xyz_um[i].astype(np.float64, copy=True)

    def volume_at(self, frame: int) -> float:
        i = self.index_of(frame)
        if i is None:
            return float("nan")
        return float(self.volume[i])


def build_track_store(observations: pd.DataFrame) -> dict[tuple[str, int], TrackSeries]:
    store: dict[tuple[str, int], TrackSeries] = {}
    for (sequence, track_id), group in observations.groupby(
        ["sequence", "track_id"],
        sort=False,
    ):
        group = group.sort_values("frame")
        key = (str(sequence).zfill(2), int(track_id))
        store[key] = TrackSeries(
            sequence=key[0],
            track_id=key[1],
            frames=group["frame"].to_numpy(dtype=np.int64, copy=True),
            xyz_um=group[["z_um", "y_um", "x_um"]].to_numpy(
                dtype=np.float32, copy=True
            ),
            xyz_vox=group[["z", "y", "x"]].to_numpy(
                dtype=np.float32, copy=True
            ),
            volume=pd.to_numeric(
                group["volume"], errors="coerce"
            ).to_numpy(dtype=np.float32, copy=True),
            boundary_um=pd.to_numeric(
                group["distance_to_boundary_um"], errors="coerce"
            ).to_numpy(dtype=np.float32, copy=True),
        )
    return store


# =============================================================================
# Identity split and global motion
# =============================================================================


def make_identity_split(
    tracks: pd.DataFrame,
    store: dict[tuple[str, int], TrackSeries],
    *,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for sequence in ("01", "02"):
        ids = sorted(
            int(track_id)
            for (seq, track_id), series in store.items()
            if seq == sequence and len(series.frames) >= 4
        )
        if len(ids) < 10:
            raise RuntimeError(
                f"Sequence {sequence} has too few usable identities: {len(ids)}"
            )
        shuffled = np.asarray(ids, dtype=np.int64)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(validation_fraction * len(shuffled))))
        n_val = min(n_val, len(shuffled) - 1)
        validation = set(int(v) for v in shuffled[:n_val])

        parent_lookup = (
            tracks.loc[tracks["sequence"].astype(str).str.zfill(2) == sequence]
            .set_index("track_id")["parent_track_id"]
            .to_dict()
        )

        for track_id in ids:
            rows.append(
                {
                    "sequence": sequence,
                    "track_id": int(track_id),
                    "parent_track_id": int(parent_lookup.get(track_id, 0)),
                    "split": (
                        "validation"
                        if track_id in validation
                        else "train"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(["sequence", "track_id"]).reset_index(drop=True)


def estimate_global_motion(
    store: dict[tuple[str, int], TrackSeries],
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[tuple[int, int], np.ndarray]]]:
    train_ids = {
        (str(row.sequence).zfill(2), int(row.track_id))
        for row in split.itertuples(index=False)
        if str(row.split) == "train"
    }

    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[tuple[int, int], np.ndarray]] = {
        "01": {},
        "02": {},
    }

    for sequence in ("01", "02"):
        displacement_by_transition: dict[
            tuple[int, int], list[np.ndarray]
        ] = {}
        for key in train_ids:
            if key[0] != sequence or key not in store:
                continue
            series = store[key]
            for i in range(1, len(series.frames)):
                f0 = int(series.frames[i - 1])
                f1 = int(series.frames[i])
                if f1 != f0 + 1:
                    continue
                delta = (
                    series.xyz_um[i].astype(np.float64)
                    - series.xyz_um[i - 1].astype(np.float64)
                )
                displacement_by_transition.setdefault((f0, f1), []).append(delta)

        for (f0, f1), values in sorted(displacement_by_transition.items()):
            matrix = np.stack(values, axis=0)
            shift = np.median(matrix, axis=0)
            lookup[sequence][(f0, f1)] = shift.astype(np.float32)
            rows.append(
                {
                    "sequence": sequence,
                    "frame_from": f0,
                    "frame_to": f1,
                    "global_shift_z_um": float(shift[0]),
                    "global_shift_y_um": float(shift[1]),
                    "global_shift_x_um": float(shift[2]),
                    "support": int(len(matrix)),
                }
            )

    return pd.DataFrame(rows), lookup


# =============================================================================
# Placement pools / online synthetic scenes
# =============================================================================


@dataclass(frozen=True)
class PlacementKey:
    sequence: str
    cut_frame: int
    gap_frames: int


@dataclass(frozen=True)
class SceneSpec:
    example_id: int
    split: str
    sequence: str
    cut_frame: int
    target_frame: int
    gap_frames: int
    source_order: tuple[int, ...]
    target_order: tuple[int, ...]
    source_lengths: tuple[int, ...]
    target_lengths: tuple[int, ...]
    hard_requested: bool
    anchor_track_id: int
    hardness_margin_um: float
    anchor_displacement_um: float

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        for name in (
            "source_order",
            "target_order",
            "source_lengths",
            "target_lengths",
        ):
            row[name] = json.dumps(row[name])
        return row


class PlacementPool:
    def __init__(
        self,
        *,
        store: dict[tuple[str, int], TrackSeries],
        split_table: pd.DataFrame,
        split_name: str,
        min_history: int,
        min_future: int,
        max_gap: int,
        tracks_per_scene: int,
    ) -> None:
        self.store = store
        self.split_name = split_name
        allowed = {
            (str(row.sequence).zfill(2), int(row.track_id))
            for row in split_table.itertuples(index=False)
            if str(row.split) == split_name
        }

        grouped: dict[PlacementKey, list[int]] = {}
        for key in sorted(allowed):
            series = store.get(key)
            if series is None:
                continue
            frames = series.frames
            if len(frames) < min_history + min_future:
                continue

            # Split each GT trajectory into strictly contiguous runs. Apparent
            # internal disappearances are not used as clean synthetic truth.
            starts = [0]
            if len(frames) > 1:
                starts.extend((np.flatnonzero(np.diff(frames) != 1) + 1).tolist())
            starts = sorted(set(int(v) for v in starts))
            ends = starts[1:] + [len(frames)]

            for i0, i1 in zip(starts, ends):
                run = frames[i0:i1]
                if len(run) < min_history + min_future:
                    continue
                run_first = int(run[0])
                run_last = int(run[-1])
                for gap in range(1, max_gap + 1):
                    first_cut = run_first + min_history - 1
                    last_cut = run_last - gap - min_future + 1
                    if last_cut < first_cut:
                        continue
                    for cut in range(first_cut, last_cut + 1):
                        grouped.setdefault(
                            PlacementKey(key[0], int(cut), int(gap)),
                            [],
                        ).append(int(key[1]))

        # A component needs several simultaneously breakable identities.
        self.by_key = {
            key: tuple(sorted(set(track_ids)))
            for key, track_ids in grouped.items()
            if len(set(track_ids)) >= tracks_per_scene
        }
        self.keys_by_sequence: dict[str, tuple[PlacementKey, ...]] = {}
        for sequence in ("01", "02"):
            self.keys_by_sequence[sequence] = tuple(
                key for key in self.by_key if key.sequence == sequence
            )

        if not self.by_key:
            raise RuntimeError(
                f"No {split_name} CE placement keys support "
                f"{tracks_per_scene} tracks per scene."
            )

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split_name,
            "keys": int(len(self.by_key)),
            "sequence_01_keys": int(len(self.keys_by_sequence["01"])),
            "sequence_02_keys": int(len(self.keys_by_sequence["02"])),
            "mean_tracks_per_key": float(
                np.mean([len(v) for v in self.by_key.values()])
            ),
        }


class SceneSampler:
    def __init__(
        self,
        *,
        pool: PlacementPool,
        store: dict[tuple[str, int], TrackSeries],
        global_motion: dict[str, dict[tuple[int, int], np.ndarray]],
        tracks_per_scene: int,
        min_history: int,
        max_history: int,
        min_future: int,
        max_future: int,
        scene_radius_um: float,
        hard_fraction: float,
        hard_proposals: int,
        seed: int,
    ) -> None:
        self.pool = pool
        self.store = store
        self.global_motion = global_motion
        self.tracks_per_scene = int(tracks_per_scene)
        self.min_history = int(min_history)
        self.max_history = int(max_history)
        self.min_future = int(min_future)
        self.max_future = int(max_future)
        self.scene_radius_um = float(scene_radius_um)
        self.hard_fraction = float(hard_fraction)
        self.hard_proposals = int(hard_proposals)
        self.rng = np.random.default_rng(seed)
        self.next_example_id = 0

    def _candidate_scene(self, hard: bool) -> SceneSpec | None:
        sequences = [
            seq
            for seq in ("01", "02")
            if len(self.pool.keys_by_sequence[seq])
        ]
        sequence = str(self.rng.choice(sequences))
        keys = self.pool.keys_by_sequence[sequence]
        key = keys[int(self.rng.integers(0, len(keys)))]
        track_ids = self.pool.by_key[key]
        anchor = int(track_ids[int(self.rng.integers(0, len(track_ids)))])

        target_frame = key.cut_frame + key.gap_frames
        anchor_target = self.store[(sequence, anchor)].xyz_at(target_frame)

        peer_rows: list[tuple[float, int]] = []
        for track_id in track_ids:
            if int(track_id) == anchor:
                continue
            point = self.store[(sequence, int(track_id))].xyz_at(target_frame)
            distance = float(np.linalg.norm(point - anchor_target))
            if distance <= self.scene_radius_um:
                peer_rows.append((distance, int(track_id)))

        if len(peer_rows) < self.tracks_per_scene - 1:
            return None

        peer_rows.sort()
        if hard:
            chosen_peers = [
                track_id
                for _, track_id in peer_rows[: self.tracks_per_scene - 1]
            ]
        else:
            candidates = np.asarray(
                [track_id for _, track_id in peer_rows],
                dtype=np.int64,
            )
            chosen_peers = [
                int(v)
                for v in self.rng.choice(
                    candidates,
                    size=self.tracks_per_scene - 1,
                    replace=False,
                ).tolist()
            ]

        selected = [anchor, *chosen_peers]
        source_order = selected.copy()
        target_order = selected.copy()
        self.rng.shuffle(source_order)
        self.rng.shuffle(target_order)

        source_lengths: list[int] = []
        target_lengths: list[int] = []
        for track_id in source_order:
            series = self.store[(sequence, int(track_id))]
            maximum = self.min(
                self.max_history,
                contiguous_history_available(series, key.cut_frame),
            )
            if maximum < self.min_history:
                return None
            source_lengths.append(
                int(self.rng.integers(self.min_history, maximum + 1))
            )

        for track_id in target_order:
            series = self.store[(sequence, int(track_id))]
            maximum = self.min(
                self.max_future,
                contiguous_future_available(series, target_frame),
            )
            if maximum < self.min_future:
                return None
            target_lengths.append(
                int(self.rng.integers(self.min_future, maximum + 1))
            )

        anchor_series = self.store[(sequence, anchor)]
        source_xyz = anchor_series.xyz_at(key.cut_frame)
        target_xyz = anchor_series.xyz_at(target_frame)
        true_distance = float(np.linalg.norm(target_xyz - source_xyz))

        wrong_distances = []
        for wrong_track in selected:
            if wrong_track == anchor:
                continue
            wrong_xyz = self.store[(sequence, wrong_track)].xyz_at(target_frame)
            wrong_distances.append(float(np.linalg.norm(wrong_xyz - source_xyz)))
        nearest_wrong = min(wrong_distances) if wrong_distances else float("inf")
        margin = float(nearest_wrong - true_distance)

        return SceneSpec(
            example_id=self.next_example_id,
            split=self.pool.split_name,
            sequence=sequence,
            cut_frame=int(key.cut_frame),
            target_frame=int(target_frame),
            gap_frames=int(key.gap_frames),
            source_order=tuple(int(v) for v in source_order),
            target_order=tuple(int(v) for v in target_order),
            source_lengths=tuple(int(v) for v in source_lengths),
            target_lengths=tuple(int(v) for v in target_lengths),
            hard_requested=bool(hard),
            anchor_track_id=int(anchor),
            hardness_margin_um=float(margin),
            anchor_displacement_um=float(true_distance),
        )

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def sample(self) -> SceneSpec:
        hard = bool(self.rng.random() < self.hard_fraction)
        proposals: list[SceneSpec] = []

        tries = self.hard_proposals if hard else 1
        attempts = max(tries * 8, 16)
        for _ in range(attempts):
            candidate = self._candidate_scene(hard=hard)
            if candidate is not None:
                proposals.append(candidate)
                if len(proposals) >= tries:
                    break

        if not proposals:
            raise RuntimeError(
                f"Could not synthesize a {self.pool.split_name} scene after "
                f"{attempts} attempts. Increase --scene-radius-um or reduce "
                "--tracks-per-scene."
            )

        if hard:
            # Lower wrong-minus-true margin is more ambiguous; if margins tie,
            # prefer larger true displacement.
            chosen = min(
                proposals,
                key=lambda item: (
                    item.hardness_margin_um,
                    -item.anchor_displacement_um,
                ),
            )
        else:
            chosen = proposals[0]

        chosen = SceneSpec(
            **{
                **asdict(chosen),
                "example_id": int(self.next_example_id),
            }
        )
        self.next_example_id += 1
        return chosen


def contiguous_history_available(series: TrackSeries, frame: int) -> int:
    i = series.index_of(frame)
    if i is None:
        return 0
    count = 1
    cursor = i
    while cursor > 0 and int(series.frames[cursor - 1]) == int(series.frames[cursor]) - 1:
        count += 1
        cursor -= 1
    return count


def contiguous_future_available(series: TrackSeries, frame: int) -> int:
    i = series.index_of(frame)
    if i is None:
        return 0
    count = 1
    cursor = i
    while (
        cursor + 1 < len(series.frames)
        and int(series.frames[cursor + 1]) == int(series.frames[cursor]) + 1
    ):
        count += 1
        cursor += 1
    return count


# =============================================================================
# Motion / pair feature construction
# =============================================================================


def cumulative_global_shift(
    lookup: dict[tuple[int, int], np.ndarray],
    start_frame: int,
    target_frame: int,
) -> tuple[np.ndarray, bool]:
    shift = np.zeros(3, dtype=np.float32)
    for frame in range(start_frame + 1, target_frame + 1):
        value = lookup.get((frame - 1, frame))
        if value is None or not np.isfinite(value).all():
            return np.zeros(3, dtype=np.float32), False
        shift += np.asarray(value, dtype=np.float32)
    return shift, True


def source_relative_velocity(
    rows: pd.DataFrame,
    lookup: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, bool]:
    if len(rows) < 2:
        return np.zeros(3, dtype=np.float32), False
    rows = rows.sort_values("frame")
    a = rows.iloc[-2]
    b = rows.iloc[-1]
    f0 = int(a.frame)
    f1 = int(b.frame)
    if f1 <= f0:
        return np.zeros(3, dtype=np.float32), False
    observed = (
        b[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float32)
        - a[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float32)
    ) / float(f1 - f0)

    # Relative velocity is only considered valid when the matching global
    # transition is available.
    global_shift = cumulative_global_shift(lookup, f0, f1)
    if not global_shift[1]:
        return np.zeros(3, dtype=np.float32), False
    relative = observed - global_shift[0] / float(f1 - f0)
    return relative.astype(np.float32), bool(np.isfinite(relative).all())


def log_volume_error(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return float("nan")
    return float(abs(math.log(float(b) / float(a))))


@dataclass
class RawScene:
    spec: SceneSpec
    tracklet_rows: list[pd.DataFrame]
    tracklet_roles: list[str]
    raw_structured: list[np.ndarray]
    structured_validity: list[np.ndarray]
    diagnostics: list[dict[str, float]]
    reliability_raw: list[np.ndarray]
    pair_frame: pd.DataFrame
    edge_index: np.ndarray
    edge_target: np.ndarray
    expected_global: np.ndarray
    expected_global_relative: np.ndarray
    prediction_valid: np.ndarray


def build_raw_scene(
    spec: SceneSpec,
    *,
    store: dict[tuple[str, int], TrackSeries],
    global_motion: dict[str, dict[tuple[int, int], np.ndarray]],
    trainer,
    global_median_volume: float,
) -> RawScene:
    sequence = spec.sequence
    motion = global_motion[sequence]

    tracklet_rows: list[pd.DataFrame] = []
    roles: list[str] = []

    for track_id, length in zip(spec.source_order, spec.source_lengths):
        first = spec.cut_frame - int(length) + 1
        rows = store[(sequence, int(track_id))].rows(
            first,
            spec.cut_frame,
            role="source",
        )
        tracklet_rows.append(rows)
        roles.append("source")

    for track_id, length in zip(spec.target_order, spec.target_lengths):
        last = spec.target_frame + int(length) - 1
        rows = store[(sequence, int(track_id))].rows(
            spec.target_frame,
            last,
            role="target",
        )
        tracklet_rows.append(rows)
        roles.append("target")

    raw_structured: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    reliability: list[np.ndarray] = []

    for rows in tracklet_rows:
        raw, valid, diag = trainer.build_raw_structured(
            rows,
            global_motion=motion,
        )
        raw_structured.append(raw)
        validity.append(valid)
        diagnostics.append(diag)
        reliability.append(
            trainer.build_reliability_raw(
                rows,
                diagnostics=diag,
                crop_valid_fraction=np.ones(len(rows), dtype=np.float32),
                global_median_volume=float(global_median_volume),
            )
        )

    n_source = len(spec.source_order)
    n_target = len(spec.target_order)
    source_local = {
        int(track_id): i
        for i, track_id in enumerate(spec.source_order)
    }
    target_local = {
        int(track_id): n_source + i
        for i, track_id in enumerate(spec.target_order)
    }

    candidate_radius = min(
        25.0,
        12.0 + 4.0 * max(spec.gap_frames - 1, 0),
    )

    edge_rows: list[dict[str, Any]] = []
    edge_pairs: list[tuple[int, int]] = []
    edge_targets: list[float] = []
    predicted_global: list[np.ndarray] = []
    predicted_relative: list[np.ndarray] = []
    valid_predictions: list[tuple[bool, bool, bool, bool]] = []

    source_rows_by_id = {
        int(track_id): tracklet_rows[i]
        for i, track_id in enumerate(spec.source_order)
    }
    target_rows_by_id = {
        int(track_id): tracklet_rows[n_source + i]
        for i, track_id in enumerate(spec.target_order)
    }

    # First pass builds raw candidate evidence.
    for source_track_id in spec.source_order:
        s_rows = source_rows_by_id[int(source_track_id)]
        s_endpoint = s_rows.iloc[-1]
        s_xyz = s_endpoint[["z_um", "y_um", "x_um"]].to_numpy(
            dtype=np.float32
        )
        s_volume = float(s_endpoint.volume)

        global_delta, global_valid = cumulative_global_shift(
            motion,
            spec.cut_frame,
            spec.target_frame,
        )
        expected_g = s_xyz + global_delta if global_valid else s_xyz.copy()

        rv, rv_valid = source_relative_velocity(s_rows, motion)
        relative_valid = bool(global_valid and rv_valid)
        expected_gr = (
            expected_g + rv * float(spec.gap_frames)
            if relative_valid
            else expected_g.copy()
        )

        for target_track_id in spec.target_order:
            t_rows = target_rows_by_id[int(target_track_id)]
            t_endpoint = t_rows.iloc[0]
            t_xyz = t_endpoint[["z_um", "y_um", "x_um"]].to_numpy(
                dtype=np.float32
            )
            t_volume = float(t_endpoint.volume)

            direct = float(np.linalg.norm(t_xyz - s_xyz))
            is_positive = int(source_track_id) == int(target_track_id)

            # Candidate search is high recall.  True links are never dropped;
            # wrong edges outside the normal gap-specific radius are omitted.
            if not is_positive and direct > candidate_radius:
                continue

            forward_error = (
                float(np.linalg.norm(t_xyz - expected_gr))
                if relative_valid
                else float("nan")
            )
            quality_error = (
                forward_error if np.isfinite(forward_error) else direct
            )
            quality = float(
                math.exp(-quality_error / max(candidate_radius, 1e-6))
            )

            edge_pairs.append(
                (
                    source_local[int(source_track_id)],
                    target_local[int(target_track_id)],
                )
            )
            edge_targets.append(float(is_positive))
            predicted_global.append(expected_g.astype(np.float32))
            predicted_relative.append(expected_gr.astype(np.float32))
            valid_predictions.append(
                (
                    bool(global_valid),
                    bool(relative_valid),
                    False,
                    False,
                )
            )
            edge_rows.append(
                {
                    "gap_frames": int(spec.gap_frames),
                    "direct_endpoint_distance_um": direct,
                    "hard_search_radius_um": float(candidate_radius),
                    "forward_error_um": forward_error,
                    "forward_history_count": int(len(s_rows)),
                    "forward_used_global_motion": bool(global_valid),
                    "forward_used_relative_velocity": bool(relative_valid),
                    "forward_uncertainty_um": np.nan,
                    "volume_log_error": log_volume_error(
                        s_volume, t_volume
                    ),
                    "target_real_observation_count": int(len(t_rows)),
                    "candidate_quality_score": quality,
                    "effective_pair_volume": (
                        float(math.sqrt(max(s_volume, 0.0) * max(t_volume, 0.0)))
                        if np.isfinite(s_volume)
                        and np.isfinite(t_volume)
                        and s_volume >= 0
                        and t_volume >= 0
                        else np.nan
                    ),
                    "small_cell_history_exception": False,
                    "_source_track_id": int(source_track_id),
                    "_target_track_id": int(target_track_id),
                }
            )

    if not edge_rows:
        raise RuntimeError("Synthetic scene unexpectedly has no candidate edges")

    pair_frame = pd.DataFrame(edge_rows)
    edge_index = np.asarray(edge_pairs, dtype=np.int64)
    edge_target = np.asarray(edge_targets, dtype=np.float32)
    expected_global_array = np.stack(predicted_global).astype(np.float32)
    expected_relative_array = np.stack(predicted_relative).astype(np.float32)
    prediction_valid_array = np.asarray(valid_predictions, dtype=bool)

    # Every selected identity must retain its true edge.
    positive_sources = set(
        edge_index[edge_target > 0.5, 0].astype(int).tolist()
    )
    positive_targets = set(
        edge_index[edge_target > 0.5, 1].astype(int).tolist()
    )
    if len(positive_sources) != n_source or len(positive_targets) != n_target:
        raise RuntimeError("A true synthetic continuation edge was lost")

    # Competition counts/ranks use candidate quality (higher is better).
    pair_frame["source_candidate_count"] = 0
    pair_frame["target_predecessor_count"] = 0
    pair_frame["source_rank"] = 0
    pair_frame["target_rank"] = 0
    pair_frame["source_score_margin"] = np.nan
    pair_frame["target_score_margin"] = np.nan
    pair_frame["mutual_best"] = False

    for local_source in sorted(set(edge_index[:, 0].tolist())):
        indices = np.flatnonzero(edge_index[:, 0] == local_source)
        qualities = pair_frame.iloc[indices]["candidate_quality_score"].to_numpy(
            dtype=np.float64
        )
        order = np.argsort(-qualities, kind="mergesort")
        ranks = np.empty(len(indices), dtype=np.int64)
        ranks[order] = np.arange(1, len(indices) + 1)
        margin = (
            float(qualities[order[0]] - qualities[order[1]])
            if len(order) > 1
            else float("nan")
        )
        pair_frame.loc[
            pair_frame.index[indices], "source_candidate_count"
        ] = int(len(indices))
        pair_frame.loc[
            pair_frame.index[indices], "source_rank"
        ] = ranks
        pair_frame.loc[
            pair_frame.index[indices], "source_score_margin"
        ] = margin

    for local_target in sorted(set(edge_index[:, 1].tolist())):
        indices = np.flatnonzero(edge_index[:, 1] == local_target)
        qualities = pair_frame.iloc[indices]["candidate_quality_score"].to_numpy(
            dtype=np.float64
        )
        order = np.argsort(-qualities, kind="mergesort")
        ranks = np.empty(len(indices), dtype=np.int64)
        ranks[order] = np.arange(1, len(indices) + 1)
        margin = (
            float(qualities[order[0]] - qualities[order[1]])
            if len(order) > 1
            else float("nan")
        )
        pair_frame.loc[
            pair_frame.index[indices], "target_predecessor_count"
        ] = int(len(indices))
        pair_frame.loc[
            pair_frame.index[indices], "target_rank"
        ] = ranks
        pair_frame.loc[
            pair_frame.index[indices], "target_score_margin"
        ] = margin

    pair_frame["mutual_best"] = (
        (pd.to_numeric(pair_frame["source_rank"]) == 1)
        & (pd.to_numeric(pair_frame["target_rank"]) == 1)
    )

    # Hide provenance columns from the model feature tensor.
    pair_frame = pair_frame.drop(
        columns=["_source_track_id", "_target_track_id"]
    )

    return RawScene(
        spec=spec,
        tracklet_rows=tracklet_rows,
        tracklet_roles=roles,
        raw_structured=raw_structured,
        structured_validity=validity,
        diagnostics=diagnostics,
        reliability_raw=reliability,
        pair_frame=pair_frame,
        edge_index=edge_index,
        edge_target=edge_target,
        expected_global=expected_global_array,
        expected_global_relative=expected_relative_array,
        prediction_valid=prediction_valid_array,
    )


# =============================================================================
# Normalization / PreparedExample conversion
# =============================================================================


def fit_training_stats(
    *,
    sampler: SceneSampler,
    scene_count: int,
    store: dict[tuple[str, int], TrackSeries],
    global_motion: dict[str, dict[tuple[int, int], np.ndarray]],
    trainer,
    global_median_volume: float,
) -> tuple[Any, list[SceneSpec]]:
    raw_values: list[np.ndarray] = []
    raw_validity: list[np.ndarray] = []
    reliability_values: list[np.ndarray] = []
    pair_frames: list[pd.DataFrame] = []
    specs: list[SceneSpec] = []

    print(
        f"[stats] synthesizing {scene_count} TRAIN-only scenes for "
        "feature normalization ...",
        flush=True,
    )
    for i in range(scene_count):
        spec = sampler.sample()
        raw = build_raw_scene(
            spec,
            store=store,
            global_motion=global_motion,
            trainer=trainer,
            global_median_volume=global_median_volume,
        )
        specs.append(spec)
        raw_values.extend(raw.raw_structured)
        raw_validity.extend(raw.structured_validity)
        reliability_values.extend(raw.reliability_raw)
        pair_frames.append(raw.pair_frame)

        if (i + 1) % 100 == 0 or i + 1 == scene_count:
            print(
                f"[stats] {i + 1}/{scene_count} scenes",
                flush=True,
            )

    structured_stats = trainer.fit_masked_stats(
        np.concatenate(raw_values, axis=0),
        np.concatenate(raw_validity, axis=0),
    )
    reliability_stats = trainer.fit_reliability_stats(
        np.stack(reliability_values, axis=0)
    )
    pair_stats = trainer.fit_pair_stats(
        pd.concat(pair_frames, ignore_index=True)
    )

    stats = trainer.FeatureStats(
        structured=structured_stats,
        reliability=reliability_stats,
        pair=pair_stats,
    )
    return stats, specs


def prepare_scene(
    spec: SceneSpec,
    *,
    store: dict[tuple[str, int], TrackSeries],
    global_motion: dict[str, dict[tuple[int, int], np.ndarray]],
    trainer,
    stats,
    global_median_volume: float,
):
    raw = build_raw_scene(
        spec,
        store=store,
        global_motion=global_motion,
        trainer=trainer,
        global_median_volume=global_median_volume,
    )

    n = len(raw.tracklet_rows)
    k_max = max(len(rows) for rows in raw.tracklet_rows)
    fingerprint_dim = trainer.ReconcilerConfig().fingerprint.embedding_dim

    structured = np.zeros((n, k_max, 48), dtype=np.float32)
    observation_mask = np.zeros((n, k_max), dtype=bool)
    times = np.zeros((n, k_max), dtype=np.float32)
    start_xyz = np.zeros((n, 3), dtype=np.float32)
    end_xyz = np.zeros((n, 3), dtype=np.float32)
    reliability = np.zeros((n, 12), dtype=np.float32)
    zero_fingerprints = np.zeros(
        (n, k_max, fingerprint_dim),
        dtype=np.float32,
    )

    for i, rows in enumerate(raw.tracklet_rows):
        k = len(rows)
        structured[i, :k] = trainer.apply_masked_stats(
            raw.raw_structured[i],
            raw.structured_validity[i],
            stats.structured,
        )
        observation_mask[i, :k] = True
        times[i, :k] = rows["frame"].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        start_xyz[i] = xyz[0]
        end_xyz[i] = xyz[-1]
        reliability[i] = (
            raw.reliability_raw[i] - stats.reliability.mean
        ) / stats.reliability.std

    pair_tensor, _ = trainer.tensorize_stage11_pair_features(
        raw.pair_frame,
        mean=stats.pair.mean,
        std=stats.pair.std,
        device="cpu",
    )

    e = len(raw.edge_index)
    expected_local = np.zeros((e, 3), dtype=np.float32)
    expected_backward = np.zeros((e, 3), dtype=np.float32)

    return trainer.PreparedExample(
        example_id=int(spec.example_id),
        tracklet_indices=np.arange(n, dtype=np.int64),
        structured=structured,
        observation_mask=observation_mask,
        times=times,
        start_xyz_um=start_xyz,
        end_xyz_um=end_xyz,
        reliability_raw=reliability.astype(np.float32, copy=True),
        crops=None,
        fingerprints_zero=zero_fingerprints,
        edge_index=raw.edge_index.astype(np.int64, copy=True),
        gap_frames=np.full(
            e,
            int(spec.gap_frames),
            dtype=np.int64,
        ),
        pair_features=pair_tensor.numpy().astype(
            np.float32,
            copy=True,
        ),
        expected_global=raw.expected_global.astype(
            np.float32,
            copy=True,
        ),
        expected_global_relative=raw.expected_global_relative.astype(
            np.float32,
            copy=True,
        ),
        expected_local=expected_local,
        expected_backward=expected_backward,
        prediction_valid=raw.prediction_valid.astype(bool, copy=True),
        edge_target=raw.edge_target.astype(np.float32, copy=True),
    )


def synthesize_fixed_set(
    sampler: SceneSampler,
    *,
    count: int,
    store,
    global_motion,
    trainer,
    stats,
    global_median_volume: float,
) -> tuple[list[Any], list[SceneSpec]]:
    prepared: list[Any] = []
    specs: list[SceneSpec] = []
    for i in range(count):
        spec = sampler.sample()
        prepared.append(
            prepare_scene(
                spec,
                store=store,
                global_motion=global_motion,
                trainer=trainer,
                stats=stats,
                global_median_volume=global_median_volume,
            )
        )
        specs.append(spec)
        if (i + 1) % 100 == 0 or i + 1 == count:
            print(
                f"[synthesis:{sampler.pool.split_name}] "
                f"{i + 1}/{count}",
                flush=True,
            )
    return prepared, specs


# =============================================================================
# Baseline / diagnostics
# =============================================================================


def geometry_baseline(examples: list[Any]) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment

    exact = 0
    source_correct = 0
    source_total = 0
    target_correct = 0
    target_total = 0
    wrong_closer_sources = 0
    positive_count = 0
    gap_counts: dict[int, int] = {}

    for example in examples:
        edge_index = example.edge_index
        target = example.edge_target > 0.5
        # Motion-relation direct distance is reconstructed from endpoints.
        distances = np.linalg.norm(
            example.start_xyz_um[edge_index[:, 1]]
            - example.end_xyz_um[edge_index[:, 0]],
            axis=1,
        )

        for source in np.unique(edge_index[:, 0]):
            idx = np.flatnonzero(edge_index[:, 0] == source)
            positive = idx[target[idx]]
            if len(positive) != 1:
                continue
            source_total += 1
            best = idx[np.argmin(distances[idx])]
            source_correct += int(best == positive[0])
            wrong = idx[~target[idx]]
            if len(wrong) and float(np.min(distances[wrong])) < float(
                distances[positive[0]]
            ):
                wrong_closer_sources += 1

        for dst in np.unique(edge_index[:, 1]):
            idx = np.flatnonzero(edge_index[:, 1] == dst)
            positive = idx[target[idx]]
            if len(positive) != 1:
                continue
            target_total += 1
            best = idx[np.argmin(distances[idx])]
            target_correct += int(best == positive[0])

        sources = sorted(np.unique(edge_index[:, 0]).tolist())
        targets = sorted(np.unique(edge_index[:, 1]).tolist())
        if len(sources) == len(targets):
            smap = {v: i for i, v in enumerate(sources)}
            tmap = {v: i for i, v in enumerate(targets)}
            cost = np.full(
                (len(sources), len(targets)),
                1e6,
                dtype=np.float64,
            )
            positive_matrix = np.zeros_like(cost, dtype=bool)
            for d, edge, label in zip(distances, edge_index, target):
                i = smap[int(edge[0])]
                j = tmap[int(edge[1])]
                cost[i, j] = float(d)
                positive_matrix[i, j] = bool(label)
            rr, cc = linear_sum_assignment(cost)
            exact += int(
                len(rr) == len(sources)
                and all(
                    positive_matrix[r, c]
                    for r, c in zip(rr, cc)
                )
            )

        positive_count += int(target.sum())
        gap = int(example.gap_frames[0]) if len(example.gap_frames) else -1
        gap_counts[gap] = gap_counts.get(gap, 0) + 1

    return {
        "scene_exact_assignment": exact / max(len(examples), 1),
        "source_top1": source_correct / max(source_total, 1),
        "target_top1": target_correct / max(target_total, 1),
        "wrong_closer_source_fraction": (
            wrong_closer_sources / max(source_total, 1)
        ),
        "positive_edges": int(positive_count),
        "gap_distribution": {
            str(k): int(v) for k, v in sorted(gap_counts.items())
        },
    }


def sequence_distribution(specs: list[SceneSpec]) -> dict[str, int]:
    result: dict[str, int] = {}
    for spec in specs:
        result[spec.sequence] = result.get(spec.sequence, 0) + 1
    return result


def gap_distribution(specs: list[SceneSpec]) -> dict[str, int]:
    result: dict[str, int] = {}
    for spec in specs:
        key = str(int(spec.gap_frames))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


# =============================================================================
# Model scope / checkpointing
# =============================================================================


def disable_untrained_branches(model: torch.nn.Module) -> list[str]:
    """Disable branches unsupported by this CE TRA-only objective.

    The appearance temporal stream is explicitly zeroed. Therefore checkpoints
    remain safe when loaded later without a special forward hook.
    """

    frozen_prefixes = (
        "tracklets.fingerprint.",
        "tracklets.appearance_stream.",
        "division_prior_head.",
        "appearance_head.",
        "termination_head.",
        "division_head.",
    )

    frozen_names: list[str] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_prefixes):
                parameter.requires_grad_(False)
                frozen_names.append(name)

                # The appearance stream must be exactly zero so random,
                # untrained appearance features cannot affect CE training or
                # later continuation inference.
                if name.startswith("tracklets.appearance_stream."):
                    parameter.zero_()

    return frozen_names


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    step: int,
    stats,
    global_median_volume: float,
    args,
    train_metrics,
    validation_metrics,
    identity_split: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "investigation": SCRIPT_NAME,
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_stats": stats.to_json(),
            "train_global_median_volume": float(global_median_volume),
            "args": vars(args),
            "train_metrics": train_metrics.as_dict(),
            "validation_metrics": validation_metrics.as_dict(),
            "training_scope": {
                "continuation": True,
                "parental_softmax": True,
                "structured_temporal": True,
                "edge_reasoner": True,
                "appearance_fingerprint": False,
                "appearance_temporal": False,
                "division": False,
                "appearance_event": False,
                "termination_event": False,
                "appearance_stream_zeroed_in_state_dict": True,
            },
            "identity_split": {
                "train_count": int(
                    (identity_split["split"] == "train").sum()
                ),
                "validation_count": int(
                    (identity_split["split"] == "validation").sum()
                ),
                "identity_overlap": 0,
            },
        },
        path,
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the current learned track reconciler continuation core on "
            "online synthetic breaks generated from clean Fluo-N3DH-CE TRA GT."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "CTC training package containing 01_GT/TRA and 02_GT/TRA. "
            "Defaults to data/external/Fluo-N3DH-CE/Fluo-N3DH-CE_train."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-cache", default=None)
    parser.add_argument(
        "--spacing",
        default="1.0,0.09,0.09",
        help="CE physical spacing Z,Y,X in micrometres.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--tracks-per-scene",
        type=int,
        default=DEFAULT_TRACKS_PER_SCENE,
    )
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_MIN_HISTORY,
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=DEFAULT_MAX_HISTORY,
    )
    parser.add_argument(
        "--min-future",
        type=int,
        default=DEFAULT_MIN_FUTURE,
    )
    parser.add_argument(
        "--max-future",
        type=int,
        default=DEFAULT_MAX_FUTURE,
    )
    parser.add_argument(
        "--scene-radius-um",
        type=float,
        default=DEFAULT_SCENE_RADIUS_UM,
    )
    parser.add_argument(
        "--hard-fraction",
        type=float,
        default=DEFAULT_HARD_FRACTION,
    )
    parser.add_argument(
        "--hard-proposals",
        type=int,
        default=DEFAULT_HARD_PROPOSALS,
    )

    parser.add_argument(
        "--stats-scenes",
        type=int,
        default=DEFAULT_STATS_SCENES,
    )
    parser.add_argument(
        "--validation-scenes",
        type=int,
        default=DEFAULT_VALIDATION_SCENES,
    )
    parser.add_argument(
        "--train-eval-scenes",
        type=int,
        default=DEFAULT_TRAIN_EVAL_SCENES,
    )

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto / cpu / cuda / cuda:N",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "off", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Build/cache GT observations, split, normalization stats and fixed "
            "validation scenes, but do not train."
        ),
    )
    parser.add_argument(
        "--rebuild-base",
        action="store_true",
        help="Reparse all CE TRA TIFFs even if the compact base cache exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Investigation-08 output directory.",
    )
    return parser


def parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in str(text).split(","))
    if len(values) != 3 or not all(v > 0 for v in values):
        raise ValueError("--spacing must be positive Z,Y,X values")
    return values  # type: ignore[return-value]


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")
    if int(args.tracks_per_scene) < 2:
        raise ValueError("--tracks-per-scene must be >=2")
    if int(args.max_gap) < 1:
        raise ValueError("--max-gap must be >=1")
    if not 1 <= int(args.min_history) <= int(args.max_history):
        raise ValueError("Require 1 <= min-history <= max-history")
    if not 1 <= int(args.min_future) <= int(args.max_future):
        raise ValueError("Require 1 <= min-future <= max-future")
    if float(args.scene_radius_um) <= 0:
        raise ValueError("--scene-radius-um must be >0")
    if not 0.0 <= float(args.hard_fraction) <= 1.0:
        raise ValueError("--hard-fraction must be in [0,1]")
    for name in (
        "hard_proposals",
        "stats_scenes",
        "validation_scenes",
        "train_eval_scenes",
        "steps",
        "batch_size",
        "log_every",
        "eval_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >0")
    if float(args.lr) <= 0 or float(args.grad_clip) <= 0:
        raise ValueError("--lr and --grad-clip must be >0")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_root = (
        resolve(args.dataset_root)
        if args.dataset_root is not None
        else default_dataset_root()
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    # Validate both sequences before touching outputs.
    for sequence in ("01", "02"):
        sequence_paths(dataset_root, sequence)

    output = (
        resolve(args.output)
        if args.output is not None
        else default_output()
    )
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Pass --overwrite to rebuild Investigation 08."
        )
    output.mkdir(parents=True, exist_ok=True)

    base_cache = (
        resolve(args.base_cache)
        if args.base_cache is not None
        else default_base_cache()
    )
    spacing = parse_spacing(args.spacing)

    trainer, _inv04 = load_trainer_modules()

    print("=" * 126, flush=True)
    print(
        "TRACK RECONCILER — INVESTIGATION 08: "
        "CE ONLINE SYNTHETIC CONTINUATION TRAINING",
        flush=True,
    )
    print("=" * 126, flush=True)
    print(f"dataset root             : {dataset_root}", flush=True)
    print(f"base cache               : {base_cache}", flush=True)
    print(f"output                   : {output}", flush=True)
    print(f"spacing ZYX um           : {spacing}", flush=True)
    print("training data            : CE TRA GT only", flush=True)
    print("sequences                : 01 + 02", flush=True)
    print(
        "augmentation             : fresh arbitrary GT track breaks online",
        flush=True,
    )
    print(
        "validation               : identity-disjoint holdout from BOTH sequences",
        flush=True,
    )
    print(
        "appearance               : disabled/zeroed; no raw TIFFs read",
        flush=True,
    )
    print(
        "division                 : preserved in man_track metadata but NOT trained",
        flush=True,
    )
    print("=" * 126, flush=True)

    started = time.perf_counter()

    observations, tracks, tra_files = build_base_cache(
        dataset_root,
        base_cache,
        spacing=spacing,
        rebuild=bool(args.rebuild_base),
    )
    store = build_track_store(observations)

    # Diagnostics for the GT-tail issue noticed during visualization.
    gt_frame_summary = {
        sequence: {
            "first_tra_frame": int(min(files)),
            "last_tra_frame": int(max(files)),
            "tra_frame_count": int(len(files)),
        }
        for sequence, files in tra_files.items()
    }
    print(
        "[GT coverage] "
        + " | ".join(
            f"{seq}: {info['first_tra_frame']}..{info['last_tra_frame']} "
            f"({info['tra_frame_count']})"
            for seq, info in gt_frame_summary.items()
        ),
        flush=True,
    )
    print(
        "[GT coverage] frames after the last TRA TIFF are intentionally ignored.",
        flush=True,
    )

    identity_split = make_identity_split(
        tracks,
        store,
        validation_fraction=float(args.validation_fraction),
        seed=seed + 17,
    )
    atomic_csv(output / "identity_split.csv", identity_split)

    train_ids = set(
        (str(row.sequence).zfill(2), int(row.track_id))
        for row in identity_split.itertuples(index=False)
        if str(row.split) == "train"
    )
    val_ids = set(
        (str(row.sequence).zfill(2), int(row.track_id))
        for row in identity_split.itertuples(index=False)
        if str(row.split) == "validation"
    )
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise RuntimeError(
            f"Identity split leakage detected: {sorted(overlap)[:10]}"
        )

    print(
        f"[split] train identities={len(train_ids):,} | "
        f"validation identities={len(val_ids):,} | overlap=0",
        flush=True,
    )
    for sequence in ("01", "02"):
        train_count = sum(1 for seq, _ in train_ids if seq == sequence)
        val_count = sum(1 for seq, _ in val_ids if seq == sequence)
        print(
            f"[split] sequence {sequence}: "
            f"train={train_count:,} validation={val_count:,}",
            flush=True,
        )

    global_motion_table, global_motion = estimate_global_motion(
        store,
        identity_split,
    )
    atomic_csv(output / "global_motion_train_only.csv", global_motion_table)

    train_volumes = []
    for key in train_ids:
        series = store.get(key)
        if series is not None:
            values = series.volume[
                np.isfinite(series.volume) & (series.volume > 0)
            ]
            if len(values):
                train_volumes.append(values)
    global_median_volume = (
        float(np.median(np.concatenate(train_volumes)))
        if train_volumes
        else 1.0
    )

    train_pool = PlacementPool(
        store=store,
        split_table=identity_split,
        split_name="train",
        min_history=int(args.min_history),
        min_future=int(args.min_future),
        max_gap=int(args.max_gap),
        tracks_per_scene=int(args.tracks_per_scene),
    )
    validation_pool = PlacementPool(
        store=store,
        split_table=identity_split,
        split_name="validation",
        min_history=int(args.min_history),
        min_future=int(args.min_future),
        max_gap=int(args.max_gap),
        tracks_per_scene=int(args.tracks_per_scene),
    )

    print(f"[placements] TRAIN {train_pool.summary()}", flush=True)
    print(f"[placements] VAL   {validation_pool.summary()}", flush=True)

    # Separate deterministic RNG streams prevent training synthesis from ever
    # changing the fixed validation set.
    stats_sampler = SceneSampler(
        pool=train_pool,
        store=store,
        global_motion=global_motion,
        tracks_per_scene=int(args.tracks_per_scene),
        min_history=int(args.min_history),
        max_history=int(args.max_history),
        min_future=int(args.min_future),
        max_future=int(args.max_future),
        scene_radius_um=float(args.scene_radius_um),
        hard_fraction=float(args.hard_fraction),
        hard_proposals=int(args.hard_proposals),
        seed=seed + 101,
    )
    train_sampler = SceneSampler(
        pool=train_pool,
        store=store,
        global_motion=global_motion,
        tracks_per_scene=int(args.tracks_per_scene),
        min_history=int(args.min_history),
        max_history=int(args.max_history),
        min_future=int(args.min_future),
        max_future=int(args.max_future),
        scene_radius_um=float(args.scene_radius_um),
        hard_fraction=float(args.hard_fraction),
        hard_proposals=int(args.hard_proposals),
        seed=seed + 202,
    )
    train_eval_sampler = SceneSampler(
        pool=train_pool,
        store=store,
        global_motion=global_motion,
        tracks_per_scene=int(args.tracks_per_scene),
        min_history=int(args.min_history),
        max_history=int(args.max_history),
        min_future=int(args.min_future),
        max_future=int(args.max_future),
        scene_radius_um=float(args.scene_radius_um),
        hard_fraction=float(args.hard_fraction),
        hard_proposals=int(args.hard_proposals),
        seed=seed + 303,
    )
    validation_sampler = SceneSampler(
        pool=validation_pool,
        store=store,
        global_motion=global_motion,
        tracks_per_scene=int(args.tracks_per_scene),
        min_history=int(args.min_history),
        max_history=int(args.max_history),
        min_future=int(args.min_future),
        max_future=int(args.max_future),
        scene_radius_um=float(args.scene_radius_um),
        hard_fraction=float(args.hard_fraction),
        hard_proposals=int(args.hard_proposals),
        seed=seed + 404,
    )

    stats, stats_specs = fit_training_stats(
        sampler=stats_sampler,
        scene_count=int(args.stats_scenes),
        store=store,
        global_motion=global_motion,
        trainer=trainer,
        global_median_volume=global_median_volume,
    )
    atomic_json(
        output / "feature_stats.json",
        {
            **stats.to_json(),
            "fit_split": "train_identities_only",
            "train_global_median_volume": global_median_volume,
            "appearance_features": "disabled",
        },
    )
    atomic_csv(
        output / "stats_scene_manifest.csv",
        pd.DataFrame([spec.to_row() for spec in stats_specs]),
    )

    print("[validation] synthesizing fixed identity-held-out scenes ...", flush=True)
    validation_prepared, validation_specs = synthesize_fixed_set(
        validation_sampler,
        count=int(args.validation_scenes),
        store=store,
        global_motion=global_motion,
        trainer=trainer,
        stats=stats,
        global_median_volume=global_median_volume,
    )
    atomic_csv(
        output / "validation_scene_manifest.csv",
        pd.DataFrame([spec.to_row() for spec in validation_specs]),
    )

    print("[train-eval] synthesizing fixed diagnostic TRAIN scenes ...", flush=True)
    train_eval_prepared, train_eval_specs = synthesize_fixed_set(
        train_eval_sampler,
        count=int(args.train_eval_scenes),
        store=store,
        global_motion=global_motion,
        trainer=trainer,
        stats=stats,
        global_median_volume=global_median_volume,
    )
    atomic_csv(
        output / "train_eval_scene_manifest.csv",
        pd.DataFrame([spec.to_row() for spec in train_eval_specs]),
    )

    baseline = geometry_baseline(validation_prepared)
    print("", flush=True)
    print("=" * 126, flush=True)
    print("SYNTHETIC CE DATASET READY", flush=True)
    print("=" * 126, flush=True)
    print(
        f"base observations         : {len(observations):,}",
        flush=True,
    )
    print(
        f"GT tracks                 : {len(tracks):,}",
        flush=True,
    )
    print(
        f"normalization scenes      : {len(stats_specs):,}",
        flush=True,
    )
    print(
        f"fixed validation scenes   : {len(validation_specs):,}",
        flush=True,
    )
    print(
        f"validation sequence mix   : "
        f"{sequence_distribution(validation_specs)}",
        flush=True,
    )
    print(
        f"validation gap mix        : "
        f"{gap_distribution(validation_specs)}",
        flush=True,
    )
    print(
        "distance-only VAL        : "
        f"exact={100*baseline['scene_exact_assignment']:.2f}% "
        f"src={100*baseline['source_top1']:.2f}% "
        f"tgt={100*baseline['target_top1']:.2f}% "
        f"wrong-closer-src={100*baseline['wrong_closer_source_fraction']:.2f}%",
        flush=True,
    )
    print("=" * 126, flush=True)

    preparation_summary = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "dataset_root": root_relative(dataset_root),
        "spacing_zyx_um": list(spacing),
        "gt_frame_coverage": gt_frame_summary,
        "base_observations": int(len(observations)),
        "declared_tracks": int(len(tracks)),
        "identity_split": {
            "train": int(len(train_ids)),
            "validation": int(len(val_ids)),
            "overlap": 0,
        },
        "placement_pool": {
            "train": train_pool.summary(),
            "validation": validation_pool.summary(),
        },
        "synthesis": {
            "tracks_per_scene": int(args.tracks_per_scene),
            "max_gap": int(args.max_gap),
            "history_range": [
                int(args.min_history),
                int(args.max_history),
            ],
            "future_range": [
                int(args.min_future),
                int(args.max_future),
            ],
            "scene_radius_um": float(args.scene_radius_um),
            "hard_fraction": float(args.hard_fraction),
            "hard_proposals": int(args.hard_proposals),
            "online_training_scenes_stored": False,
        },
        "normalization": {
            "fit_on_train_identities_only": True,
            "stats_scenes": int(args.stats_scenes),
            "global_median_volume": float(global_median_volume),
        },
        "validation_geometry_baseline": baseline,
        "training_scope": {
            "continuation_only": True,
            "appearance_disabled": True,
            "division_disabled": True,
        },
    }
    atomic_json(output / "preparation_summary.json", preparation_summary)

    if args.prepare_only:
        print(
            f"[done] prepare-only completed in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return 0

    # -------------------------------------------------------------------------
    # Model / optimizer
    # -------------------------------------------------------------------------
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    amp_mode = trainer.resolved_amp_mode(device, str(args.amp))

    config = trainer.ReconcilerConfig()
    model = trainer.TrackletReconciliationNetwork(config).to(device)
    frozen = disable_untrained_branches(model)

    trainable_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = None
    if device.type == "cuda" and amp_mode == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    print("", flush=True)
    print("=" * 126, flush=True)
    print("CE CONTINUATION TRAINING", flush=True)
    print("=" * 126, flush=True)
    print(
        f"model parameters          : "
        f"{sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    print(
        f"trainable parameters      : "
        f"{trainable_parameter_count(model):,}",
        flush=True,
    )
    print(
        f"frozen parameter tensors  : {len(frozen)}",
        flush=True,
    )
    print(f"device                    : {device}", flush=True)
    print(f"amp                       : {amp_mode}", flush=True)
    print(f"steps                     : {args.steps:,}", flush=True)
    print(f"batch size                : {args.batch_size}", flush=True)
    print(f"learning rate             : {args.lr:g}", flush=True)
    print(
        "training scenes          : ONLINE; a fresh random CE break graph "
        "is synthesized for each batch item",
        flush=True,
    )
    print("=" * 126, flush=True)

    initial_train = trainer.evaluate(
        model,
        train_eval_prepared,
        device=device,
        appearance_mode="fingerprints",
        amp_mode=amp_mode,
    )
    initial_val = trainer.evaluate(
        model,
        validation_prepared,
        device=device,
        appearance_mode="fingerprints",
        amp_mode=amp_mode,
    )
    print(
        f"[eval step=0000] "
        f"TRAIN exact={100*initial_train.exact_assignment:6.2f}% "
        f"src={100*initial_train.source_top1:6.2f}% "
        f"tgt={100*initial_train.target_top1:6.2f}% "
        f"loss={initial_train.loss:.5f} || "
        f"VAL exact={100*initial_val.exact_assignment:6.2f}% "
        f"src={100*initial_val.source_top1:6.2f}% "
        f"tgt={100*initial_val.target_top1:6.2f}% "
        f"loss={initial_val.loss:.5f}",
        flush=True,
    )

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_exact = -1.0
    best_loss = float("inf")
    running_loss = 0.0
    running_count = 0
    train_started = time.perf_counter()

    for step in range(1, int(args.steps) + 1):
        model.train()

        prepared_batch = []
        for _ in range(int(args.batch_size)):
            spec = train_sampler.sample()
            prepared_batch.append(
                prepare_scene(
                    spec,
                    store=store,
                    global_motion=global_motion,
                    trainer=trainer,
                    stats=stats,
                    global_median_volume=global_median_volume,
                )
            )

        batch = trainer.collate(
            prepared_batch,
            device=device,
            appearance_mode="fingerprints",
        )

        optimizer.zero_grad(set_to_none=True)
        with trainer.autocast_context(device, amp_mode):
            prediction = model(batch.reconciliation)
            loss = trainer.focal_binary_probability_loss(
                prediction.parental_probabilities,
                batch.edge_target,
                batch.reconciliation.edges.edge_mask,
                gamma=2.0,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                float(args.grad_clip),
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                float(args.grad_clip),
            )
            optimizer.step()

        running_loss += float(loss.detach().float().cpu())
        running_count += 1

        if step % int(args.log_every) == 0:
            elapsed = time.perf_counter() - train_started
            memory = ""
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                memory = f" peakVRAM={peak:.2f}GiB"
            print(
                f"[train step={step:05d}] "
                f"loss={running_loss/max(running_count,1):.5f} "
                f"fresh_scenes={step*int(args.batch_size):,} "
                f"time={elapsed:.1f}s{memory}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            train_metrics = trainer.evaluate(
                model,
                train_eval_prepared,
                device=device,
                appearance_mode="fingerprints",
                amp_mode=amp_mode,
            )
            validation_metrics = trainer.evaluate(
                model,
                validation_prepared,
                device=device,
                appearance_mode="fingerprints",
                amp_mode=amp_mode,
            )

            print(
                f"[eval step={step:05d}] "
                f"TRAIN exact={100*train_metrics.exact_assignment:6.2f}% "
                f"src={100*train_metrics.source_top1:6.2f}% "
                f"tgt={100*train_metrics.target_top1:6.2f}% "
                f"loss={train_metrics.loss:.5f} || "
                f"VAL exact={100*validation_metrics.exact_assignment:6.2f}% "
                f"src={100*validation_metrics.source_top1:6.2f}% "
                f"tgt={100*validation_metrics.target_top1:6.2f}% "
                f"loss={validation_metrics.loss:.5f} "
                f"p+={validation_metrics.positive_probability_mean:.4f} "
                f"p-max-={validation_metrics.negative_probability_max:.4f}",
                flush=True,
            )

            record = {
                "step": int(step),
                "fresh_training_scenes_seen": int(
                    step * int(args.batch_size)
                ),
                "elapsed_seconds": float(
                    time.perf_counter() - train_started
                ),
                "train": train_metrics.as_dict(),
                "validation": validation_metrics.as_dict(),
            }
            history.append(record)
            atomic_json(output / "history.json", history)

            better = (
                validation_metrics.exact_assignment > best_exact
                or (
                    validation_metrics.exact_assignment == best_exact
                    and validation_metrics.loss < best_loss
                )
            )
            if better:
                best_exact = float(validation_metrics.exact_assignment)
                best_loss = float(validation_metrics.loss)
                save_checkpoint(
                    checkpoint_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    stats=stats,
                    global_median_volume=global_median_volume,
                    args=args,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    identity_split=identity_split,
                )

            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                stats=stats,
                global_median_volume=global_median_volume,
                args=args,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                identity_split=identity_split,
            )

    final_train = trainer.evaluate(
        model,
        train_eval_prepared,
        device=device,
        appearance_mode="fingerprints",
        amp_mode=amp_mode,
    )
    final_val = trainer.evaluate(
        model,
        validation_prepared,
        device=device,
        appearance_mode="fingerprints",
        amp_mode=amp_mode,
    )

    elapsed_total = time.perf_counter() - started
    summary = {
        **preparation_summary,
        "training": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "fresh_scenes_seen": int(
                int(args.steps) * int(args.batch_size)
            ),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": str(device),
            "amp": str(amp_mode),
            "total_parameters": int(
                sum(p.numel() for p in model.parameters())
            ),
            "trainable_parameters": int(
                trainable_parameter_count(model)
            ),
        },
        "initial": {
            "train": initial_train.as_dict(),
            "validation": initial_val.as_dict(),
        },
        "final": {
            "train": final_train.as_dict(),
            "validation": final_val.as_dict(),
        },
        "best_validation_checkpoint": "checkpoints/best_validation.pt",
        "latest_checkpoint": "checkpoints/latest.pt",
        "elapsed_seconds": float(elapsed_total),
    }
    atomic_json(output / "summary.json", summary)

    print("", flush=True)
    print("=" * 126, flush=True)
    print("INVESTIGATION 08 COMPLETE", flush=True)
    print("=" * 126, flush=True)
    print(
        f"fresh training scenes     : "
        f"{int(args.steps)*int(args.batch_size):,}",
        flush=True,
    )
    print(
        f"FINAL TRAIN               : "
        f"exact={100*final_train.exact_assignment:.2f}% "
        f"src={100*final_train.source_top1:.2f}% "
        f"tgt={100*final_train.target_top1:.2f}% "
        f"loss={final_train.loss:.5f}",
        flush=True,
    )
    print(
        f"FINAL VALIDATION          : "
        f"exact={100*final_val.exact_assignment:.2f}% "
        f"src={100*final_val.source_top1:.2f}% "
        f"tgt={100*final_val.target_top1:.2f}% "
        f"loss={final_val.loss:.5f} "
        f"p+={final_val.positive_probability_mean:.4f} "
        f"p-max-={final_val.negative_probability_max:.4f}",
        flush=True,
    )
    print(
        f"distance-only VAL         : "
        f"exact={100*baseline['scene_exact_assignment']:.2f}% "
        f"wrong-closer={100*baseline['wrong_closer_source_fraction']:.2f}%",
        flush=True,
    )
    print(
        f"best checkpoint           : "
        f"{checkpoint_dir / 'best_validation.pt'}",
        flush=True,
    )
    print(f"output                    : {output}", flush=True)
    print(f"elapsed                   : {elapsed_total:.1f}s", flush=True)
    print("=" * 126, flush=True)
    print(
        "[scope] This checkpoint is CE-clean continuation-core pretraining. "
        "Appearance and division branches were intentionally not trained.",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
