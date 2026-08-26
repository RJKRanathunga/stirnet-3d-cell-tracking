from __future__ import annotations

r"""
Investigation 04 — hard continuation mining + overfit.

This experiment extends the easy Investigation-01/03 continuation overfit with:

* variable temporal gaps (1..4 frames),
* asymmetric fragment lengths,
* short fragments with weak/no velocity evidence,
* true shared-middle chains:

      A: ... -3 -2 -1 | B: 0 1 | C: 2 3 4 ...

  where the SAME B tracklet is both the target of A->B and the source of B->C,
* simultaneous nearby tracks, so every true continuation competes with nearby
  cross-identity alternatives,
* explicit hard mining that prefers cases where a wrong cell is geometrically
  closer than the true continuation or the true edge has geometric rank > 1.

The underlying tracks are taken only from clean, branch-free, persistent
Trackastra tracklets produced by Investigation 36. STIR-Net/Trackastra are not
rerun.

The script builds the hard dataset and, by default, immediately launches the
existing 03_continuation_overfit trainer in-process. The model itself already
supports arbitrary tracklet graphs; Investigation 03's CSV adapter is patched
at runtime so a middle tracklet may expose both an incoming and outgoing
endpoint.

Typical run from repository root:

    python .\investigations\track_reconciler\04_hard_continuation_overfit.py --overwrite

Build only:

    python .\investigations\track_reconciler\04_hard_continuation_overfit.py --build-only --overwrite

Train an already-built hard dataset:

    python .\investigations\track_reconciler\04_hard_continuation_overfit.py --train-only --overwrite-training
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_NAME = "04_hard_continuation_overfit"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_EXAMPLES = 24
DEFAULT_TRACKS_PER_EXAMPLE = 4
DEFAULT_GUARD_FRAMES = 1
DEFAULT_GROUP_RADIUS_UM = 28.0
DEFAULT_MAX_CELL_MATCH_UM = 2.5
DEFAULT_MAX_STEP_UM = 12.0
DEFAULT_MAX_VOLUME_RATIO = 2.0
DEFAULT_MIN_BOUNDARY_DISTANCE_UM = 5.0
DEFAULT_MAX_TRACK_REUSE = 4
DEFAULT_MAX_PLACEMENT_REUSE = 4
DEFAULT_CHAIN_FRACTION = 0.50
DEFAULT_SEED = 20260827
DEFAULT_TRAIN_STEPS = 1200
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_TRAIN_LR = 3.0e-4
DEFAULT_CROP_SHAPE = "13,41,41"


# =============================================================================
# Repository helpers / dynamic reuse of Investigations 01 and 03
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


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_script(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_inv01():
    return load_script(
        ROOT / "investigations" / "track_reconciler" / "01_build_overfit_dataset.py",
        "_inv04_source_01",
    )


def load_inv03():
    return load_script(
        ROOT / "investigations" / "track_reconciler" / "03_continuation_overfit.py",
        "_inv04_trainer_03",
    )


def default_source(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "36_biohub_spatial_trackastra_visualization"
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


def root_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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


# =============================================================================
# Synthetic fragmentation patterns
# =============================================================================


@dataclass(frozen=True)
class Pattern:
    name: str
    lengths: tuple[int, ...]
    gaps: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.lengths) not in (2, 3):
            raise ValueError("Only pair and 3-fragment chain patterns are supported")
        if len(self.gaps) != len(self.lengths) - 1:
            raise ValueError("gaps must have len(lengths)-1 elements")
        if min(self.lengths) < 1 or min(self.gaps) < 1:
            raise ValueError("fragment lengths and gaps must be >=1")

    @property
    def kind(self) -> str:
        return "chain" if len(self.lengths) == 3 else "pair"

    @property
    def transition_count(self) -> int:
        return len(self.gaps)

    def intervals(self, first_start: int) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        start = int(first_start)
        for fragment, length in enumerate(self.lengths):
            end = start + int(length) - 1
            result.append((start, end))
            if fragment < len(self.gaps):
                # gap_frames is target_start - source_end.
                start = end + int(self.gaps[fragment])
        return tuple(result)

    def required_interval(self, first_start: int, guard: int) -> tuple[int, int]:
        intervals = self.intervals(first_start)
        return intervals[0][0] - guard, intervals[-1][1] + guard


PATTERNS: tuple[Pattern, ...] = (
    Pattern("pair_balanced_g1", (4, 4), (1,)),
    Pattern("pair_short_target_g1", (4, 1), (1,)),
    Pattern("pair_short_source_g1", (2, 4), (1,)),
    Pattern("pair_g2", (4, 4), (2,)),
    Pattern("pair_g3", (3, 4), (3,)),
    Pattern("pair_g4", (3, 4), (4,)),
    # Requested topology: ...,-2,-1 | 0,1 | 2,3,4...
    Pattern("chain_user_case", (4, 2, 3), (1, 1)),
    # One-frame middle fragment: no internal velocity can be estimated.
    Pattern("chain_one_frame_middle", (4, 1, 4), (1, 1)),
    Pattern("chain_asymmetric", (2, 2, 4), (1, 1)),
    Pattern("chain_first_gap2", (3, 2, 3), (2, 1)),
    Pattern("chain_second_gap2", (3, 2, 3), (1, 2)),
    Pattern("chain_both_gap2", (3, 1, 3), (2, 2)),
)


def candidate_radius_um(gap_frames: int) -> float:
    """Same gap schedule as learned.track_reconciler CandidateConfig."""
    return float(min(25.0, 12.0 + 4.0 * max(int(gap_frames) - 1, 0)))


# =============================================================================
# Clean persistent original tracks
# =============================================================================


@dataclass(frozen=True)
class Eligible:
    track_id: int
    first_start: int
    intervals: tuple[tuple[int, int], ...]
    starts_xyz_um: tuple[tuple[float, float, float], ...]
    ends_xyz_um: tuple[tuple[float, float, float], ...]
    transition_midpoints_um: tuple[tuple[float, float, float], ...]
    max_step_um: float
    max_volume_ratio: float
    max_match_error_um: float
    min_boundary_distance_um: float


def consecutive_volume_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return 1.0
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        return float("inf")
    ratios = np.maximum(values[1:] / values[:-1], values[:-1] / values[1:])
    return float(ratios.max())


def eligible_track(
    track: pd.DataFrame,
    *,
    pattern: Pattern,
    first_start: int,
    guard: int,
    max_cell_match_um: float,
    max_step_um: float,
    max_volume_ratio: float,
    min_boundary_distance_um: float,
) -> Eligible | None:
    intervals = pattern.intervals(first_start)
    required_start, required_end = pattern.required_interval(first_start, guard)
    required_frames = np.arange(required_start, required_end + 1, dtype=np.int64)

    indexed = track.set_index("frame", drop=False)
    if not np.isin(required_frames, indexed.index.to_numpy(dtype=np.int64)).all():
        return None

    window = indexed.loc[required_frames].copy()
    if len(window) != len(required_frames):
        return None
    if bool(window["lineage_related"].any()):
        return None

    numeric_names = (
        "z_um", "y_um", "x_um", "volume",
        "cell_match_error_um", "distance_to_boundary_um",
    )
    numeric = window[list(numeric_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        return None
    if (window["cell_id"].to_numpy(dtype=np.int64) < 0).any():
        return None

    match_error = float(window["cell_match_error_um"].max())
    boundary = float(window["distance_to_boundary_um"].min())
    if match_error > max_cell_match_um or boundary < min_boundary_distance_um:
        return None

    xyz = window[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    maximum_step = float(steps.max()) if steps.size else 0.0
    if maximum_step > max_step_um:
        return None

    ratio = consecutive_volume_ratio(window["volume"].to_numpy(dtype=np.float64))
    if ratio > max_volume_ratio:
        return None

    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    for start, end in intervals:
        s = indexed.loc[start, ["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
        e = indexed.loc[end, ["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
        starts.append(tuple(float(v) for v in s))
        ends.append(tuple(float(v) for v in e))

    midpoints: list[tuple[float, float, float]] = []
    for transition in range(pattern.transition_count):
        a = np.asarray(ends[transition], dtype=np.float64)
        b = np.asarray(starts[transition + 1], dtype=np.float64)
        midpoints.append(tuple(float(v) for v in 0.5 * (a + b)))

    return Eligible(
        track_id=int(track["track_id"].iloc[0]),
        first_start=int(first_start),
        intervals=intervals,
        starts_xyz_um=tuple(starts),
        ends_xyz_um=tuple(ends),
        transition_midpoints_um=tuple(midpoints),
        max_step_um=maximum_step,
        max_volume_ratio=ratio,
        max_match_error_um=match_error,
        min_boundary_distance_um=boundary,
    )


def enumerate_eligible(
    observations: pd.DataFrame,
    *,
    movie_frames: int,
    guard: int,
    args: argparse.Namespace,
) -> dict[tuple[str, int], list[Eligible]]:
    by_track = {
        int(track_id): group.sort_values("frame")
        for track_id, group in observations.groupby("track_id", sort=False)
    }
    result: dict[tuple[str, int], list[Eligible]] = {}

    for pattern in PATTERNS:
        for first_start in range(guard, movie_frames):
            required_start, required_end = pattern.required_interval(first_start, guard)
            if required_start < 0 or required_end >= movie_frames:
                continue

            items: list[Eligible] = []
            for track in by_track.values():
                item = eligible_track(
                    track,
                    pattern=pattern,
                    first_start=first_start,
                    guard=guard,
                    max_cell_match_um=float(args.max_cell_match_um),
                    max_step_um=float(args.max_step_um),
                    max_volume_ratio=float(args.max_volume_ratio),
                    min_boundary_distance_um=float(args.min_boundary_distance_um),
                )
                if item is not None:
                    items.append(item)
            if items:
                result[(pattern.name, int(first_start))] = items
    return result


# =============================================================================
# Hard local group mining
# =============================================================================


def pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


@dataclass(frozen=True)
class HardGroup:
    pattern_name: str
    pattern_kind: str
    first_start: int
    track_ids: tuple[int, ...]
    wrong_closer_count: int
    source_rank_gt1_count: int
    target_rank_gt1_count: int
    min_margin_um: float
    median_margin_um: float
    mean_margin_um: float
    candidate_edges: int
    possible_edges: int
    max_gap: int
    scene_diameter_um: float

    @property
    def rank_gt1_count(self) -> int:
        return self.source_rank_gt1_count + self.target_rank_gt1_count

    @property
    def density(self) -> float:
        return self.candidate_edges / max(self.possible_edges, 1)


def compute_group(
    items: list[Eligible],
    *,
    pattern: Pattern,
    min_wrong_source: int,
    min_wrong_target: int,
) -> HardGroup | None:
    k = len(items)
    margins: list[float] = []
    wrong_closer = 0
    source_rank_gt1 = 0
    target_rank_gt1 = 0
    candidate_edges = 0

    for transition, gap in enumerate(pattern.gaps):
        source = np.asarray([x.ends_xyz_um[transition] for x in items], dtype=np.float64)
        target = np.asarray([x.starts_xyz_um[transition + 1] for x in items], dtype=np.float64)
        distance = pairwise(source, target)
        true = np.diag(distance)
        radius = candidate_radius_um(gap)
        if np.any(true > radius):
            return None

        candidate = distance <= radius
        np.fill_diagonal(candidate, True)
        wrong = candidate.copy()
        np.fill_diagonal(wrong, False)
        if np.any(wrong.sum(axis=1) < min_wrong_source):
            return None
        if np.any(wrong.sum(axis=0) < min_wrong_target):
            return None
        candidate_edges += int(candidate.sum())

        for i in range(k):
            wrong_values = distance[i][wrong[i]]
            margin = float(wrong_values.min() - true[i])
            margins.append(margin)
            wrong_closer += int(margin < 0.0)

            available = np.flatnonzero(candidate[i])
            order = available[np.argsort(distance[i, available])]
            source_rank_gt1 += int(int(np.flatnonzero(order == i)[0]) + 1 > 1)

        for j in range(k):
            available = np.flatnonzero(candidate[:, j])
            order = available[np.argsort(distance[available, j])]
            target_rank_gt1 += int(int(np.flatnonzero(order == j)[0]) + 1 > 1)

    representative = np.asarray(
        [np.mean(np.asarray(x.transition_midpoints_um), axis=0) for x in items],
        dtype=np.float64,
    )
    diameter = float(pairwise(representative, representative).max())
    m = np.asarray(margins, dtype=np.float64)
    return HardGroup(
        pattern_name=pattern.name,
        pattern_kind=pattern.kind,
        first_start=int(items[0].first_start),
        track_ids=tuple(sorted(x.track_id for x in items)),
        wrong_closer_count=int(wrong_closer),
        source_rank_gt1_count=int(source_rank_gt1),
        target_rank_gt1_count=int(target_rank_gt1),
        min_margin_um=float(m.min()),
        median_margin_um=float(np.median(m)),
        mean_margin_um=float(m.mean()),
        candidate_edges=int(candidate_edges),
        possible_edges=int(pattern.transition_count * k * k),
        max_gap=int(max(pattern.gaps)),
        scene_diameter_um=diameter,
    )


def enumerate_groups(
    eligible: dict[tuple[str, int], list[Eligible]],
    *,
    patterns: dict[str, Pattern],
    args: argparse.Namespace,
) -> list[HardGroup]:
    result: dict[tuple[str, int, tuple[int, ...]], HardGroup] = {}
    k = int(args.tracks_per_example)

    for (pattern_name, first_start), items in eligible.items():
        if len(items) < k:
            continue
        pattern = patterns[pattern_name]
        by_id = {x.track_id: x for x in items}
        ids = np.asarray(sorted(by_id), dtype=np.int64)
        representative = np.asarray(
            [np.mean(np.asarray(by_id[int(t)].transition_midpoints_um), axis=0) for t in ids],
            dtype=np.float64,
        )
        d = pairwise(representative, representative)

        for anchor in range(len(ids)):
            order = np.argsort(d[anchor])
            nearby = [
                int(index)
                for index in order
                if d[anchor, index] <= float(args.group_radius_um)
            ]
            if len(nearby) < k:
                continue

            # Explore a few nearest-neighbour variants rather than only one set.
            pool = nearby[: min(len(nearby), 8)]
            proposals: list[tuple[int, ...]] = []
            proposals.append(tuple(sorted(int(ids[i]) for i in pool[:k])))
            for replacement in range(k, len(pool)):
                chosen = list(pool[:k])
                chosen[-1] = pool[replacement]
                proposals.append(tuple(sorted(int(ids[i]) for i in chosen)))

            for chosen_ids in proposals:
                chosen_items = [by_id[value] for value in chosen_ids]
                pos = np.asarray(
                    [np.mean(np.asarray(x.transition_midpoints_um), axis=0) for x in chosen_items],
                    dtype=np.float64,
                )
                if float(pairwise(pos, pos).max()) > float(args.group_radius_um):
                    continue
                group = compute_group(
                    chosen_items,
                    pattern=pattern,
                    min_wrong_source=int(args.min_wrong_candidates_per_source),
                    min_wrong_target=int(args.min_wrong_candidates_per_target),
                )
                if group is not None:
                    result[(pattern_name, int(first_start), group.track_ids)] = group
    return list(result.values())


def hardness_key(group: HardGroup, tie: float) -> tuple[Any, ...]:
    return (
        -group.wrong_closer_count,
        -group.rank_gt1_count,
        group.min_margin_um,
        group.median_margin_um,
        -group.max_gap,
        -group.density,
        tie,
    )


def select_groups(candidates: list[HardGroup], args: argparse.Namespace) -> list[HardGroup]:
    rng = np.random.default_rng(int(args.seed))
    tiebreak = {id(c): float(rng.random()) for c in candidates}
    ordered = sorted(candidates, key=lambda c: hardness_key(c, tiebreak[id(c)]))
    chains = [c for c in ordered if c.pattern_kind == "chain"]
    pairs = [c for c in ordered if c.pattern_kind == "pair"]

    requested = int(args.examples)
    chain_quota = int(round(requested * float(args.chain_fraction)))
    pair_quota = requested - chain_quota

    track_use: Counter[int] = Counter()
    placement_use: Counter[tuple[str, int]] = Counter()
    chosen: list[HardGroup] = []
    keys: set[tuple[str, int, tuple[int, ...]]] = set()

    def admissible(c: HardGroup) -> bool:
        key = (c.pattern_name, c.first_start, c.track_ids)
        if key in keys:
            return False
        if placement_use[(c.pattern_name, c.first_start)] >= int(args.max_placement_reuse):
            return False
        return not any(track_use[t] >= int(args.max_track_reuse) for t in c.track_ids)

    def add(c: HardGroup) -> None:
        chosen.append(c)
        keys.add((c.pattern_name, c.first_start, c.track_ids))
        track_use.update(c.track_ids)
        placement_use[(c.pattern_name, c.first_start)] += 1

    for pool, quota in ((chains, chain_quota), (pairs, pair_quota)):
        count = 0
        for c in pool:
            if admissible(c):
                add(c)
                count += 1
            if count >= quota:
                break

    for c in ordered:
        if len(chosen) >= requested:
            break
        if admissible(c):
            add(c)
    return chosen


# =============================================================================
# Materialization
# =============================================================================


def global_lookup(global_motion: pd.DataFrame) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for row in global_motion.itertuples(index=False):
        shift = np.asarray([row.shift_z_um, row.shift_y_um, row.shift_x_um], dtype=np.float64)
        if np.isfinite(shift).all():
            result[int(row.frame_from)] = shift
    return result


def cumulative_global(lookup: dict[int, np.ndarray], source_end: int, target_start: int) -> tuple[np.ndarray, bool]:
    pieces: list[np.ndarray] = []
    for frame in range(int(source_end), int(target_start)):
        shift = lookup.get(frame)
        if shift is None:
            return np.zeros(3, dtype=np.float64), False
        pieces.append(shift)
    return (np.sum(np.asarray(pieces), axis=0) if pieces else np.zeros(3)), True


def source_relative_velocity(rows: pd.DataFrame, lookup: dict[int, np.ndarray]) -> tuple[np.ndarray, int, float]:
    rows = rows.sort_values("frame")
    xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    frames = rows["frame"].to_numpy(dtype=np.int64)
    residuals: list[np.ndarray] = []
    for i in range(len(rows) - 1):
        if frames[i + 1] != frames[i] + 1:
            continue
        shift = lookup.get(int(frames[i]))
        if shift is not None:
            residuals.append((xyz[i + 1] - xyz[i]) - shift)
    if not residuals:
        return np.zeros(3, dtype=np.float64), 0, float("nan")
    recent = np.asarray(residuals[-3:], dtype=np.float64)
    mean = recent.mean(axis=0)
    error = float(np.linalg.norm(recent - mean[None, :], axis=1).mean())
    return mean, int(len(recent)), error


def fragment_role(fragment: int, count: int) -> str:
    if count == 2:
        return "source" if fragment == 0 else "target"
    if fragment == 0:
        return "source"
    if fragment == count - 1:
        return "target"
    return "middle"


def materialize(
    selected: list[HardGroup],
    *,
    observations: pd.DataFrame,
    eligible: dict[tuple[str, int], list[Eligible]],
    patterns: dict[str, Pattern],
    global_motion: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_track = {
        int(track_id): group.sort_values("frame")
        for track_id, group in observations.groupby("track_id", sort=False)
    }
    eligible_map = {
        key: {x.track_id: x for x in items}
        for key, items in eligible.items()
    }
    shifts = global_lookup(global_motion)

    example_rows: list[dict[str, Any]] = []
    tracklet_rows: list[dict[str, Any]] = []
    observation_parts: list[pd.DataFrame] = []
    edge_rows: list[dict[str, Any]] = []

    for example_id, hard in enumerate(selected):
        pattern = patterns[hard.pattern_name]
        track_ids = list(hard.track_ids)
        k = len(track_ids)
        f_count = len(pattern.lengths)
        local_eligible = eligible_map[(hard.pattern_name, hard.first_start)]

        def node(identity: int, fragment: int) -> int:
            return identity * f_count + fragment

        starts: dict[tuple[int, int], np.ndarray] = {}
        ends: dict[tuple[int, int], np.ndarray] = {}
        fragment_frames: dict[tuple[int, int], tuple[int, int]] = {}
        relative: dict[tuple[int, int], tuple[np.ndarray, int, float]] = {}

        for identity, track_id in enumerate(track_ids):
            original = by_track[track_id]
            info = local_eligible[track_id]
            for fragment, (start, end) in enumerate(info.intervals):
                rows = original[(original["frame"] >= start) & (original["frame"] <= end)].copy()
                if len(rows) != int(pattern.lengths[fragment]):
                    raise RuntimeError("Persistent fragment lost observations during materialization")
                idx = node(identity, fragment)
                role = fragment_role(fragment, f_count)
                payload = rows.copy()
                payload.insert(0, "example_id", int(example_id))
                payload.insert(1, "tracklet_index", int(idx))
                payload.insert(2, "identity_index", int(identity))
                payload.insert(3, "fragment_index", int(fragment))
                payload.insert(4, "role", role)
                payload.insert(5, "sequence_index", np.arange(len(payload), dtype=np.int64))
                payload["original_track_id"] = int(track_id)
                observation_parts.append(payload)

                tracklet_rows.append(
                    {
                        "example_id": int(example_id),
                        "tracklet_index": int(idx),
                        "identity_index": int(identity),
                        "fragment_index": int(fragment),
                        "fragment_count": int(f_count),
                        "role": role,
                        "original_track_id": int(track_id),
                        "true_predecessor_tracklet_index": int(node(identity, fragment - 1) if fragment > 0 else -1),
                        "true_successor_tracklet_index": int(node(identity, fragment + 1) if fragment + 1 < f_count else -1),
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "observation_count": int(len(payload)),
                        "division_prior_target": 0,
                    }
                )
                xyz = payload[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
                starts[(identity, fragment)] = xyz[0]
                ends[(identity, fragment)] = xyz[-1]
                fragment_frames[(identity, fragment)] = (int(start), int(end))
                relative[(identity, fragment)] = source_relative_velocity(payload, shifts)

        edge_id = 0
        transition_report: list[dict[str, Any]] = []
        for transition, gap in enumerate(pattern.gaps):
            sf = transition
            tf = transition + 1
            radius = candidate_radius_um(gap)
            source_xyz = np.asarray([ends[(i, sf)] for i in range(k)], dtype=np.float64)
            target_xyz = np.asarray([starts[(i, tf)] for i in range(k)], dtype=np.float64)
            distance = pairwise(source_xyz, target_xyz)
            candidate = distance <= radius
            np.fill_diagonal(candidate, True)

            source_rank = np.zeros_like(distance, dtype=np.int64)
            target_rank = np.zeros_like(distance, dtype=np.int64)
            for i in range(k):
                available = np.flatnonzero(candidate[i])
                order = available[np.argsort(distance[i, available])]
                for rank, j in enumerate(order, start=1):
                    source_rank[i, j] = rank
            for j in range(k):
                available = np.flatnonzero(candidate[:, j])
                order = available[np.argsort(distance[available, j])]
                for rank, i in enumerate(order, start=1):
                    target_rank[i, j] = rank

            transition_margins: list[float] = []
            transition_wrong_closer = 0

            for i in range(k):
                source_end_frame = fragment_frames[(i, sf)][1]
                # All identities share the same synthetic transition frames.
                target_start_frame = fragment_frames[(i, tf)][0]
                global_delta, global_valid = cumulative_global(shifts, source_end_frame, target_start_frame)
                expected_global = source_xyz[i] + global_delta if global_valid else np.full(3, np.nan)
                rv, rv_samples, rv_error = relative[(i, sf)]
                expected_relative = (
                    expected_global + rv * float(gap)
                    if global_valid and rv_samples > 0
                    else np.full(3, np.nan)
                )

                positive_distance = float(distance[i, i])
                wrong_mask = candidate[i].copy()
                wrong_mask[i] = False
                nearest_wrong = float(distance[i][wrong_mask].min())
                margin = nearest_wrong - positive_distance
                transition_margins.append(margin)
                transition_wrong_closer += int(margin < 0.0)

                for j in range(k):
                    if not candidate[i, j]:
                        continue
                    target_position = target_xyz[j]
                    is_positive = i == j
                    edge_rows.append(
                        {
                            "example_id": int(example_id),
                            "edge_index": int(edge_id),
                            "transition_index": int(transition),
                            "source_fragment_index": int(sf),
                            "target_fragment_index": int(tf),
                            "source_tracklet_index": int(node(i, sf)),
                            "target_tracklet_index": int(node(j, tf)),
                            "source_identity_index": int(i),
                            "target_identity_index": int(j),
                            "source_original_track_id": int(track_ids[i]),
                            "target_original_track_id": int(track_ids[j]),
                            "gap_frames": int(gap),
                            "candidate_radius_um": float(radius),
                            "continuation_target": int(is_positive),
                            "direct_distance_um": float(distance[i, j]),
                            "source_distance_rank": int(source_rank[i, j]),
                            "target_distance_rank": int(target_rank[i, j]),
                            "positive_direct_distance_um": positive_distance,
                            "wrong_closer_than_positive": int((not is_positive) and distance[i, j] < positive_distance),
                            "global_prediction_valid": int(global_valid),
                            "expected_global_z_um": float(expected_global[0]),
                            "expected_global_y_um": float(expected_global[1]),
                            "expected_global_x_um": float(expected_global[2]),
                            "global_error_um": float(np.linalg.norm(target_position - expected_global)) if global_valid else np.nan,
                            "global_relative_prediction_valid": int(global_valid and rv_samples > 0),
                            "relative_velocity_samples": int(rv_samples),
                            "relative_velocity_error_um": float(rv_error),
                            "expected_global_relative_z_um": float(expected_relative[0]),
                            "expected_global_relative_y_um": float(expected_relative[1]),
                            "expected_global_relative_x_um": float(expected_relative[2]),
                            "global_relative_error_um": float(np.linalg.norm(target_position - expected_relative)) if global_valid and rv_samples > 0 else np.nan,
                        }
                    )
                    edge_id += 1

            transition_report.append(
                {
                    "transition_index": int(transition),
                    "gap_frames": int(gap),
                    "candidate_radius_um": float(radius),
                    "wrong_closer_sources": int(transition_wrong_closer),
                    "minimum_margin_um": float(np.min(transition_margins)),
                    "median_margin_um": float(np.median(transition_margins)),
                }
            )

        intervals = pattern.intervals(hard.first_start)
        example_rows.append(
            {
                "example_id": int(example_id),
                "pattern_name": pattern.name,
                "pattern_kind": pattern.kind,
                "fragment_count": int(f_count),
                "transition_count": int(pattern.transition_count),
                "first_start_frame": int(hard.first_start),
                "last_end_frame": int(intervals[-1][1]),
                "fragment_lengths": json.dumps(list(pattern.lengths)),
                "fragment_gaps": json.dumps(list(pattern.gaps)),
                "fragment_intervals": json.dumps([list(v) for v in intervals]),
                "track_count": int(k),
                "synthetic_tracklet_count": int(k * f_count),
                "candidate_edge_count": int(edge_id),
                "positive_edge_count": int(k * pattern.transition_count),
                "negative_edge_count": int(edge_id - k * pattern.transition_count),
                "original_track_ids": json.dumps(track_ids),
                "wrong_closer_source_count": int(hard.wrong_closer_count),
                "positive_source_rank_gt1_count": int(hard.source_rank_gt1_count),
                "positive_target_rank_gt1_count": int(hard.target_rank_gt1_count),
                "minimum_wrong_minus_true_um": float(hard.min_margin_um),
                "median_wrong_minus_true_um": float(hard.median_margin_um),
                "mean_wrong_minus_true_um": float(hard.mean_margin_um),
                "edge_density": float(hard.density),
                "maximum_gap_frames": int(hard.max_gap),
                "scene_diameter_um": float(hard.scene_diameter_um),
                "transition_diagnostics": json.dumps(transition_report),
            }
        )

    return (
        pd.DataFrame(example_rows),
        pd.DataFrame(tracklet_rows),
        pd.concat(observation_parts, ignore_index=True) if observation_parts else pd.DataFrame(),
        pd.DataFrame(edge_rows),
    )


# =============================================================================
# Contract checks / reporting
# =============================================================================


def validate_dataset(examples: pd.DataFrame, tracklets: pd.DataFrame, observations: pd.DataFrame, edges: pd.DataFrame) -> None:
    if examples.empty or edges.empty:
        raise RuntimeError("Hard dataset is empty")

    for example_id in examples["example_id"].astype(int):
        ex = examples[examples["example_id"].astype(int) == example_id].iloc[0]
        local_tracklets = tracklets[tracklets["example_id"].astype(int) == example_id]
        local_obs = observations[observations["example_id"].astype(int) == example_id]
        local_edges = edges[edges["example_id"].astype(int) == example_id]
        if len(local_tracklets) != int(ex.synthetic_tracklet_count):
            raise RuntimeError(f"example {example_id}: synthetic tracklet count mismatch")

        for t in local_tracklets.itertuples(index=False):
            rows = local_obs[local_obs["tracklet_index"].astype(int) == int(t.tracklet_index)].sort_values("sequence_index")
            if len(rows) != int(t.observation_count):
                raise RuntimeError(f"example {example_id}, tracklet {t.tracklet_index}: observation count mismatch")
            frames = rows["frame"].to_numpy(dtype=np.int64)
            if len(frames) > 1 and not np.all(np.diff(frames) == 1):
                raise RuntimeError(f"example {example_id}, tracklet {t.tracklet_index}: fragment not consecutive")

        for transition, group in local_edges.groupby("transition_index"):
            positives = group[group["continuation_target"].astype(int) == 1]
            if len(positives) != int(ex.track_count):
                raise RuntimeError(f"example {example_id}, transition {transition}: positive count mismatch")
            if len(positives.groupby("source_tracklet_index")) != int(ex.track_count):
                raise RuntimeError(f"example {example_id}, transition {transition}: source uniqueness failed")
            if len(positives.groupby("target_tracklet_index")) != int(ex.track_count):
                raise RuntimeError(f"example {example_id}, transition {transition}: target uniqueness failed")

    # The important topology check: the exact same middle node must appear as
    # target in transition 0 and source in transition 1.
    chain_ids = examples.loc[examples["pattern_kind"] == "chain", "example_id"].astype(int)
    for example_id in chain_ids:
        middle = tracklets[(tracklets["example_id"].astype(int) == example_id) & (tracklets["role"] == "middle")]
        local_edges = edges[edges["example_id"].astype(int) == example_id]
        for t in middle.itertuples(index=False):
            idx = int(t.tracklet_index)
            if not (local_edges["target_tracklet_index"].astype(int) == idx).any():
                raise RuntimeError(f"chain example {example_id}: middle node {idx} has no incoming edges")
            if not (local_edges["source_tracklet_index"].astype(int) == idx).any():
                raise RuntimeError(f"chain example {example_id}: middle node {idx} has no outgoing edges")


def report(candidates: list[HardGroup], selected: list[HardGroup], examples: pd.DataFrame, tracklets: pd.DataFrame, edges: pd.DataFrame) -> None:
    positives = edges[edges["continuation_target"].astype(int) == 1]
    print("", flush=True)
    print("=" * 124, flush=True)
    print("HARD CONTINUATION DATASET BUILT", flush=True)
    print("=" * 124, flush=True)
    print(f"candidate scenes         : {len(candidates):,}", flush=True)
    print(
        f"selected examples        : {len(selected):,} "
        f"({sum(x.pattern_kind == 'chain' for x in selected)} chain / "
        f"{sum(x.pattern_kind == 'pair' for x in selected)} pair)",
        flush=True,
    )
    print(f"synthetic tracklets      : {len(tracklets):,}", flush=True)
    print(f"middle tracklets         : {int((tracklets.role == 'middle').sum()):,}", flush=True)
    print(
        f"candidate edges          : {len(edges):,} "
        f"({len(positives)} positive / {int((edges.continuation_target == 0).sum())} negative)",
        flush=True,
    )
    print(f"wrong closer decisions   : {sum(x.wrong_closer_count for x in selected)}", flush=True)
    print(f"positive geometric rank>1: {sum(x.rank_gt1_count for x in selected)} (source+target)", flush=True)
    print(
        "hardness margin         : "
        f"median={examples.median_wrong_minus_true_um.median():+.3f} um, "
        f"min={examples.minimum_wrong_minus_true_um.min():+.3f} um",
        flush=True,
    )
    print(
        "positive gap distribution: "
        + str(positives.gap_frames.astype(int).value_counts().sort_index().to_dict()),
        flush=True,
    )
    print("=" * 124, flush=True)

    cols = [
        "example_id", "pattern_name", "pattern_kind", "track_count",
        "transition_count", "candidate_edge_count", "wrong_closer_source_count",
        "positive_source_rank_gt1_count", "positive_target_rank_gt1_count",
        "minimum_wrong_minus_true_um", "median_wrong_minus_true_um", "maximum_gap_frames",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", lambda x: f"{x:.3f}"):
        print(
            examples.sort_values(
                ["wrong_closer_source_count", "minimum_wrong_minus_true_um"],
                ascending=[False, True],
            )[cols].head(min(24, len(examples))).to_string(index=False),
            flush=True,
        )


# =============================================================================
# Investigation-03 adapter patches and in-process training
# =============================================================================


def patch_inv03(trainer) -> None:
    # 1) Every tracklet has both a first (incoming) and last (outgoing)
    # endpoint. This is the only structural change needed for a shared middle B.
    def endpoints_any_role(observations: pd.DataFrame):
        sources: dict[tuple[int, int], pd.Series] = {}
        targets: dict[tuple[int, int], pd.Series] = {}
        for (example_id, tracklet_index), group in observations.groupby(["example_id", "tracklet_index"]):
            group = group.sort_values("sequence_index")
            key = (int(example_id), int(tracklet_index))
            targets[key] = group.iloc[0]
            sources[key] = group.iloc[-1]
        return sources, targets

    trainer._endpoint_rows = endpoints_any_role

    # 2) Investigation 03 assumed one radius for every gap. Hard examples use
    # the production gap-specific radius stored per edge.
    original_enrich = trainer.enrich_pair_features

    def enrich_gap_specific(data, *, tracklet_diagnostics):
        frame = original_enrich(data, tracklet_diagnostics=tracklet_diagnostics)
        if "candidate_radius_um" in frame.columns:
            radius = pd.to_numeric(frame["candidate_radius_um"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            distance = pd.to_numeric(frame["direct_endpoint_distance_um"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            valid = np.isfinite(radius) & (radius > 0) & np.isfinite(distance)
            frame.loc[valid, "hard_search_radius_um"] = radius[valid]
            frame.loc[valid, "candidate_quality_score"] = np.exp(-distance[valid] / radius[valid])
        return frame

    trainer.enrich_pair_features = enrich_gap_specific

    # 3) Avoid amplifying nominally constant reliability channels.
    def reliability_stats(raw: np.ndarray):
        mean = raw.mean(axis=0).astype(np.float32)
        std = raw.std(axis=0).astype(np.float32)
        std[std < 1.0e-4] = 1.0
        for name in ("touches_boundary", "small_cell_indicator"):
            index = trainer.TRACKLET_RELIABILITY_FEATURES.index(name)
            mean[index] = 0.0
            std[index] = 1.0
        return trainer.MaskedStats(mean=mean, std=std)

    trainer.fit_reliability_stats = reliability_stats

    # 4) Pandas 3/mmap may expose read-only NumPy arrays. Make only the small
    # read-only prepared arrays writable before the existing torch.from_numpy
    # collator sees them; large crop arrays are already writable.
    original_prepare = trainer.prepare_examples

    def prepare_writable(*args, **kwargs):
        prepared, stats = original_prepare(*args, **kwargs)
        for example in prepared:
            for name, value in vars(example).items():
                if isinstance(value, np.ndarray) and not value.flags.writeable:
                    setattr(example, name, np.array(value, copy=True))
        return prepared, stats

    trainer.prepare_examples = prepare_writable


def run_training(dataset_dir: Path, training_dir: Path, args: argparse.Namespace) -> int:
    trainer = load_inv03()
    patch_inv03(trainer)

    argv = [
        str(ROOT / "investigations" / "track_reconciler" / "03_continuation_overfit.py"),
        "--sample-id", str(args.sample_id),
        "--dataset", str(dataset_dir),
        "--output", str(training_dir),
        "--steps", str(int(args.train_steps)),
        "--batch-size", str(int(args.train_batch_size)),
        "--lr", str(float(args.train_lr)),
        "--device", str(args.train_device),
        "--amp", str(args.train_amp),
        "--appearance-mode", "crops",
        "--crop-shape", str(args.crop_shape),
    ]
    if args.rebuild_crops:
        argv.append("--rebuild-crops")
    if args.overwrite or args.overwrite_training:
        argv.append("--overwrite")

    previous = list(sys.argv)
    try:
        sys.argv = argv
        return int(trainer.main())
    finally:
        sys.argv = previous


# =============================================================================
# CLI / main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine hard continuation graphs and immediately overfit the reconciler.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)

    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--tracks-per-example", type=int, default=DEFAULT_TRACKS_PER_EXAMPLE)
    parser.add_argument("--guard-frames", type=int, default=DEFAULT_GUARD_FRAMES)
    parser.add_argument("--group-radius-um", type=float, default=DEFAULT_GROUP_RADIUS_UM)
    parser.add_argument("--min-wrong-candidates-per-source", type=int, default=1)
    parser.add_argument("--min-wrong-candidates-per-target", type=int, default=1)
    parser.add_argument("--max-cell-match-um", type=float, default=DEFAULT_MAX_CELL_MATCH_UM)
    parser.add_argument("--max-step-um", type=float, default=DEFAULT_MAX_STEP_UM)
    parser.add_argument("--max-volume-ratio", type=float, default=DEFAULT_MAX_VOLUME_RATIO)
    parser.add_argument("--min-boundary-distance-um", type=float, default=DEFAULT_MIN_BOUNDARY_DISTANCE_UM)
    parser.add_argument("--max-track-reuse", type=int, default=DEFAULT_MAX_TRACK_REUSE)
    parser.add_argument("--max-placement-reuse", type=int, default=DEFAULT_MAX_PLACEMENT_REUSE)
    parser.add_argument("--chain-fraction", type=float, default=DEFAULT_CHAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-lr", type=float, default=DEFAULT_TRAIN_LR)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--train-amp", choices=("auto", "off", "fp16", "bf16"), default="off")
    parser.add_argument("--crop-shape", default=DEFAULT_CROP_SHAPE)
    parser.add_argument("--rebuild-crops", action="store_true")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-training", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("examples", "tracks_per_example", "max_track_reuse", "max_placement_reuse", "train_steps", "train_batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >0")
    if int(args.tracks_per_example) < 2:
        raise ValueError("--tracks-per-example must be >=2")
    if int(args.guard_frames) < 0:
        raise ValueError("--guard-frames must be >=0")
    if not 0.0 <= float(args.chain_fraction) <= 1.0:
        raise ValueError("--chain-fraction must be in [0,1]")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    output = resolve(args.output) if args.output is not None else default_output(args.sample_id)
    dataset_dir = output / "dataset"
    training_dir = output / "training"

    if args.train_only:
        if not (dataset_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"Hard dataset not found: {dataset_dir}")
        print(f"[04] reusing hard dataset: {dataset_dir}", flush=True)
        return run_training(dataset_dir, training_dir, args)

    if args.overwrite and output.exists():
        shutil.rmtree(output)
    elif dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(
            f"Hard dataset already exists: {dataset_dir}\n"
            "Use --overwrite to rebuild or --train-only to reuse it."
        )

    inv01 = load_inv01()
    source_root = resolve(args.source) if args.source is not None else default_source(args.sample_id)
    source = inv01.SourcePaths(source_root)
    source.validate()
    spacing = tuple(float(v) for v in inv01.DEFAULT_SPACING_ZYX_UM)

    print("=" * 124, flush=True)
    print("TRACK RECONCILER — INVESTIGATION 04: HARD CONTINUATION MINING + OVERFIT", flush=True)
    print("=" * 124, flush=True)
    print(f"sample                   : {args.sample_id}", flush=True)
    print(f"source Investigation 36  : {source.root}", flush=True)
    print(f"output                   : {output}", flush=True)
    print(f"examples requested       : {args.examples}", flush=True)
    print(f"tracks/example           : {args.tracks_per_example}", flush=True)
    print(f"chain target fraction    : {args.chain_fraction:.2f}", flush=True)
    print("patterns                 : " + ", ".join(p.name for p in PATTERNS), flush=True)
    print("=" * 124, flush=True)

    observations, lineage_ids, movie_shape = inv01.load_observations(source, spacing=spacing)
    global_motion = inv01.estimate_global_motion(observations, lineage_ids=lineage_ids)
    print(
        f"[source] rows={len(observations):,} tracklets={observations.track_id.nunique():,} "
        f"lineage-related excluded={len(lineage_ids):,}",
        flush=True,
    )
    print(
        f"[motion] transitions={len(global_motion)} median support={global_motion.support_tracks.median():.0f}",
        flush=True,
    )

    started = time.perf_counter()
    pattern_map = {p.name: p for p in PATTERNS}
    eligible = enumerate_eligible(
        observations,
        movie_frames=int(movie_shape[0]),
        guard=int(args.guard_frames),
        args=args,
    )
    eligible_count = sum(len(v) for v in eligible.values())
    print(f"[eligible] pattern-track windows={eligible_count:,} placements={len(eligible):,}", flush=True)

    candidates = enumerate_groups(eligible, patterns=pattern_map, args=args)
    print(f"[mining] ambiguous local scenes={len(candidates):,}", flush=True)
    if not candidates:
        raise RuntimeError(
            "No hard local scenes found. First try --group-radius-um 35. "
            "Do not loosen persistence/identity quality filters first."
        )

    selected = select_groups(candidates, args)
    if not selected:
        raise RuntimeError("Selection constraints rejected every hard scene")
    if len(selected) < int(args.examples):
        print(
            f"[warning] requested {args.examples} examples but selected {len(selected)} under reuse constraints.",
            flush=True,
        )

    examples, tracklets, selected_obs, edges = materialize(
        selected,
        observations=observations,
        eligible=eligible,
        patterns=pattern_map,
        global_motion=global_motion,
    )
    validate_dataset(examples, tracklets, selected_obs, edges)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(dataset_dir / "examples.csv", examples)
    atomic_csv(dataset_dir / "tracklets.csv", tracklets)
    atomic_csv(dataset_dir / "observations.csv", selected_obs)
    atomic_csv(dataset_dir / "edges.csv", edges)
    atomic_csv(dataset_dir / "global_motion.csv", global_motion)

    positives = edges[edges["continuation_target"].astype(int) == 1]
    source_refs = {
        "investigation_36_root": root_relative(source.root),
        "raw": root_relative(source.raw),
        "preprocessed": root_relative(source.preprocessed),
        "final_instances": root_relative(source.final_instances),
        "cells": root_relative(source.cells),
        "tracks": root_relative(source.tracks),
        "napari_graph": root_relative(source.napari_graph),
    }
    stats = {
        "source_track_rows": int(len(observations)),
        "source_tracklets": int(observations.track_id.nunique()),
        "lineage_related_tracklets_excluded": int(len(lineage_ids)),
        "eligible_pattern_track_windows": int(eligible_count),
        "candidate_hard_scenes": int(len(candidates)),
        "selected_examples": int(len(examples)),
        "chain_examples": int((examples.pattern_kind == "chain").sum()),
        "pair_examples": int((examples.pattern_kind == "pair").sum()),
        "synthetic_tracklets": int(len(tracklets)),
        "middle_tracklets": int((tracklets.role == "middle").sum()),
        "selected_observation_rows": int(len(selected_obs)),
        "candidate_edges": int(len(edges)),
        "positive_edges": int(len(positives)),
        "negative_edges": int((edges.continuation_target == 0).sum()),
        "wrong_closer_source_decisions": int(sum(x.wrong_closer_count for x in selected)),
        "positive_source_rank_gt1": int((positives.source_distance_rank.astype(int) > 1).sum()),
        "positive_target_rank_gt1": int((positives.target_distance_rank.astype(int) > 1).sum()),
        "minimum_wrong_minus_true_um": float(examples.minimum_wrong_minus_true_um.min()),
        "median_example_wrong_minus_true_um": float(examples.median_wrong_minus_true_um.median()),
    }
    manifest = {
        "schema_version": 2,
        "investigation": SCRIPT_NAME,
        "purpose": "Hard continuation overfit with variable gaps, asymmetric histories and shared-middle A->B->C chains.",
        "sample_id": str(args.sample_id),
        "source": source_refs,
        "movie_shape_tzyx": list(movie_shape),
        "spacing_zyx_um": list(spacing),
        "parameters": {
            "examples_requested": int(args.examples),
            "tracks_per_example": int(args.tracks_per_example),
            "guard_frames": int(args.guard_frames),
            # Fallback consumed by Investigation 03. Runtime patch uses each
            # edge's candidate_radius_um instead.
            "candidate_radius_um": 25.0,
            "candidate_radius_policy": "min(25, 12 + 4*(gap_frames-1)) um",
            "group_radius_um": float(args.group_radius_um),
            "chain_fraction": float(args.chain_fraction),
            "seed": int(args.seed),
        },
        "patterns": [
            {"name": p.name, "kind": p.kind, "lengths": list(p.lengths), "gaps": list(p.gaps)}
            for p in PATTERNS
        ],
        "stats": stats,
        "topology": {
            "shared_middle_tracklet_supported": True,
            "chain_definition": "A source -> B middle -> C target; same B node is target of transition 0 and source of transition 1.",
            "variable_fragment_length_supported": True,
            "variable_gap_supported": True,
            "division": "excluded in this continuation-only experiment",
        },
        "supervision": {
            "positive_continuation": "same original Trackastra identity across adjacent synthetic fragments",
            "negative_continuation": "nearby cross-identity edge inside the gap-specific physical gate",
            "identity_columns_are_model_inputs": False,
        },
        "files": {
            "examples": "examples.csv",
            "tracklets": "tracklets.csv",
            "observations": "observations.csv",
            "edges": "edges.csv",
            "global_motion": "global_motion.csv",
        },
    }
    atomic_json(dataset_dir / "manifest.json", manifest)

    report(candidates, selected, examples, tracklets, edges)
    print(f"[dataset] saved: {dataset_dir}", flush=True)
    print(f"[dataset] build time: {time.perf_counter() - started:.1f}s", flush=True)

    if stats["wrong_closer_source_decisions"] == 0 and (
        stats["positive_source_rank_gt1"] + stats["positive_target_rank_gt1"]
    ) == 0:
        print(
            "[warning] No selected positive loses geometric rank 1. The set is still harder through gaps/chains/asymmetry, "
            "but it is not yet a strict nearest-neighbour defeat.",
            flush=True,
        )
    else:
        print(
            "[hardness] Selected set contains decisions where nearest geometry is not the true identity.",
            flush=True,
        )

    if args.build_only:
        return 0

    print("", flush=True)
    print("[training] starting full crop -> fingerprint -> temporal -> edge-reasoner overfit ...", flush=True)
    code = run_training(dataset_dir, training_dir, args)

    training_summary_path = training_dir / "summary.json"
    training_summary = (
        json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary_path.is_file()
        else None
    )
    atomic_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "investigation": SCRIPT_NAME,
            "sample_id": str(args.sample_id),
            "dataset": root_relative(dataset_dir),
            "training": root_relative(training_dir),
            "dataset_stats": stats,
            "training_return_code": int(code),
            "training_summary": training_summary,
        },
    )

    if training_summary is not None:
        print("", flush=True)
        print("=" * 124, flush=True)
        print("INVESTIGATION 04 COMPLETE", flush=True)
        print("=" * 124, flush=True)
        print(f"strict success : {training_summary.get('strict_success')}", flush=True)
        print(f"final metrics  : {training_summary.get('final_metrics')}", flush=True)
        print(f"dataset        : {dataset_dir}", flush=True)
        print(f"training       : {training_dir}", flush=True)
        print("=" * 124, flush=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
