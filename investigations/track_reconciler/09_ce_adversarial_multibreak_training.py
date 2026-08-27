from __future__ import annotations

r"""
Investigation 09 — adversarial multi-break CE continuation training.

Hard successor to Investigation 08. Uses only clean Fluo-N3DH-CE TRA GT,
reuses the compact CE cache from 08, and synthesizes fresh multi-fragment
reconciliation graphs online.

Key synthesis patterns include:
    -1 0 1 2 | 3 | 4 5 | 6 7 8 9      -> [4,1,2,4]
    -1 0 1 2 | 3 | 4 | 5 6 7 8 9      -> [4,1,1,5]
     0 1 2   | 3 | 4 | 5 | 6 7 8      -> [3,1,1,1,3]

Several nearby identities are fragmented at the SAME boundaries, and stage
ordering is shuffled independently so the network must recover the correct
local permutation repeatedly.

Strict hard modes:
  * wrong_closer      : a wrong successor is closer than the true successor
  * motion_trap       : global+relative prediction prefers a wrong successor
  * rank_hard         : true source or target geometric rank > 1
  * direction_change  : abrupt true motion direction change
  * large_motion      : true displacement is in the CE p90 tail for that gap
  * near_tie          : nearest wrong vs true distance margin is small
  * multi_break       : dense multi-stage fragmentation regardless of geometry

Appearance and division remain disabled. This isolates continuation reasoning.

Run:
    python .\investigations\track_reconciler\09_ce_adversarial_multibreak_training.py --overwrite

Smoke run:
    python .\investigations\track_reconciler\09_ce_adversarial_multibreak_training.py ^
        --steps 100 --stats-scenes 64 --validation-scenes 32 ^
        --train-eval-scenes 32 --overwrite
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

SCRIPT_NAME = "09_ce_adversarial_multibreak_training"
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_TRACKS_PER_SCENE = 5
DEFAULT_SCENE_RADIUS_UM = 25.0
DEFAULT_STATS_SCENES = 768
DEFAULT_VALIDATION_SCENES = 384
DEFAULT_TRAIN_EVAL_SCENES = 192
DEFAULT_STEPS = 7500
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1.5e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_LOG_EVERY = 25
DEFAULT_EVAL_EVERY = 250
DEFAULT_SEED = 20260827
DEFAULT_DIRECTION_DEGREES = 70.0
DEFAULT_NEAR_TIE_MARGIN_UM = 1.5
DEFAULT_PROPOSALS = 24


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "track_reconciler").is_dir()
            and (candidate / "investigations" / "track_reconciler").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_inv08():
    return load_module(
        ROOT / "investigations" / "track_reconciler" / "08_ce_synthetic_continuation_training.py",
        "_inv09_inv08",
    )


def default_dataset_root() -> Path:
    nested = ROOT / "data" / "external" / "Fluo-N3DH-CE" / "Fluo-N3DH-CE_train"
    return nested.resolve() if nested.is_dir() else (ROOT / "data" / "external" / "Fluo-N3DH-CE").resolve()


def default_output() -> Path:
    return (ROOT / "runs" / "track_reconciler" / "investigations" / SCRIPT_NAME).resolve()


def default_base_cache() -> Path:
    return (ROOT / "runs" / "track_reconciler" / "cache" / "fluo_n3dh_ce_tra").resolve()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
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


@dataclass(frozen=True)
class FragmentPattern:
    name: str
    segment_lengths: tuple[int, ...]
    gaps: tuple[int, ...]
    weight: float

    @property
    def span_frames(self) -> int:
        return int(sum(self.segment_lengths) + sum(g - 1 for g in self.gaps))

    def ranges(self, start_frame: int) -> tuple[tuple[int, int], ...]:
        out = []
        cursor = int(start_frame)
        for i, length in enumerate(self.segment_lengths):
            first = cursor
            last = first + int(length) - 1
            out.append((first, last))
            if i < len(self.gaps):
                cursor = last + int(self.gaps[i])
        return tuple(out)


PATTERNS = (
    FragmentPattern("quad_4_1_2_4", (4, 1, 2, 4), (1, 1, 1), 1.5),
    FragmentPattern("quad_4_1_1_5", (4, 1, 1, 5), (1, 1, 1), 1.5),
    FragmentPattern("singletons_3_1_1_1_3", (3, 1, 1, 1, 3), (1, 1, 1, 1), 1.2),
    FragmentPattern("triple_4_1_4", (4, 1, 4), (1, 1), 0.8),
    FragmentPattern("mixed_gap_4_1_2_4", (4, 1, 2, 4), (1, 2, 1), 1.0),
    FragmentPattern("long_bridge_4_2_4", (4, 2, 4), (3, 2), 0.8),
    FragmentPattern("rapid_gap_3_1_1_3", (3, 1, 1, 3), (2, 1, 2), 0.8),
)
PATTERN_LOOKUP = {p.name: p for p in PATTERNS}
HARD_MODES = (
    ("wrong_closer", 0.22),
    ("motion_trap", 0.20),
    ("rank_hard", 0.15),
    ("direction_change", 0.13),
    ("large_motion", 0.10),
    ("near_tie", 0.10),
    ("multi_break", 0.10),
)


def weighted_name(rng, items):
    w = np.asarray([x[1] for x in items], dtype=np.float64)
    w /= w.sum()
    return items[int(rng.choice(len(items), p=w))][0]


def choose_pattern(rng):
    w = np.asarray([p.weight for p in PATTERNS], dtype=np.float64)
    w /= w.sum()
    return PATTERNS[int(rng.choice(len(PATTERNS), p=w))]


@dataclass(frozen=True)
class PatternPlacement:
    sequence: str
    pattern_name: str
    start_frame: int


class PatternPool:
    def __init__(self, *, store, split_table, split_name, tracks_per_scene):
        self.store = store
        self.split_name = str(split_name)
        allowed = {"01": set(), "02": set()}
        for row in split_table.itertuples(index=False):
            if str(row.split) == split_name:
                allowed[str(row.sequence).zfill(2)].add(int(row.track_id))
        self.by_key = {}
        for sequence in ("01", "02"):
            series_list = [store[(sequence, tid)] for tid in sorted(allowed[sequence]) if (sequence, tid) in store]
            if not series_list:
                continue
            first = min(int(s.frames[0]) for s in series_list)
            last = max(int(s.frames[-1]) for s in series_list)
            for pattern in PATTERNS:
                for start in range(first, last - pattern.span_frames + 2):
                    end = start + pattern.span_frames - 1
                    ids = [int(s.track_id) for s in series_list if s.has_range(start, end)]
                    if len(ids) >= int(tracks_per_scene):
                        self.by_key[PatternPlacement(sequence, pattern.name, int(start))] = tuple(ids)
        self.keys_by_pattern = {
            p.name: tuple(k for k in self.by_key if k.pattern_name == p.name)
            for p in PATTERNS
        }
        if not self.by_key:
            raise RuntimeError(f"No {split_name} multi-break placements")

    def summary(self):
        return {
            "split": self.split_name,
            "placement_count": len(self.by_key),
            "mean_tracks_per_placement": float(np.mean([len(v) for v in self.by_key.values()])),
            "pattern_counts": {name: len(keys) for name, keys in self.keys_by_pattern.items()},
        }


def angle_degrees(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def displacement_quantiles(store, train_ids):
    samples = {1: [], 2: [], 3: [], 4: []}
    for key in train_ids:
        series = store.get(key)
        if series is None:
            continue
        index = {int(f): i for i, f in enumerate(series.frames.tolist())}
        for i, frame in enumerate(series.frames.tolist()):
            for gap in samples:
                j = index.get(int(frame) + gap)
                if j is None or not series.has_range(int(frame), int(frame) + gap):
                    continue
                samples[gap].append(float(np.linalg.norm(series.xyz_um[j] - series.xyz_um[i])))
    return {
        gap: {
            "p90": float(np.quantile(v, 0.90)) if v else 0.0,
            "p95": float(np.quantile(v, 0.95)) if v else 0.0,
            "max": float(np.max(v)) if v else 0.0,
        }
        for gap, v in samples.items()
    }


@dataclass(frozen=True)
class Difficulty:
    wrong_closer_count: int
    source_rank_gt1_count: int
    target_rank_gt1_count: int
    motion_trap_count: int
    direction_change_max_deg: float
    large_motion_count: int
    minimum_wrong_minus_true_um: float
    minimum_motion_wrong_minus_true_um: float
    transition_count: int


@dataclass(frozen=True)
class HardSceneSpec:
    example_id: int
    split: str
    sequence: str
    pattern_name: str
    start_frame: int
    track_ids: tuple[int, ...]
    stage_orders: tuple[tuple[int, ...], ...]
    requested_mode: str
    strict_mode_satisfied: bool
    difficulty: Difficulty

    def to_row(self):
        return {
            "example_id": self.example_id,
            "split": self.split,
            "sequence": self.sequence,
            "pattern_name": self.pattern_name,
            "start_frame": self.start_frame,
            "track_ids": json.dumps(self.track_ids),
            "stage_orders": json.dumps(self.stage_orders),
            "requested_mode": self.requested_mode,
            "strict_mode_satisfied": int(self.strict_mode_satisfied),
            **{f"difficulty_{k}": v for k, v in asdict(self.difficulty).items()},
        }


def motion_prediction(series, source_first, source_last, target_frame, global_lookup, inv08):
    source_xyz = series.xyz_at(source_last)
    delta, global_valid = inv08.cumulative_global_shift(global_lookup, source_last, target_frame)
    expected = source_xyz + delta if global_valid else source_xyz.copy()
    rows = series.rows(source_first, source_last, role="source")
    rv, rv_valid = inv08.source_relative_velocity(rows, global_lookup)
    gap = target_frame - source_last
    if global_valid and rv_valid:
        return (expected + rv * float(gap)).astype(np.float64), True
    return expected.astype(np.float64), bool(global_valid)


def score_group(*, sequence, pattern, start_frame, track_ids, store, global_motion, displacement_q, inv08):
    ranges = pattern.ranges(start_frame)
    wc = sr = tr = mt = lm = transitions = 0
    max_angle = 0.0
    margins = []; motion_margins = []
    for stage in range(len(ranges) - 1):
        s_first, s_last = ranges[stage]
        t_first, _ = ranges[stage + 1]
        gap = t_first - s_last
        sx = np.stack([store[(sequence, tid)].xyz_at(s_last) for tid in track_ids]).astype(np.float64)
        tx = np.stack([store[(sequence, tid)].xyz_at(t_first) for tid in track_ids]).astype(np.float64)
        dist = np.linalg.norm(sx[:, None] - tx[None], axis=-1)
        radius = min(25.0, 12.0 + 4.0 * max(gap - 1, 0))
        for i, tid in enumerate(track_ids):
            transitions += 1
            true = float(dist[i, i])
            wrong_mask = np.arange(len(track_ids)) != i
            wrong = dist[i, wrong_mask]
            # Difficulty only counts wrong edges that would actually survive
            # the production candidate gate. True GT edges are always retained.
            wrong = wrong[wrong <= radius]
            nearest = float(np.min(wrong)) if len(wrong) else float("inf")
            margins.append(nearest - true)
            wc += int(nearest < true)
            sr += int(np.sum(wrong < true) > 0)
            threshold = displacement_q.get(gap, {}).get("p90", float("inf"))
            lm += int(threshold > 0 and true >= threshold)
            series = store[(sequence, tid)]
            if series.index_of(s_last - 1) is not None:
                max_angle = max(max_angle, angle_degrees(series.xyz_at(s_last) - series.xyz_at(s_last - 1), series.xyz_at(t_first) - series.xyz_at(s_last)))
            expected, ok = motion_prediction(series, s_first, s_last, t_first, global_motion[sequence], inv08)
            if ok:
                errors = np.linalg.norm(tx - expected[None], axis=1)
                eligible_wrong = errors[wrong_mask & (dist[i] <= radius)]
                true_e = float(errors[i])
                nearest_e = float(np.min(eligible_wrong)) if len(eligible_wrong) else float("inf")
                motion_margins.append(nearest_e - true_e)
                mt += int(nearest_e < true_e)
        for j in range(len(track_ids)):
            true = float(dist[j, j])
            source_mask = np.arange(len(track_ids)) != j
            wrong = dist[source_mask, j]
            wrong = wrong[wrong <= radius]
            tr += int(np.sum(wrong < true) > 0)
    return Difficulty(
        wc, sr, tr, mt, float(max_angle), lm,
        float(min(margins)) if margins else float("inf"),
        float(min(motion_margins)) if motion_margins else float("inf"),
        transitions,
    )


def mode_satisfied(mode, d, direction_deg, tie_um):
    return {
        "wrong_closer": d.wrong_closer_count > 0,
        "motion_trap": d.motion_trap_count > 0,
        "rank_hard": d.source_rank_gt1_count > 0 or d.target_rank_gt1_count > 0,
        "direction_change": d.direction_change_max_deg >= direction_deg,
        "large_motion": d.large_motion_count > 0,
        "near_tie": d.minimum_wrong_minus_true_um <= tie_um,
        "multi_break": d.transition_count > 0,
    }[mode]


def mode_score(mode, d):
    if mode == "wrong_closer": return (d.minimum_wrong_minus_true_um, -d.wrong_closer_count)
    if mode == "motion_trap": return (d.minimum_motion_wrong_minus_true_um, -d.motion_trap_count)
    if mode == "rank_hard": return (-(d.source_rank_gt1_count + d.target_rank_gt1_count), d.minimum_wrong_minus_true_um)
    if mode == "direction_change": return (-d.direction_change_max_deg,)
    if mode == "large_motion": return (-d.large_motion_count, d.minimum_wrong_minus_true_um)
    if mode == "near_tie": return (abs(d.minimum_wrong_minus_true_um),)
    return (d.minimum_wrong_minus_true_um,)


class AdversarialSampler:
    def __init__(self, *, pool, store, global_motion, displacement_q, inv08, tracks_per_scene, scene_radius_um, proposals, direction_degrees, near_tie_margin_um, seed):
        self.pool = pool; self.store = store; self.global_motion = global_motion
        self.displacement_q = displacement_q; self.inv08 = inv08
        self.tracks_per_scene = int(tracks_per_scene); self.scene_radius_um = float(scene_radius_um)
        self.proposals = int(proposals); self.direction_degrees = float(direction_degrees)
        self.near_tie_margin_um = float(near_tie_margin_um)
        self.rng = np.random.default_rng(seed); self.next_example_id = 0; self.fallback_count = 0

    def _proposal(self, mode):
        pattern = choose_pattern(self.rng); keys = self.pool.keys_by_pattern[pattern.name]
        if not keys: return None
        placement = keys[int(self.rng.integers(0, len(keys)))]
        available = self.pool.by_key[placement]
        ref = pattern.ranges(placement.start_frame)[1][0]
        anchor = int(available[int(self.rng.integers(0, len(available)))])
        axyz = self.store[(placement.sequence, anchor)].xyz_at(ref)
        peers = []
        for tid in available:
            if int(tid) == anchor: continue
            d = float(np.linalg.norm(self.store[(placement.sequence, int(tid))].xyz_at(ref) - axyz))
            if d <= self.scene_radius_um: peers.append((d, int(tid)))
        if len(peers) < self.tracks_per_scene - 1: return None
        peers.sort()
        if mode == "multi_break" and len(peers) > self.tracks_per_scene:
            n = min(len(peers), self.tracks_per_scene * 3)
            pool_ids = np.asarray([tid for _, tid in peers[:n]], dtype=np.int64)
            chosen = [int(v) for v in self.rng.choice(pool_ids, self.tracks_per_scene - 1, replace=False).tolist()]
        else:
            chosen = [tid for _, tid in peers[: self.tracks_per_scene - 1]]
        ids = tuple([anchor, *chosen])
        diff = score_group(sequence=placement.sequence, pattern=pattern, start_frame=placement.start_frame, track_ids=ids, store=self.store, global_motion=self.global_motion, displacement_q=self.displacement_q, inv08=self.inv08)
        return placement, ids, diff

    def sample(self):
        mode = weighted_name(self.rng, HARD_MODES)
        strict = []; proposals = []
        for _ in range(max(self.proposals * 5, 32)):
            item = self._proposal(mode)
            if item is None: continue
            proposals.append(item)
            if mode_satisfied(mode, item[2], self.direction_degrees, self.near_tie_margin_um):
                strict.append(item)
                if len(strict) >= self.proposals: break
        if strict:
            placement, ids, diff = min(strict, key=lambda x: mode_score(mode, x[2])); satisfied = True
        elif proposals:
            placement, ids, diff = min(proposals, key=lambda x: mode_score(mode, x[2])); satisfied = False; self.fallback_count += 1
        else:
            raise RuntimeError(f"Could not synthesize {self.pool.split_name} scene")
        pattern = PATTERN_LOOKUP[placement.pattern_name]
        stage_orders = []
        for _ in pattern.segment_lengths:
            values = list(ids); self.rng.shuffle(values); stage_orders.append(tuple(values))
        spec = HardSceneSpec(self.next_example_id, self.pool.split_name, placement.sequence, placement.pattern_name, placement.start_frame, ids, tuple(stage_orders), mode, satisfied, diff)
        self.next_example_id += 1
        return spec


@dataclass
class RawHardScene:
    spec: HardSceneSpec
    tracklet_rows: list[pd.DataFrame]
    raw_structured: list[np.ndarray]
    structured_validity: list[np.ndarray]
    reliability_raw: list[np.ndarray]
    pair_frame: pd.DataFrame
    edge_index: np.ndarray
    edge_target: np.ndarray
    gap_frames: np.ndarray
    expected_global: np.ndarray
    expected_global_relative: np.ndarray
    prediction_valid: np.ndarray


def log_volume_error(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0: return float("nan")
    return float(abs(math.log(float(b) / float(a))))


def build_raw_scene(spec, *, store, global_motion, trainer, inv08, global_median_volume):
    pattern = PATTERN_LOOKUP[spec.pattern_name]; ranges = pattern.ranges(spec.start_frame)
    sequence = spec.sequence; motion = global_motion[sequence]
    rows_list = []; local_index = {}
    for stage, order in enumerate(spec.stage_orders):
        first, last = ranges[stage]
        for tid in order:
            local_index[(stage, int(tid))] = len(rows_list)
            rows_list.append(store[(sequence, int(tid))].rows(first, last, role="middle"))
    raw_structured = []; validity = []; reliability = []
    for rows in rows_list:
        raw, valid, diag = trainer.build_raw_structured(rows, global_motion=motion)
        raw_structured.append(raw); validity.append(valid)
        reliability.append(trainer.build_reliability_raw(rows, diagnostics=diag, crop_valid_fraction=np.ones(len(rows), np.float32), global_median_volume=float(global_median_volume)))
    edge_rows = []; edge_pairs = []; targets = []; gaps = []; eg = []; egr = []; pvalid = []
    for stage in range(len(ranges) - 1):
        s_first, s_last = ranges[stage]; t_first, _ = ranges[stage + 1]
        gap = int(t_first - s_last); radius = min(25.0, 12.0 + 4.0 * max(gap - 1, 0))
        stage_edge_ids = []
        for sid in spec.stage_orders[stage]:
            slocal = local_index[(stage, int(sid))]; srows = rows_list[slocal]; send = srows.iloc[-1]
            sxyz = send[["z_um", "y_um", "x_um"]].to_numpy(np.float32); svol = float(send.volume)
            delta, gv = inv08.cumulative_global_shift(motion, s_last, t_first)
            expg = sxyz + delta if gv else sxyz.copy()
            rv, rvv = inv08.source_relative_velocity(srows, motion); grv = bool(gv and rvv)
            expgr = expg + rv * float(gap) if grv else expg.copy()
            for tid in spec.stage_orders[stage + 1]:
                tlocal = local_index[(stage + 1, int(tid))]; trows = rows_list[tlocal]; tend = trows.iloc[0]
                txyz = tend[["z_um", "y_um", "x_um"]].to_numpy(np.float32); tvol = float(tend.volume)
                direct = float(np.linalg.norm(txyz - sxyz)); positive = int(sid) == int(tid)
                if not positive and direct > radius: continue
                ferr = float(np.linalg.norm(txyz - expgr)) if grv else float("nan")
                qerr = ferr if np.isfinite(ferr) else direct
                quality = float(math.exp(-qerr / max(radius, 1e-6)))
                stage_edge_ids.append(len(edge_rows)); edge_pairs.append((slocal, tlocal)); targets.append(float(positive)); gaps.append(gap)
                eg.append(expg.astype(np.float32)); egr.append(expgr.astype(np.float32)); pvalid.append((bool(gv), bool(grv), False, False))
                edge_rows.append({
                    "gap_frames": gap, "direct_endpoint_distance_um": direct, "hard_search_radius_um": radius,
                    "forward_error_um": ferr, "forward_history_count": len(srows), "forward_used_global_motion": bool(gv),
                    "forward_used_relative_velocity": bool(grv), "forward_uncertainty_um": np.nan,
                    "volume_log_error": log_volume_error(svol, tvol), "target_real_observation_count": len(trows),
                    "candidate_quality_score": quality,
                    "effective_pair_volume": float(math.sqrt(max(svol,0)*max(tvol,0))) if np.isfinite(svol) and np.isfinite(tvol) and svol >= 0 and tvol >= 0 else np.nan,
                    "small_cell_history_exception": False,
                })
        if int(np.sum(np.asarray(targets)[stage_edge_ids] > 0.5)) != len(spec.track_ids):
            raise RuntimeError(f"Lost GT continuation at stage {stage}")
    pair = pd.DataFrame(edge_rows); edge_index = np.asarray(edge_pairs, np.int64); target = np.asarray(targets, np.float32)
    pair["source_candidate_count"] = 0; pair["target_predecessor_count"] = 0; pair["source_rank"] = 0; pair["target_rank"] = 0
    pair["source_score_margin"] = np.nan; pair["target_score_margin"] = np.nan; pair["mutual_best"] = False
    for local in np.unique(edge_index[:,0]):
        idx = np.flatnonzero(edge_index[:,0] == local); q = pair.iloc[idx]["candidate_quality_score"].to_numpy(float); order = np.argsort(-q, kind="mergesort")
        ranks = np.empty(len(idx), np.int64); ranks[order] = np.arange(1, len(idx)+1); margin = float(q[order[0]] - q[order[1]]) if len(order)>1 else np.nan
        pair.loc[pair.index[idx], "source_candidate_count"] = len(idx); pair.loc[pair.index[idx], "source_rank"] = ranks; pair.loc[pair.index[idx], "source_score_margin"] = margin
    for local in np.unique(edge_index[:,1]):
        idx = np.flatnonzero(edge_index[:,1] == local); q = pair.iloc[idx]["candidate_quality_score"].to_numpy(float); order = np.argsort(-q, kind="mergesort")
        ranks = np.empty(len(idx), np.int64); ranks[order] = np.arange(1, len(idx)+1); margin = float(q[order[0]] - q[order[1]]) if len(order)>1 else np.nan
        pair.loc[pair.index[idx], "target_predecessor_count"] = len(idx); pair.loc[pair.index[idx], "target_rank"] = ranks; pair.loc[pair.index[idx], "target_score_margin"] = margin
    pair["mutual_best"] = (pd.to_numeric(pair["source_rank"]) == 1) & (pd.to_numeric(pair["target_rank"]) == 1)
    return RawHardScene(spec, rows_list, raw_structured, validity, reliability, pair, edge_index, target, np.asarray(gaps,np.int64), np.stack(eg).astype(np.float32), np.stack(egr).astype(np.float32), np.asarray(pvalid,bool))


def fit_stats(sampler, *, count, store, global_motion, trainer, inv08, global_median_volume):
    structured=[]; validity=[]; reliability=[]; pairs=[]; specs=[]
    print(f"[stats] synthesizing {count} adversarial TRAIN-only scenes ...", flush=True)
    for i in range(count):
        spec=sampler.sample(); raw=build_raw_scene(spec,store=store,global_motion=global_motion,trainer=trainer,inv08=inv08,global_median_volume=global_median_volume)
        specs.append(spec); structured.extend(raw.raw_structured); validity.extend(raw.structured_validity); reliability.extend(raw.reliability_raw); pairs.append(raw.pair_frame)
        if (i+1)%100==0 or i+1==count: print(f"[stats] {i+1}/{count}", flush=True)
    return trainer.FeatureStats(
        structured=trainer.fit_masked_stats(np.concatenate(structured), np.concatenate(validity)),
        reliability=trainer.fit_reliability_stats(np.stack(reliability)),
        pair=trainer.fit_pair_stats(pd.concat(pairs, ignore_index=True)),
    ), specs


def prepare_scene(spec, *, store, global_motion, trainer, inv08, stats, global_median_volume):
    raw=build_raw_scene(spec,store=store,global_motion=global_motion,trainer=trainer,inv08=inv08,global_median_volume=global_median_volume)
    n=len(raw.tracklet_rows); kmax=max(len(r) for r in raw.tracklet_rows); fdim=trainer.ReconcilerConfig().fingerprint.embedding_dim
    structured=np.zeros((n,kmax,48),np.float32); mask=np.zeros((n,kmax),bool); times=np.zeros((n,kmax),np.float32)
    start=np.zeros((n,3),np.float32); end=np.zeros((n,3),np.float32); rel=np.zeros((n,12),np.float32); fingerprints=np.zeros((n,kmax,fdim),np.float32)
    for i, rows in enumerate(raw.tracklet_rows):
        k=len(rows); structured[i,:k]=trainer.apply_masked_stats(raw.raw_structured[i],raw.structured_validity[i],stats.structured); mask[i,:k]=True
        times[i,:k]=rows["frame"].to_numpy(np.float32,copy=True); xyz=rows[["z_um","y_um","x_um"]].to_numpy(np.float32,copy=True); start[i]=xyz[0]; end[i]=xyz[-1]
        rel[i]=(raw.reliability_raw[i]-stats.reliability.mean)/stats.reliability.std
    pair,_=trainer.tensorize_stage11_pair_features(raw.pair_frame,mean=stats.pair.mean,std=stats.pair.std,device="cpu")
    e=len(raw.edge_index)
    return trainer.PreparedExample(spec.example_id,np.arange(n,dtype=np.int64),structured,mask,times,start,end,rel,None,fingerprints,raw.edge_index.copy(),raw.gap_frames.copy(),pair.numpy().astype(np.float32,copy=True),raw.expected_global.copy(),raw.expected_global_relative.copy(),np.zeros((e,3),np.float32),np.zeros((e,3),np.float32),raw.prediction_valid.copy(),raw.edge_target.copy())


def fixed_set(sampler, *, count, **kwargs):
    examples=[]; specs=[]
    for i in range(count):
        spec=sampler.sample(); examples.append(prepare_scene(spec,**kwargs)); specs.append(spec)
        if (i+1)%100==0 or i+1==count: print(f"[synthesis:{sampler.pool.split_name}] {i+1}/{count}", flush=True)
    return examples,specs


def geometry_baseline(examples):
    from scipy.optimize import linear_sum_assignment
    exact=sc=st=tc=tt=wc=0
    for ex in examples:
        ei=ex.edge_index; y=ex.edge_target>0.5; d=np.linalg.norm(ex.start_xyz_um[ei[:,1]]-ex.end_xyz_um[ei[:,0]],axis=1)
        for s in np.unique(ei[:,0]):
            idx=np.flatnonzero(ei[:,0]==s); pos=idx[y[idx]]
            if len(pos)!=1: continue
            st+=1; best=idx[np.argmin(d[idx])]; sc+=int(best==pos[0]); neg=idx[~y[idx]]; wc+=int(len(neg)>0 and np.min(d[neg])<d[pos[0]])
        for t in np.unique(ei[:,1]):
            idx=np.flatnonzero(ei[:,1]==t); pos=idx[y[idx]]
            if len(pos)!=1: continue
            tt+=1; tc+=int(idx[np.argmin(d[idx])]==pos[0])
        sources=sorted(np.unique(ei[:,0]).tolist()); targets=sorted(np.unique(ei[:,1]).tolist())
        if len(sources)==len(targets):
            sm={v:i for i,v in enumerate(sources)}; tm={v:i for i,v in enumerate(targets)}; cost=np.full((len(sources),len(targets)),1e6); truth=np.zeros_like(cost,bool)
            for dist,edge,label in zip(d,ei,y): cost[sm[int(edge[0])],tm[int(edge[1])]]=float(dist); truth[sm[int(edge[0])],tm[int(edge[1])]]=bool(label)
            rr,cc=linear_sum_assignment(cost); exact+=int(len(rr)==len(sources) and all(truth[r,c] for r,c in zip(rr,cc)))
    return {"exact_assignment":exact/max(len(examples),1),"source_top1":sc/max(st,1),"target_top1":tc/max(tt,1),"wrong_closer_source_fraction":wc/max(st,1)}


def spec_summary(specs):
    modes={}; patterns={}; strict=wc=mt=rh=0
    for s in specs:
        modes[s.requested_mode]=modes.get(s.requested_mode,0)+1; patterns[s.pattern_name]=patterns.get(s.pattern_name,0)+1; strict+=int(s.strict_mode_satisfied)
        wc+=int(s.difficulty.wrong_closer_count>0); mt+=int(s.difficulty.motion_trap_count>0); rh+=int(s.difficulty.source_rank_gt1_count>0 or s.difficulty.target_rank_gt1_count>0)
    n=max(len(specs),1)
    return {"count":len(specs),"requested_modes":modes,"patterns":patterns,"strict_requested_mode_fraction":strict/n,"wrong_closer_scene_fraction":wc/n,"motion_trap_scene_fraction":mt/n,"rank_hard_scene_fraction":rh/n}


def evaluate_strata(trainer,model,examples,specs,*,device,amp_mode):
    groups={}
    for ex,s in zip(examples,specs):
        groups.setdefault(f"mode:{s.requested_mode}",[]).append(ex); groups.setdefault(f"pattern:{s.pattern_name}",[]).append(ex)
    return {name:trainer.evaluate(model,subset,device=device,appearance_mode="fingerprints",amp_mode=amp_mode).as_dict() for name,subset in groups.items()}


def disable_untrained_branches(model):
    prefixes=("tracklets.fingerprint.","tracklets.appearance_stream.","division_prior_head.","appearance_head.","termination_head.","division_head.")
    frozen=[]
    with torch.no_grad():
        for name,p in model.named_parameters():
            if any(name.startswith(x) for x in prefixes):
                p.requires_grad_(False); frozen.append(name)
                if name.startswith("tracklets.appearance_stream."): p.zero_()
    return frozen


def trainable_parameter_count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path,*,model,optimizer,step,stats,global_median_volume,args,train_metrics,validation_metrics,validation_strata):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"schema_version":1,"investigation":SCRIPT_NAME,"step":int(step),"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"feature_stats":stats.to_json(),"train_global_median_volume":float(global_median_volume),"args":vars(args),"train_metrics":train_metrics.as_dict(),"validation_metrics":validation_metrics.as_dict(),"validation_strata":validation_strata,"training_scope":{"continuation":True,"multi_break_chains":True,"adversarial_hard_mining":True,"appearance":False,"division":False,"appearance_stream_zeroed_in_state_dict":True}},path)


def parse_spacing(text):
    values=tuple(float(v.strip()) for v in str(text).split(","))
    if len(values)!=3 or not all(v>0 for v in values): raise ValueError("--spacing must be positive Z,Y,X")
    return values


def build_parser():
    p=argparse.ArgumentParser(description="Adversarial multi-break continuation training from clean Fluo-N3DH-CE TRA GT")
    p.add_argument("--dataset-root",default=None); p.add_argument("--base-cache",default=None); p.add_argument("--output",default=None); p.add_argument("--spacing",default="1.0,0.09,0.09")
    p.add_argument("--validation-fraction",type=float,default=DEFAULT_VALIDATION_FRACTION); p.add_argument("--tracks-per-scene",type=int,default=DEFAULT_TRACKS_PER_SCENE)
    p.add_argument("--scene-radius-um",type=float,default=DEFAULT_SCENE_RADIUS_UM); p.add_argument("--hard-proposals",type=int,default=DEFAULT_PROPOSALS)
    p.add_argument("--direction-threshold-deg",type=float,default=DEFAULT_DIRECTION_DEGREES); p.add_argument("--near-tie-margin-um",type=float,default=DEFAULT_NEAR_TIE_MARGIN_UM)
    p.add_argument("--stats-scenes",type=int,default=DEFAULT_STATS_SCENES); p.add_argument("--validation-scenes",type=int,default=DEFAULT_VALIDATION_SCENES); p.add_argument("--train-eval-scenes",type=int,default=DEFAULT_TRAIN_EVAL_SCENES)
    p.add_argument("--steps",type=int,default=DEFAULT_STEPS); p.add_argument("--batch-size",type=int,default=DEFAULT_BATCH_SIZE); p.add_argument("--lr",type=float,default=DEFAULT_LR); p.add_argument("--weight-decay",type=float,default=DEFAULT_WEIGHT_DECAY); p.add_argument("--grad-clip",type=float,default=DEFAULT_GRAD_CLIP)
    p.add_argument("--log-every",type=int,default=DEFAULT_LOG_EVERY); p.add_argument("--eval-every",type=int,default=DEFAULT_EVAL_EVERY); p.add_argument("--device",default="auto"); p.add_argument("--amp",choices=("auto","off","fp16","bf16"),default="auto"); p.add_argument("--seed",type=int,default=DEFAULT_SEED)
    p.add_argument("--prepare-only",action="store_true"); p.add_argument("--rebuild-base",action="store_true"); p.add_argument("--overwrite",action="store_true")
    return p


def main():
    args=build_parser().parse_args(); seed=int(args.seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    inv08=load_inv08(); trainer,_=inv08.load_trainer_modules()
    dataset_root=resolve(args.dataset_root) if args.dataset_root else default_dataset_root(); base_cache=resolve(args.base_cache) if args.base_cache else default_base_cache(); output=resolve(args.output) if args.output else default_output(); spacing=parse_spacing(args.spacing)
    if output.exists() and args.overwrite: shutil.rmtree(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Output exists: {output}\nPass --overwrite")
    output.mkdir(parents=True,exist_ok=True)
    print("="*126); print("TRACK RECONCILER — INVESTIGATION 09: CE ADVERSARIAL MULTI-BREAK TRAINING"); print("="*126)
    print(f"dataset root             : {dataset_root}"); print(f"base cache               : {base_cache}"); print(f"output                   : {output}"); print(f"simultaneous tracks      : {args.tracks_per_scene}")
    print("fragmentation            : multi-cut chains + singleton middle tracklets"); print("hard mining              : wrong-closer / motion-trap / rank / direction / displacement / near-tie"); print("appearance / division    : disabled / disabled"); print("="*126,flush=True)
    started=time.perf_counter(); observations,tracks,_=inv08.build_base_cache(dataset_root,base_cache,spacing=spacing,rebuild=bool(args.rebuild_base)); store=inv08.build_track_store(observations)
    split=inv08.make_identity_split(tracks,store,validation_fraction=float(args.validation_fraction),seed=seed+17); atomic_csv(output/"identity_split.csv",split)
    train_ids={(str(r.sequence).zfill(2),int(r.track_id)) for r in split.itertuples(index=False) if str(r.split)=="train"}; val_ids={(str(r.sequence).zfill(2),int(r.track_id)) for r in split.itertuples(index=False) if str(r.split)=="validation"}
    if train_ids & val_ids: raise RuntimeError("Identity leakage")
    gm_table,gm=inv08.estimate_global_motion(store,split); atomic_csv(output/"global_motion_train_only.csv",gm_table)
    vols=[s.volume[np.isfinite(s.volume)&(s.volume>0)] for k,s in store.items() if k in train_ids and np.any(np.isfinite(s.volume)&(s.volume>0))]; median=float(np.median(np.concatenate(vols))) if vols else 1.0
    dq=displacement_quantiles(store,train_ids); atomic_json(output/"training_motion_quantiles.json",dq); print("[motion p90] "+" | ".join(f"gap{g}={v['p90']:.3f}um" for g,v in dq.items()),flush=True)
    print("[placements] indexing multi-break windows ...",flush=True); train_pool=PatternPool(store=store,split_table=split,split_name="train",tracks_per_scene=args.tracks_per_scene); val_pool=PatternPool(store=store,split_table=split,split_name="validation",tracks_per_scene=args.tracks_per_scene)
    print(f"[placements] TRAIN {train_pool.summary()}",flush=True); print(f"[placements] VAL   {val_pool.summary()}",flush=True)
    def mk(pool,off): return AdversarialSampler(pool=pool,store=store,global_motion=gm,displacement_q=dq,inv08=inv08,tracks_per_scene=args.tracks_per_scene,scene_radius_um=args.scene_radius_um,proposals=args.hard_proposals,direction_degrees=args.direction_threshold_deg,near_tie_margin_um=args.near_tie_margin_um,seed=seed+off)
    stats_sampler=mk(train_pool,101); train_sampler=mk(train_pool,202); train_eval_sampler=mk(train_pool,303); val_sampler=mk(val_pool,404)
    stats,stats_specs=fit_stats(stats_sampler,count=args.stats_scenes,store=store,global_motion=gm,trainer=trainer,inv08=inv08,global_median_volume=median); atomic_json(output/"feature_stats.json",{**stats.to_json(),"fit_split":"train_identities_only","appearance_features":"disabled","train_global_median_volume":median}); atomic_csv(output/"stats_scene_manifest.csv",pd.DataFrame([s.to_row() for s in stats_specs]))
    kwargs=dict(store=store,global_motion=gm,trainer=trainer,inv08=inv08,stats=stats,global_median_volume=median)
    print("[validation] synthesizing fixed hard validation set ...",flush=True); val_examples,val_specs=fixed_set(val_sampler,count=args.validation_scenes,**kwargs); atomic_csv(output/"validation_scene_manifest.csv",pd.DataFrame([s.to_row() for s in val_specs]))
    print("[train-eval] synthesizing fixed hard train diagnostic set ...",flush=True); train_eval,train_eval_specs=fixed_set(train_eval_sampler,count=args.train_eval_scenes,**kwargs); atomic_csv(output/"train_eval_scene_manifest.csv",pd.DataFrame([s.to_row() for s in train_eval_specs]))
    synth=spec_summary(val_specs); baseline=geometry_baseline(val_examples)
    print("\n"+"="*126); print("ADVERSARIAL CE DATASET READY"); print("="*126); print(f"identity split           : train={len(train_ids):,} validation={len(val_ids):,} overlap=0"); print(f"validation strict modes  : {100*synth['strict_requested_mode_fraction']:.1f}%"); print(f"VAL wrong-closer scenes  : {100*synth['wrong_closer_scene_fraction']:.1f}%"); print(f"VAL motion-trap scenes   : {100*synth['motion_trap_scene_fraction']:.1f}%"); print(f"VAL rank-hard scenes     : {100*synth['rank_hard_scene_fraction']:.1f}%"); print(f"distance-only VAL        : exact={100*baseline['exact_assignment']:.2f}% src={100*baseline['source_top1']:.2f}% tgt={100*baseline['target_top1']:.2f}% wrong-closer={100*baseline['wrong_closer_source_fraction']:.2f}%"); print(f"pattern mix              : {synth['patterns']}"); print(f"requested hard modes     : {synth['requested_modes']}"); print("="*126,flush=True)
    prep={"schema_version":1,"investigation":SCRIPT_NAME,"dataset_root":root_relative(dataset_root),"base_observations":len(observations),"declared_tracks":len(tracks),"identity_split":{"train":len(train_ids),"validation":len(val_ids),"overlap":0},"patterns":[asdict(p) for p in PATTERNS],"hard_modes":dict(HARD_MODES),"training_motion_quantiles":dq,"train_pool":train_pool.summary(),"validation_pool":val_pool.summary(),"validation_synthesis":synth,"geometry_baseline":baseline,"training_scope":{"continuation_only":True,"appearance_disabled":True,"division_disabled":True,"multi_break":True,"simultaneous_nearby_breaks":True,"online_synthesis":True}}; atomic_json(output/"preparation_summary.json",prep)
    if baseline["exact_assignment"]>0.95: print("[warning] adversarial validation is still >95% solvable by distance-only assignment; inspect hard-mode fractions.",flush=True)
    if args.prepare_only: return 0
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device); amp=trainer.resolved_amp_mode(device,args.amp)
    model=trainer.TrackletReconciliationNetwork(trainer.ReconcilerConfig()).to(device); frozen=disable_untrained_branches(model); trainable=[p for p in model.parameters() if p.requires_grad]; optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay); scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" and amp=="fp16" else None
    if device.type=="cuda": torch.set_float32_matmul_precision("high"); torch.cuda.reset_peak_memory_stats(device)
    print("\n"+"="*126); print("ADVERSARIAL MULTI-BREAK CONTINUATION TRAINING"); print("="*126); print(f"parameters               : {sum(p.numel() for p in model.parameters()):,}"); print(f"trainable                : {trainable_parameter_count(model):,}"); print(f"frozen tensors           : {len(frozen)}"); print(f"device / amp             : {device} / {amp}"); print(f"steps / batch            : {args.steps:,} / {args.batch_size}"); print("="*126,flush=True)
    init_train=trainer.evaluate(model,train_eval,device=device,appearance_mode="fingerprints",amp_mode=amp); init_val=trainer.evaluate(model,val_examples,device=device,appearance_mode="fingerprints",amp_mode=amp); print(f"[eval step=00000] TRAIN exact={100*init_train.exact_assignment:6.2f}% src={100*init_train.source_top1:6.2f}% tgt={100*init_train.target_top1:6.2f}% loss={init_train.loss:.5f} || VAL exact={100*init_val.exact_assignment:6.2f}% src={100*init_val.source_top1:6.2f}% tgt={100*init_val.target_top1:6.2f}% loss={init_val.loss:.5f}",flush=True)
    ckpt=output/"checkpoints"; ckpt.mkdir(parents=True,exist_ok=True); history=[]; best_exact=-1.; best_loss=float("inf"); running=0.; rc=0; t0=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train(); batch_examples=[]
        for _ in range(args.batch_size): batch_examples.append(prepare_scene(train_sampler.sample(),**kwargs))
        batch=trainer.collate(batch_examples,device=device,appearance_mode="fingerprints"); optimizer.zero_grad(set_to_none=True)
        with trainer.autocast_context(device,amp): pred=model(batch.reconciliation); loss=trainer.focal_binary_probability_loss(pred.parental_probabilities,batch.edge_target,batch.reconciliation.edges.edge_mask,gamma=2.0)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); optimizer.step()
        running+=float(loss.detach().float().cpu()); rc+=1
        if step%args.log_every==0:
            mem=f" peakVRAM={torch.cuda.max_memory_allocated(device)/(1024**3):.2f}GiB" if device.type=="cuda" else ""; print(f"[train step={step:05d}] loss={running/max(rc,1):.5f} fresh_scenes={step*args.batch_size:,} fallback_hard={train_sampler.fallback_count} time={time.perf_counter()-t0:.1f}s{mem}",flush=True); running=0.; rc=0
        if step%args.eval_every==0 or step==args.steps:
            tm=trainer.evaluate(model,train_eval,device=device,appearance_mode="fingerprints",amp_mode=amp); vm=trainer.evaluate(model,val_examples,device=device,appearance_mode="fingerprints",amp_mode=amp); strata=evaluate_strata(trainer,model,val_examples,val_specs,device=device,amp_mode=amp)
            print(f"[eval step={step:05d}] TRAIN exact={100*tm.exact_assignment:6.2f}% src={100*tm.source_top1:6.2f}% tgt={100*tm.target_top1:6.2f}% loss={tm.loss:.5f} || VAL exact={100*vm.exact_assignment:6.2f}% src={100*vm.source_top1:6.2f}% tgt={100*vm.target_top1:6.2f}% loss={vm.loss:.5f} p+={vm.positive_probability_mean:.4f} p-max-={vm.negative_probability_max:.4f}",flush=True)
            pieces=[]
            for name in ("mode:wrong_closer","mode:motion_trap","mode:rank_hard"):
                if name in strata: pieces.append(f"{name.split(':')[1]}={100*strata[name]['exact_assignment']:.1f}%")
            if pieces: print("[val hard strata] "+" | ".join(pieces),flush=True)
            history.append({"step":step,"fresh_training_scenes_seen":step*args.batch_size,"train_fallback_hard_count":train_sampler.fallback_count,"elapsed_seconds":time.perf_counter()-t0,"train":tm.as_dict(),"validation":vm.as_dict(),"validation_strata":strata}); atomic_json(output/"history.json",history)
            better=vm.exact_assignment>best_exact or (vm.exact_assignment==best_exact and vm.loss<best_loss)
            if better: best_exact=float(vm.exact_assignment); best_loss=float(vm.loss); save_checkpoint(ckpt/"best_validation.pt",model=model,optimizer=optimizer,step=step,stats=stats,global_median_volume=median,args=args,train_metrics=tm,validation_metrics=vm,validation_strata=strata)
            save_checkpoint(ckpt/"latest.pt",model=model,optimizer=optimizer,step=step,stats=stats,global_median_volume=median,args=args,train_metrics=tm,validation_metrics=vm,validation_strata=strata)
    final_train=trainer.evaluate(model,train_eval,device=device,appearance_mode="fingerprints",amp_mode=amp); final_val=trainer.evaluate(model,val_examples,device=device,appearance_mode="fingerprints",amp_mode=amp); final_strata=evaluate_strata(trainer,model,val_examples,val_specs,device=device,amp_mode=amp)
    summary={**prep,"training":{"steps":args.steps,"batch_size":args.batch_size,"fresh_training_scenes_seen":args.steps*args.batch_size,"train_fallback_hard_count":train_sampler.fallback_count,"lr":args.lr,"device":str(device),"amp":amp,"total_parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":trainable_parameter_count(model)},"initial":{"train":init_train.as_dict(),"validation":init_val.as_dict()},"final":{"train":final_train.as_dict(),"validation":final_val.as_dict(),"validation_strata":final_strata},"best_validation_checkpoint":"checkpoints/best_validation.pt","latest_checkpoint":"checkpoints/latest.pt","elapsed_seconds":time.perf_counter()-started}; atomic_json(output/"summary.json",summary)
    print("\n"+"="*126); print("INVESTIGATION 09 COMPLETE"); print("="*126); print(f"FINAL TRAIN              : exact={100*final_train.exact_assignment:.2f}% src={100*final_train.source_top1:.2f}% tgt={100*final_train.target_top1:.2f}% loss={final_train.loss:.5f}"); print(f"FINAL VALIDATION         : exact={100*final_val.exact_assignment:.2f}% src={100*final_val.source_top1:.2f}% tgt={100*final_val.target_top1:.2f}% loss={final_val.loss:.5f} p+={final_val.positive_probability_mean:.4f} p-max-={final_val.negative_probability_max:.4f}"); print(f"distance-only VAL        : exact={100*baseline['exact_assignment']:.2f}% src={100*baseline['source_top1']:.2f}% tgt={100*baseline['target_top1']:.2f}%"); print(f"hard fallback count      : {train_sampler.fallback_count}"); print(f"best checkpoint          : {ckpt/'best_validation.pt'}"); print(f"output                   : {output}"); print("="*126,flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
