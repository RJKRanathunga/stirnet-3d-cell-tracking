from __future__ import annotations

r"""
Investigation 05 — held-out-identity generalization for the learned continuation
reconciler.

Investigations 03 and 04 answered two memorization questions:

    03: can the complete network memorize easy simultaneous breaks?       yes
    04: can it memorize variable-gap, geometrically ambiguous A->B->C?    yes

Investigation 05 asks the next question:

    Can the learned reconciler recover hard continuations for ORIGINAL
    TRACKASTRA IDENTITIES THAT NEVER APPEAR IN TRAINING?

The split is therefore NOT an example-level random split.  Validation scenes
are selected first, all original Trackastra track IDs used by those scenes are
reserved as held-out identities, and every candidate training scene containing
any held-out identity is discarded.

This avoids the most important leakage mode for the 3-D fingerprint branch:
the same physical cell track cannot appear at one temporal cut in training and
at another cut in validation.

The script also prevents feature-normalization leakage:

    structured statistics  -- fit on train only
    reliability statistics -- fit on train only
    pair-feature statistics-- fit on train only

The fixed training statistics are then applied to validation.

Dataset coverage
----------------
Selection is pattern-balanced before hardness filling, so the experiment tries
to retain all Investigation-04 regimes instead of selecting only the globally
hardest pair_g4 / chain_gap2 cases:

    pair gap 1 / 2 / 3 / 4
    short source / short target
    A -> B -> C chains
    one-frame and two-frame middle tracklets
    mixed gap chains
    wrong candidate closer than the true target
    positive geometric rank > 1

Outputs
-------
runs/track_reconciler/investigations/
  05_hard_continuation_generalization/<sample_id>/
    identity_split.csv
    train_dataset/
    validation_dataset/
    training/
      cache_train/
      cache_validation/
      checkpoints/
        best_validation.pt
        latest.pt
      feature_stats.json
      history.json
      summary.json
    summary.json

Default run
-----------
From repository root:

    python .\investigations\track_reconciler\05_hard_continuation_generalization.py ^
        --overwrite

The defaults intentionally remain modest for the first held-out experiment on
the 20-frame BioHub movie:

    64 hard training scenes
    24 hard held-out-identity validation scenes
    4 nearby identities per scene
    600 optimizer steps

A successful result is NOT defined by train=100%.  The important number is the
held-out validation exact assignment / source-top1 / target-top1 trajectory.
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
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment


SCRIPT_NAME = "05_hard_continuation_generalization"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_TRAIN_EXAMPLES = 64
DEFAULT_VALIDATION_EXAMPLES = 24
DEFAULT_TRACKS_PER_EXAMPLE = 4
DEFAULT_GUARD_FRAMES = 1
DEFAULT_GROUP_RADIUS_UM = 28.0
DEFAULT_MAX_CELL_MATCH_UM = 2.5
DEFAULT_MAX_STEP_UM = 12.0
DEFAULT_MAX_VOLUME_RATIO = 2.0
DEFAULT_MIN_BOUNDARY_DISTANCE_UM = 5.0
DEFAULT_MIN_WRONG_SOURCE = 1
DEFAULT_MIN_WRONG_TARGET = 1
DEFAULT_TRAIN_TRACK_REUSE = 6
DEFAULT_VALIDATION_TRACK_REUSE = 6
DEFAULT_PLACEMENT_REUSE = 4
DEFAULT_STEPS = 600
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 3.0e-4
DEFAULT_EVAL_EVERY = 25
DEFAULT_LOG_EVERY = 10
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_SEED = 20260827
DEFAULT_CROP_SHAPE = "13,41,41"

STRICT_POSITIVE_PROBABILITY = 0.90
STRICT_NEGATIVE_PROBABILITY = 0.10
STRICT_VALIDATION_STREAK = 3


# =============================================================================
# Repository / dynamic reuse of Investigations 01, 03 and 04
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


def load_inv04():
    return load_script(
        ROOT / "investigations" / "track_reconciler" / "04_hard_continuation_overfit.py",
        "_track_reconciler_inv04_for_05",
    )


def load_inv03():
    return load_script(
        ROOT / "investigations" / "track_reconciler" / "03_continuation_overfit.py",
        "_track_reconciler_inv03_for_05",
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
# Pattern-balanced hard-scene selection
# =============================================================================


def hard_key(inv04, group, tie: float) -> tuple[Any, ...]:
    return inv04.hardness_key(group, tie)


def balanced_select(
    inv04,
    candidates: list[Any],
    *,
    requested: int,
    max_track_reuse: int,
    max_placement_reuse: int,
    seed: int,
) -> list[Any]:
    """Round-robin patterns first, then globally hardest fill."""
    if requested <= 0 or not candidates:
        return []

    rng = np.random.default_rng(seed)
    tie = {id(group): float(rng.random()) for group in candidates}
    ordered = sorted(candidates, key=lambda g: hard_key(inv04, g, tie[id(g)]))

    pattern_order = [pattern.name for pattern in inv04.PATTERNS]
    by_pattern: dict[str, list[Any]] = {
        name: [g for g in ordered if g.pattern_name == name]
        for name in pattern_order
    }

    chosen: list[Any] = []
    chosen_keys: set[tuple[str, int, tuple[int, ...]]] = set()
    track_use: Counter[int] = Counter()
    placement_use: Counter[tuple[str, int]] = Counter()
    cursor: dict[str, int] = {name: 0 for name in pattern_order}

    def admissible(group) -> bool:
        key = (group.pattern_name, int(group.first_start), tuple(group.track_ids))
        if key in chosen_keys:
            return False
        if placement_use[(group.pattern_name, int(group.first_start))] >= max_placement_reuse:
            return False
        return not any(track_use[int(track_id)] >= max_track_reuse for track_id in group.track_ids)

    def add(group) -> None:
        chosen.append(group)
        chosen_keys.add((group.pattern_name, int(group.first_start), tuple(group.track_ids)))
        track_use.update(int(track_id) for track_id in group.track_ids)
        placement_use[(group.pattern_name, int(group.first_start))] += 1

    # Balanced round-robin. Continue cycling as long as at least one pattern
    # contributes a scene in the pass.
    while len(chosen) < requested:
        progress = False
        for pattern_name in pattern_order:
            pool = by_pattern[pattern_name]
            while cursor[pattern_name] < len(pool):
                candidate = pool[cursor[pattern_name]]
                cursor[pattern_name] += 1
                if admissible(candidate):
                    add(candidate)
                    progress = True
                    break
            if len(chosen) >= requested:
                break
        if not progress:
            break

    # Fill deficits by global hardness while preserving reuse constraints.
    for group in ordered:
        if len(chosen) >= requested:
            break
        if admissible(group):
            add(group)

    return chosen


def split_candidates_by_reserved_ids(
    candidates: list[Any],
    validation_ids: set[int],
) -> tuple[list[Any], list[Any], list[Any]]:
    train: list[Any] = []
    validation: list[Any] = []
    mixed: list[Any] = []
    for group in candidates:
        ids = set(int(value) for value in group.track_ids)
        if ids.issubset(validation_ids):
            validation.append(group)
        elif ids.isdisjoint(validation_ids):
            train.append(group)
        else:
            mixed.append(group)
    return train, validation, mixed


# =============================================================================
# Dataset writing
# =============================================================================


def source_references(source) -> dict[str, str]:
    return {
        "investigation_36_root": root_relative(source.root),
        "raw": root_relative(source.raw),
        "preprocessed": root_relative(source.preprocessed),
        "final_instances": root_relative(source.final_instances),
        "cells": root_relative(source.cells),
        "tracks": root_relative(source.tracks),
        "napari_graph": root_relative(source.napari_graph),
    }


def dataset_stats(examples: pd.DataFrame, tracklets: pd.DataFrame, observations: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    positive = edges[edges["continuation_target"].astype(int) == 1]
    original_ids = sorted(tracklets["original_track_id"].astype(int).unique().tolist())
    return {
        "examples": int(len(examples)),
        "original_track_ids": int(len(original_ids)),
        "synthetic_tracklets": int(len(tracklets)),
        "middle_tracklets": int((tracklets["role"] == "middle").sum()),
        "observation_rows": int(len(observations)),
        "candidate_edges": int(len(edges)),
        "positive_edges": int(len(positive)),
        "negative_edges": int((edges["continuation_target"].astype(int) == 0).sum()),
        "chain_examples": int((examples["pattern_kind"] == "chain").sum()),
        "pair_examples": int((examples["pattern_kind"] == "pair").sum()),
        "wrong_closer_source_decisions": int(examples["wrong_closer_source_count"].sum()),
        "positive_source_rank_gt1": int((positive["source_distance_rank"].astype(int) > 1).sum()),
        "positive_target_rank_gt1": int((positive["target_distance_rank"].astype(int) > 1).sum()),
        "minimum_wrong_minus_true_um": float(examples["minimum_wrong_minus_true_um"].min()),
        "median_example_wrong_minus_true_um": float(examples["median_wrong_minus_true_um"].median()),
        "positive_gap_distribution": {
            str(int(k)): int(v)
            for k, v in positive["gap_frames"].astype(int).value_counts().sort_index().items()
        },
        "pattern_distribution": {
            str(k): int(v)
            for k, v in examples["pattern_name"].value_counts().sort_index().items()
        },
    }


def write_dataset(
    path: Path,
    *,
    split_name: str,
    sample_id: str,
    source,
    movie_shape: tuple[int, ...],
    spacing: tuple[float, float, float],
    examples: pd.DataFrame,
    tracklets: pd.DataFrame,
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    global_motion: pd.DataFrame,
    reserved_validation_ids: set[int],
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    atomic_csv(path / "examples.csv", examples)
    atomic_csv(path / "tracklets.csv", tracklets)
    atomic_csv(path / "observations.csv", observations)
    atomic_csv(path / "edges.csv", edges)
    atomic_csv(path / "global_motion.csv", global_motion)

    stats = dataset_stats(examples, tracklets, observations, edges)
    manifest = {
        "schema_version": 3,
        "investigation": SCRIPT_NAME,
        "split": split_name,
        "sample_id": sample_id,
        "source": source_references(source),
        "movie_shape_tzyx": list(movie_shape),
        "spacing_zyx_um": list(spacing),
        "parameters": {
            "candidate_radius_um": 25.0,
            "candidate_radius_policy": "min(25, 12 + 4*(gap_frames-1)) um",
            "identity_disjoint_split": True,
        },
        "patterns": [
            {
                "name": pattern.name,
                "kind": pattern.kind,
                "lengths": list(pattern.lengths),
                "gaps": list(pattern.gaps),
            }
            for pattern in load_inv04().PATTERNS
        ],
        "stats": stats,
        "identity_split": {
            "validation_identity_count": int(len(reserved_validation_ids)),
            "validation_track_ids": sorted(int(v) for v in reserved_validation_ids),
            "rule": (
                "No original_track_id used by validation may appear in the "
                "training dataset. Mixed candidate scenes are discarded."
            ),
        },
        "files": {
            "examples": "examples.csv",
            "tracklets": "tracklets.csv",
            "observations": "observations.csv",
            "edges": "edges.csv",
            "global_motion": "global_motion.csv",
        },
    }
    atomic_json(path / "manifest.json", manifest)
    return manifest


# =============================================================================
# Geometry-only baseline
# =============================================================================


def geometry_baseline(examples: pd.DataFrame, edges: pd.DataFrame) -> dict[str, float]:
    positive = edges[edges["continuation_target"].astype(int) == 1]
    source_top1 = float((positive["source_distance_rank"].astype(int) == 1).mean())
    target_top1 = float((positive["target_distance_rank"].astype(int) == 1).mean())

    exact_count = 0
    for example_id in examples["example_id"].astype(int):
        local = edges[edges["example_id"].astype(int) == example_id]
        sources = sorted(local["source_tracklet_index"].astype(int).unique().tolist())
        targets = sorted(local["target_tracklet_index"].astype(int).unique().tolist())
        if len(sources) != len(targets):
            continue
        source_map = {value: i for i, value in enumerate(sources)}
        target_map = {value: i for i, value in enumerate(targets)}
        cost = np.full((len(sources), len(targets)), 1e6, dtype=np.float64)
        truth = np.zeros_like(cost, dtype=bool)
        for row in local.itertuples(index=False):
            i = source_map[int(row.source_tracklet_index)]
            j = target_map[int(row.target_tracklet_index)]
            cost[i, j] = float(row.direct_distance_um)
            truth[i, j] = bool(int(row.continuation_target))
        r, c = linear_sum_assignment(cost)
        exact_count += int(len(r) == len(sources) and all(truth[i, j] for i, j in zip(r, c)))

    return {
        "source_top1": source_top1,
        "target_top1": target_top1,
        "exact_assignment": exact_count / max(len(examples), 1),
    }


# =============================================================================
# Train-only statistics + validation tensorization
# =============================================================================


def prepare_examples_with_stats(
    trainer,
    data,
    *,
    crop_cache,
    appearance_mode: str,
    stats=None,
    global_median_volume: float | None = None,
):
    """Investigation-03 preparation with externally supplied train statistics."""
    motion = trainer._motion_lookup(data.global_motion)

    if global_median_volume is None:
        volume_all = pd.to_numeric(
            data.observations.get("volume", pd.Series(dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=float)
        volume_all = volume_all[np.isfinite(volume_all) & (volume_all > 0)]
        global_median_volume = float(np.median(volume_all)) if volume_all.size else 1.0

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
        raw, valid, diagnostics = trainer.build_raw_structured(rows, global_motion=motion)
        raw_by_tracklet[key] = raw
        valid_by_tracklet[key] = valid
        diagnostics_by_tracklet[key] = diagnostics
        all_raw.append(raw)
        all_valid.append(valid)

        if crop_cache is not None:
            row_ids = rows["_observation_row"].to_numpy(dtype=np.int64, copy=True)
            crop_fraction_by_tracklet[key] = np.asarray(
                crop_cache.valid_fraction[row_ids], dtype=np.float32
            )
        else:
            crop_fraction_by_tracklet[key] = np.ones(len(rows), dtype=np.float32)

    if stats is None:
        structured_stats = trainer.fit_masked_stats(
            np.concatenate(all_raw, axis=0),
            np.concatenate(all_valid, axis=0),
        )
    else:
        structured_stats = stats.structured

    reliability_raw_by_tracklet: dict[tuple[int, int], np.ndarray] = {}
    reliability_all: list[np.ndarray] = []
    for (example_id, tracklet_index), rows in data.observations.groupby(
        ["example_id", "tracklet_index"], sort=False
    ):
        rows = rows.sort_values("sequence_index")
        key = (int(example_id), int(tracklet_index))
        reliability = trainer.build_reliability_raw(
            rows,
            diagnostics=diagnostics_by_tracklet[key],
            crop_valid_fraction=crop_fraction_by_tracklet[key],
            global_median_volume=float(global_median_volume),
        )
        reliability_raw_by_tracklet[key] = reliability
        reliability_all.append(reliability)

    if stats is None:
        reliability_stats = trainer.fit_reliability_stats(np.stack(reliability_all, axis=0))
    else:
        reliability_stats = stats.reliability

    enriched_edges = trainer.enrich_pair_features(
        data,
        tracklet_diagnostics=diagnostics_by_tracklet,
    )
    if stats is None:
        pair_stats = trainer.fit_pair_stats(enriched_edges)
    else:
        pair_stats = stats.pair

    pair_tensor, _ = trainer.tensorize_stage11_pair_features(
        enriched_edges,
        mean=pair_stats.mean,
        std=pair_stats.std,
        device="cpu",
    )
    enriched_edges = enriched_edges.copy()
    enriched_edges["_pair_row"] = np.arange(len(enriched_edges), dtype=np.int64)
    pair_matrix = pair_tensor.numpy().astype(np.float32, copy=True)

    if stats is None:
        stats = trainer.FeatureStats(
            structured=structured_stats,
            reliability=reliability_stats,
            pair=pair_stats,
        )

    prepared: list[Any] = []
    fingerprint_dim = trainer.ReconcilerConfig().fingerprint.embedding_dim

    for example_id in sorted(data.examples["example_id"].astype(int).unique()):
        tracklets = data.tracklets[
            data.tracklets["example_id"].astype(int) == example_id
        ].sort_values("tracklet_index")
        edges = enriched_edges[
            enriched_edges["example_id"].astype(int) == example_id
        ].sort_values("edge_index")

        tracklet_indices = tracklets["tracklet_index"].astype(int).to_numpy(copy=True)
        local_index = {int(value): i for i, value in enumerate(tracklet_indices)}
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
            (n, len(trainer.TRACKLET_RELIABILITY_FEATURES)), dtype=np.float32
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
            structured[local, :k] = trainer.apply_masked_stats(
                raw_by_tracklet[key],
                valid_by_tracklet[key],
                structured_stats,
            )
            observation_mask[local, :k] = True
            times[local, :k] = rows["frame"].to_numpy(dtype=np.float32, copy=True)
            xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float32, copy=True)
            start_xyz[local] = xyz[0]
            end_xyz[local] = xyz[-1]
            reliability_raw[local] = reliability_raw_by_tracklet[key]

            if crops is not None:
                row_ids = rows["_observation_row"].to_numpy(dtype=np.int64, copy=True)
                crops[local, :k] = np.asarray(
                    crop_cache.crops[row_ids], dtype=np.float16
                )

        reliability = (
            reliability_raw - reliability_stats.mean[None, :]
        ) / reliability_stats.std[None, :]

        e = len(edges)
        edge_index = np.zeros((e, 2), dtype=np.int64)
        for row_number, edge in enumerate(edges.itertuples(index=False)):
            edge_index[row_number, 0] = local_index[int(edge.source_tracklet_index)]
            edge_index[row_number, 1] = local_index[int(edge.target_tracklet_index)]

        def xyz_from_columns(prefix: str) -> tuple[np.ndarray, np.ndarray]:
            values = np.zeros((e, 3), dtype=np.float32)
            valid = np.ones(e, dtype=bool)
            for axis, axis_name in enumerate(("z", "y", "x")):
                name = f"{prefix}_{axis_name}_um"
                if name not in edges:
                    valid[:] = False
                    continue
                column = pd.to_numeric(edges[name], errors="coerce").to_numpy(
                    dtype=np.float32, copy=True
                )
                valid &= np.isfinite(column)
                values[:, axis] = np.nan_to_num(column, nan=0.0)
            values[~valid] = 0.0
            return values, valid

        expected_global, global_valid = xyz_from_columns("expected_global")
        expected_relative, relative_valid = xyz_from_columns("expected_global_relative")
        expected_local = np.zeros((e, 3), dtype=np.float32)
        expected_backward = np.zeros((e, 3), dtype=np.float32)
        local_valid = np.zeros(e, dtype=bool)
        backward_valid = np.zeros(e, dtype=bool)

        pair_rows = edges["_pair_row"].to_numpy(dtype=np.int64, copy=True)
        prepared.append(
            trainer.PreparedExample(
                example_id=int(example_id),
                tracklet_indices=tracklet_indices.astype(np.int64, copy=True),
                structured=structured,
                observation_mask=observation_mask,
                times=times,
                start_xyz_um=start_xyz,
                end_xyz_um=end_xyz,
                reliability_raw=reliability.astype(np.float32, copy=True),
                crops=crops,
                fingerprints_zero=zero_fingerprints,
                edge_index=edge_index,
                gap_frames=edges["gap_frames"].to_numpy(dtype=np.int64, copy=True),
                pair_features=pair_matrix[pair_rows].copy(),
                expected_global=expected_global,
                expected_global_relative=expected_relative,
                expected_local=expected_local,
                expected_backward=expected_backward,
                prediction_valid=np.stack(
                    (global_valid, relative_valid, local_valid, backward_valid),
                    axis=-1,
                ),
                edge_target=edges["continuation_target"].to_numpy(
                    dtype=np.float32, copy=True
                ),
            )
        )

    return prepared, stats, float(global_median_volume)


# =============================================================================
# Evaluation strata
# =============================================================================


def subset_prepared(prepared: list[Any], ids: set[int]) -> list[Any]:
    return [example for example in prepared if int(example.example_id) in ids]


def validation_strata(examples: pd.DataFrame, edges: pd.DataFrame) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {
        "all": set(examples["example_id"].astype(int).tolist()),
        "pair": set(examples.loc[examples["pattern_kind"] == "pair", "example_id"].astype(int).tolist()),
        "chain": set(examples.loc[examples["pattern_kind"] == "chain", "example_id"].astype(int).tolist()),
        "wrong_closer": set(examples.loc[examples["wrong_closer_source_count"].astype(int) > 0, "example_id"].astype(int).tolist()),
    }
    positive = edges[edges["continuation_target"].astype(int) == 1]
    for gap in (1, 2, 3, 4):
        ids = set(
            positive.loc[positive["gap_frames"].astype(int) == gap, "example_id"]
            .astype(int)
            .tolist()
        )
        if ids:
            result[f"contains_gap_{gap}"] = ids
    for pattern_name, group in examples.groupby("pattern_name"):
        result[f"pattern:{pattern_name}"] = set(group["example_id"].astype(int).tolist())
    return result


def evaluate_strata(
    trainer,
    model,
    prepared: list[Any],
    *,
    strata: dict[str, set[int]],
    device: torch.device,
    amp_mode: str,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name, ids in strata.items():
        subset = subset_prepared(prepared, ids)
        if not subset:
            continue
        metrics = trainer.evaluate(
            model,
            subset,
            device=device,
            appearance_mode="crops",
            amp_mode=amp_mode,
        )
        output[name] = metrics.as_dict()
    return output


# =============================================================================
# Checkpoints / metrics
# =============================================================================


def strict_success(metrics) -> bool:
    return bool(
        metrics.exact_assignment >= 1.0
        and metrics.source_top1 >= 1.0
        and metrics.target_top1 >= 1.0
        and metrics.positive_probability_mean >= STRICT_POSITIVE_PROBABILITY
        and metrics.negative_probability_max <= STRICT_NEGATIVE_PROBABILITY
    )


def save_generalization_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    step: int,
    train_metrics,
    validation_metrics,
    stats,
    global_median_volume: float,
    args: argparse.Namespace,
    train_dataset,
    validation_dataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "investigation": SCRIPT_NAME,
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics.as_dict(),
            "validation_metrics": validation_metrics.as_dict(),
            "feature_stats": stats.to_json(),
            "train_global_median_volume": float(global_median_volume),
            "args": vars(args),
            "train_dataset": root_relative(train_dataset.root),
            "validation_dataset": root_relative(validation_dataset.root),
        },
        path,
    )


def print_dual_metrics(step: int, train_metrics, validation_metrics) -> None:
    print(
        f"[eval step={step:04d}] "
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


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build hard train/validation continuation datasets with completely "
            "disjoint original Trackastra identities, train on train only, and "
            "measure held-out generalization."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)

    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--validation-examples", type=int, default=DEFAULT_VALIDATION_EXAMPLES)
    parser.add_argument("--tracks-per-example", type=int, default=DEFAULT_TRACKS_PER_EXAMPLE)
    parser.add_argument("--guard-frames", type=int, default=DEFAULT_GUARD_FRAMES)
    parser.add_argument("--group-radius-um", type=float, default=DEFAULT_GROUP_RADIUS_UM)
    parser.add_argument("--max-cell-match-um", type=float, default=DEFAULT_MAX_CELL_MATCH_UM)
    parser.add_argument("--max-step-um", type=float, default=DEFAULT_MAX_STEP_UM)
    parser.add_argument("--max-volume-ratio", type=float, default=DEFAULT_MAX_VOLUME_RATIO)
    parser.add_argument("--min-boundary-distance-um", type=float, default=DEFAULT_MIN_BOUNDARY_DISTANCE_UM)
    parser.add_argument("--min-wrong-candidates-per-source", type=int, default=DEFAULT_MIN_WRONG_SOURCE)
    parser.add_argument("--min-wrong-candidates-per-target", type=int, default=DEFAULT_MIN_WRONG_TARGET)
    parser.add_argument("--train-max-track-reuse", type=int, default=DEFAULT_TRAIN_TRACK_REUSE)
    parser.add_argument("--validation-max-track-reuse", type=int, default=DEFAULT_VALIDATION_TRACK_REUSE)
    parser.add_argument("--max-placement-reuse", type=int, default=DEFAULT_PLACEMENT_REUSE)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="off")
    parser.add_argument("--crop-shape", default=DEFAULT_CROP_SHAPE)
    parser.add_argument("--rebuild-crops", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Run every requested step even after held-out strict success.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "train_examples",
        "validation_examples",
        "tracks_per_example",
        "train_max_track_reuse",
        "validation_max_track_reuse",
        "max_placement_reuse",
        "steps",
        "batch_size",
        "eval_every",
        "log_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if int(args.tracks_per_example) < 2:
        raise ValueError("--tracks-per-example must be >=2")
    for name in (
        "group_radius_um",
        "max_cell_match_um",
        "max_step_um",
        "max_volume_ratio",
        "min_boundary_distance_um",
        "lr",
        "grad_clip",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output = resolve(args.output) if args.output is not None else default_output(args.sample_id)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}\nPass --overwrite to rebuild Investigation 05.")
    output.mkdir(parents=True, exist_ok=True)

    inv04 = load_inv04()
    inv01 = inv04.load_inv01()
    trainer = load_inv03()
    inv04.patch_inv03(trainer)

    source_root = resolve(args.source) if args.source is not None else default_source(args.sample_id)
    source = inv01.SourcePaths(source_root)
    source.validate()
    spacing = tuple(float(value) for value in inv01.DEFAULT_SPACING_ZYX_UM)

    print("=" * 126, flush=True)
    print("TRACK RECONCILER — INVESTIGATION 05: HELD-OUT IDENTITY GENERALIZATION", flush=True)
    print("=" * 126, flush=True)
    print(f"sample                   : {args.sample_id}", flush=True)
    print(f"source Investigation 36  : {source.root}", flush=True)
    print(f"output                   : {output}", flush=True)
    print(f"train examples requested : {args.train_examples}", flush=True)
    print(f"validation requested     : {args.validation_examples}", flush=True)
    print(f"tracks/example           : {args.tracks_per_example}", flush=True)
    print("split rule               : original Trackastra track IDs are disjoint", flush=True)
    print("normalization rule       : fit TRAIN only; apply unchanged to validation", flush=True)
    print("=" * 126, flush=True)

    observations, lineage_ids, movie_shape = inv01.load_observations(source, spacing=spacing)
    global_motion = inv01.estimate_global_motion(observations, lineage_ids=lineage_ids)

    mining_args = argparse.Namespace(
        tracks_per_example=int(args.tracks_per_example),
        group_radius_um=float(args.group_radius_um),
        min_wrong_candidates_per_source=int(args.min_wrong_candidates_per_source),
        min_wrong_candidates_per_target=int(args.min_wrong_candidates_per_target),
        max_cell_match_um=float(args.max_cell_match_um),
        max_step_um=float(args.max_step_um),
        max_volume_ratio=float(args.max_volume_ratio),
        min_boundary_distance_um=float(args.min_boundary_distance_um),
    )
    patterns = {pattern.name: pattern for pattern in inv04.PATTERNS}

    started = time.perf_counter()
    eligible = inv04.enumerate_eligible(
        observations,
        movie_frames=int(movie_shape[0]),
        guard=int(args.guard_frames),
        args=mining_args,
    )
    candidates = inv04.enumerate_groups(eligible, patterns=patterns, args=mining_args)
    if not candidates:
        raise RuntimeError("No hard candidate scenes were found")

    print(
        f"[mining] eligible windows={sum(len(v) for v in eligible.values()):,} "
        f"hard scenes={len(candidates):,}",
        flush=True,
    )

    # 1) Select validation seed scenes from the full candidate pool.
    validation_seed = balanced_select(
        inv04,
        candidates,
        requested=int(args.validation_examples),
        max_track_reuse=int(args.validation_max_track_reuse),
        max_placement_reuse=int(args.max_placement_reuse),
        seed=int(args.seed) + 11,
    )
    if len(validation_seed) < int(args.validation_examples):
        raise RuntimeError(
            f"Could select only {len(validation_seed)} validation seed scenes; "
            "reduce --validation-examples or increase reuse limits."
        )
    validation_ids = {
        int(track_id)
        for group in validation_seed
        for track_id in group.track_ids
    }

    # 2) Remove every mixed scene, then re-select balanced scenes independently
    # inside the identity-pure train/validation pools.
    train_pool, validation_pool, mixed_pool = split_candidates_by_reserved_ids(
        candidates, validation_ids
    )
    selected_validation = balanced_select(
        inv04,
        validation_pool,
        requested=int(args.validation_examples),
        max_track_reuse=int(args.validation_max_track_reuse),
        max_placement_reuse=int(args.max_placement_reuse),
        seed=int(args.seed) + 23,
    )
    selected_train = balanced_select(
        inv04,
        train_pool,
        requested=int(args.train_examples),
        max_track_reuse=int(args.train_max_track_reuse),
        max_placement_reuse=int(args.max_placement_reuse),
        seed=int(args.seed) + 37,
    )
    if len(selected_validation) < int(args.validation_examples):
        raise RuntimeError(
            f"Identity-pure validation pool produced only {len(selected_validation)} scenes."
        )
    if len(selected_train) < int(args.train_examples):
        print(
            f"[warning] requested {args.train_examples} train scenes, selected "
            f"{len(selected_train)} after held-out identity exclusion.",
            flush=True,
        )

    train_original_ids = {
        int(track_id) for group in selected_train for track_id in group.track_ids
    }
    validation_original_ids = {
        int(track_id) for group in selected_validation for track_id in group.track_ids
    }
    leakage = train_original_ids.intersection(validation_original_ids)
    if leakage:
        raise RuntimeError(f"Identity leakage detected before materialization: {sorted(leakage)}")

    all_eligible_ids = sorted(
        {
            int(item.track_id)
            for items in eligible.values()
            for item in items
        }
    )
    split_rows = []
    for track_id in all_eligible_ids:
        if track_id in validation_ids:
            split = "validation_reserved"
        else:
            split = "train_pool"
        split_rows.append(
            {
                "original_track_id": int(track_id),
                "split": split,
                "used_in_train_examples": int(track_id in train_original_ids),
                "used_in_validation_examples": int(track_id in validation_original_ids),
            }
        )
    atomic_csv(output / "identity_split.csv", pd.DataFrame(split_rows))

    train_examples, train_tracklets, train_obs, train_edges = inv04.materialize(
        selected_train,
        observations=observations,
        eligible=eligible,
        patterns=patterns,
        global_motion=global_motion,
    )
    val_examples, val_tracklets, val_obs, val_edges = inv04.materialize(
        selected_validation,
        observations=observations,
        eligible=eligible,
        patterns=patterns,
        global_motion=global_motion,
    )
    inv04.validate_dataset(train_examples, train_tracklets, train_obs, train_edges)
    inv04.validate_dataset(val_examples, val_tracklets, val_obs, val_edges)

    materialized_train_ids = set(train_tracklets["original_track_id"].astype(int))
    materialized_val_ids = set(val_tracklets["original_track_id"].astype(int))
    leakage = materialized_train_ids.intersection(materialized_val_ids)
    if leakage:
        raise RuntimeError(f"Identity leakage after materialization: {sorted(leakage)}")

    train_dir = output / "train_dataset"
    validation_dir = output / "validation_dataset"
    train_manifest = write_dataset(
        train_dir,
        split_name="train",
        sample_id=str(args.sample_id),
        source=source,
        movie_shape=movie_shape,
        spacing=spacing,
        examples=train_examples,
        tracklets=train_tracklets,
        observations=train_obs,
        edges=train_edges,
        global_motion=global_motion,
        reserved_validation_ids=validation_ids,
    )
    validation_manifest = write_dataset(
        validation_dir,
        split_name="validation",
        sample_id=str(args.sample_id),
        source=source,
        movie_shape=movie_shape,
        spacing=spacing,
        examples=val_examples,
        tracklets=val_tracklets,
        observations=val_obs,
        edges=val_edges,
        global_motion=global_motion,
        reserved_validation_ids=validation_ids,
    )

    train_geometry = geometry_baseline(train_examples, train_edges)
    validation_geometry = geometry_baseline(val_examples, val_edges)

    print("", flush=True)
    print("=" * 126, flush=True)
    print("IDENTITY-DISJOINT DATASETS", flush=True)
    print("=" * 126, flush=True)
    print(
        f"reserved validation IDs : {len(validation_ids)} | "
        f"train-used IDs={len(materialized_train_ids)} | "
        f"validation-used IDs={len(materialized_val_ids)} | overlap=0",
        flush=True,
    )
    print(
        f"candidate pools          : train={len(train_pool):,} "
        f"validation={len(validation_pool):,} mixed-discarded={len(mixed_pool):,}",
        flush=True,
    )
    print(
        f"selected scenes          : train={len(train_examples)} "
        f"validation={len(val_examples)}",
        flush=True,
    )
    print(
        "validation hardness     : "
        f"wrong-closer={validation_manifest['stats']['wrong_closer_source_decisions']} "
        f"src-rank>1={validation_manifest['stats']['positive_source_rank_gt1']} "
        f"tgt-rank>1={validation_manifest['stats']['positive_target_rank_gt1']} "
        f"min-margin={validation_manifest['stats']['minimum_wrong_minus_true_um']:+.3f}um",
        flush=True,
    )
    print(
        "geometry baseline VAL   : "
        f"exact={100*validation_geometry['exact_assignment']:.2f}% "
        f"src_top1={100*validation_geometry['source_top1']:.2f}% "
        f"tgt_top1={100*validation_geometry['target_top1']:.2f}%",
        flush=True,
    )
    print(
        f"pattern coverage train   : {train_manifest['stats']['pattern_distribution']}",
        flush=True,
    )
    print(
        f"pattern coverage val     : {validation_manifest['stats']['pattern_distribution']}",
        flush=True,
    )
    print("=" * 126, flush=True)

    # -------------------------------------------------------------------------
    # Load datasets through the existing Investigation-03 contract.
    # -------------------------------------------------------------------------
    train_data = trainer.SourceDataset.load(train_dir)
    validation_data = trainer.SourceDataset.load(validation_dir)

    crop_shape = trainer.parse_triplet_int(args.crop_shape, name="crop-shape")
    training_dir = output / "training"
    checkpoint_dir = training_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    amp_mode = trainer.resolved_amp_mode(device, str(args.amp))

    train_crop_cache = trainer.build_crop_cache(
        train_data,
        cache_dir=training_dir / "cache_train",
        shape_zyx=crop_shape,
        distance_clip_um=float(trainer.DEFAULT_DISTANCE_CLIP_UM),
        rebuild=bool(args.rebuild_crops),
    )
    validation_crop_cache = trainer.build_crop_cache(
        validation_data,
        cache_dir=training_dir / "cache_validation",
        shape_zyx=crop_shape,
        distance_clip_um=float(trainer.DEFAULT_DISTANCE_CLIP_UM),
        rebuild=bool(args.rebuild_crops),
    )

    print("[features] fitting normalization statistics on TRAIN ONLY ...", flush=True)
    train_prepared, stats, train_global_median_volume = prepare_examples_with_stats(
        trainer,
        train_data,
        crop_cache=train_crop_cache,
        appearance_mode="crops",
        stats=None,
        global_median_volume=None,
    )
    print("[features] applying fixed TRAIN statistics to validation ...", flush=True)
    validation_prepared, _same_stats, _ = prepare_examples_with_stats(
        trainer,
        validation_data,
        crop_cache=validation_crop_cache,
        appearance_mode="crops",
        stats=stats,
        global_median_volume=train_global_median_volume,
    )
    atomic_json(
        training_dir / "feature_stats.json",
        {
            **stats.to_json(),
            "fit_split": "train_only",
            "train_global_median_volume": float(train_global_median_volume),
        },
    )

    config = trainer.ReconcilerConfig()
    model = trainer.TrackletReconciliationNetwork(config).to(device)
    trainer.disable_all_dropout(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=1.0e-5,
    )
    scaler = None
    if device.type == "cuda" and amp_mode == "fp16":
        scaler = torch.amp.GradScaler("cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[model] trainable parameters={trainer.trainable_parameter_count(model):,} "
        f"device={device} amp={amp_mode}",
        flush=True,
    )

    initial_train = trainer.evaluate(
        model,
        train_prepared,
        device=device,
        appearance_mode="crops",
        amp_mode=amp_mode,
    )
    initial_validation = trainer.evaluate(
        model,
        validation_prepared,
        device=device,
        appearance_mode="crops",
        amp_mode=amp_mode,
    )
    print_dual_metrics(0, initial_train, initial_validation)

    strata = validation_strata(val_examples, val_edges)
    initial_strata = evaluate_strata(
        trainer,
        model,
        validation_prepared,
        strata=strata,
        device=device,
        amp_mode=amp_mode,
    )

    rng = np.random.default_rng(int(args.seed) + 101)
    order = np.arange(len(train_prepared), dtype=np.int64)
    cursor = len(order)
    running_loss = 0.0
    running_count = 0
    best_validation_exact = -1.0
    best_validation_loss = float("inf")
    validation_strict_streak = 0
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    def next_indices() -> np.ndarray:
        nonlocal cursor, order
        batch_size = min(int(args.batch_size), len(train_prepared))
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        result = order[cursor : cursor + batch_size]
        cursor += batch_size
        return result

    step = 0
    for step in range(1, int(args.steps) + 1):
        model.train()
        batch_examples = [train_prepared[int(i)] for i in next_indices()]
        batch = trainer.collate(
            batch_examples,
            device=device,
            appearance_mode="crops",
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

        running_loss += float(loss.detach().float().cpu())
        running_count += 1

        if step % int(args.log_every) == 0:
            elapsed = time.perf_counter() - training_started
            memory = ""
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                memory = f" peakVRAM={peak:.2f}GiB"
            print(
                f"[train step={step:04d}] minibatch_loss="
                f"{running_loss/max(running_count,1):.5f} "
                f"time={elapsed:.1f}s{memory}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            train_metrics = trainer.evaluate(
                model,
                train_prepared,
                device=device,
                appearance_mode="crops",
                amp_mode=amp_mode,
            )
            validation_metrics = trainer.evaluate(
                model,
                validation_prepared,
                device=device,
                appearance_mode="crops",
                amp_mode=amp_mode,
            )
            print_dual_metrics(step, train_metrics, validation_metrics)

            strata_metrics = evaluate_strata(
                trainer,
                model,
                validation_prepared,
                strata=strata,
                device=device,
                amp_mode=amp_mode,
            )
            hard_metrics = strata_metrics.get("wrong_closer")
            chain_metrics = strata_metrics.get("chain")
            if hard_metrics is not None or chain_metrics is not None:
                pieces = []
                if hard_metrics is not None:
                    pieces.append(
                        f"wrong_closer exact={100*hard_metrics['exact_assignment']:.1f}% "
                        f"src={100*hard_metrics['source_top1']:.1f}%"
                    )
                if chain_metrics is not None:
                    pieces.append(
                        f"chain exact={100*chain_metrics['exact_assignment']:.1f}% "
                        f"src={100*chain_metrics['source_top1']:.1f}%"
                    )
                print("[val strata] " + " | ".join(pieces), flush=True)

            record = {
                "step": int(step),
                "elapsed_seconds": float(time.perf_counter() - training_started),
                "train": train_metrics.as_dict(),
                "validation": validation_metrics.as_dict(),
                "validation_strata": strata_metrics,
            }
            history.append(record)
            atomic_json(training_dir / "history.json", history)

            better = (
                validation_metrics.exact_assignment > best_validation_exact
                or (
                    validation_metrics.exact_assignment == best_validation_exact
                    and validation_metrics.loss < best_validation_loss
                )
            )
            if better:
                best_validation_exact = validation_metrics.exact_assignment
                best_validation_loss = validation_metrics.loss
                save_generalization_checkpoint(
                    checkpoint_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    stats=stats,
                    global_median_volume=train_global_median_volume,
                    args=args,
                    train_dataset=train_data,
                    validation_dataset=validation_data,
                )

            save_generalization_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                stats=stats,
                global_median_volume=train_global_median_volume,
                args=args,
                train_dataset=train_data,
                validation_dataset=validation_data,
            )

            if strict_success(validation_metrics):
                validation_strict_streak += 1
                print(
                    f"[held-out success] strict validation check "
                    f"{validation_strict_streak}/{STRICT_VALIDATION_STREAK}",
                    flush=True,
                )
            else:
                validation_strict_streak = 0

            if (
                not args.no_early_stop
                and validation_strict_streak >= STRICT_VALIDATION_STREAK
            ):
                print(
                    "[held-out success] identity-disjoint validation reached "
                    "exact assignment and strict probability separation.",
                    flush=True,
                )
                break

    final_train = trainer.evaluate(
        model,
        train_prepared,
        device=device,
        appearance_mode="crops",
        amp_mode=amp_mode,
    )
    final_validation = trainer.evaluate(
        model,
        validation_prepared,
        device=device,
        appearance_mode="crops",
        amp_mode=amp_mode,
    )
    final_strata = evaluate_strata(
        trainer,
        model,
        validation_prepared,
        strata=strata,
        device=device,
        amp_mode=amp_mode,
    )

    elapsed = time.perf_counter() - training_started
    print("", flush=True)
    print("=" * 126, flush=True)
    print("FINAL HELD-OUT GENERALIZATION", flush=True)
    print("=" * 126, flush=True)
    print_dual_metrics(step, final_train, final_validation)
    print(
        f"geometry baseline VAL   : exact={100*validation_geometry['exact_assignment']:.2f}% "
        f"src={100*validation_geometry['source_top1']:.2f}% "
        f"tgt={100*validation_geometry['target_top1']:.2f}%",
        flush=True,
    )
    print(f"elapsed                  : {elapsed:.1f}s", flush=True)
    print(f"best validation checkpoint: {checkpoint_dir / 'best_validation.pt'}", flush=True)
    print("=" * 126, flush=True)

    summary = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "sample_id": str(args.sample_id),
        "identity_overlap": 0,
        "reserved_validation_identity_count": int(len(validation_ids)),
        "used_train_identity_count": int(len(materialized_train_ids)),
        "used_validation_identity_count": int(len(materialized_val_ids)),
        "candidate_pool": {
            "all": int(len(candidates)),
            "train_pure": int(len(train_pool)),
            "validation_pure": int(len(validation_pool)),
            "mixed_discarded": int(len(mixed_pool)),
        },
        "train_dataset": train_manifest,
        "validation_dataset": validation_manifest,
        "geometry_baseline": {
            "train": train_geometry,
            "validation": validation_geometry,
        },
        "initial": {
            "train": initial_train.as_dict(),
            "validation": initial_validation.as_dict(),
            "validation_strata": initial_strata,
        },
        "final": {
            "step": int(step),
            "train": final_train.as_dict(),
            "validation": final_validation.as_dict(),
            "validation_strata": final_strata,
            "strict_validation_success": bool(strict_success(final_validation)),
        },
        "normalization": {
            "fit_split": "train_only",
            "validation_statistics_used": False,
        },
        "best_validation_checkpoint": "training/checkpoints/best_validation.pt",
        "latest_checkpoint": "training/checkpoints/latest.pt",
    }
    atomic_json(training_dir / "summary.json", summary)
    atomic_json(output / "summary.json", summary)

    if final_train.exact_assignment >= 0.999 and final_validation.exact_assignment < 0.80:
        print(
            "[interpretation] The network memorized training but generalization "
            "remains weak. Do not expand pseudo-labelled training yet; inspect "
            "which validation strata fail.",
            flush=True,
        )
    elif final_validation.exact_assignment >= 0.90:
        print(
            "[interpretation] Strong held-out-identity continuation generalization "
            "was observed. The next useful experiment is real Trackastra-break repair.",
            flush=True,
        )
    else:
        print(
            "[interpretation] Partial held-out generalization. Use the saved "
            "pattern/gap strata to identify the remaining failure regime.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
