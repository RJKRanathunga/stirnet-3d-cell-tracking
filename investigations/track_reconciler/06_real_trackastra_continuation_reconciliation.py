from __future__ import annotations

r"""
Investigation 06 — real Trackastra continuation reconciliation.

Purpose
-------
Investigations 03-05 validated the learned continuation network on synthetic
breaks.  This investigation applies the best Investigation-05 checkpoint to
the REAL non-boundary Trackastra breaks in Investigation 36.

This remains continuation-only:
    * no explicit division repair head is used
    * no birth/death learning
    * Trackastra child->parent "lineage" membership is NOT treated as truth and
      is NOT excluded from candidate repair
    * boundary entries/exits are excluded from candidate repair

Trackastra lineage membership is retained only as a diagnostic flag. In this
BioHub sequence many lineage-flagged tracklets are actually broken
continuations, while only a small subset appear to be true divisions.

The script deliberately uses conservative language:

    "reconciler-corrected"
        A previously broken endpoint disappeared after applying an accepted
        learned continuation link.

    "still broken"
        A non-boundary, endpoint remains after reconciliation.

These are NOT manual-ground-truth labels.  The current 20-frame movie does not
yet contain manual tracking truth.  Trackastra lineage flags are also not used
as ground truth.  The Napari viewer is therefore an explicit human inspection
step.

Pipeline
--------
Investigation-36 Trackastra tracks
        |
        +-- identify ALL interior ended/new tracklets
        |
        +-- build gap-1..4 physical candidate edges
        |
        +-- form local components per target-start frame
        |
        +-- use 4-observation endpoint context + five-channel 3-D crops
        |
        +-- load Investigation-05 best_validation.pt
        |
        +-- learned continuation probabilities
        |
        +-- conservative mutual-best + probability gate
        |
        +-- merge accepted tracklet identities
        |
        +-- Stage-09 before/after endpoint comparison
        |
        +-- Notebook-09-style Napari visualization

Viewer layers
-------------
CORRECTED - Reconciled Trajectories
CORRECTED - Accepted Links
CORRECTED - Removed Ends
CORRECTED - Removed Starts

STILL BROKEN - Ended Tracks
STILL BROKEN - New Tracks

Original / reconciled full tracks, boundary events and Trackastra
lineage-FLAGGED tracks are also available as context layers.

Default run
-----------
From repository root:

    python .\investigations\track_reconciler\06_real_trackastra_continuation_reconciliation.py --overwrite

Inference without opening Napari:

    python .\investigations\track_reconciler\06_real_trackastra_continuation_reconciliation.py --overwrite --no-viewer

Open an existing completed result:

    python .\investigations\track_reconciler\06_real_trackastra_continuation_reconciliation.py --viewer-only

Important
---------
The default acceptance threshold is deliberately conservative (0.90) because
Investigation 05 showed excellent held-out top-1 generalization but a gap-4
negative could still reach roughly 0.53.  A proposed link must also be the
learned model's best edge for both its source and its target unless
--allow-non-mutual is explicitly supplied.
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


SCRIPT_NAME = "06_real_trackastra_continuation_reconciliation"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"

DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_BOUNDARY_MARGIN_UM = 4.0
DEFAULT_MAX_GAP = 4
DEFAULT_SOURCE_HISTORY = 4
DEFAULT_TARGET_FUTURE = 4
DEFAULT_PROBABILITY_THRESHOLD = 0.90
DEFAULT_MAX_COMPONENT_EDGES = 96
DEFAULT_CROP_SHAPE = "13,41,41"
DEFAULT_DISTANCE_CLIP_UM = 5.0


# =============================================================================
# Repository / dynamic imports
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


def root_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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
        ROOT
        / "investigations"
        / "track_reconciler"
        / "01_build_overfit_dataset.py",
        "_track_reconciler_inv01_for_06",
    )


def load_inv03():
    return load_script(
        ROOT
        / "investigations"
        / "track_reconciler"
        / "03_continuation_overfit.py",
        "_track_reconciler_inv03_for_06",
    )


def load_inv04():
    return load_script(
        ROOT
        / "investigations"
        / "track_reconciler"
        / "04_hard_continuation_overfit.py",
        "_track_reconciler_inv04_for_06",
    )


def load_inv05():
    return load_script(
        ROOT
        / "investigations"
        / "track_reconciler"
        / "05_hard_continuation_generalization.py",
        "_track_reconciler_inv05_for_06",
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


def default_checkpoint(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "investigations"
        / "05_hard_continuation_generalization"
        / sample_id
        / "training"
        / "checkpoints"
        / "best_validation.pt"
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


def parse_triplet_int(text: str, *, name: str) -> tuple[int, int, int]:
    values = tuple(int(token.strip()) for token in str(text).split(","))
    if len(values) != 3 or not all(value > 0 for value in values):
        raise ValueError(f"{name} must contain three positive integers")
    if not all(value % 2 == 1 for value in values):
        raise ValueError(f"{name} should use odd dimensions")
    return values  # type: ignore[return-value]


# =============================================================================
# Candidate discovery
# =============================================================================


def candidate_radius_um(gap_frames: int) -> float:
    return float(min(25.0, 12.0 + 4.0 * max(int(gap_frames) - 1, 0)))


def pair_distance_um(a: pd.Series, b: pd.Series) -> float:
    av = a[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    bv = b[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    return float(np.linalg.norm(bv - av))


def global_shift_lookup(global_motion: pd.DataFrame) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for row in global_motion.itertuples(index=False):
        value = np.asarray(
            [row.shift_z_um, row.shift_y_um, row.shift_x_um],
            dtype=np.float64,
        )
        if np.isfinite(value).all():
            lookup[int(row.frame_from)] = value
    return lookup


def cumulative_global_shift(
    lookup: dict[int, np.ndarray],
    *,
    source_end_frame: int,
    target_start_frame: int,
) -> tuple[np.ndarray, bool]:
    values: list[np.ndarray] = []
    for frame in range(int(source_end_frame), int(target_start_frame)):
        shift = lookup.get(frame)
        if shift is None:
            return np.zeros(3, dtype=np.float64), False
        values.append(shift)
    if not values:
        return np.zeros(3, dtype=np.float64), True
    return np.sum(np.asarray(values), axis=0), True


def source_relative_velocity(
    rows: pd.DataFrame,
    *,
    lookup: dict[int, np.ndarray],
) -> tuple[np.ndarray, int, float]:
    rows = rows.sort_values("frame")
    xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    frames = rows["frame"].to_numpy(dtype=np.int64)
    residuals: list[np.ndarray] = []

    for index in range(len(rows) - 1):
        f0 = int(frames[index])
        f1 = int(frames[index + 1])
        if f1 != f0 + 1:
            continue
        shift = lookup.get(f0)
        if shift is None:
            continue
        residuals.append((xyz[index + 1] - xyz[index]) - shift)

    if not residuals:
        return np.zeros(3, dtype=np.float64), 0, 0.0

    recent = np.asarray(residuals[-3:], dtype=np.float64)
    center = recent.mean(axis=0)
    scatter = float(
        np.linalg.norm(recent - center[None, :], axis=1).mean()
    )
    return center, int(len(recent)), scatter


@dataclass(frozen=True)
class RawCandidate:
    source_track_id: int
    target_track_id: int
    source_end_frame: int
    target_start_frame: int
    gap_frames: int
    radius_um: float
    direct_distance_um: float


@dataclass(frozen=True)
class CandidateComponent:
    target_start_frame: int
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    edges: tuple[RawCandidate, ...]


def endpoint_rows(
    observations: pd.DataFrame,
) -> tuple[dict[int, pd.Series], dict[int, pd.Series], pd.DataFrame]:
    summary = (
        observations.groupby("track_id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            observation_count=("frame", "size"),
        )
        .reset_index()
    )
    start_rows: dict[int, pd.Series] = {}
    end_rows: dict[int, pd.Series] = {}
    for track_id, group in observations.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        start_rows[int(track_id)] = group.iloc[0]
        end_rows[int(track_id)] = group.iloc[-1]
    return start_rows, end_rows, summary


def discover_raw_candidates(
    observations: pd.DataFrame,
    *,
    lineage_ids: set[int],
    movie_frames: int,
    max_gap: int,
    boundary_margin_um: float,
    stage9_source_ids: set[int] | None = None,
    stage9_target_ids: set[int] | None = None,
) -> tuple[list[RawCandidate], dict[str, int]]:
    starts, ends, summary = endpoint_rows(observations)

    # Prefer the exact Stage-09 endpoint classifier used by the viewer.  It can
    # use cell bounding boxes when cells_all.csv provides them, while the
    # synthetic-data builder's distance_to_boundary_um is centroid based.
    #
    # IMPORTANT: Trackastra child->parent membership is NOT a filter here.
    # In this BioHub sequence many lineage-flagged tracklets are actually
    # broken continuations. We retain lineage membership only as a diagnostic.
    if stage9_source_ids is not None and stage9_target_ids is not None:
        source_ids = {int(value) for value in stage9_source_ids}
        target_ids = {int(value) for value in stage9_target_ids}
    else:
        source_ids: set[int] = set()
        target_ids: set[int] = set()

        for row in summary.itertuples(index=False):
            tid = int(row.track_id)

            if int(row.last_frame) < movie_frames - 1:
                endpoint = ends[tid]
                boundary = float(endpoint.get("distance_to_boundary_um", np.nan))
                if np.isfinite(boundary) and boundary > boundary_margin_um:
                    source_ids.add(tid)

            if int(row.first_frame) > 0:
                endpoint = starts[tid]
                boundary = float(endpoint.get("distance_to_boundary_um", np.nan))
                if np.isfinite(boundary) and boundary > boundary_margin_um:
                    target_ids.add(tid)

    candidates: list[RawCandidate] = []
    for target_id in sorted(target_ids):
        target = starts[target_id]
        tf = int(target.frame)

        for source_id in sorted(source_ids):
            if source_id == target_id:
                continue
            source = ends[source_id]
            sf = int(source.frame)
            gap = tf - sf
            if gap < 1 or gap > max_gap:
                continue

            radius = candidate_radius_um(gap)
            distance = pair_distance_um(source, target)
            if distance > radius:
                continue

            candidates.append(
                RawCandidate(
                    source_track_id=int(source_id),
                    target_track_id=int(target_id),
                    source_end_frame=sf,
                    target_start_frame=tf,
                    gap_frames=int(gap),
                    radius_um=float(radius),
                    direct_distance_um=float(distance),
                )
            )

    return candidates, {
        "interior_ended_tracks": int(len(source_ids)),
        "interior_new_tracks": int(len(target_ids)),
        "lineage_flagged_interior_ended_tracks": int(
            len(source_ids.intersection(lineage_ids))
        ),
        "lineage_flagged_interior_new_tracks": int(
            len(target_ids.intersection(lineage_ids))
        ),
        "candidate_source_tracks": int(
            len({candidate.source_track_id for candidate in candidates})
        ),
        "candidate_target_tracks": int(
            len({candidate.target_track_id for candidate in candidates})
        ),
    }


def split_components(
    candidates: list[RawCandidate],
    *,
    max_component_edges: int,
) -> tuple[list[CandidateComponent], list[CandidateComponent]]:
    """
    Build spatial candidate components independently for each target-start frame.

    This bounds edge-attention cost and preserves all source competition for the
    same newly starting tracks.  A real middle fragment may appear as a target
    in one component and later as a source in another component; global accepted
    links are reconciled afterward, allowing A -> B -> C chains.
    """
    by_frame: dict[int, list[RawCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate.target_start_frame].append(candidate)

    accepted_components: list[CandidateComponent] = []
    oversized_components: list[CandidateComponent] = []

    for target_frame, frame_edges in sorted(by_frame.items()):
        adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        edges_by_node: dict[tuple[str, int], list[RawCandidate]] = defaultdict(list)

        for edge in frame_edges:
            s = ("s", int(edge.source_track_id))
            t = ("t", int(edge.target_track_id))
            adjacency[s].add(t)
            adjacency[t].add(s)
            edges_by_node[s].append(edge)
            edges_by_node[t].append(edge)

        remaining = set(adjacency)
        while remaining:
            seed = min(remaining)
            queue = deque([seed])
            nodes: set[tuple[str, int]] = set()

            while queue:
                node = queue.popleft()
                if node in nodes:
                    continue
                nodes.add(node)
                for neighbour in adjacency[node]:
                    if neighbour not in nodes:
                        queue.append(neighbour)

            remaining.difference_update(nodes)

            local_edges: dict[
                tuple[int, int, int, int], RawCandidate
            ] = {}
            for node in nodes:
                for edge in edges_by_node[node]:
                    key = (
                        edge.source_track_id,
                        edge.target_track_id,
                        edge.source_end_frame,
                        edge.target_start_frame,
                    )
                    local_edges[key] = edge

            edges = tuple(
                sorted(
                    local_edges.values(),
                    key=lambda e: (
                        e.source_track_id,
                        e.target_track_id,
                    ),
                )
            )
            sources = tuple(
                sorted(
                    node_id
                    for role, node_id in nodes
                    if role == "s"
                )
            )
            targets = tuple(
                sorted(
                    node_id
                    for role, node_id in nodes
                    if role == "t"
                )
            )
            component = CandidateComponent(
                target_start_frame=int(target_frame),
                sources=sources,
                targets=targets,
                edges=edges,
            )

            if len(edges) > int(max_component_edges):
                oversized_components.append(component)
            else:
                accepted_components.append(component)

    return accepted_components, oversized_components


# =============================================================================
# Real dataset materialization in Investigation-03 contract
# =============================================================================


def source_window(
    by_track: dict[int, pd.DataFrame],
    track_id: int,
    history: int,
) -> pd.DataFrame:
    return by_track[int(track_id)].sort_values("frame").tail(int(history)).copy()


def target_window(
    by_track: dict[int, pd.DataFrame],
    track_id: int,
    future: int,
) -> pd.DataFrame:
    return by_track[int(track_id)].sort_values("frame").head(int(future)).copy()


def materialize_real_dataset(
    components: list[CandidateComponent],
    *,
    observations: pd.DataFrame,
    global_motion: pd.DataFrame,
    source_history: int,
    target_future: int,
    lineage_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_track = {
        int(track_id): group.sort_values("frame")
        for track_id, group in observations.groupby("track_id", sort=False)
    }
    global_lookup = global_shift_lookup(global_motion)
    lineage_ids_for_materialization = set(int(v) for v in lineage_ids)

    example_rows: list[dict[str, Any]] = []
    tracklet_rows: list[dict[str, Any]] = []
    observation_parts: list[pd.DataFrame] = []
    edge_rows: list[dict[str, Any]] = []

    for example_id, component in enumerate(components):
        source_ids = list(component.sources)
        target_ids = list(component.targets)

        # Source/target role copies are intentionally separate within one
        # target-start-frame component. A physical Trackastra track can appear
        # in another component later under the opposite role.
        tracklet_index: dict[tuple[str, int], int] = {}
        local = 0

        for source_id in source_ids:
            tracklet_index[("s", source_id)] = local
            rows = source_window(by_track, source_id, source_history)
            payload = rows.copy()
            payload.insert(0, "example_id", int(example_id))
            payload.insert(1, "tracklet_index", int(local))
            payload.insert(2, "sequence_index", np.arange(len(payload), dtype=np.int64))
            payload.insert(3, "role", "source")
            payload["original_track_id"] = int(source_id)
            observation_parts.append(payload)

            tracklet_rows.append(
                {
                    "example_id": int(example_id),
                    "tracklet_index": int(local),
                    "role": "source",
                    "original_track_id": int(source_id),
                    "trackastra_lineage_flag": int(source_id in lineage_ids_for_materialization),
                    "start_frame": int(rows["frame"].min()),
                    "end_frame": int(rows["frame"].max()),
                    "observation_count": int(len(rows)),
                }
            )
            local += 1

        for target_id in target_ids:
            tracklet_index[("t", target_id)] = local
            rows = target_window(by_track, target_id, target_future)
            payload = rows.copy()
            payload.insert(0, "example_id", int(example_id))
            payload.insert(1, "tracklet_index", int(local))
            payload.insert(2, "sequence_index", np.arange(len(payload), dtype=np.int64))
            payload.insert(3, "role", "target")
            payload["original_track_id"] = int(target_id)
            observation_parts.append(payload)

            tracklet_rows.append(
                {
                    "example_id": int(example_id),
                    "tracklet_index": int(local),
                    "role": "target",
                    "original_track_id": int(target_id),
                    "trackastra_lineage_flag": int(target_id in lineage_ids_for_materialization),
                    "start_frame": int(rows["frame"].min()),
                    "end_frame": int(rows["frame"].max()),
                    "observation_count": int(len(rows)),
                }
            )
            local += 1

        local_edges = list(component.edges)

        # Geometry ranks are computed within the actual component.
        source_to_edges: dict[int, list[int]] = defaultdict(list)
        target_to_edges: dict[int, list[int]] = defaultdict(list)
        for index, edge in enumerate(local_edges):
            source_to_edges[edge.source_track_id].append(index)
            target_to_edges[edge.target_track_id].append(index)

        source_rank: dict[int, int] = {}
        target_rank: dict[int, int] = {}
        for indices in source_to_edges.values():
            ordered = sorted(indices, key=lambda i: local_edges[i].direct_distance_um)
            for rank, index in enumerate(ordered, start=1):
                source_rank[index] = rank
        for indices in target_to_edges.values():
            ordered = sorted(indices, key=lambda i: local_edges[i].direct_distance_um)
            for rank, index in enumerate(ordered, start=1):
                target_rank[index] = rank

        for edge_index, edge in enumerate(local_edges):
            source_rows = source_window(
                by_track,
                edge.source_track_id,
                source_history,
            )
            source_endpoint = source_rows.sort_values("frame").iloc[-1]
            target_endpoint = by_track[edge.target_track_id].sort_values("frame").iloc[0]

            source_xyz = source_endpoint[
                ["z_um", "y_um", "x_um"]
            ].to_numpy(dtype=np.float64)
            target_xyz = target_endpoint[
                ["z_um", "y_um", "x_um"]
            ].to_numpy(dtype=np.float64)

            global_delta, global_valid = cumulative_global_shift(
                global_lookup,
                source_end_frame=edge.source_end_frame,
                target_start_frame=edge.target_start_frame,
            )
            expected_global = (
                source_xyz + global_delta
                if global_valid
                else np.full(3, np.nan, dtype=np.float64)
            )

            relative_velocity, rv_samples, rv_scatter = source_relative_velocity(
                source_rows,
                lookup=global_lookup,
            )
            relative_valid = bool(global_valid and rv_samples > 0)
            expected_relative = (
                expected_global + relative_velocity * float(edge.gap_frames)
                if relative_valid
                else np.full(3, np.nan, dtype=np.float64)
            )

            row = {
                "example_id": int(example_id),
                "edge_index": int(edge_index),
                "source_tracklet_index": int(
                    tracklet_index[("s", edge.source_track_id)]
                ),
                "target_tracklet_index": int(
                    tracklet_index[("t", edge.target_track_id)]
                ),
                "source_original_track_id": int(edge.source_track_id),
                "target_original_track_id": int(edge.target_track_id),
                "source_trackastra_lineage_flag": int(
                    edge.source_track_id in lineage_ids_for_materialization
                ),
                "target_trackastra_lineage_flag": int(
                    edge.target_track_id in lineage_ids_for_materialization
                ),
                "source_end_frame": int(edge.source_end_frame),
                "target_start_frame": int(edge.target_start_frame),
                "gap_frames": int(edge.gap_frames),
                "candidate_radius_um": float(edge.radius_um),
                "direct_distance_um": float(edge.direct_distance_um),
                "source_distance_rank": int(source_rank[edge_index]),
                "target_distance_rank": int(target_rank[edge_index]),
                # There is no real tracking GT yet. This dummy field only
                # satisfies the Investigation-03 tensor contract; it is never
                # used as a loss or metric in Investigation 06.
                "continuation_target": 0,
                "global_prediction_valid": int(global_valid),
                "global_relative_prediction_valid": int(relative_valid),
                "relative_velocity_samples": int(rv_samples),
                "relative_velocity_error_um": float(rv_scatter),
                "expected_global_z_um": float(expected_global[0]),
                "expected_global_y_um": float(expected_global[1]),
                "expected_global_x_um": float(expected_global[2]),
                "global_error_um": (
                    float(np.linalg.norm(target_xyz - expected_global))
                    if global_valid
                    else np.nan
                ),
                "expected_global_relative_z_um": float(expected_relative[0]),
                "expected_global_relative_y_um": float(expected_relative[1]),
                "expected_global_relative_x_um": float(expected_relative[2]),
                "global_relative_error_um": (
                    float(np.linalg.norm(target_xyz - expected_relative))
                    if relative_valid
                    else np.nan
                ),
            }
            edge_rows.append(row)

        example_rows.append(
            {
                "example_id": int(example_id),
                "target_start_frame": int(component.target_start_frame),
                "source_track_count": int(len(source_ids)),
                "target_track_count": int(len(target_ids)),
                "candidate_edge_count": int(len(local_edges)),
                "source_track_ids": json.dumps(source_ids),
                "target_track_ids": json.dumps(target_ids),
            }
        )

    examples = pd.DataFrame(example_rows)
    tracklets = pd.DataFrame(tracklet_rows)
    observations_out = (
        pd.concat(observation_parts, ignore_index=True)
        if observation_parts
        else pd.DataFrame()
    )
    edges = pd.DataFrame(edge_rows)
    return examples, tracklets, observations_out, edges


def write_real_dataset(
    path: Path,
    *,
    sample_id: str,
    source,
    movie_shape: tuple[int, ...],
    spacing: tuple[float, float, float],
    examples: pd.DataFrame,
    tracklets: pd.DataFrame,
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    global_motion: pd.DataFrame,
    max_gap: int,
    source_history: int,
    target_future: int,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_csv(path / "examples.csv", examples)
    atomic_csv(path / "tracklets.csv", tracklets)
    atomic_csv(path / "observations.csv", observations)
    atomic_csv(path / "edges.csv", edges)
    atomic_csv(path / "global_motion.csv", global_motion)

    manifest = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "sample_id": str(sample_id),
        "movie_shape_tzyx": list(movie_shape),
        "spacing_zyx_um": list(spacing),
        "source": {
            "investigation_36_root": root_relative(source.root),
            "raw": root_relative(source.raw),
            "preprocessed": root_relative(source.preprocessed),
            "final_instances": root_relative(source.final_instances),
            "cells": root_relative(source.cells),
            "tracks": root_relative(source.tracks),
            "napari_graph": root_relative(source.napari_graph),
        },
        "parameters": {
            "candidate_radius_um": 25.0,
            "candidate_radius_policy": "min(25, 12 + 4*(gap_frames-1)) um",
            "max_gap_frames": int(max_gap),
            "source_history_observations": int(source_history),
            "target_future_observations": int(target_future),
            "manual_tracking_ground_truth": False,
        },
        "stats": {
            "examples": int(len(examples)),
            "synthetic_contract_tracklets": int(len(tracklets)),
            "observation_rows": int(len(observations)),
            "candidate_edges": int(len(edges)),
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


# =============================================================================
# Checkpoint / feature-stat loading
# =============================================================================


def stats_from_checkpoint(trainer, checkpoint: dict[str, Any]):
    payload = checkpoint.get("feature_stats")
    if not isinstance(payload, dict):
        raise KeyError("Checkpoint does not contain feature_stats")

    def masked(name: str):
        block = payload.get(name)
        if not isinstance(block, dict):
            raise KeyError(f"feature_stats missing {name!r}")
        return trainer.MaskedStats(
            mean=np.asarray(block["mean"], dtype=np.float32),
            std=np.asarray(block["std"], dtype=np.float32),
        )

    stats = trainer.FeatureStats(
        structured=masked("structured"),
        reliability=masked("reliability"),
        pair=masked("pair"),
    )
    return stats


def load_model(
    trainer,
    checkpoint_path: Path,
    *,
    device: torch.device,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Investigation-05 checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")

    model = trainer.TrackletReconciliationNetwork(
        trainer.ReconcilerConfig()
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.disable_all_dropout(model)
    model.eval()

    stats = stats_from_checkpoint(trainer, checkpoint)
    median_volume = float(
        checkpoint.get("train_global_median_volume", 1.0)
    )
    return model, stats, median_volume, checkpoint


# =============================================================================
# Inference
# =============================================================================


def infer_edges(
    trainer,
    model,
    prepared: list[Any],
    data,
    *,
    device: torch.device,
    amp_mode: str,
) -> pd.DataFrame:
    scored_parts: list[pd.DataFrame] = []

    by_example = {
        int(example_id): group.sort_values("edge_index").copy()
        for example_id, group in data.edges.groupby("example_id", sort=False)
    }

    model.eval()
    with torch.no_grad():
        for position, example in enumerate(prepared, start=1):
            batch = trainer.collate(
                [example],
                device=device,
                appearance_mode="crops",
            )
            with trainer.autocast_context(device, amp_mode):
                output = model(batch.reconciliation)

            mask = batch.reconciliation.edges.edge_mask[0]
            count = int(mask.sum().item())
            example_id = int(example.example_id)
            rows = by_example[example_id].iloc[:count].copy()

            rows["continuation_logit"] = (
                output.continuation_logits[0, :count]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            rows["continuation_probability"] = (
                output.parental_probabilities[0, :count]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            rows["no_parent_probability"] = (
                output.no_parent_probability[0, :count]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            scored_parts.append(rows)

            if position % 20 == 0 or position == len(prepared):
                print(
                    f"[inference] {position:4d}/{len(prepared):4d} components",
                    flush=True,
                )

    scored = (
        pd.concat(scored_parts, ignore_index=True)
        if scored_parts
        else data.edges.iloc[0:0].copy()
    )

    if scored.empty:
        return scored

    scored["model_source_rank"] = 0
    scored["model_target_rank"] = 0
    scored["model_source_margin"] = np.nan
    scored["model_target_margin"] = np.nan

    # Global source rank matters because a source ending at frame t can be a
    # candidate for new tracks beginning at several future frames.
    for _, group in scored.groupby("source_original_track_id"):
        order = group.sort_values(
            ["continuation_probability", "continuation_logit"],
            ascending=False,
        )
        probabilities = order["continuation_probability"].to_numpy(dtype=float)
        for rank, dataframe_index in enumerate(order.index, start=1):
            scored.loc[dataframe_index, "model_source_rank"] = rank
            others = np.delete(
                group["continuation_probability"].to_numpy(dtype=float),
                np.flatnonzero(group.index.to_numpy() == dataframe_index)[0],
            )
            scored.loc[dataframe_index, "model_source_margin"] = (
                float(
                    scored.loc[dataframe_index, "continuation_probability"]
                    - others.max()
                )
                if others.size
                else np.nan
            )

    for _, group in scored.groupby("target_original_track_id"):
        order = group.sort_values(
            ["continuation_probability", "continuation_logit"],
            ascending=False,
        )
        for rank, dataframe_index in enumerate(order.index, start=1):
            scored.loc[dataframe_index, "model_target_rank"] = rank
            others = np.delete(
                group["continuation_probability"].to_numpy(dtype=float),
                np.flatnonzero(group.index.to_numpy() == dataframe_index)[0],
            )
            scored.loc[dataframe_index, "model_target_margin"] = (
                float(
                    scored.loc[dataframe_index, "continuation_probability"]
                    - others.max()
                )
                if others.size
                else np.nan
            )

    scored["model_mutual_best"] = (
        (scored["model_source_rank"].astype(int) == 1)
        & (scored["model_target_rank"].astype(int) == 1)
    )
    return scored


def select_accepted_links(
    scored: pd.DataFrame,
    *,
    probability_threshold: float,
    require_mutual_best: bool,
) -> pd.DataFrame:
    if scored.empty:
        result = scored.copy()
        result["accepted"] = pd.Series(dtype=bool)
        return result

    eligible = scored[
        scored["continuation_probability"].astype(float)
        >= float(probability_threshold)
    ].copy()
    if require_mutual_best:
        eligible = eligible[
            eligible["model_mutual_best"].astype(bool)
        ].copy()

    eligible = eligible.sort_values(
        ["continuation_probability", "continuation_logit"],
        ascending=False,
    )

    used_sources: set[int] = set()
    used_targets: set[int] = set()
    accepted_indices: list[int] = []

    for index, row in eligible.iterrows():
        source = int(row.source_original_track_id)
        target = int(row.target_original_track_id)
        if source in used_sources or target in used_targets:
            continue
        if int(row.target_start_frame) <= int(row.source_end_frame):
            continue

        accepted_indices.append(int(index))
        used_sources.add(source)
        used_targets.add(target)

    result = scored.copy()
    result["accepted"] = False
    if accepted_indices:
        result.loc[accepted_indices, "accepted"] = True

    return result


# =============================================================================
# Track-ID reconciliation
# =============================================================================


class UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def reconcile_track_ids(
    original_tracks: pd.DataFrame,
    accepted_links: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int]]:
    original_tracks = original_tracks.copy()
    original_tracks["track_id"] = pd.to_numeric(
        original_tracks["track_id"], errors="raise"
    ).astype(np.int64)
    original_tracks["frame"] = pd.to_numeric(
        original_tracks["frame"], errors="raise"
    ).astype(np.int64)

    track_ids = sorted(
        original_tracks["track_id"].astype(int).unique().tolist()
    )
    uf = UnionFind(track_ids)

    if not accepted_links.empty:
        ordered = accepted_links.sort_values(
            ["source_end_frame", "target_start_frame"]
        )
        for row in ordered.itertuples(index=False):
            uf.union(
                int(row.source_original_track_id),
                int(row.target_original_track_id),
            )

    summary = (
        original_tracks.groupby("track_id")
        .agg(first_frame=("frame", "min"))
        .reset_index()
    )
    first_frame = {
        int(row.track_id): int(row.first_frame)
        for row in summary.itertuples(index=False)
    }

    components: dict[int, list[int]] = defaultdict(list)
    for track_id in track_ids:
        components[uf.find(track_id)].append(track_id)

    mapping: dict[int, int] = {}
    for members in components.values():
        canonical = min(
            members,
            key=lambda tid: (first_frame[int(tid)], int(tid)),
        )
        for member in members:
            mapping[int(member)] = int(canonical)

    reconciled = original_tracks.copy()
    reconciled.insert(
        0,
        "original_track_id",
        reconciled["track_id"].astype(np.int64),
    )
    reconciled["track_id"] = (
        reconciled["track_id"].map(mapping).astype(np.int64)
    )

    duplicated = reconciled.duplicated(
        ["track_id", "frame"], keep=False
    )
    if duplicated.any():
        rows = reconciled.loc[
            duplicated,
            ["track_id", "frame", "original_track_id"],
        ].sort_values(["track_id", "frame"])
        raise RuntimeError(
            "Accepted links created overlapping observations in one "
            f"reconciled track:\n{rows.head(40)}"
        )

    return reconciled.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True), mapping


def remap_lineage_graph(
    graph_path: Path,
    mapping: dict[int, int],
) -> dict[str, Any]:
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("napari_graph.json must contain a dict")

    result: dict[str, Any] = {}
    for child, parent in payload.items():
        child_new = int(mapping.get(int(child), int(child)))
        if isinstance(parent, (list, tuple)):
            parents = [
                int(mapping.get(int(value), int(value)))
                for value in parent
            ]
            # Preserve deterministic uniqueness.
            parent_new: Any = sorted(set(parents))
        elif parent is None:
            parent_new = None
        else:
            parent_new = int(mapping.get(int(parent), int(parent)))
        result[str(child_new)] = parent_new
    return result


# =============================================================================
# Stage-09 endpoint comparison
# =============================================================================


def filtered_endpoint_groups(
    endpoint_module,
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    spatial_shape_zyx: tuple[int, int, int],
    *,
    spacing: tuple[float, float, float],
    boundary_margin_um: float,
    excluded_track_ids: set[int],
):
    groups = endpoint_module.prepare_endpoint_track_groups(
        tracks,
        cells,
        spatial_shape_zyx,
        voxel_size_zyx=spacing,
        boundary_margin_um=float(boundary_margin_um),
    )

    def filt(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "track_id" not in frame:
            return frame.copy()
        return frame.loc[
            ~pd.to_numeric(frame["track_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .isin(excluded_track_ids)
        ].copy()

    return endpoint_module.EndpointTrackGroups(
        track_summary=groups.track_summary.copy(),
        new_track_endpoints=filt(groups.new_track_endpoints),
        ended_track_endpoints=filt(groups.ended_track_endpoints),
        new_failure_tracks=filt(groups.new_failure_tracks),
        ended_failure_tracks=filt(groups.ended_failure_tracks),
        boundary_entry_tracks=filt(groups.boundary_entry_tracks),
        boundary_exit_tracks=filt(groups.boundary_exit_tracks),
    )


def endpoint_keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
    if frame.empty:
        return set()
    return {
        (int(row.frame), int(row.cell_id))
        for row in frame.itertuples(index=False)
    }


def annotate_accepted_endpoint_removal(
    accepted: pd.DataFrame,
    *,
    corrected_ended: pd.DataFrame,
    corrected_new: pd.DataFrame,
) -> pd.DataFrame:
    if accepted.empty:
        return accepted.copy()

    ended_keys = endpoint_keys(corrected_ended)
    new_keys = endpoint_keys(corrected_new)

    result = accepted.copy()
    result["source_end_removed"] = [
        (int(row.source_end_frame), int(row.source_cell_id)) in ended_keys
        if hasattr(row, "source_cell_id")
        else False
        for row in result.itertuples(index=False)
    ]
    result["target_start_removed"] = [
        (int(row.target_start_frame), int(row.target_cell_id)) in new_keys
        if hasattr(row, "target_cell_id")
        else False
        for row in result.itertuples(index=False)
    ]
    return result


def attach_endpoint_cell_ids(
    scored: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    starts, ends, _ = endpoint_rows(observations)
    result = scored.copy()
    result["source_cell_id"] = [
        int(ends[int(track_id)].cell_id)
        for track_id in result["source_original_track_id"].astype(int)
    ]
    result["target_cell_id"] = [
        int(starts[int(track_id)].cell_id)
        for track_id in result["target_original_track_id"].astype(int)
    ]
    return result


def build_still_broken_cases(
    after_groups,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    best_source = {}
    best_target = {}
    if not scored.empty:
        for track_id, group in scored.groupby("source_original_track_id"):
            best_source[int(track_id)] = group.sort_values(
                "continuation_probability", ascending=False
            ).iloc[0]
        for track_id, group in scored.groupby("target_original_track_id"):
            best_target[int(track_id)] = group.sort_values(
                "continuation_probability", ascending=False
            ).iloc[0]

    for event_type, endpoints, lookup in (
        ("ended", after_groups.ended_track_endpoints, best_source),
        ("new", after_groups.new_track_endpoints, best_target),
    ):
        if endpoints.empty:
            continue
        failure = endpoints.loc[
            ~endpoints["is_boundary_endpoint"].astype(bool)
        ]
        for endpoint in failure.itertuples(index=False):
            original_id = int(
                getattr(endpoint, "original_track_id", endpoint.track_id)
            )
            best = lookup.get(original_id)
            rows.append(
                {
                    "event_type": event_type,
                    "frame": int(endpoint.frame),
                    "track_id": int(endpoint.track_id),
                    "original_track_id": int(original_id),
                    "cell_id": int(endpoint.cell_id),
                    "z": float(endpoint.z),
                    "y": float(endpoint.y),
                    "x": float(endpoint.x),
                    "best_candidate_probability": (
                        float(best.continuation_probability)
                        if best is not None
                        else np.nan
                    ),
                    "best_candidate_other_track_id": (
                        int(
                            best.target_original_track_id
                            if event_type == "ended"
                            else best.source_original_track_id
                        )
                        if best is not None
                        else -1
                    ),
                    "best_candidate_gap_frames": (
                        int(best.gap_frames)
                        if best is not None
                        else -1
                    ),
                    "candidate_available": bool(best is not None),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================


def points_array(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 4), dtype=float)
    return frame[["frame", "z", "y", "x"]].to_numpy(dtype=float)


def accepted_link_tracks(
    accepted: pd.DataFrame,
    observations: pd.DataFrame,
) -> np.ndarray:
    if accepted.empty:
        return np.empty((0, 5), dtype=float)

    starts, ends, _ = endpoint_rows(observations)
    rows: list[list[float]] = []
    for link_id, edge in enumerate(
        accepted.itertuples(index=False), start=1
    ):
        source = ends[int(edge.source_original_track_id)]
        target = starts[int(edge.target_original_track_id)]
        rows.append(
            [
                float(link_id),
                float(source.frame),
                float(source.z),
                float(source.y),
                float(source.x),
            ]
        )
        rows.append(
            [
                float(link_id),
                float(target.frame),
                float(target.z),
                float(target.y),
                float(target.x),
            ]
        )
    return np.asarray(rows, dtype=float)


def open_viewer(
    *,
    sample_id: str,
    source,
    output: Path,
    spacing: tuple[float, float, float],
    boundary_margin_um: float,
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for Investigation 06 visualization."
        ) from exc

    endpoint_module = importlib.import_module(
        "src.09_visualization.step02_endpoints"
    )
    napari_layers = importlib.import_module(
        "src.09_visualization.napari_layers"
    )

    raw = np.load(source.raw, mmap_mode="r", allow_pickle=False)
    preprocessed = np.load(
        source.preprocessed, mmap_mode="r", allow_pickle=False
    )
    final_instances = np.load(
        source.final_instances, mmap_mode="r", allow_pickle=False
    )
    cells = pd.read_csv(source.cells)
    original_tracks = pd.read_csv(source.tracks)
    reconciled = pd.read_csv(output / "reconciled_tracks.csv")
    accepted = pd.read_csv(output / "accepted_links.csv")
    corrected_ended = pd.read_csv(output / "corrected_ended_endpoints.csv")
    corrected_new = pd.read_csv(output / "corrected_new_endpoints.csv")
    still_ended_tracks = pd.read_csv(output / "still_broken_ended_tracks.csv")
    still_new_tracks = pd.read_csv(output / "still_broken_new_tracks.csv")

    # Diagnostic only: these IDs are still passed through the reconciler.
    lineage_ids = load_inv01().load_lineage_track_ids(source.napari_graph)
    lineage_tracks = original_tracks[
        original_tracks["track_id"].astype(int).isin(lineage_ids)
    ].copy()

    touched_original = set()
    if not accepted.empty:
        touched_original.update(
            accepted["source_original_track_id"].astype(int).tolist()
        )
        touched_original.update(
            accepted["target_original_track_id"].astype(int).tolist()
        )
    corrected_track_ids = set(
        reconciled.loc[
            reconciled["original_track_id"].astype(int).isin(touched_original),
            "track_id",
        ].astype(int)
    )
    corrected_tracks = reconciled[
        reconciled["track_id"].astype(int).isin(corrected_track_ids)
    ].copy()

    scale = (1.0, *spacing)
    viewer = napari.Viewer(ndisplay=3)

    low, high = np.percentile(np.asarray(raw), [1.0, 99.8])
    viewer.add_image(
        raw,
        name="Raw Volume",
        scale=scale,
        rendering="mip",
        colormap="gray",
        contrast_limits=[float(low), float(high)],
    )
    viewer.add_image(
        preprocessed,
        name="Preprocessed Volume",
        scale=scale,
        rendering="mip",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=False,
    )
    viewer.add_labels(
        final_instances,
        name="Spatial Final Instances",
        scale=scale,
        opacity=1.0,
        visible=False,
    )

    if not original_tracks.empty:
        original_layer = viewer.add_tracks(
            original_tracks[
                ["track_id", "frame", "z", "y", "x"]
            ].to_numpy(float),
            name="CONTEXT - Original Trackastra Tracks",
            scale=scale,
            tail_length=20,
        )
        original_layer.visible = False

    if not reconciled.empty:
        reconciled_layer = viewer.add_tracks(
            reconciled[
                ["track_id", "frame", "z", "y", "x"]
            ].to_numpy(float),
            name="CONTEXT - All Reconciled Tracks",
            scale=scale,
            tail_length=20,
        )
        reconciled_layer.visible = False

    if not corrected_tracks.empty:
        corrected_layer = viewer.add_tracks(
            corrected_tracks[
                ["track_id", "frame", "z", "y", "x"]
            ].to_numpy(float),
            name="CORRECTED - Reconciled Trajectories",
            scale=scale,
            tail_length=20,
        )
        corrected_layer.visible = True

    accepted_track_data = accepted_link_tracks(
        accepted,
        original_tracks,
    )
    if len(accepted_track_data):
        accepted_layer = viewer.add_tracks(
            accepted_track_data,
            name="CORRECTED - Accepted Links",
            scale=scale,
            tail_length=20,
        )
        accepted_layer.visible = True

    if not corrected_ended.empty:
        viewer.add_points(
            points_array(corrected_ended),
            name="CORRECTED - Removed Ends",
            scale=scale,
            size=6,
            face_color="lime",
            properties={
                "track_id": corrected_ended["track_id"].to_numpy(),
                "cell_id": corrected_ended["cell_id"].to_numpy(),
            },
        )

    if not corrected_new.empty:
        viewer.add_points(
            points_array(corrected_new),
            name="CORRECTED - Removed Starts",
            scale=scale,
            size=6,
            face_color="green",
            properties={
                "track_id": corrected_new["track_id"].to_numpy(),
                "cell_id": corrected_new["cell_id"].to_numpy(),
            },
        )

    napari_layers.add_track_group(
        viewer,
        still_ended_tracks,
        track_name="STILL BROKEN - Ended Tracks",
        point_name="STILL BROKEN - Ended Centroids",
        color="red",
        scale=scale,
        visible=True,
    )
    napari_layers.add_track_group(
        viewer,
        still_new_tracks,
        track_name="STILL BROKEN - New Tracks",
        point_name="STILL BROKEN - New Centroids",
        color="yellow",
        scale=scale,
        visible=True,
    )

    if not lineage_tracks.empty:
        lineage_layer = viewer.add_tracks(
            lineage_tracks[
                ["track_id", "frame", "z", "y", "x"]
            ].to_numpy(float),
            name="FLAGGED - Trackastra Lineage Graph Tracks",
            scale=scale,
            tail_length=20,
        )
        lineage_layer.visible = False

    # Boundary layers are recomputed from the reconciled result and hidden.
    after_groups = endpoint_module.prepare_endpoint_track_groups(
        reconciled,
        cells,
        raw.shape[-3:],
        voxel_size_zyx=spacing,
        boundary_margin_um=boundary_margin_um,
    )
    napari_layers.add_track_group(
        viewer,
        after_groups.boundary_entry_tracks,
        track_name="EXCLUDED - Boundary Entry Tracks",
        point_name="EXCLUDED - Boundary Entry Centroids",
        color="cyan",
        scale=scale,
        visible=False,
    )
    napari_layers.add_track_group(
        viewer,
        after_groups.boundary_exit_tracks,
        track_name="EXCLUDED - Boundary Exit Tracks",
        point_name="EXCLUDED - Boundary Exit Centroids",
        color="orange",
        scale=scale,
        visible=False,
    )

    focus_frames: list[int] = []
    if not corrected_new.empty:
        focus_frames.extend(corrected_new["frame"].astype(int).tolist())
    if not corrected_ended.empty:
        focus_frames.extend(corrected_ended["frame"].astype(int).tolist())
    if not focus_frames and not still_new_tracks.empty:
        focus_frames.extend(still_new_tracks["frame"].astype(int).tolist())

    if focus_frames:
        try:
            viewer.dims.set_current_step(0, int(min(focus_frames)))
        except Exception:
            pass

    print("", flush=True)
    print("=" * 118, flush=True)
    print("INVESTIGATION 06 — NOTEBOOK-09 STYLE REAL BREAK VIEWER", flush=True)
    print("=" * 118, flush=True)
    print(
        "Green CORRECTED layers = accepted learned reconciliation effects.",
        flush=True,
    )
    print(
        "Red/yellow STILL BROKEN layers = interior endpoints that remain.",
        flush=True,
    )
    print(
        "These categories are reconciliation state, not manual tracking ground truth.",
        flush=True,
    )
    print("=" * 118, flush=True)

    napari.run()


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the learned continuation reconciler to real Trackastra "
            "breaks from Investigation 36 and visualize Stage-09-style "
            "before/after endpoint changes."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--source", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)

    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument(
        "--boundary-margin-um",
        type=float,
        default=DEFAULT_BOUNDARY_MARGIN_UM,
    )
    parser.add_argument(
        "--source-history",
        type=int,
        default=DEFAULT_SOURCE_HISTORY,
    )
    parser.add_argument(
        "--target-future",
        type=int,
        default=DEFAULT_TARGET_FUTURE,
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--allow-non-mutual",
        action="store_true",
        help=(
            "Allow threshold-passing edges that are not the learned model's "
            "best edge for both source and target. Not recommended initially."
        ),
    )
    parser.add_argument(
        "--max-component-edges",
        type=int,
        default=DEFAULT_MAX_COMPONENT_EDGES,
    )
    parser.add_argument("--crop-shape", default=DEFAULT_CROP_SHAPE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        choices=("auto", "off", "fp16", "bf16"),
        default="off",
    )
    parser.add_argument("--rebuild-crops", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Open previously generated Investigation-06 outputs.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= int(args.max_gap) <= 12:
        raise ValueError("--max-gap must be in [1,12]")
    if int(args.source_history) < 1:
        raise ValueError("--source-history must be >=1")
    if int(args.target_future) < 1:
        raise ValueError("--target-future must be >=1")
    if int(args.max_component_edges) < 1:
        raise ValueError("--max-component-edges must be >=1")
    if not 0.0 < float(args.prob_threshold) < 1.0:
        raise ValueError("--prob-threshold must be between 0 and 1")
    if float(args.boundary_margin_um) < 0:
        raise ValueError("--boundary-margin-um must be >=0")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    source_root = (
        resolve(args.source)
        if args.source is not None
        else default_source(args.sample_id)
    )
    checkpoint_path = (
        resolve(args.checkpoint)
        if args.checkpoint is not None
        else default_checkpoint(args.sample_id)
    )
    output = (
        resolve(args.output)
        if args.output is not None
        else default_output(args.sample_id)
    )

    inv01 = load_inv01()
    source = inv01.SourcePaths(source_root)
    source.validate()

    spacing = tuple(float(value) for value in DEFAULT_SPACING_ZYX_UM)

    if args.viewer_only:
        required = (
            output / "reconciled_tracks.csv",
            output / "accepted_links.csv",
            output / "corrected_ended_endpoints.csv",
            output / "corrected_new_endpoints.csv",
            output / "still_broken_ended_tracks.csv",
            output / "still_broken_new_tracks.csv",
            output / "summary.json",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Investigation-06 output is incomplete:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        open_viewer(
            sample_id=str(args.sample_id),
            source=source,
            output=output,
            spacing=spacing,
            boundary_margin_um=float(args.boundary_margin_um),
        )
        return 0

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Pass --overwrite to rebuild, or --viewer-only to inspect it."
        )
    output.mkdir(parents=True, exist_ok=True)

    trainer = load_inv03()
    inv04 = load_inv04()
    inv05 = load_inv05()
    inv04.patch_inv03(trainer)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_mode = trainer.resolved_amp_mode(device, str(args.amp))

    print("=" * 126, flush=True)
    print(
        "TRACK RECONCILER — INVESTIGATION 06: REAL TRACKASTRA CONTINUATION RECONCILIATION",
        flush=True,
    )
    print("=" * 126, flush=True)
    print(f"sample                   : {args.sample_id}", flush=True)
    print(f"source Investigation 36  : {source.root}", flush=True)
    print(f"checkpoint               : {checkpoint_path}", flush=True)
    print(f"output                   : {output}", flush=True)
    print(f"max gap                  : {args.max_gap}", flush=True)
    print(f"probability threshold    : {args.prob_threshold:.3f}", flush=True)
    print(
        "mutual-best required     : "
        f"{not bool(args.allow_non_mutual)}",
        flush=True,
    )
    print(
        "manual tracking GT       : unavailable; viewer inspection required",
        flush=True,
    )
    print("=" * 126, flush=True)

    started = time.perf_counter()

    observations, lineage_ids, movie_shape = inv01.load_observations(
        source,
        spacing=spacing,
    )
    global_motion = inv01.estimate_global_motion(
        observations,
        lineage_ids=lineage_ids,
    )

    # Candidate discovery and final visualization use the exact same Stage-09
    # bbox-aware boundary classification.
    endpoint_module = importlib.import_module(
        "src.09_visualization.step02_endpoints"
    )
    original_tracks_for_endpoints = pd.read_csv(source.tracks)
    cells_for_endpoints = pd.read_csv(source.cells)
    stage9_original_groups = endpoint_module.prepare_endpoint_track_groups(
        original_tracks_for_endpoints,
        cells_for_endpoints,
        movie_shape[-3:],
        voxel_size_zyx=spacing,
        boundary_margin_um=float(args.boundary_margin_um),
    )
    stage9_source_ids = set(
        stage9_original_groups.ended_failure_tracks["track_id"]
        .astype(int)
        .tolist()
    )
    stage9_target_ids = set(
        stage9_original_groups.new_failure_tracks["track_id"]
        .astype(int)
        .tolist()
    )

    raw_candidates, discovery_stats = discover_raw_candidates(
        observations,
        lineage_ids=lineage_ids,
        movie_frames=int(movie_shape[0]),
        max_gap=int(args.max_gap),
        boundary_margin_um=float(args.boundary_margin_um),
        stage9_source_ids=stage9_source_ids,
        stage9_target_ids=stage9_target_ids,
    )
    components, oversized = split_components(
        raw_candidates,
        max_component_edges=int(args.max_component_edges),
    )

    print(
        "[candidates] "
        f"ended={discovery_stats['interior_ended_tracks']} "
        f"new={discovery_stats['interior_new_tracks']} "
        f"edges={len(raw_candidates)} components={len(components)} "
        f"oversized_skipped={len(oversized)}",
        flush=True,
    )

    dataset_dir = output / "real_dataset"
    examples, tracklets, real_observations, edges = materialize_real_dataset(
        components,
        observations=observations,
        global_motion=global_motion,
        source_history=int(args.source_history),
        target_future=int(args.target_future),
        lineage_ids=lineage_ids,
    )
    write_real_dataset(
        dataset_dir,
        sample_id=str(args.sample_id),
        source=source,
        movie_shape=movie_shape,
        spacing=spacing,
        examples=examples,
        tracklets=tracklets,
        observations=real_observations,
        edges=edges,
        global_motion=global_motion,
        max_gap=int(args.max_gap),
        source_history=int(args.source_history),
        target_future=int(args.target_future),
    )

    if oversized:
        oversized_rows = []
        for index, component in enumerate(oversized):
            oversized_rows.append(
                {
                    "component_index": int(index),
                    "target_start_frame": int(component.target_start_frame),
                    "source_count": int(len(component.sources)),
                    "target_count": int(len(component.targets)),
                    "edge_count": int(len(component.edges)),
                    "source_track_ids": json.dumps(list(component.sources)),
                    "target_track_ids": json.dumps(list(component.targets)),
                }
            )
        atomic_csv(
            output / "oversized_skipped_components.csv",
            pd.DataFrame(oversized_rows),
        )
    else:
        atomic_csv(
            output / "oversized_skipped_components.csv",
            pd.DataFrame(
                columns=[
                    "component_index",
                    "target_start_frame",
                    "source_count",
                    "target_count",
                    "edge_count",
                    "source_track_ids",
                    "target_track_ids",
                ]
            ),
        )

    scored = edges.copy()
    if not edges.empty:
        data = trainer.SourceDataset.load(dataset_dir)
        crop_shape = parse_triplet_int(
            args.crop_shape,
            name="--crop-shape",
        )
        crop_cache = trainer.build_crop_cache(
            data,
            cache_dir=output / "cache",
            shape_zyx=crop_shape,
            distance_clip_um=float(DEFAULT_DISTANCE_CLIP_UM),
            rebuild=bool(args.rebuild_crops),
        )

        model, stats, train_global_median_volume, checkpoint = load_model(
            trainer,
            checkpoint_path,
            device=device,
        )

        print(
            "[features] applying Investigation-05 TRAIN-ONLY normalization "
            "statistics to real Trackastra candidates ...",
            flush=True,
        )
        prepared, _same_stats, _ = inv05.prepare_examples_with_stats(
            trainer,
            data,
            crop_cache=crop_cache,
            appearance_mode="crops",
            stats=stats,
            global_median_volume=float(train_global_median_volume),
        )

        print(
            f"[model] components={len(prepared)} "
            f"trainable_parameters={trainer.trainable_parameter_count(model):,} "
            f"device={device} amp={amp_mode}",
            flush=True,
        )
        scored = infer_edges(
            trainer,
            model,
            prepared,
            data,
            device=device,
            amp_mode=amp_mode,
        )
    else:
        print(
            "[model] No candidate edges were found; skipping neural inference.",
            flush=True,
        )

    scored = attach_endpoint_cell_ids(scored, observations)
    scored = select_accepted_links(
        scored,
        probability_threshold=float(args.prob_threshold),
        require_mutual_best=not bool(args.allow_non_mutual),
    )
    atomic_csv(output / "candidate_edges_scored.csv", scored)

    accepted = (
        scored.loc[scored["accepted"].astype(bool)].copy()
        if "accepted" in scored
        else scored.iloc[0:0].copy()
    )

    original_tracks = pd.read_csv(source.tracks)
    cells = pd.read_csv(source.cells)
    reconciled, mapping = reconcile_track_ids(
        original_tracks,
        accepted,
    )
    atomic_csv(output / "reconciled_tracks.csv", reconciled)
    atomic_json(
        output / "track_id_mapping.json",
        {str(key): int(value) for key, value in sorted(mapping.items())},
    )
    atomic_json(
        output / "reconciled_napari_graph.json",
        remap_lineage_graph(source.napari_graph, mapping),
    )

    endpoint_module = importlib.import_module(
        "src.09_visualization.step02_endpoints"
    )
    comparison_module = importlib.import_module(
        "src.09_visualization.step05_comparison"
    )

    before_groups = filtered_endpoint_groups(
        endpoint_module,
        original_tracks,
        cells,
        movie_shape[-3:],
        spacing=spacing,
        boundary_margin_um=float(args.boundary_margin_um),
        excluded_track_ids=set(),
    )
    after_groups = filtered_endpoint_groups(
        endpoint_module,
        reconciled,
        cells,
        movie_shape[-3:],
        spacing=spacing,
        boundary_margin_um=float(args.boundary_margin_um),
        excluded_track_ids=set(),
    )

    before_snapshot = comparison_module.prepare_stage9_snapshot(
        before_groups,
        sample_id=str(args.sample_id),
        boundary_margin_um=float(args.boundary_margin_um),
        voxel_size_zyx=spacing,
        spatial_shape_zyx=movie_shape[-3:],
        stage8_metadata={
            "source": "Investigation 36 Trackastra",
            "investigation": SCRIPT_NAME,
        },
    )
    after_snapshot = comparison_module.prepare_stage9_snapshot(
        after_groups,
        sample_id=str(args.sample_id),
        boundary_margin_um=float(args.boundary_margin_um),
        voxel_size_zyx=spacing,
        spatial_shape_zyx=movie_shape[-3:],
        stage8_metadata={
            "source": "Investigation 06 reconciled tracks",
            "investigation": SCRIPT_NAME,
        },
    )

    comparison_module.save_stage9_snapshot(
        before_snapshot,
        output / "stage9_before",
        overwrite=True,
    )
    comparison_module.save_stage9_snapshot(
        after_snapshot,
        output / "stage9_after",
        overwrite=True,
    )
    comparison = comparison_module.compare_stage9_snapshots(
        before_snapshot,
        after_snapshot,
    )

    corrected_ended = comparison.removed_ended_endpoints.copy()
    corrected_new = comparison.removed_new_endpoints.copy()
    still_ended_endpoints = after_snapshot.ended_endpoints.copy()
    still_new_endpoints = after_snapshot.new_endpoints.copy()
    still_ended_tracks = after_snapshot.ended_tracks.copy()
    still_new_tracks = after_snapshot.new_tracks.copy()

    atomic_csv(
        output / "corrected_ended_endpoints.csv",
        corrected_ended,
    )
    atomic_csv(
        output / "corrected_new_endpoints.csv",
        corrected_new,
    )
    atomic_csv(
        output / "still_broken_ended_endpoints.csv",
        still_ended_endpoints,
    )
    atomic_csv(
        output / "still_broken_new_endpoints.csv",
        still_new_endpoints,
    )
    atomic_csv(
        output / "still_broken_ended_tracks.csv",
        still_ended_tracks,
    )
    atomic_csv(
        output / "still_broken_new_tracks.csv",
        still_new_tracks,
    )
    atomic_csv(
        output / "newly_introduced_ended_endpoints.csv",
        comparison.added_ended_endpoints,
    )
    atomic_csv(
        output / "newly_introduced_new_endpoints.csv",
        comparison.added_new_endpoints,
    )

    accepted = annotate_accepted_endpoint_removal(
        accepted,
        corrected_ended=corrected_ended,
        corrected_new=corrected_new,
    )
    atomic_csv(output / "accepted_links.csv", accepted)

    still_cases = build_still_broken_cases(
        after_groups,
        scored,
    )
    atomic_csv(output / "still_broken_cases.csv", still_cases)

    before_ended = int(before_snapshot.ended_tracks["track_id"].nunique())
    before_new = int(before_snapshot.new_tracks["track_id"].nunique())
    after_ended = int(after_snapshot.ended_tracks["track_id"].nunique())
    after_new = int(after_snapshot.new_tracks["track_id"].nunique())

    accepted_probability = (
        accepted["continuation_probability"].astype(float)
        if not accepted.empty
        else pd.Series(dtype=float)
    )
    rejected_probability = (
        scored.loc[
            ~scored["accepted"].astype(bool),
            "continuation_probability",
        ].astype(float)
        if not scored.empty and "accepted" in scored
        else pd.Series(dtype=float)
    )

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "sample_id": str(args.sample_id),
        "checkpoint": root_relative(checkpoint_path),
        "manual_tracking_ground_truth_available": False,
        "classification_semantics": {
            "reconciler_corrected": (
                "A previously non-boundary Trackastra endpoint "
                "was removed by an accepted learned continuation link."
            ),
            "still_broken": (
                "A non-boundary endpoint remains after applying "
                "accepted continuation links."
            ),
            "warning": (
                "Neither category is manual ground truth. Napari inspection "
                "is required before treating proposed links as confirmed."
            ),
        },
        "parameters": {
            "max_gap_frames": int(args.max_gap),
            "boundary_margin_um": float(args.boundary_margin_um),
            "source_history_observations": int(args.source_history),
            "target_future_observations": int(args.target_future),
            "probability_threshold": float(args.prob_threshold),
            "require_model_mutual_best": not bool(args.allow_non_mutual),
            "max_component_edges": int(args.max_component_edges),
            "crop_shape_zyx": list(
                parse_triplet_int(args.crop_shape, name="--crop-shape")
            ),
        },
        "discovery": {
            **discovery_stats,
            "raw_candidate_edges": int(len(raw_candidates)),
            "candidate_components": int(len(components)),
            "oversized_components_skipped": int(len(oversized)),
            "trackastra_lineage_flagged_track_ids": int(len(lineage_ids)),
            "lineage_flags_used_as_exclusion": False,
        },
        "model": {
            "scored_edges": int(len(scored)),
            "accepted_links": int(len(accepted)),
            "accepted_probability_mean": (
                float(accepted_probability.mean())
                if len(accepted_probability)
                else None
            ),
            "accepted_probability_min": (
                float(accepted_probability.min())
                if len(accepted_probability)
                else None
            ),
            "rejected_probability_max": (
                float(rejected_probability.max())
                if len(rejected_probability)
                else None
            ),
        },
        "stage9_before_after": {
            "before_ended_tracks": before_ended,
            "after_ended_tracks": after_ended,
            "corrected_ended_tracks": int(
                comparison.removed_ended_tracks["track_id"].nunique()
            ),
            "before_new_tracks": before_new,
            "after_new_tracks": after_new,
            "corrected_new_tracks": int(
                comparison.removed_new_tracks["track_id"].nunique()
            ),
            "corrected_ended_endpoints": int(len(corrected_ended)),
            "corrected_new_endpoints": int(len(corrected_new)),
            "still_broken_ended_endpoints": int(
                len(still_ended_endpoints)
            ),
            "still_broken_new_endpoints": int(
                len(still_new_endpoints)
            ),
            "newly_introduced_ended_endpoints": int(
                len(comparison.added_ended_endpoints)
            ),
            "newly_introduced_new_endpoints": int(
                len(comparison.added_new_endpoints)
            ),
        },
        "elapsed_seconds": float(elapsed),
        "outputs": {
            "candidate_edges_scored": "candidate_edges_scored.csv",
            "accepted_links": "accepted_links.csv",
            "reconciled_tracks": "reconciled_tracks.csv",
            "corrected_ended_endpoints": "corrected_ended_endpoints.csv",
            "corrected_new_endpoints": "corrected_new_endpoints.csv",
            "still_broken_cases": "still_broken_cases.csv",
            "stage9_before": "stage9_before",
            "stage9_after": "stage9_after",
        },
    }
    atomic_json(output / "summary.json", summary)

    print("", flush=True)
    print("=" * 126, flush=True)
    print("REAL TRACKASTRA CONTINUATION RECONCILIATION", flush=True)
    print("=" * 126, flush=True)
    print(
        f"original interior ended  : {before_ended}",
        flush=True,
    )
    print(
        f"original interior new    : {before_new}",
        flush=True,
    )
    print(
        f"candidate edges scored   : {len(scored)}",
        flush=True,
    )
    print(
        f"accepted learned links   : {len(accepted)}",
        flush=True,
    )
    print(
        f"corrected ended endpoints: {len(corrected_ended)}",
        flush=True,
    )
    print(
        f"corrected new endpoints  : {len(corrected_new)}",
        flush=True,
    )
    print(
        f"still broken ended       : {len(still_ended_endpoints)}",
        flush=True,
    )
    print(
        f"still broken new         : {len(still_new_endpoints)}",
        flush=True,
    )
    print(
        "new breaks introduced   : "
        f"{len(comparison.added_ended_endpoints) + len(comparison.added_new_endpoints)}",
        flush=True,
    )
    print(
        f"elapsed                  : {elapsed:.1f}s",
        flush=True,
    )
    print(f"output                   : {output}", flush=True)
    print("=" * 126, flush=True)
    print(
        "[interpretation] 'corrected' means removed by the reconciler, not "
        "manually verified. Inspect CORRECTED and STILL BROKEN layers in Napari.",
        flush=True,
    )

    if not args.no_viewer:
        open_viewer(
            sample_id=str(args.sample_id),
            source=source,
            output=output,
            spacing=spacing,
            boundary_margin_um=float(args.boundary_margin_um),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
