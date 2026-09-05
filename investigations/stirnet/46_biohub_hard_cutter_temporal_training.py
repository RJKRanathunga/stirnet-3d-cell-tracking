from __future__ import annotations

r"""
Investigation 46 — BioHub hard tracklet cutter + enriched temporal STIR-Net training.

Scientific question
-------------------
Investigation 45 showed three important facts on the curated BioHub volume:

1. 44/44 curated spatial merges retained a Trackastra parent and therefore did
   not appear as newborn tracks.
2. Global-motion compensation tightened accepted temporal continuations
   substantially.
3. Relative component volume was an exceptionally strong merge indicator, while
   parent-volume change, motion-prediction error, and nearby broken tracks also
   carried useful signal.

Investigation 46 tests the resulting architecture:

    frozen/current spatial instances
        -> provisional Trackastra graph
        -> global-motion estimate
        -> synchronous high-recall HARD TRACKLET CUTTER
        -> global-motion-compensated temporal coordinates
        -> explicit volume / cut / break / hypothesis features
        -> temporal STIR-Net fine-tuning
        -> split-only spatial correction

The cutter is NOT a final tracker. It is deliberately allowed to over-cut
because its output is only evidence for temporal STIR-Net. Final tracking is
still expected to be rebuilt after corrected spatial instances are produced.

Important cutter design
-----------------------
The cutter inherits only physical-consistency ideas from the legacy classical
tracker. It does NOT reuse Hungarian assignment.

For each accepted one-to-one Trackastra edge i(t-1) -> j(t), the cutter uses:

    current / local-median volume
    positive current / parent volume growth
    stabilized predecessor prediction error
    nearby PRE-EXISTING broken predecessor tracks
    rejected plausible Trackastra hypotheses, when Inv45 cache exists
    boundary-aware volume reliability

All cutter decisions are computed from a frozen copy of the original
provisional graph and are then applied synchronously. Cutter-created breaks
therefore cannot recursively trigger more cuts.

Temporal coordinates
--------------------
The temporal graph is expressed in the target frame's coordinate system:

    p_temporal(tau | target=t)
        = p_source(tau) - G_tau + G_t

where G is cumulative global motion.

At tau=t this equals the original source coordinate, so RAG/instance query
positions remain aligned while past/future temporal detections have global
motion removed.

Experiment-local feature injection
----------------------------------
Investigation 35's direct Trackastra adapter intentionally leaves heavyweight
morphology fields neutral. Investigation 46 reuses those previously-neutral
32-D node channels for explicit temporal-consistency evidence without changing
the production model tensor shapes:

    GX_LOG_VOLUME       frame-local log volume ratio
    GX_BBOX[0]          normalized local-volume anomaly
    GX_BBOX[1]          normalized positive parent growth
    GX_BBOX[2]          hard-cut incoming flag
    GX_PCA[0]           stabilized prediction error / dref
    GX_PCA[1]           nearby broken-track count
    GX_PCA[2]           normalized cutter score
    GX_ELONGATION       rejected plausible predecessor count
    GX_FLATNESS         plausible predecessor count
    GX_INTENSITY_MEAN   original accepted association score
    GX_INTENSITY_STD    hard-cut outgoing flag

This is experiment-local. Production graph semantics are not modified.

Training
--------
Investigation 46 V3 freezes the enriched temporal candidate. The V2 smoke run
showed that this candidate was already strong before fine-tuning and that even
small candidate updates reduced CUT/exact recovery.

V3 therefore trains only a small DISCRETE WRITE SELECTOR. For editable RAG
edges, frozen spatial and candidate predictions define:

    HELP:
        spatial wrong, candidate correct -> WRITE target = 1

    SUPPRESS:
        spatial correct, candidate wrong -> WRITE target = 0

    NEUTRAL:
        both correct or both wrong -> excluded from selector training

The selector is trained on balanced HELP/SUPPRESS minibatches. Its features are
the frozen RAG edge embedding, Investigation-42/V11 candidate scalars, and the
explicit physical evidence introduced by Investigation 46.

Final inference is discrete, not interpolated:

    SUPPRESS -> use spatial logit exactly
    WRITE    -> use candidate logit exactly

The write threshold is calibrated only on training-frame HELP/SUPPRESS examples,
maximizing HELP recall under a configurable harmful-write constraint. Validation
never participates in threshold calibration.

Typical command:

    python .\investigations\stirnet\46_biohub_hard_cutter_temporal_training_v3.py `
        --selector-steps 400 `
        --selector-lr 3e-4 `
        --selector-eval-every 50 `
        --selector-max-suppress-write-rate 0.10 `
        --print-every 25

Default initializer
-------------------
If it exists, this script automatically uses:

    runs/stirnet/investigations/
      42_biohub_curated_temporal_finetune/<volume>/best.pt

which is the useful step-300 temporal checkpoint from Investigation 42.
Otherwise it falls back to the spatial checkpoint resolved by curation.

Default spatial cache
---------------------
If available, the exact persisted-label Investigation-42 spatial cache is
reused automatically.

Additional commands
-------------------
Audit cutter only:

    python .\investigations\stirnet\46_biohub_hard_cutter_temporal_training_v3.py --audit-only

Short selector smoke run:

    python .\investigations\stirnet\46_biohub_hard_cutter_temporal_training_v3.py `
        --selector-steps 80 `
        --selector-eval-every 20 `
        --print-every 10

The 30-39 validation range is a DEVELOPMENT split already used repeatedly in
Investigations 42/45. Use a fresh untouched holdout before production claims.
"""

import argparse
import dataclasses
import gc
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


# =============================================================================
# Repository / dynamic investigation imports
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "dataset_curation").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV42 = load_module(
    ROOT / "investigations" / "stirnet" / "42_biohub_curated_temporal_finetune.py",
    "_inv46_inv42",
)
INV45 = load_module(
    ROOT / "investigations" / "stirnet" / "45_biohub_temporal_evidence_audit.py",
    "_inv46_inv45",
)
INV35 = INV42.INV35

SCRIPT_NAME = "46_biohub_hard_cutter_temporal_training_v3"
OBJECTIVE_VERSION = 3

DEFAULT_CUTTER_TARGET_CLEAN_BREAK = 0.08
DEFAULT_CUTTER_MIN_THRESHOLD = 2.0
DEFAULT_CUTTER_NEAR_RADIUS_DREF = 2.5
DEFAULT_CUTTER_BOUNDARY_VOLUME_SCALE = 0.12
DEFAULT_CUTTER_PLAUSIBLE_SCORE = 0.20

LEGACY_VOLUME_LOG_SCALE = float(np.log(1.25))

# Cutter score weights. Volume deliberately dominates; secondary evidence raises
# recall without making nearby broken tracks an independent cut trigger.
W_LOCAL_VOLUME = 1.25
W_PARENT_GROWTH = 1.00
W_PREDICTION_ERROR = 0.55
W_NEARBY_BROKEN = 0.75
W_REJECTED_HYPOTHESIS = 0.50

# Motion error below this level is treated as ordinary residual motion.
PREDICTION_ERROR_FLOOR_UM = 1.5
PREDICTION_ERROR_SCALE_UM = 1.5

STATE: "Inv46State | None" = None
ACTIVE_MODEL = None


# =============================================================================
# Generic helpers
# =============================================================================


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def atomic_json(path: Path, payload: Any) -> None:
    INV42.atomic_json(path, payload)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def torch_load(path: Path, map_location="cpu"):
    return INV42.torch_load(path, map_location=map_location)


def edge_score(data: dict[str, Any]) -> float:
    for key in ("weight", "score", "probability", "confidence"):
        value = data.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return 0.0


def safe_log_ratio(a: float, b: float) -> float:
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return 0.0
    if float(a) <= 0 or float(b) <= 0:
        return 0.0
    return float(math.log(float(a) / float(b)))


def equivalent_sphere_diameter_um(volume_um3: float) -> float:
    if not math.isfinite(volume_um3) or volume_um3 <= 0:
        return 1.0
    return 2.0 * ((3.0 * volume_um3) / (4.0 * math.pi)) ** (1.0 / 3.0)


def source_boundary_distance_um(
    coords_zyx: Sequence[float],
    *,
    shape_zyx: Sequence[int],
    spacing_zyx_um: Sequence[float],
) -> float:
    coords = np.asarray(coords_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    lower = coords * spacing
    upper = (shape - 1.0 - coords) * spacing
    return float(np.min(np.concatenate([lower, upper])))


def is_boundary(
    coords_zyx: Sequence[float],
    *,
    shape_zyx: Sequence[int],
    spacing_zyx_um: Sequence[float],
    margin_um: float = 4.0,
) -> bool:
    return source_boundary_distance_um(
        coords_zyx,
        shape_zyx=shape_zyx,
        spacing_zyx_um=spacing_zyx_um,
    ) <= float(margin_um)


def dataclass_replace_loss(base, *, total: Tensor):
    return dataclasses.replace(base, total=total)


# =============================================================================
# Experiment state
# =============================================================================


@dataclass
class NodeEvidence:
    frame: int
    cell_id: int
    volume_um3: float
    local_median_volume_um3: float
    local_log_ratio: float = 0.0
    parent_log_ratio: float = 0.0
    positive_parent_growth_units: float = 0.0
    prediction_error_um: float = 0.0
    nearby_broken_count: int = 0
    plausible_predecessor_count: int = 0
    rejected_plausible_count: int = 0
    incoming_association_score: float = 0.0
    cutter_score: float = 0.0
    hard_cut_incoming: bool = False
    hard_cut_outgoing: bool = False
    boundary: bool = False


@dataclass
class Inv46State:
    args: argparse.Namespace
    paths: Any
    spacing: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    motion: Any
    motion_metrics: dict[str, Any]
    node_evidence: dict[int, NodeEvidence]
    cutter_rows: pd.DataFrame
    cutter_threshold: float
    cutter_metrics: dict[str, Any]
    hypothesis_cache: Path | None
    hypothesis_used: bool


# =============================================================================
# Inv45 paths / motion / labels
# =============================================================================


def inv45_paths_from_inv42(paths, *, output: Path):
    trackastra_root = paths.preprocessed / "trackastra"
    record = INV45.BioHubCatalog(paths.data_root).get(
        paths.sample,
        split=paths.split,
    )
    inference_id = record.paths.inference_id()
    if inference_id is None:
        raise RuntimeError("Canonical inference has no inference_id")
    return INV45.Paths(
        sample=paths.sample,
        split=paths.split,
        annotation_set=paths.annotation_set,
        zarr=paths.zarr,
        final_instances=paths.base_instances,
        track_graph=paths.track_graph,
        trackastra_summary=trackastra_root / "summary.json",
        bootstrap_motion=trackastra_root / "bootstrap_motion.csv",
        instance_annotations=paths.instance_annotations,
        track_annotations=paths.track_annotations,
        output=output,
        frame_count=int(paths.frame_count),
        inference_id=str(inference_id),
    )


def load_hypothesis_index(
    sample: str,
    *,
    override: Path | None,
    plausible_score: float,
) -> tuple[dict[tuple[int, int], list[tuple[int, int, float]]], Path | None]:
    path = (
        resolve(override)
        if override is not None
        else (
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / "45_biohub_temporal_evidence_audit"
            / sample
            / "hypothesis_cache"
            / "edges.csv"
        ).resolve()
    )
    if not path.is_file():
        print(
            "[Inv46 hypotheses] Inv45 hypothesis cache not found; "
            "continuing without rejected-candidate counts.",
            flush=True,
        )
        return {}, None

    frame = pd.read_csv(path)
    required = {
        "source_frame",
        "source_label",
        "target_frame",
        "target_label",
        "score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Inv45 hypothesis cache {path} is missing columns {missing}"
        )

    index: dict[tuple[int, int], list[tuple[int, int, float]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        score = float(row.score)
        if score < float(plausible_score):
            continue
        index[(int(row.target_frame), int(row.target_label))].append(
            (int(row.source_frame), int(row.source_label), score)
        )
    for key in index:
        index[key].sort(key=lambda item: item[2], reverse=True)

    print(
        f"[Inv46 hypotheses] loaded {path} "
        f"targets={len(index):,} plausible_score>={float(plausible_score):.2f}",
        flush=True,
    )
    return dict(index), path


# =============================================================================
# Hard cutter feature construction
# =============================================================================


def canonical_node_maps(graph) -> tuple[
    dict[tuple[int, int], int],
    dict[int, list[int]],
]:
    by_cell: dict[tuple[int, int], int] = {}
    by_frame: dict[int, list[int]] = defaultdict(list)

    for node_id, data in graph.nodes(data=True):
        if "inv42_cell_id" not in data:
            continue
        t = int(data["time"])
        cell_id = int(data["inv42_cell_id"])
        key = (t, cell_id)
        previous = by_cell.get(key)
        if previous is not None and previous != int(node_id):
            raise RuntimeError(
                f"Multiple canonical Trackastra nodes resolve to {key}: "
                f"{previous}, {node_id}"
            )
        by_cell[key] = int(node_id)
        by_frame[t].append(int(node_id))

    return by_cell, dict(by_frame)


def node_source_coords(graph, node_id: int) -> np.ndarray:
    value = graph.nodes[int(node_id)].get("inv35_coords_zyx")
    if value is None:
        raise KeyError(f"Node {node_id} lacks inv35_coords_zyx after Inv42 audit")
    coords = np.asarray(value, dtype=np.float64).reshape(-1)
    if coords.size != 3 or not np.isfinite(coords).all():
        raise RuntimeError(f"Node {node_id} has invalid canonical coordinates")
    return coords


def node_volume_um3(
    graph,
    node_id: int,
    *,
    voxel_volume_um3: float,
) -> float:
    voxels = max(int(graph.nodes[int(node_id)].get("inv35_volume_voxels", 1)), 1)
    return float(voxels) * float(voxel_volume_um3)


def stable_position_um(
    graph,
    node_id: int,
    *,
    cumulative_motion_zyx: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> np.ndarray:
    t = int(graph.nodes[int(node_id)]["time"])
    return (
        node_source_coords(graph, node_id) - cumulative_motion_zyx[t]
    ) * spacing_zyx_um


def unique_previous_node(graph, node_id: int) -> int | None:
    t = int(graph.nodes[int(node_id)]["time"])
    predecessors = [
        int(v)
        for v in graph.predecessors(int(node_id))
        if int(graph.nodes[int(v)]["time"]) == t - 1
    ]
    if len(predecessors) != 1:
        return None
    previous = predecessors[0]
    if graph.out_degree(previous) != 1:
        return None
    return previous


def predicted_target_position_um(
    graph,
    source: int,
    *,
    cumulative_motion_zyx: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> tuple[np.ndarray, bool]:
    p1 = stable_position_um(
        graph,
        source,
        cumulative_motion_zyx=cumulative_motion_zyx,
        spacing_zyx_um=spacing_zyx_um,
    )
    previous = unique_previous_node(graph, source)
    if previous is None:
        return p1, False
    p0 = stable_position_um(
        graph,
        previous,
        cumulative_motion_zyx=cumulative_motion_zyx,
        spacing_zyx_um=spacing_zyx_um,
    )
    return p1 + (p1 - p0), True


def build_frame_volume_stats(
    graph,
    by_frame: dict[int, list[int]],
    *,
    voxel_volume_um3: float,
) -> dict[int, tuple[float, float]]:
    stats: dict[int, tuple[float, float]] = {}
    for t, node_ids in by_frame.items():
        volumes = np.asarray(
            [
                node_volume_um3(
                    graph,
                    node_id,
                    voxel_volume_um3=voxel_volume_um3,
                )
                for node_id in node_ids
            ],
            dtype=np.float64,
        )
        if volumes.size == 0:
            continue
        median = max(float(np.median(volumes)), 1.0e-6)
        dref = equivalent_sphere_diameter_um(median)
        stats[int(t)] = (median, dref)
    return stats


def preexisting_broken_nodes_by_frame(
    graph,
    by_frame: dict[int, list[int]],
    *,
    shape_zyx: Sequence[int],
    spacing_zyx_um: Sequence[float],
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for t, node_ids in by_frame.items():
        rows = []
        for node_id in node_ids:
            # Only true pre-existing Trackastra endpoints. Cutter-created
            # endpoints are never fed back into this list.
            if int(graph.out_degree(node_id)) != 0:
                continue
            if is_boundary(
                node_source_coords(graph, node_id),
                shape_zyx=shape_zyx,
                spacing_zyx_um=spacing_zyx_um,
            ):
                continue
            rows.append(int(node_id))
        out[int(t)] = rows
    return out


def cutter_score(
    *,
    local_log_ratio: float,
    parent_log_ratio: float,
    prediction_error_um: float,
    nearby_broken_count: int,
    rejected_plausible_count: int,
    boundary: bool,
    boundary_volume_scale: float,
) -> tuple[float, dict[str, float]]:
    local_units = max(0.0, float(local_log_ratio)) / LEGACY_VOLUME_LOG_SCALE
    growth_units = max(0.0, float(parent_log_ratio)) / LEGACY_VOLUME_LOG_SCALE
    motion_units = max(
        0.0,
        (float(prediction_error_um) - PREDICTION_ERROR_FLOOR_UM)
        / PREDICTION_ERROR_SCALE_UM,
    )
    broken_units = min(max(int(nearby_broken_count), 0), 3)
    rejected_units = min(max(int(rejected_plausible_count), 0), 3)

    volume_reliability = (
        float(boundary_volume_scale)
        if bool(boundary)
        else 1.0
    )

    local_term = (
        volume_reliability * W_LOCAL_VOLUME * local_units
    )
    growth_term = (
        volume_reliability * W_PARENT_GROWTH * growth_units
    )
    motion_term = W_PREDICTION_ERROR * motion_units
    broken_term = W_NEARBY_BROKEN * float(broken_units)
    rejected_term = W_REJECTED_HYPOTHESIS * float(rejected_units)

    score = (
        local_term
        + growth_term
        + motion_term
        + broken_term
        + rejected_term
    )
    return float(score), {
        "local_term": float(local_term),
        "growth_term": float(growth_term),
        "motion_term": float(motion_term),
        "broken_term": float(broken_term),
        "rejected_term": float(rejected_term),
    }


def build_cutter_rows(
    graph,
    *,
    classification: pd.DataFrame,
    motion,
    spacing: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
    near_radius_dref: float,
    boundary_volume_scale: float,
    hypothesis_index: dict[tuple[int, int], list[tuple[int, int, float]]],
) -> tuple[pd.DataFrame, dict[int, NodeEvidence]]:
    spacing_np = np.asarray(spacing, dtype=np.float64)
    cumulative = np.asarray(motion.cumulative_float_zyx, dtype=np.float64)
    voxel_volume = float(np.prod(spacing_np))

    by_cell, by_frame = canonical_node_maps(graph)
    frame_stats = build_frame_volume_stats(
        graph,
        by_frame,
        voxel_volume_um3=voxel_volume,
    )
    broken_by_frame = preexisting_broken_nodes_by_frame(
        graph,
        by_frame,
        shape_zyx=shape_zyx,
        spacing_zyx_um=spacing,
    )

    class_map = {
        (int(row.frame), int(row.instance_id)): str(row.target_class)
        for row in classification.itertuples(index=False)
    }

    node_evidence: dict[int, NodeEvidence] = {}
    for node_id, data in graph.nodes(data=True):
        if "inv42_cell_id" not in data:
            continue
        t = int(data["time"])
        cell_id = int(data["inv42_cell_id"])
        volume = node_volume_um3(
            graph,
            int(node_id),
            voxel_volume_um3=voxel_volume,
        )
        median_volume = frame_stats.get(t, (volume, 1.0))[0]
        node_evidence[int(node_id)] = NodeEvidence(
            frame=t,
            cell_id=cell_id,
            volume_um3=float(volume),
            local_median_volume_um3=float(median_volume),
            local_log_ratio=safe_log_ratio(volume, median_volume),
            boundary=is_boundary(
                node_source_coords(graph, int(node_id)),
                shape_zyx=shape_zyx,
                spacing_zyx_um=spacing,
            ),
        )

    rows: list[dict[str, Any]] = []
    original_indegree = {
        int(node_id): int(graph.in_degree(node_id))
        for node_id in graph.nodes
    }

    for source, target, edge_data in list(graph.edges(data=True)):
        source = int(source)
        target = int(target)
        if source not in node_evidence or target not in node_evidence:
            continue

        ts = int(graph.nodes[source]["time"])
        tt = int(graph.nodes[target]["time"])
        if tt != ts + 1:
            continue

        # Hard cutter acts only on the same one-to-one joins that define a
        # tracklet. Division/fusion-like branches are left for temporal reasoning.
        if int(graph.out_degree(source)) != 1 or int(original_indegree[target]) != 1:
            continue

        source_ev = node_evidence[source]
        target_ev = node_evidence[target]

        parent_log = safe_log_ratio(
            target_ev.volume_um3,
            source_ev.volume_um3,
        )
        target_ev.parent_log_ratio = float(parent_log)
        target_ev.positive_parent_growth_units = (
            max(0.0, parent_log) / LEGACY_VOLUME_LOG_SCALE
        )

        predicted, used_velocity = predicted_target_position_um(
            graph,
            source,
            cumulative_motion_zyx=cumulative,
            spacing_zyx_um=spacing_np,
        )
        target_position = stable_position_um(
            graph,
            target,
            cumulative_motion_zyx=cumulative,
            spacing_zyx_um=spacing_np,
        )
        prediction_error = float(np.linalg.norm(predicted - target_position))
        target_ev.prediction_error_um = prediction_error

        median_volume, dref = frame_stats.get(
            tt,
            (target_ev.local_median_volume_um3, 1.0),
        )
        near_radius_um = float(near_radius_dref) * float(dref)
        nearby_broken = 0
        for broken_node in broken_by_frame.get(ts, []):
            if broken_node == source:
                continue
            broken_position = stable_position_um(
                graph,
                broken_node,
                cumulative_motion_zyx=cumulative,
                spacing_zyx_um=spacing_np,
            )
            if float(np.linalg.norm(broken_position - target_position)) <= near_radius_um:
                nearby_broken += 1
        target_ev.nearby_broken_count = int(nearby_broken)

        plausible = list(
            hypothesis_index.get((tt, target_ev.cell_id), [])
        )
        rejected = [
            item
            for item in plausible
            if not (
                int(item[0]) == ts
                and int(item[1]) == source_ev.cell_id
            )
        ]
        target_ev.plausible_predecessor_count = int(len(plausible))
        target_ev.rejected_plausible_count = int(len(rejected))

        association_score = edge_score(dict(edge_data))
        target_ev.incoming_association_score = float(association_score)

        boundary_pair = bool(source_ev.boundary or target_ev.boundary)
        score, terms = cutter_score(
            local_log_ratio=target_ev.local_log_ratio,
            parent_log_ratio=parent_log,
            prediction_error_um=prediction_error,
            nearby_broken_count=nearby_broken,
            rejected_plausible_count=len(rejected),
            boundary=boundary_pair,
            boundary_volume_scale=boundary_volume_scale,
        )
        target_ev.cutter_score = max(target_ev.cutter_score, score)

        target_class = class_map.get((tt, target_ev.cell_id), "unreviewed")

        rows.append(
            {
                "source_node": source,
                "target_node": target,
                "source_frame": ts,
                "target_frame": tt,
                "source_cell_id": source_ev.cell_id,
                "target_cell_id": target_ev.cell_id,
                "target_class": target_class,
                "eligible_one_to_one": True,
                "source_boundary": bool(source_ev.boundary),
                "target_boundary": bool(target_ev.boundary),
                "boundary_pair": boundary_pair,
                "source_volume_um3": source_ev.volume_um3,
                "target_volume_um3": target_ev.volume_um3,
                "local_median_volume_um3": float(median_volume),
                "current_over_local_median": (
                    target_ev.volume_um3 / max(float(median_volume), 1.0e-6)
                ),
                "current_over_parent": (
                    target_ev.volume_um3 / max(source_ev.volume_um3, 1.0e-6)
                ),
                "local_log_ratio": target_ev.local_log_ratio,
                "parent_log_ratio": parent_log,
                "prediction_error_um": prediction_error,
                "prediction_used_velocity": bool(used_velocity),
                "near_radius_um": near_radius_um,
                "nearby_preexisting_broken_count": int(nearby_broken),
                "plausible_predecessor_count": int(len(plausible)),
                "rejected_plausible_count": int(len(rejected)),
                "association_score": float(association_score),
                "cutter_score": float(score),
                **terms,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "Hard cutter found zero canonical one-to-one Trackastra edges."
        )
    return frame, node_evidence


def choose_threshold(
    frame: pd.DataFrame,
    *,
    train_frames: set[int],
    target_clean_break_rate: float,
    minimum_threshold: float,
    override: float | None,
) -> tuple[float, dict[str, Any]]:
    labelled = frame[
        frame["target_frame"].isin(sorted(train_frames))
        & frame["target_class"].isin(["merge", "clean"])
    ].copy()
    merge = labelled[labelled["target_class"] == "merge"]
    clean = labelled[labelled["target_class"] == "clean"]

    if len(merge) == 0:
        raise RuntimeError(
            "No curated merge-retaining eligible edges occur in training frames."
        )
    if len(clean) == 0:
        raise RuntimeError("No trusted clean eligible edges occur in training frames.")

    def metrics(threshold: float) -> tuple[float, float, int, int]:
        merge_cut = int((merge["cutter_score"] >= threshold).sum())
        clean_cut = int((clean["cutter_score"] >= threshold).sum())
        return (
            merge_cut / max(len(merge), 1),
            clean_cut / max(len(clean), 1),
            merge_cut,
            clean_cut,
        )

    if override is not None:
        threshold = float(override)
        recall, clean_rate, merge_cut, clean_cut = metrics(threshold)
        return threshold, {
            "mode": "explicit_threshold",
            "threshold": threshold,
            "train_merge_edges": int(len(merge)),
            "train_clean_edges": int(len(clean)),
            "train_merge_break_recall": float(recall),
            "train_clean_break_rate": float(clean_rate),
            "train_merge_edges_cut": int(merge_cut),
            "train_clean_edges_cut": int(clean_cut),
        }

    scores = sorted(
        {
            float(v)
            for v in labelled["cutter_score"].tolist()
            if math.isfinite(float(v))
        },
        reverse=True,
    )
    if not scores:
        raise RuntimeError("No finite cutter scores available for calibration.")

    candidates = [
        max(float(minimum_threshold), scores[0] + 1.0e-6)
    ]
    candidates.extend(
        score
        for score in scores
        if score >= float(minimum_threshold)
    )

    feasible: list[tuple[float, float, float, int, int]] = []
    for threshold in candidates:
        recall, clean_rate, merge_cut, clean_cut = metrics(threshold)
        if clean_rate <= float(target_clean_break_rate) + 1.0e-12:
            feasible.append(
                (recall, -clean_rate, threshold, merge_cut, clean_cut)
            )

    if not feasible:
        threshold = max(float(minimum_threshold), scores[0] + 1.0e-6)
    else:
        # Primary: highest merge recall under the allowed clean over-cut rate.
        # Secondary: lower clean break rate. Tertiary: lower threshold.
        feasible.sort(
            key=lambda row: (row[0], row[1], -row[2]),
            reverse=True,
        )
        threshold = float(feasible[0][2])

    recall, clean_rate, merge_cut, clean_cut = metrics(threshold)
    return threshold, {
        "mode": "calibrated_on_train_only",
        "target_clean_break_rate": float(target_clean_break_rate),
        "minimum_threshold": float(minimum_threshold),
        "threshold": float(threshold),
        "train_merge_edges": int(len(merge)),
        "train_clean_edges": int(len(clean)),
        "train_merge_break_recall": float(recall),
        "train_clean_break_rate": float(clean_rate),
        "train_merge_edges_cut": int(merge_cut),
        "train_clean_edges_cut": int(clean_cut),
    }


def split_metrics(
    frame: pd.DataFrame,
    *,
    target_frames: set[int],
    threshold: float,
) -> dict[str, Any]:
    labelled = frame[
        frame["target_frame"].isin(sorted(target_frames))
        & frame["target_class"].isin(["merge", "clean"])
    ]
    merge = labelled[labelled["target_class"] == "merge"]
    clean = labelled[labelled["target_class"] == "clean"]

    merge_cut = int((merge["cutter_score"] >= threshold).sum())
    clean_cut = int((clean["cutter_score"] >= threshold).sum())

    return {
        "merge_edges": int(len(merge)),
        "merge_edges_cut": int(merge_cut),
        "merge_break_recall": float(
            merge_cut / max(len(merge), 1)
        ),
        "clean_edges": int(len(clean)),
        "clean_edges_cut": int(clean_cut),
        "clean_break_rate": float(
            clean_cut / max(len(clean), 1)
        ),
        "clean_keep_rate": float(
            1.0 - clean_cut / max(len(clean), 1)
        ),
    }


def component_newborn_metrics(
    graph_before,
    graph_after,
    *,
    classification: pd.DataFrame,
    frames: set[int],
) -> dict[str, Any]:
    def map_nodes(graph):
        out = {}
        for node_id, data in graph.nodes(data=True):
            if "inv42_cell_id" in data:
                out[(int(data["time"]), int(data["inv42_cell_id"]))] = int(node_id)
        return out

    before_nodes = map_nodes(graph_before)
    after_nodes = map_nodes(graph_after)
    merge_rows = classification[
        classification["frame"].isin(sorted(frames))
        & (classification["target_class"] == "merge")
    ]

    total = before_parent = after_newborn = missing = 0
    for row in merge_rows.itertuples(index=False):
        key = (int(row.frame), int(row.instance_id))
        before = before_nodes.get(key)
        after = after_nodes.get(key)
        if before is None or after is None:
            missing += 1
            continue
        total += 1
        before_parent += int(graph_before.in_degree(before) > 0)
        after_newborn += int(graph_after.in_degree(after) == 0)

    return {
        "merge_components_present": int(total),
        "merge_components_missing_graph_node": int(missing),
        "merge_retains_parent_before_rate": float(
            before_parent / max(total, 1)
        ),
        "merge_newborn_after_rate": float(
            after_newborn / max(total, 1)
        ),
        "merge_newborn_after_count": int(after_newborn),
    }


def apply_hard_cutter(
    graph,
    *,
    frame: pd.DataFrame,
    node_evidence: dict[int, NodeEvidence],
    threshold: float,
) -> tuple[Any, pd.DataFrame]:
    # The original graph is frozen for feature construction. Apply all decisions
    # at once to a copy so cutter-created breaks cannot cascade.
    result = graph.copy()
    decisions = frame.copy()
    decisions["hard_cut"] = (
        decisions["cutter_score"].to_numpy(np.float64) >= float(threshold)
    )

    cut_edges: list[tuple[int, int]] = []
    for row in decisions.itertuples(index=False):
        if not bool(row.hard_cut):
            continue
        source = int(row.source_node)
        target = int(row.target_node)
        if result.has_edge(source, target):
            cut_edges.append((source, target))
        if source in node_evidence:
            node_evidence[source].hard_cut_outgoing = True
        if target in node_evidence:
            node_evidence[target].hard_cut_incoming = True

    result.remove_edges_from(cut_edges)
    return result, decisions


# =============================================================================
# Experiment-local temporal input
# =============================================================================


def make_temporal_input_46(
    graph,
    *,
    runtime,
    paths,
    temporal_radius: int,
    spacing: Sequence[float],
    device: torch.device,
):
    global STATE
    if STATE is None:
        raise RuntimeError("Inv46 state is not initialized.")

    node_ids, available_offsets = INV35.selected_nodes(
        graph,
        runtime.t,
        paths.frame_count,
        temporal_radius,
    )
    if not node_ids:
        return INV35.direct_temporal_input(
            graph,
            target_t=runtime.t,
            frame_count=paths.frame_count,
            temporal_radius=int(temporal_radius),
            spacing=spacing,
            dref_um=float(runtime.dref_um),
            shape_zyx=runtime.target.shape,
            device=device,
        )

    cumulative = np.asarray(
        STATE.motion.cumulative_float_zyx,
        dtype=np.float64,
    )
    target_motion = cumulative[int(runtime.t)]

    # Temporarily express temporal detections in target-frame coordinates.
    saved_coords: dict[int, Any] = {}
    try:
        for node_id in node_ids:
            data = graph.nodes[int(node_id)]
            source_coords = np.asarray(
                data["inv35_coords_zyx"],
                dtype=np.float64,
            )
            t = int(data["time"])
            saved_coords[int(node_id)] = data["inv35_coords_zyx"]
            data["inv35_coords_zyx"] = (
                source_coords - cumulative[t] + target_motion
            ).astype(np.float32)

        temporal_input = INV35.direct_temporal_input(
            graph,
            target_t=runtime.t,
            frame_count=paths.frame_count,
            temporal_radius=int(temporal_radius),
            spacing=spacing,
            dref_um=float(runtime.dref_um),
            shape_zyx=runtime.target.shape,
            device=device,
        )
    finally:
        for node_id, value in saved_coords.items():
            graph.nodes[int(node_id)]["inv35_coords_zyx"] = value

    # Row order is guaranteed by INV35.selected_nodes/direct_temporal_input.
    graph_x = temporal_input.graph_x
    if int(graph_x.shape[0]) != len(node_ids):
        raise RuntimeError(
            "Inv46 temporal row contract mismatch: "
            f"{graph_x.shape[0]} != {len(node_ids)}"
        )

    spacing_np = np.asarray(spacing, dtype=np.float64)
    shape_zyx = tuple(int(v) for v in runtime.target.shape)
    dref = max(float(runtime.dref_um), 1.0e-6)

    raw_boundary: list[bool] = []
    times: list[int] = []

    for row, node_id in enumerate(node_ids):
        node_id = int(node_id)
        data = graph.nodes[node_id]
        evidence = STATE.node_evidence.get(node_id)
        t = int(data["time"])
        times.append(t - int(runtime.t))

        source_coords = node_source_coords(graph, node_id)
        boundary_distance = source_boundary_distance_um(
            source_coords,
            shape_zyx=shape_zyx,
            spacing_zyx_um=spacing,
        )
        boundary = bool(boundary_distance <= 4.0)
        raw_boundary.append(boundary)

        if evidence is not None:
            local_units = (
                evidence.local_log_ratio / LEGACY_VOLUME_LOG_SCALE
            )
            growth_units = evidence.positive_parent_growth_units

            graph_x[row, INV35.GX_LOG_VOLUME] = float(
                np.clip(evidence.local_log_ratio, -3.0, 3.0)
            )
            graph_x[row, INV35.GX_BBOX.start + 0] = float(
                np.clip(local_units / 5.0, -1.0, 1.0)
            )
            graph_x[row, INV35.GX_BBOX.start + 1] = float(
                np.clip(growth_units / 5.0, 0.0, 1.0)
            )
            graph_x[row, INV35.GX_BBOX.start + 2] = float(
                evidence.hard_cut_incoming
            )

            graph_x[row, INV35.GX_PCA.start + 0] = float(
                np.clip(evidence.prediction_error_um / (5.0 * dref), 0.0, 1.0)
            )
            graph_x[row, INV35.GX_PCA.start + 1] = float(
                min(evidence.nearby_broken_count, 3) / 3.0
            )
            threshold_scale = max(float(STATE.cutter_threshold), 1.0)
            graph_x[row, INV35.GX_PCA.start + 2] = float(
                np.tanh(evidence.cutter_score / threshold_scale)
            )

            graph_x[row, INV35.GX_ELONGATION] = float(
                min(evidence.rejected_plausible_count, 3) / 3.0
            )
            graph_x[row, INV35.GX_FLATNESS] = float(
                min(evidence.plausible_predecessor_count, 4) / 4.0
            )
            graph_x[row, INV35.GX_INTENSITY_MEAN] = float(
                np.clip(evidence.incoming_association_score, 0.0, 1.0)
            )
            graph_x[row, INV35.GX_INTENSITY_STD] = float(
                evidence.hard_cut_outgoing
            )

        # Correct source-volume boundary semantics after target-relative motion
        # compensation. Motion compensation must never move the biological FOV.
        graph_x[row, INV35.GX_VOLUME_BOUNDARY] = float(
            boundary_distance / dref
        )
        graph_x[row, INV35.GX_PATCH_BOUNDARY] = float(
            boundary_distance / dref
        )
        graph_x[row, INV35.GX_BOUNDARY] = float(boundary)

    # Recompute the tracklet boundary/start/end status from SOURCE coordinates.
    tracklet_id = temporal_input.tracklet_id.detach().cpu().numpy()
    times_np = np.asarray(times, dtype=np.int64)
    boundary_np = np.asarray(raw_boundary, dtype=bool)

    status = temporal_input.temporal_status
    if len(available_offsets):
        min_available = min(available_offsets)
        max_available = max(available_offsets)
        unique_tracklets = sorted(set(int(v) for v in tracklet_id.tolist()))
        for tracklet in unique_tracklets:
            rows = np.flatnonzero(tracklet_id == tracklet)
            if rows.size == 0:
                continue
            ts = times_np[rows]
            boundary = bool(boundary_np[rows].any())
            interior_start = bool(ts.min() > min_available and not boundary)
            interior_end = bool(ts.max() < max_available and not boundary)

            status[tracklet, 1] = float(interior_start)
            status[tracklet, 2] = float(interior_end)
            status[tracklet, 5] = float(boundary)

            for row in rows.tolist():
                graph_x[row, INV35.GX_INTERIOR_START] = float(
                    times_np[row] == ts.min() and interior_start
                )
                graph_x[row, INV35.GX_INTERIOR_END] = float(
                    times_np[row] == ts.max() and interior_end
                )

    return temporal_input


# =============================================================================
# Investigation 46 shared physical-selector feature helpers
# =============================================================================


ORIGINAL_PRINT_METRICS = INV42.print_temporal_metrics
ORIGINAL_GATE_SCALARS = INV42.selective_gate_scalar_features

PHYSICAL_GATE_DIM = 7

# Lazily populated CPU-side current-component evidence.
COMPONENT_PHYSICAL_CACHE: dict[int, np.ndarray] = {}
BASE_INSTANCE_MOVIE = None
CELL_EVIDENCE_BY_KEY: dict[tuple[int, int], NodeEvidence] | None = None


def _cell_evidence_index() -> dict[tuple[int, int], NodeEvidence]:
    global CELL_EVIDENCE_BY_KEY
    if CELL_EVIDENCE_BY_KEY is not None:
        return CELL_EVIDENCE_BY_KEY
    if STATE is None:
        raise RuntimeError("Inv46 state is not initialized")
    result: dict[tuple[int, int], NodeEvidence] = {}
    for evidence in STATE.node_evidence.values():
        result[(int(evidence.frame), int(evidence.cell_id))] = evidence
    CELL_EVIDENCE_BY_KEY = result
    return result


def _base_instance_movie():
    global BASE_INSTANCE_MOVIE
    if BASE_INSTANCE_MOVIE is None:
        if STATE is None:
            raise RuntimeError("Inv46 state is not initialized")
        BASE_INSTANCE_MOVIE = np.load(
            STATE.paths.base_instances,
            mmap_mode="r",
            allow_pickle=False,
        )
    return BASE_INSTANCE_MOVIE


def component_physical_features_46(runtime, case) -> Tensor:
    """Return [current_component, 7] normalized physical evidence."""

    t = int(runtime.t)
    cached = COMPONENT_PHYSICAL_CACHE.get(t)
    if cached is None:
        base_movie = _base_instance_movie()
        base_labels = np.asarray(base_movie[t])

        supervoxels = (
            runtime.rag.supervoxel_labels[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        node_sv = (
            runtime.rag.node_supervoxel_id
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        lookup = INV35.sv_label_lookup(
            supervoxels,
            base_labels,
            name=f"inv46 gate current identity t={t}",
        )
        node_cell = lookup[node_sv]

        current = (
            case.node_current_component
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        component_count = int(case.split_target.numel())
        rows = np.zeros((component_count, PHYSICAL_GATE_DIM), dtype=np.float32)
        evidence_index = _cell_evidence_index()

        for component in range(component_count):
            node_rows = np.flatnonzero(current == component)
            if node_rows.size == 0:
                continue

            ids = node_cell[node_rows]
            ids = ids[ids > 0]
            if ids.size == 0:
                continue

            unique, counts = np.unique(ids, return_counts=True)
            cell_id = int(unique[int(np.argmax(counts))])
            evidence = evidence_index.get((t, cell_id))
            if evidence is None:
                continue

            dref = max(float(runtime.dref_um), 1.0e-6)
            threshold = (
                max(float(STATE.cutter_threshold), 1.0e-6)
                if STATE is not None
                else 1.0
            )
            local_units = (
                float(evidence.local_log_ratio) / LEGACY_VOLUME_LOG_SCALE
            )

            rows[component] = np.asarray(
                [
                    np.clip(local_units / 5.0, -1.0, 1.0),
                    np.clip(
                        float(evidence.positive_parent_growth_units) / 5.0,
                        0.0,
                        1.0,
                    ),
                    float(evidence.hard_cut_incoming),
                    np.clip(
                        float(evidence.prediction_error_um) / (5.0 * dref),
                        0.0,
                        1.0,
                    ),
                    min(int(evidence.nearby_broken_count), 3) / 3.0,
                    math.tanh(float(evidence.cutter_score) / threshold),
                    min(int(evidence.rejected_plausible_count), 3) / 3.0,
                ],
                dtype=np.float32,
            )

        COMPONENT_PHYSICAL_CACHE[t] = rows
        cached = rows

    return torch.from_numpy(cached).to(
        device=case.rag.spatial_edge_logits.device,
        dtype=torch.float32,
    )


def physical_gate_edge_features_46(runtime, case) -> Tensor:
    if case.rag.edge_index.shape[1] == 0:
        return case.rag.spatial_edge_logits.new_zeros(
            (0, PHYSICAL_GATE_DIM),
            dtype=torch.float32,
        )

    component = case.node_current_component.long()
    component_features = component_physical_features_46(runtime, case)
    if component.numel() and int(component.max().item()) >= int(component_features.shape[0]):
        raise RuntimeError(
            "Inv46 current-component index exceeds physical-feature table: "
            f"max={int(component.max().item())} rows={int(component_features.shape[0])}"
        )
    src, dst = case.rag.edge_index
    left = component_features[component[src]]
    right = component_features[component[dst]]
    return 0.5 * (left + right)


def selective_gate_scalar_features_46(
    model,
    *,
    runtime,
    case,
    base_reasoning,
    candidate_logits,
) -> Tensor:
    original = ORIGINAL_GATE_SCALARS(
        model,
        case=case,
        base_reasoning=base_reasoning,
        candidate_logits=candidate_logits,
    )
    physical = physical_gate_edge_features_46(runtime, case)
    if original.shape[0] != physical.shape[0]:
        raise RuntimeError(
            "Inv46 gate scalar/physical edge row mismatch: "
            f"{original.shape[0]} != {physical.shape[0]}"
        )
    return torch.cat([original.float(), physical.float()], dim=-1)


# =============================================================================
# CLI / main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = INV42.build_parser()
    parser.description = (
        "Investigation 46 V3: hard tracklet cutter and frozen enriched temporal candidate."
    )
    parser.add_argument("--cutter-threshold", type=float, default=None)
    parser.add_argument(
        "--cutter-target-clean-break-rate",
        type=float,
        default=DEFAULT_CUTTER_TARGET_CLEAN_BREAK,
    )
    parser.add_argument(
        "--cutter-min-threshold",
        type=float,
        default=DEFAULT_CUTTER_MIN_THRESHOLD,
    )
    parser.add_argument(
        "--cutter-near-radius-dref",
        type=float,
        default=DEFAULT_CUTTER_NEAR_RADIUS_DREF,
    )
    parser.add_argument(
        "--cutter-boundary-volume-scale",
        type=float,
        default=DEFAULT_CUTTER_BOUNDARY_VOLUME_SCALE,
    )
    parser.add_argument(
        "--cutter-plausible-score",
        type=float,
        default=DEFAULT_CUTTER_PLAUSIBLE_SCORE,
    )
    parser.add_argument("--hypothesis-cache", type=Path, default=None)
    # Keep inherited Investigation-42 validator-compatible values. V3 does not
    # optimize the candidate/gate through those legacy flags.
    parser.set_defaults(steps=1, lr=1e-5, candidate_weight=0.0, split_weight=0.0)
    return parser


def validate_inv46_args(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.cutter_target_clean_break_rate) < 1.0:
        raise ValueError("--cutter-target-clean-break-rate must be in [0,1)")
    if float(args.cutter_min_threshold) < 0:
        raise ValueError("--cutter-min-threshold must be >= 0")
    if args.cutter_threshold is not None and float(args.cutter_threshold) < 0:
        raise ValueError("--cutter-threshold must be >= 0")
    if float(args.cutter_near_radius_dref) <= 0:
        raise ValueError("--cutter-near-radius-dref must be > 0")
    if not 0.0 <= float(args.cutter_boundary_volume_scale) <= 1.0:
        raise ValueError("--cutter-boundary-volume-scale must be in [0,1]")
    if not 0.0 <= float(args.cutter_plausible_score) <= 1.0:
        raise ValueError("--cutter-plausible-score must be in [0,1]")
    if args.gate_resume is not None:
        raise ValueError("--gate-resume is not used by Investigation 46 V3")


def auto_defaults(args: argparse.Namespace) -> None:
    if args.output is None:
        args.output = (
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / SCRIPT_NAME
            / str(args.sample_id)
        ).resolve()

    if args.spatial_cache_root is None:
        candidate = (
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / "42_biohub_curated_temporal_finetune"
            / str(args.sample_id)
            / "spatial_cache"
        ).resolve()
        if candidate.is_dir():
            args.spatial_cache_root = candidate
            print(
                f"[Inv46] reusing Investigation-42 spatial cache: {candidate}",
                flush=True,
            )

    if args.resume is None:
        candidate = (
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / "42_biohub_curated_temporal_finetune"
            / str(args.sample_id)
            / "best.pt"
        ).resolve()
        if candidate.is_file():
            args.resume = candidate
            print(
                f"[Inv46] temporal initializer: {candidate}",
                flush=True,
            )
        else:
            print(
                "[Inv46] Investigation-42 best.pt was not found; temporal modules "
                "will initialize from the curation spatial checkpoint.",
                flush=True,
            )


def write_manifest_46(
    paths,
    *,
    args,
    reviewed,
    train_frames,
    val_frames,
    spacing,
) -> None:
    payload = {
        "version": OBJECTIVE_VERSION,
        "investigation": SCRIPT_NAME,
        "sample_id": paths.sample,
        "split": paths.split,
        "annotation_set": paths.annotation_set,
        "reviewed_frames": list(map(int, reviewed)),
        "train_frames": list(map(int, train_frames)),
        "val_frames": list(map(int, val_frames)),
        "temporal_radius": int(args.temporal_radius),
        "spacing_zyx_um": list(map(float, spacing)),
        "initializer": str(args.resume) if args.resume is not None else str(paths.checkpoint),
        "spatial_cache": str(paths.spatial_cache),
        "track_graph": str(paths.track_graph),
        "cutter_contract": {
            "synchronous": True,
            "recursive_cascade": False,
            "one_to_one_tracklet_edges_only": True,
            "volume_scale_inherited_from_legacy": float(LEGACY_VOLUME_LOG_SCALE),
            "target_clean_break_rate": float(args.cutter_target_clean_break_rate),
            "explicit_threshold": (
                None if args.cutter_threshold is None else float(args.cutter_threshold)
            ),
            "boundary_volume_scale": float(args.cutter_boundary_volume_scale),
            "near_radius_dref": float(args.cutter_near_radius_dref),
        },
        "temporal_contract": {
            "target_relative_global_motion_compensation": True,
            "frame_local_volume_feature": True,
            "parent_growth_feature": True,
            "hard_cut_flags": True,
            "nearby_preexisting_break_feature": True,
            "inv45_hypothesis_counts_when_available": True,
            "split_only": True,
            "manual_tracking_overrides": False,
        },
        "training_contract": {
            "two_stage": True,
            "phase_a": {
                "candidate_only": True,
                "temporal_encoder_trainable": True,
                "temporal_reasoner_trainable_except_old_gate": True,
                "selective_gate_frozen": True,
                "steps": int(args.candidate_steps),
                "lr": float(args.candidate_lr),
                "candidate_keep_weight": float(args.candidate_preservation_weight),
                "split_weight": float(args.split_weight),
                "baseline_is_selectable": True,
            },
            "phase_b": {
                "candidate_frozen": True,
                "old_four_scalar_gate_frozen": True,
                "enriched_selective_gate_trainable": True,
                "physical_gate_feature_dim": int(PHYSICAL_GATE_DIM),
                "steps": int(args.gate_steps),
                "lr": float(args.gate_lr),
                "final_loss_weight": float(args.gate_final_weight),
                "anchor_weight": float(args.gate_anchor_weight),
                "candidate_invariant": True,
            },
            "spatial_frozen": True,
            "instance_tokenizer_frozen": True,
        },
    }
    atomic_json(paths.output / "dataset_manifest.json", payload)



# =============================================================================
# Investigation 46 V3 — frozen candidate + discrete HELP/SUPPRESS selector
# =============================================================================


class Inv46DiscreteWriteSelector(torch.nn.Module):
    """Small classifier deciding whether to use candidate or spatial exactly."""

    ORIGINAL_SCALAR_DIM = 9
    SCALAR_DIM = ORIGINAL_SCALAR_DIM + PHYSICAL_GATE_DIM

    def __init__(self, edge_embedding_dim: int) -> None:
        super().__init__()
        edge_embedding_dim = int(edge_embedding_dim)
        if edge_embedding_dim <= 0:
            raise ValueError("edge_embedding_dim must be positive")
        self.edge_embedding_dim = edge_embedding_dim
        self.edge_norm = torch.nn.LayerNorm(edge_embedding_dim)
        self.scalar_norm = torch.nn.LayerNorm(self.SCALAR_DIM)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(edge_embedding_dim + self.SCALAR_DIM, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, *, edge_embedding: Tensor, scalar_features: Tensor) -> Tensor:
        if edge_embedding.ndim != 2 or scalar_features.ndim != 2:
            raise ValueError("selector inputs must be [E,D] and [E,F]")
        if edge_embedding.shape[0] != scalar_features.shape[0]:
            raise ValueError("selector row mismatch")
        if int(edge_embedding.shape[1]) != self.edge_embedding_dim:
            raise ValueError("selector edge embedding dimension mismatch")
        if int(scalar_features.shape[1]) != self.SCALAR_DIM:
            raise ValueError(
                f"selector scalar dim={scalar_features.shape[1]} expected={self.SCALAR_DIM}"
            )
        edge = self.edge_norm(edge_embedding.float())
        scalar = self.scalar_norm(scalar_features.float())
        return self.net(torch.cat([edge, scalar], dim=-1)).squeeze(-1)


@dataclass
class SelectorDatasetV3:
    edge_embedding: Tensor
    scalar_features: Tensor
    target_write: Tensor
    metadata: pd.DataFrame

    @property
    def help_count(self) -> int:
        return int((self.target_write > 0.5).sum().item())

    @property
    def suppress_count(self) -> int:
        return int((self.target_write <= 0.5).sum().item())


def frozen_candidate_bundle_v3(model, *, runtime, case, encoded, temporal=None):
    state = encoded.temporal if temporal is None else temporal
    base_reasoning = model.instance_temporal(
        encoded.instances,
        case.rag,
        state,
        encoded.dref_t,
    )
    candidate_logits = case.rag.spatial_edge_logits + base_reasoning.edge_temporal_delta
    scalar_features = selective_gate_scalar_features_46(
        model,
        runtime=runtime,
        case=case,
        base_reasoning=base_reasoning,
        candidate_logits=candidate_logits,
    )
    return base_reasoning, candidate_logits, scalar_features


def help_suppress_masks_v3(model, case, candidate_logits):
    threshold = float(model.cfg.partition.final_merge_threshold)
    target_keep = case.target_keep.bool()
    spatial_pred = torch.sigmoid(case.rag.spatial_edge_logits) >= threshold
    candidate_pred = torch.sigmoid(candidate_logits) >= threshold
    spatial_correct = spatial_pred == target_keep
    candidate_correct = candidate_pred == target_keep
    help_mask = case.editable & candidate_correct & ~spatial_correct
    suppress_mask = case.editable & ~candidate_correct & spatial_correct
    return help_mask, suppress_mask


def reason_case_v3(
    model,
    selector: Inv46DiscreteWriteSelector,
    *,
    runtime,
    case,
    encoded,
    write_threshold: float,
    temporal=None,
):
    base_reasoning, candidate_logits, scalar_features = frozen_candidate_bundle_v3(
        model,
        runtime=runtime,
        case=case,
        encoded=encoded,
        temporal=temporal,
    )
    selector_logits = selector(
        edge_embedding=case.rag.edge_embeddings.detach(),
        scalar_features=scalar_features.detach(),
    )
    write_probability = torch.sigmoid(selector_logits)
    write_mask = case.editable & (write_probability >= float(write_threshold))

    # Central V3 change: exact discrete source selection, no interpolation.
    final_logits = torch.where(
        write_mask,
        candidate_logits,
        case.rag.spatial_edge_logits,
    )
    reasoning = dataclasses.replace(
        base_reasoning,
        edge_temporal_gate=write_probability.to(base_reasoning.edge_temporal_gate.dtype),
        final_edge_logits=final_logits,
    )
    return INV42.ReasonedCase(
        reasoning=reasoning,
        final_logits=final_logits,
        candidate_logits=candidate_logits,
        gate_logits=selector_logits,
        base_gate=base_reasoning.edge_temporal_gate.detach(),
    )


@torch.no_grad()
def extract_selector_dataset_v3(
    model,
    *,
    frames: Sequence[int],
    loader,
    graph,
    paths,
    temporal_radius: int,
    spacing: Sequence[float],
    device: torch.device,
) -> SelectorDatasetV3:
    model.eval()
    edge_chunks=[]; scalar_chunks=[]; target_chunks=[]; meta=[]
    for t in frames:
        runtime=loader.load(int(t))
        case=INV42.build_real_case(runtime)
        temporal_input=make_temporal_input_46(
            graph,
            runtime=runtime,
            paths=paths,
            temporal_radius=int(temporal_radius),
            spacing=spacing,
            device=device,
        )
        encoded=INV42.encode_case(
            model,
            runtime=runtime,
            case=case,
            temporal_input=temporal_input,
            spacing=spacing,
            device=device,
        )
        _, candidate_logits, scalar_features=frozen_candidate_bundle_v3(
            model,
            runtime=runtime,
            case=case,
            encoded=encoded,
        )
        help_mask,suppress_mask=help_suppress_masks_v3(model,case,candidate_logits)
        chosen=torch.nonzero(help_mask|suppress_mask,as_tuple=False).flatten()
        if chosen.numel()==0:
            continue
        labels=help_mask[chosen].float()
        edge_chunks.append(case.rag.edge_embeddings[chosen].detach().float().cpu())
        scalar_chunks.append(scalar_features[chosen].detach().float().cpu())
        target_chunks.append(labels.detach().float().cpu())
        phys=physical_gate_edge_features_46(runtime,case)[chosen].detach().float().cpu()
        spatial=case.rag.spatial_edge_logits[chosen].detach().float().cpu()
        candidate=candidate_logits[chosen].detach().float().cpu()
        for k,edge_row in enumerate(chosen.tolist()):
            meta.append({
                'frame':int(t),
                'edge_row':int(edge_row),
                'class':'HELP' if labels[k].item()>0.5 else 'SUPPRESS',
                'target_write':int(labels[k].item()>0.5),
                'spatial_logit':float(spatial[k].item()),
                'candidate_logit':float(candidate[k].item()),
                'local_volume_feature':float(phys[k,0].item()),
                'parent_growth_feature':float(phys[k,1].item()),
                'hard_cut_feature':float(phys[k,2].item()),
                'prediction_error_feature':float(phys[k,3].item()),
                'nearby_broken_feature':float(phys[k,4].item()),
                'cutter_score_feature':float(phys[k,5].item()),
                'rejected_predecessor_feature':float(phys[k,6].item()),
            })
    if not target_chunks:
        raise RuntimeError('No HELP/SUPPRESS selector examples found in training frames')
    ds=SelectorDatasetV3(
        edge_embedding=torch.cat(edge_chunks,dim=0),
        scalar_features=torch.cat(scalar_chunks,dim=0),
        target_write=torch.cat(target_chunks,dim=0),
        metadata=pd.DataFrame(meta),
    )
    if ds.help_count<=0 or ds.suppress_count<=0:
        raise RuntimeError(
            f'Selector needs both classes: HELP={ds.help_count} SUPPRESS={ds.suppress_count}'
        )
    return ds


def selector_auc_v3(probability: np.ndarray, target: np.ndarray) -> float:
    probability=np.asarray(probability,dtype=np.float64)
    target=np.asarray(target,dtype=np.int64)
    pos=probability[target==1]; neg=probability[target==0]
    if pos.size==0 or neg.size==0:
        return 0.5
    greater=float((pos[:,None]>neg[None,:]).sum())
    equal=float((pos[:,None]==neg[None,:]).sum())
    return (greater+0.5*equal)/float(pos.size*neg.size)


def selector_threshold_metrics_v3(probability,target,threshold:float)->dict[str,Any]:
    p=np.asarray(probability,dtype=np.float64)
    y=np.asarray(target,dtype=np.int64)
    write=p>=float(threshold)
    help_mask=y==1; suppress_mask=y==0
    help_recall=float(write[help_mask].mean()) if help_mask.any() else 0.0
    suppress_write=float(write[suppress_mask].mean()) if suppress_mask.any() else 0.0
    return {
        'threshold':float(threshold),
        'help_recall':help_recall,
        'suppress_write_rate':suppress_write,
        'balanced_accuracy':0.5*(help_recall+1.0-suppress_write),
        'auc':selector_auc_v3(p,y),
        'help_count':int(help_mask.sum()),
        'suppress_count':int(suppress_mask.sum()),
    }


def choose_selector_threshold_v3(probability,target,*,max_suppress_write_rate:float,override):
    p=np.asarray(probability,dtype=np.float64)
    y=np.asarray(target,dtype=np.int64)
    if override is not None:
        th=float(override)
        return th,selector_threshold_metrics_v3(p,y,th)
    candidates=[1.0+1e-7]+sorted(set(float(v) for v in p.tolist()),reverse=True)+[0.0]
    feasible=[]
    for th in candidates:
        m=selector_threshold_metrics_v3(p,y,th)
        if m['suppress_write_rate']<=float(max_suppress_write_rate)+1e-12:
            feasible.append((m['help_recall'],-m['suppress_write_rate'],float(th),m))
    if not feasible:
        th=1.0+1e-7
        return th,selector_threshold_metrics_v3(p,y,th)
    feasible.sort(key=lambda row:(row[0],row[1],row[2]),reverse=True)
    return float(feasible[0][2]),dict(feasible[0][3])


@torch.no_grad()
def selector_probabilities_v3(selector,dataset,device):
    selector.eval()
    logits=selector(
        edge_embedding=dataset.edge_embedding.to(device),
        scalar_features=dataset.scalar_features.to(device),
    )
    return torch.sigmoid(logits).detach().float().cpu().numpy().astype(np.float64,copy=False)


@torch.no_grad()
def evaluate_v3(
    model,selector,*,frames,loader,graph,paths,temporal_radius,spacing,device,write_threshold
):
    model.eval(); selector.eval()
    real_acc=INV42.metric_accumulator(); candidate_acc=INV42.metric_accumulator()
    oracle_acc=INV42.metric_accumulator(); contentless_acc=INV42.metric_accumulator(); shuffled_acc=INV42.metric_accumulator()
    hp=sp=0.0; hc=sc=hw=sw=0
    for t in frames:
        runtime=loader.load(int(t)); case=INV42.build_real_case(runtime)
        temporal_input=make_temporal_input_46(
            graph,runtime=runtime,paths=paths,temporal_radius=int(temporal_radius),spacing=spacing,device=device
        )
        encoded=INV42.encode_case(
            model,runtime=runtime,case=case,temporal_input=temporal_input,spacing=spacing,device=device
        )
        full=reason_case_v3(
            model,selector,runtime=runtime,case=case,encoded=encoded,write_threshold=float(write_threshold)
        )
        INV42.update_metric_accumulator(real_acc,model=model,runtime=runtime,case=case,logits=full.final_logits)
        INV42.update_metric_accumulator(candidate_acc,model=model,runtime=runtime,case=case,logits=full.candidate_logits)
        help_mask,suppress_mask=help_suppress_masks_v3(model,case,full.candidate_logits)
        oracle_logits=torch.where(help_mask,full.candidate_logits,case.rag.spatial_edge_logits)
        INV42.update_metric_accumulator(oracle_acc,model=model,runtime=runtime,case=case,logits=oracle_logits)
        prob=full.reasoning.edge_temporal_gate.float(); write=case.editable&(prob>=float(write_threshold))
        if bool(help_mask.any()):
            hp+=float(prob[help_mask].sum().detach().cpu()); hc+=int(help_mask.sum().item()); hw+=int(write[help_mask].sum().item())
        if bool(suppress_mask.any()):
            sp+=float(prob[suppress_mask].sum().detach().cpu()); sc+=int(suppress_mask.sum().item()); sw+=int(write[suppress_mask].sum().item())
        for corruption,acc,seed in (
            ('contentless',contentless_acc,46_300_001+int(t)),('shuffled',shuffled_acc,46_400_001+int(t))
        ):
            state=INV42.corrupt_temporal_state(encoded.temporal,corruption=corruption,seed=seed)
            corrupted=reason_case_v3(
                model,selector,runtime=runtime,case=case,encoded=encoded,temporal=state,write_threshold=float(write_threshold)
            )
            INV42.update_metric_accumulator(acc,model=model,runtime=runtime,case=case,logits=corrupted.final_logits)
    real=INV42.finalize_metrics(real_acc); candidate=INV42.finalize_metrics(candidate_acc); oracle=INV42.finalize_metrics(oracle_acc)
    contentless=INV42.finalize_metrics(contentless_acc); shuffled=INV42.finalize_metrics(shuffled_acc)
    control_exact=0.5*(contentless['exact_bad_component_recovery']+shuffled['exact_bad_component_recovery'])
    control_cut=0.5*(contentless['cut_accuracy']+shuffled['cut_accuracy'])
    score=(3.0*real['cut_accuracy']+2.0*real['exact_bad_component_recovery']+1.5*real['keep_accuracy']
           -4.0*real['clean_false_split_rate']-0.75*control_exact-0.25*control_cut)
    safe=bool(real['cut_accuracy']>0 and real['keep_accuracy']>=0.98 and real['clean_false_split_rate']<=0.005 and real['split_only_violations']==0)
    strict=bool(real['cut_accuracy']>=0.80 and real['keep_accuracy']>=0.98 and real['exact_bad_component_recovery']>=0.60 and real['clean_false_split_rate']<=0.02 and real['split_only_violations']==0)
    return {
        'real':real,'candidate':candidate,'oracle_selector':oracle,
        'selector':{
            'write_threshold':float(write_threshold),
            'helpful_edges':int(hc),'harmful_edges':int(sc),
            'helpful_probability_mean':float(hp/max(hc,1)),
            'harmful_probability_mean':float(sp/max(sc,1)),
            'probability_separation':float(hp/max(hc,1)-sp/max(sc,1)),
            'help_write_rate':float(hw/max(hc,1)),'suppress_write_rate':float(sw/max(sc,1)),
        },
        'contentless':contentless,'shuffled':shuffled,
        'checkpoint_score':float(score),'safe_pass':safe,'strict_pass':strict,
    }


def print_metrics_v3(title,step,metrics,train_selector=None):
    r=metrics['real']; c=metrics['candidate']; o=metrics['oracle_selector']; s=metrics['selector']
    print('\n'+'='*112,flush=True); print(f'{title} @ STEP {step}',flush=True); print('='*112,flush=True)
    print(f"final CUT/KEEP            : {r['cut_accuracy']:.4f} / {r['keep_accuracy']:.4f}",flush=True)
    print(f"final exact/false split   : {r['exact_bad_component_recovery']:.4f} / {r['clean_false_split_rate']:.4f}",flush=True)
    print(f"candidate CUT/KEEP        : {c['cut_accuracy']:.4f} / {c['keep_accuracy']:.4f}",flush=True)
    print(f"candidate exact           : {c['exact_bad_component_recovery']:.4f}",flush=True)
    print(f"oracle CUT/KEEP           : {o['cut_accuracy']:.4f} / {o['keep_accuracy']:.4f}",flush=True)
    print(f"oracle exact              : {o['exact_bad_component_recovery']:.4f}",flush=True)
    print(f"selector threshold        : {s['write_threshold']:.6f}",flush=True)
    print(f"HELP prob/write           : {s['helpful_probability_mean']:.4f} / {s['help_write_rate']:.4f} (n={s['helpful_edges']})",flush=True)
    print(f"SUPPRESS prob/write       : {s['harmful_probability_mean']:.4f} / {s['suppress_write_rate']:.4f} (n={s['harmful_edges']})",flush=True)
    print(f"selector separation       : {s['probability_separation']:.4f}",flush=True)
    if train_selector is not None:
        print(f"TRAIN HELP recall         : {train_selector['help_recall']:.4f}",flush=True)
        print(f"TRAIN suppress write rate : {train_selector['suppress_write_rate']:.4f}",flush=True)
        print(f"TRAIN selector AUC        : {train_selector['auc']:.4f}",flush=True)
    print(f"contentless CUT/exact     : {metrics['contentless']['cut_accuracy']:.4f} / {metrics['contentless']['exact_bad_component_recovery']:.4f}",flush=True)
    print(f"shuffled CUT/exact        : {metrics['shuffled']['cut_accuracy']:.4f} / {metrics['shuffled']['exact_bad_component_recovery']:.4f}",flush=True)
    print(f"SAFE / STRICT             : {metrics['safe_pass']} / {metrics['strict_pass']}",flush=True)
    print(f"checkpoint score          : {metrics['checkpoint_score']:.5f}",flush=True); print('='*112,flush=True)


def rank_v3(metrics):
    return (int(bool(metrics.get('strict_pass',False))),int(bool(metrics.get('safe_pass',False))),float(metrics['checkpoint_score']))


def cp_v3(paths):
    return {
        'best':paths.output/'best_selector_v3.pt',
        'best_metrics':paths.output/'best_selector_metrics_v3.json',
        'final':paths.output/'final_selector_v3.pt',
        'history':paths.output/'selector_history_v3.json',
        'train_examples':paths.output/'selector_train_examples_v3.csv',
        'initial':paths.output/'initial_selector_metrics_v3.json',
    }


def save_v3(path,*,model,selector,threshold,calibration,metrics,args,step):
    if STATE is None: raise RuntimeError('Inv46 state missing')
    payload={
        'format':'inv46_discrete_write_selector_v3','investigation':SCRIPT_NAME,'objective_version':OBJECTIVE_VERSION,
        'selector_step':int(step),'write_threshold':float(threshold),'selector_calibration':calibration,
        'temporal_encoder_state_dict':model.temporal_encoder.state_dict(),
        'instance_temporal_state_dict':model.instance_temporal.state_dict(),
        'selector_state_dict':selector.state_dict(),'validation_metrics':metrics,'cutter':STATE.cutter_metrics,'motion':STATE.motion_metrics,'args':vars(args),
        'notes':{'temporal_candidate_frozen':True,'balanced_help_suppress':True,'threshold_train_only':True,'hard_candidate_or_spatial':True,'continuous_interpolation_removed':True},
    }
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload,tmp); os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def restore_v3(path,*,model,selector):
    payload=torch_load(path,map_location='cpu')
    if not isinstance(payload,dict) or payload.get('format')!='inv46_discrete_write_selector_v3':
        raise RuntimeError(f'Invalid V3 checkpoint {path}')
    model.temporal_encoder.load_state_dict(payload['temporal_encoder_state_dict'],strict=True)
    model.instance_temporal.load_state_dict(payload['instance_temporal_state_dict'],strict=True)
    selector.load_state_dict(payload['selector_state_dict'],strict=True)
    return payload


def train_discrete_selector_v3(*,paths,train_frames,val_frames,loader,graph,spacing,device,args):
    cp=cp_v3(paths)
    initializer=resolve(args.resume) if args.resume is not None else paths.checkpoint
    payload,model=INV42.load_model(initializer,device)
    initializer_step=int(payload.get('global_step',0)) if isinstance(payload,dict) else 0
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    selector=Inv46DiscreteWriteSelector(int(model.cfg.partition.rag_hidden_dim)).to(device)
    ds=extract_selector_dataset_v3(model,frames=train_frames,loader=loader,graph=graph,paths=paths,temporal_radius=int(args.temporal_radius),spacing=spacing,device=device)
    atomic_csv(cp['train_examples'],ds.metadata)
    print('\n'+'='*112,flush=True); print('INVESTIGATION 46 V3 — BALANCED DISCRETE WRITE SELECTOR',flush=True); print('='*112,flush=True)
    print(f'initializer             : {initializer}',flush=True); print(f'initializer global step : {initializer_step}',flush=True)
    print('temporal candidate      : FROZEN',flush=True); print('final write             : HARD candidate-or-spatial',flush=True)
    print(f'train examples          : {len(ds.target_write):,} (HELP={ds.help_count}, SUPPRESS={ds.suppress_count})',flush=True)
    print(f'selector parameters     : {sum(p.numel() for p in selector.parameters()):,}',flush=True)
    print(f'selector steps/lr       : {int(args.selector_steps)} / {float(args.selector_lr):.3g}',flush=True)
    print(f'max TRAIN suppress write: {float(args.selector_max_suppress_write_rate):.3f}',flush=True); print('='*112,flush=True)
    initial=evaluate_v3(model,selector,frames=val_frames,loader=loader,graph=graph,paths=paths,temporal_radius=int(args.temporal_radius),spacing=spacing,device=device,write_threshold=1.0+1e-6)
    candidate_reference=dict(initial['candidate']); atomic_json(cp['initial'],initial); print_metrics_v3('INV46 V3 FROZEN CANDIDATE BASELINE',0,initial)
    help_idx=torch.nonzero(ds.target_write>0.5,as_tuple=False).flatten(); suppress_idx=torch.nonzero(ds.target_write<=0.5,as_tuple=False).flatten()
    opt=torch.optim.AdamW(selector.parameters(),lr=float(args.selector_lr),weight_decay=float(args.selector_weight_decay))
    gen=torch.Generator(device='cpu'); gen.manual_seed(int(args.seed)+46503)
    history=[]; best_rank=None; best_step=-1; stale=0
    for step in range(1,int(args.selector_steps)+1):
        selector.train(True); opt.zero_grad(set_to_none=True); n=int(args.selector_batch_per_class)
        hi=help_idx[torch.randint(0,int(help_idx.numel()),(n,),generator=gen)]; si=suppress_idx[torch.randint(0,int(suppress_idx.numel()),(n,),generator=gen)]
        batch=torch.cat([hi,si]); batch=batch[torch.randperm(int(batch.numel()),generator=gen)]
        logits=selector(edge_embedding=ds.edge_embedding[batch].to(device),scalar_features=ds.scalar_features[batch].to(device)); target=ds.target_write[batch].to(device)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
        if not bool(torch.isfinite(loss.detach())): raise FloatingPointError(f'non-finite selector loss at {step}')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(selector.parameters(),float(args.grad_clip)); grad_norm=float(torch.as_tensor(grad).detach().cpu())
        if not math.isfinite(grad_norm): raise FloatingPointError(f'non-finite selector grad at {step}')
        opt.step(); history.append({'step':int(step),'loss':float(loss.detach().cpu()),'grad_norm':grad_norm})
        if step==1 or step%int(args.print_every)==0:
            print(f'[selector {step:04d}/{int(args.selector_steps)}] loss={float(loss.detach().cpu()):.5f} grad={grad_norm:.3f}',flush=True)
        if step%int(args.selector_eval_every)==0 or step==int(args.selector_steps):
            probs=selector_probabilities_v3(selector,ds,device); target_np=ds.target_write.numpy().astype(np.int64,copy=False)
            threshold,cal=choose_selector_threshold_v3(probs,target_np,max_suppress_write_rate=float(args.selector_max_suppress_write_rate),override=args.selector_threshold)
            metrics=evaluate_v3(model,selector,frames=val_frames,loader=loader,graph=graph,paths=paths,temporal_radius=int(args.temporal_radius),spacing=spacing,device=device,write_threshold=threshold)
            INV42.assert_candidate_invariant(candidate_reference,metrics['candidate']); print_metrics_v3('INV46 V3 VALIDATION',step,metrics,train_selector=cal)
            rank=rank_v3(metrics)
            if best_rank is None or rank>best_rank:
                best_rank=rank; best_step=step; stale=0; save_v3(cp['best'],model=model,selector=selector,threshold=threshold,calibration=cal,metrics=metrics,args=args,step=step); atomic_json(cp['best_metrics'],metrics)
                print(f'[best selector] step={step} rank={rank} threshold={threshold:.6f}',flush=True)
            else:
                stale+=1; print(f'[selector selection] rank={rank} best={best_rank} stale={stale}',flush=True)
            atomic_json(cp['history'],history)
            if int(args.selector_patience)>0 and stale>=int(args.selector_patience):
                print('[selector early-stop] no validation improvement.',flush=True); break
    if not cp['best'].is_file(): raise RuntimeError('No best V3 selector checkpoint produced')
    probs=selector_probabilities_v3(selector,ds,device); target_np=ds.target_write.numpy().astype(np.int64,copy=False)
    threshold,cal=choose_selector_threshold_v3(probs,target_np,max_suppress_write_rate=float(args.selector_max_suppress_write_rate),override=args.selector_threshold)
    metrics=evaluate_v3(model,selector,frames=val_frames,loader=loader,graph=graph,paths=paths,temporal_radius=int(args.temporal_radius),spacing=spacing,device=device,write_threshold=threshold)
    save_v3(cp['final'],model=model,selector=selector,threshold=threshold,calibration=cal,metrics=metrics,args=args,step=len(history)); atomic_json(cp['history'],history)
    selected=restore_v3(cp['best'],model=model,selector=selector); threshold=float(selected['write_threshold'])
    metrics=evaluate_v3(model,selector,frames=val_frames,loader=loader,graph=graph,paths=paths,temporal_radius=int(args.temporal_radius),spacing=spacing,device=device,write_threshold=threshold)
    INV42.assert_candidate_invariant(candidate_reference,metrics['candidate']); print_metrics_v3('INV46 V3 SELECTED FINAL',best_step,metrics,train_selector=selected['selector_calibration'])
    print('\n'+'='*112,flush=True); print('INVESTIGATION 46 V3 TRAINING COMPLETE',flush=True); print('='*112,flush=True)
    print(f"best selector : {cp['best']}",flush=True); print(f"last selector : {cp['final']}",flush=True); print(f'selected threshold: {threshold:.6f}',flush=True); print('='*112,flush=True)


# Extend the cutter/data parser with V3 selector controls.
_BUILD_PARSER_BASE = build_parser
_VALIDATE_ARGS_BASE = validate_inv46_args


def build_parser() -> argparse.ArgumentParser:
    parser=_BUILD_PARSER_BASE()
    parser.description='Investigation 46 V3: frozen enriched temporal candidate + balanced discrete write selector.'
    parser.add_argument('--selector-steps',type=int,default=400)
    parser.add_argument('--selector-lr',type=float,default=3.0e-4)
    parser.add_argument('--selector-weight-decay',type=float,default=1.0e-4)
    parser.add_argument('--selector-batch-per-class',type=int,default=32)
    parser.add_argument('--selector-eval-every',type=int,default=50)
    parser.add_argument('--selector-patience',type=int,default=6)
    parser.add_argument('--selector-max-suppress-write-rate',type=float,default=0.10)
    parser.add_argument('--selector-threshold',type=float,default=None)
    parser.set_defaults(print_every=25)
    return parser


def validate_inv46_args(args: argparse.Namespace) -> None:
    _VALIDATE_ARGS_BASE(args)
    if int(args.selector_steps)<1: raise ValueError('--selector-steps must be >=1')
    if float(args.selector_lr)<=0: raise ValueError('--selector-lr must be >0')
    if float(args.selector_weight_decay)<0: raise ValueError('--selector-weight-decay must be >=0')
    if int(args.selector_batch_per_class)<1: raise ValueError('--selector-batch-per-class must be >=1')
    if int(args.selector_eval_every)<1: raise ValueError('--selector-eval-every must be >=1')
    if int(args.selector_patience)<0: raise ValueError('--selector-patience must be >=0')
    if not 0.0<=float(args.selector_max_suppress_write_rate)<=1.0: raise ValueError('--selector-max-suppress-write-rate must be in [0,1]')
    if args.selector_threshold is not None and not 0.0<=float(args.selector_threshold)<=1.0: raise ValueError('--selector-threshold must be in [0,1]')


def write_manifest_46(paths,*,args,reviewed,train_frames,val_frames,spacing)->None:
    payload={
        'version':OBJECTIVE_VERSION,'investigation':SCRIPT_NAME,'sample_id':paths.sample,'split':paths.split,'annotation_set':paths.annotation_set,
        'reviewed_frames':list(map(int,reviewed)),'train_frames':list(map(int,train_frames)),'val_frames':list(map(int,val_frames)),
        'temporal_radius':int(args.temporal_radius),'spacing_zyx_um':list(map(float,spacing)),
        'initializer':str(args.resume) if args.resume is not None else str(paths.checkpoint),'spatial_cache':str(paths.spatial_cache),'track_graph':str(paths.track_graph),
        'cutter_contract':{'synchronous':True,'recursive_cascade':False,'one_to_one_tracklet_edges_only':True,'target_clean_break_rate':float(args.cutter_target_clean_break_rate),'boundary_volume_scale':float(args.cutter_boundary_volume_scale),'near_radius_dref':float(args.cutter_near_radius_dref)},
        'temporal_contract':{'global_motion_compensated':True,'volume_features':True,'hard_cut_flags':True,'candidate_frozen':True,'split_only':True},
        'selector_contract':{'balanced_help_suppress':True,'neutral_excluded':True,'physical_feature_dim':int(PHYSICAL_GATE_DIM),'steps':int(args.selector_steps),'lr':float(args.selector_lr),'threshold_train_only':True,'max_suppress_write_rate':float(args.selector_max_suppress_write_rate),'hard_candidate_or_spatial':True,'continuous_interpolation':False},
    }
    atomic_json(paths.output/'dataset_manifest_v3.json',payload)


def main() -> int:
    global STATE

    args = build_parser().parse_args()
    validate_inv46_args(args)
    auto_defaults(args)

    reviewed = INV42.parse_frame_spec(args.reviewed_frames)
    train_frames = INV42.parse_frame_spec(args.train_frames)
    val_frames = INV42.parse_frame_spec(args.val_frames)
    INV42.validate_args(
        args,
        reviewed=reviewed,
        train_frames=train_frames,
        val_frames=val_frames,
    )
    spacing = INV42.parse_spacing(args.spacing)

    paths = INV42.make_paths(args)
    paths.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 46 — HARD TRACKLET CUTTER + TEMPORAL TRAINING", flush=True)
    print("=" * 112, flush=True)
    print(f"repository        : {ROOT}", flush=True)
    print(f"volume            : {paths.split}/{paths.sample}", flush=True)
    print(f"annotation set    : {paths.annotation_set}", flush=True)
    print(f"reviewed frames   : {list(reviewed)}", flush=True)
    print(f"train frames      : {list(train_frames)}", flush=True)
    print(f"validation frames : {list(val_frames)}", flush=True)
    print(f"initializer       : {args.resume or paths.checkpoint}", flush=True)
    print(f"spatial cache     : {paths.spatial_cache}", flush=True)
    print(f"output            : {paths.output}", flush=True)
    print("=" * 112, flush=True)

    write_manifest_46(
        paths,
        args=args,
        reviewed=reviewed,
        train_frames=train_frames,
        val_frames=val_frames,
        spacing=spacing,
    )

    reviewed_set = set(reviewed)
    ignored_ids, ignored_records = INV42.load_ignored_ids(paths, reviewed_set)
    target_store = INV42.TargetFrameStore(paths, reviewed, ignored_ids)
    ignore_metrics = INV42.validate_ignored_ids(target_store)
    ignore_metrics["ignore_records_touching_reviewed_frames"] = int(
        len(ignored_records)
    )

    requested_cache_frames = tuple(
        sorted(set(train_frames) | set(val_frames))
    )
    INV42.prepare_spatial_cache(
        paths,
        frames=requested_cache_frames,
        spacing=spacing,
        device=device,
        args=args,
    )
    spatial_cache_metrics = INV42.validate_spatial_cache(
        paths,
        requested_cache_frames,
    )

    # Canonical graph identity enrichment stays exactly Investigation 42.
    graph_original = INV42.load_track_graph(paths)
    coordinate_metrics = INV42.audit_and_enrich_track_graph(
        graph_original,
        paths=paths,
        target_frames=requested_cache_frames,
        spacing=spacing,
        temporal_radius=int(args.temporal_radius),
        max_error_um=float(args.max_coordinate_error_um),
    )

    movie = np.load(
        paths.base_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    shape_zyx = tuple(int(v) for v in movie.shape[1:])

    inv45_paths = inv45_paths_from_inv42(
        paths,
        output=paths.output / "motion",
    )
    motion, motion_metrics = INV45.load_motion(
        inv45_paths,
        shape_zyx,
        tuple(float(v) for v in spacing),
    )
    print(
        "[Inv46 motion] "
        f"source={motion_metrics['source']} "
        f"median={motion_metrics['pairwise_motion_median_um']:.3f}um "
        f"p90={motion_metrics['pairwise_motion_p90_um']:.3f}um",
        flush=True,
    )

    classification = INV45.classify_components(
        inv45_paths,
        movie,
        reviewed,
        INV45.ignored_ids(inv45_paths, reviewed_set),
    )

    hypothesis_index, hypothesis_path = load_hypothesis_index(
        paths.sample,
        override=args.hypothesis_cache,
        plausible_score=float(args.cutter_plausible_score),
    )

    cutter_rows, node_evidence = build_cutter_rows(
        graph_original,
        classification=classification,
        motion=motion,
        spacing=tuple(float(v) for v in spacing),
        shape_zyx=shape_zyx,
        near_radius_dref=float(args.cutter_near_radius_dref),
        boundary_volume_scale=float(args.cutter_boundary_volume_scale),
        hypothesis_index=hypothesis_index,
    )

    threshold, calibration = choose_threshold(
        cutter_rows,
        train_frames=set(train_frames),
        target_clean_break_rate=float(args.cutter_target_clean_break_rate),
        minimum_threshold=float(args.cutter_min_threshold),
        override=args.cutter_threshold,
    )

    graph_sanitized, cutter_decisions = apply_hard_cutter(
        graph_original,
        frame=cutter_rows,
        node_evidence=node_evidence,
        threshold=threshold,
    )

    train_cutter = split_metrics(
        cutter_decisions,
        target_frames=set(train_frames),
        threshold=threshold,
    )
    val_cutter = split_metrics(
        cutter_decisions,
        target_frames=set(val_frames),
        threshold=threshold,
    )
    train_newborn = component_newborn_metrics(
        graph_original,
        graph_sanitized,
        classification=classification,
        frames=set(train_frames),
    )
    val_newborn = component_newborn_metrics(
        graph_original,
        graph_sanitized,
        classification=classification,
        frames=set(val_frames),
    )

    cutter_metrics = {
        "threshold_calibration": calibration,
        "train": train_cutter,
        "validation": val_cutter,
        "train_newborn": train_newborn,
        "validation_newborn": val_newborn,
        "total_eligible_edges": int(len(cutter_decisions)),
        "total_hard_cuts": int(cutter_decisions["hard_cut"].sum()),
        "hypothesis_cache": (
            str(hypothesis_path) if hypothesis_path is not None else None
        ),
        "hypothesis_used": bool(hypothesis_path is not None),
        "weights": {
            "local_volume": W_LOCAL_VOLUME,
            "parent_growth": W_PARENT_GROWTH,
            "prediction_error": W_PREDICTION_ERROR,
            "nearby_broken": W_NEARBY_BROKEN,
            "rejected_hypothesis": W_REJECTED_HYPOTHESIS,
        },
    }

    atomic_csv(paths.output / "hard_cutter_edges.csv", cutter_decisions)
    atomic_json(paths.output / "hard_cutter_metrics.json", cutter_metrics)
    atomic_json(paths.output / "motion_metrics.json", motion_metrics)

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 46 — HARD CUTTER AUDIT", flush=True)
    print("=" * 112, flush=True)
    print(f"threshold                    : {threshold:.4f}", flush=True)
    print(
        f"train merge-link break recall: {train_cutter['merge_break_recall']:.4f} "
        f"({train_cutter['merge_edges_cut']}/{train_cutter['merge_edges']})",
        flush=True,
    )
    print(
        f"train clean break rate        : {train_cutter['clean_break_rate']:.4f} "
        f"({train_cutter['clean_edges_cut']}/{train_cutter['clean_edges']})",
        flush=True,
    )
    print(
        f"val merge-link break recall  : {val_cutter['merge_break_recall']:.4f} "
        f"({val_cutter['merge_edges_cut']}/{val_cutter['merge_edges']})",
        flush=True,
    )
    print(
        f"val clean break rate          : {val_cutter['clean_break_rate']:.4f} "
        f"({val_cutter['clean_edges_cut']}/{val_cutter['clean_edges']})",
        flush=True,
    )
    print(
        f"val merge newborn after cut   : "
        f"{val_newborn['merge_newborn_after_rate']:.4f} "
        f"({val_newborn['merge_newborn_after_count']}/"
        f"{val_newborn['merge_components_present']})",
        flush=True,
    )
    print(
        f"total graph edges cut         : "
        f"{int(cutter_decisions['hard_cut'].sum())}",
        flush=True,
    )
    print(
        f"Inv45 hypotheses used         : {bool(hypothesis_path is not None)}",
        flush=True,
    )
    print("=" * 112, flush=True)

    STATE = Inv46State(
        args=args,
        paths=paths,
        spacing=tuple(float(v) for v in spacing),
        shape_zyx=shape_zyx,
        motion=motion,
        motion_metrics=motion_metrics,
        node_evidence=node_evidence,
        cutter_rows=cutter_decisions,
        cutter_threshold=float(threshold),
        cutter_metrics=cutter_metrics,
        hypothesis_cache=hypothesis_path,
        hypothesis_used=bool(hypothesis_path is not None),
    )

    # The curated spatial-target audit remains Investigation 42's exact logic.
    loader = INV42.RuntimeLoader(paths, target_store, device)
    operations = INV42.load_spatial_operations(paths)
    audit = INV42.audit_dataset(
        train_frames=train_frames,
        val_frames=val_frames,
        loader=loader,
        operations=operations,
        ignore_metrics=ignore_metrics,
        spatial_cache_metrics=spatial_cache_metrics,
        coordinate_metrics=coordinate_metrics,
    )
    audit["inv46_hard_cutter"] = cutter_metrics
    audit["inv46_motion"] = motion_metrics
    atomic_json(paths.output / "audit.json", audit)
    INV42.print_audit(audit)

    if args.audit_only:
        print(
            "Investigation-46 cutter audit passed. No temporal training was "
            "started because --audit-only was supplied.",
            flush=True,
        )
        return 0

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 46 V3 — DISCRETE WRITE SELECTOR", flush=True)
    print("=" * 112, flush=True)
    print("temporal coordinates : target-relative global-motion compensated", flush=True)
    print("hard cutter          : ACTIVE, synchronous, non-recursive", flush=True)
    print("temporal candidate   : FROZEN", flush=True)
    print("final write          : HARD candidate-or-spatial", flush=True)
    print("selector training    : balanced HELP/SUPPRESS", flush=True)
    print("=" * 112, flush=True)

    train_discrete_selector_v3(
        paths=paths, train_frames=train_frames, val_frames=val_frames, loader=loader,
        graph=graph_sanitized, spacing=spacing, device=device, args=args,
    )

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 46 V3 COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print(f"output             : {paths.output}", flush=True)
    print(f"hard cutter edges  : {paths.output / 'hard_cutter_edges.csv'}", flush=True)
    print(f"cutter metrics     : {paths.output / 'hard_cutter_metrics.json'}", flush=True)
    cp = cp_v3(paths)
    print(f"best selector      : {cp['best']}", flush=True)
    print(f"last selector      : {cp['final']}", flush=True)
    print(f"train examples     : {cp['train_examples']}", flush=True)
    print("=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
