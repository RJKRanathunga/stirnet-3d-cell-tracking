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
The mature spatial network and InstanceTokenizer remain frozen.

Trainable:
    temporal_encoder
    instance_temporal reasoner except its original 4-scalar edge_gate
    Investigation-42 selective-write gate calibrator

The loss is Investigation 42 V11's curated split-only objective plus:
    candidate_weight * ungated temporal candidate CUT/KEEP loss
    split_weight * component split-head loss

The original candidate-invariance assertion is intentionally disabled because
the purpose of Investigation 46 is to raise the temporal candidate ceiling.

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

Typical commands
----------------
Audit cutter only:

    python .\investigations\stirnet\46_biohub_hard_cutter_temporal_training.py --audit-only

First training run:

    python .\investigations\stirnet\46_biohub_hard_cutter_temporal_training.py `
        --steps 600 `
        --lr 5e-5 `
        --candidate-weight 0.5 `
        --eval-every 50 `
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

SCRIPT_NAME = "46_biohub_hard_cutter_temporal_training"
OBJECTIVE_VERSION = 1

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
# Temporal training overrides for Investigation 42's mature training loop
# =============================================================================


ORIGINAL_TRAINING_LOSS = INV42.training_loss
ORIGINAL_PRINT_METRICS = INV42.print_temporal_metrics


def configure_temporal_training_46(model, gate_calibrator):
    global ACTIVE_MODEL
    ACTIVE_MODEL = model

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable: list[Tensor] = []

    for parameter in model.temporal_encoder.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    # Train the temporal candidate/reasoner but keep the old under-specified
    # 4-scalar production edge_gate frozen. The experiment-local V11 calibrator
    # remains responsible for selective writing.
    for name, parameter in model.instance_temporal.named_parameters():
        if name.startswith("edge_gate."):
            parameter.requires_grad_(False)
            continue
        parameter.requires_grad_(True)
        trainable.append(parameter)

    for parameter in gate_calibrator.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    model.eval()
    gate_calibrator.train(True)
    return trainable


def parameter_audit_46(model, gate_calibrator):
    temporal_encoder = int(
        sum(
            p.numel()
            for p in model.temporal_encoder.parameters()
            if p.requires_grad
        )
    )
    temporal_reasoner = int(
        sum(
            p.numel()
            for p in model.instance_temporal.parameters()
            if p.requires_grad
        )
    )
    frozen_edge_gate = int(
        sum(
            p.numel()
            for p in model.instance_temporal.edge_gate.parameters()
            if not p.requires_grad
        )
    )
    calibrator = int(
        sum(
            p.numel()
            for p in gate_calibrator.parameters()
            if p.requires_grad
        )
    )
    total = temporal_encoder + temporal_reasoner + calibrator

    if total <= 0:
        raise RuntimeError("Investigation 46 has zero trainable parameters.")

    return {
        "temporal_encoder": temporal_encoder,
        "temporal_reasoner_excluding_old_gate": temporal_reasoner,
        "frozen_old_edge_gate": frozen_edge_gate,
        "selective_gate_calibrator": calibrator,
        "total": total,
    }


def training_loss_46(
    model,
    gate_calibrator,
    *,
    case,
    encoded,
    full,
    args,
    rng,
    corruption,
):
    base = ORIGINAL_TRAINING_LOSS(
        model,
        gate_calibrator,
        case=case,
        encoded=encoded,
        full=full,
        args=args,
        rng=rng,
        corruption=corruption,
    )
    total = (
        base.total
        + float(args.candidate_weight) * base.candidate_total
        + float(args.split_weight) * base.split
    )
    return dataclass_replace_loss(base, total=total)


def no_candidate_invariant(reference, current) -> None:
    # Candidate change is the objective of Investigation 46.
    return None


def print_temporal_metrics_46(title: str, step: int, metrics: dict[str, Any]) -> None:
    ORIGINAL_PRINT_METRICS(
        title.replace("INVESTIGATION 42", "INVESTIGATION 46"),
        step,
        metrics,
    )


def save_training_state_46(
    path: Path,
    *,
    gate_calibrator,
    optimizer,
    scaler,
    step: int,
    config,
    paths,
    args,
    audit,
    validation,
    base_checkpoint: Path,
) -> None:
    global ACTIVE_MODEL, STATE
    if ACTIVE_MODEL is None:
        raise RuntimeError("Inv46 active model is unavailable during checkpoint save.")

    payload = {
        "format": "inv46_hard_cutter_temporal_v1",
        "investigation": SCRIPT_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "global_step": int(step),
        "initializer_checkpoint": str(base_checkpoint),
        "initializer_checkpoint_sha256": INV42.sha256(base_checkpoint),
        "temporal_encoder_state_dict": ACTIVE_MODEL.temporal_encoder.state_dict(),
        "instance_temporal_state_dict": ACTIVE_MODEL.instance_temporal.state_dict(),
        "selective_gate_state_dict": gate_calibrator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "training_config": config,
        "args": vars(args),
        "parameter_audit": audit,
        "validation_metrics": validation or {},
        "cutter": (
            STATE.cutter_metrics if STATE is not None else {}
        ),
        "notes": {
            "spatial_model_frozen": True,
            "instance_tokenizer_frozen": True,
            "temporal_encoder_trained": True,
            "temporal_candidate_trained": True,
            "old_four_scalar_edge_gate_frozen": True,
            "selective_gate_trained": True,
            "hard_tracklet_cutter": True,
            "global_motion_compensated_temporal_coordinates": True,
            "manual_tracking_overrides_used": False,
            "temporal_action": "split_only",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def install_training_overrides() -> None:
    INV42.configure_temporal_training = configure_temporal_training_46
    INV42.parameter_audit = parameter_audit_46
    INV42.training_loss = training_loss_46
    INV42.assert_candidate_invariant = no_candidate_invariant
    INV42.make_temporal_input = make_temporal_input_46
    INV42.save_training_state = save_training_state_46
    INV42.print_temporal_metrics = print_temporal_metrics_46


# =============================================================================
# CLI / main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = INV42.build_parser()
    parser.description = (
        "Investigation 46: calibrate an aggressive physical hard tracklet cutter "
        "and fine-tune STIR-Net temporal reasoning on the resulting enriched graph."
    )

    parser.add_argument(
        "--cutter-threshold",
        type=float,
        default=None,
        help=(
            "Explicit hard-cutter threshold. Default: calibrate on training "
            "frames only under --cutter-target-clean-break-rate."
        ),
    )
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
    parser.add_argument(
        "--hypothesis-cache",
        type=Path,
        default=None,
        help=(
            "Optional Investigation-45 hypothesis edges.csv. If omitted, the "
            "standard Inv45 run path is used when present."
        ),
    )

    # Investigation-46 defaults: starting from the step-300 temporal checkpoint
    # usually means --steps 600 performs ~300 additional optimizer steps.
    parser.set_defaults(
        steps=600,
        lr=5.0e-5,
        weight_decay=1.0e-4,
        candidate_weight=0.5,
        candidate_preservation_weight=1.0,
        split_weight=0.05,
        preservation_weight=2.0,
        cut_frame_probability=0.80,
        eval_every=50,
        print_every=10,
    )
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
        raise ValueError(
            "Investigation 46 does not support --gate-resume because the temporal "
            "parameter topology is intentionally different. Start from --resume "
            "(the step-300 full temporal checkpoint) instead."
        )


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
            "spatial_frozen": True,
            "instance_tokenizer_frozen": True,
            "temporal_encoder_trainable": True,
            "temporal_reasoner_trainable": True,
            "old_four_scalar_gate_frozen": True,
            "selective_gate_trainable": True,
            "candidate_auxiliary_loss": float(args.candidate_weight),
            "split_loss": float(args.split_weight),
        },
    }
    atomic_json(paths.output / "dataset_manifest.json", payload)


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

    install_training_overrides()

    initializer_path = (
        resolve(args.resume)
        if args.resume is not None
        else paths.checkpoint
    )
    initializer_payload = torch_load(initializer_path, map_location="cpu")
    initializer_step = (
        int(initializer_payload.get("global_step", 0))
        if isinstance(initializer_payload, dict)
        else 0
    )
    if int(args.steps) <= initializer_step:
        raise ValueError(
            f"--steps={int(args.steps)} must be greater than initializer "
            f"global_step={initializer_step}. For example, use "
            f"--steps {initializer_step + 300}."
        )
    del initializer_payload

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 46 — TEMPORAL FINE-TUNING", flush=True)
    print("=" * 112, flush=True)
    print("temporal coordinates : target-relative global-motion compensated", flush=True)
    print("hard cutter          : ACTIVE, synchronous, non-recursive", flush=True)
    print("volume features      : ACTIVE", flush=True)
    print(
        f"candidate aux weight : {float(args.candidate_weight):g}",
        flush=True,
    )
    print(
        f"split-head weight    : {float(args.split_weight):g}",
        flush=True,
    )
    print("old 4-scalar gate    : FROZEN", flush=True)
    print("V11 selective gate   : TRAINABLE", flush=True)
    print("temporal encoder     : TRAINABLE", flush=True)
    print("temporal candidate   : TRAINABLE", flush=True)
    print("=" * 112, flush=True)

    INV42.train(
        paths=paths,
        train_frames=train_frames,
        val_frames=val_frames,
        loader=loader,
        graph=graph_sanitized,
        spacing=spacing,
        device=device,
        args=args,
    )

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 46 COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print(f"output             : {paths.output}", flush=True)
    print(f"hard cutter edges  : {paths.output / 'hard_cutter_edges.csv'}", flush=True)
    print(f"cutter metrics     : {paths.output / 'hard_cutter_metrics.json'}", flush=True)
    print(f"best checkpoint    : {paths.best}", flush=True)
    print(f"best safe          : {paths.best_safe}", flush=True)
    print("=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
