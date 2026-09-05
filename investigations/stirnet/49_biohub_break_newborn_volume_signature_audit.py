from __future__ import annotations

r"""
Investigation 49 — break → newborn → volume-conservation signature audit.

This is a DIAGNOSTIC-ONLY experiment.  It does not train or modify STIR-Net.

Question
--------
For each of the 12 trusted validation merge components in frames 30..39, test
the specific temporal signature:

    t-1                                         t
    ----------------------------------          --------------------------
    wrongly assigned Trackastra parent  --X--> merged current instance
                                             (edge removed by hard cutter)
    another nearby track ends / breaks  ---->  same stabilized location

                                                   ↓ after the cut

                                             NEWBORN current track

and then test the physical volume signature:

    V_expected_previous
        = V_wrong_parent(t-1)
          + sum(V_other_nearby_broken_tracks(t-1))

    V_current(t) ≈ V_expected_previous

and:

    V_current(t) > V_wrong_parent(t-1)

The point is NOT to fit a learned model.  The point is to answer, explicitly:

    How many of the 12 curated merge failures have this signature?

The script also evaluates trusted clean components in the same validation
frames.  That is important because "unique" should mean both:

    high recall on the 12 merge events
    AND
    rare among clean components.

Definitions
-----------
NEWBORN AFTER CUT:
    Current canonical Trackastra node has no accepted predecessor in the
    sanitized / hard-cut graph.

WRONG PARENT BEFORE CUT:
    A t-1 -> t accepted edge exists in the persisted pre-cutter Trackastra
    graph, but that edge is absent from the sanitized graph.  The t-1 endpoint
    is therefore the cell Trackastra had assigned to the current merged object
    before the hard cutter removed that association.

BROKEN TRACK AT t-1:
    A canonical t-1 node has no accepted future successor in the sanitized
    graph.  The wrong parent is excluded from the "other nearby broken" set so
    its volume is not counted twice.

SAME PLACE AFTER GLOBAL SHIFT REMOVAL:
    Global t-1 -> t displacement is estimated robustly from the many surviving
    accepted Trackastra continuations in the sanitized graph.  A t-1 broken
    endpoint is translated by this global displacement, then compared with the
    current component centroid in physical micrometres.

VOLUME CLOSE:
    symmetric relative error

        abs(V_current - V_previous_sum) / max(V_current, V_previous_sum)

    is <= --volume-rel-error-max.

The default headline uses:
    nearby radius     = 1.0 * current dref
    volume tolerance  = 25%

but the experiment ALWAYS writes a radius/tolerance sweep, so conclusions do
not depend on one arbitrary threshold.

Run from repository root
------------------------

    python .\investigations\stirnet\49_biohub_break_newborn_volume_signature_audit.py

Useful optional controls:

    --nearby-radius-dref 1.0
    --volume-rel-error-max 0.25
    --radius-sweep 0.5,0.75,1.0,1.25,1.5,2.0
    --volume-error-sweep 0.10,0.15,0.20,0.25,0.30,0.40,0.50
"""

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Repository / Investigation 48 reuse
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (Path.cwd().resolve(), here.parent, *here.parents):
        if (
            (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "dataset_curation").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Could not find repository root. Run this from the cell-tracking repo."
    )


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


INV48_PATHS = (
    ROOT
    / "investigations"
    / "stirnet"
    / "48_biohub_event_conditioned_spatial_cut.py",
    ROOT
    / "investigations"
    / "stirnet"
    / "48_biohub_event_conditioned_spatial_cut.py",
)

INV48_PATH = next((p for p in INV48_PATHS if p.is_file()), None)
if INV48_PATH is None:
    raise FileNotFoundError(
        "Could not find Investigation 48. Expected one of:\n"
        + "\n".join(str(p) for p in INV48_PATHS)
    )

INV48 = load_module(INV48_PATH, "_inv49_inv48")
INV47 = INV48.INV47
INV46 = INV48.INV46
INV42 = INV48.INV42

SAMPLE = str(INV42.DEFAULT_SAMPLE)
SCRIPT_NAME = "49_biohub_break_newborn_volume_signature_audit"


# =============================================================================
# CLI
# =============================================================================


def parse_float_list(text: str) -> tuple[float, ...]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"all values must be >0, got {value}"
            )
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    # Reuse the exact Inv48/47 setup contract and checkpoint defaults.
    parser = INV48.build_parser()
    parser.description = (
        "Investigation 49: audit the break→newborn→volume-conservation "
        "signature on curated BioHub merge failures."
    )

    parser.add_argument(
        "--signature-output",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / SCRIPT_NAME
            / SAMPLE
        ).resolve(),
        help="Investigation-49 diagnostic output directory.",
    )
    parser.add_argument(
        "--nearby-radius-dref",
        type=float,
        default=1.0,
        help=(
            "Headline radius around the current component centroid after "
            "global-shift compensation, in units of current dref."
        ),
    )
    parser.add_argument(
        "--volume-rel-error-max",
        type=float,
        default=0.25,
        help=(
            "Headline maximum symmetric relative error between current "
            "volume and summed previous candidate-cell volumes."
        ),
    )
    parser.add_argument(
        "--radius-sweep",
        type=parse_float_list,
        default=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        help="Comma-separated dref radius sweep.",
    )
    parser.add_argument(
        "--volume-error-sweep",
        type=parse_float_list,
        default=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
        help="Comma-separated relative-volume-error sweep.",
    )
    parser.add_argument(
        "--min-global-motion-links",
        type=int,
        default=25,
        help=(
            "Minimum surviving accepted adjacent links required for the "
            "robust global t-1→t displacement estimate."
        ),
    )
    return parser


def validate_args(args) -> None:
    if float(args.nearby_radius_dref) <= 0:
        raise ValueError("--nearby-radius-dref must be >0")
    if not 0 < float(args.volume_rel_error_max) < 1:
        raise ValueError("--volume-rel-error-max must be in (0,1)")
    if int(args.min_global_motion_links) < 3:
        raise ValueError("--min-global-motion-links must be >=3")


# =============================================================================
# Small file helpers
# =============================================================================


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


# =============================================================================
# Canonical graph access
# =============================================================================


def graph_node_time(data: dict[str, Any]) -> int | None:
    for key in ("time", "frame", "t", "timepoint"):
        if key in data:
            try:
                return int(data[key])
            except Exception:
                pass
    return None


def graph_node_cell_id(data: dict[str, Any]) -> int | None:
    for key in (
        "inv42_cell_id",
        "cell_id",
        "label",
        "instance_id",
    ):
        if key in data:
            try:
                value = int(data[key])
            except Exception:
                continue
            if value > 0:
                return value
    return None


def graph_node_position_um(
    data: dict[str, Any],
    spacing_zyx_um: Sequence[float],
) -> np.ndarray | None:
    # Known canonical coordinate used by Inv42/46.
    voxel_keys = (
        "inv35_coords_zyx",
        "coords_zyx",
        "centroid_zyx",
        "center_zyx",
        "position_zyx",
    )
    um_keys = (
        "coords_zyx_um",
        "centroid_zyx_um",
        "center_zyx_um",
        "position_zyx_um",
        "stable_coords_zyx_um",
        "stabilized_coords_zyx_um",
    )

    for key in um_keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.isfinite(arr[:3]).all():
            return arr[:3].astype(np.float64, copy=True)

    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    for key in voxel_keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.isfinite(arr[:3]).all():
            return arr[:3] * spacing

    # Defensive scalar-column fallbacks.
    for triplet, is_um in (
        (("z_um", "y_um", "x_um"), True),
        (("z", "y", "x"), False),
        (("centroid_z", "centroid_y", "centroid_x"), False),
    ):
        if all(k in data for k in triplet):
            arr = np.asarray(
                [data[k] for k in triplet],
                dtype=np.float64,
            )
            if np.isfinite(arr).all():
                return arr if is_um else arr * spacing

    return None


def canonical_node_tables(
    graph,
    spacing_zyx_um: Sequence[float],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[tuple[int, int], list[int]],
]:
    nodes: dict[int, dict[str, Any]] = {}
    by_time_cell: dict[tuple[int, int], list[int]] = defaultdict(list)

    for node_id, attrs in graph.nodes(data=True):
        node_id = int(node_id)
        attrs = dict(attrs or {})
        t = graph_node_time(attrs)
        cell_id = graph_node_cell_id(attrs)
        pos_um = graph_node_position_um(attrs, spacing_zyx_um)
        if t is None or cell_id is None or pos_um is None:
            continue

        row = {
            "node_id": node_id,
            "time": int(t),
            "cell_id": int(cell_id),
            "position_um": pos_um,
            "attrs": attrs,
        }
        nodes[node_id] = row
        by_time_cell[(int(t), int(cell_id))].append(node_id)

    return nodes, by_time_cell


def orient_edge(
    u: int,
    v: int,
    node_rows: dict[int, dict[str, Any]],
) -> tuple[int, int] | None:
    if int(u) not in node_rows or int(v) not in node_rows:
        return None
    tu = int(node_rows[int(u)]["time"])
    tv = int(node_rows[int(v)]["time"])
    if tu == tv:
        return None
    return (int(u), int(v)) if tu < tv else (int(v), int(u))


def oriented_edges(
    graph,
    node_rows: dict[int, dict[str, Any]],
) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for u, v in graph.edges():
        oriented = orient_edge(int(u), int(v), node_rows)
        if oriented is not None:
            result.add(oriented)
    return result


def resolve_raw_track_graph_path(paths) -> Path:
    # First inspect explicit path attributes.
    for name in (
        "track_graph",
        "track_graph_path",
        "track_graph_pkl",
    ):
        value = getattr(paths, name, None)
        if value is None:
            continue
        candidate = Path(value).resolve()
        if candidate.is_file():
            return candidate

    # The current repository stores:
    #   <preprocessed>/<split>/<sample>/movies/final_instances.npy
    #   <preprocessed>/<split>/<sample>/trackastra/track_graph.pkl
    base_instances = Path(paths.base_instances).resolve()
    candidate = (
        base_instances.parent.parent
        / "trackastra"
        / "track_graph.pkl"
    )
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Could not resolve persisted Trackastra graph. "
        f"final_instances={base_instances}"
    )


def load_raw_graph(path: Path):
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(
            f"{path} did not contain a NetworkX-like graph: {type(graph)}"
        )
    return graph


# =============================================================================
# Robust consecutive-frame global motion
# =============================================================================


def robust_global_pair_shift_um(
    *,
    t_prev: int,
    t_cur: int,
    sanitized_edges: set[tuple[int, int]],
    node_rows: dict[int, dict[str, Any]],
    min_links: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    displacements = []

    for src, dst in sanitized_edges:
        src_row = node_rows[src]
        dst_row = node_rows[dst]
        if (
            int(src_row["time"]) == int(t_prev)
            and int(dst_row["time"]) == int(t_cur)
        ):
            displacements.append(
                np.asarray(dst_row["position_um"], dtype=np.float64)
                - np.asarray(src_row["position_um"], dtype=np.float64)
            )

    if len(displacements) < int(min_links):
        raise RuntimeError(
            f"t={t_prev}->{t_cur}: only {len(displacements)} accepted "
            f"adjacent links; need >= {min_links} for global-motion estimate"
        )

    values = np.stack(displacements, axis=0)
    initial = np.median(values, axis=0)
    residual = np.linalg.norm(values - initial[None, :], axis=1)

    median_residual = float(np.median(residual))
    mad = float(
        np.median(np.abs(residual - median_residual))
    )
    robust_scale = 1.4826 * mad

    if robust_scale > 1.0e-9:
        keep = residual <= (
            median_residual + 3.5 * robust_scale
        )
    else:
        keep = np.ones(len(values), dtype=bool)

    if int(keep.sum()) < int(min_links):
        keep = np.ones(len(values), dtype=bool)

    shift = np.median(values[keep], axis=0)
    residual_after = np.linalg.norm(
        values[keep] - shift[None, :],
        axis=1,
    )

    audit = {
        "frame_prev": int(t_prev),
        "frame_cur": int(t_cur),
        "link_count": int(len(values)),
        "inlier_count": int(keep.sum()),
        "shift_z_um": float(shift[0]),
        "shift_y_um": float(shift[1]),
        "shift_x_um": float(shift[2]),
        "shift_norm_um": float(np.linalg.norm(shift)),
        "residual_median_um": float(
            np.median(residual_after)
        ),
        "residual_p90_um": float(
            np.quantile(residual_after, 0.90)
        ),
    }
    return shift.astype(np.float64), audit


# =============================================================================
# Fast instance volumes
# =============================================================================


def frame_label_volumes_um3(
    labels: np.ndarray,
    voxel_volume_um3: float,
) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected [Z,Y,X] label frame, got {labels.shape}"
        )
    flat = labels.reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(flat)
    return counts.astype(np.float64) * float(voxel_volume_um3)


def lookup_volume(
    volumes: np.ndarray,
    cell_id: int,
) -> float:
    cell_id = int(cell_id)
    if cell_id <= 0 or cell_id >= len(volumes):
        return 0.0
    return float(volumes[cell_id])


# =============================================================================
# Topology helpers
# =============================================================================


def predecessor_successor_maps(
    edges: Iterable[tuple[int, int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    pred: dict[int, list[int]] = defaultdict(list)
    succ: dict[int, list[int]] = defaultdict(list)
    for src, dst in edges:
        succ[int(src)].append(int(dst))
        pred[int(dst)].append(int(src))
    return pred, succ


def has_any_past_predecessor(
    node_id: int,
    pred: dict[int, list[int]],
    node_rows: dict[int, dict[str, Any]],
) -> bool:
    t = int(node_rows[node_id]["time"])
    return any(
        int(node_rows[p]["time"]) < t
        for p in pred.get(node_id, [])
        if p in node_rows
    )


def has_any_future_successor(
    node_id: int,
    succ: dict[int, list[int]],
    node_rows: dict[int, dict[str, Any]],
) -> bool:
    t = int(node_rows[node_id]["time"])
    return any(
        int(node_rows[s]["time"]) > t
        for s in succ.get(node_id, [])
        if s in node_rows
    )


def select_component_node(
    by_time_cell: dict[tuple[int, int], list[int]],
    *,
    t: int,
    cell_id: int,
) -> tuple[int | None, int]:
    candidates = list(
        by_time_cell.get((int(t), int(cell_id)), [])
    )
    if not candidates:
        return None, 0
    # There should normally be exactly one canonical node per persisted cell.
    return int(candidates[0]), len(candidates)


# =============================================================================
# Per-component signal extraction
# =============================================================================


def json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def component_signature_row(
    *,
    t: int,
    component: int,
    is_merge: bool,
    cell_id: int,
    current_node_id: int | None,
    duplicate_current_nodes: int,
    current_volume_um3: float,
    dref_um: float,
    radius_dref: float,
    volume_error_max: float,
    node_rows: dict[int, dict[str, Any]],
    raw_edges: set[tuple[int, int]],
    sanitized_edges: set[tuple[int, int]],
    sanitized_pred: dict[int, list[int]],
    sanitized_succ: dict[int, list[int]],
    broken_nodes_prev: Sequence[int],
    volumes_prev: np.ndarray,
    global_shift_um: np.ndarray,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "frame": int(t),
        "component": int(component),
        "target_merge": bool(is_merge),
        "current_cell_id": int(cell_id),
        "current_node_id": (
            int(current_node_id)
            if current_node_id is not None
            else -1
        ),
        "canonical_node_multiplicity": int(
            duplicate_current_nodes
        ),
        "current_volume_um3": float(current_volume_um3),
        "dref_um": float(dref_um),
        "nearby_radius_dref": float(radius_dref),
        "nearby_radius_um": float(radius_dref * dref_um),
        "volume_rel_error_max": float(volume_error_max),
    }

    if current_node_id is None:
        base.update(
            {
                "newborn_after_cut": False,
                "raw_prev_parent_count": 0,
                "removed_prev_parent_count": 0,
                "wrong_parent_unambiguous": False,
                "wrong_parent_cell_id": -1,
                "wrong_parent_volume_um3": 0.0,
                "nearby_other_broken_count": 0,
                "previous_volume_sum_um3": 0.0,
                "current_over_previous_sum": float("nan"),
                "symmetric_volume_error": float("nan"),
                "current_larger_than_wrong_parent": False,
                "temporal_pair_signal": False,
                "volume_sum_close": False,
                "full_signature": False,
                "failure_reason": "no_canonical_current_track_node",
                "nearby_other_broken_cell_ids": "[]",
                "nearby_other_broken_distances_um": "[]",
                "nearby_other_broken_volumes_um3": "[]",
            }
        )
        return base

    current_node_id = int(current_node_id)
    t_prev = int(t) - 1

    newborn_after_cut = not has_any_past_predecessor(
        current_node_id,
        sanitized_pred,
        node_rows,
    )

    raw_prev_parents = sorted(
        src
        for src, dst in raw_edges
        if dst == current_node_id
        and int(node_rows[src]["time"]) == t_prev
    )
    sanitized_prev_parents = sorted(
        src
        for src, dst in sanitized_edges
        if dst == current_node_id
        and int(node_rows[src]["time"]) == t_prev
    )
    removed_prev_parents = sorted(
        set(raw_prev_parents) - set(sanitized_prev_parents)
    )

    wrong_parent_unambiguous = (
        len(removed_prev_parents) == 1
    )
    wrong_parent_node = (
        int(removed_prev_parents[0])
        if wrong_parent_unambiguous
        else None
    )

    wrong_parent_cell_id = -1
    wrong_parent_volume = 0.0
    if wrong_parent_node is not None:
        wrong_parent_cell_id = int(
            node_rows[wrong_parent_node]["cell_id"]
        )
        wrong_parent_volume = lookup_volume(
            volumes_prev,
            wrong_parent_cell_id,
        )

    current_position = np.asarray(
        node_rows[current_node_id]["position_um"],
        dtype=np.float64,
    )

    radius_um = float(radius_dref) * float(dref_um)

    nearby_records = []
    for broken_node in broken_nodes_prev:
        broken_node = int(broken_node)
        if broken_node == wrong_parent_node:
            # Do not count the wrongly assigned parent twice.
            continue

        row = node_rows[broken_node]
        shifted_position = (
            np.asarray(row["position_um"], dtype=np.float64)
            + np.asarray(global_shift_um, dtype=np.float64)
        )
        distance_um = float(
            np.linalg.norm(
                shifted_position - current_position
            )
        )
        if distance_um > radius_um:
            continue

        broken_cell_id = int(row["cell_id"])
        broken_volume = lookup_volume(
            volumes_prev,
            broken_cell_id,
        )
        nearby_records.append(
            {
                "node_id": broken_node,
                "cell_id": broken_cell_id,
                "distance_um": distance_um,
                "volume_um3": broken_volume,
            }
        )

    nearby_records.sort(
        key=lambda item: (
            item["distance_um"],
            item["cell_id"],
        )
    )

    nearby_volume_sum = float(
        sum(item["volume_um3"] for item in nearby_records)
    )
    previous_volume_sum = (
        float(wrong_parent_volume) + nearby_volume_sum
    )

    if current_volume_um3 > 0 and previous_volume_sum > 0:
        current_over_previous_sum = (
            float(current_volume_um3)
            / float(previous_volume_sum)
        )
        symmetric_error = (
            abs(
                float(current_volume_um3)
                - float(previous_volume_sum)
            )
            / max(
                float(current_volume_um3),
                float(previous_volume_sum),
            )
        )
    else:
        current_over_previous_sum = float("nan")
        symmetric_error = float("nan")

    current_larger_than_wrong_parent = (
        wrong_parent_unambiguous
        and float(current_volume_um3)
        > float(wrong_parent_volume)
    )

    temporal_pair_signal = bool(
        newborn_after_cut
        and wrong_parent_unambiguous
        and len(nearby_records) >= 1
    )
    volume_sum_close = bool(
        math.isfinite(symmetric_error)
        and symmetric_error <= float(volume_error_max)
    )
    full_signature = bool(
        temporal_pair_signal
        and current_larger_than_wrong_parent
        and volume_sum_close
    )

    failure_reasons = []
    if not newborn_after_cut:
        failure_reasons.append("not_newborn_after_cut")
    if len(removed_prev_parents) == 0:
        failure_reasons.append("no_removed_tminus1_parent")
    elif len(removed_prev_parents) > 1:
        failure_reasons.append("ambiguous_removed_parents")
    if len(nearby_records) == 0:
        failure_reasons.append("no_other_nearby_broken_track")
    if (
        wrong_parent_unambiguous
        and not current_larger_than_wrong_parent
    ):
        failure_reasons.append(
            "current_not_larger_than_wrong_parent"
        )
    if not volume_sum_close:
        failure_reasons.append("previous_volume_sum_not_close")

    base.update(
        {
            "newborn_after_cut": bool(newborn_after_cut),
            "raw_prev_parent_count": int(
                len(raw_prev_parents)
            ),
            "sanitized_prev_parent_count": int(
                len(sanitized_prev_parents)
            ),
            "removed_prev_parent_count": int(
                len(removed_prev_parents)
            ),
            "wrong_parent_unambiguous": bool(
                wrong_parent_unambiguous
            ),
            "wrong_parent_node_id": (
                int(wrong_parent_node)
                if wrong_parent_node is not None
                else -1
            ),
            "wrong_parent_cell_id": int(
                wrong_parent_cell_id
            ),
            "wrong_parent_volume_um3": float(
                wrong_parent_volume
            ),
            "current_over_wrong_parent": (
                float(current_volume_um3 / wrong_parent_volume)
                if wrong_parent_volume > 0
                else float("nan")
            ),
            "nearby_other_broken_count": int(
                len(nearby_records)
            ),
            "nearby_other_broken_cell_ids": json_list(
                [r["cell_id"] for r in nearby_records]
            ),
            "nearby_other_broken_distances_um": json_list(
                [
                    round(float(r["distance_um"]), 6)
                    for r in nearby_records
                ]
            ),
            "nearby_other_broken_volumes_um3": json_list(
                [
                    round(float(r["volume_um3"]), 6)
                    for r in nearby_records
                ]
            ),
            "nearby_other_broken_volume_sum_um3": float(
                nearby_volume_sum
            ),
            "previous_volume_sum_um3": float(
                previous_volume_sum
            ),
            "current_over_previous_sum": float(
                current_over_previous_sum
            ),
            "symmetric_volume_error": float(
                symmetric_error
            ),
            "current_larger_than_wrong_parent": bool(
                current_larger_than_wrong_parent
            ),
            "temporal_pair_signal": bool(
                temporal_pair_signal
            ),
            "volume_sum_close": bool(volume_sum_close),
            "full_signature": bool(full_signature),
            "failure_reason": (
                "PASS"
                if full_signature
                else ";".join(failure_reasons)
            ),
        }
    )
    return base


# =============================================================================
# Radius / volume-tolerance sweep
# =============================================================================


def recompute_signature_at(
    row: pd.Series,
    *,
    radius_dref: float,
    volume_error_max: float,
) -> dict[str, Any]:
    distances = json.loads(
        row["all_other_broken_distances_um"]
    )
    volumes = json.loads(
        row["all_other_broken_volumes_um3"]
    )

    radius_um = float(radius_dref) * float(row["dref_um"])
    selected_volumes = [
        float(v)
        for d, v in zip(distances, volumes)
        if float(d) <= radius_um
    ]
    nearby_count = len(selected_volumes)

    previous_sum = (
        float(row["wrong_parent_volume_um3"])
        + float(sum(selected_volumes))
    )
    current = float(row["current_volume_um3"])

    if current > 0 and previous_sum > 0:
        sym_error = abs(current - previous_sum) / max(
            current,
            previous_sum,
        )
    else:
        sym_error = float("nan")

    temporal = bool(
        row["newborn_after_cut"]
        and row["wrong_parent_unambiguous"]
        and nearby_count >= 1
    )
    larger = bool(
        row["wrong_parent_unambiguous"]
        and current > float(row["wrong_parent_volume_um3"])
    )
    close = bool(
        math.isfinite(sym_error)
        and sym_error <= float(volume_error_max)
    )
    full = bool(temporal and larger and close)

    return {
        "nearby_count": int(nearby_count),
        "previous_sum": float(previous_sum),
        "sym_error": float(sym_error),
        "temporal": bool(temporal),
        "larger": bool(larger),
        "close": bool(close),
        "full": bool(full),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    signature_output = Path(args.signature_output).resolve()
    signature_output.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 128, flush=True)
    print(
        "INVESTIGATION 49 — BREAK → NEWBORN → VOLUME SIGNATURE AUDIT",
        flush=True,
    )
    print("=" * 128, flush=True)
    print(f"repository           : {ROOT}", flush=True)
    print(f"Inv48 source         : {INV48_PATH}", flush=True)
    print(f"sample               : {SAMPLE}", flush=True)
    print(f"signature output     : {signature_output}", flush=True)
    print(
        f"headline radius      : {float(args.nearby_radius_dref):.3f} dref",
        flush=True,
    )
    print(
        f"headline volume error: <= {float(args.volume_rel_error_max):.3f}",
        flush=True,
    )
    print(
        "training             : NONE — audit only",
        flush=True,
    )
    print("=" * 128, flush=True)

    # Reconstruct the exact Inv47/48 hard-cutter state.
    context = INV47.prepare(args)
    loader = context["loader"]
    graph_sanitized = context["graph_sanitized"]
    paths = context["paths"]
    spacing = tuple(float(v) for v in context["spacing"])
    val_frames = tuple(int(v) for v in context["val_frames"])

    voxel_volume_um3 = float(np.prod(np.asarray(spacing)))
    base_movie = np.load(
        paths.base_instances,
        mmap_mode="r",
        allow_pickle=False,
    )

    node_rows, by_time_cell = canonical_node_tables(
        graph_sanitized,
        spacing,
    )
    if not node_rows:
        raise RuntimeError(
            "No canonical temporal nodes could be read from graph_sanitized"
        )

    raw_graph_path = resolve_raw_track_graph_path(paths)
    raw_graph = load_raw_graph(raw_graph_path)

    sanitized_edges = oriented_edges(
        graph_sanitized,
        node_rows,
    )

    # Only edges whose endpoints survived the canonical-node audit are
    # comparable to the sanitized graph.
    raw_edges = oriented_edges(
        raw_graph,
        node_rows,
    )

    removed_edges = raw_edges - sanitized_edges

    print(
        f"[graph] canonical nodes={len(node_rows):,} "
        f"raw comparable edges={len(raw_edges):,} "
        f"sanitized edges={len(sanitized_edges):,} "
        f"removed comparable edges={len(removed_edges):,}",
        flush=True,
    )
    print(
        f"[graph] raw persisted graph: {raw_graph_path}",
        flush=True,
    )

    sanitized_pred, sanitized_succ = (
        predecessor_successor_maps(sanitized_edges)
    )

    # Robust global shift for every consecutive validation transition.
    motion_rows = []
    pair_shift: dict[int, np.ndarray] = {}
    for t in val_frames:
        if int(t) <= 0:
            continue
        shift, audit = robust_global_pair_shift_um(
            t_prev=int(t) - 1,
            t_cur=int(t),
            sanitized_edges=sanitized_edges,
            node_rows=node_rows,
            min_links=int(args.min_global_motion_links),
        )
        pair_shift[int(t)] = shift
        motion_rows.append(audit)
        print(
            f"[motion t={int(t)-1:03d}->{int(t):03d}] "
            f"links={audit['link_count']} "
            f"inliers={audit['inlier_count']} "
            f"|shift|={audit['shift_norm_um']:.3f}um "
            f"resid_med={audit['residual_median_um']:.3f}um",
            flush=True,
        )

    atomic_csv(
        signature_output / "pairwise_global_motion_audit.csv",
        pd.DataFrame(motion_rows),
    )

    # Pre-index true broken endpoints at each frame AFTER hard cutting.
    broken_by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, row in node_rows.items():
        t = int(row["time"])
        if not has_any_future_successor(
            node_id,
            sanitized_succ,
            node_rows,
        ):
            broken_by_frame[t].append(int(node_id))

    # Cache volumes for only the frames needed.
    needed_frames = sorted(
        set(val_frames)
        | {int(t) - 1 for t in val_frames if int(t) > 0}
    )
    volume_cache: dict[int, np.ndarray] = {}
    for t in needed_frames:
        volume_cache[int(t)] = frame_label_volumes_um3(
            np.asarray(base_movie[int(t)]),
            voxel_volume_um3,
        )

    # -------------------------------------------------------------------------
    # First pass: store every broken candidate out to the maximum sweep radius.
    # This lets us recompute the headline and sweep without rerunning setup.
    # -------------------------------------------------------------------------
    max_radius_dref = max(
        max(float(v) for v in args.radius_sweep),
        float(args.nearby_radius_dref),
    )

    all_rows: list[dict[str, Any]] = []

    for ordinal, t in enumerate(val_frames, 1):
        runtime = loader.load(int(t))
        case = INV42.build_real_case(runtime)
        cell_ids = INV48.component_cell_ids(runtime, case)

        trusted = (
            case.split_valid
            & case.metric_component_valid
        ).detach().cpu().numpy().astype(bool)
        target_merge = (
            case.split_target.detach().cpu().numpy()
            > 0.5
        )

        shift = pair_shift[int(t)]
        current_volumes = volume_cache[int(t)]
        prev_volumes = volume_cache[int(t) - 1]
        broken_prev = broken_by_frame[int(t) - 1]

        for component in np.flatnonzero(trusted).tolist():
            component = int(component)
            cell_id = int(cell_ids[component])
            current_node_id, multiplicity = select_component_node(
                by_time_cell,
                t=int(t),
                cell_id=cell_id,
            )
            current_volume = lookup_volume(
                current_volumes,
                cell_id,
            )

            # Compute a broad record first so the sweep can apply any smaller
            # requested radius without recomputing graph geometry.
            row = component_signature_row(
                t=int(t),
                component=component,
                is_merge=bool(target_merge[component]),
                cell_id=cell_id,
                current_node_id=current_node_id,
                duplicate_current_nodes=multiplicity,
                current_volume_um3=current_volume,
                dref_um=float(runtime.dref_um),
                radius_dref=float(max_radius_dref),
                volume_error_max=float(
                    args.volume_rel_error_max
                ),
                node_rows=node_rows,
                raw_edges=raw_edges,
                sanitized_edges=sanitized_edges,
                sanitized_pred=sanitized_pred,
                sanitized_succ=sanitized_succ,
                broken_nodes_prev=broken_prev,
                volumes_prev=prev_volumes,
                global_shift_um=shift,
            )

            # Preserve the broad candidate lists.
            row["all_other_broken_cell_ids"] = row[
                "nearby_other_broken_cell_ids"
            ]
            row["all_other_broken_distances_um"] = row[
                "nearby_other_broken_distances_um"
            ]
            row["all_other_broken_volumes_um3"] = row[
                "nearby_other_broken_volumes_um3"
            ]

            # Recompute the actual headline at the user-selected radius.
            headline = recompute_signature_at(
                pd.Series(row),
                radius_dref=float(args.nearby_radius_dref),
                volume_error_max=float(
                    args.volume_rel_error_max
                ),
            )
            row.update(
                {
                    "nearby_radius_dref": float(
                        args.nearby_radius_dref
                    ),
                    "nearby_radius_um": float(
                        args.nearby_radius_dref
                        * runtime.dref_um
                    ),
                    "nearby_other_broken_count": int(
                        headline["nearby_count"]
                    ),
                    "previous_volume_sum_um3": float(
                        headline["previous_sum"]
                    ),
                    "symmetric_volume_error": float(
                        headline["sym_error"]
                    ),
                    "temporal_pair_signal": bool(
                        headline["temporal"]
                    ),
                    "current_larger_than_wrong_parent": bool(
                        headline["larger"]
                    ),
                    "volume_sum_close": bool(
                        headline["close"]
                    ),
                    "full_signature": bool(
                        headline["full"]
                    ),
                }
            )

            all_rows.append(row)

        print(
            f"[frame {int(t):03d}] "
            f"{ordinal}/{len(val_frames)} "
            f"trusted={int(trusted.sum())} "
            f"merge={int((trusted & target_merge).sum())} "
            f"broken(t-1)={len(broken_prev)}",
            flush=True,
        )

    all_df = pd.DataFrame(all_rows)
    merge_df = all_df[all_df["target_merge"]].copy()
    clean_df = all_df[~all_df["target_merge"]].copy()

    if len(merge_df) != 12:
        raise RuntimeError(
            "Expected the known 12 trusted validation merge components, "
            f"found {len(merge_df)}. Annotation/cache contract changed."
        )

    # -------------------------------------------------------------------------
    # Stage counts — exactly the user's hypothesis, one condition at a time.
    # -------------------------------------------------------------------------
    stage_definitions = [
        (
            "A_newborn_after_cut",
            lambda df: df["newborn_after_cut"].astype(bool),
        ),
        (
            "B_newborn_plus_other_nearby_broken",
            lambda df: (
                df["newborn_after_cut"].astype(bool)
                & (df["nearby_other_broken_count"] >= 1)
            ),
        ),
        (
            "C_B_plus_removed_wrong_parent",
            lambda df: (
                df["newborn_after_cut"].astype(bool)
                & (df["nearby_other_broken_count"] >= 1)
                & df["wrong_parent_unambiguous"].astype(bool)
            ),
        ),
        (
            "D_C_plus_current_larger_than_wrong_parent",
            lambda df: (
                df["newborn_after_cut"].astype(bool)
                & (df["nearby_other_broken_count"] >= 1)
                & df["wrong_parent_unambiguous"].astype(bool)
                & df[
                    "current_larger_than_wrong_parent"
                ].astype(bool)
            ),
        ),
        (
            "E_full_signature_plus_volume_conservation",
            lambda df: df["full_signature"].astype(bool),
        ),
    ]

    stage_rows = []
    for name, predicate in stage_definitions:
        merge_mask = np.asarray(
            predicate(merge_df),
            dtype=bool,
        )
        clean_mask = np.asarray(
            predicate(clean_df),
            dtype=bool,
        )
        tp = int(merge_mask.sum())
        fp = int(clean_mask.sum())
        stage_rows.append(
            {
                "stage": name,
                "merge_hits": tp,
                "merge_total": int(len(merge_df)),
                "merge_recall": float(
                    tp / max(len(merge_df), 1)
                ),
                "clean_hits": fp,
                "clean_total": int(len(clean_df)),
                "clean_rate": float(
                    fp / max(len(clean_df), 1)
                ),
                "precision_among_flagged": float(
                    tp / max(tp + fp, 1)
                ),
            }
        )

    stages_df = pd.DataFrame(stage_rows)

    # -------------------------------------------------------------------------
    # Full radius / volume threshold sweep.
    # -------------------------------------------------------------------------
    sweep_rows = []
    for radius_dref in args.radius_sweep:
        for volume_error in args.volume_error_sweep:
            merge_flags = []
            clean_flags = []

            for _idx, row in merge_df.iterrows():
                result = recompute_signature_at(
                    row,
                    radius_dref=float(radius_dref),
                    volume_error_max=float(volume_error),
                )
                merge_flags.append(bool(result["full"]))

            for _idx, row in clean_df.iterrows():
                result = recompute_signature_at(
                    row,
                    radius_dref=float(radius_dref),
                    volume_error_max=float(volume_error),
                )
                clean_flags.append(bool(result["full"]))

            tp = int(sum(merge_flags))
            fp = int(sum(clean_flags))
            sweep_rows.append(
                {
                    "radius_dref": float(radius_dref),
                    "volume_rel_error_max": float(
                        volume_error
                    ),
                    "merge_hits": tp,
                    "merge_total": int(len(merge_df)),
                    "merge_recall": float(
                        tp / max(len(merge_df), 1)
                    ),
                    "clean_hits": fp,
                    "clean_total": int(len(clean_df)),
                    "clean_rate": float(
                        fp / max(len(clean_df), 1)
                    ),
                    "precision_among_flagged": float(
                        tp / max(tp + fp, 1)
                    ),
                }
            )

    sweep_df = pd.DataFrame(sweep_rows)

    # Human-focused 12-row table.
    merge_columns = [
        "frame",
        "component",
        "current_cell_id",
        "newborn_after_cut",
        "removed_prev_parent_count",
        "wrong_parent_cell_id",
        "wrong_parent_volume_um3",
        "nearby_other_broken_count",
        "nearby_other_broken_cell_ids",
        "nearby_other_broken_distances_um",
        "nearby_other_broken_volumes_um3",
        "current_volume_um3",
        "previous_volume_sum_um3",
        "current_over_wrong_parent",
        "current_over_previous_sum",
        "symmetric_volume_error",
        "current_larger_than_wrong_parent",
        "temporal_pair_signal",
        "volume_sum_close",
        "full_signature",
        "failure_reason",
    ]
    merge_report = merge_df[merge_columns].sort_values(
        ["frame", "component"]
    )

    # Save before printing so partial terminal output still leaves artifacts.
    atomic_csv(
        signature_output
        / "validation_12_merge_signature_audit.csv",
        merge_report,
    )
    atomic_csv(
        signature_output
        / "validation_all_trusted_components_signature_audit.csv",
        all_df,
    )
    atomic_csv(
        signature_output
        / "signature_stage_counts.csv",
        stages_df,
    )
    atomic_csv(
        signature_output
        / "radius_volume_tolerance_sweep.csv",
        sweep_df,
    )

    print("\n" + "=" * 160, flush=True)
    print(
        "INV49 — THE 12 CURATED MERGE EVENTS",
        flush=True,
    )
    print("=" * 160, flush=True)
    display_cols = [
        "frame",
        "component",
        "current_cell_id",
        "newborn_after_cut",
        "wrong_parent_cell_id",
        "nearby_other_broken_count",
        "current_volume_um3",
        "wrong_parent_volume_um3",
        "previous_volume_sum_um3",
        "symmetric_volume_error",
        "current_larger_than_wrong_parent",
        "full_signature",
        "failure_reason",
    ]
    with pd.option_context(
        "display.max_rows",
        50,
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(
            merge_report[display_cols].to_string(index=False),
            flush=True,
        )

    print("\n" + "=" * 128, flush=True)
    print(
        "INV49 — STAGED SIGNATURE COUNTS",
        flush=True,
    )
    print("=" * 128, flush=True)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(stages_df.to_string(index=False), flush=True)

    full_merge_hits = int(
        merge_df["full_signature"].astype(bool).sum()
    )
    full_clean_hits = int(
        clean_df["full_signature"].astype(bool).sum()
    )

    temporal_merge_hits = int(
        merge_df["temporal_pair_signal"].astype(bool).sum()
    )
    temporal_clean_hits = int(
        clean_df["temporal_pair_signal"].astype(bool).sum()
    )

    summary = {
        "investigation": SCRIPT_NAME,
        "sample": SAMPLE,
        "validation_frames": list(val_frames),
        "validation_is_untouched_holdout": False,
        "headline": {
            "nearby_radius_dref": float(
                args.nearby_radius_dref
            ),
            "volume_rel_error_max": float(
                args.volume_rel_error_max
            ),
            "merge_components": int(len(merge_df)),
            "clean_components": int(len(clean_df)),
            "temporal_pair_signal_merge_hits": int(
                temporal_merge_hits
            ),
            "temporal_pair_signal_merge_recall": float(
                temporal_merge_hits / max(len(merge_df), 1)
            ),
            "temporal_pair_signal_clean_hits": int(
                temporal_clean_hits
            ),
            "temporal_pair_signal_clean_rate": float(
                temporal_clean_hits / max(len(clean_df), 1)
            ),
            "full_signature_merge_hits": int(
                full_merge_hits
            ),
            "full_signature_merge_recall": float(
                full_merge_hits / max(len(merge_df), 1)
            ),
            "full_signature_clean_hits": int(
                full_clean_hits
            ),
            "full_signature_clean_rate": float(
                full_clean_hits / max(len(clean_df), 1)
            ),
            "full_signature_precision": float(
                full_merge_hits
                / max(
                    full_merge_hits + full_clean_hits,
                    1,
                )
            ),
        },
        "definitions": {
            "newborn_after_cut": (
                "current canonical node has no accepted predecessor "
                "in sanitized hard-cut graph"
            ),
            "wrong_parent_before_cut": (
                "t-1->t raw Trackastra edge into current node that "
                "is absent after hard cutting"
            ),
            "other_nearby_broken": (
                "t-1 canonical node with no future successor after "
                "hard cutting, excluding wrong parent"
            ),
            "global_shift": (
                "robust median t-1->t displacement of surviving "
                "accepted canonical links"
            ),
            "volume_sum": (
                "wrong-parent volume + other nearby broken-cell volumes"
            ),
            "volume_error": (
                "abs(current-sum)/max(current,sum)"
            ),
        },
        "raw_track_graph": str(raw_graph_path),
        "removed_comparable_edges": int(len(removed_edges)),
        "outputs": {
            "merge_audit_csv": str(
                signature_output
                / "validation_12_merge_signature_audit.csv"
            ),
            "all_components_csv": str(
                signature_output
                / "validation_all_trusted_components_signature_audit.csv"
            ),
            "stage_counts_csv": str(
                signature_output
                / "signature_stage_counts.csv"
            ),
            "sweep_csv": str(
                signature_output
                / "radius_volume_tolerance_sweep.csv"
            ),
            "motion_csv": str(
                signature_output
                / "pairwise_global_motion_audit.csv"
            ),
        },
    }
    atomic_json(
        signature_output / "summary.json",
        summary,
    )

    print("\n" + "=" * 128, flush=True)
    print("INVESTIGATION 49 — HEADLINE", flush=True)
    print("=" * 128, flush=True)
    print(
        f"Temporal pair signal "
        f"(newborn + removed wrong parent + >=1 other nearby broken track): "
        f"{temporal_merge_hits}/{len(merge_df)} merges",
        flush=True,
    )
    print(
        f"Same temporal pair signal among clean controls: "
        f"{temporal_clean_hits}/{len(clean_df)}",
        flush=True,
    )
    print(
        f"FULL signal (+ current>wrong-parent + volume conservation): "
        f"{full_merge_hits}/{len(merge_df)} merges",
        flush=True,
    )
    print(
        f"FULL signal among clean controls: "
        f"{full_clean_hits}/{len(clean_df)}",
        flush=True,
    )
    print(
        f"12-case table : "
        f"{signature_output / 'validation_12_merge_signature_audit.csv'}",
        flush=True,
    )
    print(
        f"threshold sweep: "
        f"{signature_output / 'radius_volume_tolerance_sweep.csv'}",
        flush=True,
    )
    print(
        f"summary       : {signature_output / 'summary.json'}",
        flush=True,
    )
    print("=" * 128, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
