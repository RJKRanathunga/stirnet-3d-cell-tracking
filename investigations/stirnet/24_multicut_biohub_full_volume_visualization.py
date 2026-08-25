from __future__ import annotations

r"""
Investigation 24 — production multicut BioHub full-volume A/B visualization.

Scientific purpose
------------------
Investigation 21 showed that signed minimum-cost multicut at q=0.845 fixes the
positive-only union-find transitivity problem. That solver is now installed in
production STIR-Net.

This investigation visualizes ONLY that partitioner change on BioHub:

    SAME h100 checkpoint
    SAME saved watershed supervoxels
    SAME saved RAG nodes / edges
    SAME saved h100 edge probabilities
    ------------------------------------------------
    OLD: threshold(0.845) + union-find
    NEW: production signed multicut, q=0.845

There is deliberately NO dense-model rerun and NO RAG-network rerun. This makes
the comparison exactly attributable to the new partition method and finishes
much faster than another 20-frame full-volume inference.

Why not use Investigation 18 unchanged?
---------------------------------------
Investigation 18 is close and several of its viewing ideas are reused here, but
its yellow/magenta "changed RAG edge" layers mean:

    old checkpoint probability crossed threshold -> new checkpoint probability

For this experiment the probabilities are IDENTICAL by construction, so those
layers would report almost no changes even when multicut changes the final
partition. Investigation 24 therefore visualizes partition-membership changes
directly.

Default baseline
----------------
The historical morphology-v2 h100 BioHub output produced before multicut was
promoted to production:

    runs/stirnet/evaluation/12_biohub_full_volume_spatial_inference/
        44b6_0113de3b/morphology_v2_h100/step000600

Required per frame:
    partition/watershed_supervoxels.npy
    partition/spatial_partition.npy
    rag/rag_state.npz

Output
------
    runs/stirnet/evaluation/24_multicut_biohub_full_volume_visualization/
        44b6_0113de3b/h100_q0p845/

Per frame:
    partition/watershed_supervoxels.npy
    partition/spatial_partition.npy                <- NEW multicut partition
    diff/new_split_boundaries.npy                 <- GREEN
    diff/removed_old_boundaries.npy               <- RED
    diff/changed_components.npy                   <- CYAN mask
    diff/multicut_split_instances.npy             <- NEW labels only where an
                                                      old UF component split
    diff/multicut_merged_instances.npy            <- NEW labels only where
                                                      old UF components merged
    diff/restored_repulsive_supervoxels.npy       <- YELLOW
    diff/sacrificed_attractive_supervoxels.npy    <- MAGENTA
    diff/summary.json

Top-level:
    summary.json
    per_frame.jsonl

Important Napari layers
-----------------------
    OLD union-find partition
    NEW multicut partition

    GREEN   Multicut NEW split boundaries
            A boundary present now but absent in union-find.

    NEW split instances
            Label-colored new multicut instances produced by splitting one
            historical union-find component. This is the easiest layer for
            seeing the "new cells" created by multicut.

    YELLOW  Restored repulsive evidence
            Supervoxels on edges with p<q that union-find nevertheless put in
            the same component, but multicut now separates.

    RED     Old boundaries removed by multicut
            Hidden by default.

    MAGENTA Attractive edges sacrificed by global consistency
            p>=q, union-find connected, multicut separates. Hidden by default.

No BioHub GT is used, so these layers show "multicut corrections/changes", not
guaranteed biological correctness. Visual inspection decides whether a newly
separated object corresponds to a real cell.

Typical command
---------------
From the repository root:

    python .\investigations\stirnet\24_multicut_biohub_full_volume_visualization.py

Materialize comparison without opening Napari:

    python .\investigations\stirnet\24_multicut_biohub_full_volume_visualization.py --no-viewer

Re-open existing artifacts:

    python .\investigations\stirnet\24_multicut_biohub_full_volume_visualization.py --viewer-only

A short frame subset:

    python .\investigations\stirnet\24_multicut_biohub_full_volume_visualization.py --timepoints 0-4
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_NAME = "24_multicut_biohub_full_volume_visualization"
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_NEUTRAL_PROBABILITY = 0.845
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
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


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, np.asarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def link_or_copy_npy(source: Path, target: Path) -> None:
    """Hard-link immutable baseline arrays when possible, otherwise copy."""
    import shutil

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def default_baseline(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "12_biohub_full_volume_spatial_inference"
        / sample_id
        / "morphology_v2_h100"
        / "step000600"
    ).resolve()


def default_output(sample_id: str, q: float) -> Path:
    q_token = f"{q:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / SCRIPT_NAME
        / sample_id
        / f"h100_q{q_token}"
    ).resolve()


def completed_frames(root: Path) -> list[int]:
    rows = []
    for path in root.glob("t[0-9][0-9][0-9]"):
        if (
            path.is_dir()
            and (path / "partition/watershed_supervoxels.npy").is_file()
            and (path / "partition/spatial_partition.npy").is_file()
            and (path / "rag/rag_state.npz").is_file()
        ):
            rows.append(int(path.name[1:]))
    if not rows:
        raise FileNotFoundError(
            f"No complete watershed/partition/RAG frames found below {root}"
        )
    return sorted(rows)


def parse_timepoints(text: str, available: list[int]) -> list[int]:
    token = text.strip().lower()
    if token in {"all", "*"}:
        return list(available)

    allowed = set(available)
    selected: set[int] = set()
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            a, b = int(left), int(right)
            if b < a:
                raise ValueError(f"Invalid timepoint range: {item}")
            selected.update(range(a, b + 1))
        else:
            selected.add(int(item))

    missing = sorted(selected - allowed)
    if missing:
        raise ValueError(
            f"Requested frames are unavailable: {missing}; available={available}"
        )
    if not selected:
        raise ValueError("No timepoints selected")
    return sorted(selected)


def _positive_label_count(labels: np.ndarray) -> int:
    ids = np.unique(np.asarray(labels))
    return int(np.count_nonzero(ids > 0))


def _partition_boundary_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3-D labels, got {labels.shape}")

    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        a = labels[left]
        b = labels[right]
        changed = (a > 0) & (b > 0) & (a != b)
        boundary[left] |= changed
        boundary[right] |= changed
    return boundary


def _sv_component_lookup(
    watershed: np.ndarray,
    partition: np.ndarray,
) -> dict[int, int]:
    """Map each positive watershed SV ID to exactly one partition component."""
    ws = np.asarray(watershed).reshape(-1)
    part = np.asarray(partition).reshape(-1)
    positive = ws > 0
    ws = ws[positive].astype(np.int64, copy=False)
    part = part[positive].astype(np.int64, copy=False)

    if ws.size == 0:
        return {}

    pairs = np.unique(np.stack([ws, part], axis=1), axis=0)
    lookup: dict[int, int] = {}
    for sv_id, component_id in pairs.tolist():
        sv_id = int(sv_id)
        component_id = int(component_id)
        if sv_id in lookup and lookup[sv_id] != component_id:
            raise RuntimeError(
                f"Partition split atomic supervoxel {sv_id}: "
                f"{lookup[sv_id]} vs {component_id}"
            )
        lookup[sv_id] = component_id
    return lookup


def _node_components_from_partition(
    node_supervoxel_id: np.ndarray,
    watershed: np.ndarray,
    partition: np.ndarray,
) -> np.ndarray:
    lookup = _sv_component_lookup(watershed, partition)
    result = np.empty(len(node_supervoxel_id), dtype=np.int64)
    for i, sv_id in enumerate(node_supervoxel_id.tolist()):
        try:
            result[i] = lookup[int(sv_id)]
        except KeyError as exc:
            raise KeyError(
                f"RAG node supervoxel ID {sv_id} is absent from watershed partition"
            ) from exc

    # Compact arbitrary raster component IDs to 0..K-1 for comparisons.
    compact: dict[int, int] = {}
    for i, value in enumerate(result.tolist()):
        if int(value) not in compact:
            compact[int(value)] = len(compact)
        result[i] = compact[int(value)]
    return result


def _rasterize_node_components(
    watershed: np.ndarray,
    node_supervoxel_id: np.ndarray,
    node_component: np.ndarray,
) -> np.ndarray:
    max_sv = int(
        max(
            int(np.max(watershed)) if watershed.size else 0,
            int(np.max(node_supervoxel_id)) if node_supervoxel_id.size else 0,
        )
    )
    mapping = np.zeros(max_sv + 1, dtype=np.int32)
    for sv_id, component in zip(
        node_supervoxel_id.tolist(),
        node_component.tolist(),
    ):
        sv_id = int(sv_id)
        if sv_id > 0:
            mapping[sv_id] = int(component) + 1

    ws = np.asarray(watershed, dtype=np.int64)
    if ws.size and int(ws.max()) >= len(mapping):
        raise IndexError("Watershed supervoxel ID exceeds RAG mapping")
    return mapping[ws]


def _component_relation_maps(
    old_component: np.ndarray,
    new_component: np.ndarray,
) -> tuple[set[int], set[int], dict[int, set[int]], dict[int, set[int]]]:
    """Return old components split by multicut and new components formed by merges."""
    old_to_new: dict[int, set[int]] = {}
    new_to_old: dict[int, set[int]] = {}

    for old, new in zip(old_component.tolist(), new_component.tolist()):
        old_to_new.setdefault(int(old), set()).add(int(new))
        new_to_old.setdefault(int(new), set()).add(int(old))

    split_old = {
        old for old, new_set in old_to_new.items()
        if len(new_set) > 1
    }
    merged_new = {
        new for new, old_set in new_to_old.items()
        if len(old_set) > 1
    }
    return split_old, merged_new, old_to_new, new_to_old


def _masked_new_partition(
    watershed: np.ndarray,
    node_supervoxel_id: np.ndarray,
    new_component: np.ndarray,
    selected_nodes: np.ndarray,
) -> np.ndarray:
    """Label only selected nodes using their NEW multicut component IDs."""
    max_sv = int(
        max(
            int(np.max(watershed)) if watershed.size else 0,
            int(np.max(node_supervoxel_id)) if node_supervoxel_id.size else 0,
        )
    )
    mapping = np.zeros(max_sv + 1, dtype=np.int32)
    for node_index in np.flatnonzero(selected_nodes):
        sv_id = int(node_supervoxel_id[node_index])
        mapping[sv_id] = int(new_component[node_index]) + 1
    return mapping[np.asarray(watershed, dtype=np.int64)]


def _binary_sv_mask(
    watershed: np.ndarray,
    sv_ids: np.ndarray,
) -> np.ndarray:
    if sv_ids.size == 0:
        return np.zeros(watershed.shape, dtype=np.uint8)
    return np.isin(
        np.asarray(watershed),
        np.asarray(sv_ids, dtype=np.int64),
    ).astype(np.uint8)


def _edge_points(
    *,
    edge_rows: np.ndarray,
    edge_index: np.ndarray,
    node_centroid_um: np.ndarray | None,
    volume_shape_zyx: tuple[int, int, int],
    spacing_zyx_um: tuple[float, float, float],
    displayed_time_index: int,
) -> np.ndarray:
    if node_centroid_um is None or edge_rows.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    endpoints = edge_index[:, edge_rows]
    a_um = node_centroid_um[endpoints[0]]
    b_um = node_centroid_um[endpoints[1]]
    midpoint_centered_um = 0.5 * (a_um + b_um)

    spacing = np.asarray(spacing_zyx_um, dtype=np.float32)
    center_voxel = 0.5 * (
        np.asarray(volume_shape_zyx, dtype=np.float32) - 1.0
    )
    midpoint_zyx = midpoint_centered_um / spacing[None, :] + center_voxel[None, :]

    return np.concatenate(
        [
            np.full(
                (midpoint_zyx.shape[0], 1),
                float(displayed_time_index),
                dtype=np.float32,
            ),
            midpoint_zyx.astype(np.float32, copy=False),
        ],
        axis=1,
    )


def materialize_frame(
    *,
    frame: int,
    displayed_time_index: int,
    baseline_root: Path,
    output_root: Path,
    neutral_probability: float,
    spacing_zyx_um: tuple[float, float, float],
    overwrite: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Run the CURRENT PRODUCTION multicut solver on one saved h100 RAG."""
    from learned.stirnet.model.config import PartitionConfig
    from learned.stirnet.model.partition.partitioner import (
        _signed_costs,
        _solve_multicut_graph,
    )

    source_frame = baseline_root / f"t{frame:03d}"
    target_frame = output_root / f"t{frame:03d}"
    success_path = target_frame / "_SUCCESS.json"

    if success_path.is_file() and not overwrite:
        row = json.loads(success_path.read_text(encoding="utf-8"))
        split_points = np.load(
            target_frame / "diff/restored_repulsive_edge_points.npy",
            allow_pickle=False,
        )
        attractive_points = np.load(
            target_frame / "diff/sacrificed_attractive_edge_points.npy",
            allow_pickle=False,
        )
        merged_points = np.load(
            target_frame / "diff/old_separate_new_same_edge_points.npy",
            allow_pickle=False,
        )
        return row, split_points, attractive_points, merged_points

    watershed = np.asarray(
        np.load(
            source_frame / "partition/watershed_supervoxels.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    old_partition = np.asarray(
        np.load(
            source_frame / "partition/spatial_partition.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    )

    with np.load(
        source_frame / "rag/rag_state.npz",
        allow_pickle=False,
    ) as rag:
        required = (
            "node_supervoxel_id",
            "edge_index",
            "spatial_edge_probability",
        )
        missing = [key for key in required if key not in rag]
        if missing:
            raise KeyError(
                f"t{frame:03d} rag_state.npz missing: {missing}"
            )

        node_sv = np.asarray(rag["node_supervoxel_id"], dtype=np.int64)
        edge_index = np.asarray(rag["edge_index"], dtype=np.int64)
        probability = np.asarray(
            rag["spatial_edge_probability"],
            dtype=np.float64,
        ).reshape(-1)
        node_centroid_um = (
            np.asarray(rag["node_centroid_um"], dtype=np.float32)
            if "node_centroid_um" in rag
            else None
        )

    node_count = len(node_sv)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"t{frame:03d}: edge_index must be [2,E], got {edge_index.shape}"
        )
    if edge_index.shape[1] != len(probability):
        raise ValueError(
            f"t{frame:03d}: edge probability length does not match edge_index"
        )
    if edge_index.size and (
        int(edge_index.min()) < 0
        or int(edge_index.max()) >= node_count
    ):
        raise IndexError(f"t{frame:03d}: RAG edge references invalid node")

    old_component = _node_components_from_partition(
        node_sv,
        watershed,
        old_partition,
    )

    # Historical union-find must directly connect every accepted edge. If this
    # is not true, the chosen baseline is not the intended pre-multicut output.
    old_same = (
        old_component[edge_index[0]] == old_component[edge_index[1]]
        if edge_index.shape[1]
        else np.zeros(0, dtype=bool)
    )
    accepted = probability >= float(neutral_probability)
    accepted_cut_old = int(np.count_nonzero(accepted & ~old_same))
    if accepted_cut_old:
        raise RuntimeError(
            f"t{frame:03d}: baseline is not compatible with historical "
            f"threshold+union-find at q={neutral_probability}; "
            f"{accepted_cut_old} accepted edges are cut."
        )

    cfg = PartitionConfig()
    cfg.spatial_partition_backend = "multicut"

    started = time.perf_counter()
    edge_cut, new_component = _solve_multicut_graph(
        node_count=node_count,
        edges=edge_index,
        probability=probability,
        neutral_probability=float(neutral_probability),
        cfg=cfg,
    )
    solve_seconds = time.perf_counter() - started

    new_partition = _rasterize_node_components(
        watershed,
        node_sv,
        new_component,
    )
    new_same = (
        new_component[edge_index[0]] == new_component[edge_index[1]]
        if edge_index.shape[1]
        else np.zeros(0, dtype=bool)
    )

    repulsive = probability < float(neutral_probability)
    attractive = ~repulsive

    # This is the central Problem-1 diagnostic:
    # union-find had these rejected edges inside one connected component, while
    # multicut now puts their endpoints into different components.
    restored_repulsive = repulsive & old_same & ~new_same

    # Multicut may need to cut an individually attractive edge to satisfy the
    # globally stronger signed partition objective.
    sacrificed_attractive = attractive & old_same & ~new_same

    # Also expose the reverse grouping change if it occurs.
    old_separate_new_same = ~old_same & new_same

    split_old, merged_new, old_to_new, new_to_old = _component_relation_maps(
        old_component,
        new_component,
    )
    split_nodes = np.asarray(
        [int(value) in split_old for value in old_component],
        dtype=bool,
    )
    merged_nodes = np.asarray(
        [int(value) in merged_new for value in new_component],
        dtype=bool,
    )

    old_boundary = _partition_boundary_mask(old_partition)
    new_boundary = _partition_boundary_mask(new_partition)
    new_split_boundary = new_boundary & ~old_boundary
    removed_old_boundary = old_boundary & ~new_boundary

    changed_nodes = old_component != old_component  # allocate shape
    # Component integer IDs are arbitrary, so compare membership sets rather
    # than raw IDs.
    changed_nodes[:] = False
    for node in range(node_count):
        old_members = np.flatnonzero(old_component == old_component[node])
        new_members = np.flatnonzero(new_component == new_component[node])
        if not np.array_equal(old_members, new_members):
            changed_nodes[node] = True

    changed_sv = node_sv[changed_nodes]
    restored_sv = np.unique(
        node_sv[
            edge_index[:, np.flatnonzero(restored_repulsive)].reshape(-1)
        ]
    ) if np.any(restored_repulsive) else np.zeros(0, dtype=np.int64)
    sacrificed_sv = np.unique(
        node_sv[
            edge_index[:, np.flatnonzero(sacrificed_attractive)].reshape(-1)
        ]
    ) if np.any(sacrificed_attractive) else np.zeros(0, dtype=np.int64)

    split_instances = _masked_new_partition(
        watershed,
        node_sv,
        new_component,
        split_nodes,
    )
    merged_instances = _masked_new_partition(
        watershed,
        node_sv,
        new_component,
        merged_nodes,
    )
    changed_components = _binary_sv_mask(watershed, changed_sv)
    restored_repulsive_mask = _binary_sv_mask(watershed, restored_sv)
    sacrificed_attractive_mask = _binary_sv_mask(watershed, sacrificed_sv)

    restored_points = _edge_points(
        edge_rows=np.flatnonzero(restored_repulsive),
        edge_index=edge_index,
        node_centroid_um=node_centroid_um,
        volume_shape_zyx=tuple(int(v) for v in watershed.shape),
        spacing_zyx_um=spacing_zyx_um,
        displayed_time_index=displayed_time_index,
    )
    attractive_points = _edge_points(
        edge_rows=np.flatnonzero(sacrificed_attractive),
        edge_index=edge_index,
        node_centroid_um=node_centroid_um,
        volume_shape_zyx=tuple(int(v) for v in watershed.shape),
        spacing_zyx_um=spacing_zyx_um,
        displayed_time_index=displayed_time_index,
    )
    merged_points = _edge_points(
        edge_rows=np.flatnonzero(old_separate_new_same),
        edge_index=edge_index,
        node_centroid_um=node_centroid_um,
        volume_shape_zyx=tuple(int(v) for v in watershed.shape),
        spacing_zyx_um=spacing_zyx_um,
        displayed_time_index=displayed_time_index,
    )

    # Persist Investigation-13-like partition layout plus multicut-specific
    # difference artifacts.
    link_or_copy_npy(
        source_frame / "partition/watershed_supervoxels.npy",
        target_frame / "partition/watershed_supervoxels.npy",
    )
    atomic_npy(
        target_frame / "partition/spatial_partition.npy",
        new_partition.astype(np.int32, copy=False),
    )
    atomic_npy(
        target_frame / "diff/new_split_boundaries.npy",
        new_split_boundary.astype(np.uint8),
    )
    atomic_npy(
        target_frame / "diff/removed_old_boundaries.npy",
        removed_old_boundary.astype(np.uint8),
    )
    atomic_npy(
        target_frame / "diff/changed_components.npy",
        changed_components,
    )
    atomic_npy(
        target_frame / "diff/multicut_split_instances.npy",
        split_instances.astype(np.int32, copy=False),
    )
    atomic_npy(
        target_frame / "diff/multicut_merged_instances.npy",
        merged_instances.astype(np.int32, copy=False),
    )
    atomic_npy(
        target_frame / "diff/restored_repulsive_supervoxels.npy",
        restored_repulsive_mask,
    )
    atomic_npy(
        target_frame / "diff/sacrificed_attractive_supervoxels.npy",
        sacrificed_attractive_mask,
    )
    atomic_npy(
        target_frame / "diff/restored_repulsive_edge_points.npy",
        restored_points,
    )
    atomic_npy(
        target_frame / "diff/sacrificed_attractive_edge_points.npy",
        attractive_points,
    )
    atomic_npy(
        target_frame / "diff/old_separate_new_same_edge_points.npy",
        merged_points,
    )

    costs = _signed_costs(
        probability,
        neutral_probability=float(neutral_probability),
        epsilon=float(cfg.multicut_probability_epsilon),
    )

    old_cut = ~old_same
    new_cut = ~new_same
    old_energy = float(np.dot(costs, old_cut.astype(np.float64)))
    new_energy = float(np.dot(costs, new_cut.astype(np.float64)))

    row = {
        "timepoint": int(frame),
        "neutral_probability": float(neutral_probability),
        "node_count": int(node_count),
        "edge_count": int(edge_index.shape[1]),
        "old_union_find_instance_count": _positive_label_count(old_partition),
        "new_multicut_instance_count": _positive_label_count(new_partition),
        "instance_delta_new_minus_old": (
            _positive_label_count(new_partition)
            - _positive_label_count(old_partition)
        ),
        "old_component_count": int(np.unique(old_component).size),
        "new_component_count": int(np.unique(new_component).size),
        "old_components_split_by_multicut": int(len(split_old)),
        "new_components_formed_from_multiple_old_components": int(len(merged_new)),
        "new_split_boundary_voxel_count": int(np.count_nonzero(new_split_boundary)),
        "removed_old_boundary_voxel_count": int(np.count_nonzero(removed_old_boundary)),
        "changed_node_count": int(np.count_nonzero(changed_nodes)),
        "repulsive_edge_count": int(np.count_nonzero(repulsive)),
        "attractive_edge_count": int(np.count_nonzero(attractive)),
        "repulsive_inside_old_union_find": int(
            np.count_nonzero(repulsive & old_same)
        ),
        "repulsive_inside_new_multicut": int(
            np.count_nonzero(repulsive & new_same)
        ),
        "restored_repulsive_edge_count": int(
            np.count_nonzero(restored_repulsive)
        ),
        "sacrificed_attractive_edge_count": int(
            np.count_nonzero(sacrificed_attractive)
        ),
        "old_separate_new_same_edge_count": int(
            np.count_nonzero(old_separate_new_same)
        ),
        "old_signed_energy": old_energy,
        "new_signed_energy": new_energy,
        "signed_energy_improvement_old_minus_new": old_energy - new_energy,
        "multicut_solve_seconds": float(solve_seconds),
        "old_to_new_component_cardinality": {
            str(old): len(values)
            for old, values in old_to_new.items()
            if len(values) > 1
        },
        "new_to_old_component_cardinality": {
            str(new): len(values)
            for new, values in new_to_old.items()
            if len(values) > 1
        },
    }

    atomic_json(target_frame / "diff/summary.json", row)
    atomic_json(success_path, row)
    return row, restored_points, attractive_points, merged_points


def materialize_run(
    *,
    baseline_root: Path,
    output_root: Path,
    frames: list[int],
    neutral_probability: float,
    spacing_zyx_um: tuple[float, float, float],
    overwrite: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    per_frame_path = output_root / "per_frame.jsonl"
    per_frame_path.unlink(missing_ok=True)

    rows = []
    restored_points = []
    attractive_points = []
    merged_points = []

    print("=" * 118)
    print("INVESTIGATION 24 — PRODUCTION MULTICUT BIOHUB A/B")
    print("=" * 118)
    print("baseline union-find :", baseline_root)
    print("output multicut     :", output_root)
    print("frames              :", frames)
    print("neutral q           :", neutral_probability)
    print("dense inference     : NOT rerun")
    print("RAG network         : NOT rerun")
    print("edge probabilities  : IDENTICAL old/new")
    print("=" * 118, flush=True)

    for displayed_index, frame in enumerate(frames):
        row, split_pts, attr_pts, merge_pts = materialize_frame(
            frame=frame,
            displayed_time_index=displayed_index,
            baseline_root=baseline_root,
            output_root=output_root,
            neutral_probability=neutral_probability,
            spacing_zyx_um=spacing_zyx_um,
            overwrite=overwrite,
        )
        rows.append(row)
        append_jsonl(per_frame_path, row)
        if split_pts.size:
            restored_points.append(split_pts)
        if attr_pts.size:
            attractive_points.append(attr_pts)
        if merge_pts.size:
            merged_points.append(merge_pts)

        print(
            f"[t{frame:03d}] "
            f"instances {row['old_union_find_instance_count']} -> "
            f"{row['new_multicut_instance_count']} | "
            f"repulsive-inside {row['repulsive_inside_old_union_find']} -> "
            f"{row['repulsive_inside_new_multicut']} | "
            f"restored={row['restored_repulsive_edge_count']} | "
            f"split-old-components={row['old_components_split_by_multicut']} | "
            f"{row['multicut_solve_seconds']:.3f}s",
            flush=True,
        )

    def total(key: str):
        return sum(row[key] for row in rows)

    summary = {
        "status": "success",
        "experiment": SCRIPT_NAME,
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "frames": frames,
        "frame_count": len(frames),
        "neutral_probability": float(neutral_probability),
        "comparison_invariant": (
            "same watershed + same h100 RAG probabilities; partitioner only"
        ),
        "old_union_find_instance_count_total": int(
            total("old_union_find_instance_count")
        ),
        "new_multicut_instance_count_total": int(
            total("new_multicut_instance_count")
        ),
        "instance_delta_total": int(total("instance_delta_new_minus_old")),
        "old_components_split_by_multicut_total": int(
            total("old_components_split_by_multicut")
        ),
        "new_components_formed_from_multiple_old_components_total": int(
            total("new_components_formed_from_multiple_old_components")
        ),
        "repulsive_inside_old_union_find_total": int(
            total("repulsive_inside_old_union_find")
        ),
        "repulsive_inside_new_multicut_total": int(
            total("repulsive_inside_new_multicut")
        ),
        "restored_repulsive_edge_count_total": int(
            total("restored_repulsive_edge_count")
        ),
        "sacrificed_attractive_edge_count_total": int(
            total("sacrificed_attractive_edge_count")
        ),
        "old_separate_new_same_edge_count_total": int(
            total("old_separate_new_same_edge_count")
        ),
        "new_split_boundary_voxel_count_total": int(
            total("new_split_boundary_voxel_count")
        ),
        "removed_old_boundary_voxel_count_total": int(
            total("removed_old_boundary_voxel_count")
        ),
        "old_signed_energy_total": float(total("old_signed_energy")),
        "new_signed_energy_total": float(total("new_signed_energy")),
        "signed_energy_improvement_total": float(
            total("signed_energy_improvement_old_minus_new")
        ),
        "multicut_solve_seconds_total": float(
            total("multicut_solve_seconds")
        ),
        "per_frame": rows,
    }
    atomic_json(output_root / "summary.json", summary)

    # Top-level point arrays make viewer-only mode independent of selected
    # source frame paths.
    atomic_npy(
        output_root / "restored_repulsive_edge_points.npy",
        np.concatenate(restored_points, axis=0)
        if restored_points
        else np.zeros((0, 4), dtype=np.float32),
    )
    atomic_npy(
        output_root / "sacrificed_attractive_edge_points.npy",
        np.concatenate(attractive_points, axis=0)
        if attractive_points
        else np.zeros((0, 4), dtype=np.float32),
    )
    atomic_npy(
        output_root / "old_separate_new_same_edge_points.npy",
        np.concatenate(merged_points, axis=0)
        if merged_points
        else np.zeros((0, 4), dtype=np.float32),
    )

    print("=" * 118)
    print("INVESTIGATION 24 SUMMARY")
    print("=" * 118)
    print(
        "instances total          :",
        summary["old_union_find_instance_count_total"],
        "->",
        summary["new_multicut_instance_count_total"],
        f"(delta={summary['instance_delta_total']:+d})",
    )
    print(
        "repulsive inside comps   :",
        summary["repulsive_inside_old_union_find_total"],
        "->",
        summary["repulsive_inside_new_multicut_total"],
    )
    print(
        "restored repulsive edges :",
        summary["restored_repulsive_edge_count_total"],
    )
    print(
        "old components split     :",
        summary["old_components_split_by_multicut_total"],
    )
    print(
        "attractive edges cut     :",
        summary["sacrificed_attractive_edge_count_total"],
    )
    print(
        "signed energy improvement:",
        f"{summary['signed_energy_improvement_total']:.6f}",
    )
    print(
        "multicut solver total    :",
        f"{summary['multicut_solve_seconds_total']:.3f}s",
    )
    print("summary                   :", output_root / "summary.json")
    print("=" * 118)
    return summary


def open_viewer(
    *,
    sample_id: str,
    baseline_root: Path,
    output_root: Path,
    frames: list[int],
    stage6_root_override: str | None,
    sample_zarr_override: str | None,
    spacing_zyx_um: tuple[float, float, float],
) -> None:
    viewer_script = (
        ROOT
        / "investigations"
        / "stirnet"
        / "data"
        / "13_biohub_full_volume_spatial_results_viewer.py"
    )
    V13 = _load_module(
        viewer_script,
        "_investigation13_for_investigation24",
    )

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is not installed. Install visualization dependencies "
            "or run with --no-viewer."
        ) from exc

    stage6_root = V13.resolve_stage6_root(
        sample_id,
        stage6_root_override,
    )
    sample_zarr = V13.resolve_sample_zarr(
        sample_id,
        sample_zarr_override,
    )

    scale_4d = (1.0, *spacing_zyx_um)

    raw, _ = V13.load_raw_time_series(sample_zarr, frames)
    preprocessed, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "preprocessing"),
        name="Stage-6 preprocessing",
    )
    source_labels, _ = V13.stack_npy(
        V13.stage6_paths(stage6_root, frames, "segmentation"),
        name="Stage-6 source segmentation",
    )

    watershed, _ = V13.stack_npy(
        V13.inference_paths(
            baseline_root,
            frames,
            "partition/watershed_supervoxels.npy",
        ),
        name="shared watershed supervoxels",
    )
    old_partition, _ = V13.stack_npy(
        V13.inference_paths(
            baseline_root,
            frames,
            "partition/spatial_partition.npy",
        ),
        name="old union-find partition",
    )
    new_partition, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "partition/spatial_partition.npy",
        ),
        name="new multicut partition",
    )

    new_split_boundaries, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/new_split_boundaries.npy",
        ),
        name="multicut new split boundaries",
    )
    removed_boundaries, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/removed_old_boundaries.npy",
        ),
        name="multicut removed boundaries",
    )
    changed_components, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/changed_components.npy",
        ),
        name="changed components",
    )
    split_instances, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/multicut_split_instances.npy",
        ),
        name="new split instances",
    )
    merged_instances, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/multicut_merged_instances.npy",
        ),
        name="new merged instances",
    )
    restored_repulsive, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/restored_repulsive_supervoxels.npy",
        ),
        name="restored repulsive supervoxels",
    )
    sacrificed_attractive, _ = V13.stack_npy(
        V13.inference_paths(
            output_root,
            frames,
            "diff/sacrificed_attractive_supervoxels.npy",
        ),
        name="sacrificed attractive supervoxels",
    )

    restored_points = np.load(
        output_root / "restored_repulsive_edge_points.npy",
        allow_pickle=False,
    )
    attractive_points = np.load(
        output_root / "sacrificed_attractive_edge_points.npy",
        allow_pickle=False,
    )
    merged_points = np.load(
        output_root / "old_separate_new_same_edge_points.npy",
        allow_pickle=False,
    )

    viewer = napari.Viewer(
        title=(
            f"STIR-Net Multicut A/B | {sample_id} | "
            f"union-find -> signed multicut q={DEFAULT_NEUTRAL_PROBABILITY}"
        )
    )
    viewer.add_image(
        raw,
        name="Raw BioHub volume",
        scale=scale_4d,
        colormap="gray",
        visible=False,
    )
    viewer.add_image(
        preprocessed,
        name="Stage-6 preprocessing",
        scale=scale_4d,
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=True,
    )
    viewer.add_labels(
        source_labels,
        name="Stage-6 source segmentation",
        scale=scale_4d,
        opacity=0.35,
        visible=False,
    )
    viewer.add_labels(
        watershed,
        name="Shared watershed supervoxels",
        scale=scale_4d,
        opacity=0.42,
        visible=False,
    )
    viewer.add_labels(
        old_partition,
        name="OLD h100 union-find partition",
        scale=scale_4d,
        opacity=0.58,
        visible=False,
    )
    viewer.add_labels(
        new_partition,
        name="NEW h100 MULTICUT partition",
        scale=scale_4d,
        opacity=0.48,
        visible=True,
    )

    # The most useful layer for seeing the newly separated cell instances:
    # each new multicut component gets its own label color, while unaffected
    # components are transparent/background.
    viewer.add_labels(
        split_instances,
        name="MULTICUT new split instances ONLY",
        scale=scale_4d,
        opacity=0.78,
        visible=True,
    )

    viewer.add_image(
        new_split_boundaries,
        name="MULTICUT NEW boundaries (green)",
        scale=scale_4d,
        colormap="green",
        contrast_limits=(0.0, 1.0),
        opacity=1.0,
        blending="additive",
        visible=True,
    )
    viewer.add_image(
        restored_repulsive,
        name="MULTICUT restored repulsive evidence (yellow)",
        scale=scale_4d,
        colormap="yellow",
        contrast_limits=(0.0, 1.0),
        opacity=0.28,
        blending="additive",
        visible=True,
    )
    viewer.add_image(
        changed_components,
        name="MULTICUT all changed components (cyan)",
        scale=scale_4d,
        colormap="cyan",
        contrast_limits=(0.0, 1.0),
        opacity=0.20,
        blending="additive",
        visible=False,
    )
    viewer.add_image(
        removed_boundaries,
        name="MULTICUT removed OLD boundaries (red)",
        scale=scale_4d,
        colormap="red",
        contrast_limits=(0.0, 1.0),
        opacity=1.0,
        blending="additive",
        visible=False,
    )
    viewer.add_labels(
        merged_instances,
        name="MULTICUT newly merged instances ONLY",
        scale=scale_4d,
        opacity=0.70,
        visible=False,
    )
    viewer.add_image(
        sacrificed_attractive,
        name="MULTICUT globally cut attractive edges (magenta)",
        scale=scale_4d,
        colormap="magenta",
        contrast_limits=(0.0, 1.0),
        opacity=0.30,
        blending="additive",
        visible=False,
    )

    if restored_points.size:
        viewer.add_points(
            restored_points,
            name="Restored repulsive edge midpoints",
            scale=scale_4d,
            size=6.0,
            face_color="yellow",
            border_color="black",
            opacity=0.95,
            visible=True,
        )
    if attractive_points.size:
        viewer.add_points(
            attractive_points,
            name="Globally sacrificed attractive edge midpoints",
            scale=scale_4d,
            size=6.0,
            face_color="magenta",
            border_color="black",
            opacity=0.95,
            visible=False,
        )
    if merged_points.size:
        viewer.add_points(
            merged_points,
            name="Old-separate -> multicut-same edge midpoints",
            scale=scale_4d,
            size=6.0,
            face_color="cyan",
            border_color="black",
            opacity=0.95,
            visible=False,
        )

    summary = json.loads(
        (output_root / "summary.json").read_text(encoding="utf-8")
    )
    change_by_frame = {
        int(row["timepoint"]): (
            int(row["new_split_boundary_voxel_count"])
            + int(row["removed_old_boundary_voxel_count"])
        )
        for row in summary["per_frame"]
    }
    strongest_frame = max(
        frames,
        key=lambda frame: change_by_frame.get(frame, 0),
    )
    start_index = frames.index(strongest_frame)

    viewer.dims.set_current_step(0, start_index)
    viewer.dims.ndisplay = 3

    print(
        "[viewer] starting on strongest multicut-change frame: "
        f"t{strongest_frame:03d} "
        f"(boundary-change voxels={change_by_frame[strongest_frame]:,})",
        flush=True,
    )
    print(
        "[viewer] IMPORTANT:\n"
        "  NEW h100 MULTICUT partition\n"
        "      = complete new instance segmentation.\n"
        "  MULTICUT new split instances ONLY\n"
        "      = only new components created by splitting an old union-find component.\n"
        "  GREEN\n"
        "      = new internal cell boundary introduced by multicut.\n"
        "  YELLOW\n"
        "      = rejected/repulsive RAG evidence that union-find had overridden\n"
        "        through transitive connectivity and multicut now restores.\n"
        "  RED (hidden)\n"
        "      = old boundary removed by multicut.\n"
        "  MAGENTA (hidden)\n"
        "      = individually attractive edge cut by the global multicut solution.",
        flush=True,
    )
    print(
        "[viewer] T index -> BioHub frame: "
        + ", ".join(
            f"{i}->t{frame:03d}" for i, frame in enumerate(frames)
        ),
        flush=True,
    )

    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare historical h100 threshold+union-find against the current "
            "production signed multicut using identical saved BioHub RAG scores."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timepoints", default="all")
    parser.add_argument(
        "--neutral-probability",
        type=float,
        default=DEFAULT_NEUTRAL_PROBABILITY,
    )
    parser.add_argument(
        "--spacing",
        default=",".join(str(v) for v in DEFAULT_SPACING_ZYX_UM),
        help="Physical spacing Z,Y,X in um for point visualization.",
    )
    parser.add_argument("--stage6-root", default=None)
    parser.add_argument("--sample-zarr", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viewer-only", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.neutral_probability < 1.0:
        raise ValueError("--neutral-probability must lie in (0,1)")

    spacing = tuple(
        float(token.strip()) for token in args.spacing.split(",")
    )
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("--spacing must contain three positive Z,Y,X values")

    baseline_root = (
        resolve(args.baseline_dir)
        if args.baseline_dir
        else default_baseline(args.sample_id)
    )
    if not baseline_root.is_dir():
        raise FileNotFoundError(
            f"Historical h100 union-find baseline not found: {baseline_root}\n"
            "Pass --baseline-dir explicitly."
        )

    available = completed_frames(baseline_root)
    frames = parse_timepoints(args.timepoints, available)

    output_root = (
        resolve(args.output_dir)
        if args.output_dir
        else default_output(args.sample_id, args.neutral_probability)
    )

    if args.viewer_only:
        if not (output_root / "summary.json").is_file():
            raise FileNotFoundError(
                f"--viewer-only requested but no summary exists: {output_root}"
            )
    else:
        materialize_run(
            baseline_root=baseline_root,
            output_root=output_root,
            frames=frames,
            neutral_probability=args.neutral_probability,
            spacing_zyx_um=spacing,
            overwrite=args.overwrite,
        )

    if not args.no_viewer:
        open_viewer(
            sample_id=args.sample_id,
            baseline_root=baseline_root,
            output_root=output_root,
            frames=frames,
            stage6_root_override=args.stage6_root,
            sample_zarr_override=args.sample_zarr,
            spacing_zyx_um=spacing,
        )


if __name__ == "__main__":
    main()
