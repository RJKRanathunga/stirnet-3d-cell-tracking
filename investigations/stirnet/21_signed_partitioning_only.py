from __future__ import annotations

r"""
STIR-Net Investigation 21 — signed partitioning only.

Scientific question
-------------------
How much of the current BioHub failure is caused by the partitioner rather than
by the learned RAG edge scores?

This investigation does NOT rerun STIR-Net and does NOT train anything. It
consumes the persistent Investigation-12 RAG artifacts and replaces only the
final graph partitioning rule.

Primary solver: binary multicut with lazy cycle constraints
-----------------------------------------------------------
For edge e with predicted P(same instance)=p_e and configurable neutral
probability q, define

    c_e = logit(p_e) - logit(q)
    y_e = 0  -> endpoints same component
    y_e = 1  -> edge cut

and minimize sum(c_e*y_e) subject to multicut cycle consistency. Positive costs
are attractive; negative costs are repulsive. Cycle constraints are added
lazily using scipy.optimize.milp (HiGHS), already available through SciPy.

The neutral q is intentionally swept because production threshold 0.845 is an
acceptance threshold, not necessarily a calibrated P(same)=0.5 point.

Optional solver: mutex-style greedy signed agglomeration.

Output directories are Investigation-13-compatible, so each variant can be
passed directly as --compare-dir.

Recommended smoke:

    python .\investigations\stirnet\21_signed_partitioning_only.py `
        --inference-dir .\runs\stirnet\evaluation\12_biohub_full_volume_spatial_inference\44b6_0113de3b\morphology_v2_h100\step000600 `
        --timepoints 0-2 `
        --neutral-probabilities 0.50,0.70,0.845

Full run: omit --timepoints or use --timepoints all.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EXPERIMENT_NAME = "21_signed_partitioning_only"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
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
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, value, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_frames(root: Path) -> list[int]:
    rows = []
    for path in root.glob("t[0-9][0-9][0-9]"):
        if path.is_dir() and (path / "_SUCCESS.json").is_file():
            rows.append(int(path.name[1:]))
    if not rows:
        raise FileNotFoundError(f"No completed tXXX frames below {root}")
    return sorted(rows)


def parse_frames(text: str, available: Iterable[int]) -> list[int]:
    available = sorted(int(v) for v in available)
    allowed = set(available)
    token = text.strip().lower()
    if token in {"all", "*"}:
        return available
    selected: set[int] = set()
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            a, b = int(left), int(right)
            if b < a:
                raise ValueError(f"Invalid frame range: {item}")
            selected.update(range(a, b + 1))
        else:
            selected.add(int(item))
    missing = sorted(selected - allowed)
    if missing:
        raise ValueError(f"Frames not completed: {missing}; available={available}")
    if not selected:
        raise ValueError("No frames selected")
    return sorted(selected)


def parse_float_list(text: str, *, name: str) -> list[float]:
    result = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not result or any(not 0.0 < v < 1.0 for v in result):
        raise ValueError(f"{name} must contain values strictly in (0,1)")
    return result


def parse_solver_list(text: str) -> list[str]:
    rows = [v.strip().lower() for v in text.split(",") if v.strip()]
    allowed = {"multicut", "mutex"}
    unknown = sorted(set(rows) - allowed)
    if unknown:
        raise ValueError(f"Unknown solvers: {unknown}; allowed={sorted(allowed)}")
    if not rows:
        raise ValueError("--solvers cannot be empty")
    return list(dict.fromkeys(rows))


def label_token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def components_from_uncut(node_count: int, edges: np.ndarray, cut: np.ndarray) -> np.ndarray:
    dsu = DSU(node_count)
    for row in np.flatnonzero(~cut):
        dsu.union(int(edges[0, row]), int(edges[1, row]))
    roots = [dsu.find(i) for i in range(node_count)]
    mapping: dict[int, int] = {}
    result = np.empty(node_count, dtype=np.int32)
    for i, root in enumerate(roots):
        if root not in mapping:
            mapping[root] = len(mapping)
        result[i] = mapping[root]
    return result


def rasterize_components(watershed: np.ndarray, node_supervoxel_id: np.ndarray, node_component: np.ndarray) -> np.ndarray:
    max_sv = int(max(int(np.max(watershed)) if watershed.size else 0, int(np.max(node_supervoxel_id)) if node_supervoxel_id.size else 0))
    mapping = np.zeros(max_sv + 1, dtype=np.int32)
    if len(node_supervoxel_id) != len(node_component):
        raise ValueError("node_supervoxel_id and node_component length mismatch")
    for sv, comp in zip(node_supervoxel_id, node_component):
        if int(sv) > 0:
            mapping[int(sv)] = int(comp) + 1
    if int(np.max(watershed)) >= len(mapping):
        raise IndexError("Watershed contains a supervoxel ID absent from mapping")
    return mapping[np.asarray(watershed, dtype=np.int64)]


def internal_boundary(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a, b = labels[tuple(lo)], labels[tuple(hi)]
        changed = (a > 0) & (b > 0) & (a != b)
        low_pad = [(0, 0)] * 3
        high_pad = [(0, 0)] * 3
        low_pad[axis] = (0, 1)
        high_pad[axis] = (1, 0)
        boundary |= np.pad(changed, low_pad)
        boundary |= np.pad(changed, high_pad)
    return boundary


def shifted_log_odds(probability: np.ndarray, neutral: float) -> np.ndarray:
    eps = 1e-6
    p = np.clip(np.asarray(probability, dtype=np.float64), eps, 1.0 - eps)
    q = float(np.clip(neutral, eps, 1.0 - eps))
    return np.log(p / (1.0 - p)) - math.log(q / (1.0 - q))


def _uncut_adjacency(node_count: int, edges: np.ndarray, cut: np.ndarray) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for edge_row in np.flatnonzero(~cut):
        u, v = int(edges[0, edge_row]), int(edges[1, edge_row])
        adjacency[u].append((v, int(edge_row)))
        adjacency[v].append((u, int(edge_row)))
    return adjacency


def _find_path_edges(adjacency: list[list[tuple[int, int]]], source: int, target: int) -> list[int] | None:
    if source == target:
        return []
    parent_node = {source: -1}
    parent_edge: dict[int, int] = {}
    queue: deque[int] = deque([source])
    while queue:
        u = queue.popleft()
        for v, edge_row in adjacency[u]:
            if v in parent_node:
                continue
            parent_node[v] = u
            parent_edge[v] = edge_row
            if v == target:
                path: list[int] = []
                cur = target
                while cur != source:
                    path.append(parent_edge[cur])
                    cur = parent_node[cur]
                return list(reversed(path))
            queue.append(v)
    return None


def violated_cycle_constraints(node_count: int, edges: np.ndarray, cut: np.ndarray, *, max_constraints: int) -> list[tuple[int, tuple[int, ...]]]:
    adjacency = _uncut_adjacency(node_count, edges, cut)
    component = components_from_uncut(node_count, edges, cut)
    violations: list[tuple[int, tuple[int, ...]]] = []
    for edge_row in np.flatnonzero(cut):
        u, v = int(edges[0, edge_row]), int(edges[1, edge_row])
        if component[u] != component[v]:
            continue
        path = _find_path_edges(adjacency, u, v)
        if path is not None:
            violations.append((int(edge_row), tuple(int(x) for x in path)))
        if len(violations) >= max_constraints:
            break
    return violations


def solve_multicut(node_count: int, edges: np.ndarray, costs: np.ndarray, *, max_rounds: int, max_constraints_per_round: int, time_limit_seconds: float, mip_rel_gap: float) -> dict[str, Any]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    edge_count = int(edges.shape[1])
    if edge_count == 0:
        return {"cut": np.zeros(0, dtype=bool), "component": np.arange(node_count, dtype=np.int32), "objective": 0.0, "rounds": 0, "constraint_count": 0, "status": "empty_graph", "solver_message": "No edges", "solve_seconds": 0.0}

    rows: list[tuple[int, tuple[int, ...]]] = []
    row_keys: set[tuple[int, tuple[int, ...]]] = set()
    started = time.perf_counter()
    last_result = None
    last_cut = None

    for round_index in range(max_rounds + 1):
        constraints = None
        if rows:
            data, row_index, col_index = [], [], []
            for r, (cut_edge, path_edges) in enumerate(rows):
                row_index.append(r); col_index.append(cut_edge); data.append(1.0)
                for edge_row in path_edges:
                    row_index.append(r); col_index.append(edge_row); data.append(-1.0)
            A = csr_matrix((data, (row_index, col_index)), shape=(len(rows), edge_count), dtype=np.float64)
            constraints = LinearConstraint(A, lb=np.full(len(rows), -np.inf), ub=np.zeros(len(rows)))

        options = {"presolve": True, "disp": False, "mip_rel_gap": float(mip_rel_gap)}
        if time_limit_seconds > 0:
            remaining = max(0.01, float(time_limit_seconds) - (time.perf_counter() - started))
            options["time_limit"] = remaining

        result = milp(
            c=np.asarray(costs, dtype=np.float64),
            integrality=np.ones(edge_count, dtype=np.uint8),
            bounds=Bounds(np.zeros(edge_count), np.ones(edge_count)),
            constraints=constraints,
            options=options,
        )
        last_result = result
        if result.x is None:
            raise RuntimeError(f"HiGHS returned no incumbent: status={result.status}, message={result.message}")
        cut = np.asarray(result.x >= 0.5, dtype=bool)
        last_cut = cut
        violations = violated_cycle_constraints(node_count, edges, cut, max_constraints=max_constraints_per_round)
        new_rows = 0
        for key in violations:
            if key not in row_keys:
                row_keys.add(key); rows.append(key); new_rows += 1
        if not violations:
            component = components_from_uncut(node_count, edges, cut)
            return {"cut": cut, "component": component, "objective": float(np.dot(costs, cut.astype(np.float64))), "rounds": round_index + 1, "constraint_count": len(rows), "status": int(result.status), "solver_message": str(result.message), "solve_seconds": float(time.perf_counter() - started)}
        if new_rows == 0:
            raise RuntimeError("Cycle-separation stalled on repeated violated constraints")
        if time_limit_seconds > 0 and time.perf_counter() - started >= time_limit_seconds:
            break

    remaining = [] if last_cut is None else violated_cycle_constraints(node_count, edges, last_cut, max_constraints=1)
    raise RuntimeError(
        "Multicut cutting-plane loop did not reach cycle consistency: "
        f"rounds={max_rounds}, constraints={len(rows)}, remaining={bool(remaining)}, "
        f"last_status={getattr(last_result, 'status', None)}"
    )


def solve_mutex_greedy(node_count: int, edges: np.ndarray, costs: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    dsu = DSU(node_count)
    mutex: list[set[int]] = [set() for _ in range(node_count)]

    def clean(root: int) -> set[int]:
        root = dsu.find(root)
        mutex[root] = {dsu.find(v) for v in mutex[root] if dsu.find(v) != root}
        return mutex[root]

    def are_mutex(a: int, b: int) -> bool:
        ra, rb = dsu.find(a), dsu.find(b)
        return ra != rb and (rb in clean(ra) or ra in clean(rb))

    def add_mutex(a: int, b: int) -> None:
        ra, rb = dsu.find(a), dsu.find(b)
        if ra != rb:
            mutex[ra].add(rb); mutex[rb].add(ra)

    def merge(a: int, b: int) -> bool:
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            return True
        if are_mutex(ra, rb):
            return False
        neighbors = clean(ra) | clean(rb)
        new_root = dsu.union(ra, rb)
        mutex[new_root] = set()
        for neighbor in neighbors:
            nr = dsu.find(neighbor)
            if nr == new_root:
                continue
            mutex[new_root].add(nr)
            clean(nr)
            mutex[nr].discard(ra); mutex[nr].discard(rb); mutex[nr].add(new_root)
        return True

    for edge_row in np.argsort(-np.abs(costs), kind="stable"):
        u, v = int(edges[0, edge_row]), int(edges[1, edge_row])
        if costs[edge_row] < 0:
            add_mutex(u, v)
        elif costs[edge_row] > 0:
            merge(u, v)

    roots = [dsu.find(i) for i in range(node_count)]
    mapping: dict[int, int] = {}
    component = np.empty(node_count, dtype=np.int32)
    for i, root in enumerate(roots):
        root = dsu.find(root)
        if root not in mapping:
            mapping[root] = len(mapping)
        component[i] = mapping[root]
    cut = component[edges[0]] != component[edges[1]]
    return {"cut": np.asarray(cut, dtype=bool), "component": component, "objective": float(np.dot(costs, cut.astype(np.float64))), "rounds": 1, "constraint_count": 0, "status": "greedy", "solver_message": "mutex-style absolute-confidence greedy", "solve_seconds": float(time.perf_counter() - started)}


def load_frame_graph(frame_dir: Path) -> dict[str, np.ndarray]:
    watershed = np.load(frame_dir / "partition" / "watershed_supervoxels.npy", mmap_mode="r", allow_pickle=False)
    baseline_partition = np.load(frame_dir / "partition" / "spatial_partition.npy", mmap_mode="r", allow_pickle=False)
    with np.load(frame_dir / "rag" / "rag_state.npz", allow_pickle=False) as rag:
        required = ("node_supervoxel_id", "edge_index", "spatial_edge_probability", "partition_node_component")
        missing = [name for name in required if name not in rag]
        if missing:
            raise KeyError(f"{frame_dir}: rag_state.npz missing {missing}")
        result = {
            "watershed": np.asarray(watershed),
            "baseline_partition": np.asarray(baseline_partition),
            "node_supervoxel_id": np.asarray(rag["node_supervoxel_id"], dtype=np.int64),
            "edge_index": np.asarray(rag["edge_index"], dtype=np.int64),
            "probability": np.asarray(rag["spatial_edge_probability"], dtype=np.float64).reshape(-1),
            "baseline_component": np.asarray(rag["partition_node_component"], dtype=np.int64).reshape(-1),
        }
    node_count = len(result["node_supervoxel_id"])
    edges = result["edge_index"]
    if edges.ndim != 2 or edges.shape[0] != 2 or len(result["probability"]) != edges.shape[1]:
        raise ValueError(f"{frame_dir}: malformed RAG arrays")
    if len(result["baseline_component"]) != node_count:
        raise ValueError(f"{frame_dir}: baseline component length mismatch")
    if edges.size and (edges.min() < 0 or edges.max() >= node_count):
        raise IndexError(f"{frame_dir}: edge_index references invalid node")
    return result


def partition_metrics(graph: dict[str, np.ndarray], *, component: np.ndarray, cut: np.ndarray, costs: np.ndarray, neutral: float, signed_partition: np.ndarray) -> dict[str, Any]:
    edges, p = graph["edge_index"], graph["probability"]
    baseline_component = graph["baseline_component"]
    baseline_same = baseline_component[edges[0]] == baseline_component[edges[1]]
    signed_same = component[edges[0]] == component[edges[1]]
    repulsive, attractive = costs < 0, costs > 0
    baseline_boundary = internal_boundary(graph["baseline_partition"])
    signed_boundary = internal_boundary(signed_partition)
    return {
        "node_count": int(len(component)),
        "edge_count": int(edges.shape[1]),
        "neutral_probability": float(neutral),
        "baseline_component_count": int(np.unique(baseline_component).size if len(baseline_component) else 0),
        "signed_component_count": int(np.unique(component).size if len(component) else 0),
        "cut_edge_count": int(np.count_nonzero(cut)),
        "uncut_edge_count": int(np.count_nonzero(~cut)),
        "repulsive_edge_count": int(np.count_nonzero(repulsive)),
        "attractive_edge_count": int(np.count_nonzero(attractive)),
        "repulsive_inside_baseline_count": int(np.count_nonzero(repulsive & baseline_same)),
        "repulsive_inside_signed_count": int(np.count_nonzero(repulsive & signed_same)),
        "attractive_cut_baseline_count": int(np.count_nonzero(attractive & ~baseline_same)),
        "attractive_cut_signed_count": int(np.count_nonzero(attractive & ~signed_same)),
        "rejected_at_production_threshold_inside_baseline_count": int(np.count_nonzero((p < 0.845) & baseline_same)),
        "rejected_at_production_threshold_inside_signed_count": int(np.count_nonzero((p < 0.845) & signed_same)),
        "changed_internal_boundary_voxels": int(np.count_nonzero(baseline_boundary ^ signed_boundary)),
        "baseline_internal_boundary_voxels": int(np.count_nonzero(baseline_boundary)),
        "signed_internal_boundary_voxels": int(np.count_nonzero(signed_boundary)),
        "energy": float(np.dot(costs, cut.astype(np.float64))),
    }


def output_root_for(source_root: Path, source_manifest: dict, explicit_output_root: str | None) -> Path:
    if explicit_output_root:
        return resolve(explicit_output_root)
    sample = str(source_manifest.get("sample_id") or source_manifest.get("sample") or source_root.parent.name)
    run_label = str(source_manifest.get("run_label") or source_root.parent.name or "source")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_label)
    return (ROOT / "runs" / "stirnet" / "evaluation" / EXPERIMENT_NAME / sample / safe).resolve()


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"frame_count": len(rows)}
    if not rows:
        return result
    keys = ("baseline_component_count", "signed_component_count", "repulsive_inside_baseline_count", "repulsive_inside_signed_count", "attractive_cut_baseline_count", "attractive_cut_signed_count", "changed_internal_boundary_voxels", "energy", "solve_seconds")
    for key in keys:
        vals = [float(row[key]) for row in rows]
        result[f"{key}_sum"] = float(sum(vals))
        result[f"{key}_mean"] = float(sum(vals) / len(vals))
    result["frames_with_partition_change"] = int(sum(row["changed_internal_boundary_voxels"] > 0 for row in rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigation 21: signed graph partitioning with fixed saved RAG probabilities.")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--timepoints", default="all")
    parser.add_argument("--neutral-probabilities", default="0.50,0.70,0.845")
    parser.add_argument("--solvers", default="multicut", help='Comma-separated subset of "multicut,mutex".')
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cutting-rounds", type=int, default=50)
    parser.add_argument("--max-constraints-per-round", type=int, default=512)
    parser.add_argument("--milp-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    args = parser.parse_args()

    source_root = resolve(args.inference_dir)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not (source_root / "manifest.json").is_file() or not (source_root / "summary.json").is_file():
        raise FileNotFoundError(f"Expected Investigation-12-style manifest/summary in {source_root}")

    source_manifest = read_json(source_root / "manifest.json")
    source_summary = read_json(source_root / "summary.json")
    frames = parse_frames(args.timepoints, completed_frames(source_root))
    neutrals = parse_float_list(args.neutral_probabilities, name="--neutral-probabilities")
    solvers = parse_solver_list(args.solvers)
    base_output = output_root_for(source_root, source_manifest, args.output_root)

    print("=" * 112)
    print("STIR-Net Investigation 21 — signed partitioning only")
    print("=" * 112)
    print("source inference :", source_root)
    print("frames           :", frames)
    print("solvers          :", solvers)
    print("neutral q sweep  :", neutrals)
    print("output root      :", base_output)
    print("learned weights  : UNCHANGED")
    print("RAG probabilities: UNCHANGED")
    print("=" * 112, flush=True)

    all_variants = {}
    for solver_name in solvers:
        for neutral in neutrals:
            variant = f"{solver_name}_q{label_token(neutral)}"
            variant_root = base_output / variant
            if variant_root.exists() and args.overwrite:
                shutil.rmtree(variant_root)
            if variant_root.exists():
                raise FileExistsError(f"{variant_root} exists; pass --overwrite")
            variant_root.mkdir(parents=True, exist_ok=True)
            frame_rows = []

            for frame in frames:
                source_frame = source_root / f"t{frame:03d}"
                graph = load_frame_graph(source_frame)
                costs = shifted_log_odds(graph["probability"], neutral)
                if solver_name == "multicut":
                    solved = solve_multicut(len(graph["node_supervoxel_id"]), graph["edge_index"], costs, max_rounds=args.max_cutting_rounds, max_constraints_per_round=args.max_constraints_per_round, time_limit_seconds=args.milp_time_limit, mip_rel_gap=args.mip_rel_gap)
                else:
                    solved = solve_mutex_greedy(len(graph["node_supervoxel_id"]), graph["edge_index"], costs)

                signed_partition = rasterize_components(graph["watershed"], graph["node_supervoxel_id"], solved["component"])
                target_frame = variant_root / f"t{frame:03d}"
                link_or_copy(source_frame / "partition" / "watershed_supervoxels.npy", target_frame / "partition" / "watershed_supervoxels.npy")
                atomic_npy(target_frame / "partition" / "spatial_partition.npy", signed_partition.astype(np.int32, copy=False))
                atomic_npz(
                    target_frame / "signed" / "signed_state.npz",
                    node_component=np.asarray(solved["component"], dtype=np.int32),
                    edge_cut=np.asarray(solved["cut"], dtype=np.uint8),
                    edge_probability=graph["probability"].astype(np.float32),
                    signed_cost=costs.astype(np.float32),
                    edge_index=graph["edge_index"].astype(np.int32),
                    node_supervoxel_id=graph["node_supervoxel_id"].astype(np.int32),
                )
                metrics = partition_metrics(graph, component=solved["component"], cut=solved["cut"], costs=costs, neutral=neutral, signed_partition=signed_partition)
                metrics.update({"frame": int(frame), "solver": solver_name, "solver_rounds": int(solved["rounds"]), "solver_constraint_count": int(solved["constraint_count"]), "solver_status": str(solved["status"]), "solver_message": str(solved["solver_message"]), "solve_seconds": float(solved["solve_seconds"])})
                atomic_json(target_frame / "_SUCCESS.json", metrics)
                frame_rows.append(metrics)
                print(f"[{variant}] t{frame:03d} components {metrics['baseline_component_count']} -> {metrics['signed_component_count']} | repulsive-inside {metrics['repulsive_inside_baseline_count']} -> {metrics['repulsive_inside_signed_count']} | changed-boundary={metrics['changed_internal_boundary_voxels']} | {metrics['solve_seconds']:.2f}s", flush=True)

            manifest = dict(source_manifest)
            manifest.update({"experiment": EXPERIMENT_NAME, "run_label": variant, "source_inference_dir": str(source_root), "signed_solver": solver_name, "signed_neutral_probability": float(neutral), "completed_frames": frames, "partition_source": "saved_rag_probabilities_only"})
            summary = {"experiment": EXPERIMENT_NAME, "source_inference_dir": str(source_root), "source_summary": source_summary, "solver": solver_name, "neutral_probability": float(neutral), "frames": frame_rows, **aggregate_summary(frame_rows)}
            atomic_json(variant_root / "manifest.json", manifest)
            atomic_json(variant_root / "summary.json", summary)
            all_variants[variant] = {k: v for k, v in summary.items() if k not in {"frames", "source_summary"}}
            print("[done]", variant_root, flush=True)

    atomic_json(base_output / "sweep_summary.json", {"experiment": EXPERIMENT_NAME, "source_inference_dir": str(source_root), "variants": all_variants})
    print("=" * 112)
    print("Investigation 21 complete:", base_output)
    print("Use any variant directory directly as Investigation-13 --compare-dir.")
    print("=" * 112)


if __name__ == "__main__":
    main()
