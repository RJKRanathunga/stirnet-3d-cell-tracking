from __future__ import annotations

# DATASET_CURATION_REFACTOR_CURRENT_V1: migrated current merge-suspect exporter

r"""
BioHub merge-suspect exporter for the supervoxel instance annotator.

# STIRNET_MERGE_SUSPECT_EXPORT_V1

Purpose
-------
Produce a compact, threshold-independent suspect score for every CURRENT
Investigation-25 predicted instance. The output is consumed by
02_supervoxel_instance_annotator.py.

No-leak rule
------------
Manual corrected labels are NEVER read by this script. Inference is reconstructed
from Investigation-25 current labels, Investigation-24 atomic supervoxels,
Investigation-12 compact RAG state, and temporal-v4 caches built from the current
spatial movie itself.

Scoring backend
---------------
V1 uses the Investigation-31 causal temporal checkpoint contract. This is the
existing leak-free graph-only inference path whose instance IDs align exactly
with Investigation-25. It does not silently use Investigation-35 prepared
observer caches, because those caches were prepared around manual/synthetic
candidate locations. A future online production-observer backend can emit the
same tXXX.npz contract without changing the annotator.

Score per current instance i
----------------------------
    split_probability(i) = sigmoid(FULL split logit)

    graph_cut_evidence(i) = top-k mean of FULL internal-edge CUT probabilities

    causal_cut_evidence(i) = top-k mean of positive FULL-vs-
        max(SHUFFLED, CONTENTLESS) internal-edge CUT gains

    suspect_score(i) =
        (w_split*split + w_graph*graph + w_causal*causal)
        * (support_floor + (1-support_floor)*temporal_support)

All scores are exported. The display threshold is applied only by 02_.

Default output
--------------
    evaluation/segmentation/suspects/<sample-id>/
        manifest.json
        t000.npz
        t001.npz
        ...

Typical usage
-------------
    python .\evaluation\segmentation\scripts\03_biohub_merge_suspect_export.py ^
        --sample-id 44b6_0113de3b --timepoints all

Then:
    python .\evaluation\segmentation\scripts\02_supervoxel_instance_annotator.py ^
        --timepoints all --suspect-threshold 0.70
"""

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    raise RuntimeError("Could not resolve cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV30 = load_module(
    ROOT / "investigations/stirnet/30_biohub_temporal_partition_overfit.py",
    "_stirnet_inv30_for_suspects",
)
INV31 = load_module(
    ROOT / "investigations/stirnet/31_biohub_causal_temporal_overfit.py",
    "_stirnet_inv31_for_suspects",
)

from learned.stirnet.data.trackastra_cache import load_cache


SCRIPT_NAME = "03_biohub_merge_suspect_export"
DEFAULT_SAMPLE_ID = INV30.DEFAULT_SAMPLE_ID
DEFAULT_SPACING_ZYX_UM = INV30.DEFAULT_SPACING_ZYX_UM
DEFAULT_SPATIAL_PRIOR_LOGIT = INV30.DEFAULT_SPATIAL_PRIOR_LOGIT
DEFAULT_CHECKPOINT = (
    ROOT / "runs" / "stirnet" / "evaluation"
    / "31_biohub_causal_temporal_overfit"
    / INV30.DEFAULT_SAMPLE_ID / "best.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "evaluation" / "segmentation" / "suspects"
SCORE_FORMAT_VERSION = 1
SCORE_METHOD = "inv31_causal_instance_graph_v1"


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        if value.ndim == 0:
            return jsonable(value.detach().cpu().item())
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def parse_device(text: str) -> torch.device:
    token = str(text).strip().lower()
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(token)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def completed_spatial_frames(root: Path) -> list[int]:
    result = []
    if not root.is_dir():
        return result
    for frame_dir in root.glob("t[0-9][0-9][0-9]"):
        if (frame_dir / "partition" / "after_split_only.npy").is_file():
            result.append(int(frame_dir.name[1:]))
    return sorted(result)


def parse_timepoints(text: str, available: list[int]) -> list[int]:
    token = str(text).strip().lower()
    if token in {"all", "*"}:
        return list(available)
    selected: set[int] = set()
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            first, last = int(left), int(right)
            if last < first:
                raise ValueError(f"Invalid timepoint range: {item}")
            selected.update(range(first, last + 1))
        else:
            selected.add(int(item))
    if not selected:
        raise ValueError("No timepoints selected.")
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(f"Unavailable frames {missing}; available={available}")
    return sorted(selected)


def topk_mean(values: np.ndarray, k: int) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    count = min(max(int(k), 1), int(values.size))
    if count == values.size:
        return float(values.mean())
    index = np.argpartition(values, values.size - count)[-count:]
    return float(values[index].mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-instance BioHub merge-suspect scores."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--timepoints", default="all")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inv12", type=Path, default=None)
    parser.add_argument("--inv24", type=Path, default=None)
    parser.add_argument("--inv25", type=Path, default=None)
    parser.add_argument("--zarr", type=Path, default=None)
    parser.add_argument("--temporal-work-root", type=Path, default=None)
    parser.add_argument(
        "--spacing-zyx-um",
        type=float,
        nargs=3,
        default=DEFAULT_SPACING_ZYX_UM,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--dref-um", type=float, default=None)
    parser.add_argument(
        "--spatial-prior-logit",
        type=float,
        default=DEFAULT_SPATIAL_PRIOR_LOGIT,
    )
    parser.add_argument("--trackastra-model", default="ctc")
    parser.add_argument("--trackastra-mode", default="greedy")
    parser.add_argument("--trackastra-device", default="cuda")
    parser.add_argument("--complete-candidate-graph", action="store_true")
    parser.add_argument("--rebuild-temporal", action="store_true")
    parser.add_argument("--top-k-edges", type=int, default=3)
    parser.add_argument("--split-weight", type=float, default=0.55)
    parser.add_argument("--graph-weight", type=float, default=0.30)
    parser.add_argument("--causal-weight", type=float, default=0.15)
    parser.add_argument("--support-floor", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-scores", action="store_true")
    parser.add_argument("--print-top", type=int, default=10)
    return parser.parse_args()


def make_inv30_paths(args: argparse.Namespace):
    output = (
        resolve(args.temporal_work_root)
        if args.temporal_work_root is not None
        else (INV30.DEFAULT_OUTPUT_ROOT / args.sample_id).resolve()
    )
    namespace = argparse.Namespace(
        sample_id=str(args.sample_id),
        inv12=args.inv12,
        inv24=args.inv24,
        inv25=args.inv25,
        annotations=None,
        zarr=args.zarr,
        output=output,
    )
    return INV30.make_paths(namespace)


def validate_spatial_artifacts(paths, frame_count: int) -> None:
    missing: list[Path] = []
    for t in range(frame_count):
        for path in (
            paths.rag_state(t),
            paths.supervoxels(t),
            paths.base_instances(t),
        ):
            if not path.is_file():
                missing.append(path)
    if not paths.zarr.exists():
        missing.append(paths.zarr)
    if missing:
        preview = "\n".join(f"  {p}" for p in missing[:40])
        suffix = f"\n  ... and {len(missing)-40} more" if len(missing) > 40 else ""
        raise FileNotFoundError(
            "Suspect export requires current spatial artifacts only; missing:\n"
            + preview + suffix
        )


def resolve_dref_and_base_movie(
    paths,
    frame_count: int,
    spacing: tuple[float, float, float],
    override: float | None,
    *,
    rebuild_movie: bool,
) -> tuple[float, Path]:
    base_path = INV30.assemble_base_instance_movie(
        paths, frame_count, rebuild=bool(rebuild_movie)
    )
    base_movie = np.load(base_path, mmap_mode="r")
    dref_um = (
        float(override)
        if override is not None
        else float(INV30.resolve_movie_dref_um(base_movie, spacing))
    )
    if not math.isfinite(dref_um) or dref_um <= 0:
        raise RuntimeError(f"Invalid dref_um: {dref_um}")
    return dref_um, base_path


def ensure_temporal_caches(
    paths,
    *,
    frame_count: int,
    spacing: tuple[float, float, float],
    dref_um: float,
    base_movie_path: Path,
    args: argparse.Namespace,
) -> None:
    ready = all(paths.temporal_cache(t).is_file() for t in range(frame_count))
    if ready and not args.rebuild_temporal:
        load_cache(paths.temporal_cache(0))
        print(f"[temporal] reusing {frame_count} temporal-v4 caches")
        return

    first = np.load(paths.base_instances(0), mmap_mode="r", allow_pickle=False)
    raw_movie_path = INV30.assemble_raw_movie(
        paths,
        frame_count,
        tuple(int(v) for v in first.shape),
        rebuild=bool(args.rebuild_temporal),
    )
    graph_path, tracked_masks_path = INV30.prepare_trackastra(
        paths,
        raw_movie_path,
        base_movie_path,
        model_name=str(args.trackastra_model),
        mode=str(args.trackastra_mode),
        device=str(args.trackastra_device),
        rebuild=bool(args.rebuild_temporal),
    )
    with graph_path.open("rb") as handle:
        track_graph = pickle.load(handle)
    tracked_movie = np.load(tracked_masks_path, mmap_mode="r")
    raw_movie = np.load(raw_movie_path, mmap_mode="r")
    INV30.build_temporal_caches(
        paths,
        track_graph=track_graph,
        tracked_movie=tracked_movie,
        raw_movie=raw_movie,
        frame_count=frame_count,
        spacing=spacing,
        dref_um=float(dref_um),
        temporal_radius=int(INV30.DEFAULT_TEMPORAL_RADIUS),
        complete_candidate_graph=bool(args.complete_candidate_graph),
        rebuild=bool(args.rebuild_temporal),
    )


def hydrate_model_config(payload: dict[str, Any]):
    cfg = INV30.ModelConfig()
    saved = payload.get("model_config")
    if not isinstance(saved, dict):
        raise KeyError("Checkpoint does not contain model_config.")
    for section_name, section_values in saved.items():
        if not hasattr(cfg, section_name) or not isinstance(section_values, dict):
            continue
        section = getattr(cfg, section_name)
        for key, value in section_values.items():
            if hasattr(section, key):
                setattr(section, key, value)
    cfg.validate()
    return cfg


def load_scoring_model(checkpoint_path: Path, device: torch.device):
    payload = torch_load(checkpoint_path)
    if "temporal_model_state_dict" not in payload:
        investigation = payload.get("investigation") or payload.get("extra", {}).get("investigation")
        raise RuntimeError(
            "Suspect-export v1 expects an Investigation-31-style causal temporal "
            "checkpoint containing temporal_model_state_dict. "
            f"Got investigation={investigation!r}. Investigation-35 prepared "
            "observer caches are intentionally not used for annotation mining."
        )
    cfg = hydrate_model_config(payload)
    temporal_model = INV30.TemporalOnlyModel(cfg)
    temporal_model.load_state_dict(payload["temporal_model_state_dict"], strict=True)
    frozen_spatial = INV30.FrozenCachedSpatialRepresentation(cfg)
    temporal_model.to(device).eval()
    frozen_spatial.to(device).eval()
    if any(True for _ in frozen_spatial.parameters()):
        raise RuntimeError("Frozen cached-spatial representation has parameters.")
    return payload, cfg, temporal_model, frozen_spatial


def load_inference_case(
    paths,
    t: int,
    spacing: tuple[float, float, float],
    spatial_prior_logit: float,
):
    """Construct an Investigation-30 SpatialFrameCase without reading manual GT."""
    supervoxels = np.asarray(
        np.load(paths.supervoxels(t), mmap_mode="r", allow_pickle=False)
    )
    base = np.asarray(
        np.load(paths.base_instances(t), mmap_mode="r", allow_pickle=False)
    )
    if supervoxels.shape != base.shape:
        raise ValueError(f"t={t}: supervoxel/base shape mismatch")

    base_by_sv = INV30.sv_label_lookup(
        supervoxels, base, name=f"t={t} current spatial"
    )
    with np.load(paths.rag_state(t), allow_pickle=False) as rag:
        required = ("node_supervoxel_id", "edge_index", "spatial_edge_probability")
        missing = [key for key in required if key not in rag]
        if missing:
            raise KeyError(f"t={t}: rag_state missing {missing}")
        node_sv_np = np.asarray(rag["node_supervoxel_id"], dtype=np.int64).reshape(-1)
        edge_index_np = np.asarray(rag["edge_index"], dtype=np.int64)
        probability_np = np.asarray(
            rag["spatial_edge_probability"], dtype=np.float32
        ).reshape(-1)
        centroid_np = (
            np.asarray(rag["node_centroid_um"], dtype=np.float32)
            if "node_centroid_um" in rag else None
        )

    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(f"t={t}: edge_index must be [2,E]")
    if edge_index_np.shape[1] != len(probability_np):
        raise ValueError(f"t={t}: edge/probability mismatch")
    if node_sv_np.size and (
        node_sv_np.min() <= 0 or node_sv_np.max() >= len(base_by_sv)
    ):
        raise RuntimeError(f"t={t}: RAG/SV ID mismatch")

    if centroid_np is None:
        centroid_np = INV30.compute_sv_centroids_um(supervoxels, node_sv_np, spacing)
    counts = INV30.sv_voxel_counts(
        supervoxels, int(supervoxels.max(initial=0))
    )
    node_volume_np = counts[node_sv_np].astype(np.float32)
    node_base_np = base_by_sv[node_sv_np]
    if np.any(node_base_np <= 0):
        bad = node_sv_np[node_base_np <= 0][:20].tolist()
        raise RuntimeError(f"t={t}: RAG nodes outside current instances: {bad}")

    base_ids = np.unique(node_base_np)
    base_ids = base_ids[base_ids > 0]
    base_to_index = {int(label): i for i, label in enumerate(base_ids.tolist())}
    node_instance_np = np.asarray(
        [base_to_index[int(label)] for label in node_base_np], dtype=np.int64
    )
    src, dst = edge_index_np
    same_base = node_base_np[src] == node_base_np[dst]
    spatial_logits = np.where(
        same_base,
        float(spatial_prior_logit),
        -float(spatial_prior_logit),
    ).astype(np.float32)
    edge_base = np.where(same_base, node_base_np[src], 0).astype(np.int64)
    false_edges = np.zeros(edge_index_np.shape[1], dtype=bool)

    return INV30.SpatialFrameCase(
        t=int(t),
        shape_zyx=tuple(int(v) for v in supervoxels.shape),
        supervoxels=supervoxels,
        base_labels=base,
        manual_labels=base,  # dummy only; scoring never uses target fields
        node_sv=torch.from_numpy(node_sv_np.copy()).long(),
        node_centroid_um=torch.from_numpy(centroid_np.copy()).float(),
        node_volume_voxels=torch.from_numpy(node_volume_np.copy()).float(),
        edge_index=torch.from_numpy(edge_index_np.copy()).long(),
        raw_spatial_probability=torch.from_numpy(probability_np.copy()).float(),
        spatial_edge_logits=torch.from_numpy(spatial_logits.copy()).float(),
        node_base_label=torch.from_numpy(node_base_np.copy()).long(),
        node_manual_label=torch.from_numpy(node_base_np.copy()).long(),
        node_instance_index=torch.from_numpy(node_instance_np.copy()).long(),
        instance_base_ids=torch.from_numpy(base_ids.copy()).long(),
        instance_split_target=torch.zeros(len(base_ids), dtype=torch.float32),
        eligible_edge_mask=torch.from_numpy(same_base.copy()).bool(),
        keep_edge_mask=torch.from_numpy(same_base.copy()).bool(),
        cut_edge_mask=torch.from_numpy(false_edges).bool(),
        edge_base_label=torch.from_numpy(edge_base.copy()).long(),
        changed_base_ids=(),
    )


@torch.inference_mode()
def score_frame(
    *,
    case,
    temporal_payload: dict[str, Any],
    temporal_model,
    frozen_spatial,
    device: torch.device,
    dref_um: float,
    top_k_edges: int,
    split_weight: float,
    graph_weight: float,
    causal_weight: float,
    support_floor: float,
) -> dict[str, np.ndarray]:
    rag, full = INV31.forward_eval_ablation(
        temporal_model, frozen_spatial, case, temporal_payload,
        device=device, dref_um=float(dref_um), ablation="full",
    )
    _, shuffled = INV31.forward_eval_ablation(
        temporal_model, frozen_spatial, case, temporal_payload,
        device=device, dref_um=float(dref_um), ablation="shuffled",
    )
    _, contentless = INV31.forward_eval_ablation(
        temporal_model, frozen_spatial, case, temporal_payload,
        device=device, dref_um=float(dref_um), ablation="contentless",
    )

    instance_ids = case.instance_base_ids.cpu().numpy().astype(np.int64, copy=False)
    split_probability = full.split_logits.float().sigmoid().cpu().numpy().astype(np.float32)
    support_t = (
        full.temporal_support[:, 0]
        if full.temporal_support.ndim == 2 else full.temporal_support
    )
    temporal_support = support_t.float().clamp(0, 1).cpu().numpy().astype(np.float32)
    entropy_t = (
        full.temporal_attention_entropy[:, 0]
        if full.temporal_attention_entropy.ndim == 2
        else full.temporal_attention_entropy
    )
    temporal_entropy = entropy_t.float().cpu().numpy().astype(np.float32)
    if len(split_probability) != len(instance_ids):
        raise RuntimeError(
            f"split-logit/instance mismatch: {len(split_probability)} vs {len(instance_ids)}"
        )

    full_keep = full.final_edge_logits.float().sigmoid().cpu().numpy()
    shuffled_keep = shuffled.final_edge_logits.float().sigmoid().cpu().numpy()
    contentless_keep = contentless.final_edge_logits.float().sigmoid().cpu().numpy()
    spatial_keep = rag.spatial_edge_logits.float().sigmoid().cpu().numpy()
    gate = full.edge_temporal_gate.float().cpu().numpy()

    full_cut = 1.0 - full_keep
    causal_gain = np.maximum(
        full_cut - np.maximum(1.0 - shuffled_keep, 1.0 - contentless_keep),
        0.0,
    )
    override = np.maximum(spatial_keep - full_keep, 0.0) * np.clip(gate, 0, 1)
    edge_base = case.edge_base_label.cpu().numpy()

    graph_evidence = np.zeros(len(instance_ids), np.float32)
    causal_evidence = np.zeros(len(instance_ids), np.float32)
    override_evidence = np.zeros(len(instance_ids), np.float32)
    strongest_cut = np.zeros(len(instance_ids), np.float32)
    min_final_keep = np.ones(len(instance_ids), np.float32)
    internal_edge_count = np.zeros(len(instance_ids), np.int32)

    for row, instance_id in enumerate(instance_ids.tolist()):
        edge_rows = np.flatnonzero(edge_base == int(instance_id))
        internal_edge_count[row] = int(edge_rows.size)
        if edge_rows.size == 0:
            continue
        cuts = full_cut[edge_rows]
        graph_evidence[row] = topk_mean(cuts, top_k_edges)
        causal_evidence[row] = topk_mean(causal_gain[edge_rows], top_k_edges)
        override_evidence[row] = topk_mean(override[edge_rows], top_k_edges)
        strongest_cut[row] = float(cuts.max())
        min_final_keep[row] = float(full_keep[edge_rows].min())

    raw_score = (
        float(split_weight) * split_probability
        + float(graph_weight) * graph_evidence
        + float(causal_weight) * causal_evidence
    )
    support_factor = (
        float(support_floor)
        + (1.0 - float(support_floor)) * temporal_support
    )
    suspect_score = np.clip(raw_score * support_factor, 0, 1).astype(np.float32)

    return {
        "instance_id": instance_ids.astype(np.int32, copy=False),
        "suspect_score": suspect_score,
        "split_probability": split_probability,
        "graph_cut_evidence": graph_evidence,
        "causal_cut_evidence": causal_evidence,
        "temporal_support": temporal_support,
        "temporal_attention_entropy": temporal_entropy,
        "temporal_override_evidence": override_evidence,
        "strongest_cut_probability": strongest_cut,
        "min_final_keep_probability": min_final_keep,
        "internal_edge_count": internal_edge_count,
    }


def main() -> None:
    args = parse_args()
    weights = (float(args.split_weight), float(args.graph_weight), float(args.causal_weight))
    if any(v < 0 for v in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("split/graph/causal weights must be nonnegative and sum to 1")
    if not 0 <= float(args.support_floor) <= 1:
        raise ValueError("support-floor must be in [0,1]")
    if int(args.top_k_edges) < 1:
        raise ValueError("top-k-edges must be >= 1")

    paths = make_inv30_paths(args)
    available = completed_spatial_frames(paths.inv25)
    if not available:
        raise FileNotFoundError(f"No completed spatial frames below:\n  {paths.inv25}")
    expected = list(range(max(available) + 1))
    if available != expected:
        raise RuntimeError(
            "Temporal suspect mining requires a contiguous movie starting at t=0; "
            f"found {available}"
        )
    frame_count = len(available)
    selected = parse_timepoints(args.timepoints, available)
    validate_spatial_artifacts(paths, frame_count)

    checkpoint_path = (
        resolve(args.checkpoint) if args.checkpoint is not None
        else DEFAULT_CHECKPOINT.resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Causal temporal checkpoint missing:\n  {checkpoint_path}\n"
            "Pass --checkpoint if it is elsewhere."
        )
    output = (
        resolve(args.output_dir) if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / args.sample_id).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)

    spacing = tuple(float(v) for v in args.spacing_zyx_um)
    device = parse_device(args.device)
    dref_um, base_movie_path = resolve_dref_and_base_movie(
        paths,
        frame_count,
        spacing,
        args.dref_um,
        rebuild_movie=bool(args.rebuild_temporal),
    )
    ensure_temporal_caches(
        paths,
        frame_count=frame_count,
        spacing=spacing,
        dref_um=dref_um,
        base_movie_path=base_movie_path,
        args=args,
    )
    checkpoint_payload, _cfg, temporal_model, frozen_spatial = load_scoring_model(
        checkpoint_path, device
    )

    checkpoint_sig = file_signature(checkpoint_path)
    score_parameters = {
        "method": SCORE_METHOD,
        "top_k_edges": int(args.top_k_edges),
        "split_weight": float(args.split_weight),
        "graph_weight": float(args.graph_weight),
        "causal_weight": float(args.causal_weight),
        "support_floor": float(args.support_floor),
        "spatial_prior_logit": float(args.spatial_prior_logit),
    }
    manifest_path = output / "manifest.json"
    old_manifest = None
    if manifest_path.is_file():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            old_manifest = None
    reusable = bool(
        not args.rebuild_scores
        and isinstance(old_manifest, dict)
        and old_manifest.get("format_version") == SCORE_FORMAT_VERSION
        and old_manifest.get("checkpoint_signature") == checkpoint_sig
        and old_manifest.get("score_parameters") == score_parameters
        and abs(float(old_manifest.get("dref_um", -1)) - dref_um) < 1e-6
    )

    print("=" * 104)
    print("BIOHUB MERGE-SUSPECT EXPORT")
    print("=" * 104)
    print(f"sample             : {args.sample_id}")
    print(f"selected frames    : {selected}")
    print(f"checkpoint         : {checkpoint_path}")
    print(f"checkpoint step    : {checkpoint_payload.get('step', '?')}")
    print(f"device             : {device}")
    print(f"dref_um            : {dref_um:.5f}")
    print(f"temporal cache     : {paths.temporal_cache_dir}")
    print(f"output             : {output}")
    print("manual annotations : NOT READ")
    print("=" * 104)

    per_frame = []
    for frame in selected:
        target = output / f"t{frame:03d}.npz"
        if reusable and target.is_file():
            with np.load(target, allow_pickle=False) as cached:
                ids = np.asarray(cached["instance_id"])
                scores = np.asarray(cached["suspect_score"])
            print(
                f"[t{frame:03d}] reuse | instances={len(ids)} "
                f"max={float(scores.max()) if len(scores) else 0.0:.4f}"
            )
            per_frame.append({
                "timepoint": int(frame),
                "instance_count": int(len(ids)),
                "max_score": float(scores.max()) if len(scores) else 0.0,
                "reused": True,
            })
            continue

        case = load_inference_case(
            paths, frame, spacing, float(args.spatial_prior_logit)
        )
        temporal_payload = load_cache(paths.temporal_cache(frame))
        arrays = score_frame(
            case=case,
            temporal_payload=temporal_payload,
            temporal_model=temporal_model,
            frozen_spatial=frozen_spatial,
            device=device,
            dref_um=dref_um,
            top_k_edges=int(args.top_k_edges),
            split_weight=float(args.split_weight),
            graph_weight=float(args.graph_weight),
            causal_weight=float(args.causal_weight),
            support_floor=float(args.support_floor),
        )
        atomic_npz(target, **arrays)
        ids = arrays["instance_id"]
        scores = arrays["suspect_score"]
        order = np.argsort(-scores)
        print(
            f"[t{frame:03d}] instances={len(ids)} "
            f"max={float(scores.max()) if len(scores) else 0.0:.4f} "
            f"mean={float(scores.mean()) if len(scores) else 0.0:.4f}"
        )
        top_n = min(max(int(args.print_top), 0), len(order))
        if top_n:
            print("           top: " + " | ".join(
                f"id={int(ids[i])}: {float(scores[i]):.3f}"
                for i in order[:top_n]
            ))
        per_frame.append({
            "timepoint": int(frame),
            "instance_count": int(len(ids)),
            "max_score": float(scores.max()) if len(scores) else 0.0,
            "mean_score": float(scores.mean()) if len(scores) else 0.0,
            "reused": False,
        })

    manifest = {
        "format_version": SCORE_FORMAT_VERSION,
        "status": "success",
        "experiment": SCRIPT_NAME,
        "score_method": SCORE_METHOD,
        "sample_id": str(args.sample_id),
        "available_frames": available,
        "exported_frames": selected,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_signature": checkpoint_sig,
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "checkpoint_investigation": checkpoint_payload.get("investigation"),
        "spacing_zyx_um": spacing,
        "dref_um": float(dref_um),
        "score_parameters": score_parameters,
        "manual_annotations_read": False,
        "temporal_source": "current Investigation-25 movie via Trackastra/temporal-v4",
        "threshold_applied_during_export": False,
        "per_frame": per_frame,
    }
    atomic_json(manifest_path, manifest)
    print("=" * 104)
    print("SUSPECT EXPORT COMPLETE")
    print(f"manifest : {manifest_path}")
    print("threshold: NOT APPLIED; 02_ applies it at display time")
    print("=" * 104)


if __name__ == "__main__":
    main()
