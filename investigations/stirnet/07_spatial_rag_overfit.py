from __future__ import annotations

"""
Stage 07: isolated overfit of STIR-Net's learned spatial RAG on candidate 2.

The experiment restores the latest successful Stage-01 joint/full checkpoint,
runs learned dense geometry once, builds the current production watershed +
face-level safety-guard proposal, caches all expensive 3-D reductions, and then
trains ONLY:

    RAGBuilder.node_projection
    SpatialRAGNetwork

The production RAGCriterion supplies valid MERGE/SEPARATE edge targets and the
production GraphPartitioner is used for the final grouping check.

Default ``--proposal-source auto`` prefers the learned proposal. If that proposal
is unsafe or lacks at least one valid MERGE and one valid SEPARATE edge, Stage 07
falls back to an oracle proposal topology generated through the current
watershed/guard. Even in that fallback, RAG node/edge evidence still comes from
the learned Stage-01 D0/dense-geometry predictions.

Local CUDA only.

Run:
    python investigations/stirnet/07_spatial_rag_overfit.py

Visualize:
    python investigations/stirnet/07_spatial_rag_overfit.py --visualize
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _import_adjacent(module_name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage01 = _import_adjacent("_stirnet_stage01_for_stage07", "01_dense_geometry_overfit.py")
stage03 = _import_adjacent("_stirnet_stage03_for_stage07", "03_watershed_supervoxel_debug.py")

from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.partition.graph_net import SpatialRAGNetwork
from learned.stirnet.model.partition.partitioner import GraphPartitioner
from learned.stirnet.model.partition.rag import RAGBuilder, RAGCriterion, RAGTargets
from learned.stirnet.model.partition.statistics import build_supervoxel_statistics
from learned.stirnet.model.partition.watershed import LearnedGeometryWatershed
from learned.stirnet.model.types import GeometryState, RAGState

try:
    from tqdm.auto import trange as tqdm_trange
except ImportError:
    tqdm_trange = None


EXPECTED_CANDIDATE_INDEX = 2
DEFAULT_SEED = 230525
DEFAULT_STEPS = 1000
DEFAULT_LR = 1e-3
DEFAULT_EVAL_EVERY = 20
GRAD_CLIP_NORM = 5.0
PASS_STREAK = 3
MIN_PROPOSAL_ATOMIC_RECOVERABILITY = 0.95

PASS = {
    "rag_bce_max": 0.05,
    "edge_accuracy_min": 1.0,
    "merge_recall_min": 1.0,
    "separate_accuracy_min": 1.0,
    "min_merge_probability_min": 0.95,
    "max_separate_probability_max": 0.05,
    "pairwise_accuracy_min": 1.0,
}

DEFAULT_SAMPLE = REPOSITORY_ROOT / "data" / "learned" / "stirnet" / "debug_crop.pt"
DEFAULT_STAGE01 = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "dense_geometry_debug"
    / "latest_success.pt"
)
DEFAULT_RESULTS = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "spatial_rag_overfit"
)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_peak_gib() -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated() / 1024**3,
        torch.cuda.max_memory_reserved() / 1024**3,
    )


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def state_dict_cpu(module: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _probability_to_logit(x: Tensor, eps: float = 1e-4) -> Tensor:
    x = x.float().clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def oracle_geometry(batch) -> GeometryState:
    return GeometryState(
        foreground_logits=_probability_to_logit(batch.targets["foreground"]),
        surface_logits=_probability_to_logit(batch.targets["surface"]),
        separator_logits=_probability_to_logit(batch.targets["separator"]),
        sdf=batch.targets["sdf"].float(),
        flow=batch.targets["flow"].float(),
        centroid_offset=batch.targets["centroid_offset"].float(),
        seed_logits=_probability_to_logit(batch.targets["seed"]),
        features=None,
        feature_spacing_um=batch.spacing_um,
    )


def validate_checkpoint(checkpoint: dict[str, Any], batch) -> str:
    stage = str(checkpoint.get("stage", ""))
    if stage not in {"joint", "full"}:
        raise RuntimeError(
            "Stage 07 needs a successful Stage-01 joint/full checkpoint; "
            f"found {stage!r}."
        )
    sample_index = batch.selection.get("candidate_index")
    if sample_index != EXPECTED_CANDIDATE_INDEX:
        raise RuntimeError(
            f"debug_crop.pt is candidate {sample_index}; expected "
            f"candidate {EXPECTED_CANDIDATE_INDEX}."
        )
    ckpt_sel = dict(checkpoint.get("sample_selection", {}))
    ckpt_index = ckpt_sel.get("candidate_index")
    if ckpt_index is not None and int(ckpt_index) != int(sample_index):
        raise RuntimeError(
            f"Stage-01 checkpoint candidate={ckpt_index}, sample={sample_index}."
        )
    ckpt_shape = checkpoint.get("sample_shape_zyx")
    shape = list(batch.spatial_inputs.shape[-3:])
    if ckpt_shape is not None and list(ckpt_shape) != shape:
        raise RuntimeError(
            f"Stage-01 checkpoint shape={ckpt_shape}, sample shape={shape}."
        )
    return stage


@torch.no_grad()
def run_frozen_stage01(batch, checkpoint_path: Path, cfg, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-01 checkpoint: {checkpoint_path}\n"
            "Run 01_dense_geometry_overfit.py --stage joint first."
        )
    checkpoint = torch_load(checkpoint_path)
    checkpoint_stage = validate_checkpoint(checkpoint, batch)

    model = stage01.DenseGeometryOnlyModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    acquisition = model.acquisition(batch.spacing_um, batch.dref_um)
    stem = model.evidence_stem(batch.spatial_inputs, acquisition)
    _, decoded = model.spatial_backbone(
        stem, batch.spacing_um, acquisition, padding_mask=None
    )
    geometry = model.geometry_decoder(decoded.d0, acquisition)
    del model, acquisition, stem
    return geometry, decoded, checkpoint_stage


@dataclass
class Bundle:
    source: str
    labels: Tensor
    summary: dict[str, Any]
    statistics: list[Any]
    template: RAGState
    targets: RAGTargets
    positives: int
    negatives: int
    atomic_min: float


def complete_gt_ids(batch) -> list[int]:
    ids = [int(v) for v in batch.selection.get("complete_gt_ids", [])]
    if ids:
        return ids
    return [int(v) for v in torch.unique(batch.gt_labels).tolist() if int(v) > 0]


@torch.no_grad()
def proposal_labels(source: str, predicted_geometry, predicted_cache, batch, cfg) -> Tensor:
    ws = LearnedGeometryWatershed(cfg.partition, cfg.geometry).to(predicted_geometry.sdf.device)
    if source == "predicted":
        return ws(
            predicted_geometry,
            batch.spacing_um,
            batch.dref_um,
            derived_cache=predicted_cache,
        )[0]

    oracle = oracle_geometry(batch)
    cache = build_geometry_derived_cache(oracle, cfg.partition, padding_mask=None)
    labels = ws(
        oracle,
        batch.spacing_um,
        batch.dref_um,
        derived_cache=cache,
    )[0]
    del oracle, cache
    return labels


@torch.no_grad()
def make_bundle(
    source: str,
    labels: Tensor,
    predicted_geometry,
    predicted_cache,
    decoded,
    batch,
    cfg,
    builder: RAGBuilder,
    criterion: RAGCriterion,
) -> Bundle:
    summary, _ = stage03.supervoxel_diagnostics(
        labels.detach().cpu().numpy().astype(np.int32, copy=False),
        batch.gt_labels.detach().cpu().numpy().astype(np.int32, copy=False),
        complete_gt_ids(batch),
        cfg,
    )
    statistics = build_supervoxel_statistics(
        [labels],
        batch.spatial_inputs,
        predicted_geometry,
        batch.spacing_um,
        (decoded.d0, decoded.d1, decoded.d2),
        derived=predicted_cache,
    )
    template = builder(
        [labels],
        decoded.d0,
        batch.spatial_inputs,
        predicted_geometry,
        batch.spacing_um,
        batch.dref_um,
        statistics_by_batch=statistics,
        derived_cache=predicted_cache,
    )
    targets = criterion.build_targets(
        template, batch.gt_labels.unsqueeze(0).to(labels.device)
    )
    valid = targets.valid
    pos = valid & (targets.target > 0.5)
    neg = valid & (targets.target <= 0.5)
    return Bundle(
        source=source,
        labels=labels,
        summary=summary,
        statistics=statistics,
        template=template,
        targets=targets,
        positives=int(pos.sum().item()),
        negatives=int(neg.sum().item()),
        atomic_min=float(summary["atomic_min_complete_recoverable_fraction"]),
    )


def bundle_usable(bundle: Bundle) -> tuple[bool, list[str]]:
    failures = []
    if bundle.atomic_min < MIN_PROPOSAL_ATOMIC_RECOVERABILITY:
        failures.append(
            f"atomic ceiling {bundle.atomic_min:.3f} < "
            f"{MIN_PROPOSAL_ATOMIC_RECOVERABILITY:.3f}"
        )
    if bundle.positives < 1:
        failures.append("no valid MERGE edge")
    if bundle.negatives < 1:
        failures.append("no valid SEPARATE edge")
    return not failures, failures


def print_bundle(bundle: Bundle) -> None:
    print("\n" + "=" * 102)
    print(f"RAG pre-flight — {bundle.source.upper()} proposal")
    print("=" * 102)
    print(f"Supervoxels                   : {int(bundle.labels.max().item())}")
    print(f"RAG nodes                     : {bundle.template.node_features.shape[0]}")
    print(f"Adjacency edges               : {bundle.template.edge_index.shape[1]}")
    print(f"Valid MERGE edges             : {bundle.positives}")
    print(f"Valid SEPARATE edges          : {bundle.negatives}")
    print(
        "Legacy cross-GT SVs           : "
        f"{bundle.summary['cross_gt_unsafe_supervoxel_count']}"
    )
    print(f"Atomic min complete recoverable: {bundle.atomic_min:.4f}")
    print(
        "RAG-valid SV fraction          : "
        f"{bundle.summary['rag_valid_supervoxel_fraction']:.4f}"
    )
    print("=" * 102)


def edge_rows(rag: RAGState, targets: RAGTargets, probs: Tensor | None = None):
    rows = []
    for e in range(rag.edge_index.shape[1]):
        a = int(rag.edge_index[0, e].item())
        b = int(rag.edge_index[1, e].item())
        valid = bool(targets.valid[e].item())
        target_value = float(targets.target[e].item())
        row = {
            "edge_row": e,
            "supervoxel_a": int(rag.node_supervoxel_id[a].item()),
            "supervoxel_b": int(rag.node_supervoxel_id[b].item()),
            "dominant_gt_a": int(targets.dominant_gt[a].item()),
            "dominant_gt_b": int(targets.dominant_gt[b].item()),
            "valid": valid,
            "target": (
                "merge" if valid and target_value > 0.5
                else "separate" if valid
                else "invalid"
            ),
        }
        if probs is not None:
            row["probability"] = float(probs[e].detach().float().cpu().item())
        rows.append(row)
    return rows


def print_edges(rows, final: bool = False) -> None:
    print("\nRAG edges")
    header = (
        f"{'edge':>5s} {'SV':>9s} {'GT':>9s} {'valid':>6s} "
        f"{'target':>9s}" + (f" {'prob':>8s}" if final else "")
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['edge_row']:5d} "
            f"{r['supervoxel_a']}-{r['supervoxel_b']:>5d} "
            f"{r['dominant_gt_a']}-{r['dominant_gt_b']:>5d} "
            f"{str(r['valid']):>6s} {r['target']:>9s}"
        )
        if final:
            line += f" {r.get('probability', float('nan')):8.4f}"
        print(line)


def grouping_metrics(partition, targets: RAGTargets, cfg) -> dict[str, float | int]:
    valid_nodes = (
        (targets.dominant_gt > 0)
        & (targets.node_purity >= cfg.partition.rag_min_node_purity)
        & (targets.node_gt_support >= cfg.partition.rag_min_node_gt_support)
    )
    rows = torch.nonzero(valid_nodes, as_tuple=False).flatten()
    if rows.numel() < 2:
        return {
            "valid_node_count": int(rows.numel()),
            "pair_count": 0,
            "pairwise_accuracy": 1.0,
            "false_merge_pairs": 0,
            "false_split_pairs": 0,
        }
    i, j = torch.triu_indices(
        rows.numel(), rows.numel(), offset=1, device=rows.device
    )
    a, b = rows[i], rows[j]
    expected = targets.dominant_gt[a] == targets.dominant_gt[b]
    predicted = partition.node_component[a] == partition.node_component[b]
    return {
        "valid_node_count": int(rows.numel()),
        "pair_count": int(expected.numel()),
        "pairwise_accuracy": float((expected == predicted).float().mean().item()),
        "false_merge_pairs": int((predicted & ~expected).sum().item()),
        "false_split_pairs": int((~predicted & expected).sum().item()),
    }


@torch.no_grad()
def oracle_edge_ceiling(bundle: Bundle, partitioner, cfg):
    logits = bundle.template.edge_features.new_full(
        (bundle.template.edge_index.shape[1],), -10.0
    )
    logits[bundle.targets.valid & (bundle.targets.target > 0.5)] = 10.0
    partition = partitioner(
        bundle.template, logits, cfg.partition.spatial_merge_threshold
    )
    return grouping_metrics(partition, bundle.targets, cfg), partition


class RAGOverfitModel(nn.Module):
    """Production trainable RAG with cached full-volume reductions."""

    def __init__(self, builder, network, bundle: Bundle, dref_um: Tensor):
        super().__init__()
        self.builder = builder
        self.network = network
        self.bundle = bundle
        self.register_buffer("_dref_um", dref_um.detach().clone(), persistent=False)

    def current_rag(self) -> RAGState:
        template = self.bundle.template
        node_chunks, centroid_chunks, count_chunks = [], [], []
        for b, stats in enumerate(self.bundle.statistics):
            start = int(template.node_offsets[b].item())
            stop = int(template.node_offsets[b + 1].item())
            reference = template.node_features[start:stop]
            nodes, centroids, counts = self.builder._node_rows_from_statistics(
                stats, reference, self._dref_um[b]
            )
            node_chunks.append(nodes)
            centroid_chunks.append(centroids)
            count_chunks.append(counts)
        node_features = torch.cat(node_chunks, 0)
        return replace(
            template,
            node_features=node_features,
            node_centroid_um=torch.cat(centroid_chunks, 0),
            node_volume_voxels=torch.cat(count_chunks, 0),
            node_embeddings=node_features.new_zeros(
                (node_features.shape[0], self.network.cfg.rag_hidden_dim)
            ),
            edge_embeddings=template.edge_features.new_zeros(
                (template.edge_features.shape[0], self.network.cfg.rag_hidden_dim)
            ),
            spatial_edge_logits=template.edge_features.new_zeros(
                (template.edge_features.shape[0],)
            ),
        )

    def forward(self) -> RAGState:
        return self.network(self.current_rag())


@torch.no_grad()
def evaluate(model, criterion, partitioner, targets, gt_batched, cfg):
    was_training = model.training
    model.eval()
    rag = model()
    loss_metrics = criterion(rag, gt_batched, targets=targets)
    probs = rag.spatial_edge_logits.sigmoid()
    valid = targets.valid
    pos = valid & (targets.target > 0.5)
    neg = valid & (targets.target <= 0.5)
    pred = probs >= 0.5
    truth = targets.target > 0.5

    partition = partitioner(rag, rag.spatial_edge_logits, cfg.partition.spatial_merge_threshold)
    group = grouping_metrics(partition, targets, cfg)
    metrics = {
        "rag_bce": float(loss_metrics["rag_bce"].item()),
        "edge_accuracy": float((pred[valid] == truth[valid]).float().mean().item()),
        "merge_recall": float(pred[pos].float().mean().item()),
        "separate_accuracy": float((~pred[neg]).float().mean().item()),
        "min_merge_probability": float(probs[pos].min().item()),
        "max_separate_probability": float(probs[neg].max().item()),
        "valid_edge_count": int(valid.sum().item()),
        "merge_edge_count": int(pos.sum().item()),
        "separate_edge_count": int(neg.sum().item()),
        "valid_edge_fraction": float(loss_metrics["rag_valid_edge_fraction"].item()),
        "mean_node_purity": float(loss_metrics["rag_mean_node_purity"].item()),
        **{f"partition_{k}": v for k, v in group.items()},
    }
    if was_training:
        model.train()
    return metrics, rag, partition


def accepted(m: dict[str, float | int]) -> tuple[bool, list[str]]:
    failures = []
    checks = [
        (float(m["rag_bce"]) <= PASS["rag_bce_max"], f"rag_bce={m['rag_bce']:.5f}"),
        (float(m["edge_accuracy"]) >= 1.0, f"edge_accuracy={m['edge_accuracy']:.4f}"),
        (float(m["merge_recall"]) >= 1.0, f"merge_recall={m['merge_recall']:.4f}"),
        (
            float(m["separate_accuracy"]) >= 1.0,
            f"separate_accuracy={m['separate_accuracy']:.4f}",
        ),
        (
            float(m["min_merge_probability"]) >= PASS["min_merge_probability_min"],
            f"min_merge_probability={m['min_merge_probability']:.4f}",
        ),
        (
            float(m["max_separate_probability"]) <= PASS["max_separate_probability_max"],
            f"max_separate_probability={m['max_separate_probability']:.4f}",
        ),
        (
            float(m["partition_pairwise_accuracy"]) >= 1.0,
            f"partition_pairwise_accuracy={m['partition_pairwise_accuracy']:.4f}",
        ),
        (
            int(m["partition_false_merge_pairs"]) == 0,
            f"partition_false_merge_pairs={m['partition_false_merge_pairs']}",
        ),
        (
            int(m["partition_false_split_pairs"]) == 0,
            f"partition_false_split_pairs={m['partition_false_split_pairs']}",
        ),
    ]
    for ok, text in checks:
        if not ok:
            failures.append(text)
    return not failures, failures


def print_metrics(step: int, m: dict[str, float | int]) -> None:
    print(
        f"step={step:04d} "
        f"bce={m['rag_bce']:.5f} "
        f"edgeAcc={m['edge_accuracy']:.3f} "
        f"mergeRec={m['merge_recall']:.3f} "
        f"sepAcc={m['separate_accuracy']:.3f} "
        f"pMergeMin={m['min_merge_probability']:.3f} "
        f"pSepMax={m['max_separate_probability']:.3f} "
        f"groupAcc={m['partition_pairwise_accuracy']:.3f} "
        f"FM={m['partition_false_merge_pairs']} "
        f"FS={m['partition_false_split_pairs']}"
    )


@dataclass
class Options:
    steps: int
    lr: float
    eval_every: int
    seed: int
    proposal_source: str
    sample: Path
    stage01_checkpoint: Path
    results: Path


def train(opts: Options) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 07 requires the local CUDA GPU.")

    device = torch.device("cuda")
    set_seed(opts.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = stage01.build_debug_config()
    batch = stage01.load_debug_batch(opts.sample, device)
    if batch.selection.get("candidate_index") != EXPECTED_CANDIDATE_INDEX:
        raise RuntimeError(
            f"Expected candidate 2, found {batch.selection.get('candidate_index')}."
        )

    print("\n" + "=" * 108)
    print("STIR-Net Stage 07 — isolated Spatial RAG overfit")
    print("=" * 108)
    print(f"GPU                     : {torch.cuda.get_device_name(device)}")
    print(f"Candidate               : {EXPECTED_CANDIDATE_INDEX}")
    print(f"Crop                    : {tuple(batch.gt_labels.shape)}")
    print(
        "Merge case              : "
        f"source {batch.selection.get('merge_source_id')} -> "
        f"GT {batch.selection.get('merge_gt_ids')}"
    )
    print(f"Free GT                 : {batch.selection.get('free_gt_id')}")
    print(f"Proposal mode           : {opts.proposal_source}")
    print("Trainable               : RAGBuilder.node_projection + SpatialRAGNetwork")
    print("Frozen                  : dense geometry + watershed + safety guard")
    print(f"Learning rate           : {opts.lr:.3g}")
    print(f"Max steps               : {opts.steps}")
    print(f"Eval every              : {opts.eval_every}")
    print("=" * 108)

    setup_start = time.perf_counter()
    print("\n[setup 1/5] Restoring/running Stage-01 learned geometry once ...", flush=True)
    predicted_geometry, decoded, checkpoint_stage = run_frozen_stage01(
        batch, opts.stage01_checkpoint, cfg, device
    )
    predicted_cache = build_geometry_derived_cache(
        predicted_geometry, cfg.partition, padding_mask=None
    )
    print(f"[setup] Stage-01 checkpoint stage: {checkpoint_stage}")

    builder = RAGBuilder(cfg.partition, cfg.spatial).to(device)
    network = SpatialRAGNetwork(
        cfg.partition, builder.node_feature_dim, builder.edge_feature_dim
    ).to(device)
    criterion = RAGCriterion(cfg.partition).to(device)
    partitioner = GraphPartitioner().to(device)

    sources = (
        [opts.proposal_source]
        if opts.proposal_source != "auto"
        else ["predicted", "oracle"]
    )
    attempts = []
    bundle = None
    for source in sources:
        print(f"\n[setup 2/5] Building {source} proposal ...", flush=True)
        labels = proposal_labels(
            source, predicted_geometry, predicted_cache, batch, cfg
        )
        candidate = make_bundle(
            source,
            labels,
            predicted_geometry,
            predicted_cache,
            decoded,
            batch,
            cfg,
            builder,
            criterion,
        )
        print_bundle(candidate)
        useful, reasons = bundle_usable(candidate)
        attempts.append(
            {
                "source": source,
                "supervoxel_count": int(labels.max().item()),
                "node_count": int(candidate.template.node_features.shape[0]),
                "edge_count": int(candidate.template.edge_index.shape[1]),
                "valid_merge_edges": candidate.positives,
                "valid_separate_edges": candidate.negatives,
                "atomic_min_complete": candidate.atomic_min,
                "usable": useful,
                "failures": reasons,
            }
        )
        if useful:
            bundle = candidate
            break
        print("[pre-flight] unsuitable:")
        for reason in reasons:
            print(f"  - {reason}")
        if opts.proposal_source != "auto":
            raise RuntimeError(f"Forced {source} proposal is unsuitable: {reasons}")
        print("[pre-flight] AUTO falling back to oracle proposal topology.")

    if bundle is None:
        raise RuntimeError("No safe two-class RAG proposal is available.")

    print(f"\n[setup] Selected proposal: {bundle.source.upper()}")
    print_edges(edge_rows(bundle.template, bundle.targets))

    print("\n[setup 3/5] Checking perfect-edge GraphPartitioner ceiling ...", flush=True)
    oracle_group, oracle_partition = oracle_edge_ceiling(bundle, partitioner, cfg)
    print(
        f"Oracle-edge groupAcc={oracle_group['pairwise_accuracy']:.4f} "
        f"falseMerge={oracle_group['false_merge_pairs']} "
        f"falseSplit={oracle_group['false_split_pairs']}"
    )
    if (
        oracle_group["pairwise_accuracy"] < 1.0
        or oracle_group["false_merge_pairs"]
        or oracle_group["false_split_pairs"]
    ):
        raise RuntimeError(
            "Perfect edge decisions cannot recover the GT-consistent grouping; "
            "do not train the RAG on this graph."
        )

    print("\n[setup 4/5] Caching graph reductions and releasing 3-D tensors ...", flush=True)
    raw_cpu = batch.spatial_inputs[0, 0].detach().float().cpu()
    gt_cpu = batch.gt_labels.detach().cpu().long()
    sv_cpu = bundle.labels.detach().cpu().long()
    oracle_partition_cpu = oracle_partition.labels[0].detach().cpu().long()
    gt_batched = batch.gt_labels.unsqueeze(0).to(device)
    dref = batch.dref_um.detach().clone()

    del predicted_geometry, predicted_cache, decoded
    torch.cuda.empty_cache()

    model = RAGOverfitModel(builder, network, bundle, dref).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=opts.lr, weight_decay=0.0)
    print(f"[setup] Trainable RAG parameters: {trainable_count:,}")

    print("\n[setup 5/5] Evaluating fresh random RAG ...", flush=True)
    initial_metrics, _, _ = evaluate(
        model, criterion, partitioner, bundle.targets, gt_batched, cfg
    )
    print_metrics(0, initial_metrics)
    setup_seconds = time.perf_counter() - setup_start
    print(f"[setup] complete in {duration(setup_seconds)}")

    history = [{"step": 0, "metrics": initial_metrics}]
    pass_streak = 0
    stopped_early = False
    train_start = time.perf_counter()

    iterator = (
        tqdm_trange(1, opts.steps + 1, desc="stirnet:rag", unit="step", dynamic_ncols=True)
        if tqdm_trange is not None
        else range(1, opts.steps + 1)
    )
    has_tqdm = tqdm_trange is not None

    for step in iterator:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rag = model()
        loss = criterion(rag, gt_batched, targets=bundle.targets)["rag_bce"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite RAG loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM)
        optimizer.step()

        if has_tqdm:
            allocated, _ = gpu_peak_gib()
            iterator.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                mem=f"{allocated:.2f}G",
                refresh=(step % 5 == 0),
            )

        if step == 1 or step % opts.eval_every == 0 or step == opts.steps:
            metrics, _, _ = evaluate(
                model, criterion, partitioner, bundle.targets, gt_batched, cfg
            )
            passed_now, failures = accepted(metrics)
            pass_streak = pass_streak + 1 if passed_now else 0
            history.append(
                {
                    "step": step,
                    "grad_norm": float(
                        torch.as_tensor(grad_norm).detach().float().cpu().item()
                    ),
                    "passed": passed_now,
                    "pass_streak": pass_streak,
                    "failures": failures,
                    "metrics": metrics,
                    "elapsed_seconds": time.perf_counter() - train_start,
                }
            )
            text = (
                f"[eval] step={step:04d} bce={metrics['rag_bce']:.5f} "
                f"edgeAcc={metrics['edge_accuracy']:.3f} "
                f"pMergeMin={metrics['min_merge_probability']:.3f} "
                f"pSepMax={metrics['max_separate_probability']:.3f} "
                f"groupAcc={metrics['partition_pairwise_accuracy']:.3f} "
                f"pass={passed_now} streak={pass_streak}/{PASS_STREAK}"
            )
            if has_tqdm:
                iterator.write(text)
            else:
                print(text)

            if passed_now and pass_streak >= PASS_STREAK:
                stopped_early = True
                if has_tqdm:
                    iterator.write("[EARLY STOP] strict RAG memorization is stable.")
                break

    final_metrics, final_rag, final_partition = evaluate(
        model, criterion, partitioner, bundle.targets, gt_batched, cfg
    )
    final_pass, final_failures = accepted(final_metrics)
    probs = final_rag.spatial_edge_logits.sigmoid()
    final_edges = edge_rows(final_rag, bundle.targets, probs)
    steps_completed = int(history[-1]["step"])
    train_seconds = time.perf_counter() - train_start
    allocated, reserved = gpu_peak_gib()

    print("\n" + "=" * 108)
    print("Stage 07 final evaluation")
    print("=" * 108)
    print_metrics(steps_completed, final_metrics)
    print_edges(final_edges, final=True)
    print(f"\nAcceptance               : {'PASS' if final_pass else 'NOT YET PASSING'}")
    for failure in final_failures:
        print(f"  - {failure}")
    print(f"Proposal source          : {bundle.source}")
    print(f"Proposal atomic ceiling  : {bundle.atomic_min:.4f}")
    print(f"Completed steps          : {steps_completed}")
    print(f"Stopped early            : {stopped_early}")
    print(f"Setup time               : {duration(setup_seconds)}")
    print(f"Training time            : {duration(train_seconds)}")
    print(
        f"Peak CUDA                : {allocated:.3f} GiB allocated / "
        f"{reserved:.3f} GiB reserved"
    )
    print("=" * 108)

    opts.results.mkdir(parents=True, exist_ok=True)
    attempt = {
        "format_version": 1,
        "stage": "07_spatial_rag_overfit",
        "passed": final_pass,
        "failures": final_failures,
        "candidate_index": EXPECTED_CANDIDATE_INDEX,
        "proposal_source_requested": opts.proposal_source,
        "proposal_source_used": bundle.source,
        "proposal_attempts": attempts,
        "stage01_checkpoint": str(opts.stage01_checkpoint),
        "stage01_checkpoint_stage": checkpoint_stage,
        "steps_completed": steps_completed,
        "max_steps": opts.steps,
        "learning_rate": opts.lr,
        "eval_every": opts.eval_every,
        "seed": opts.seed,
        "trainable_parameters": trainable_count,
        "stopped_early": stopped_early,
        "setup_seconds": setup_seconds,
        "training_seconds": train_seconds,
        "peak_cuda_allocated_gib": allocated,
        "peak_cuda_reserved_gib": reserved,
        "proposal_atomic_min_complete_recoverable_fraction": bundle.atomic_min,
        "oracle_edge_grouping": oracle_group,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "pass_thresholds": PASS,
        "history": history,
        "edge_rows": final_edges,
        "timestamp_unix": time.time(),
    }
    atomic_json(opts.results / "last_attempt.json", attempt)

    artifact = {
        "format_version": 1,
        "kind": "stirnet_spatial_rag_overfit_attempt",
        "attempt": attempt,
        "raw": raw_cpu.half(),
        "gt_labels": gt_cpu,
        "safe_supervoxels": sv_cpu,
        "oracle_edge_partition": oracle_partition_cpu,
        "rag_partition": final_partition.labels[0].detach().cpu().long(),
        "edge_probabilities": probs.detach().cpu().float(),
        "edge_targets": bundle.targets.target.detach().cpu().float(),
        "edge_valid": bundle.targets.valid.detach().cpu().bool(),
        "edge_index": final_rag.edge_index.detach().cpu().long(),
        "node_supervoxel_id": final_rag.node_supervoxel_id.detach().cpu().long(),
        "dominant_gt": bundle.targets.dominant_gt.detach().cpu().long(),
        "node_purity": bundle.targets.node_purity.detach().cpu().float(),
        "node_gt_support": bundle.targets.node_gt_support.detach().cpu().float(),
        "spacing_um": batch.spacing_um[0].detach().cpu().float(),
    }
    atomic_torch_save(opts.results / "last_attempt.pt", artifact)

    if final_pass:
        success = {
            **artifact,
            "kind": "stirnet_spatial_rag_overfit_success",
            "rag_builder_state": state_dict_cpu(model.builder),
            "rag_network_state": state_dict_cpu(model.network),
        }
        atomic_torch_save(opts.results / "latest_success.pt", success)
        atomic_json(opts.results / "latest_success.json", attempt)
        print("\nSaved latest successful Stage-07 checkpoint.")
    else:
        print("\nNo success checkpoint written; any previous success is preserved.")

    print(f"Attempt summary : {opts.results / 'last_attempt.json'}")
    print(f"Attempt artifact: {opts.results / 'last_attempt.pt'}")
    return attempt


def visualize(results: Path) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for --visualize.") from exc

    success = results / "latest_success.pt"
    attempt = results / "last_attempt.pt"
    path = success if success.exists() else attempt
    if not path.exists():
        raise FileNotFoundError(f"No Stage-07 result found in {results}.")

    data = torch_load(path)
    info = data["attempt"]
    scale = tuple(float(v) for v in data["spacing_um"].tolist())

    print(f"\nStage-07 visualization: {path.name}")
    print(f"Passed          : {info['passed']}")
    print(f"Proposal source : {info['proposal_source_used']}")
    print_edges(info["edge_rows"], final=True)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        data["raw"].float().numpy(),
        name="Raw",
        scale=scale,
        colormap="gray",
        visible=True,
    )
    viewer.add_labels(
        data["gt_labels"].numpy(),
        name="GT labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        data["safe_supervoxels"].numpy(),
        name="Safe supervoxels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        data["rag_partition"].numpy(),
        name="RAG predicted partition",
        scale=scale,
        visible=True,
    )
    print("Napari loads exactly four layers.")
    napari.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--proposal-source",
        choices=("auto", "predicted", "oracle"),
        default="auto",
    )
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--stage01-checkpoint", type=Path, default=DEFAULT_STAGE01
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.sample_path = args.sample_path.resolve()
    args.stage01_checkpoint = args.stage01_checkpoint.resolve()
    args.result_dir = args.result_dir.resolve()

    if args.visualize:
        visualize(args.result_dir)
        return
    if args.steps < 1 or args.eval_every < 1 or args.learning_rate <= 0:
        raise ValueError("steps/eval-every must be positive and learning-rate > 0")

    train(
        Options(
            steps=args.steps,
            lr=args.learning_rate,
            eval_every=args.eval_every,
            seed=args.seed,
            proposal_source=args.proposal_source,
            sample=args.sample_path,
            stage01_checkpoint=args.stage01_checkpoint,
            results=args.result_dir,
        )
    )


if __name__ == "__main__":
    main()
