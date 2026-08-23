from __future__ import annotations

"""
STIR-Net Investigation 05 — evaluate the spatial checkpoint after the first
300 geometry + 300 spatial/RAG steps.

Defaults:
  primary  = checkpoint_step_000600.pt
  baseline = checkpoint_step_000300.pt
  sample   = Zebrafish_2
  crops    = 6 deterministic production 32x192x192 crops

The script reuses Investigation 04 for the exact NIS3D/source/crop/geometry
path, then adds:
  * step-300 -> step-600 geometry regression
  * watershed/supervoxel safety and recoverability
  * learned RAG merge/cut metrics
  * learned spatial-instance metrics
  * oracle RAG ceilings
  * spatial merge-threshold sweep
  * NPZ + edge CSV + optional Napari visualization

Run from repository root:
  python investigations/stirnet/data/05_nis3d_spatial_checkpoint_eval.py

Optional:
  python investigations/stirnet/data/05_nis3d_spatial_checkpoint_eval.py --napari
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_NAME = "05_nis3d_spatial_checkpoint_eval"
DEFAULT_PRIMARY = Path(
    "runs/stirnet/training/01_nis3d_spatial_training/recovery/"
    "nis3d_zebrafish_spatial_v1/checkpoint_step_000600.pt"
)
DEFAULT_BASELINE = Path(
    "runs/stirnet/training/01_nis3d_spatial_training/recovery/"
    "nis3d_zebrafish_spatial_v1/checkpoint_step_000300.pt"
)
EDGE_FEATURE_NAMES = (
    "separator_mean", "separator_max",
    "surface_mean", "surface_max",
    "foreground_mean", "abs_sdf_mean",
    "flow_disagreement", "centroid_offset_disagreement",
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "learned").is_dir() and (p / "src").is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    raise RuntimeError("Could not resolve cell-tracking repository root.")


ROOT = repo_root()
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None


def load_eval04():
    path = ROOT / "investigations/stirnet/data/04_nis3d_geometry_checkpoint_eval.py"
    if not path.exists():
        raise FileNotFoundError(f"Required Investigation 04 is missing: {path}")
    name = "_stirnet_eval04"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E04 = load_eval04()


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else str(x)
    if isinstance(v, np.integer):
        return int(v)
    if torch.is_tensor(v):
        return jsonable(v.item()) if v.numel() == 1 else v.detach().cpu().tolist()
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return str(v)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = ["checkpoint_role", "checkpoint_step", "sample", "crop_index",
                 "candidate_type", "threshold"]
    keys = [k for k in preferred if any(k in r for r in rows)]
    seen = set(keys)
    for k in sorted({k for r in rows for k in r}):
        if k not in seen:
            keys.append(k); seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            cooked = {}
            for k in keys:
                v = row.get(k)
                cooked[k] = json.dumps(jsonable(v)) if isinstance(v, (list, tuple, dict)) else v
            w.writerow(cooked)


def finite(v: Any) -> bool:
    return isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and math.isfinite(float(v))


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def prefixed(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def aggregate(rows: list[dict[str, Any]], exclude=("checkpoint_step", "crop_index")) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    if not rows:
        return result
    excluded = set(exclude)
    keys = sorted({k for r in rows for k, v in r.items() if k not in excluded and finite(v)})
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and finite(r[k])]
        if vals:
            result[f"mean_{k}"] = float(np.mean(vals))
            result[f"min_{k}"] = float(np.min(vals))
            result[f"max_{k}"] = float(np.max(vals))
    return result


# --------------------------------------------------------------------------------------
# Label/partition diagnostics
# --------------------------------------------------------------------------------------

def foreground_dice(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
    p = torch.as_tensor(pred).bool()[valid]
    g = torch.as_tensor(gt).bool()[valid]
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    den = 2 * tp + fp + fn
    return 1.0 if den == 0 else 2.0 * tp / den


def partition_quality(pred, gt, valid) -> dict[str, Any]:
    from learned.stirnet.model.utils.contingency import label_contingency
    pred = torch.as_tensor(pred).detach().cpu().long()
    gt = torch.as_tensor(gt).detach().cpu().long()
    valid = torch.as_tensor(valid).detach().cpu().bool()
    t = label_contingency(pred, gt, valid_mask=valid)
    nr, nc = int(t.row_ids.numel()), int(t.column_ids.numel())
    out: dict[str, Any] = {
        "pred_instance_count": nr,
        "gt_instance_count": nc,
        "foreground_dice": foreground_dice(pred > 0, gt > 0, valid),
    }
    if nr == 0 or nc == 0:
        out.update(
            mean_best_gt_iou=0.0 if nc else float("nan"),
            mean_best_gt_coverage=0.0 if nc else float("nan"),
            gt_recall_iou50=0.0 if nc else float("nan"),
            gt_recall_iou75=0.0 if nc else float("nan"),
            gt_missing_fraction=1.0 if nc else float("nan"),
            mean_best_pred_iou=0.0 if nr else float("nan"),
            pred_precision_iou50=0.0 if nr else float("nan"),
            merged_pred_count=0,
            merged_pred_fraction=0.0 if nr else float("nan"),
            split_gt_count=0,
            split_gt_fraction=0.0 if nc else float("nan"),
        )
        return out

    inter = t.intersections.float()
    rc = t.row_counts.float()[:, None]
    cc = t.column_counts.float()[None, :]
    iou = inter / (rc + cc - inter).clamp_min(1)
    cov = inter / cc.clamp_min(1)
    pf = inter / rc.clamp_min(1)
    best_gt_iou = iou.max(0).values
    best_pred_iou = iou.max(1).values
    best_gt_cov = cov.max(0).values

    # Deliberately simple, symmetric "meaningful overlap" diagnostic.
    meaningful = (inter > 0) & (pf >= 0.10) & (cov >= 0.10)
    merged = meaningful.sum(1) >= 2
    split = meaningful.sum(0) >= 2

    out.update(
        mean_best_gt_iou=float(best_gt_iou.mean()),
        mean_best_gt_coverage=float(best_gt_cov.mean()),
        gt_recall_iou50=float((best_gt_iou >= 0.50).float().mean()),
        gt_recall_iou75=float((best_gt_iou >= 0.75).float().mean()),
        gt_missing_fraction=float((best_gt_cov < 0.25).float().mean()),
        mean_best_pred_iou=float(best_pred_iou.mean()),
        pred_precision_iou50=float((best_pred_iou >= 0.50).float().mean()),
        merged_pred_count=int(merged.sum()),
        merged_pred_fraction=float(merged.float().mean()),
        split_gt_count=int(split.sum()),
        split_gt_fraction=float(split.float().mean()),
    )
    return out


def supervoxel_audit(sv, gt, valid, *, min_purity: float, min_support: float) -> dict[str, Any]:
    from learned.stirnet.model.utils.contingency import label_contingency
    sv = torch.as_tensor(sv).detach().cpu().long()
    gt = torch.as_tensor(gt).detach().cpu().long()
    valid = torch.as_tensor(valid).detach().cpu().bool()
    t = label_contingency(sv, gt, valid_mask=valid)
    n = int(t.row_ids.numel())
    if not n:
        return dict(
            supervoxel_count=0,
            mean_node_purity=float("nan"),
            mean_node_gt_support=float("nan"),
            criterion_unsafe_node_count=0,
            criterion_unsafe_node_fraction=float("nan"),
            strict_cross_gt_node_count=0,
            strict_cross_gt_node_fraction=float("nan"),
            atomic_non_dominant_overlap_fraction=float("nan"),
        )

    inter = t.intersections.float()
    fg = inter.sum(1)
    support = fg / t.row_counts.float().clamp_min(1)
    dominant = inter.max(1).values if inter.shape[1] else torch.zeros_like(fg)
    purity = dominant / fg.clamp_min(1)
    unsafe = (support >= min_support) & (purity < min_purity)

    strict = torch.zeros(n, dtype=torch.bool)
    if inter.shape[1] >= 2:
        top2, cols = torch.topk(inter, 2, dim=1)
        second = top2[:, 1]
        second_col = cols[:, 1]
        strict = (
            (support >= min_support)
            & (second / fg.clamp_min(1) >= 0.05)
            & (second / t.column_counts[second_col].float().clamp_min(1) >= 0.05)
        )

    return dict(
        supervoxel_count=n,
        mean_node_purity=float(purity.mean()),
        mean_node_gt_support=float(support.mean()),
        criterion_unsafe_node_count=int(unsafe.sum()),
        criterion_unsafe_node_fraction=float(unsafe.float().mean()),
        strict_cross_gt_node_count=int(strict.sum()),
        strict_cross_gt_node_fraction=float(strict.float().mean()),
        atomic_non_dominant_overlap_fraction=safe_div(
            float((fg - dominant).clamp_min(0).sum()), float(fg.sum())
        ),
    )


def roc_auc(prob: torch.Tensor, target: torch.Tensor) -> float:
    p = prob.detach().cpu().double().numpy()
    y = target.detach().cpu().bool().numpy()
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(p, method="average")
    return (float(ranks[y].sum()) - npos * (npos + 1) / 2.0) / (npos * nneg)


def edge_metrics(prob, targets, threshold: float) -> dict[str, Any]:
    prob = torch.as_tensor(prob).detach().cpu().float()
    valid = targets["valid"].bool()
    truth = targets["target"].bool()
    p = prob[valid]; y = truth[valid]; pred = p >= threshold
    total, n = int(valid.numel()), int(valid.sum())
    pos = int(y.sum()) if n else 0
    neg = n - pos
    if n:
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_div(2 * precision * recall, precision + recall) if finite(precision) and finite(recall) else float("nan")
        balanced = 0.5 * (recall + specificity) if finite(recall) and finite(specificity) else float("nan")
        auc = roc_auc(p, y)
        pos_mean = float(p[y].mean()) if pos else float("nan")
        neg_mean = float(p[~y].mean()) if neg else float("nan")
    else:
        tp = fp = fn = tn = 0
        precision = recall = specificity = f1 = balanced = auc = float("nan")
        pos_mean = neg_mean = float("nan")
    return dict(
        edge_count=total,
        valid_edge_count=n,
        valid_edge_fraction=safe_div(n, total),
        positive_merge_edge_count=pos,
        negative_cut_edge_count=neg,
        merge_precision=precision,
        merge_recall=recall,
        merge_f1=f1,
        balanced_accuracy=balanced,
        roc_auc=auc,
        false_merge_count=fp,
        false_cut_count=fn,
        true_merge_count=tp,
        true_cut_count=tn,
        positive_probability_mean=pos_mean,
        negative_probability_mean=neg_mean,
        probability_margin_mean=(pos_mean - neg_mean) if finite(pos_mean) and finite(neg_mean) else float("nan"),
    )


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n)); self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.r[a] < self.r[b]: a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]: self.r[a] += 1


def repartition(sv, edge_index, merge_mask, *, node_count: int, node_offset: int) -> torch.Tensor:
    sv = torch.as_tensor(sv).detach().cpu().long()
    edges = torch.as_tensor(edge_index).detach().cpu().long()
    merge = torch.as_tensor(merge_mask).detach().cpu().bool()
    uf = UnionFind(node_count)
    for e in torch.nonzero(merge, as_tuple=False).flatten().tolist():
        a = int(edges[0, e]) - node_offset
        b = int(edges[1, e]) - node_offset
        if 0 <= a < node_count and 0 <= b < node_count:
            uf.union(a, b)
    roots: dict[int, int] = {}
    comps = []
    for i in range(node_count):
        root = uf.find(i)
        if root not in roots: roots[root] = len(roots) + 1
        comps.append(roots[root])
    mapping = torch.zeros(node_count + 1, dtype=torch.long)
    if comps: mapping[1:] = torch.tensor(comps)
    return mapping[sv]


# --------------------------------------------------------------------------------------
# Forward / graph extraction
# --------------------------------------------------------------------------------------

def extract_graph(output, rag_targets):
    rag = output.rag
    graph = dict(
        edge_index=rag.edge_index.detach().cpu().long(),
        edge_features=rag.edge_features.detach().float().cpu(),
        edge_prob=rag.spatial_edge_logits.detach().float().sigmoid().cpu(),
        node_centroid_um=rag.node_centroid_um.detach().float().cpu(),
        node_supervoxel_id=rag.node_supervoxel_id.detach().cpu().long(),
        node_offsets=rag.node_offsets.detach().cpu().long(),
    )
    targets = dict(
        target=rag_targets.target.detach().cpu().float(),
        valid=rag_targets.valid.detach().cpu().bool(),
        node_purity=rag_targets.node_purity.detach().cpu().float(),
        node_gt_support=rag_targets.node_gt_support.detach().cpu().float(),
        dominant_gt=rag_targets.dominant_gt.detach().cpu().long(),
    )
    return graph, targets


def oracle_labels(sv, graph, targets):
    edges = graph["edge_index"]
    dom = targets["dominant_gt"]
    a, b = dom[edges[0]], dom[edges[1]]
    same = (a > 0) & (a == b)
    start = int(graph["node_offsets"][0])
    n = int(graph["node_offsets"][1] - graph["node_offsets"][0])
    dominant = repartition(sv, edges, same, node_count=n, node_offset=start)
    valid = repartition(sv, edges, same & targets["valid"], node_count=n, node_offset=start)
    return valid, dominant


def run_checkpoint(
    *,
    role: str, step: int, model, model_cfg, train_cfg, criterion,
    crop, record, geom_targets, device: torch.device,
):
    from learned.stirnet.training.trainer import model_forward_from_batch, move_batch_to_device

    batch = move_batch_to_device(crop.batch, device)
    gt_device = crop.gt_labels.to(device)
    valid_device = crop.batch["supervision_valid_mask"].to(device)
    context, amp_name = E04._amp_context(device, train_cfg.amp_dtype)

    t0 = time.perf_counter()
    with torch.inference_mode():
        with context:
            output = model_forward_from_batch(
                model, batch, use_temporal=False, execution_stage="spatial",
                apply_existence_filter=False,
            )
            metrics = criterion(
                output, gt_device, batch["spacing_um"], batch["dref_um"],
                stage="spatial_partition",
                current_labels=batch.get("instance_labels"),
                precomputed_geometry_targets=geom_targets,
                supervision_valid_mask=valid_device,
            )
            rag_targets = criterion.rag.build_targets(
                output.rag, gt_device, valid_mask=valid_device
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    geometry = E04._compute_metrics(
        output=output, targets=geom_targets, crop=crop, record=record,
        model_cfg=model_cfg, criterion_metrics=metrics,
    )

    sv = output.rag.supervoxel_labels[0].detach().cpu().long()
    spatial = output.spatial_partition.labels[0].detach().cpu().long()
    gt = crop.gt_labels[0].detach().cpu().long()
    valid = crop.batch["supervision_valid_mask"][0].detach().cpu().bool()
    graph, rt = extract_graph(output, rag_targets)
    ov, od = oracle_labels(sv, graph, rt)

    row = dict(
        checkpoint_role=role,
        checkpoint_step=step,
        candidate_type=record.candidate_type,
        amp_dtype=amp_name,
        inference_seconds=elapsed,
        **geometry,
        **prefixed("proposal", supervoxel_audit(
            sv, gt, valid,
            min_purity=float(model_cfg.partition.rag_min_node_purity),
            min_support=float(model_cfg.partition.rag_min_node_gt_support),
        )),
        **prefixed("sv", partition_quality(sv, gt, valid)),
        **prefixed("rag", edge_metrics(
            graph["edge_prob"], rt,
            float(model_cfg.partition.spatial_merge_threshold)
        )),
        **prefixed("spatial", partition_quality(spatial, gt, valid)),
        **prefixed("oracle_valid", partition_quality(ov, gt, valid)),
        **prefixed("oracle_dominant", partition_quality(od, gt, valid)),
    )
    for k in (
        "spatial_rag_bce", "spatial_rag_accuracy", "rag_valid_edge_fraction",
        "rag_mean_node_purity", "rag_impure_node_fraction",
        "rag_mean_node_gt_support", "rag_low_support_node_fraction",
    ):
        if k in metrics:
            row[f"criterion_{k}"] = float(metrics[k].detach().float().cpu())

    arrays = dict(
        pred_foreground=output.geometry.foreground_logits[0, 0].detach().float().sigmoid().cpu(),
        pred_surface=output.geometry.surface_logits[0, 0].detach().float().sigmoid().cpu(),
        pred_separator=output.geometry.separator_logits[0, 0].detach().float().sigmoid().cpu(),
        pred_seed=output.geometry.seed_logits[0, 0].detach().float().sigmoid().cpu(),
        pred_sdf=output.geometry.sdf[0, 0].detach().float().cpu(),
        pred_flow=output.geometry.flow[0].detach().float().cpu(),
        pred_centroid_offset=output.geometry.centroid_offset[0].detach().float().cpu(),
        supervoxels=sv,
        spatial_partition=spatial,
        oracle_valid=ov,
        oracle_dominant=od,
    )

    del output, metrics, rag_targets, batch, gt_device, valid_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, arrays, graph, rt, gt, valid


def edge_rows(graph, targets, threshold: float) -> list[dict[str, Any]]:
    rows = []
    edges = graph["edge_index"]
    feat = graph["edge_features"]
    prob = graph["edge_prob"]
    valid = targets["valid"]; truth = targets["target"].bool()
    purity = targets["node_purity"]; support = targets["node_gt_support"]
    dom = targets["dominant_gt"]; ids = graph["node_supervoxel_id"]
    cent = graph["node_centroid_um"]
    for e in range(edges.shape[1]):
        a, b = int(edges[0, e]), int(edges[1, e])
        p = float(prob[e]); v = bool(valid[e]); y = bool(truth[e]); pred = p >= threshold
        outcome = (
            "invalid" if not v else
            "true_merge" if pred and y else
            "false_merge" if pred and not y else
            "false_cut" if (not pred) and y else
            "true_cut"
        )
        row = dict(
            edge_row=e, node_a=a, node_b=b,
            sv_a=int(ids[a]), sv_b=int(ids[b]),
            valid=v, target_merge=y, predicted_merge=pred,
            probability=p, outcome=outcome,
            purity_a=float(purity[a]), purity_b=float(purity[b]),
            support_a=float(support[a]), support_b=float(support[b]),
            dominant_gt_a=int(dom[a]), dominant_gt_b=int(dom[b]),
            z_a_um=float(cent[a, 0]), y_a_um=float(cent[a, 1]), x_a_um=float(cent[a, 2]),
            z_b_um=float(cent[b, 0]), y_b_um=float(cent[b, 1]), x_b_um=float(cent[b, 2]),
        )
        for i, name in enumerate(EDGE_FEATURE_NAMES):
            row[name] = float(feat[e, i])
        rows.append(row)
    return rows


def threshold_sweep(graph, sv, gt, valid, thresholds, *, sample, crop_index, step):
    start = int(graph["node_offsets"][0])
    n = int(graph["node_offsets"][1] - graph["node_offsets"][0])
    rows = []
    for th in thresholds:
        labels = repartition(
            sv, graph["edge_index"], graph["edge_prob"] >= th,
            node_count=n, node_offset=start,
        )
        rows.append(dict(
            checkpoint_step=step, sample=sample, crop_index=crop_index,
            threshold=th, **partition_quality(labels, gt, valid)
        ))
    return rows


# --------------------------------------------------------------------------------------
# Artifacts / visualization
# --------------------------------------------------------------------------------------

def np_array(t, *, fp16=False):
    a = torch.as_tensor(t).detach().cpu().numpy()
    return a.astype(np.float16) if fp16 and np.issubdtype(a.dtype, np.floating) else a


def save_npz(path: Path, crop, record, *, primary, graph, targets, step, baseline=None, baseline_step=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        normalized_raw=np_array(crop.batch["spatial_inputs"][0, 0], fp16=True),
        source_labels=np_array(crop.batch["instance_labels"][0]).astype(np.int32),
        gt_labels=np_array(crop.gt_labels[0]).astype(np.int32),
        supervision_valid=np_array(crop.batch["supervision_valid_mask"][0]).astype(np.uint8),
        spacing_zyx_um=np_array(crop.batch["spacing_um"][0]).astype(np.float32),
        dref_um=np.asarray([float(crop.batch["dref_um"][0])], np.float32),
        crop_bounds_zyx=np.asarray([[s.start, s.stop] for s in record.slices_zyx], np.int32),
        primary_step=np.asarray([step], np.int32),
        edge_index=np_array(graph["edge_index"]).astype(np.int32),
        edge_probability=np_array(graph["edge_prob"]).astype(np.float16),
        edge_target=np_array(targets["target"]).astype(np.uint8),
        edge_valid=np_array(targets["valid"]).astype(np.uint8),
        node_purity=np_array(targets["node_purity"]).astype(np.float16),
        node_gt_support=np_array(targets["node_gt_support"]).astype(np.float16),
        node_dominant_gt=np_array(targets["dominant_gt"]).astype(np.int32),
    )
    for k, v in primary.items():
        payload[f"primary_{k}"] = (
            np_array(v).astype(np.int32)
            if k in {"supervoxels", "spatial_partition", "oracle_valid", "oracle_dominant"}
            else np_array(v, fp16=True)
        )
    if baseline is not None:
        payload["baseline_step"] = np.asarray([baseline_step], np.int32)
        for k, v in baseline.items():
            payload[f"baseline_{k}"] = (
                np_array(v).astype(np.int32)
                if k in {"supervoxels", "spatial_partition", "oracle_valid", "oracle_dominant"}
                else np_array(v, fp16=True)
            )
    np.savez_compressed(path, **payload)


def save_montage(path: Path, npz_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    with np.load(npz_path, allow_pickle=False) as d:
        gt, valid = d["gt_labels"], d["supervision_valid"].astype(bool)
        score = ((gt > 0) & valid).sum((1, 2))
        z = int(score.argmax()) if score.size and score.max() else gt.shape[0] // 2
        panels = [
            ("raw", d["normalized_raw"][z], "gray", None, None),
            ("source", d["source_labels"][z], "nipy_spectral", None, None),
            ("GT", gt[z], "nipy_spectral", None, None),
            ("foreground", d["primary_pred_foreground"][z], "magma", 0, 1),
            ("separator", d["primary_pred_separator"][z], "magma", 0, 1),
            ("seed", d["primary_pred_seed"][z], "magma", 0, 1),
            ("supervoxels", d["primary_supervoxels"][z], "nipy_spectral", None, None),
            ("spatial", d["primary_spatial_partition"][z], "nipy_spectral", None, None),
            ("oracle valid", d["primary_oracle_valid"][z], "nipy_spectral", None, None),
            ("oracle dominant", d["primary_oracle_dominant"][z], "nipy_spectral", None, None),
            ("SDF", d["primary_pred_sdf"][z], "coolwarm", None, None),
            ("valid", d["supervision_valid"][z], "gray", 0, 1),
        ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    for ax, (name, img, cmap, vmin, vmax) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name); ax.axis("off")
    fig.suptitle(f"{title} | z={z}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def open_napari(npz_path: Path) -> None:
    import napari
    d = np.load(npz_path, allow_pickle=False)
    scale = tuple(float(x) for x in d["spacing_zyx_um"])
    v = napari.Viewer(title=f"STIR-Net spatial eval — {npz_path.parent.name}", ndisplay=3)
    v.add_image(d["normalized_raw"].astype(np.float32), name="normalized raw", scale=scale)
    v.add_labels(d["source_labels"].astype(np.int32), name="source labels", scale=scale, visible=False)
    v.add_labels(d["gt_labels"].astype(np.int32), name="GT labels", scale=scale)
    v.add_image(d["primary_pred_foreground"].astype(np.float32), name="pred foreground", scale=scale, visible=False)
    v.add_image(d["primary_pred_separator"].astype(np.float32), name="pred separator", scale=scale, visible=False)
    v.add_image(d["primary_pred_seed"].astype(np.float32), name="pred seed", scale=scale, visible=False)
    v.add_labels(d["primary_supervoxels"].astype(np.int32), name="watershed supervoxels", scale=scale, visible=False)
    v.add_labels(d["primary_spatial_partition"].astype(np.int32), name="learned spatial partition", scale=scale)
    v.add_labels(d["primary_oracle_valid"].astype(np.int32), name="oracle valid", scale=scale, visible=False)
    v.add_labels(d["primary_oracle_dominant"].astype(np.int32), name="oracle dominant", scale=scale, visible=False)
    if "baseline_spatial_partition" in d:
        v.add_labels(d["baseline_spatial_partition"].astype(np.int32), name="step300 untrained-RAG partition", scale=scale, visible=False)
        v.add_image(d["baseline_pred_separator"].astype(np.float32), name="step300 separator", scale=scale, visible=False)
    napari.run()
    d.close()


# --------------------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------------------

def evaluate(args):
    from learned.stirnet.training.criterion import StirNetCriterion
    from learned.stirnet.training.crop_target_cache import StaticCropTargetCache
    from learned.stirnet.training.prepared_geometry import compose_source_conditioned_geometry_targets

    primary_path = resolve_path(args.checkpoint)
    baseline_path = None if args.no_baseline else resolve_path(args.baseline_checkpoint)
    if not primary_path.exists():
        raise FileNotFoundError(
            f"Step-600 checkpoint not found:\n  {primary_path}\n"
            "Download checkpoint_step_000600.pt after the training run finishes."
        )
    if baseline_path is not None and not baseline_path.exists():
        raise FileNotFoundError(f"Step-300 baseline not found: {baseline_path}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    pckpt, pmodel, mcfg, tcfg = E04._load_checkpoint_model(primary_path, device)
    pstep = int(pckpt["global_step"])
    pcrit = StirNetCriterion(mcfg, tcfg.loss).to(device).eval()

    bmodel = bcfg = btcfg = bcrit = None
    bstep = None
    if baseline_path is not None:
        bckpt, bmodel, bcfg, btcfg = E04._load_checkpoint_model(baseline_path, device)
        bstep = int(bckpt["global_step"])
        bcrit = StirNetCriterion(bcfg, btcfg.loss).to(device).eval()
        if asdict(mcfg) != asdict(bcfg):
            raise ValueError("Step 300 and step 600 ModelConfig differ; comparison would be invalid.")

    samples = tuple(x.strip() for x in args.samples.split(",") if x.strip())
    if not samples:
        raise ValueError("No samples requested.")
    if "Zebrafish_1" in samples and not args.allow_zebrafish1:
        raise ValueError(
            "Investigation 05 defaults to exact-source spatial debugging and intentionally "
            "does not use the Zebrafish_1 crop-proxy domain. Pass --allow-zebrafish1 "
            "only if you explicitly want to attempt its full-source path."
        )

    nis3d_root = E04._discover_nis3d_root(
        samples, None if args.nis3d_root is None else Path(args.nis3d_root)
    )
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    if not thresholds or any(not 0 < x < 1 for x in thresholds):
        raise ValueError("Thresholds must be in (0,1).")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = resolve_path(args.output_dir) if args.output_dir else (
        ROOT / "runs/stirnet/evaluation" / SCRIPT_NAME /
        f"checkpoint_step_{pstep:06d}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "runs/stirnet/evaluation" / SCRIPT_NAME / "cache"
    source_cache = cache / "source"
    target_cache = StaticCropTargetCache(max_memory_entries=4, disk_dir=cache / "static_gt")

    print("=" * 116)
    print("STIR-Net Investigation 05 — spatial checkpoint evaluation")
    print("=" * 116)
    print(f"Primary       : {primary_path} (step {pstep})")
    print(f"Baseline      : {baseline_path} (step {bstep})")
    print(f"Samples       : {list(samples)}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU           : {prop.name} ({prop.total_memory / 2**30:.2f} GiB)")
    print(f"Crop          : {tuple(tcfg.curriculum.refinement_crop_shape_zyx)}")
    print(f"Crops/sample  : {args.crops_per_sample}")
    print(f"RAG threshold : {mcfg.partition.spatial_merge_threshold:.2f}")
    print(f"Sweep         : {list(thresholds)}")
    print(f"Output        : {output}")
    print("=" * 116)

    all_rows, primary_rows, baseline_rows, sweep_rows = [], [], [], []
    selected, reports = {}, {}
    first_npz = None
    started = time.perf_counter()

    for sample in samples:
        batch, report = E04._prepare_source_batch(
            nis3d_root, sample,
            source_cache_root=source_cache,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        )
        reports[sample] = report
        manifest = E04._build_eval_manifest(batch, tcfg)
        records = E04._select_eval_records(
            manifest, total=args.crops_per_sample, merge_fraction=args.merge_fraction
        )
        selected[sample] = [
            dict(
                crop_index=i, candidate_type=r.candidate_type,
                bounds_zyx=[[s.start, s.stop] for s in r.slices_zyx],
                complete_cell_count=len(r.complete_cell_ids),
                partial_cell_count=len(r.partial_cell_ids),
                merge_source_ids=list(r.merge_source_ids),
            )
            for i, r in enumerate(records)
        ]

        print(
            f"[data] {sample}: GT={report['gt_ids_kept']} "
            f"source={report['source_current_instance_count']} "
            f"valid={report['supervision_valid_fraction']:.4f} "
            f"dref={report['model_dref_um']:.4f}um"
        )

        for i, record in enumerate(records):
            spec = E04._record_to_spec(
                record,
                full_shape=tuple(int(v) for v in batch["gt_labels"].shape[-3:]),
                spacing=batch["spacing_um"][record.batch_index],
            )
            crop = E04._prepare_eval_crop(batch, spec, train_cfg=tcfg)
            c = tcfg.curriculum
            static, cache_stats = target_cache.get_or_build_batch(
                batch, batch["gt_labels"], [spec],
                spacing_um=batch["spacing_um"], dref_um=batch["dref_um"],
                geometry_config=mcfg.geometry, backend=args.target_backend,
                gpu_min_voxels=int(tcfg.geometry_target_gpu_min_voxels),
                halo_um=float(c.refinement_crop_target_halo_um),
            )
            targets = compose_source_conditioned_geometry_targets(
                static, crop.gt_labels, crop.batch.get("instance_labels"),
                crop.batch["spacing_um"], geometry_config=mcfg.geometry,
                backend=args.target_backend,
                gpu_min_voxels=int(tcfg.geometry_target_gpu_min_voxels),
            )

            b_arrays = None
            if bmodel is not None:
                brow, b_arrays, _, _, _, _ = run_checkpoint(
                    role="baseline_geometry", step=bstep, model=bmodel,
                    model_cfg=bcfg, train_cfg=btcfg, criterion=bcrit,
                    crop=crop, record=record, geom_targets=targets, device=device,
                )
                brow.update(sample=sample, crop_index=i)
                baseline_rows.append(brow); all_rows.append(brow)
                append_jsonl(output / "per_crop_metrics.jsonl", brow)

            prow, p_arrays, graph, rt, gt, valid = run_checkpoint(
                role="primary_spatial", step=pstep, model=pmodel,
                model_cfg=mcfg, train_cfg=tcfg, criterion=pcrit,
                crop=crop, record=record, geom_targets=targets, device=device,
            )
            prow.update(
                sample=sample, crop_index=i,
                target_cache_memory_hits=int(cache_stats["memory_hits"]),
                target_cache_disk_hits=int(cache_stats["disk_hits"]),
                target_cache_misses=int(cache_stats["misses"]),
            )
            primary_rows.append(prow); all_rows.append(prow)
            append_jsonl(output / "per_crop_metrics.jsonl", prow)

            sweep_rows += threshold_sweep(
                graph, p_arrays["supervoxels"], gt, valid, thresholds,
                sample=sample, crop_index=i, step=pstep,
            )

            crop_dir = output / sample / f"crop_{i:02d}_{record.candidate_type}"
            npz_path = crop_dir / "spatial_eval.npz"
            save_npz(
                npz_path, crop, record,
                primary=p_arrays, graph=graph, targets=rt, step=pstep,
                baseline=b_arrays, baseline_step=bstep,
            )
            write_csv(
                crop_dir / "edges.csv",
                edge_rows(graph, rt, float(mcfg.partition.spatial_merge_threshold)),
            )
            if not args.no_png:
                save_montage(
                    crop_dir / "spatial_montage.png", npz_path,
                    f"{sample} crop {i} {record.candidate_type} step {pstep}"
                )
            first_npz = first_npz or npz_path

            print(
                f"[eval] {sample} crop={i:02d} type={record.candidate_type:8s} "
                f"FG={prow['pred_foreground_dice']:.3f} "
                f"SEP={prow['separator_dice_at_0p5']:.3f} "
                f"SV={int(prow['proposal_supervoxel_count'])} "
                f"crossSV={int(prow['proposal_strict_cross_gt_node_count'])} "
                f"RAG_F1={prow['rag_merge_f1']:.3f} "
                f"spR50={prow['spatial_gt_recall_iou50']:.3f} "
                f"oracleR50={prow['oracle_valid_gt_recall_iou50']:.3f}"
            )
            del crop, static, targets, p_arrays, graph, rt
            if b_arrays is not None:
                del b_arrays
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(output / "per_crop_metrics.csv", all_rows)
    write_csv(output / "threshold_sweep.csv", sweep_rows)
    write_json(output / "selected_crops.json", selected)
    write_json(output / "sample_report.json", reports)

    # Geometry step300 -> step600 comparison on exactly the same crops.
    comparison = []
    if baseline_rows:
        bmap = {(r["sample"], r["crop_index"]): r for r in baseline_rows}
        for p in primary_rows:
            b = bmap[(p["sample"], p["crop_index"])]
            for key in (
                "pred_foreground_dice", "pred_foreground_recall",
                "surface_dice_at_0p5", "separator_dice_at_0p5",
                "seed_dice_at_0p5", "sdf_mae", "flow_cosine",
                "centroid_offset_cosine", "geometry_loss",
            ):
                if finite(p.get(key)) and finite(b.get(key)):
                    comparison.append(dict(
                        sample=p["sample"], crop_index=p["crop_index"], metric=key,
                        baseline_step=bstep, primary_step=pstep,
                        baseline_value=float(b[key]), primary_value=float(p[key]),
                        delta_primary_minus_baseline=float(p[key]) - float(b[key]),
                    ))
    write_csv(output / "geometry_comparison.csv", comparison)

    pagg = aggregate(primary_rows)
    bagg = aggregate(baseline_rows) if baseline_rows else None

    threshold_summary = []
    for th in thresholds:
        rows = [r for r in sweep_rows if abs(r["threshold"] - th) < 1e-9]
        threshold_summary.append(dict(threshold=th, **aggregate(rows, exclude=("checkpoint_step", "crop_index", "threshold"))))

    flags = []
    oracle = pagg.get("mean_oracle_valid_gt_recall_iou50")
    learned = pagg.get("mean_spatial_gt_recall_iou50")
    oracle_dom = pagg.get("mean_oracle_dominant_gt_recall_iou50")
    cross = pagg.get("mean_proposal_strict_cross_gt_node_fraction")
    if finite(oracle_dom) and oracle_dom < 0.80:
        flags.append("proposal/topology: dominant-GT oracle recall@0.5 is below 0.80")
    if finite(cross) and cross > 0.02:
        flags.append("proposal/topology: >2% of supervoxels are strict cross-GT atomic nodes")
    if finite(oracle) and finite(learned) and oracle - learned > 0.10:
        flags.append("RAG: valid-edge oracle exceeds learned spatial recall@0.5 by >0.10")
    if threshold_summary:
        best = max(threshold_summary, key=lambda r: (r.get("mean_gt_recall_iou50", -1), r.get("mean_mean_best_gt_iou", -1)))
        default = min(threshold_summary, key=lambda r: abs(r["threshold"] - float(mcfg.partition.spatial_merge_threshold)))
        if (
            finite(best.get("mean_gt_recall_iou50"))
            and finite(default.get("mean_gt_recall_iou50"))
            and best["mean_gt_recall_iou50"] - default["mean_gt_recall_iou50"] > 0.05
        ):
            flags.append(f"calibration: threshold {best['threshold']:.2f} beats default by >0.05 recall@0.5")
    if bagg:
        if finite(pagg.get("mean_pred_foreground_dice")) and finite(bagg.get("mean_pred_foreground_dice")):
            if pagg["mean_pred_foreground_dice"] < bagg["mean_pred_foreground_dice"] - 0.03:
                flags.append("geometry regression: foreground Dice dropped >0.03 from step300")
        if finite(pagg.get("mean_separator_dice_at_0p5")) and finite(bagg.get("mean_separator_dice_at_0p5")):
            if pagg["mean_separator_dice_at_0p5"] < bagg["mean_separator_dice_at_0p5"] - 0.05:
                flags.append("geometry regression: separator Dice dropped >0.05 from step300")
    if not flags:
        flags.append("no large failure crossed the evaluator's conservative heuristic thresholds")

    elapsed = time.perf_counter() - started
    summary = dict(
        status="success",
        primary_checkpoint=str(primary_path), primary_step=pstep,
        baseline_checkpoint=None if baseline_path is None else str(baseline_path),
        baseline_step=bstep,
        samples=list(samples), device=str(device),
        crop_shape_zyx=list(tcfg.curriculum.refinement_crop_shape_zyx),
        crops_per_sample=args.crops_per_sample,
        spatial_merge_threshold=float(mcfg.partition.spatial_merge_threshold),
        watershed_backend=mcfg.partition.watershed_backend,
        elapsed_seconds=elapsed,
        output_root=str(output),
        primary_aggregate=pagg,
        baseline_aggregate=bagg,
        threshold_summary=threshold_summary,
        diagnostic_clues=flags,
    )
    write_json(output / "summary.json", summary)

    print("\n" + "=" * 116)
    print("SPATIAL EVALUATION COMPLETE")
    print("=" * 116)
    print(f"Output                : {output}")
    print(f"Geometry FG Dice      : {pagg.get('mean_pred_foreground_dice', float('nan')):.3f}")
    print(f"Separator Dice        : {pagg.get('mean_separator_dice_at_0p5', float('nan')):.3f}")
    print(f"Strict cross-GT SV    : {pagg.get('mean_proposal_strict_cross_gt_node_count', float('nan')):.2f}/crop")
    print(f"RAG merge F1          : {pagg.get('mean_rag_merge_f1', float('nan')):.3f}")
    print(f"Spatial GT recall@.50 : {pagg.get('mean_spatial_gt_recall_iou50', float('nan')):.3f}")
    print(f"Oracle-valid R@.50    : {pagg.get('mean_oracle_valid_gt_recall_iou50', float('nan')):.3f}")
    print(f"Oracle-dom R@.50      : {pagg.get('mean_oracle_dominant_gt_recall_iou50', float('nan')):.3f}")
    print("Clues:")
    for x in flags:
        print(f"  - {x}")
    print("=" * 116)

    if args.napari:
        if first_npz is None:
            raise RuntimeError("No NPZ artifact was created.")
        open_napari(first_npz)
    return summary


def parser():
    p = argparse.ArgumentParser(description="Evaluate STIR-Net step-600 spatial checkpoint.")
    p.add_argument("--checkpoint", default=str(DEFAULT_PRIMARY))
    p.add_argument("--baseline-checkpoint", default=str(DEFAULT_BASELINE))
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--samples", default="Zebrafish_2")
    p.add_argument("--allow-zebrafish1", action="store_true")
    p.add_argument("--nis3d-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--crops-per-sample", type=int, default=6)
    p.add_argument("--merge-fraction", type=float, default=0.5)
    p.add_argument("--confidence-ignore-margin-um", type=float, default=1.0)
    p.add_argument("--target-backend", choices=("auto", "scipy", "cupy"), default="auto")
    p.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--napari", action="store_true")
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.crops_per_sample < 1: p.error("--crops-per-sample must be >= 1")
    if not 0 <= a.merge_fraction <= 1: p.error("--merge-fraction must be in [0,1]")
    summary = evaluate(a)
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
