from __future__ import annotations

r"""
STIR-Net Investigation 22 — counterfactual component rescoring.

Scientific question
-------------------
Does an atomic-supervoxel edge fail because the classifier is being asked the
question at the wrong state?

Canonical failure:

    A -- B -- C

GT says A+B is one cell and C is another.  Atomic B can look incomplete, so the
model may incorrectly score B-C as MERGE.  After the correct A+B contraction,
the meaningful question is (A+B)-C.

This investigation performs exactly that counterfactual test with the CURRENT
trained model and frozen dense geometry:

    1. run a normal NIS3D crop;
    2. find valid GT-positive A-B edges that the model would plausibly merge;
    3. find GT-negative C neighbours incident to A or B;
    4. record the original dangerous A/B-C probability;
    5. merge A+B in the watershed label state only;
    6. rebuild the production RAG/morphology representation for the new state;
    7. record P((A+B),C).

No weights are changed.  The experiment measures whether component-aware
rescoring itself fixes the error.

The script reuses Investigation-17's NIS3D loading/crop helpers so the data and
validity semantics remain identical to prior morphology training.

Key outputs
-----------
    cases.csv
    summary.json
    per_crop.jsonl

Important metrics:
    false_merge_before_count
    false_merge_after_count
    rescued_false_merge_count
    worsened_to_false_merge_count
    mean_probability_delta

A large rescued fraction is direct evidence for iterative component-level
agglomeration/rescoring.

Recommended smoke:

    python .\investigations\stirnet\22_counterfactual_component_rescoring.py `
        --checkpoint .\runs\stirnet\investigations\19_morphology_rag_v2_headroom_training\recovery\drosophila_12_morphology_rag_v2_headroom_h100\checkpoint_step_000600.pt `
        --max-crops 2 `
        --max-positive-merges-per-crop 2

Recommended diagnostic run:

    python .\investigations\stirnet\22_counterfactual_component_rescoring.py `
        --checkpoint <checkpoint.pt> `
        --max-crops 12 `
        --max-positive-merges-per-crop 6
"""

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_NAME = "22_counterfactual_component_rescoring"
DEFAULT_SAMPLES = "Drosophila_1,Drosophila_2"
DEFAULT_SPACING_XYZ = "0.20312639,0.20312639,0.79099447"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_inv17_support():
    path = ROOT / "investigations" / "stirnet" / "17_morphology_rag_multicrop_training.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_stirnet_inv17_support_for_inv22", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import support module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def resolve_checkpoint(value: str) -> Path:
    path = resolve(value)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    preferred = path / "best_checkpoint.pt"
    if preferred.is_file():
        return preferred
    rows = sorted(path.glob("checkpoint_step_*.pt"))
    if rows:
        return rows[-1]
    rows = sorted(path.glob("**/best_checkpoint.pt")) or sorted(path.glob("**/checkpoint_step_*.pt"))
    if rows:
        return rows[-1]
    raise FileNotFoundError(f"No checkpoint found below {path}")


def parse_triplet(text: str, *, cast=float) -> tuple:
    values = tuple(cast(v.strip()) for v in text.split(","))
    if len(values) != 3:
        raise ValueError(f"Expected three comma-separated values, got {text!r}")
    return values


def parse_samples(text: str) -> tuple[str, ...]:
    rows = tuple(v.strip() for v in text.split(",") if v.strip())
    if not rows:
        raise ValueError("--samples cannot be empty")
    return rows


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_two_labels_contiguous(labels, label_a: int, label_b: int):
    """Merge two positive label IDs and compact all positive IDs to 1..K."""
    import torch

    if label_a == label_b or label_a <= 0 or label_b <= 0:
        raise ValueError("Need two distinct positive label IDs")
    n = int(labels.max().item())
    if label_a > n or label_b > n:
        raise IndexError((label_a, label_b, n))

    canonical = min(label_a, label_b)
    absorbed = max(label_a, label_b)
    group_key = []
    for old in range(1, n + 1):
        group_key.append(canonical if old == absorbed else old)

    key_to_new: dict[int, int] = {}
    old_to_new = torch.zeros(n + 1, device=labels.device, dtype=torch.long)
    for old, key in enumerate(group_key, start=1):
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new) + 1
        old_to_new[old] = key_to_new[key]

    merged = old_to_new[labels.long()]
    return merged, old_to_new


def find_edge_row(edge_index, node_a: int, node_b: int) -> int | None:
    import torch
    if edge_index.shape[1] == 0:
        return None
    a = edge_index[0]
    b = edge_index[1]
    mask = ((a == node_a) & (b == node_b)) | ((a == node_b) & (b == node_a))
    rows = torch.nonzero(mask, as_tuple=False).flatten()
    return None if rows.numel() == 0 else int(rows[0].item())


def candidate_positive_groups(output, targets, *, merge_threshold: float, min_external_probability: float, max_groups: int):
    """Return promising GT-safe A-B contractions and their GT-negative neighbours."""
    import torch

    rag = output.rag
    p = rag.spatial_edge_logits.detach().float().sigmoid()
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5) & (p >= merge_threshold)
    negative = valid & (targets.target <= 0.5) & (p >= min_external_probability)
    positive_rows = torch.nonzero(positive, as_tuple=False).flatten()
    negative_rows = torch.nonzero(negative, as_tuple=False).flatten()

    groups = []
    seen = set()
    for pos_row_tensor in positive_rows:
        pos_row = int(pos_row_tensor.item())
        a = int(rag.edge_index[0, pos_row].item())
        b = int(rag.edge_index[1, pos_row].item())
        key_ab = tuple(sorted((a, b)))
        if key_ab in seen:
            continue

        external = []
        for neg_row_tensor in negative_rows:
            neg_row = int(neg_row_tensor.item())
            u = int(rag.edge_index[0, neg_row].item())
            v = int(rag.edge_index[1, neg_row].item())
            if u in key_ab and v not in key_ab:
                c = v
            elif v in key_ab and u not in key_ab:
                c = u
            else:
                continue
            external.append((float(p[neg_row].item()), neg_row, c, u, v))

        if not external:
            continue
        external.sort(reverse=True, key=lambda row: row[0])
        groups.append(
            {
                "a": key_ab[0],
                "b": key_ab[1],
                "positive_edge_row": pos_row,
                "positive_probability": float(p[pos_row].item()),
                "external": external,
                "priority": (external[0][0], float(p[pos_row].item())),
            }
        )
        seen.add(key_ab)

    groups.sort(reverse=True, key=lambda row: row["priority"])
    return groups[:max_groups]


def write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample", "manifest_index", "case_index", "atomic_a_node", "atomic_b_node",
        "atomic_c_node", "atomic_a_sv", "atomic_b_sv", "atomic_c_sv",
        "ab_probability", "before_edge_probability", "after_edge_probability",
        "probability_delta", "before_false_merge", "after_false_merge", "rescued",
        "worsened", "new_ab_label", "new_c_label", "new_edge_row",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigation 22: counterfactual merge-then-rescore diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--spacing-xyz", default=DEFAULT_SPACING_XYZ)
    parser.add_argument("--crop-shape-zyx", default="32,192,192")
    parser.add_argument("--validation-crops-per-sample", type=int, default=6)
    parser.add_argument("--max-crops", type=int, default=12)
    parser.add_argument("--max-positive-merges-per-crop", type=int, default=6)
    parser.add_argument("--max-external-neighbours-per-merge", type=int, default=8)
    parser.add_argument("--min-external-probability", type=float, default=0.50)
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--confidence-ignore-margin-um", type=float, default=0.0)
    parser.add_argument("--partial-ignore-margin-um", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17022)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-rescued-volumes", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion

    if not torch.cuda.is_available():
        raise RuntimeError("Investigation 22 currently requires CUDA")
    seed_everything(args.seed)
    support = load_inv17_support()
    checkpoint = resolve_checkpoint(args.checkpoint)
    samples = parse_samples(args.samples)
    xyz = parse_triplet(args.spacing_xyz, cast=float)
    spacing_zyx = (float(xyz[2]), float(xyz[1]), float(xyz[0]))
    crop_shape = tuple(int(v) for v in parse_triplet(args.crop_shape_zyx, cast=int))

    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    model, model_cfg, checkpoint_payload, transfer = support._build_morphology_model(checkpoint, device="cuda")
    model.eval()
    merge_threshold = float(args.merge_threshold if args.merge_threshold is not None else model_cfg.partition.spatial_merge_threshold)
    criterion = RAGCriterion(model_cfg.partition)

    nis3d_root = support._discover_nis3d_root(samples, data_dir=args.data_dir, execution_mode="local")
    cache_root = ROOT / "runs" / "stirnet" / "cache" / EXPERIMENT_NAME
    source_batches = {}
    source_reports = {}
    for sample in samples:
        batch, report = support._prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_zyx,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
            cache_root=cache_root,
            cache_namespace="component_rescore_v1",
        )
        source_batches[sample] = batch
        source_reports[sample] = report

    splits = support._build_split(
        source_batches,
        crop_shape_zyx=crop_shape,
        validation_crops_per_sample=args.validation_crops_per_sample,
    )

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = resolve(args.output_dir) if args.output_dir else (
        ROOT / "runs" / "stirnet" / "investigations" / EXPERIMENT_NAME / f"{now}_counterfactual_rescore"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_global_step": int(checkpoint_payload.get("global_step", 0)),
        "samples": list(samples),
        "spacing_zyx_um": list(spacing_zyx),
        "crop_shape_zyx": list(crop_shape),
        "merge_threshold": merge_threshold,
        "min_external_probability": float(args.min_external_probability),
        "max_crops": int(args.max_crops),
        "max_positive_merges_per_crop": int(args.max_positive_merges_per_crop),
        "transfer": transfer,
        "source_reports": source_reports,
    }
    atomic_json(output_dir / "manifest.json", manifest)

    crop_plan = []
    for sample in samples:
        for index in splits[sample]["validation_indices"]:
            crop_plan.append((sample, int(index)))
    crop_plan = crop_plan[: max(0, args.max_crops)] if args.max_crops > 0 else crop_plan

    all_rows: list[dict[str, Any]] = []
    case_ordinal = 0
    rescued_volume_ordinal = 0
    started = time.perf_counter()

    print("=" * 112)
    print("STIR-Net Investigation 22 — counterfactual component rescoring")
    print("=" * 112)
    print("checkpoint       :", checkpoint)
    print("samples          :", samples)
    print("crops            :", len(crop_plan))
    print("merge threshold  :", merge_threshold)
    print("external p floor :", args.min_external_probability)
    print("output           :", output_dir)
    print("=" * 112, flush=True)

    for crop_number, (sample, manifest_index) in enumerate(crop_plan, start=1):
        record = splits[sample]["records"][manifest_index]
        crop_cpu, _ = support._materialize_crop(
            source_batches[sample], record, partial_ignore_margin_um=args.partial_ignore_margin_um
        )
        crop = support._move_crop_to_cuda(crop_cpu)

        with torch.no_grad(), support._autocast_context(amp_dtype):
            geometry_forward = model(
                crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"), execution_stage="geometry",
            )
            output = model(
                crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"), execution_stage="spatial",
                precomputed_geometry=geometry_forward,
            )

        targets = criterion.build_targets(output.rag, crop["gt_labels"], valid_mask=crop.get("supervision_valid_mask"))
        groups = candidate_positive_groups(
            output, targets,
            merge_threshold=merge_threshold,
            min_external_probability=args.min_external_probability,
            max_groups=args.max_positive_merges_per_crop,
        )
        crop_rows = []

        for group in groups:
            a, b = int(group["a"]), int(group["b"])
            sv_a = int(output.rag.node_supervoxel_id[a].item())
            sv_b = int(output.rag.node_supervoxel_id[b].item())
            atomic_labels = output.rag.supervoxel_labels[0]
            merged_labels, old_to_new = merge_two_labels_contiguous(atomic_labels, sv_a, sv_b)
            new_ab_label = int(old_to_new[sv_a].item())

            with torch.no_grad(), support._autocast_context(amp_dtype):
                new_rag, _ = model._rag_from_supervoxels(
                    [merged_labels],
                    geometry_forward.geometry,
                    geometry_forward.decoded_spatial,
                    crop["spatial_inputs"],
                    crop["spacing_um"],
                    crop["dref_um"],
                    profile_prefix="inv22_counterfactual",
                )
            new_targets = criterion.build_targets(new_rag, crop["gt_labels"], valid_mask=crop.get("supervision_valid_mask"))
            new_probability = new_rag.spatial_edge_logits.detach().float().sigmoid()

            seen_c = set()
            for before_p, neg_row, c, _, _ in group["external"][: args.max_external_neighbours_per_merge]:
                c = int(c)
                sv_c = int(output.rag.node_supervoxel_id[c].item())
                if sv_c in seen_c:
                    continue
                seen_c.add(sv_c)
                new_c_label = int(old_to_new[sv_c].item())
                if new_c_label == new_ab_label:
                    continue
                new_edge_row = find_edge_row(new_rag.edge_index, new_ab_label - 1, new_c_label - 1)
                if new_edge_row is None:
                    continue
                if not bool(new_targets.valid[new_edge_row].item()) or float(new_targets.target[new_edge_row].item()) > 0.5:
                    continue

                after_p = float(new_probability[new_edge_row].item())
                before_false = float(before_p) >= merge_threshold
                after_false = after_p >= merge_threshold
                row = {
                    "sample": sample,
                    "manifest_index": manifest_index,
                    "case_index": case_ordinal,
                    "atomic_a_node": a,
                    "atomic_b_node": b,
                    "atomic_c_node": c,
                    "atomic_a_sv": sv_a,
                    "atomic_b_sv": sv_b,
                    "atomic_c_sv": sv_c,
                    "ab_probability": float(group["positive_probability"]),
                    "before_edge_probability": float(before_p),
                    "after_edge_probability": after_p,
                    "probability_delta": after_p - float(before_p),
                    "before_false_merge": int(before_false),
                    "after_false_merge": int(after_false),
                    "rescued": int(before_false and not after_false),
                    "worsened": int((not before_false) and after_false),
                    "new_ab_label": new_ab_label,
                    "new_c_label": new_c_label,
                    "new_edge_row": int(new_edge_row),
                }
                case_ordinal += 1
                crop_rows.append(row)
                all_rows.append(row)

            if args.save_rescued_volumes and any(row["rescued"] for row in crop_rows):
                # Save one compact diagnostic volume per rescued contraction, not per C neighbour.
                rescued_volume_ordinal += 1
                probabilities = geometry_forward.geometry.probabilities()
                np.savez_compressed(
                    output_dir / f"rescued_crop_{rescued_volume_ordinal:03d}_{sample}_m{manifest_index:03d}.npz",
                    raw=crop["spatial_inputs"][0, 0].detach().float().cpu().numpy().astype(np.float16),
                    separator=probabilities["separator"][0, 0].detach().float().cpu().numpy().astype(np.float16),
                    gt=crop["gt_labels"][0].detach().cpu().numpy().astype(np.int32),
                    atomic_supervoxels=atomic_labels.detach().cpu().numpy().astype(np.int32),
                    counterfactual_supervoxels=merged_labels.detach().cpu().numpy().astype(np.int32),
                    merged_sv_ids=np.asarray([sv_a, sv_b], dtype=np.int32),
                )

        crop_report = {
            "sample": sample,
            "manifest_index": manifest_index,
            "candidate_positive_contractions": len(groups),
            "counterfactual_edges": len(crop_rows),
            "rescued": int(sum(row["rescued"] for row in crop_rows)),
            "worsened": int(sum(row["worsened"] for row in crop_rows)),
            "mean_delta": safe_mean([float(row["probability_delta"]) for row in crop_rows]),
        }
        append_jsonl(output_dir / "per_crop.jsonl", crop_report)
        print(
            f"[{crop_number:02d}/{len(crop_plan):02d}] {sample} crop={manifest_index} "
            f"AB={len(groups)} edges={len(crop_rows)} rescued={crop_report['rescued']} "
            f"mean Δp={crop_report['mean_delta']:+.4f}", flush=True
        )

        del crop, crop_cpu, output, geometry_forward, targets
        torch.cuda.empty_cache()
        gc.collect()

    write_cases_csv(output_dir / "cases.csv", all_rows)
    before = [float(row["before_edge_probability"]) for row in all_rows]
    after = [float(row["after_edge_probability"]) for row in all_rows]
    deltas = [float(row["probability_delta"]) for row in all_rows]
    before_false = int(sum(row["before_false_merge"] for row in all_rows))
    after_false = int(sum(row["after_false_merge"] for row in all_rows))
    rescued = int(sum(row["rescued"] for row in all_rows))
    worsened = int(sum(row["worsened"] for row in all_rows))

    summary = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(checkpoint),
        "case_count": len(all_rows),
        "false_merge_before_count": before_false,
        "false_merge_after_count": after_false,
        "rescued_false_merge_count": rescued,
        "rescued_fraction_of_before_false_merges": float(rescued / before_false) if before_false else 0.0,
        "worsened_to_false_merge_count": worsened,
        "mean_before_probability": safe_mean(before),
        "mean_after_probability": safe_mean(after),
        "mean_probability_delta": safe_mean(deltas),
        "fraction_probability_decreased": float(sum(v < 0 for v in deltas) / len(deltas)) if deltas else 0.0,
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation": {
            "rescued_high": "component state is a major missing variable; proceed to component-state training/agglomeration",
            "rescued_low": "atomic-to-component state change alone does not solve most false merges; inspect separator/graph evidence",
        },
    }
    atomic_json(output_dir / "summary.json", summary)

    print("=" * 112)
    print("Investigation 22 complete")
    print(json.dumps(summary, indent=2))
    print("cases:", output_dir / "cases.csv")
    print("=" * 112)


if __name__ == "__main__":
    main()
