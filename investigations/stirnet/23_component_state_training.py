# STIRNET_INV23_PRODUCTIVE_COMPONENT_MINING_V1
# STIRNET_INV23_SHARED_SOURCE_CACHE_V1
from __future__ import annotations

r"""
STIR-Net Investigation 23 — train on component states.

Purpose
-------
Train the existing morphology-aware RAG on the states it would encounter after
correct supervoxel contractions, instead of supervising only the original
atomic-supervoxel graph.

The motivating failure is:

    A -- B -- C

where GT says A+B is one cell and C is another.  B alone can look incomplete and
B-C can be scored as a merge.  After a correct A+B contraction, the model should
see the whole AB component and classify AB-C.

Training construction
---------------------
For each NIS3D crop:

1. Freeze dense geometry and compute the normal atomic watershed/RAG.
2. Use GT-valid same-cell RAG edges to make 1..N *safe* contractions.  By
   default we prefer same-cell edges that the current model already accepts, so
   the state resembles an on-policy agglomeration trajectory.
3. Relabel the watershed into the resulting component state.
4. Rebuild the CURRENT production RAG/morphology representation on that state.
   The node morphology encoder now sees the complete merged component mask.
5. Supervise edges incident to newly merged components with class-balanced BCE.
6. Add a lower-weight context BCE on unchanged / non-focus edges in the same
   state to reduce regression on ordinary atomic relationships.

Dense geometry and the legacy RAG remain frozen.  By default only the morphology
builder + morphology residual projections are trainable.

This is deliberately an investigation, not a production partitioner.  It does
not yet implement the iterative inference loop; it trains the scorer that such a
loop would need.

Validation
----------
Validation reports BOTH:

* atomic graph metrics (must not regress badly), and
* component-state focus metrics after deterministic GT-safe contractions.

Best checkpoint ranking first enforces an atomic positive-acceptance guard, then
prefers lower component-state false-merge rate, then lower component-state BCE.

Recommended smoke:

    python .\investigations\stirnet\23_component_state_training.py `
        --checkpoint .\runs\stirnet\investigations\19_morphology_rag_v2_headroom_training\recovery\drosophila_12_morphology_rag_v2_headroom_h100\checkpoint_step_000600.pt `
        --max-steps 4 `
        --validation-every 2 `
        --validation-crops-per-sample 2 `
        --max-state-merges 2 `
        --run-name component_state_smoke

Recommended first run:

    python .\investigations\stirnet\23_component_state_training.py `
        --checkpoint <h100-or-chosen-morphology-checkpoint.pt> `
        --max-steps 600 `
        --max-state-merges 3 `
        --run-name drosophila_12_component_state_v1

Resume an Investigation-23 checkpoint:

    python .\investigations\stirnet\23_component_state_training.py `
        --checkpoint <checkpoint_step_XXXXXX.pt> `
        --resume `
        --max-steps 600
"""

import argparse
import gc
import hashlib
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

from tqdm import tqdm

EXPERIMENT_NAME = "23_component_state_training"
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
    spec = importlib.util.spec_from_file_location("_stirnet_inv17_support_for_inv23", path)
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
    best = path / "best_checkpoint.pt"
    if best.is_file():
        return best
    rows = sorted(path.glob("checkpoint_step_*.pt"))
    if rows:
        return rows[-1]
    rows = sorted(path.glob("**/best_checkpoint.pt")) or sorted(path.glob("**/checkpoint_step_*.pt"))
    if rows:
        return rows[-1]
    raise FileNotFoundError(f"No checkpoint found below {path}")


def torch_load(path: Path, *, map_location="cpu") -> dict:
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_triplet(text: str, *, cast=float) -> tuple:
    values = tuple(cast(v.strip()) for v in text.split(","))
    if len(values) != 3:
        raise ValueError(f"Expected three comma-separated values: {text}")
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


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def build_component_state(atomic_rag, atomic_targets, *, probabilities, target_merges: int, merge_threshold: float, rng: random.Random, require_model_accepted: bool):
    """GT-safe graph contractions; prefer growth around an already merged component."""
    import torch

    node_count = int(atomic_rag.node_features.shape[0])
    if node_count == 0 or target_merges <= 0:
        return None

    valid_positive = atomic_targets.valid.bool() & (atomic_targets.target > 0.5)
    rows = torch.nonzero(valid_positive, as_tuple=False).flatten().detach().cpu().tolist()
    if require_model_accepted:
        accepted = [r for r in rows if float(probabilities[r]) >= merge_threshold]
        rows = accepted
    if not rows:
        return None

    # High-confidence same-GT contractions are the most plausible on-policy moves.
    rows = sorted(rows, key=lambda r: float(probabilities[r]), reverse=True)
    # Slight diversity among similarly strong edges without throwing away the on-policy bias.
    head = rows[: min(len(rows), max(8, 4 * target_merges))]
    rng.shuffle(head)
    head.sort(key=lambda r: float(probabilities[r]), reverse=True)
    candidates = head + [r for r in rows if r not in set(head)]

    dsu = DSU(node_count)
    selected_rows: list[int] = []
    active_roots: set[int] = set()

    while len(selected_rows) < target_merges:
        best = None
        # Prefer an edge that expands an already contracted component.
        for edge_row in candidates:
            u = int(atomic_rag.edge_index[0, edge_row].item())
            v = int(atomic_rag.edge_index[1, edge_row].item())
            ru, rv = dsu.find(u), dsu.find(v)
            if ru == rv:
                continue
            touches_active = ru in active_roots or rv in active_roots
            score = (1 if touches_active else 0, float(probabilities[edge_row]))
            if best is None or score > best[0]:
                best = (score, edge_row, u, v)
        if best is None:
            break
        _, edge_row, u, v = best
        old_roots = {dsu.find(u), dsu.find(v)}
        new_root = dsu.union(u, v)
        active_roots.difference_update(old_roots)
        active_roots.add(new_root)
        selected_rows.append(int(edge_row))

    if not selected_rows:
        return None

    root_to_component: dict[int, int] = {}
    node_to_component = [0] * node_count
    members: dict[int, list[int]] = {}
    for node in range(node_count):
        root = dsu.find(node)
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component)
        comp = root_to_component[root]
        node_to_component[node] = comp
        members.setdefault(comp, []).append(node)

    merged_components = sorted(comp for comp, rows_ in members.items() if len(rows_) > 1)
    if not merged_components:
        return None

    atomic_labels = atomic_rag.supervoxel_labels[0]
    max_sv = int(atomic_labels.max().item())
    sv_to_component_label = torch.zeros(max_sv + 1, device=atomic_labels.device, dtype=torch.long)
    for node, comp in enumerate(node_to_component):
        sv = int(atomic_rag.node_supervoxel_id[node].item())
        if sv > max_sv:
            raise IndexError("node_supervoxel_id exceeds atomic label maximum")
        sv_to_component_label[sv] = comp + 1
    state_labels = sv_to_component_label[atomic_labels.long()]

    return {
        "labels": state_labels,
        "node_to_component": node_to_component,
        "merged_components": merged_components,
        "selected_edge_rows": selected_rows,
        "selected_probabilities": [float(probabilities[r]) for r in selected_rows],
    }


def focus_mask_for_state(state_rag, targets, merged_components: list[int]):
    import torch
    if state_rag.edge_index.shape[1] == 0:
        return targets.valid.bool()
    merged = torch.zeros(state_rag.node_features.shape[0], device=state_rag.node_features.device, dtype=torch.bool)
    ids = [v for v in merged_components if 0 <= int(v) < len(merged)]
    if ids:
        merged[torch.as_tensor(ids, device=merged.device, dtype=torch.long)] = True
    incident = merged[state_rag.edge_index[0]] | merged[state_rag.edge_index[1]]
    return targets.valid.bool() & incident


def balanced_bce(logits, targets, mask, *, max_per_class: int, hard_negative_first: bool = True):
    import torch
    import torch.nn.functional as F

    positive = torch.nonzero(mask & (targets.target > 0.5), as_tuple=False).flatten()
    negative = torch.nonzero(mask & (targets.target <= 0.5), as_tuple=False).flatten()

    if positive.numel() > max_per_class:
        positions = torch.linspace(0, positive.numel() - 1, max_per_class, device=positive.device).round().long()
        positive = positive[positions]
    if negative.numel() > max_per_class:
        if hard_negative_first:
            p = logits.detach().sigmoid()[negative]
            order = torch.argsort(p, descending=True)
            negative = negative[order[:max_per_class]]
        else:
            positions = torch.linspace(0, negative.numel() - 1, max_per_class, device=negative.device).round().long()
            negative = negative[positions]

    pieces = []
    if positive.numel():
        pieces.append(F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive])))
    if negative.numel():
        pieces.append(F.binary_cross_entropy_with_logits(logits[negative], torch.zeros_like(logits[negative])))
    if not pieces:
        return logits.sum() * 0.0, {"positive": 0, "negative": 0, "total": 0}
    return torch.stack(pieces).mean(), {
        "positive": int(positive.numel()),
        "negative": int(negative.numel()),
        "total": int(positive.numel() + negative.numel()),
    }


# ======================================================================================
# Productive component-state crop mining
# ======================================================================================

MINING_VERSION = 1


def _mining_fingerprint(
    *,
    checkpoint: Path,
    samples,
    spacing_zyx,
    crop_shape,
    splits,
    confidence_ignore_margin_um: float,
    partial_ignore_margin_um: float,
    max_state_merges: int,
    require_model_accepted: bool,
    mining_max_crops_per_sample: int,
    seed: int,
) -> str:
    payload = {
        "version": MINING_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": int(checkpoint.stat().st_size),
        "checkpoint_mtime_ns": int(checkpoint.stat().st_mtime_ns),
        "samples": list(samples),
        "spacing_zyx": [float(v) for v in spacing_zyx],
        "crop_shape": [int(v) for v in crop_shape],
        "confidence_ignore_margin_um": float(confidence_ignore_margin_um),
        "partial_ignore_margin_um": float(partial_ignore_margin_um),
        "max_state_merges": int(max_state_merges),
        "require_model_accepted": bool(require_model_accepted),
        "mining_max_crops_per_sample": int(mining_max_crops_per_sample),
        "seed": int(seed),
        "split": {
            sample: {
                "train_indices": [int(v) for v in splits[sample]["train_indices"]],
                "validation_indices": [int(v) for v in splits[sample]["validation_indices"]],
            }
            for sample in samples
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _spread_indices(rows: list[int], maximum: int) -> list[int]:
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    if maximum == 1:
        return [rows[len(rows) // 2]]

    result = []
    used = set()
    for i in range(maximum):
        position = round(i * (len(rows) - 1) / (maximum - 1))
        value = int(rows[position])
        if value not in used:
            used.add(value)
            result.append(value)
    if len(result) < maximum:
        for value in rows:
            value = int(value)
            if value not in used:
                used.add(value)
                result.append(value)
                if len(result) >= maximum:
                    break
    return result


def _attach_mining_lookups(profile: dict) -> dict:
    for sample_row in profile["samples"].values():
        sample_row["_by_index"] = {
            int(row["manifest_index"]): row
            for row in sample_row["rows"]
        }
    return profile


def _public_mining_profile(profile: dict) -> dict:
    result = {key: value for key, value in profile.items() if key != "_runtime"}
    result["samples"] = {}
    for sample, row in profile["samples"].items():
        result["samples"][sample] = {
            key: value
            for key, value in row.items()
            if key != "_by_index"
        }
    return result


def mine_productive_component_crops(
    *,
    model,
    criterion,
    support,
    source_batches,
    splits,
    samples,
    checkpoint: Path,
    spacing_zyx,
    crop_shape,
    amp_dtype: str,
    confidence_ignore_margin_um: float,
    partial_ignore_margin_um: float,
    max_state_merges: int,
    merge_threshold: float,
    require_model_accepted: bool,
    mining_max_crops_per_sample: int,
    seed: int,
    cache_root: Path,
) -> dict:
    import torch

    fingerprint = _mining_fingerprint(
        checkpoint=checkpoint,
        samples=samples,
        spacing_zyx=spacing_zyx,
        crop_shape=crop_shape,
        splits=splits,
        confidence_ignore_margin_um=confidence_ignore_margin_um,
        partial_ignore_margin_um=partial_ignore_margin_um,
        max_state_merges=max_state_merges,
        require_model_accepted=require_model_accepted,
        mining_max_crops_per_sample=mining_max_crops_per_sample,
        seed=seed,
    )
    cache_path = cache_root / f"{fingerprint}.json"

    if cache_path.is_file():
        print(f"[mining] cache HIT: {cache_path}", flush=True)
        return _attach_mining_lookups(
            json.loads(cache_path.read_text(encoding="utf-8"))
        )

    print(
        f"[mining] cache MISS: profiling component-state crops -> {cache_path}",
        flush=True,
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    morphology = model.rag_builder.morphology_builder
    if morphology is None:
        raise RuntimeError("Morphology builder is required for component mining")
    morphology.eval()

    profile = {
        "version": MINING_VERSION,
        "fingerprint": fingerprint,
        "checkpoint": str(checkpoint),
        "max_state_merges": int(max_state_merges),
        "require_model_accepted": bool(require_model_accepted),
        "merge_threshold": float(merge_threshold),
        "samples": {},
    }

    total_candidate = 0
    total_productive = 0
    total_hard = 0
    total_focus_positive = 0
    total_focus_negative = 0
    total_hard_negative = 0
    mining_started = time.perf_counter()

    try:
        for sample_ordinal, sample in enumerate(samples):
            split_row = splits[sample]
            candidate_indices = _spread_indices(
                [int(v) for v in split_row["train_indices"]],
                int(mining_max_crops_per_sample),
            )
            rows = []
            productive_indices = []
            hard_indices = []
            state_capable_indices = []

            bar = tqdm(
                total=len(candidate_indices),
                desc=f"Mine {sample}",
                unit="crop",
                dynamic_ncols=True,
                leave=False,
                colour="green",
                file=sys.stdout,
            )

            for manifest_index in candidate_indices:
                record = split_row["records"][manifest_index]
                crop_cpu, _ = support._materialize_crop(
                    source_batches[sample],
                    record,
                    partial_ignore_margin_um=partial_ignore_margin_um,
                )
                crop = support._move_crop_to_cuda(crop_cpu)

                with torch.no_grad(), support._autocast_context(amp_dtype):
                    geometry_forward = model(
                        crop["spatial_inputs"],
                        crop["spacing_um"],
                        crop["dref_um"],
                        spatial_padding_mask=crop.get("spatial_padding_mask"),
                        execution_stage="geometry",
                    )
                    atomic_output = model(
                        crop["spatial_inputs"],
                        crop["spacing_um"],
                        crop["dref_um"],
                        spatial_padding_mask=crop.get("spatial_padding_mask"),
                        execution_stage="spatial",
                        precomputed_geometry=geometry_forward,
                    )

                atomic_targets = criterion.build_targets(
                    atomic_output.rag,
                    crop["gt_labels"],
                    valid_mask=crop.get("supervision_valid_mask"),
                )
                atomic_probability = (
                    atomic_output.rag.spatial_edge_logits
                    .detach()
                    .float()
                    .sigmoid()
                    .cpu()
                    .tolist()
                )

                depth_rows = []
                productive_depths = []
                hard_depths = []
                crop_focus_positive = 0
                crop_focus_negative = 0
                crop_hard_negative = 0

                for depth in range(1, int(max_state_merges) + 1):
                    state_rng = random.Random(
                        int(seed)
                        + 1000003 * sample_ordinal
                        + 1009 * int(manifest_index)
                        + 97 * depth
                    )
                    state = build_component_state(
                        atomic_output.rag,
                        atomic_targets,
                        probabilities=atomic_probability,
                        target_merges=depth,
                        merge_threshold=merge_threshold,
                        rng=state_rng,
                        require_model_accepted=require_model_accepted,
                    )
                    if state is None:
                        continue

                    with torch.no_grad(), support._autocast_context(amp_dtype):
                        state_rag, _ = model._rag_from_supervoxels(
                            [state["labels"]],
                            geometry_forward.geometry,
                            geometry_forward.decoded_spatial,
                            crop["spatial_inputs"],
                            crop["spacing_um"],
                            crop["dref_um"],
                            profile_prefix="inv23_mining_state",
                        )

                    state_targets = criterion.build_targets(
                        state_rag,
                        crop["gt_labels"],
                        valid_mask=crop.get("supervision_valid_mask"),
                    )
                    focus = focus_mask_for_state(
                        state_rag,
                        state_targets,
                        state["merged_components"],
                    )
                    positive = focus & (state_targets.target > 0.5)
                    negative = focus & (state_targets.target <= 0.5)
                    probability = state_rag.spatial_edge_logits.detach().float().sigmoid()
                    hard_negative = negative & (
                        probability >= float(merge_threshold)
                    )

                    positive_count = int(positive.sum().item())
                    negative_count = int(negative.sum().item())
                    hard_negative_count = int(hard_negative.sum().item())
                    focus_count = int(focus.sum().item())

                    if focus_count > 0:
                        state_capable_indices.append(int(manifest_index))
                    if negative_count > 0:
                        productive_depths.append(int(depth))
                    if hard_negative_count > 0:
                        hard_depths.append(int(depth))

                    crop_focus_positive += positive_count
                    crop_focus_negative += negative_count
                    crop_hard_negative += hard_negative_count

                    depth_rows.append(
                        {
                            "depth": int(depth),
                            "actual_state_depth": int(len(state["selected_edge_rows"])),
                            "focus_edge_count": focus_count,
                            "focus_positive_count": positive_count,
                            "focus_negative_count": negative_count,
                            "hard_negative_count": hard_negative_count,
                            "selected_merge_probabilities": [
                                float(v) for v in state["selected_probabilities"]
                            ],
                        }
                    )

                    del state_rag, state_targets, focus
                    del positive, negative, probability, hard_negative

                productive_depths = sorted(set(productive_depths))
                hard_depths = sorted(set(hard_depths))
                if productive_depths:
                    productive_indices.append(int(manifest_index))
                if hard_depths:
                    hard_indices.append(int(manifest_index))

                rows.append(
                    {
                        "manifest_index": int(manifest_index),
                        "productive": bool(productive_depths),
                        "hard": bool(hard_depths),
                        "productive_depths": productive_depths,
                        "hard_depths": hard_depths,
                        "focus_positive_count": int(crop_focus_positive),
                        "focus_negative_count": int(crop_focus_negative),
                        "hard_negative_count": int(crop_hard_negative),
                        "depths": depth_rows,
                    }
                )

                del crop, crop_cpu, atomic_output, geometry_forward, atomic_targets
                torch.cuda.empty_cache()
                gc.collect()
                bar.update(1)

            bar.close()

            productive_indices = sorted(set(productive_indices))
            hard_indices = sorted(set(hard_indices))
            state_capable_indices = sorted(set(state_capable_indices))
            productive_set = set(productive_indices)
            candidate_set = set(candidate_indices)
            ordinary_indices = [
                int(v)
                for v in split_row["train_indices"]
                if int(v) not in productive_set
            ]

            sample_summary = {
                "candidate_indices": candidate_indices,
                "candidate_count": len(candidate_indices),
                "full_train_count": len(split_row["train_indices"]),
                "state_capable_indices": state_capable_indices,
                "state_capable_count": len(state_capable_indices),
                "productive_indices": productive_indices,
                "productive_count": len(productive_indices),
                "hard_indices": hard_indices,
                "hard_count": len(hard_indices),
                "ordinary_indices": ordinary_indices,
                "ordinary_count": len(ordinary_indices),
                "focus_positive_count": int(
                    sum(row["focus_positive_count"] for row in rows)
                ),
                "focus_negative_count": int(
                    sum(row["focus_negative_count"] for row in rows)
                ),
                "hard_negative_count": int(
                    sum(row["hard_negative_count"] for row in rows)
                ),
                "unmined_train_indices": [
                    int(v)
                    for v in split_row["train_indices"]
                    if int(v) not in candidate_set
                ],
                "rows": rows,
            }
            profile["samples"][sample] = sample_summary

            total_candidate += sample_summary["candidate_count"]
            total_productive += sample_summary["productive_count"]
            total_hard += sample_summary["hard_count"]
            total_focus_positive += sample_summary["focus_positive_count"]
            total_focus_negative += sample_summary["focus_negative_count"]
            total_hard_negative += sample_summary["hard_negative_count"]

            print(
                f"[mining] {sample}: "
                f"candidate={sample_summary['candidate_count']} "
                f"productive={sample_summary['productive_count']} "
                f"hard={sample_summary['hard_count']} "
                f"focus P/N={sample_summary['focus_positive_count']}/"
                f"{sample_summary['focus_negative_count']} "
                f"hard-neg={sample_summary['hard_negative_count']}",
                flush=True,
            )
    finally:
        model.eval()
        morphology.train()

    profile["summary"] = {
        "candidate_crop_count": int(total_candidate),
        "productive_crop_count": int(total_productive),
        "hard_crop_count": int(total_hard),
        "focus_positive_count": int(total_focus_positive),
        "focus_negative_count": int(total_focus_negative),
        "hard_negative_count": int(total_hard_negative),
        "productive_fraction": safe_ratio(total_productive, total_candidate),
        "hard_fraction": safe_ratio(total_hard, total_candidate),
        "elapsed_seconds": float(time.perf_counter() - mining_started),
    }

    if total_productive == 0:
        raise RuntimeError(
            "Component-state mining found zero productive training crops. "
            "Do not start the long run; inspect state construction first."
        )

    atomic_json(cache_path, _public_mining_profile(profile))
    print(
        "[mining] COMPLETE: "
        f"productive={total_productive}/{total_candidate} "
        f"hard={total_hard}/{total_candidate} "
        f"focus P/N={total_focus_positive}/{total_focus_negative} "
        f"hard-neg={total_hard_negative} "
        f"time={profile['summary']['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return _attach_mining_lookups(profile)


def choose_training_crop(
    *,
    support,
    splits,
    mining_profile: dict,
    samples,
    attempt: int,
    optimizer_step: int,
    sample_steps: dict,
    productive_crop_fraction: float,
    hard_component_crop_fraction: float,
    max_state_merges: int,
    seed: int,
):
    rng = random.Random(
        int(seed) * 1000003
        + int(attempt) * 9176
        + int(optimizer_step) * 101
    )

    base_sample = samples[attempt % len(samples)]
    want_productive = rng.random() < float(productive_crop_fraction)

    if want_productive:
        sample_order = [
            base_sample,
            *[sample for sample in samples if sample != base_sample],
        ]
        sample = next(
            (
                candidate
                for candidate in sample_order
                if mining_profile["samples"][candidate]["productive_indices"]
            ),
            None,
        )
        if sample is not None:
            sample_row = mining_profile["samples"][sample]
            want_hard = (
                bool(sample_row["hard_indices"])
                and rng.random() < float(hard_component_crop_fraction)
            )
            pool_name = "hard" if want_hard else "productive"
            pool = (
                sample_row["hard_indices"]
                if want_hard
                else sample_row["productive_indices"]
            )
            ordinal = sample_steps[sample][pool_name]
            sample_steps[sample][pool_name] += 1
            manifest_index = int(pool[ordinal % len(pool)])

            mined_row = sample_row["_by_index"][manifest_index]
            depths = (
                mined_row["hard_depths"]
                if want_hard and mined_row["hard_depths"]
                else mined_row["productive_depths"]
            )
            if not depths:
                raise RuntimeError(
                    f"Mined productive crop {sample}/{manifest_index} "
                    "has no productive depth."
                )
            target_depth = int(depths[ordinal % len(depths)])
            return {
                "sample": sample,
                "manifest_index": manifest_index,
                "provenance": f"mined_{pool_name}",
                "sampling_pool": pool_name,
                "target_depth": target_depth,
            }

    sample = base_sample
    ordinal = sample_steps[sample]["ordinary"]
    sample_steps[sample]["ordinary"] += 1
    split_row = splits[sample]
    manifest_index, provenance = support._training_manifest_index(
        split_row,
        sample_local_step=ordinal,
    )
    target_depth = 1 + (
        (optimizer_step + attempt + seed) % max_state_merges
    )
    return {
        "sample": sample,
        "manifest_index": int(manifest_index),
        "provenance": provenance,
        "sampling_pool": "ordinary",
        "target_depth": int(target_depth),
    }



def configure_trainability(model, scope: str):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    morphology = model.rag_builder.morphology_builder
    if morphology is None:
        raise RuntimeError("Morphology must be enabled")

    if scope == "all_morphology":
        for parameter in morphology.parameters():
            parameter.requires_grad_(True)
        for module in (model.rag_network.node_morphology_projection, model.rag_network.edge_morphology_projection):
            if module is None:
                raise RuntimeError("Morphology projection missing")
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif scope == "edge_only":
        for module in (morphology.edge_encoder, morphology.edge_scale_fusion, model.rag_network.edge_morphology_projection):
            if module is None:
                raise RuntimeError("Edge morphology module missing")
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    else:
        raise ValueError(scope)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    model.eval()
    # Keep trainable morphology modules in train mode; legacy RAG stays eval/deterministic.
    morphology.train()
    return trainable


def zero_accumulator() -> dict[str, float]:
    return {
        "edges": 0.0, "positive": 0.0, "negative": 0.0, "bce_sum": 0.0,
        "false_merge": 0.0, "positive_accept": 0.0, "prob_sum": 0.0,
    }


def add_metrics(acc: dict[str, float], logits, targets, mask, *, merge_threshold: float) -> None:
    import torch.nn.functional as F
    p = logits.detach().float().sigmoid()
    mask = mask.bool()
    positive = mask & (targets.target > 0.5)
    negative = mask & (targets.target <= 0.5)
    if bool(mask.any()):
        acc["bce_sum"] += float(F.binary_cross_entropy_with_logits(logits[mask].float(), targets.target[mask].float(), reduction="sum").cpu())
        acc["prob_sum"] += float(p[mask].sum().cpu())
    acc["edges"] += float(mask.sum().item())
    acc["positive"] += float(positive.sum().item())
    acc["negative"] += float(negative.sum().item())
    acc["false_merge"] += float((negative & (p >= merge_threshold)).sum().item())
    acc["positive_accept"] += float((positive & (p >= merge_threshold)).sum().item())


def finalize_metrics(acc: dict[str, float]) -> dict[str, Any]:
    return {
        "edge_count": int(acc["edges"]),
        "positive_edge_count": int(acc["positive"]),
        "negative_edge_count": int(acc["negative"]),
        "bce": safe_ratio(acc["bce_sum"], acc["edges"]),
        "false_merge_count": int(acc["false_merge"]),
        "false_merge_rate": safe_ratio(acc["false_merge"], acc["negative"]),
        "positive_accept_count": int(acc["positive_accept"]),
        "positive_accept_rate": safe_ratio(acc["positive_accept"], acc["positive"]),
        "mean_probability": safe_ratio(acc["prob_sum"], acc["edges"]),
    }


def evaluate(model, model_cfg, support, source_batches, splits, *, amp_dtype: str, partial_ignore_margin_um: float, merge_threshold: float, max_state_merges: int, require_model_accepted: bool, seed: int):
    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion

    criterion = RAGCriterion(model_cfg.partition)
    atomic_acc = zero_accumulator()
    focus_acc = zero_accumulator()
    state_count = 0
    model.eval()
    model.rag_builder.morphology_builder.eval()

    for sample_ordinal, (sample, source_batch) in enumerate(source_batches.items()):
        for crop_ordinal, manifest_index in enumerate(splits[sample]["validation_indices"]):
            record = splits[sample]["records"][manifest_index]
            crop_cpu, _ = support._materialize_crop(source_batch, record, partial_ignore_margin_um=partial_ignore_margin_um)
            crop = support._move_crop_to_cuda(crop_cpu)
            with torch.no_grad(), support._autocast_context(amp_dtype):
                geometry_forward = model(crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"], spatial_padding_mask=crop.get("spatial_padding_mask"), execution_stage="geometry")
                atomic_output = model(crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"], spatial_padding_mask=crop.get("spatial_padding_mask"), execution_stage="spatial", precomputed_geometry=geometry_forward)
            atomic_targets = criterion.build_targets(atomic_output.rag, crop["gt_labels"], valid_mask=crop.get("supervision_valid_mask"))
            add_metrics(atomic_acc, atomic_output.rag.spatial_edge_logits, atomic_targets, atomic_targets.valid, merge_threshold=merge_threshold)

            rng = random.Random(seed + 1009 * sample_ordinal + 97 * crop_ordinal + int(manifest_index))
            p = atomic_output.rag.spatial_edge_logits.detach().float().sigmoid().cpu().tolist()
            state = build_component_state(atomic_output.rag, atomic_targets, probabilities=p, target_merges=max_state_merges, merge_threshold=merge_threshold, rng=rng, require_model_accepted=require_model_accepted)
            if state is not None:
                with torch.no_grad(), support._autocast_context(amp_dtype):
                    state_rag, _ = model._rag_from_supervoxels([state["labels"]], geometry_forward.geometry, geometry_forward.decoded_spatial, crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"], profile_prefix="inv23_validation_state")
                state_targets = criterion.build_targets(state_rag, crop["gt_labels"], valid_mask=crop.get("supervision_valid_mask"))
                focus = focus_mask_for_state(state_rag, state_targets, state["merged_components"])
                add_metrics(focus_acc, state_rag.spatial_edge_logits, state_targets, focus, merge_threshold=merge_threshold)
                state_count += 1

            del crop, crop_cpu, atomic_output, geometry_forward, atomic_targets
            torch.cuda.empty_cache()
            gc.collect()

    model.eval()
    model.rag_builder.morphology_builder.train()
    return {"atomic": finalize_metrics(atomic_acc), "component_focus": finalize_metrics(focus_acc), "component_state_crop_count": state_count}


def validation_rank(metrics: dict, baseline: dict, *, allowed_atomic_positive_drop: float):
    current_pa = metrics["atomic"]["positive_accept_rate"]
    baseline_pa = baseline["atomic"]["positive_accept_rate"]
    violation = max(0.0, baseline_pa - allowed_atomic_positive_drop - current_pa)
    return (
        int(violation > 0),
        float(violation),
        float(metrics["component_focus"]["false_merge_rate"]),
        float(metrics["component_focus"]["bce"]),
        float(metrics["atomic"]["false_merge_rate"]),
        -float(metrics["atomic"]["positive_accept_rate"]),
    )


def make_scaler(enabled: bool):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def save_training_checkpoint(path: Path, *, model, model_cfg, optimizer, scaler, step: int, run_dir: Path, starting_checkpoint: Path, baseline_validation: dict, best_metrics: dict | None, best_rank, args) -> None:
    from learned.stirnet.training.checkpoint import save_checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_checkpoint(
            tmp,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            model_config=model_cfg,
            training_config={"experiment": EXPERIMENT_NAME, "legacy_rag_frozen": True},
            extra={
                "experiment": EXPERIMENT_NAME,
                "run_dir": str(run_dir),
                "starting_checkpoint": str(starting_checkpoint),
                "baseline_validation": baseline_validation,
                "best_metrics": best_metrics,
                "best_rank": None if best_rank is None else list(best_rank),
                "max_state_merges": int(args.max_state_merges),
                "context_loss_weight": float(args.context_loss_weight),
                "train_scope": args.train_scope,
            },
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigation 23: morphology RAG training on GT-safe merged component states")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--spacing-xyz", default=DEFAULT_SPACING_XYZ)
    parser.add_argument("--crop-shape-zyx", default="32,192,192")
    parser.add_argument("--validation-crops-per-sample", type=int, default=6)
    parser.add_argument("--confidence-ignore-margin-um", type=float, default=1.0)
    parser.add_argument("--partial-ignore-margin-um", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-state-merges", type=int, default=3)
    parser.add_argument(
        "--productive-crop-fraction",
        type=float,
        default=0.75,
        help=(
            "Fraction of optimizer steps deliberately drawn from mined crops "
            "that produce GT-different neighbours around a merged component."
        ),
    )
    parser.add_argument(
        "--hard-component-crop-fraction",
        type=float,
        default=0.25,
        help=(
            "Within productive draws, fraction preferentially drawn from crops "
            "with a component-state false merge at the starting checkpoint."
        ),
    )
    parser.add_argument(
        "--mining-max-crops-per-sample",
        type=int,
        default=0,
        help=(
            "Cap one-time pre-mining crops per sample; 0 mines the full train "
            "split. Mining is cached by checkpoint/settings."
        ),
    )
    parser.add_argument("--require-model-accepted-state-merges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-focus-edges-per-class", type=int, default=32)
    parser.add_argument("--max-context-edges-per-class", type=int, default=16)
    parser.add_argument("--context-loss-weight", type=float, default=0.25)
    parser.add_argument("--allowed-atomic-positive-drop", type=float, default=0.02)
    parser.add_argument("--train-scope", choices=("all_morphology", "edge_only"), default="all_morphology")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17023)
    parser.add_argument("--run-name", default="drosophila_12_component_state_v1")
    args = parser.parse_args()

    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion

    if not torch.cuda.is_available():
        raise RuntimeError("Investigation 23 currently requires CUDA")
    if args.max_state_merges < 1:
        raise ValueError("--max-state-merges must be >=1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >=1")
    if not 0.0 <= args.productive_crop_fraction <= 1.0:
        raise ValueError("--productive-crop-fraction must be in [0,1]")
    if not 0.0 <= args.hard_component_crop_fraction <= 1.0:
        raise ValueError("--hard-component-crop-fraction must be in [0,1]")
    if args.mining_max_crops_per_sample < 0:
        raise ValueError("--mining-max-crops-per-sample must be >=0")

    seed_everything(args.seed)
    support = load_inv17_support()
    checkpoint = resolve_checkpoint(args.checkpoint)
    checkpoint_payload = torch_load(checkpoint, map_location="cpu")
    samples = parse_samples(args.samples)
    xyz = parse_triplet(args.spacing_xyz, cast=float)
    spacing_zyx = (float(xyz[2]), float(xyz[1]), float(xyz[0]))
    crop_shape = tuple(int(v) for v in parse_triplet(args.crop_shape_zyx, cast=int))
    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    model, model_cfg, _, transfer = support._build_morphology_model(checkpoint, device="cuda")
    trainable = configure_trainability(model, args.train_scope)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = make_scaler(amp_dtype == "fp16")
    criterion = RAGCriterion(model_cfg.partition)
    merge_threshold = float(model_cfg.partition.spatial_merge_threshold)

    start_step = 0
    resume_extra = {}
    if args.resume:
        if checkpoint_payload.get("extra", {}).get("experiment") != EXPERIMENT_NAME:
            raise ValueError("--resume requires an Investigation-23 checkpoint")
        if "optimizer" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer"])
        if "scaler" in checkpoint_payload:
            scaler.load_state_dict(checkpoint_payload["scaler"])
        start_step = int(checkpoint_payload.get("global_step", 0))
        resume_extra = dict(checkpoint_payload.get("extra", {}))

    nis3d_root = support._discover_nis3d_root(
        samples,
        data_dir=args.data_dir,
        execution_mode="local",
    )

    # Reuse the exact source cache already populated by Investigation 17.
    # The cache is dominated by a full-volume int32 current-label map, so
    # duplicating it in every investigation wastes several GiB.
    cache_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "17_morphology_rag_multicrop_training"
        / "cache"
    )
    print(f"[cache] shared NIS3D source cache: {cache_root}", flush=True)

    source_batches = {}
    source_reports = {}
    for sample in samples:
        cache_namespace = support._data_signature(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_zyx,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        )
        expected_cache = (
            cache_root
            / "source"
            / cache_namespace
            / f"{sample}.pt"
        )
        print(
            f"[cache] {sample}: "
            f"{'HIT' if expected_cache.is_file() else 'MISS'} "
            f"{expected_cache}",
            flush=True,
        )

        batch, report = support._prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_zyx,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
            cache_root=cache_root,
            cache_namespace=cache_namespace,
        )
        source_batches[sample] = batch
        source_reports[sample] = report

    splits = support._build_split(source_batches, crop_shape_zyx=crop_shape, validation_crops_per_sample=args.validation_crops_per_sample)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "stirnet" / "investigations" / EXPERIMENT_NAME / "attempts" / f"{timestamp}_{args.run_name}"
    recovery_dir = ROOT / "runs" / "stirnet" / "investigations" / EXPERIMENT_NAME / "recovery" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)

    mining_cache_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / EXPERIMENT_NAME
        / "mining_cache"
    )
    mining_profile = mine_productive_component_crops(
        model=model,
        criterion=criterion,
        support=support,
        source_batches=source_batches,
        splits=splits,
        samples=samples,
        checkpoint=checkpoint,
        spacing_zyx=spacing_zyx,
        crop_shape=crop_shape,
        amp_dtype=amp_dtype,
        confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        partial_ignore_margin_um=args.partial_ignore_margin_um,
        max_state_merges=args.max_state_merges,
        merge_threshold=merge_threshold,
        require_model_accepted=args.require_model_accepted_state_merges,
        mining_max_crops_per_sample=args.mining_max_crops_per_sample,
        seed=args.seed,
        cache_root=mining_cache_root,
    )
    atomic_json(
        run_dir / "component_state_mining.json",
        _public_mining_profile(mining_profile),
    )

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "run_name": args.run_name,
        "checkpoint": str(checkpoint),
        "resume": bool(args.resume),
        "start_step": int(start_step),
        "max_steps": int(args.max_steps),
        "samples": list(samples),
        "spacing_zyx_um": list(spacing_zyx),
        "crop_shape_zyx": list(crop_shape),
        "merge_threshold": merge_threshold,
        "max_state_merges": int(args.max_state_merges),
        "require_model_accepted_state_merges": bool(args.require_model_accepted_state_merges),
        "productive_crop_fraction": float(args.productive_crop_fraction),
        "hard_component_crop_fraction": float(args.hard_component_crop_fraction),
        "mining_max_crops_per_sample": int(args.mining_max_crops_per_sample),
        "component_state_mining": _public_mining_profile(mining_profile)["summary"],
        "context_loss_weight": float(args.context_loss_weight),
        "train_scope": args.train_scope,
        "amp_dtype": amp_dtype,
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "transfer": transfer,
        "source_reports": source_reports,
    }
    atomic_json(run_dir / "manifest.json", manifest)

    print("=" * 112)
    print("STIR-Net Investigation 23 — train on component states")
    print("=" * 112)
    print("checkpoint       :", checkpoint)
    print("start step       :", start_step)
    print("target step      :", args.max_steps)
    print("samples          :", samples)
    print("max state merges :", args.max_state_merges)
    print(
        "productive mix   :",
        f"{args.productive_crop_fraction:.2f} "
        f"(hard within productive={args.hard_component_crop_fraction:.2f})",
    )
    print(
        "mined crops      :",
        f"productive={mining_profile['summary']['productive_crop_count']} "
        f"hard={mining_profile['summary']['hard_crop_count']} "
        f"candidate={mining_profile['summary']['candidate_crop_count']}",
    )
    print(
        "mined focus P/N  :",
        f"{mining_profile['summary']['focus_positive_count']}/"
        f"{mining_profile['summary']['focus_negative_count']} "
        f"(hard-neg={mining_profile['summary']['hard_negative_count']})",
    )
    print("train scope      :", args.train_scope)
    print("trainable params :", f"{sum(p.numel() for p in trainable):,}")
    print("merge threshold  :", merge_threshold)
    print("run dir          :", run_dir)
    print("=" * 112, flush=True)

    baseline_validation = resume_extra.get("baseline_validation")
    if baseline_validation is None:
        print("[validation] computing step-0 baseline ...", flush=True)
        baseline_validation = evaluate(
            model, model_cfg, support, source_batches, splits,
            amp_dtype=amp_dtype,
            partial_ignore_margin_um=args.partial_ignore_margin_um,
            merge_threshold=merge_threshold,
            max_state_merges=args.max_state_merges,
            require_model_accepted=args.require_model_accepted_state_merges,
            seed=args.seed + 50000,
        )
        atomic_json(run_dir / "baseline_validation.json", baseline_validation)
        print(json.dumps(baseline_validation, indent=2), flush=True)

    best_metrics = resume_extra.get("best_metrics")
    best_rank = tuple(resume_extra["best_rank"]) if resume_extra.get("best_rank") is not None else None

    sample_steps = {
        sample: {"ordinary": 0, "productive": 0, "hard": 0}
        for sample in samples
    }
    sampling_counts = {"ordinary": 0, "productive": 0, "hard": 0}
    optimizer_step = int(start_step)
    attempt = 0
    skipped = 0
    started = time.perf_counter()
    progress = tqdm(total=args.max_steps, initial=optimizer_step, desc="Component-state RAG", unit="step", dynamic_ncols=True, smoothing=0.10, mininterval=0.5, leave=True, colour="green", file=sys.stdout)

    while optimizer_step < args.max_steps:
        choice = choose_training_crop(
            support=support,
            splits=splits,
            mining_profile=mining_profile,
            samples=samples,
            attempt=attempt,
            optimizer_step=optimizer_step,
            sample_steps=sample_steps,
            productive_crop_fraction=args.productive_crop_fraction,
            hard_component_crop_fraction=args.hard_component_crop_fraction,
            max_state_merges=args.max_state_merges,
            seed=args.seed,
        )
        attempt += 1
        sample = choice["sample"]
        manifest_index = int(choice["manifest_index"])
        provenance = choice["provenance"]
        sampling_pool = choice["sampling_pool"]
        target_depth = int(choice["target_depth"])
        split_row = splits[sample]
        record = split_row["records"][manifest_index]

        crop_cpu, _ = support._materialize_crop(source_batches[sample], record, partial_ignore_margin_um=args.partial_ignore_margin_um)
        crop = support._move_crop_to_cuda(crop_cpu)

        # Frozen dense geometry and no-grad atomic state used only to choose safe contractions.
        with torch.no_grad(), support._autocast_context(amp_dtype):
            geometry_forward = model(crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"], spatial_padding_mask=crop.get("spatial_padding_mask"), execution_stage="geometry")
            atomic_output = model(crop["spatial_inputs"], crop["spacing_um"], crop["dref_um"], spatial_padding_mask=crop.get("spatial_padding_mask"), execution_stage="spatial", precomputed_geometry=geometry_forward)
        atomic_targets = criterion.build_targets(atomic_output.rag, crop["gt_labels"], valid_mask=crop.get("supervision_valid_mask"))
        atomic_p = atomic_output.rag.spatial_edge_logits.detach().float().sigmoid().cpu().tolist()
        state_rng = random.Random(args.seed * 1000003 + optimizer_step * 97 + attempt)
        state = build_component_state(
            atomic_output.rag,
            atomic_targets,
            probabilities=atomic_p,
            target_merges=target_depth,
            merge_threshold=merge_threshold,
            rng=state_rng,
            require_model_accepted=args.require_model_accepted_state_merges,
        )
        if state is None:
            skipped += 1
            del crop, crop_cpu, atomic_output, geometry_forward, atomic_targets
            torch.cuda.empty_cache(); gc.collect()
            if skipped > max(1000, args.max_steps * 20):
                raise RuntimeError("Too many crops could not form a GT-safe component state")
            continue

        optimizer.zero_grad(set_to_none=True)
        with support._autocast_context(amp_dtype):
            state_rag, _ = model._rag_from_supervoxels(
                [state["labels"]],
                geometry_forward.geometry,
                geometry_forward.decoded_spatial,
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                profile_prefix="inv23_train_state",
            )
            state_targets = criterion.build_targets(state_rag, crop["gt_labels"], valid_mask=crop.get("supervision_valid_mask"))
            focus = focus_mask_for_state(state_rag, state_targets, state["merged_components"])
            context = state_targets.valid.bool() & ~focus
            focus_loss, focus_counts = balanced_bce(state_rag.spatial_edge_logits, state_targets, focus, max_per_class=args.max_focus_edges_per_class, hard_negative_first=True)
            context_loss, context_counts = balanced_bce(state_rag.spatial_edge_logits, state_targets, context, max_per_class=args.max_context_edges_per_class, hard_negative_first=False)
            loss = focus_loss + float(args.context_loss_weight) * context_loss

        if focus_counts["total"] == 0:
            skipped += 1
            del crop, crop_cpu, atomic_output, geometry_forward, atomic_targets, state_rag, state_targets
            torch.cuda.empty_cache(); gc.collect()
            continue

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer_step += 1
        sampling_counts[sampling_pool] += 1
        progress.update(1)
        progress.set_postfix({
            "loss": f"{float(loss.detach().float().cpu()):.4f}",
            "focus": focus_counts["total"],
            "ctx": context_counts["total"],
            "depth": len(state["selected_edge_rows"]),
            "pool": sampling_pool[0].upper(),
            "skip": skipped,
        })

        append_jsonl(
            run_dir / "training.jsonl",
            {
                "step": optimizer_step,
                "sample": sample,
                "manifest_index": int(manifest_index),
                "provenance": provenance,
                "sampling_pool": sampling_pool,
                "requested_state_depth": int(target_depth),
                "actual_state_depth": len(state["selected_edge_rows"]),
                "selected_merge_probabilities": state["selected_probabilities"],
                "focus_positive": focus_counts["positive"],
                "focus_negative": focus_counts["negative"],
                "context_positive": context_counts["positive"],
                "context_negative": context_counts["negative"],
                "focus_loss": float(focus_loss.detach().float().cpu()),
                "context_loss": float(context_loss.detach().float().cpu()),
                "total_loss": float(loss.detach().float().cpu()),
            },
        )

        should_validate = optimizer_step % args.validation_every == 0 or optimizer_step == args.max_steps
        should_checkpoint = optimizer_step % args.checkpoint_every == 0 or optimizer_step == args.max_steps
        validation = None
        if should_validate:
            validation = evaluate(
                model, model_cfg, support, source_batches, splits,
                amp_dtype=amp_dtype,
                partial_ignore_margin_um=args.partial_ignore_margin_um,
                merge_threshold=merge_threshold,
                max_state_merges=args.max_state_merges,
                require_model_accepted=args.require_model_accepted_state_merges,
                seed=args.seed + 50000,
            )
            rank = validation_rank(validation, baseline_validation, allowed_atomic_positive_drop=args.allowed_atomic_positive_drop)
            append_jsonl(run_dir / "validation.jsonl", {"step": optimizer_step, "rank": list(rank), **validation})
            print(
                f"\n[val {optimizer_step}] atomic FM={validation['atomic']['false_merge_rate']:.5f} "
                f"PA={validation['atomic']['positive_accept_rate']:.5f} | "
                f"component FM={validation['component_focus']['false_merge_rate']:.5f} "
                f"PA={validation['component_focus']['positive_accept_rate']:.5f} "
                f"BCE={validation['component_focus']['bce']:.5f}", flush=True
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_metrics = validation
                save_training_checkpoint(
                    recovery_dir / "best_checkpoint.pt",
                    model=model, model_cfg=model_cfg, optimizer=optimizer, scaler=scaler,
                    step=optimizer_step, run_dir=run_dir, starting_checkpoint=checkpoint,
                    baseline_validation=baseline_validation, best_metrics=best_metrics,
                    best_rank=best_rank, args=args,
                )
                print(f"[best] step {optimizer_step}: {best_rank}", flush=True)

        if should_checkpoint:
            save_training_checkpoint(
                recovery_dir / f"checkpoint_step_{optimizer_step:06d}.pt",
                model=model, model_cfg=model_cfg, optimizer=optimizer, scaler=scaler,
                step=optimizer_step, run_dir=run_dir, starting_checkpoint=checkpoint,
                baseline_validation=baseline_validation, best_metrics=best_metrics,
                best_rank=best_rank, args=args,
            )

        del crop, crop_cpu, atomic_output, geometry_forward, atomic_targets, state_rag, state_targets
        torch.cuda.empty_cache(); gc.collect()

    progress.close()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "run_name": args.run_name,
        "starting_checkpoint": str(checkpoint),
        "final_step": optimizer_step,
        "skipped_attempts": skipped,
        "sampling_counts": sampling_counts,
        "component_state_mining": _public_mining_profile(mining_profile)["summary"],
        "baseline_validation": baseline_validation,
        "best_validation": best_metrics,
        "best_rank": None if best_rank is None else list(best_rank),
        "recovery_dir": str(recovery_dir),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    atomic_json(run_dir / "summary.json", summary)
    print("=" * 112)
    print("Investigation 23 complete")
    print("best checkpoint:", recovery_dir / "best_checkpoint.pt")
    print("summary        :", run_dir / "summary.json")
    print("=" * 112)


if __name__ == "__main__":
    main()
