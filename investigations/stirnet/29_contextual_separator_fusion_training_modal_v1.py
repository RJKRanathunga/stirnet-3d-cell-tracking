from __future__ import annotations

"""
Investigation 29 — contextual separator late-fusion probe on Modal.

Frozen h100 RAG produces:
    contextual edge embedding e_rag
    base merge logit L_base
    exact 12-D separator/contact vector s_sep

Only this experiment-local head is trained:
    delta = Fusion(e_rag, s_sep, L_base)
    L_final = L_base + delta

The final residual layer is zero-initialized, so step 0 exactly reproduces h100.
The residual is signed: it may either lower or raise merge confidence.

Training uses a balanced anti-shortcut sampler:
    25% GT-different + strong separator
    25% GT-same      + strong separator
    25% GT-different + weak separator
    25% GT-same      + weak separator
and balances Drosophila_1 / Drosophila_2 inside each stratum when possible.
Validation uses the natural fixed held-out crop distribution.

First 300 steps:
    modal run investigations/stirnet/29_contextual_separator_fusion_training.py

Continue the SAME run from 300 to 600 later:
    modal run investigations/stirnet/29_contextual_separator_fusion_training.py --target-step 600
"""

import gc
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time
from typing import Any

import modal


# ======================================================================================
# Modal packaging
# ======================================================================================

EXPERIMENT_NAME = "29_contextual_separator_fusion_training"
DEFAULT_RUN_NAME = "drosophila_12_contextual_separator_fusion_h100_v1"

LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_H100_CHECKPOINT = (
    LOCAL_REPO_ROOT
    / "runs/stirnet/investigations/19_morphology_rag_v2_headroom_training"
    / "recovery/drosophila_12_morphology_rag_v2_headroom_h100"
    / "checkpoint_step_000600.pt"
)
if not LOCAL_H100_CHECKPOINT.is_file():
    raise FileNotFoundError(
        "Local h100 checkpoint is required so Modal can package it:\n"
        f"{LOCAL_H100_CHECKPOINT}"
    )

REMOTE_REPO_ROOT = "/workspace/cell-tracking"
REMOTE_INVESTIGATIONS = f"{REMOTE_REPO_ROOT}/investigations/stirnet"
REMOTE_RUNS_ROOT = f"{REMOTE_REPO_ROOT}/runs"
REMOTE_H100 = "/workspace/bootstrap/h100_step000600.pt"
NIS3D_ROOT = "/external/NIS3D"

external_volume = modal.Volume.from_name("external")
runs_volume = modal.Volume.from_name("stirnet-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
        "networkx==3.6.1",
        "tifffile==2026.3.3",
        "imagecodecs==2026.3.6",
        "tqdm==4.69.0",
        "pandas==3.0.5",
        "psutil==7.2.2",
        "cupy-cuda13x[ctk]==14.2.0",
    )
    .workdir(REMOTE_REPO_ROOT)
    .add_local_dir(
        LOCAL_REPO_ROOT / "learned",
        remote_path=f"{REMOTE_REPO_ROOT}/learned",
    )
    .add_local_dir(
        LOCAL_REPO_ROOT / "investigations/stirnet",
        remote_path=REMOTE_INVESTIGATIONS,
    )
    .add_local_file(
        LOCAL_REPO_ROOT / "pyproject.toml",
        remote_path=f"{REMOTE_REPO_ROOT}/pyproject.toml",
    )
    .add_local_file(
        LOCAL_H100_CHECKPOINT,
        remote_path=REMOTE_H100,
    )
)

app = modal.App("stirnet-investigation-29-contextual-separator-fusion")


# ======================================================================================
# Experiment constants
# ======================================================================================

SAMPLES = ("Drosophila_1", "Drosophila_2")
SPACING_XYZ = "0.20312639,0.20312639,0.79099447"
CROP_SHAPE_ZYX = (32, 192, 192)
VALIDATION_CROPS_PER_SAMPLE = 6

MERGE_THRESHOLD = 0.845
STRONG_MEAN = 0.55
STRONG_MAX = 0.85
STRONG_COVERAGE70 = 0.25

BATCH_SIZE = 256
LR = 5e-4
WEIGHT_DECAY = 1e-4
DELTA_L1_WEIGHT = 0.01
MAX_ABS_DELTA = 8.0
VALIDATE_EVERY = 25
CHECKPOINT_EVERY = 25
LOG_EVERY = 10
SEED = 29_001


def _ensure_path() -> None:
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)


def _load_inv26():
    _ensure_path()
    path = Path(REMOTE_INVESTIGATIONS) / "26_separator_aware_rag_barrier_training.py"
    spec = importlib.util.spec_from_file_location("_inv26_for_inv29", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import current Investigation 26: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _torch_load(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(path: Path, payload: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    def conv(v):
        import numpy as np
        import torch
        if v is None or isinstance(v, (str, bool, int)):
            return v
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if isinstance(v, np.generic):
            return conv(v.item())
        if isinstance(v, Path):
            return str(v)
        if torch.is_tensor(v):
            t = v.detach().cpu()
            return conv(t.item()) if t.numel() == 1 else t.tolist()
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return str(v)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(conv(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def _amp_name() -> str:
    import torch
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


# ======================================================================================
# Experiment-local fusion head
# ======================================================================================


def _make_head(context_dim: int, separator_dim: int):
    import torch
    from torch import nn

    class ContextualSeparatorFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_norm = nn.LayerNorm(context_dim)
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, 96), nn.SiLU(),
                nn.Linear(96, 64), nn.SiLU(),
            )
            self.separator_norm = nn.LayerNorm(separator_dim)
            self.separator_encoder = nn.Sequential(
                nn.Linear(separator_dim, 48), nn.SiLU(),
                nn.Linear(48, 32), nn.SiLU(),
            )
            self.base_encoder = nn.Sequential(
                nn.Linear(1, 16), nn.SiLU(),
                nn.Linear(16, 16), nn.SiLU(),
            )
            self.gate = nn.Sequential(
                nn.Linear(64 + 32 + 16, 64), nn.SiLU(),
                nn.Linear(64, 32), nn.Sigmoid(),
            )
            self.context_to_interaction = nn.Sequential(
                nn.Linear(64, 32), nn.SiLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(64 + 32 + 32 + 16, 96), nn.SiLU(),
                nn.Linear(96, 64), nn.SiLU(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)

        def forward(self, context, separator, base_logit):
            c = self.context_encoder(self.context_norm(context.float()))
            s = self.separator_encoder(self.separator_norm(separator.float()))
            b = self.base_encoder(torch.tanh(base_logit.float()[:, None] / 4.0))
            g = self.gate(torch.cat([c, s, b], dim=-1))
            gs = g * s
            interaction = self.context_to_interaction(c) * gs
            raw = self.fusion(torch.cat([c, gs, interaction, b], dim=-1)).squeeze(-1)
            delta = MAX_ABS_DELTA * torch.tanh(raw / MAX_ABS_DELTA)
            return base_logit.float() + delta, delta, g.mean(dim=-1)

    return ContextualSeparatorFusion()


# ======================================================================================
# Frozen feature cache
# ======================================================================================


def _strong(features):
    return (
        (features[:, 0] >= STRONG_MEAN)
        | ((features[:, 1] >= STRONG_MAX) & (features[:, 4] >= STRONG_COVERAGE70))
    )


def _new_bucket():
    return {k: [] for k in (
        "context", "separator", "base_logit", "target", "strong", "sample_id", "manifest_index"
    )}


def _append_edges(bucket, result, sample_id: int, manifest_index: int) -> int:
    import torch
    valid = result["targets"].valid.bool()
    if not bool(valid.any()):
        return 0
    context = result["rag"].edge_embeddings[valid].detach().float().cpu()
    separator = result["features"][valid].detach().float().cpu()
    base = result["base_logits"][valid].detach().float().cpu()
    target = result["targets"].target[valid].detach().float().cpu()
    strong = _strong(separator).cpu()
    n = int(target.numel())
    bucket["context"].append(context)
    bucket["separator"].append(separator)
    bucket["base_logit"].append(base)
    bucket["target"].append(target)
    bucket["strong"].append(strong)
    bucket["sample_id"].append(torch.full((n,), sample_id, dtype=torch.long))
    bucket["manifest_index"].append(torch.full((n,), manifest_index, dtype=torch.long))
    return n


def _finish_bucket(bucket):
    import torch
    if not bucket["target"]:
        raise RuntimeError("No valid RAG edges were cached")
    return {k: torch.cat(v, dim=0) for k, v in bucket.items()}


def _stratum_counts(cache):
    counts = {}
    same = cache["target"] >= 0.5
    for sid, sample in enumerate(SAMPLES):
        for relation, rel_mask in (("different", ~same), ("same", same)):
            for sep_name, sep_flag in (("weak", False), ("strong", True)):
                mask = (
                    (cache["sample_id"] == sid)
                    & rel_mask
                    & (cache["strong"] == sep_flag)
                )
                counts[f"{sample}:{relation}:{sep_name}"] = int(mask.sum())
    return counts


def _precompute(inv26, support, model, cfg, feature_cache_path: Path, preprocessing_cache: Path):
    import torch
    from learned.stirnet.model.partition.rag import RAGCriterion
    from tqdm import tqdm

    spacing_override = support._parse_spacing_xyz_override(SPACING_XYZ)
    nis3d_root = support._discover_nis3d_root(
        SAMPLES,
        data_dir=NIS3D_ROOT,
        execution_mode="local",
    )
    criterion = RAGCriterion(cfg.partition).to("cuda")
    criterion.eval()
    amp = _amp_name()

    train_bucket = _new_bucket()
    val_bucket = _new_bucket()
    reports = {}
    splits_report = {}
    start = time.perf_counter()

    for sid, sample in enumerate(SAMPLES):
        print(f"\n[cache] preparing {sample}", flush=True)
        signature = support._data_signature(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_override,
            confidence_ignore_margin_um=1.0,
        )
        source_batch, report = support._prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_override,
            confidence_ignore_margin_um=1.0,
            cache_root=preprocessing_cache,
            cache_namespace=signature,
        )
        reports[sample] = report
        split = support._build_split(
            {sample: source_batch},
            crop_shape_zyx=CROP_SHAPE_ZYX,
            validation_crops_per_sample=VALIDATION_CROPS_PER_SAMPLE,
        )[sample]
        train_indices = [int(x) for x in split["train_indices"]]
        val_indices = [int(x) for x in split["validation_indices"]]
        splits_report[sample] = {
            "manifest": len(split["records"]),
            "train": len(train_indices),
            "validation": len(val_indices),
            "validation_indices": val_indices,
        }
        jobs = [("train", x) for x in train_indices] + [("validation", x) for x in val_indices]
        progress = tqdm(
            total=len(jobs), desc=f"Frozen features {sample}", unit="crop",
            dynamic_ncols=True, colour="green", file=sys.stdout,
        )
        tr_edges = 0
        va_edges = 0
        for split_name, idx in jobs:
            crop_cpu, _ = support._materialize_crop(
                source_batch,
                split["records"][idx],
                partial_ignore_margin_um=1.0,
            )
            crop = support._move_crop_to_cuda(crop_cpu)
            with torch.no_grad():
                result = inv26._forward_crop(
                    model=model,
                    crop=crop,
                    rag_criterion=criterion,
                    amp_dtype=amp,
                    training=False,
                )
            if split_name == "train":
                added = _append_edges(train_bucket, result, sid, idx)
                tr_edges += added
            else:
                added = _append_edges(val_bucket, result, sid, idx)
                va_edges += added
            progress.set_postfix({"split": split_name[:3], "idx": idx, "edges": added})
            progress.update(1)
            del result, crop, crop_cpu
            torch.cuda.empty_cache()
        progress.close()
        print(f"[cache] {sample}: train edges={tr_edges}, validation edges={va_edges}", flush=True)
        del source_batch, split
        gc.collect()
        torch.cuda.empty_cache()

    train = _finish_bucket(train_bucket)
    validation = _finish_bucket(val_bucket)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": REMOTE_H100,
        "samples": list(SAMPLES),
        "spacing_xyz": SPACING_XYZ,
        "crop_shape_zyx": list(CROP_SHAPE_ZYX),
        "train": train,
        "validation": validation,
        "train_stratum_counts": _stratum_counts(train),
        "validation_stratum_counts": _stratum_counts(validation),
        "data_reports": reports,
        "split_reports": splits_report,
        "elapsed_seconds": time.perf_counter() - start,
    }
    _atomic_torch_save(feature_cache_path, payload)
    runs_volume.commit()
    print(f"[cache] persisted {feature_cache_path}", flush=True)
    for key, value in payload["train_stratum_counts"].items():
        print(f"  {key}: {value}", flush=True)
    return payload


# ======================================================================================
# Balanced sampler and evaluation
# ======================================================================================


def _build_pools(cache):
    import torch
    same = cache["target"] >= 0.5
    pools = {}
    fallback = {}
    for same_flag in (False, True):
        for strong_flag in (False, True):
            rel = same if same_flag else ~same
            mask = rel & (cache["strong"] == strong_flag)
            fallback[(same_flag, strong_flag)] = torch.nonzero(mask, as_tuple=False).flatten()
            if fallback[(same_flag, strong_flag)].numel() == 0:
                raise RuntimeError(f"Empty required stratum same={same_flag}, strong={strong_flag}")
            for sid in range(len(SAMPLES)):
                local = mask & (cache["sample_id"] == sid)
                pools[(sid, same_flag, strong_flag)] = torch.nonzero(local, as_tuple=False).flatten()
                if pools[(sid, same_flag, strong_flag)].numel() == 0:
                    print(
                        f"[sampler] empty {SAMPLES[sid]} same={same_flag} strong={strong_flag}; "
                        "using cross-dataset fallback",
                        flush=True,
                    )
    return pools, fallback


def _sample_indices(pools, fallback, batch_size: int):
    import torch
    groups = len(SAMPLES) * 4
    base = batch_size // groups
    remainder = batch_size % groups
    pieces = []
    g = 0
    for sid in range(len(SAMPLES)):
        for same_flag in (False, True):
            for strong_flag in (False, True):
                count = base + int(g < remainder)
                g += 1
                pool = pools[(sid, same_flag, strong_flag)]
                if pool.numel() == 0:
                    pool = fallback[(same_flag, strong_flag)]
                draw = torch.randint(0, int(pool.numel()), (count,))
                pieces.append(pool[draw])
    idx = torch.cat(pieces)
    return idx[torch.randperm(idx.numel())]


def _evaluate(head, cache, device):
    import torch
    import torch.nn.functional as F

    c = cache["context"].to(device)
    s = cache["separator"].to(device)
    b = cache["base_logit"].to(device)
    y = cache["target"].to(device)
    strong = cache["strong"].to(device)
    sample_id = cache["sample_id"].to(device)
    qlogit = _logit(MERGE_THRESHOLD)

    with torch.no_grad():
        f, delta, gate = head(c, s, b)
        same = y >= 0.5
        different = ~same
        bm = b >= qlogit
        fm = f >= qlogit
        bc = (bm & same) | ((~bm) & different)
        fc = (fm & same) | ((~fm) & different)
        metrics = {
            "edge_count": int(y.numel()),
            "same_count": int(same.sum()),
            "different_count": int(different.sum()),
            "base_bce": float(F.binary_cross_entropy_with_logits(b, y).cpu()),
            "final_bce": float(F.binary_cross_entropy_with_logits(f, y).cpu()),
            "base_false_merge_count": int((bm & different).sum()),
            "final_false_merge_count": int((fm & different).sum()),
            "base_false_split_count": int(((~bm) & same).sum()),
            "final_false_split_count": int(((~fm) & same).sum()),
            "fixed_h100_errors": int(((~bc) & fc).sum()),
            "broken_h100_correct_edges": int((bc & (~fc)).sum()),
            "delta_abs_mean": float(delta.abs().mean().cpu()),
            "delta_mean": float(delta.mean().cpu()),
            "gate_mean": float(gate.mean().cpu()),
        }
        metrics["base_threshold_error_count"] = metrics["base_false_merge_count"] + metrics["base_false_split_count"]
        metrics["final_threshold_error_count"] = metrics["final_false_merge_count"] + metrics["final_false_split_count"]
        metrics["base_same_acceptance"] = float((bm & same).sum()) / max(int(same.sum()), 1)
        metrics["final_same_acceptance"] = float((fm & same).sum()) / max(int(same.sum()), 1)
        metrics["base_different_rejection"] = float(((~bm) & different).sum()) / max(int(different.sum()), 1)
        metrics["final_different_rejection"] = float(((~fm) & different).sum()) / max(int(different.sum()), 1)

        for same_flag, relation in ((False, "different"), (True, "same")):
            rel = same if same_flag else different
            for strong_flag, sep_name in ((False, "weak"), (True, "strong")):
                mask = rel & (strong == strong_flag)
                prefix = f"{relation}_{sep_name}"
                n = int(mask.sum())
                metrics[f"{prefix}_count"] = n
                if n:
                    correct = fm[mask] if same_flag else ~fm[mask]
                    metrics[f"{prefix}_accuracy"] = float(correct.float().mean().cpu())
                    metrics[f"{prefix}_gate_mean"] = float(gate[mask].mean().cpu())
                    metrics[f"{prefix}_delta_mean"] = float(delta[mask].mean().cpu())
                else:
                    metrics[f"{prefix}_accuracy"] = None
                    metrics[f"{prefix}_gate_mean"] = None
                    metrics[f"{prefix}_delta_mean"] = None

        per_sample = {}
        for sid, sample in enumerate(SAMPLES):
            mask = sample_id == sid
            sm = same[mask]
            df = ~sm
            bmm = bm[mask]
            fmm = fm[mask]
            per_sample[sample] = {
                "edge_count": int(mask.sum()),
                "base_false_merge_count": int((bmm & df).sum()),
                "final_false_merge_count": int((fmm & df).sum()),
                "base_false_split_count": int(((~bmm) & sm).sum()),
                "final_false_split_count": int(((~fmm) & sm).sum()),
            }
        metrics["per_sample"] = per_sample

    del c, s, b, y, strong, sample_id
    torch.cuda.empty_cache()
    return metrics


def _best_key(metrics):
    return (
        int(metrics["final_threshold_error_count"]),
        float(metrics["final_bce"]),
        int(metrics["broken_h100_correct_edges"]),
    )


# ======================================================================================
# Modal training function
# ======================================================================================


@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={
        "/external": external_volume,
        REMOTE_RUNS_ROOT: runs_volume,
    },
)
def train(target_step: int = 300, run_name: str = DEFAULT_RUN_NAME) -> dict:
    _ensure_path()
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    if target_step < 1:
        raise ValueError("target_step must be positive")
    _seed_all(SEED)

    inv26 = _load_inv26()
    support = inv26._load_inv17_support()
    props = torch.cuda.get_device_properties(0)

    recovery = (
        Path(REMOTE_RUNS_ROOT)
        / "stirnet/investigations"
        / EXPERIMENT_NAME
        / "recovery"
        / run_name
    )
    recovery.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    attempt = (
        Path(REMOTE_RUNS_ROOT)
        / "stirnet/investigations"
        / EXPERIMENT_NAME
        / "attempts"
        / f"{stamp}_{run_name}_target{target_step:04d}"
    )
    attempt.mkdir(parents=True, exist_ok=False)

    feature_cache_path = recovery / "edge_feature_cache.pt"
    latest_path = recovery / "latest_fusion_checkpoint.pt"
    best_path = recovery / "best_fusion_checkpoint.pt"
    baseline_path = recovery / "baseline_metrics.json"
    history_path = recovery / "history.jsonl"
    validation_path = recovery / "validation.jsonl"

    print("=" * 112, flush=True)
    print("STIR-Net Investigation 29 — contextual separator late fusion", flush=True)
    print("=" * 112, flush=True)
    print(f"GPU             : {props.name} ({props.total_memory / 2**30:.1f} GiB)", flush=True)
    print(f"NIS3D           : {NIS3D_ROOT}", flush=True)
    print(f"Base checkpoint : {REMOTE_H100}", flush=True)
    print(f"Target step     : {target_step}", flush=True)
    print(f"Recovery        : {recovery}", flush=True)
    print("Trainable       : experiment-local fusion head ONLY", flush=True)
    print("Sampler         : 4 strata x 2 datasets when available", flush=True)
    print("=" * 112, flush=True)

    model, cfg, base_payload, transfer_report, _ = inv26._build_model_from_checkpoint(
        Path(REMOTE_H100), device="cuda"
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    preprocessing_cache = (
        Path(REMOTE_RUNS_ROOT)
        / "stirnet/investigations/17_morphology_rag_multicrop_training/cache"
    )
    if feature_cache_path.is_file():
        print(f"[cache] reusing {feature_cache_path}", flush=True)
        features = _torch_load(feature_cache_path)
    else:
        features = _precompute(
            inv26, support, model, cfg,
            feature_cache_path, preprocessing_cache,
        )

    train_cache = features["train"]
    val_cache = features["validation"]
    context_dim = int(train_cache["context"].shape[1])
    separator_dim = int(train_cache["separator"].shape[1])
    if separator_dim != 12:
        raise RuntimeError(f"Expected 12 separator/contact features, got {separator_dim}")

    print(f"[cache] train edges={train_cache['target'].numel()}", flush=True)
    print(f"[cache] validation edges={val_cache['target'].numel()}", flush=True)
    print(f"[cache] context_dim={context_dim}, separator_dim={separator_dim}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    head = _make_head(context_dim, separator_dim).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pools, fallback = _build_pools(train_cache)

    start_step = 0
    best_step = None
    best_metrics = None
    if latest_path.is_file():
        state = _torch_load(latest_path)
        head.load_state_dict(state["head_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = int(state["global_step"])
        best_step = state.get("best_step")
        best_metrics = state.get("best_metrics")
        print(f"[resume] step {start_step}", flush=True)

    # True frozen-h100 baseline is stored once. On later continuation runs we reload it.
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        zero_head = _make_head(context_dim, separator_dim).cuda().eval()
        baseline = _evaluate(zero_head, val_cache, torch.device("cuda"))
        del zero_head
        _atomic_json(baseline_path, baseline)
        runs_volume.commit()

    print(
        f"[baseline] BCE={baseline['base_bce']:.6f} | "
        f"FM={baseline['base_false_merge_count']} | "
        f"FS={baseline['base_false_split_count']} | "
        f"errors={baseline['base_threshold_error_count']}",
        flush=True,
    )

    if start_step >= target_step:
        return {
            "status": "already_complete",
            "global_step": start_step,
            "best_step": best_step,
            "recovery_dir": str(recovery),
            "best_metrics": best_metrics,
        }

    head.train()
    tr_c = train_cache["context"]
    tr_s = train_cache["separator"]
    tr_b = train_cache["base_logit"]
    tr_y = train_cache["target"]

    progress = tqdm(
        total=target_step,
        initial=start_step,
        desc="Contextual fusion",
        unit="step",
        dynamic_ncols=True,
        colour="green",
        file=sys.stdout,
    )
    t0 = time.perf_counter()

    for step in range(start_step + 1, target_step + 1):
        idx = _sample_indices(pools, fallback, BATCH_SIZE)
        c = tr_c[idx].cuda(non_blocking=True)
        s = tr_s[idx].cuda(non_blocking=True)
        b = tr_b[idx].cuda(non_blocking=True)
        y = tr_y[idx].cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        final, delta, gate = head(c, s, b)
        bce = F.binary_cross_entropy_with_logits(final, y.float())
        delta_l1 = delta.abs().mean()
        loss = bce + DELTA_L1_WEIGHT * delta_l1
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0).cpu())
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "delta_l1": float(delta_l1.detach().cpu()),
            "delta_mean": float(delta.detach().mean().cpu()),
            "gate_mean": float(gate.detach().mean().cpu()),
            "gradient_norm": grad_norm,
            "elapsed_seconds": time.perf_counter() - t0,
        }
        _append_jsonl(history_path, row)

        if step == 1 or step % LOG_EVERY == 0:
            progress.set_postfix({
                "loss": f"{row['loss']:.4f}",
                "bce": f"{row['bce']:.4f}",
                "|d|": f"{row['delta_l1']:.3f}",
                "gate": f"{row['gate_mean']:.3f}",
            }, refresh=False)

        if step % VALIDATE_EVERY == 0 or step == target_step:
            head.eval()
            val = _evaluate(head, val_cache, torch.device("cuda"))
            train_audit = _evaluate(head, train_cache, torch.device("cuda"))
            _append_jsonl(validation_path, {
                "step": step,
                "validation": val,
                "train_audit": train_audit,
                "elapsed_seconds": time.perf_counter() - t0,
            })
            print(
                f"\n[val] step={step:04d} | "
                f"BCE {baseline['base_bce']:.5f}->{val['final_bce']:.5f} | "
                f"FM {baseline['base_false_merge_count']}->{val['final_false_merge_count']} | "
                f"FS {baseline['base_false_split_count']}->{val['final_false_split_count']} | "
                f"errors {baseline['base_threshold_error_count']}->{val['final_threshold_error_count']} | "
                f"fixed={val['fixed_h100_errors']} broken={val['broken_h100_correct_edges']}",
                flush=True,
            )
            print(
                "[val strata] "
                f"same+strong={val['same_strong_accuracy']} | "
                f"diff+strong={val['different_strong_accuracy']} | "
                f"same+weak={val['same_weak_accuracy']} | "
                f"diff+weak={val['different_weak_accuracy']}",
                flush=True,
            )
            if best_metrics is None or _best_key(val) < _best_key(best_metrics):
                best_metrics = val
                best_step = step
                _atomic_torch_save(best_path, {
                    "experiment": EXPERIMENT_NAME,
                    "global_step": step,
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_step": best_step,
                    "best_metrics": best_metrics,
                    "baseline_metrics": baseline,
                    "context_dim": context_dim,
                    "separator_dim": separator_dim,
                })
                print(f"[checkpoint] new best step {step}", flush=True)
            head.train()

        if step % CHECKPOINT_EVERY == 0 or step == target_step:
            _atomic_torch_save(latest_path, {
                "experiment": EXPERIMENT_NAME,
                "global_step": step,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_step": best_step,
                "best_metrics": best_metrics,
                "baseline_metrics": baseline,
                "context_dim": context_dim,
                "separator_dim": separator_dim,
            })
            _atomic_json(recovery / "checkpoint_index.json", {
                "latest_step": step,
                "latest_checkpoint": str(latest_path),
                "best_step": best_step,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
            })
            runs_volume.commit()
            print(f"[persisted] step={step} best_step={best_step}", flush=True)

        progress.update(1)
        del c, s, b, y, final, delta, gate, bce, delta_l1, loss

    progress.close()
    head.eval()
    final_metrics = _evaluate(head, val_cache, torch.device("cuda"))
    summary = {
        "status": "success",
        "experiment": EXPERIMENT_NAME,
        "run_name": run_name,
        "global_step": target_step,
        "base_checkpoint_step": base_payload.get("global_step"),
        "transfer_report": transfer_report,
        "baseline_metrics": baseline,
        "final_metrics": final_metrics,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "train_stratum_counts": features.get("train_stratum_counts"),
        "validation_stratum_counts": features.get("validation_stratum_counts"),
        "recovery_dir": str(recovery),
        "attempt_dir": str(attempt),
        "feature_cache": str(feature_cache_path),
        "elapsed_training_seconds": time.perf_counter() - t0,
    }
    _atomic_json(attempt / "summary.json", summary)
    _atomic_json(recovery / "latest_summary.json", summary)
    runs_volume.commit()

    print("\n" + "=" * 112, flush=True)
    print("INVESTIGATION 29 COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print(f"BCE         : {baseline['base_bce']:.6f} -> {final_metrics['final_bce']:.6f}", flush=True)
    print(f"False merges: {baseline['base_false_merge_count']} -> {final_metrics['final_false_merge_count']}", flush=True)
    print(f"False splits: {baseline['base_false_split_count']} -> {final_metrics['final_false_split_count']}", flush=True)
    print(f"Errors      : {baseline['base_threshold_error_count']} -> {final_metrics['final_threshold_error_count']}", flush=True)
    print(f"Fixed/broken: {final_metrics['fixed_h100_errors']} / {final_metrics['broken_h100_correct_edges']}", flush=True)
    print(f"Best step   : {best_step}", flush=True)
    print(f"Recovery    : {recovery}", flush=True)
    print("=" * 112, flush=True)
    return summary


@app.local_entrypoint()
def main(target_step: int = 300, run_name: str = DEFAULT_RUN_NAME) -> None:
    report = train.remote(target_step=target_step, run_name=run_name)
    print(json.dumps({
        "status": report.get("status"),
        "global_step": report.get("global_step"),
        "best_step": report.get("best_step"),
        "recovery_dir": report.get("recovery_dir"),
        "final_metrics": report.get("final_metrics"),
    }, indent=2, default=str))
