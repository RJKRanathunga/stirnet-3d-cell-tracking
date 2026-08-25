from __future__ import annotations

r"""
STIR-Net Investigation 26 — production separator-aware RAG barrier training.

Goal
----
Test whether the NEW production ``SeparatorAwareBarrier`` can teach the spatial
RAG to respect separator evidence that the dense geometry network already gets
right.

This is deliberately a short, conservative 300-step probe.

Architecture under test
-----------------------
The production model now computes:

    base_merge_logit
            |
            + exact separator/contact features
            |   + morphology-v2 edge embedding
            |          |
            |          v
            |   SeparatorAwareBarrier
            |          |
            |          v
            |      barrier >= 0
            |          |
            +----------+
                       v

    final_merge_logit = base_merge_logit - barrier

Only the production ``rag_network.separator_barrier`` is trainable by default.

FROZEN:
    acquisition
    evidence stem
    spatial backbone
    dense geometry decoder / separator predictor
    watershed
    RAG builder and morphology-v2 encoders
    legacy RAG message passing / classifier
    multicut parameters

TRAINABLE:
    rag_network.separator_barrier

Why crop mining happens BEFORE training
---------------------------------------
A 300-step targeted run is unsafe if only a few useful separator-backed cases
exist.  Therefore this script first scans the fixed training crop manifest with
the frozen h100 baseline.

A "good" crop contains at least one valid GT-negative RAG edge satisfying:

    strong separator evidence
    AND
    base p_merge >= --good-crop-min-base-probability

The default strong-separator definition is exactly the current production loss
definition:

    separator_mean >= 0.55

OR

    separator_max >= 0.85
    AND physical coverage(separator >= 0.70) >= 0.25

The default good-crop base probability is 0.50.  Edges already above the
production q=0.845 threshold are separately counted as HARD false merges and
receive highest priority.

At least 15 DISTINCT good training crops are required by default.  If fewer are
found, the script writes the mining report and exits BEFORE optimization.

Sampling
--------
Default:
    75% draws -> mined good-crop pool
    25% draws -> ordinary train-manifest pool

Inside each crop, the edge loss remains class-balanced and reserves 75% of the
negative quota for strong-separator negatives, prioritized by base p_merge.

Early stopping
--------------
The script validates every 50 optimizer steps and writes recovery checkpoints
at 100, 200, and 300 by default.

At the decision checkpoints 100 and 200 it evaluates:

1. fixed held-out validation crops:
       positive acceptance at q=0.845 must remain safe;

2. fixed target audit over the mined good-crop pool:
       strong-separator bad-edge rate must improve, OR
       mean strong-separator negative p_merge must fall enough.

Default learning gates:

    step 100:
        >= 15% reduction in target bad-edge rate
        OR >= 0.05 drop in mean strong-negative p_merge

    step 200:
        >= 30% reduction in target bad-edge rate
        OR >= 0.10 drop in mean strong-negative p_merge

Safety gate:
    held-out positive acceptance may not fall by > 0.02 absolute.

At an early-stop checkpoint the sequence is intentionally:

    validate
    -> SAVE checkpoint
    -> write decision JSON
    -> terminate

so a stopped run is fully recoverable/inspectable.

Important
---------
"Target audit" uses the mined training-target pool and is ONLY a learning-speed
diagnostic.  It is not treated as held-out generalization.  Safety/generalization
comes from the fixed validation manifest.

Typical command
---------------
From the repository root:

    python .\investigations\stirnet\26_separator_aware_rag_barrier_training.py

Default starting checkpoint:

    runs/stirnet/investigations/
        19_morphology_rag_v2_headroom_training/
        recovery/
        drosophila_12_morphology_rag_v2_headroom_h100/

Quick smoke:

    python .\investigations\stirnet\26_separator_aware_rag_barrier_training.py `
        --max-steps 4 `
        --validation-every 2 `
        --checkpoint-every 2 `
        --min-good-crops 2 `
        --disable-early-stop `
        --run-name separator_barrier_26_smoke
"""

import argparse
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


EXPERIMENT_NAME = "26_separator_aware_rag_barrier_training"
DEFAULT_SAMPLES = "Drosophila_1,Drosophila_2"
DEFAULT_SPACING_XYZ = "0.20312639,0.20312639,0.79099447"
DEFAULT_CHECKPOINT = (
    "runs/stirnet/investigations/"
    "19_morphology_rag_v2_headroom_training/"
    "recovery/"
    "drosophila_12_morphology_rag_v2_headroom_h100/"
    "checkpoint_step_000600.pt"
)


# ======================================================================================
# Repository / support
# ======================================================================================


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
        ):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not resolve repository root. Run from the cell-tracking repo."
    )


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_inv17_support():
    path = (
        ROOT
        / "investigations"
        / "stirnet"
        / "17_morphology_rag_multicrop_training.py"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Investigation 17 support is required: "
            f"{path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_stirnet_inv17_support_for_inv26",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# Generic helpers
# ======================================================================================


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
        ),
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(payload),
                sort_keys=True,
            )
        )
        handle.write("\n")
        handle.flush()


def _torch_load(path: Path, *, map_location="cpu") -> dict:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()

    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    best = path / "best_checkpoint.pt"
    if best.is_file():
        return best

    direct = sorted(path.glob("checkpoint_step_*.pt"))
    if direct:
        return direct[-1]

    nested_best = sorted(path.glob("**/best_checkpoint.pt"))
    if nested_best:
        return nested_best[-1]

    nested = sorted(path.glob("**/checkpoint_step_*.pt"))
    if nested:
        return nested[-1]

    raise FileNotFoundError(f"No checkpoint found inside {path}")


def _hydrate_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    if not is_dataclass(instance):
        raise TypeError("Expected a dataclass instance")

    allowed = {field.name for field in fields(instance)}
    for key, value in payload.items():
        if key not in allowed:
            continue

        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _hydrate_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, (tuple, list)):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)

    return instance


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(amp_dtype: str):
    if amp_dtype == "bf16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    if amp_dtype == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = float(
            torch.linalg.vector_norm(
                parameter.grad.detach().float()
            ).cpu()
        )
        total = math.hypot(total, norm)
    return total


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ======================================================================================
# Model construction / resume
# ======================================================================================


def _is_inv26_checkpoint(payload: dict) -> bool:
    extra = payload.get("extra", {})
    return (
        isinstance(extra, dict)
        and extra.get("experiment") == EXPERIMENT_NAME
        and isinstance(payload.get("model"), dict)
    )


def _build_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
):
    from learned.stirnet import StirNet, StirNetConfig

    payload = _torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    cfg = StirNetConfig()
    checkpoint_cfg = payload.get("model_config")
    if isinstance(checkpoint_cfg, dict):
        _hydrate_dataclass(cfg, checkpoint_cfg)

    # Investigation 26 tests the production branch, not an experiment-local copy.
    cfg.partition.rag_morphology_enabled = True
    cfg.partition.rag_morphology_detach_geometry = True
    cfg.partition.rag_separator_barrier_enabled = True
    cfg.partition.rag_separator_barrier_use_morphology = True
    cfg.partition.rag_separator_barrier_detach_geometry = True
    cfg.validate()

    model = StirNet(cfg)
    historical = payload.get("model")
    if not isinstance(historical, dict):
        raise ValueError(
            f"Checkpoint does not contain model state_dict: {checkpoint_path}"
        )

    exact_resume = _is_inv26_checkpoint(payload)

    if exact_resume:
        model.load_state_dict(historical, strict=True)
        transfer_report = {
            "exact_inv26_resume": True,
            "checkpoint_tensor_count": len(historical),
            "transferred_tensor_count": len(historical),
            "missing_non_barrier": [],
            "missing_barrier": [],
            "mismatched": {},
        }
    else:
        current = model.state_dict()
        compatible = {}
        mismatched = {}

        for name, value in historical.items():
            if name not in current:
                continue
            if tuple(current[name].shape) == tuple(value.shape):
                compatible[name] = value
            else:
                mismatched[name] = {
                    "checkpoint": tuple(value.shape),
                    "current": tuple(current[name].shape),
                }

        model.load_state_dict(compatible, strict=False)
        missing = sorted(set(current) - set(compatible))
        missing_barrier = [
            name
            for name in missing
            if name.startswith("rag_network.separator_barrier.")
        ]
        missing_non_barrier = [
            name
            for name in missing
            if not name.startswith("rag_network.separator_barrier.")
        ]

        # Starting from the current h100 checkpoint should introduce ONLY the
        # new production separator barrier. Any other omitted model tensor means
        # the experiment is no longer isolated.
        if missing_non_barrier:
            preview = "\n".join(
                f"  - {name}"
                for name in missing_non_barrier[:40]
            )
            raise RuntimeError(
                "Starting checkpoint is not a clean current h100 transfer. "
                "Non-barrier tensors are missing:\n"
                + preview
            )

        transfer_report = {
            "exact_inv26_resume": False,
            "checkpoint_tensor_count": len(historical),
            "transferred_tensor_count": len(compatible),
            "missing_non_barrier": missing_non_barrier,
            "missing_barrier": missing_barrier,
            "mismatched": mismatched,
        }

    model = model.to(device)

    # Freeze absolutely everything, then reopen only the production barrier.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    barrier = model.rag_network.separator_barrier
    if barrier is None:
        raise RuntimeError(
            "Production separator barrier was enabled but not instantiated"
        )

    for parameter in barrier.parameters():
        parameter.requires_grad_(True)

    model.eval()
    barrier.train()

    trainable = [
        parameter
        for parameter in barrier.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Separator barrier has no trainable parameters")

    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("rag_network.separator_barrier.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Unexpected non-barrier trainable tensors: "
            + ", ".join(unexpected_trainable[:20])
        )

    return model, cfg, payload, transfer_report, trainable


# ======================================================================================
# Production forward / targets
# ======================================================================================


def _forward_crop(
    *,
    model,
    crop,
    rag_criterion,
    amp_dtype: str,
    training: bool,
):
    # Frozen dense geometry is computed once and never receives barrier gradient.
    with torch.no_grad(), _autocast(amp_dtype):
        geometry_output = model(
            crop["spatial_inputs"],
            crop["spacing_um"],
            crop["dref_um"],
            spatial_padding_mask=crop.get("spatial_padding_mask"),
            execution_stage="geometry",
        )

    context = _autocast(amp_dtype)
    if training:
        with context:
            output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"),
                execution_stage="spatial",
                precomputed_geometry=geometry_output,
            )
    else:
        with torch.no_grad(), context:
            output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get("spatial_padding_mask"),
                execution_stage="spatial",
                precomputed_geometry=geometry_output,
            )

    rag = output.rag
    if rag.base_spatial_edge_logits is None:
        raise RuntimeError(
            "Production RAG did not expose base_spatial_edge_logits"
        )
    if rag.separator_barrier_features is None:
        raise RuntimeError(
            "Production RAG did not expose separator_barrier_features"
        )
    if rag.separator_barrier_score is None:
        raise RuntimeError(
            "Production RAG did not expose separator_barrier_score"
        )
    if rag.separator_barrier_correction is None:
        raise RuntimeError(
            "Production RAG did not expose separator_barrier_correction"
        )

    targets = rag_criterion.build_targets(
        rag,
        crop["gt_labels"],
        valid_mask=crop.get("supervision_valid_mask"),
    )

    return {
        "geometry_output": geometry_output,
        "output": output,
        "rag": rag,
        "targets": targets,
        "base_logits": rag.base_spatial_edge_logits,
        "final_logits": rag.spatial_edge_logits,
        "features": rag.separator_barrier_features,
        "score": rag.separator_barrier_score,
        "correction": rag.separator_barrier_correction,
    }


def _strong_separator_mask(
    result,
    loss_cfg,
):
    features = result["features"]
    return (
        features[:, 0]
        >= float(loss_cfg.separator_barrier_negative_mean_min)
    ) | (
        (
            features[:, 1]
            >= float(loss_cfg.separator_barrier_negative_max_min)
        )
        & (
            features[:, 4]
            >= float(
                loss_cfg.separator_barrier_negative_coverage70_min
            )
        )
    )


# ======================================================================================
# Crop mining
# ======================================================================================


def _mine_good_crops(
    *,
    support,
    model,
    source_batches,
    splits,
    rag_criterion,
    loss_cfg,
    amp_dtype: str,
    partial_ignore_margin_um: float,
    good_crop_min_base_probability: float,
    merge_threshold: float,
    max_good_crops: int,
):
    """Mine fixed distinct crops before any optimizer update."""
    jobs = []
    for sample, split in splits.items():
        for manifest_index in split["train_indices"]:
            jobs.append((sample, int(manifest_index)))

    # Deterministic interleaving/shuffle prevents one dataset from monopolizing
    # the first --max-good-crops candidates.
    rng = random.Random(19_260_001)
    rng.shuffle(jobs)

    good = []
    ordinary = []
    scan_rows = []

    progress = tqdm(
        total=len(jobs),
        desc="Mining separator crops",
        unit="crop",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    for sample, manifest_index in jobs:
        record = splits[sample]["records"][manifest_index]
        crop_cpu, _ = support._materialize_crop(
            source_batches[sample],
            record,
            partial_ignore_margin_um=partial_ignore_margin_um,
        )
        crop = support._move_crop_to_cuda(crop_cpu)

        with torch.no_grad():
            result = _forward_crop(
                model=model,
                crop=crop,
                rag_criterion=rag_criterion,
                amp_dtype=amp_dtype,
                training=False,
            )

        targets = result["targets"]
        valid = targets.valid.bool()
        negative = valid & (targets.target <= 0.5)
        strong = _strong_separator_mask(result, loss_cfg)
        strong_negative = negative & strong

        base_p = result["base_logits"].detach().sigmoid()
        good_edge = (
            strong_negative
            & (base_p >= good_crop_min_base_probability)
        )
        hard_edge = (
            strong_negative
            & (base_p >= merge_threshold)
        )

        row = {
            "sample": sample,
            "manifest_index": manifest_index,
            "valid_edges": int(valid.sum().item()),
            "strong_negative_edges": int(
                strong_negative.sum().item()
            ),
            "good_edges": int(good_edge.sum().item()),
            "hard_edges": int(hard_edge.sum().item()),
            "max_strong_negative_base_probability": (
                float(base_p[strong_negative].max().cpu())
                if bool(strong_negative.any())
                else 0.0
            ),
            "mean_strong_negative_base_probability": (
                float(base_p[strong_negative].mean().cpu())
                if bool(strong_negative.any())
                else 0.0
            ),
        }
        scan_rows.append(row)

        if bool(good_edge.any()):
            good.append(row)
        else:
            ordinary.append(
                {
                    "sample": sample,
                    "manifest_index": manifest_index,
                }
            )

        progress.set_postfix(
            {
                "good": len(good),
                "hard": sum(
                    int(item["hard_edges"] > 0)
                    for item in good
                ),
                "sample": sample.replace("Drosophila_", "D"),
                "idx": manifest_index,
            },
            refresh=False,
        )
        progress.update(1)

        del result, crop, crop_cpu
        torch.cuda.empty_cache()

        if max_good_crops > 0 and len(good) >= max_good_crops:
            break

    progress.close()

    # Hard current production failures first, then highest baseline p_merge.
    good.sort(
        key=lambda row: (
            int(row["hard_edges"] > 0),
            row["hard_edges"],
            row["max_strong_negative_base_probability"],
            row["good_edges"],
        ),
        reverse=True,
    )

    return {
        "good": good,
        "ordinary": ordinary,
        "scan_rows": scan_rows,
        "scanned_crop_count": len(scan_rows),
        "hard_good_crop_count": sum(
            int(row["hard_edges"] > 0)
            for row in good
        ),
        "good_edge_count": sum(
            row["good_edges"]
            for row in good
        ),
        "hard_edge_count": sum(
            row["hard_edges"]
            for row in good
        ),
    }


# ======================================================================================
# Edge selection / losses
# ======================================================================================


def _spread_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        count,
        device=indices.device,
    ).round().long()
    return indices[positions]


def _select_edges(
    *,
    result,
    max_edges_per_class: int,
    separator_negative_fraction: float,
    good_crop_min_base_probability: float,
    loss_cfg,
):
    targets = result["targets"]
    valid = targets.valid.bool()
    positive = torch.nonzero(
        valid & (targets.target > 0.5),
        as_tuple=False,
    ).flatten()
    negative = torch.nonzero(
        valid & (targets.target <= 0.5),
        as_tuple=False,
    ).flatten()

    selected_positive = _spread_indices(
        positive,
        min(max_edges_per_class, int(positive.numel())),
    )

    if selected_positive.numel() and negative.numel():
        negative_quota = min(
            int(selected_positive.numel()),
            max_edges_per_class,
            int(negative.numel()),
        )
    else:
        negative_quota = min(
            max_edges_per_class,
            int(negative.numel()),
        )

    strong = _strong_separator_mask(result, loss_cfg)
    base_p = result["base_logits"].detach().sigmoid()

    strong_negative = negative[strong[negative]]
    priority_negative = strong_negative[
        base_p[strong_negative] >= good_crop_min_base_probability
    ]

    # Prefer current wrong/ambiguous separator-backed edges.
    priority_score = (
        base_p[priority_negative]
        + 0.10 * result["features"][priority_negative, 0].detach()
        + 0.05 * result["features"][priority_negative, 4].detach()
    )
    if priority_negative.numel():
        order = torch.argsort(priority_score, descending=True)
        priority_negative = priority_negative[order]

    reserved = min(
        int(round(negative_quota * separator_negative_fraction)),
        int(priority_negative.numel()),
    )
    selected_priority = priority_negative[:reserved]

    used = set(
        int(value)
        for value in selected_priority.detach().cpu().tolist()
    )
    remaining_negative = torch.as_tensor(
        [
            int(value)
            for value in negative.detach().cpu().tolist()
            if int(value) not in used
        ],
        device=negative.device,
        dtype=negative.dtype,
    )
    selected_ordinary = _spread_indices(
        remaining_negative,
        negative_quota - reserved,
    )
    selected_negative = torch.cat(
        [selected_priority, selected_ordinary],
        dim=0,
    )

    return {
        "positive_indices": selected_positive,
        "negative_indices": selected_negative,
        "priority_negative_indices": selected_priority,
        "valid_positive_count": int(positive.numel()),
        "valid_negative_count": int(negative.numel()),
        "strong_negative_count": int(strong_negative.numel()),
        "priority_negative_count": int(priority_negative.numel()),
    }


def _balanced_edge_bce(
    logits: torch.Tensor,
    selected: dict[str, Any],
):
    pieces = []
    positive = selected["positive_indices"]
    negative = selected["negative_indices"]

    positive_loss = logits.sum() * 0
    negative_loss = logits.sum() * 0

    if positive.numel():
        positive_loss = F.binary_cross_entropy_with_logits(
            logits[positive],
            torch.ones_like(logits[positive]),
        )
        pieces.append(positive_loss)

    if negative.numel():
        negative_loss = F.binary_cross_entropy_with_logits(
            logits[negative],
            torch.zeros_like(logits[negative]),
        )
        pieces.append(negative_loss)

    if not pieces:
        return logits.sum() * 0, positive_loss, negative_loss

    return (
        torch.stack(pieces).mean(),
        positive_loss,
        negative_loss,
    )


def _positive_preservation(
    *,
    result,
    merge_threshold: float,
):
    targets = result["targets"]
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    base_p = result["base_logits"].detach().sigmoid()
    preserve = positive & (base_p >= merge_threshold)

    if not bool(preserve.any()):
        zero = result["final_logits"].sum() * 0
        return zero, zero, preserve

    logit_loss = F.smooth_l1_loss(
        result["final_logits"][preserve],
        result["base_logits"].detach()[preserve],
        beta=1.0,
    )
    correction_loss = result["correction"][preserve].mean()

    return logit_loss, correction_loss, preserve


# ======================================================================================
# Evaluation
# ======================================================================================


def _new_eval_accumulator():
    return {
        "crop_count": 0.0,
        "valid_edge_count": 0.0,
        "positive_edge_count": 0.0,
        "negative_edge_count": 0.0,
        "strong_negative_count": 0.0,
        "bad_strong_negative_count": 0.0,
        "hard_strong_negative_count": 0.0,
        "positive_accept_count": 0.0,
        "false_merge_count": 0.0,
        "bce_sum": 0.0,
        "strong_negative_probability_sum": 0.0,
        "positive_probability_sum": 0.0,
        "correction_strong_negative_sum": 0.0,
        "correction_positive_sum": 0.0,
    }


def _add_eval(acc, row):
    for key, value in row.items():
        acc[key] = acc.get(key, 0.0) + float(value)


def _finalize_eval(acc):
    def ratio(a, b):
        return float(a / b) if b > 0 else 0.0

    valid = acc["valid_edge_count"]
    positive = acc["positive_edge_count"]
    negative = acc["negative_edge_count"]
    strong_negative = acc["strong_negative_count"]

    return {
        "crop_count": int(acc["crop_count"]),
        "valid_edge_count": int(valid),
        "positive_edge_count": int(positive),
        "negative_edge_count": int(negative),
        "strong_negative_count": int(strong_negative),
        "bad_strong_negative_count": int(
            acc["bad_strong_negative_count"]
        ),
        "bad_strong_negative_rate": ratio(
            acc["bad_strong_negative_count"],
            strong_negative,
        ),
        "hard_strong_negative_count": int(
            acc["hard_strong_negative_count"]
        ),
        "hard_strong_negative_rate": ratio(
            acc["hard_strong_negative_count"],
            strong_negative,
        ),
        "positive_accept_count": int(
            acc["positive_accept_count"]
        ),
        "positive_accept_rate": ratio(
            acc["positive_accept_count"],
            positive,
        ),
        "false_merge_count": int(acc["false_merge_count"]),
        "false_merge_rate": ratio(
            acc["false_merge_count"],
            negative,
        ),
        "bce": ratio(acc["bce_sum"], valid),
        "mean_strong_negative_probability": ratio(
            acc["strong_negative_probability_sum"],
            strong_negative,
        ),
        "mean_positive_probability": ratio(
            acc["positive_probability_sum"],
            positive,
        ),
        "mean_correction_strong_negative": ratio(
            acc["correction_strong_negative_sum"],
            strong_negative,
        ),
        "mean_correction_positive": ratio(
            acc["correction_positive_sum"],
            positive,
        ),
    }


def _eval_contribution(
    *,
    result,
    loss_cfg,
    good_crop_min_base_probability: float,
    merge_threshold: float,
    use_base_logits: bool,
):
    logits = (
        result["base_logits"]
        if use_base_logits
        else result["final_logits"]
    )
    probability = logits.detach().sigmoid()
    targets = result["targets"]
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    negative = valid & ~positive
    strong = _strong_separator_mask(result, loss_cfg)
    strong_negative = negative & strong

    def count(mask):
        return float(mask.sum().item())

    def sum_values(mask, values):
        if not bool(mask.any()):
            return 0.0
        return float(
            values[mask].detach().float().sum().cpu()
        )

    correction = (
        torch.zeros_like(result["correction"])
        if use_base_logits
        else result["correction"].detach()
    )

    return {
        "crop_count": 1.0,
        "valid_edge_count": count(valid),
        "positive_edge_count": count(positive),
        "negative_edge_count": count(negative),
        "strong_negative_count": count(strong_negative),
        "bad_strong_negative_count": count(
            strong_negative
            & (probability >= good_crop_min_base_probability)
        ),
        "hard_strong_negative_count": count(
            strong_negative
            & (probability >= merge_threshold)
        ),
        "positive_accept_count": count(
            positive
            & (probability >= merge_threshold)
        ),
        "false_merge_count": count(
            negative
            & (probability >= merge_threshold)
        ),
        "bce_sum": (
            float(
                F.binary_cross_entropy_with_logits(
                    logits[valid],
                    targets.target[valid],
                    reduction="sum",
                ).detach().float().cpu()
            )
            if bool(valid.any())
            else 0.0
        ),
        "strong_negative_probability_sum": sum_values(
            strong_negative,
            probability,
        ),
        "positive_probability_sum": sum_values(
            positive,
            probability,
        ),
        "correction_strong_negative_sum": sum_values(
            strong_negative,
            correction,
        ),
        "correction_positive_sum": sum_values(
            positive,
            correction,
        ),
    }


def _evaluate_jobs(
    *,
    support,
    model,
    source_batches,
    splits,
    jobs,
    rag_criterion,
    loss_cfg,
    amp_dtype: str,
    partial_ignore_margin_um: float,
    good_crop_min_base_probability: float,
    merge_threshold: float,
    description: str,
):
    candidate = _new_eval_accumulator()
    baseline = _new_eval_accumulator()

    progress = tqdm(
        total=len(jobs),
        desc=description,
        unit="crop",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    model.eval()
    model.rag_network.separator_barrier.eval()

    for row in jobs:
        sample = row["sample"]
        manifest_index = int(row["manifest_index"])
        record = splits[sample]["records"][manifest_index]

        crop_cpu, _ = support._materialize_crop(
            source_batches[sample],
            record,
            partial_ignore_margin_um=partial_ignore_margin_um,
        )
        crop = support._move_crop_to_cuda(crop_cpu)

        with torch.no_grad():
            result = _forward_crop(
                model=model,
                crop=crop,
                rag_criterion=rag_criterion,
                amp_dtype=amp_dtype,
                training=False,
            )

        _add_eval(
            candidate,
            _eval_contribution(
                result=result,
                loss_cfg=loss_cfg,
                good_crop_min_base_probability=(
                    good_crop_min_base_probability
                ),
                merge_threshold=merge_threshold,
                use_base_logits=False,
            ),
        )
        _add_eval(
            baseline,
            _eval_contribution(
                result=result,
                loss_cfg=loss_cfg,
                good_crop_min_base_probability=(
                    good_crop_min_base_probability
                ),
                merge_threshold=merge_threshold,
                use_base_logits=True,
            ),
        )

        progress.set_postfix(
            {
                "sample": sample.replace("Drosophila_", "D"),
                "idx": manifest_index,
            },
            refresh=False,
        )
        progress.update(1)

        del result, crop, crop_cpu
        torch.cuda.empty_cache()

    progress.close()
    model.rag_network.separator_barrier.train()

    return {
        "candidate": _finalize_eval(candidate),
        "base_without_barrier": _finalize_eval(baseline),
    }


def _validation_jobs_from_split(splits):
    jobs = []
    for sample, split in splits.items():
        for manifest_index in split["validation_indices"]:
            jobs.append(
                {
                    "sample": sample,
                    "manifest_index": int(manifest_index),
                }
            )
    return jobs


# ======================================================================================
# Early-stop logic
# ======================================================================================


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return float((baseline - candidate) / baseline)


def _early_stop_decision(
    *,
    step: int,
    heldout_baseline: dict,
    heldout_candidate: dict,
    target_baseline: dict,
    target_candidate: dict,
    args,
):
    if step not in {100, 200}:
        return {
            "decision_checkpoint": False,
            "stop": False,
            "reasons": [],
        }

    positive_drop = (
        heldout_baseline["positive_accept_rate"]
        - heldout_candidate["positive_accept_rate"]
    )

    baseline_bad = target_baseline["bad_strong_negative_rate"]
    candidate_bad = target_candidate["bad_strong_negative_rate"]
    bad_relative_reduction = _relative_reduction(
        baseline_bad,
        candidate_bad,
    )

    mean_probability_drop = (
        target_baseline["mean_strong_negative_probability"]
        - target_candidate["mean_strong_negative_probability"]
    )

    if step == 100:
        required_relative = args.early_stop_step100_min_bad_reduction
        required_probability_drop = (
            args.early_stop_step100_min_probability_drop
        )
    else:
        required_relative = args.early_stop_step200_min_bad_reduction
        required_probability_drop = (
            args.early_stop_step200_min_probability_drop
        )

    learning_ok = (
        bad_relative_reduction >= required_relative
        or mean_probability_drop >= required_probability_drop
    )

    safety_ok = (
        positive_drop
        <= args.early_stop_max_positive_accept_drop
    )

    # Extra selectivity diagnostic. This is deliberately NOT a hard condition:
    # the loss/probability gates are more direct, but this tells us whether the
    # branch is preferentially acting on the desired edges.
    correction_selectivity = (
        target_candidate["mean_correction_strong_negative"]
        - heldout_candidate["mean_correction_positive"]
    )

    reasons = []
    if not learning_ok:
        reasons.append(
            "target separator-backed edges are not improving fast enough"
        )
    if not safety_ok:
        reasons.append(
            "held-out legitimate merge acceptance degraded beyond safety floor"
        )

    return {
        "decision_checkpoint": True,
        "stop": bool(not learning_ok or not safety_ok),
        "learning_ok": bool(learning_ok),
        "safety_ok": bool(safety_ok),
        "reasons": reasons,
        "positive_accept_drop": float(positive_drop),
        "target_bad_relative_reduction": float(
            bad_relative_reduction
        ),
        "target_mean_probability_drop": float(
            mean_probability_drop
        ),
        "correction_selectivity": float(
            correction_selectivity
        ),
        "required_bad_relative_reduction": float(
            required_relative
        ),
        "required_probability_drop": float(
            required_probability_drop
        ),
        "max_positive_accept_drop": float(
            args.early_stop_max_positive_accept_drop
        ),
    }


# ======================================================================================
# Checkpointing
# ======================================================================================


def _save_checkpoint(
    *,
    path: Path,
    model,
    model_cfg,
    optimizer,
    scaler,
    global_step: int,
    source_checkpoint: Path,
    mining_summary: dict,
    heldout_baseline: dict,
    target_baseline: dict,
    validation: dict | None,
    target_audit: dict | None,
    decision: dict | None,
    config_payload: dict,
):
    payload = {
        "architecture": "STIR-Net",
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "training_config": {
            "experiment": EXPERIMENT_NAME,
            "max_steps": int(config_payload["max_steps"]),
        },
        "extra": {
            "experiment": EXPERIMENT_NAME,
            "source_checkpoint": str(source_checkpoint),
            "mining_summary": mining_summary,
            "heldout_baseline": heldout_baseline,
            "target_baseline": target_baseline,
            "validation": validation,
            "target_audit": target_audit,
            "early_stop_decision": decision,
            "config": config_payload,
        },
    }
    _atomic_torch_save(path, payload)


# ======================================================================================
# Main training
# ======================================================================================


def _training_impl(args) -> dict[str, Any]:
    from learned.stirnet.model.partition.rag import RAGCriterion
    from learned.stirnet.training.config import LossConfig
    from learned.stirnet.training.criterion import (
        _separator_barrier_auxiliary,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Investigation 26 requires CUDA."
        )

    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.min_good_crops < 1:
        raise ValueError("--min-good-crops must be positive")
    if not 0.0 <= args.good_crop_fraction <= 1.0:
        raise ValueError("--good-crop-fraction must be in [0,1]")
    if not 0.0 <= args.separator_negative_fraction <= 1.0:
        raise ValueError(
            "--separator-negative-fraction must be in [0,1]"
        )
    if not 0.0 <= args.good_crop_min_base_probability <= 1.0:
        raise ValueError(
            "--good-crop-min-base-probability must be in [0,1]"
        )
    if not 0.0 < args.merge_threshold < 1.0:
        raise ValueError("--merge-threshold must be in (0,1)")

    support = _load_inv17_support()
    _seed_everything(args.seed)

    source_checkpoint = _resolve_checkpoint(args.checkpoint)

    (
        model,
        model_cfg,
        source_payload,
        transfer_report,
        trainable_parameters,
    ) = _build_model_from_checkpoint(
        source_checkpoint,
        device="cuda",
    )

    exact_resume = bool(
        transfer_report["exact_inv26_resume"]
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": trainable_parameters,
                "lr": args.barrier_lr,
                "weight_decay": args.weight_decay,
                "name": "production_separator_barrier",
            }
        ]
    )

    amp_dtype = (
        "bf16"
        if torch.cuda.is_bf16_supported()
        else "fp16"
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_dtype == "fp16"),
    )

    global_step = 0
    if exact_resume:
        global_step = int(
            source_payload.get("global_step", 0)
        )
        if "optimizer" in source_payload:
            optimizer.load_state_dict(
                source_payload["optimizer"]
            )
        if "scaler" in source_payload:
            scaler.load_state_dict(
                source_payload["scaler"]
            )
        # Reassert model modes after state restoration.
        model.eval()
        model.rag_network.separator_barrier.train()

    samples = tuple(
        token.strip()
        for token in args.samples.split(",")
        if token.strip()
    )
    if not samples:
        raise ValueError("No samples requested")

    crop_shape = support._parse_shape_zyx(
        args.crop_shape_zyx
    )
    spacing_override_zyx_um = (
        support._parse_spacing_xyz_override(
            args.spacing_xyz
        )
    )
    nis3d_root = support._discover_nis3d_root(
        samples,
        data_dir=args.data_dir,
        execution_mode="local",
    )

    experiment_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / EXPERIMENT_NAME
    )
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        experiment_root
        / "attempts"
        / f"{timestamp}_{args.run_name}"
    )
    recovery_dir = (
        experiment_root
        / "recovery"
        / args.run_name
    )
    cache_root = experiment_root / "cache"

    run_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    props = torch.cuda.get_device_properties(0)

    print("=" * 124, flush=True)
    print(
        "STIR-Net Investigation 26 — production separator-aware RAG barrier",
        flush=True,
    )
    print("=" * 124, flush=True)
    print(f"GPU                       : {props.name}", flush=True)
    print(
        f"VRAM                      : "
        f"{props.total_memory / 2**30:.2f} GiB",
        flush=True,
    )
    print(f"Starting checkpoint       : {source_checkpoint}", flush=True)
    print(f"Exact Inv26 resume        : {exact_resume}", flush=True)
    print(
        "Transferred tensors       : "
        f"{transfer_report['transferred_tensor_count']}/"
        f"{transfer_report['checkpoint_tensor_count']}",
        flush=True,
    )
    print(
        "New barrier tensors       : "
        f"{len(transfer_report['missing_barrier'])}",
        flush=True,
    )
    print(
        "Trainable parameters      : "
        f"{sum(p.numel() for p in trainable_parameters):,}",
        flush=True,
    )
    print("Trainable scope           : production separator barrier ONLY", flush=True)
    print("Dense geometry            : FROZEN", flush=True)
    print("Morphology-v2             : FROZEN", flush=True)
    print("Legacy RAG                : FROZEN", flush=True)
    print(f"Max steps                 : {args.max_steps}", flush=True)
    print(f"Min good crops            : {args.min_good_crops}", flush=True)
    print(
        "Good crop definition      : strong GT-negative edge "
        f"with base p >= {args.good_crop_min_base_probability:.3f}",
        flush=True,
    )
    print(
        f"Good-crop draw fraction   : {args.good_crop_fraction:.2f}",
        flush=True,
    )
    print(
        f"Production merge q        : {args.merge_threshold:.3f}",
        flush=True,
    )
    print(
        f"Validation every          : {args.validation_every} steps",
        flush=True,
    )
    print(
        f"Checkpoint every          : {args.checkpoint_every} steps",
        flush=True,
    )
    print(
        "Early-stop decisions      : "
        + ("OFF" if args.disable_early_stop else "steps 100 and 200"),
        flush=True,
    )
    print("=" * 124, flush=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    source_batches = {}
    sample_reports = {}

    for sample in samples:
        print(f"[data] preparing {sample} ...", flush=True)
        signature = support._data_signature(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=(
                spacing_override_zyx_um
            ),
            confidence_ignore_margin_um=(
                args.confidence_ignore_margin_um
            ),
        )
        source_batch, report = support._prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=(
                spacing_override_zyx_um
            ),
            confidence_ignore_margin_um=(
                args.confidence_ignore_margin_um
            ),
            cache_root=cache_root,
            cache_namespace=signature,
        )
        source_batches[sample] = source_batch
        sample_reports[sample] = report
        print(
            f"[data] {sample}: "
            f"shape={tuple(report['shape_zyx'])} "
            f"GT={report['gt_ids_kept']} "
            f"source={report['source_instance_count']} "
            f"dref={report['model_dref_um']:.4f}um",
            flush=True,
        )

    splits = support._build_split(
        source_batches,
        crop_shape_zyx=crop_shape,
        validation_crops_per_sample=(
            args.validation_crops_per_sample
        ),
    )

    split_summary = {}
    for sample in samples:
        row = splits[sample]
        split_summary[sample] = {
            "manifest_record_count": len(row["records"]),
            "train_indices": [
                int(value)
                for value in row["train_indices"]
            ],
            "validation_indices": [
                int(value)
                for value in row["validation_indices"]
            ],
        }
        print(
            f"[split] {sample}: "
            f"manifest={len(row['records'])} "
            f"train={len(row['train_indices'])} "
            f"val={row['validation_indices']}",
            flush=True,
        )

    rag_criterion = RAGCriterion(
        model_cfg.partition
    ).to("cuda")
    rag_criterion.eval()

    loss_cfg = LossConfig()
    loss_cfg.separator_barrier_semantic_weight = (
        args.semantic_weight
    )
    loss_cfg.separator_barrier_margin_weight = (
        args.margin_weight
    )

    # Keep the production strong-separator thresholds by default, but expose
    # explicit experiment overrides.
    if args.strong_separator_mean_min is not None:
        loss_cfg.separator_barrier_negative_mean_min = (
            args.strong_separator_mean_min
        )
    if args.strong_separator_max_min is not None:
        loss_cfg.separator_barrier_negative_max_min = (
            args.strong_separator_max_min
        )
    if args.strong_separator_coverage70_min is not None:
        loss_cfg.separator_barrier_negative_coverage70_min = (
            args.strong_separator_coverage70_min
        )
    if args.signed_margin is not None:
        loss_cfg.separator_barrier_signed_margin = (
            args.signed_margin
        )

    # ------------------------------------------------------------------
    # Pre-training mining gate.
    # ------------------------------------------------------------------
    print(
        "[mining] scanning frozen h100 RAG for useful separator-backed crops ...",
        flush=True,
    )
    mining = _mine_good_crops(
        support=support,
        model=model,
        source_batches=source_batches,
        splits=splits,
        rag_criterion=rag_criterion,
        loss_cfg=loss_cfg,
        amp_dtype=amp_dtype,
        partial_ignore_margin_um=(
            args.partial_ignore_margin_um
        ),
        good_crop_min_base_probability=(
            args.good_crop_min_base_probability
        ),
        merge_threshold=args.merge_threshold,
        max_good_crops=args.max_good_crops,
    )

    mining_summary = {
        "scanned_crop_count": mining["scanned_crop_count"],
        "good_crop_count": len(mining["good"]),
        "hard_good_crop_count": mining[
            "hard_good_crop_count"
        ],
        "good_edge_count": mining["good_edge_count"],
        "hard_edge_count": mining["hard_edge_count"],
        "min_good_crops_required": args.min_good_crops,
        "good_crop_min_base_probability": (
            args.good_crop_min_base_probability
        ),
        "good_crops": mining["good"],
    }
    _atomic_json(
        run_dir / "mining_summary.json",
        mining_summary,
    )
    _atomic_json(
        run_dir / "mining_scan.json",
        mining["scan_rows"],
    )

    print(
        "[mining] "
        f"scanned={mining['scanned_crop_count']} "
        f"good={len(mining['good'])} "
        f"hard-good={mining['hard_good_crop_count']} "
        f"good-edges={mining['good_edge_count']} "
        f"hard-edges={mining['hard_edge_count']}",
        flush=True,
    )

    if len(mining["good"]) < args.min_good_crops:
        summary = {
            "status": "stopped_insufficient_good_crops",
            "reason": (
                f"Only {len(mining['good'])} good crops were found; "
                f"minimum is {args.min_good_crops}. No optimizer step was run."
            ),
            "global_step": global_step,
            "source_checkpoint": str(source_checkpoint),
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "mining": mining_summary,
        }
        _atomic_json(
            run_dir / "summary.json",
            summary,
        )
        print("=" * 124, flush=True)
        print(
            "[STOP] Insufficient distinct useful crops. "
            "Training was NOT started.",
            flush=True,
        )
        print(
            f"       found={len(mining['good'])} "
            f"required={args.min_good_crops}",
            flush=True,
        )
        print("=" * 124, flush=True)
        return summary

    good_jobs = [
        {
            "sample": row["sample"],
            "manifest_index": int(row["manifest_index"]),
        }
        for row in mining["good"]
    ]

    ordinary_jobs = []
    good_keys = {
        (row["sample"], int(row["manifest_index"]))
        for row in mining["good"]
    }
    for sample, split in splits.items():
        for manifest_index in split["train_indices"]:
            key = (sample, int(manifest_index))
            if key not in good_keys:
                ordinary_jobs.append(
                    {
                        "sample": sample,
                        "manifest_index": int(manifest_index),
                    }
                )

    if not ordinary_jobs:
        # A safe fallback: ordinary draws can use the full train pool, but good
        # draws are still explicitly scheduled separately.
        ordinary_jobs = [
            {
                "sample": sample,
                "manifest_index": int(manifest_index),
            }
            for sample, split in splits.items()
            for manifest_index in split["train_indices"]
        ]

    validation_jobs = _validation_jobs_from_split(
        splits
    )

    # ------------------------------------------------------------------
    # Fixed baselines.
    # ------------------------------------------------------------------
    print(
        "[baseline] fixed held-out validation ...",
        flush=True,
    )
    heldout0 = _evaluate_jobs(
        support=support,
        model=model,
        source_batches=source_batches,
        splits=splits,
        jobs=validation_jobs,
        rag_criterion=rag_criterion,
        loss_cfg=loss_cfg,
        amp_dtype=amp_dtype,
        partial_ignore_margin_um=(
            args.partial_ignore_margin_um
        ),
        good_crop_min_base_probability=(
            args.good_crop_min_base_probability
        ),
        merge_threshold=args.merge_threshold,
        description="Inv26 heldout baseline",
    )
    heldout_baseline = heldout0["base_without_barrier"]

    print(
        "[baseline] fixed target audit over mined good crops ...",
        flush=True,
    )
    target0 = _evaluate_jobs(
        support=support,
        model=model,
        source_batches=source_batches,
        splits=splits,
        jobs=good_jobs,
        rag_criterion=rag_criterion,
        loss_cfg=loss_cfg,
        amp_dtype=amp_dtype,
        partial_ignore_margin_um=(
            args.partial_ignore_margin_um
        ),
        good_crop_min_base_probability=(
            args.good_crop_min_base_probability
        ),
        merge_threshold=args.merge_threshold,
        description="Inv26 target baseline",
    )
    target_baseline = target0["base_without_barrier"]

    print(
        "[baseline] heldout "
        f"FM={heldout_baseline['false_merge_count']}/"
        f"{heldout_baseline['negative_edge_count']} "
        f"PA={heldout_baseline['positive_accept_rate']:.4f} | "
        "target "
        f"bad={target_baseline['bad_strong_negative_count']}/"
        f"{target_baseline['strong_negative_count']} "
        f"hard={target_baseline['hard_strong_negative_count']} "
        f"meanP={target_baseline['mean_strong_negative_probability']:.4f}",
        flush=True,
    )

    config_payload = {
        "experiment": EXPERIMENT_NAME,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_global_step": int(
            source_payload.get("global_step", -1)
        ),
        "transfer_report": transfer_report,
        "model": model_cfg.to_dict(),
        "samples": list(samples),
        "sample_reports": sample_reports,
        "split": split_summary,
        "max_steps": int(args.max_steps),
        "min_good_crops": int(args.min_good_crops),
        "max_good_crops": int(args.max_good_crops),
        "good_crop_fraction": float(args.good_crop_fraction),
        "good_crop_min_base_probability": float(
            args.good_crop_min_base_probability
        ),
        "separator_negative_fraction": float(
            args.separator_negative_fraction
        ),
        "max_edges_per_class": int(args.max_edges_per_class),
        "barrier_lr": float(args.barrier_lr),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "semantic_weight": float(args.semantic_weight),
        "margin_weight": float(args.margin_weight),
        "positive_preservation_weight": float(
            args.positive_preservation_weight
        ),
        "positive_barrier_weight": float(
            args.positive_barrier_weight
        ),
        "merge_threshold": float(args.merge_threshold),
        "strong_separator_mean_min": float(
            loss_cfg.separator_barrier_negative_mean_min
        ),
        "strong_separator_max_min": float(
            loss_cfg.separator_barrier_negative_max_min
        ),
        "strong_separator_coverage70_min": float(
            loss_cfg.separator_barrier_negative_coverage70_min
        ),
        "signed_margin": float(
            loss_cfg.separator_barrier_signed_margin
        ),
        "early_stop_enabled": not args.disable_early_stop,
        "early_stop_step100_min_bad_reduction": float(
            args.early_stop_step100_min_bad_reduction
        ),
        "early_stop_step200_min_bad_reduction": float(
            args.early_stop_step200_min_bad_reduction
        ),
        "early_stop_step100_min_probability_drop": float(
            args.early_stop_step100_min_probability_drop
        ),
        "early_stop_step200_min_probability_drop": float(
            args.early_stop_step200_min_probability_drop
        ),
        "early_stop_max_positive_accept_drop": float(
            args.early_stop_max_positive_accept_drop
        ),
        "validation_every": int(args.validation_every),
        "checkpoint_every": int(args.checkpoint_every),
        "amp_dtype": amp_dtype,
        "seed": int(args.seed),
    }

    _atomic_json(
        run_dir / "config.json",
        config_payload,
    )
    _atomic_json(
        run_dir / "heldout_baseline.json",
        heldout_baseline,
    )
    _atomic_json(
        run_dir / "target_baseline.json",
        target_baseline,
    )

    # Exact resume uses its optimizer/global step, but mining and baselines are
    # recomputed under the same deterministic crop definitions. This keeps the
    # early-stop decision auditable after resume.
    if global_step >= args.max_steps:
        summary = {
            "status": "already_complete",
            "global_step": global_step,
            "max_steps": args.max_steps,
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
        }
        _atomic_json(
            run_dir / "summary.json",
            summary,
        )
        return summary

    history_path = run_dir / "history.jsonl"
    validation_path = run_dir / "validation.jsonl"
    decision_path = run_dir / "early_stop_decisions.jsonl"

    rng = random.Random(args.seed + 26)
    good_cycle = good_jobs.copy()
    ordinary_cycle = ordinary_jobs.copy()
    rng.shuffle(good_cycle)
    rng.shuffle(ordinary_cycle)
    good_cursor = 0
    ordinary_cursor = 0

    def next_job(step: int):
        nonlocal good_cursor, ordinary_cursor
        # Deterministic fractional scheduler without independent Bernoulli noise.
        # Bresenham-like deterministic fraction scheduler. For the default
        # f=0.75 this yields exactly 3 good draws and 1 ordinary draw per
        # four-step block (up to phase), without Bernoulli sampling noise.
        use_good = (
            math.floor((step + 1) * args.good_crop_fraction)
            > math.floor(step * args.good_crop_fraction)
        )

        if use_good:
            if good_cursor >= len(good_cycle):
                rng.shuffle(good_cycle)
                good_cursor = 0
            job = good_cycle[good_cursor]
            good_cursor += 1
            return dict(job), "good"

        if ordinary_cursor >= len(ordinary_cycle):
            rng.shuffle(ordinary_cycle)
            ordinary_cursor = 0
        job = ordinary_cycle[ordinary_cursor]
        ordinary_cursor += 1
        return dict(job), "ordinary"

    best_rank = None
    best_step = None
    best_path = recovery_dir / "best_checkpoint.pt"

    started = time.perf_counter()
    skipped = 0
    consecutive_skips = 0
    early_stopped = False
    early_stop_reason = None
    last_validation = None
    last_target_audit = None
    last_decision = None

    progress = tqdm(
        total=args.max_steps,
        initial=global_step,
        desc="STIR-Net sep barrier",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    try:
        while global_step < args.max_steps:
            job, provenance = next_job(global_step)
            sample = job["sample"]
            manifest_index = int(job["manifest_index"])
            record = splits[sample]["records"][manifest_index]

            crop_cpu, _ = support._materialize_crop(
                source_batches[sample],
                record,
                partial_ignore_margin_um=(
                    args.partial_ignore_margin_um
                ),
            )
            crop = support._move_crop_to_cuda(crop_cpu)

            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()

            result = _forward_crop(
                model=model,
                crop=crop,
                rag_criterion=rag_criterion,
                amp_dtype=amp_dtype,
                training=True,
            )

            selected = _select_edges(
                result=result,
                max_edges_per_class=args.max_edges_per_class,
                separator_negative_fraction=(
                    args.separator_negative_fraction
                ),
                good_crop_min_base_probability=(
                    args.good_crop_min_base_probability
                ),
                loss_cfg=loss_cfg,
            )

            if (
                selected["positive_indices"].numel() == 0
                and selected["negative_indices"].numel() == 0
            ):
                skipped += 1
                consecutive_skips += 1
                progress.write(
                    f"[skip] {sample} idx={manifest_index} "
                    "no supervised selected edge"
                )
                del result, selected, crop, crop_cpu
                torch.cuda.empty_cache()

                if consecutive_skips >= 128:
                    raise RuntimeError(
                        "128 consecutive crop attempts produced no selected "
                        "supervised edge; aborting."
                    )
                continue

            consecutive_skips = 0

            edge_bce, positive_bce, negative_bce = (
                _balanced_edge_bce(
                    result["final_logits"],
                    selected,
                )
            )

            auxiliary = _separator_barrier_auxiliary(
                result["rag"],
                result["targets"],
                loss_cfg,
                neutral_probability=args.merge_threshold,
            )
            auxiliary_loss = (
                args.semantic_weight
                * auxiliary["separator_barrier_semantic"]
                + args.margin_weight
                * auxiliary["separator_barrier_margin"]
            )

            (
                preservation_loss,
                positive_correction,
                preserve_mask,
            ) = _positive_preservation(
                result=result,
                merge_threshold=args.merge_threshold,
            )

            total_loss = (
                edge_bce
                + auxiliary_loss
                + args.positive_preservation_weight
                * preservation_loss
                + args.positive_barrier_weight
                * positive_correction
            )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = _gradient_norm(
                trainable_parameters
            )
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )

            if not math.isfinite(grad_norm):
                raise RuntimeError(
                    f"Non-finite barrier gradient at step {global_step + 1}"
                )

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress.update(1)

            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started

            targets = result["targets"]
            valid = targets.valid.bool()
            negative = valid & (targets.target <= 0.5)
            positive = valid & (targets.target > 0.5)
            strong_negative = (
                negative
                & _strong_separator_mask(result, loss_cfg)
            )
            final_p = result["final_logits"].detach().sigmoid()

            def mean(mask, values):
                if not bool(mask.any()):
                    return 0.0
                return float(
                    values[mask].detach().float().mean().cpu()
                )

            history_row = {
                "step": global_step,
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "sample": sample,
                "manifest_index": manifest_index,
                "provenance": provenance,
                "loss": float(total_loss.detach().float().cpu()),
                "edge_bce": float(edge_bce.detach().float().cpu()),
                "positive_bce": float(
                    positive_bce.detach().float().cpu()
                ),
                "negative_bce": float(
                    negative_bce.detach().float().cpu()
                ),
                "separator_semantic": float(
                    auxiliary[
                        "separator_barrier_semantic"
                    ].detach().float().cpu()
                ),
                "separator_margin": float(
                    auxiliary[
                        "separator_barrier_margin"
                    ].detach().float().cpu()
                ),
                "strong_negative_count": int(
                    strong_negative.sum().item()
                ),
                "strong_false_merge_count": int(
                    (
                        strong_negative
                        & (final_p >= args.merge_threshold)
                    ).sum().item()
                ),
                "mean_barrier_strong_negative": mean(
                    strong_negative,
                    result["correction"],
                ),
                "mean_barrier_positive": mean(
                    positive,
                    result["correction"],
                ),
                "mean_p_strong_negative": mean(
                    strong_negative,
                    final_p,
                ),
                "grad_norm": grad_norm,
                "step_seconds": step_seconds,
                "peak_vram_gib": (
                    torch.cuda.max_memory_allocated()
                    / 2**30
                ),
                "selected_positive": int(
                    selected["positive_indices"].numel()
                ),
                "selected_negative": int(
                    selected["negative_indices"].numel()
                ),
                "selected_separator_priority": int(
                    selected[
                        "priority_negative_indices"
                    ].numel()
                ),
            }
            _append_jsonl(history_path, history_row)

            progress.set_postfix(
                {
                    "loss": f"{history_row['loss']:.3f}",
                    "SFM": history_row[
                        "strong_false_merge_count"
                    ],
                    "bar-": f"{history_row['mean_barrier_strong_negative']:.3f}",
                    "bar+": f"{history_row['mean_barrier_positive']:.3f}",
                    "src": provenance[0].upper(),
                },
                refresh=False,
            )

            should_validate = (
                global_step % args.validation_every == 0
                or global_step in {100, 200}
                or global_step == args.max_steps
            )

            if should_validate:
                last_validation = _evaluate_jobs(
                    support=support,
                    model=model,
                    source_batches=source_batches,
                    splits=splits,
                    jobs=validation_jobs,
                    rag_criterion=rag_criterion,
                    loss_cfg=loss_cfg,
                    amp_dtype=amp_dtype,
                    partial_ignore_margin_um=(
                        args.partial_ignore_margin_um
                    ),
                    good_crop_min_base_probability=(
                        args.good_crop_min_base_probability
                    ),
                    merge_threshold=args.merge_threshold,
                    description=f"Inv26 heldout s{global_step}",
                )
                last_target_audit = _evaluate_jobs(
                    support=support,
                    model=model,
                    source_batches=source_batches,
                    splits=splits,
                    jobs=good_jobs,
                    rag_criterion=rag_criterion,
                    loss_cfg=loss_cfg,
                    amp_dtype=amp_dtype,
                    partial_ignore_margin_um=(
                        args.partial_ignore_margin_um
                    ),
                    good_crop_min_base_probability=(
                        args.good_crop_min_base_probability
                    ),
                    merge_threshold=args.merge_threshold,
                    description=f"Inv26 target s{global_step}",
                )

                heldout_candidate = last_validation["candidate"]
                target_candidate = last_target_audit["candidate"]

                last_decision = _early_stop_decision(
                    step=global_step,
                    heldout_baseline=heldout_baseline,
                    heldout_candidate=heldout_candidate,
                    target_baseline=target_baseline,
                    target_candidate=target_candidate,
                    args=args,
                )

                validation_row = {
                    "step": global_step,
                    "heldout": last_validation,
                    "target_audit": last_target_audit,
                    "early_stop_decision": last_decision,
                }
                _append_jsonl(
                    validation_path,
                    validation_row,
                )

                progress.write(
                    "[validation] "
                    f"step={global_step} "
                    f"heldout FM={heldout_candidate['false_merge_count']}/"
                    f"{heldout_candidate['negative_edge_count']} "
                    f"PA={heldout_candidate['positive_accept_rate']:.4f} | "
                    f"target bad={target_candidate['bad_strong_negative_count']}/"
                    f"{target_candidate['strong_negative_count']} "
                    f"hard={target_candidate['hard_strong_negative_count']} "
                    f"meanP={target_candidate['mean_strong_negative_probability']:.4f} "
                    f"bar-={target_candidate['mean_correction_strong_negative']:.4f} "
                    f"bar+={heldout_candidate['mean_correction_positive']:.4f}"
                )

                positive_floor_ok = (
                    heldout_candidate["positive_accept_rate"]
                    >= heldout_baseline["positive_accept_rate"]
                    - args.early_stop_max_positive_accept_drop
                )
                rank = (
                    int(not positive_floor_ok),
                    target_candidate["bad_strong_negative_rate"],
                    target_candidate["hard_strong_negative_rate"],
                    heldout_candidate["false_merge_rate"],
                    heldout_candidate["bce"],
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_step = global_step
                    _save_checkpoint(
                        path=best_path,
                        model=model,
                        model_cfg=model_cfg,
                        optimizer=optimizer,
                        scaler=scaler,
                        global_step=global_step,
                        source_checkpoint=source_checkpoint,
                        mining_summary=mining_summary,
                        heldout_baseline=heldout_baseline,
                        target_baseline=target_baseline,
                        validation=last_validation,
                        target_audit=last_target_audit,
                        decision=last_decision,
                        config_payload=config_payload,
                    )
                    progress.write(
                        f"[best] step={global_step} -> {best_path}"
                    )

            should_checkpoint = (
                global_step % args.checkpoint_every == 0
                or global_step in {100, 200}
                or global_step == args.max_steps
            )

            checkpoint_path = None
            if should_checkpoint:
                checkpoint_path = (
                    recovery_dir
                    / f"checkpoint_step_{global_step:06d}.pt"
                )
                _save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    model_cfg=model_cfg,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    source_checkpoint=source_checkpoint,
                    mining_summary=mining_summary,
                    heldout_baseline=heldout_baseline,
                    target_baseline=target_baseline,
                    validation=last_validation,
                    target_audit=last_target_audit,
                    decision=last_decision,
                    config_payload=config_payload,
                )
                progress.write(
                    f"[checkpoint] {checkpoint_path}"
                )

            if (
                not args.disable_early_stop
                and last_decision is not None
                and last_decision.get(
                    "decision_checkpoint",
                    False,
                )
                and last_decision.get("stop", False)
            ):
                # The checkpoint has already been saved above.
                if checkpoint_path is None:
                    raise RuntimeError(
                        "Early-stop decision reached without checkpoint save"
                    )

                early_stopped = True
                early_stop_reason = "; ".join(
                    last_decision.get("reasons", [])
                )
                decision_record = {
                    "step": global_step,
                    "checkpoint": str(checkpoint_path),
                    **last_decision,
                }
                _append_jsonl(
                    decision_path,
                    decision_record,
                )
                progress.write(
                    "[EARLY STOP] "
                    f"step={global_step}: {early_stop_reason}"
                )
                break

            del (
                result,
                selected,
                crop,
                crop_cpu,
                total_loss,
                edge_bce,
            )
            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

    except BaseException as exc:
        progress.close()
        failure = {
            "status": "failed",
            "global_step": global_step,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
        }
        _atomic_json(
            run_dir / "failure.json",
            failure,
        )
        raise

    elapsed = time.perf_counter() - started

    status = (
        "early_stopped"
        if early_stopped
        else "success"
    )
    summary = {
        "status": status,
        "global_step": global_step,
        "max_steps": args.max_steps,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "elapsed_seconds": elapsed,
        "elapsed_human": _duration(elapsed),
        "skipped_crops": skipped,
        "source_checkpoint": str(source_checkpoint),
        "best_step": best_step,
        "best_checkpoint": (
            str(best_path)
            if best_path.is_file()
            else None
        ),
        "recovery_dir": str(recovery_dir),
        "run_dir": str(run_dir),
        "mining": mining_summary,
        "heldout_baseline": heldout_baseline,
        "target_baseline": target_baseline,
        "last_validation": last_validation,
        "last_target_audit": last_target_audit,
        "last_early_stop_decision": last_decision,
    }
    _atomic_json(
        run_dir / "summary.json",
        summary,
    )

    print("=" * 124, flush=True)
    print("INVESTIGATION 26 COMPLETE", flush=True)
    print("=" * 124, flush=True)
    print(f"Status              : {status}", flush=True)
    print(f"Optimizer steps     : {global_step}", flush=True)
    print(f"Good crops          : {len(good_jobs)}", flush=True)
    print(
        f"Hard-good crops     : {mining['hard_good_crop_count']}",
        flush=True,
    )
    print(f"Skipped crops       : {skipped}", flush=True)
    print(f"Elapsed             : {_duration(elapsed)}", flush=True)
    print(f"Best step           : {best_step}", flush=True)
    print(
        f"Best checkpoint     : {summary['best_checkpoint']}",
        flush=True,
    )
    if early_stopped:
        print(
            f"Early-stop reason   : {early_stop_reason}",
            flush=True,
        )
    print(f"Recovery dir        : {recovery_dir}", flush=True)
    print(f"Run summary         : {run_dir / 'summary.json'}", flush=True)
    print("=" * 124, flush=True)

    return summary


# ======================================================================================
# CLI
# ======================================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the production STIR-Net separator-aware RAG barrier for a "
            "300-step value probe with crop-mining and early-stop safety gates."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Starting h100 checkpoint/file/directory. An Investigation-26 "
            "checkpoint resumes exactly."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="drosophila_12_separator_barrier_prod_300",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--validation-crops-per-sample",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--samples",
        default=DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--data-dir",
        default="external/NIS3D/NIS3D",
    )
    parser.add_argument(
        "--spacing-xyz",
        default=DEFAULT_SPACING_XYZ,
    )
    parser.add_argument(
        "--crop-shape-zyx",
        default="32,192,192",
    )
    parser.add_argument(
        "--confidence-ignore-margin-um",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--partial-ignore-margin-um",
        type=float,
        default=1.0,
    )

    # Mining / sampling.
    parser.add_argument(
        "--min-good-crops",
        type=int,
        default=15,
        help=(
            "Abort before optimization if fewer distinct useful crops exist."
        ),
    )
    parser.add_argument(
        "--max-good-crops",
        type=int,
        default=64,
        help=(
            "Stop mining once this many good crops are found. Set 0 to scan all."
        ),
    )
    parser.add_argument(
        "--good-crop-min-base-probability",
        type=float,
        default=0.50,
        help=(
            "A good crop needs a strong-separator GT-negative edge with frozen "
            "base p_merge at least this large. Production-hard p>=0.845 edges "
            "are tracked/prioritized separately."
        ),
    )
    parser.add_argument(
        "--good-crop-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--separator-negative-fraction",
        type=float,
        default=0.75,
        help=(
            "Fraction of selected negative edge quota preferentially reserved "
            "for strong-separator wrong/ambiguous negatives."
        ),
    )
    parser.add_argument(
        "--max-edges-per-class",
        type=int,
        default=64,
    )

    # Optimization.
    parser.add_argument(
        "--barrier-lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--positive-preservation-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--positive-barrier-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.845,
    )

    # Optional experiment override of current production separator target rules.
    parser.add_argument(
        "--strong-separator-mean-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--strong-separator-max-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--strong-separator-coverage70-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--signed-margin",
        type=float,
        default=None,
    )

    # Early stop.
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
    )
    parser.add_argument(
        "--early-stop-step100-min-bad-reduction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--early-stop-step200-min-bad-reduction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--early-stop-step100-min-probability-drop",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--early-stop-step200-min-probability-drop",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--early-stop-max-positive-accept-drop",
        type=float,
        default=0.02,
        help=(
            "Absolute held-out positive-acceptance drop allowed relative to "
            "the frozen h100 baseline."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=260826,
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = _training_impl(args)
    print(
        json.dumps(
            _jsonable(summary),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
