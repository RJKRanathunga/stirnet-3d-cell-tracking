from __future__ import annotations

"""
STIR-Net Experiment 30: cost-bounded overfit + first-failure oracle diagnosis.

Scientific question
-------------------
After a fair but cost-conscious cloud overfit, where is the first supported
failure in the current STIR-Net V1 causal chain?

This script is intentionally not a general trainer. It targets the current
BlastoSPIM first-overfit scene, the reduced STIR-Net debugging architecture,
and the cloud runtime profile.

Important time-budget behavior
------------------------------
The caller should also enforce a Modal Function timeout of 3600 seconds.

Internally, this script defaults to 3480 seconds. It stops starting new GPU
work before the internal deadline so scalar statistics can be flushed. If the
time budget is exhausted, it writes only scalar/JSON/CSV statistics and exits.
It does NOT create a new tensor/checkpoint artifact on the timeout path.

Normal completed stages may create model checkpoints. Existing checkpoints are
not deleted if a later timeout occurs.
"""

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from learned.stirnet import StirNet
from learned.stirnet.debugging.acceptance.first_overfit import (
    _reduced_config,
    build_real_batch,
)
from learned.stirnet.model.losses import local_matched_mask_losses
from learned.stirnet.model.matcher import target_ids
from learned.stirnet.model.query_builder import QUERY_SPATIAL_PROPOSAL
from learned.stirnet.model.runtime_profiles import (
    apply_runtime_profile,
    describe_runtime_profile,
)
from learned.stirnet.training.checkpoint import load_checkpoint, save_checkpoint
from learned.stirnet.training.trainer import (
    Trainer,
    model_forward_from_batch,
    move_batch_to_device,
)


SEED_DEFAULT = 40266
SOURCE_ID = 9
EXIST_THRESHOLD = 0.50
SOURCE_NEIGHBORHOOD_DREF = 1.25

EXPECTED_CURRENT_CELLS = 36
EXPECTED_GT_CELLS = 33
EXPECTED_TEMPORAL_TRACKLETS = 52
EXPECTED_REQUIRED_QUERIES = 140

# The Modal wrapper has the true 3600 s hard timeout. This smaller internal
# cap leaves time to flush statistics and commit the result Volume.
DEFAULT_INTERNAL_HARD_LIMIT_SECONDS = 3480
DEFAULT_FINALIZATION_RESERVE_SECONDS = 90

# Cost-aware defaults. The L40S joint benchmark is ~5 s/step, so these maxima
# leave room for diagnostics/oracle work inside the one-hour envelope.
DEFAULT_QUERY_MIN_STEPS = 50
DEFAULT_QUERY_MAX_STEPS = 75
DEFAULT_LOCAL_MIN_STEPS = 150
DEFAULT_LOCAL_MAX_STEPS = 250
DEFAULT_JOINT_MIN_STEPS = 125
DEFAULT_JOINT_MAX_STEPS = 200

QUERY_EVAL_EVERY = 25
LOCAL_EVAL_EVERY = 25
JOINT_EVAL_EVERY = 25

# Local/native target used as an early-stop success threshold, not as a claim
# about final competition quality.
LOCAL_TARGET_MEAN_DICE = 0.85
LOCAL_TARGET_MIN_DICE = 0.60


class TimeBudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    hard_seconds: float
    finalization_reserve_seconds: float = DEFAULT_FINALIZATION_RESERVE_SECONDS
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.hard_seconds - self.elapsed)

    def require_gpu_window(
        self,
        label: str,
        estimated_seconds: float = 0.0,
        extra_reserve_seconds: float = 0.0,
    ) -> None:
        required = (
            self.finalization_reserve_seconds
            + max(0.0, float(estimated_seconds))
            + max(0.0, float(extra_reserve_seconds))
        )
        if self.remaining <= required:
            raise TimeBudgetExceeded(
                f"Time budget reached before {label}: "
                f"remaining={self.remaining:.1f}s required={required:.1f}s"
            )


@dataclass
class ExperimentState:
    run_dir: Path
    budget: Budget
    training_rows: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    oracle: dict[str, Any] = field(default_factory=dict)
    last_stage: str = "startup"
    last_global_step: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError(
                "Experiment statistics must be scalar. "
                f"Refusing to serialize tensor shape={tuple(value.shape)}."
            )
        return value.detach().cpu().item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
        handle.flush()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = [jsonable(dict(row)) for row in rows]
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                flat[key] = value
            writer.writerow(flat)


def log(state: ExperimentState, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with (state.run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def record_training(state: ExperimentState, row: dict[str, Any]) -> None:
    row = {"time": utc_now(), **row}
    state.training_rows.append(row)
    append_jsonl(state.run_dir / "training_metrics.jsonl", row)
    state.last_stage = str(row.get("stage", state.last_stage))
    state.last_global_step = int(row.get("global_step", state.last_global_step))


def record_diagnostic(state: ExperimentState, row: dict[str, Any]) -> None:
    row = {"time": utc_now(), **row}
    state.diagnostics.append(row)
    append_jsonl(state.run_dir / "diagnostics.jsonl", row)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def final_scalar_flush(
    state: ExperimentState,
    *,
    status: str,
    reason: str | None = None,
    error: BaseException | None = None,
) -> None:
    """
    Persist statistics only.

    This function intentionally never calls torch.save/save_checkpoint.
    """
    payload: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "finished_at": utc_now(),
        "elapsed_seconds": state.budget.elapsed,
        "remaining_seconds": state.budget.remaining,
        "last_stage": state.last_stage,
        "last_global_step": state.last_global_step,
        "training_rows": len(state.training_rows),
        "diagnostic_rows": len(state.diagnostics),
        "stages": state.stages,
        "oracle": state.oracle,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
        payload["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    atomic_json(state.run_dir / "terminal_summary.json", payload)
    write_csv(state.run_dir / "training_metrics.csv", state.training_rows)
    write_csv(state.run_dir / "diagnostics.csv", state.diagnostics)
    atomic_json(
        state.run_dir / "stage_summary.json",
        {"stages": state.stages},
    )
    atomic_json(
        state.run_dir / "oracle_results.json",
        state.oracle,
    )


def model_only_checkpoint(
    state: ExperimentState,
    path: Path,
    *,
    trainer: Trainer,
    cfg,
    label: str,
    extra: dict[str, Any] | None = None,
) -> bool:
    """
    Save only when there is enough time margin.

    We deliberately skip tensor writes close to the hard time limit.
    """
    if state.budget.remaining < 180:
        log(
            state,
            f"Skipping tensor checkpoint {path.name}: only "
            f"{state.budget.remaining:.1f}s remain.",
        )
        return False
    save_checkpoint(
        path,
        model=trainer.model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        step=trainer.global_step,
        config=cfg,
        extra={"label": label, **(extra or {})},
    )
    return True


def full_checkpoint(
    state: ExperimentState,
    path: Path,
    *,
    trainer: Trainer,
    cfg,
    label: str,
    extra: dict[str, Any] | None = None,
) -> bool:
    if state.budget.remaining < 180:
        log(
            state,
            f"Skipping full checkpoint {path.name}: only "
            f"{state.budget.remaining:.1f}s remain.",
        )
        return False
    save_checkpoint(
        path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.global_step,
        config=cfg,
        extra={"label": label, **(extra or {})},
    )
    return True


def validate_scene(sample: dict[str, Any]) -> None:
    failures = []
    checks = {
        "current_count": EXPECTED_CURRENT_CELLS,
        "target_count": EXPECTED_GT_CELLS,
        "temporal_tracklets": EXPECTED_TEMPORAL_TRACKLETS,
        "required_queries": EXPECTED_REQUIRED_QUERIES,
    }
    for key, expected in checks.items():
        actual = int(sample[key])
        if actual != expected:
            failures.append(f"{key}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError(
            "Prepared first-overfit sample no longer matches the benchmark "
            "reference scene:\n  " + "\n  ".join(failures)
        )


def discover_warm_start(runs_root: Path) -> Path | None:
    preferred_names = (
        "checkpoint_joint_best_native.pt",
        "checkpoint_joint_best.pt",
        "checkpoint_evening_final.pt",
        "checkpoint_after_local_mask.pt",
        "checkpoint_after_query.pt",
        "checkpoint_best_spatial_query.pt",
    )
    for name in preferred_names:
        candidates = list(runs_root.rglob(name)) if runs_root.exists() else []
        candidates = [p for p in candidates if p.is_file()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def source9_context(batch_cpu: dict[str, Any]) -> dict[str, Any]:
    target = batch_cpu["targets"][0]
    labels = torch.as_tensor(target["label_map"]).cpu().numpy().astype(np.int32)
    current = batch_cpu["instance_labels"][0].cpu().numpy().astype(np.int32)

    gt_ids = target_ids(target).detach().cpu().numpy().astype(int)
    gt_centers = torch.as_tensor(target["centers_cellscale"]).detach().cpu().float()

    source_gt_ids = np.unique(labels[current == SOURCE_ID])
    source_gt_ids = source_gt_ids[source_gt_ids > 0].astype(int)
    source_set = set(source_gt_ids.tolist())
    source_rows = torch.tensor(
        [i for i, gt_id in enumerate(gt_ids.tolist()) if int(gt_id) in source_set],
        dtype=torch.long,
    )
    source_centers = gt_centers[source_rows]

    missing_gt_ids: list[int] = []
    for gt_id in gt_ids.tolist():
        if not np.any(current[labels == int(gt_id)] > 0):
            missing_gt_ids.append(int(gt_id))

    noisy_source_ids: list[int] = []
    for source_id in np.unique(current):
        if int(source_id) <= 0:
            continue
        if not np.any(labels[current == int(source_id)] > 0):
            noisy_source_ids.append(int(source_id))

    return {
        "target": target,
        "gt_labels": labels,
        "current_labels": current,
        "all_gt_ids": gt_ids,
        "all_gt_centers": gt_centers,
        "source9_gt_ids": source_gt_ids,
        "source9_gt_rows": source_rows,
        "source9_gt_centers": source_centers,
        "missing_gt_ids": missing_gt_ids,
        "noisy_source_ids": noisy_source_ids,
    }


def valid_proposal_rows(outputs) -> torch.Tensor:
    valid = ~outputs.query_padding_mask[0]
    proposal = outputs.query_types[0] == QUERY_SPATIAL_PROPOSAL
    return torch.nonzero(valid & proposal, as_tuple=False).flatten()


def gt_pairing(
    outputs,
    gt_ids_ordered: np.ndarray,
    gt_centers_ordered: torch.Tensor,
) -> dict[str, Any]:
    qrows = valid_proposal_rows(outputs)
    refs = (
        outputs.query_initial_references_cellscale[0, qrows]
        .detach()
        .float()
        .cpu()
    )
    if len(qrows) == 0 or len(gt_centers_ordered) == 0:
        return {
            "query_rows": torch.empty(0, dtype=torch.long),
            "gt_rows": torch.empty(0, dtype=torch.long),
            "gt_ids": np.asarray([], dtype=int),
            "distances": torch.empty(0),
        }

    distances = torch.cdist(refs, gt_centers_ordered.float().cpu())
    q_np, g_np = linear_sum_assignment(distances.numpy())
    q_local = torch.as_tensor(q_np, dtype=torch.long)
    g_rows = torch.as_tensor(g_np, dtype=torch.long)

    order = torch.argsort(g_rows)
    q_local = q_local[order]
    g_rows = g_rows[order]

    return {
        "query_rows": qrows.detach().cpu()[q_local],
        "gt_rows": g_rows,
        "gt_ids": np.asarray(gt_ids_ordered, dtype=int)[g_rows.numpy()],
        "distances": distances[q_local, g_rows],
    }


def proposal_metrics(
    refs: torch.Tensor,
    gt_centers: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    refs = refs.detach().float().cpu()
    gt_centers = gt_centers.detach().float().cpu()

    if len(gt_centers) == 0:
        return {f"{prefix}_proposal_count": int(len(refs)), f"{prefix}_gt_count": 0}
    if len(refs) == 0:
        return {
            f"{prefix}_proposal_count": 0,
            f"{prefix}_gt_count": int(len(gt_centers)),
            f"{prefix}_recall_0p25": 0.0,
            f"{prefix}_recall_0p5": 0.0,
            f"{prefix}_recall_1p0": 0.0,
            f"{prefix}_nearest_mean": float("inf"),
            f"{prefix}_nearest_max": float("inf"),
        }

    distance = torch.cdist(refs, gt_centers)
    nearest = distance.min(dim=0).values
    return {
        f"{prefix}_proposal_count": int(len(refs)),
        f"{prefix}_gt_count": int(len(gt_centers)),
        f"{prefix}_recall_0p25": float((nearest <= 0.25).float().mean()),
        f"{prefix}_recall_0p5": float((nearest <= 0.50).float().mean()),
        f"{prefix}_recall_1p0": float((nearest <= 1.00).float().mean()),
        f"{prefix}_nearest_mean": float(nearest.mean()),
        f"{prefix}_nearest_max": float(nearest.max()),
    }


def selection_metrics(
    outputs,
    source_centers: torch.Tensor,
    *,
    threshold: float = EXIST_THRESHOLD,
) -> dict[str, Any]:
    qrows = valid_proposal_rows(outputs)
    refs = (
        outputs.query_initial_references_cellscale[0, qrows]
        .detach()
        .float()
        .cpu()
    )
    scores = outputs.exist_logits[0, qrows].sigmoid().detach().float().cpu()

    if len(source_centers) == 0 or len(refs) == 0:
        return {
            "selected": 0,
            "unique_gt_covered": 0,
            "missing_gt": int(len(source_centers)),
            "duplicates": 0,
            "exactly_one_gt": 0,
        }

    distance = torch.cdist(refs, source_centers.float().cpu())
    nearest_distance, nearest_gt = distance.min(dim=1)
    cluster = nearest_distance <= SOURCE_NEIGHBORHOOD_DREF
    selected = cluster & (scores >= threshold)
    assigned = selected & (nearest_distance <= 1.0)
    assigned_gt = nearest_gt[assigned]

    counts = torch.bincount(assigned_gt, minlength=len(source_centers))
    unique = int((counts > 0).sum())
    return {
        "selected": int(selected.sum()),
        "selected_assigned": int(assigned.sum()),
        "unique_gt_covered": unique,
        "missing_gt": int(len(source_centers) - unique),
        "duplicates": int(torch.clamp(counts - 1, min=0).sum()),
        "exactly_one_gt": int((counts == 1).sum()),
        "mean_selected_exist": (
            float(scores[selected].mean()) if bool(selected.any()) else float("nan")
        ),
    }


def eval_local_pairing(
    decoder,
    outputs,
    context: dict[str, Any],
    pairing: dict[str, Any],
    *,
    use_gt_anchors: bool,
) -> dict[str, Any]:
    qrows = pairing["query_rows"]
    grows = pairing["gt_rows"]
    if len(qrows) == 0:
        return {
            "cell_count": 0,
            "hard_dice_mean": float("nan"),
            "hard_dice_min": float("nan"),
            "soft_dice_mean": float("nan"),
            "support_coverage_min": float("nan"),
            "volume_ratio_mean": float("nan"),
        }

    qrows_dev = qrows.to(outputs.query_embeddings.device)
    query_embeddings = outputs.query_embeddings[0, qrows_dev].detach()
    if use_gt_anchors:
        anchors = context["source9_gt_centers"][grows].to(
            outputs.query_embeddings.device
        )
    else:
        anchors = outputs.query_initial_references_cellscale[0, qrows_dev].detach()

    gt_labels = context["gt_labels"]
    rows = []

    for local_index, gt_id in enumerate(pairing["gt_ids"].tolist()):
        with torch.no_grad():
            prediction = decoder.decode_one(
                outputs.d0_features,
                outputs.spatial_inputs,
                outputs.dense_outputs,
                query_embeddings[local_index],
                anchors[local_index],
                outputs.spacing_um[0],
                outputs.dref_um[0],
                batch_index=0,
            )

        if prediction.logits is None:
            raise RuntimeError("Local decoder returned no logits.")

        target_crop = torch.as_tensor(
            gt_labels[prediction.slices] == int(gt_id),
            dtype=torch.bool,
            device=prediction.logits.device,
        )
        support = prediction.support.to(prediction.logits.device)
        probability = prediction.logits.float().sigmoid()
        hard = probability >= 0.5

        target_total = float(np.count_nonzero(gt_labels == int(gt_id)))
        target_inside = float((target_crop & support).sum().detach().cpu())

        supported_probability = probability * support.float()
        supported_hard = hard & support

        soft_intersection = float(
            (supported_probability * target_crop.float()).sum().detach().cpu()
        )
        predicted_soft = float(supported_probability.sum().detach().cpu())
        soft_dice = (2.0 * soft_intersection + 1e-6) / (
            predicted_soft + target_total + 1e-6
        )

        hard_intersection = float(
            (supported_hard & target_crop).sum().detach().cpu()
        )
        predicted_hard = float(supported_hard.sum().detach().cpu())
        hard_dice = (2.0 * hard_intersection + 1e-6) / (
            predicted_hard + target_total + 1e-6
        )

        rows.append(
            {
                "hard_dice": hard_dice,
                "soft_dice": soft_dice,
                "support_coverage": target_inside / max(target_total, 1.0),
                "volume_ratio": predicted_hard / max(target_total, 1.0),
            }
        )

    return {
        "cell_count": len(rows),
        "hard_dice_mean": float(np.mean([r["hard_dice"] for r in rows])),
        "hard_dice_min": float(np.min([r["hard_dice"] for r in rows])),
        "soft_dice_mean": float(np.mean([r["soft_dice"] for r in rows])),
        "support_coverage_min": float(
            np.min([r["support_coverage"] for r in rows])
        ),
        "volume_ratio_mean": float(np.mean([r["volume_ratio"] for r in rows])),
    }


def full_forward(
    state: ExperimentState,
    trainer: Trainer,
    batch_cpu: dict[str, Any],
    *,
    return_debug: bool = True,
):
    state.budget.require_gpu_window("diagnostic forward", estimated_seconds=15)
    trainer.model.eval()
    trainer.criterion.eval()
    gpu_batch = move_batch_to_device(batch_cpu, trainer.device)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model_forward_from_batch(
            trainer.model,
            gpu_batch,
            return_debug=return_debug,
            temporal_memory_ablation="full",
            temporal_routing_ablation="full",
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    del gpu_batch
    record_diagnostic(
        state,
        {
            "stage": "forward",
            "global_step": trainer.global_step,
            "seconds": elapsed,
            "peak_allocated_gib": peak_alloc,
            "peak_reserved_gib": peak_reserved,
            "query_count": int(outputs.exist_logits.shape[1]),
        },
    )
    return outputs


def source9_diagnostic(
    state: ExperimentState,
    trainer: Trainer,
    batch_cpu: dict[str, Any],
    context: dict[str, Any],
    *,
    tag: str,
) -> dict[str, Any]:
    outputs = full_forward(state, trainer, batch_cpu, return_debug=False)
    pairing = gt_pairing(
        outputs,
        context["source9_gt_ids"],
        context["source9_gt_centers"],
    )

    mask_metrics = eval_local_pairing(
        trainer.model.local_mask_decoder,
        outputs,
        context,
        pairing,
        use_gt_anchors=False,
    )

    qrows = pairing["query_rows"].to(outputs.centers_cellscale.device)
    grows = pairing["gt_rows"]
    if len(qrows):
        paired_gt = context["source9_gt_centers"][grows]
        initial = (
            outputs.query_initial_references_cellscale[0, qrows]
            .detach()
            .float()
            .cpu()
        )
        final = outputs.centers_cellscale[0, qrows].detach().float().cpu()
        initial_error = float(
            torch.linalg.vector_norm(initial - paired_gt, dim=-1).mean()
        )
        final_error = float(
            torch.linalg.vector_norm(final - paired_gt, dim=-1).mean()
        )
    else:
        initial_error = float("nan")
        final_error = float("nan")

    proposals = outputs.proposals
    if proposals is not None:
        valid = ~proposals.padding_mask[0]
        refs = proposals.references_cellscale[0, valid].detach().float().cpu()
        coverage = proposal_metrics(
            refs,
            context["source9_gt_centers"],
            "source9",
        )
    else:
        coverage = {}

    selection = selection_metrics(
        outputs,
        context["source9_gt_centers"],
        threshold=EXIST_THRESHOLD,
    )

    summary = {
        "stage": tag,
        "global_step": trainer.global_step,
        **{f"mask_{k}": v for k, v in mask_metrics.items()},
        "initial_center_error_mean_dref": initial_error,
        "final_center_error_mean_dref": final_error,
        **coverage,
        **{f"selection_{k}": v for k, v in selection.items()},
    }

    record_diagnostic(state, summary)
    del outputs
    cleanup_cuda()
    return summary


@dataclass
class PlateauTracker:
    mode: str
    min_delta: float
    patience_evals: int
    best: float | None = None
    no_improve: int = 0

    def update(self, value: float) -> bool:
        if not math.isfinite(value):
            self.no_improve += 1
            return self.no_improve >= self.patience_evals

        if self.best is None:
            self.best = value
            self.no_improve = 0
            return False

        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.no_improve = 0
        else:
            self.no_improve += 1
        return self.no_improve >= self.patience_evals


def run_training_phase(
    state: ExperimentState,
    trainer: Trainer,
    cfg,
    batch_cpu: dict[str, Any],
    context: dict[str, Any],
    *,
    phase_name: str,
    min_steps: int,
    max_steps: int,
    eval_every: int,
    tracker: PlateauTracker,
    diagnostic: bool,
) -> dict[str, Any]:
    successful = 0
    started = time.monotonic()
    step_ema = 6.0
    best_diag_score = -float("inf")
    best_diag: dict[str, Any] | None = None
    stop_reason = "max_steps"

    log(
        state,
        f"START TRAINING {phase_name}: min={min_steps}, max={max_steps}, "
        f"global={trainer.global_step}, curriculum={trainer.curriculum.apply(trainer.global_step).name}",
    )

    while successful < max_steps:
        state.budget.require_gpu_window(
            f"{phase_name} optimizer step",
            estimated_seconds=max(10.0, 1.5 * step_ema),
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        step_started = time.perf_counter()

        metrics = trainer.train_step(batch_cpu)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        step_ema = 0.85 * step_ema + 0.15 * elapsed
        successful += 1

        if not all(math.isfinite(float(v)) for v in metrics.values()):
            raise RuntimeError(
                f"Non-finite training metric in {phase_name} step {successful}: {metrics}"
            )

        sampled = getattr(trainer.criterion, "last_local_sampled_requests", [])
        originals = getattr(
            trainer.criterion, "last_local_original_request_counts", []
        )

        row = {
            "stage": phase_name,
            "curriculum_stage": trainer.curriculum_stage.name,
            "global_step": trainer.global_step,
            "phase_step": successful,
            "seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "local_sampled": len(sampled),
            "local_original": int(sum(originals)) if originals else 0,
            **{k: float(v) for k, v in metrics.items()},
        }
        record_training(state, row)

        if successful == 1 or successful % 10 == 0:
            log(
                state,
                f"{phase_name} n={successful:04d} global={trainer.global_step:04d} "
                f"loss={metrics['loss']:.4f} dice_hi={metrics['dice_hi']:.4f} "
                f"coarse={metrics['dice_coarse']:.4f} center={metrics['center']:.4f} "
                f"local={len(sampled)} sec={elapsed:.2f}",
            )

        if successful % eval_every != 0:
            continue

        if diagnostic:
            diag = source9_diagnostic(
                state,
                trainer,
                batch_cpu,
                context,
                tag=f"{phase_name}_eval",
            )
            score = float(diag["mask_hard_dice_mean"])
            if math.isfinite(score) and score > best_diag_score:
                best_diag_score = score
                best_diag = dict(diag)
                if state.budget.remaining >= 240:
                    model_only_checkpoint(
                        state,
                        state.run_dir / f"checkpoint_{phase_name}_best.pt",
                        trainer=trainer,
                        cfg=cfg,
                        label=f"{phase_name}_best",
                        extra={"diagnostic": diag},
                    )

            plateau = tracker.update(score)

            target_met = (
                phase_name == "local_mask_bootstrap"
                and score >= LOCAL_TARGET_MEAN_DICE
                and float(diag["mask_hard_dice_min"]) >= LOCAL_TARGET_MIN_DICE
            )
            if phase_name == "joint":
                target_met = (
                    score >= LOCAL_TARGET_MEAN_DICE
                    and int(diag.get("selection_missing_gt", 999)) == 0
                    and int(diag.get("selection_duplicates", 999)) <= 1
                    and float(diag.get("source9_recall_0p5", 0.0)) >= 0.999
                )

            if successful >= min_steps and target_met:
                stop_reason = "target_met"
                break
            if successful >= min_steps and plateau:
                stop_reason = "plateau"
                break
        else:
            # Query/temporal refresh uses the scalar training objective. Average
            # the most recent few query rows to avoid reacting to one point.
            phase_rows = [
                r
                for r in state.training_rows
                if r.get("stage") == phase_name
            ]
            recent = phase_rows[-min(5, len(phase_rows)) :]
            score = float(np.mean([r["loss"] for r in recent]))
            plateau = tracker.update(score)
            record_diagnostic(
                state,
                {
                    "stage": f"{phase_name}_eval",
                    "global_step": trainer.global_step,
                    "recent_loss_mean": score,
                },
            )
            if successful >= min_steps and plateau:
                stop_reason = "plateau"
                break

    summary = {
        "phase": phase_name,
        "successful_steps": successful,
        "elapsed_seconds": time.monotonic() - started,
        "stop_reason": stop_reason,
        "global_step": trainer.global_step,
        "best_diagnostic": best_diag,
    }
    state.stages.append(summary)
    log(state, f"DONE TRAINING {phase_name}: {summary}")
    return summary


def centers_to_native_voxels(
    centers_cellscale: torch.Tensor,
    shape: tuple[int, int, int],
    spacing_um: torch.Tensor,
    dref_um: torch.Tensor,
) -> torch.Tensor:
    centers = centers_cellscale.to(device=spacing_um.device, dtype=torch.float32)
    shape_tensor = torch.tensor(
        shape, device=spacing_um.device, dtype=torch.float32
    )
    extent = (shape_tensor - 1.0) * spacing_um.float()
    centers_um = centers * dref_um.float()
    voxels = torch.round(
        (centers_um + 0.5 * extent[None])
        / spacing_um.float()[None].clamp_min(1e-8)
    ).long()
    return torch.minimum(
        torch.maximum(voxels, torch.zeros_like(voxels)),
        shape_tensor.long()[None] - 1,
    )


def make_spatial_only_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out = dict(batch)

    node_keys = {
        "graph_x",
        "tracklet_id",
        "node_instance_grid",
        "node_history_valid",
        "node_observed_ref_um",
        "node_time_offset",
        "node_ids",
        "node_event_features",
    }
    edge_index_keys = {"graph_edge_index", "accepted_association_edge_index"}
    edge_attr_keys = {"graph_edge_attr", "accepted_association_edge_attr"}
    tracklet_keys = {
        "temporal_ref_um",
        "temporal_status",
        "temporal_batch",
        "history_support",
        "history_support_valid",
        "history_support_dt",
        "history_support_center_um",
        "history_support_extent_um",
        "best_current_component_id",
        "best_component_overlap",
        "second_best_component_overlap",
    }
    hyp_index_keys = {"hypothesis_edge_index"}
    hyp_attr_keys = {"hypothesis_edge_attr"}

    for key in node_keys | edge_attr_keys | tracklet_keys | hyp_attr_keys:
        value = out.get(key)
        if torch.is_tensor(value):
            out[key] = value[:0]

    for key in edge_index_keys | hyp_index_keys:
        value = out.get(key)
        if torch.is_tensor(value):
            out[key] = value[:, :0]

    return out


def capacity_probe(
    state: ExperimentState,
    trainer: Trainer,
    outputs,
    context: dict[str, Any],
    *,
    max_steps: int = 60,
) -> dict[str, Any]:
    """
    Conditional only: run if the trained actual-query + GT-anchor gate fails.
    """
    if state.budget.remaining < 300:
        return {"status": "skipped_low_time"}

    decoder = copy.deepcopy(trainer.model.local_mask_decoder).to(trainer.device)
    decoder.train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=2e-3, weight_decay=1e-4)

    gt_ids = context["source9_gt_ids"].tolist()
    gt_anchors = context["source9_gt_centers"].to(trainer.device)
    common_query = torch.zeros(
        trainer.cfg.decoder.d_model,
        device=trainer.device,
        dtype=torch.float32,
    )

    best_mean = -1.0
    best_min = -1.0
    completed = 0

    for step in range(max_steps):
        if state.budget.remaining < 180:
            break
        chosen = int(torch.randint(0, len(gt_ids), (1,), device=trainer.device).item())
        gt_id = int(gt_ids[chosen])

        optimizer.zero_grad(set_to_none=True)
        prediction = decoder.decode_one(
            outputs.d0_features,
            outputs.spatial_inputs,
            outputs.dense_outputs,
            common_query,
            gt_anchors[chosen],
            outputs.spacing_um[0],
            outputs.dref_um[0],
            batch_index=0,
        )
        if prediction.logits is None:
            raise RuntimeError("Capacity probe decoder returned no logits.")

        target_crop = torch.as_tensor(
            context["gt_labels"][prediction.slices] == gt_id,
            dtype=torch.float32,
            device=trainer.device,
        )
        support = prediction.support.to(trainer.device)
        target_crop = target_crop * support.float()

        dice, focal = local_matched_mask_losses(
            prediction.logits[None],
            target_crop[None],
            support[None],
            alpha=trainer.cfg.losses.mask_focal_alpha_pos,
            gamma=trainer.cfg.losses.mask_focal_gamma,
        )
        loss = trainer.cfg.losses.dice_hi * dice + trainer.cfg.losses.focal_hi * focal
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        completed += 1

        if completed % 15 == 0:
            # Evaluate all nine with the same neutral query and GT anchors.
            rows = []
            decoder.eval()
            for index, gt_id_eval in enumerate(gt_ids):
                with torch.no_grad():
                    pred = decoder.decode_one(
                        outputs.d0_features,
                        outputs.spatial_inputs,
                        outputs.dense_outputs,
                        common_query,
                        gt_anchors[index],
                        outputs.spacing_um[0],
                        outputs.dref_um[0],
                        batch_index=0,
                    )
                target = torch.as_tensor(
                    context["gt_labels"][pred.slices] == int(gt_id_eval),
                    dtype=torch.bool,
                    device=trainer.device,
                )
                support_eval = pred.support.to(trainer.device)
                hard = (pred.logits.float().sigmoid() >= 0.5) & support_eval
                gt_total = float(
                    np.count_nonzero(context["gt_labels"] == int(gt_id_eval))
                )
                inter = float((hard & target).sum().detach().cpu())
                pred_total = float(hard.sum().detach().cpu())
                rows.append(
                    (2.0 * inter + 1e-6) / (pred_total + gt_total + 1e-6)
                )
            decoder.train()
            best_mean = max(best_mean, float(np.mean(rows)))
            best_min = max(best_min, float(np.min(rows)))
            if best_mean >= LOCAL_TARGET_MEAN_DICE and best_min >= LOCAL_TARGET_MIN_DICE:
                break

    del decoder
    cleanup_cuda()
    return {
        "status": "completed",
        "steps": completed,
        "best_hard_dice_mean": best_mean,
        "best_hard_dice_min": best_min,
    }


def source_crop_slices(
    context: dict[str, Any],
    spacing_um: np.ndarray,
    dref_um: float,
    margin_dref: float = 1.5,
):
    current = context["current_labels"]
    gt = context["gt_labels"]
    coords = np.argwhere(current == SOURCE_ID)
    if len(coords) == 0:
        coords = np.argwhere(np.isin(gt, context["source9_gt_ids"]))

    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    margin = np.ceil(margin_dref * dref_um / spacing_um).astype(int)
    low = np.maximum(0, low - margin)
    high = np.minimum(np.asarray(gt.shape), high + margin)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def compose_fragmentation_stats(
    state: ExperimentState,
    trainer: Trainer,
    outputs,
    context: dict[str, Any],
    qrows: torch.Tensor,
    *,
    use_existence: bool,
) -> dict[str, Any]:
    """
    Compute composition statistics in memory only. No label-map tensor is saved.
    """
    spacing = outputs.spacing_um[0].detach().cpu().numpy().astype(float)
    dref = float(outputs.dref_um[0].detach().cpu())
    crop = source_crop_slices(context, spacing, dref)

    crop_shape = tuple(sl.stop - sl.start for sl in crop)
    best_score = torch.full(
        crop_shape, -torch.inf, device=trainer.device, dtype=torch.float32
    )
    winner = torch.zeros(crop_shape, device=trainer.device, dtype=torch.int32)

    for destination, qrow in enumerate(qrows.tolist(), start=1):
        state.budget.require_gpu_window(
            "source9 streamed composition",
            estimated_seconds=5,
        )
        selected = torch.tensor([qrow], device=trainer.device, dtype=torch.long)
        with torch.no_grad():
            rendered = trainer.model.render_masks(outputs, [selected])[0][0]
        probability = rendered[crop].float().sigmoid()
        valid = probability >= 0.5

        if use_existence:
            exist = float(outputs.exist_logits[0, qrow].sigmoid().detach().cpu())
            score = probability * exist
        else:
            score = probability

        score = score.masked_fill(~valid, -torch.inf)
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        winner = torch.where(
            better, torch.full_like(winner, destination), winner
        )
        del rendered, probability, score, better

    winner[~torch.isfinite(best_score)] = 0
    predicted = winner.detach().cpu().numpy()
    gt_crop = context["gt_labels"][crop]

    fragment_counts = []
    best_dices = []
    missing = 0

    for gt_id in context["source9_gt_ids"].tolist():
        gt_mask = gt_crop == int(gt_id)
        gt_count = int(gt_mask.sum())
        pred_ids = np.unique(predicted[gt_mask])
        pred_ids = pred_ids[pred_ids > 0]

        significant = 0
        best_dice = 0.0
        for pred_id in pred_ids.tolist():
            pred_mask = predicted == int(pred_id)
            intersection = int(np.count_nonzero(gt_mask & pred_mask))
            if intersection >= max(3, int(0.05 * max(gt_count, 1))):
                significant += 1
            dice = 2.0 * intersection / max(
                int(gt_mask.sum()) + int(pred_mask.sum()), 1
            )
            best_dice = max(best_dice, dice)

        if significant == 0:
            missing += 1
        fragment_counts.append(significant)
        best_dices.append(best_dice)

    return {
        "mean_best_composed_dice": float(np.mean(best_dices)),
        "missing_gt": int(missing),
        "gt_with_multiple_fragments": int(
            np.count_nonzero(np.asarray(fragment_counts) > 1)
        ),
        "mean_fragments_per_gt": float(np.mean(fragment_counts)),
        "predicted_instance_count": int(np.count_nonzero(np.unique(predicted) > 0)),
    }


def run_oracle_first_failure(
    state: ExperimentState,
    trainer: Trainer,
    batch_cpu: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {"mode": "first_failure"}

    state.budget.require_gpu_window("oracle full forward", estimated_seconds=20)
    outputs = full_forward(state, trainer, batch_cpu, return_debug=True)
    pairing = gt_pairing(
        outputs,
        context["source9_gt_ids"],
        context["source9_gt_centers"],
    )

    # G1: trained actual query + perfect GT anchor.
    g1 = eval_local_pairing(
        trainer.model.local_mask_decoder,
        outputs,
        context,
        pairing,
        use_gt_anchors=True,
    )
    results["G1_trained_query_gt_anchor"] = g1

    if (
        g1["hard_dice_mean"] < 0.70
        or g1["hard_dice_min"] < 0.40
    ):
        results["first_failure"] = "G1_trained_query_gt_anchor"
        results["G1b_capacity_probe"] = capacity_probe(
            state, trainer, outputs, context
        )
        return results

    # G2: predicted immutable anchor, same GT assignment.
    g2 = eval_local_pairing(
        trainer.model.local_mask_decoder,
        outputs,
        context,
        pairing,
        use_gt_anchors=False,
    )

    qrows = pairing["query_rows"].to(outputs.centers_cellscale.device)
    grows = pairing["gt_rows"]
    paired_gt = context["source9_gt_centers"][grows]
    initial = (
        outputs.query_initial_references_cellscale[0, qrows]
        .detach()
        .float()
        .cpu()
    )
    final = outputs.centers_cellscale[0, qrows].detach().float().cpu()
    g2["initial_center_error_mean_dref"] = float(
        torch.linalg.vector_norm(initial - paired_gt, dim=-1).mean()
    )
    g2["final_center_error_mean_dref"] = float(
        torch.linalg.vector_norm(final - paired_gt, dim=-1).mean()
    )
    results["G2_predicted_anchor"] = g2

    if (
        g2["hard_dice_mean"] + 0.10 < g1["hard_dice_mean"]
        or g2["support_coverage_min"] < 0.98
    ):
        results["first_failure"] = "G2_predicted_anchor"
        return results

    # G3: proposal pool recall.
    proposals = outputs.proposals
    if proposals is None:
        results["first_failure"] = "G3_missing_proposal_state"
        return results

    valid = ~proposals.padding_mask[0]
    learned = valid & ~proposals.fallback_mask[0]
    all_refs = (
        proposals.references_cellscale[0, valid].detach().float().cpu()
    )
    learned_refs = (
        proposals.references_cellscale[0, learned].detach().float().cpu()
    )

    g3 = {}
    g3.update(
        proposal_metrics(
            all_refs,
            context["all_gt_centers"],
            "all_gt_all",
        )
    )
    g3.update(
        proposal_metrics(
            learned_refs,
            context["all_gt_centers"],
            "all_gt_learned",
        )
    )
    g3.update(
        proposal_metrics(
            all_refs,
            context["source9_gt_centers"],
            "source9_all",
        )
    )
    g3.update(
        proposal_metrics(
            learned_refs,
            context["source9_gt_centers"],
            "source9_learned",
        )
    )
    results["G3_proposal_pool"] = g3

    if (
        float(g3["source9_all_recall_0p5"]) < 0.999
        or float(g3["all_gt_all_recall_0p5"]) < 0.90
    ):
        results["first_failure"] = "G3_proposal_pool"
        return results

    # G4: one-hypothesis-per-cell selection.
    g4 = selection_metrics(
        outputs,
        context["source9_gt_centers"],
        threshold=EXIST_THRESHOLD,
    )
    results["G4_selection"] = g4

    if (
        int(g4["missing_gt"]) > 0
        or int(g4["duplicates"]) > 1
    ):
        results["first_failure"] = "G4_selection"
        return results

    # G5: oracle score field through the real NMS vs learned peaks.
    score_logits = outputs.dense_outputs["proposal_score_logits"]
    shape = tuple(int(v) for v in score_logits.shape[-3:])
    gt_centers_dev = context["all_gt_centers"].to(trainer.device)
    gt_voxels = centers_to_native_voxels(
        gt_centers_dev,
        shape,
        outputs.spacing_um[0],
        outputs.dref_um[0],
    )

    oracle_logits = torch.full(
        shape, -20.0, device=trainer.device, dtype=torch.float32
    )
    oracle_logits[
        gt_voxels[:, 0],
        gt_voxels[:, 1],
        gt_voxels[:, 2],
    ] = 20.0

    oracle_voxels = trainer.model.spatial_proposal_generator._learned_centers(
        oracle_logits,
        outputs.spacing_um[0],
        outputs.dref_um[0],
        None,
        int(trainer.cfg.proposals.max_proposals),
    )
    oracle_refs_um = (
        trainer.model.spatial_proposal_generator._relative_voxel_centers_um(
            oracle_voxels,
            shape,
            outputs.spacing_um[0],
        )
    )
    oracle_refs = (
        oracle_refs_um / outputs.dref_um[0].float().clamp_min(1e-8)
    ).detach().cpu()

    g5 = {}
    g5.update(
        proposal_metrics(
            oracle_refs,
            context["all_gt_centers"],
            "oracle_nms",
        )
    )
    g5.update(
        proposal_metrics(
            learned_refs,
            context["all_gt_centers"],
            "learned_peaks",
        )
    )
    results["G5_score_field_vs_nms"] = g5

    if float(g5["oracle_nms_recall_0p5"]) < 0.99:
        results["first_failure"] = "G5_nms_extraction"
        return results
    if (
        float(g5["learned_peaks_recall_0p5"]) + 0.05
        < float(g5["oracle_nms_recall_0p5"])
    ):
        results["first_failure"] = "G5_learned_score_field"
        return results

    # G6: missing-cell/noisy-source evidence only if the current frame supports it.
    qrows_all = valid_proposal_rows(outputs)
    scores_all = outputs.exist_logits[0, qrows_all].sigmoid().detach().float().cpu()
    source_ids_all = (
        outputs.source_instance_ids[0, qrows_all].detach().cpu().long()
    )
    missing_stats: list[dict[str, Any]] = []
    gt_id_to_row = {
        int(gt_id): row
        for row, gt_id in enumerate(context["all_gt_ids"].tolist())
    }
    refs_all = (
        outputs.query_initial_references_cellscale[0, qrows_all]
        .detach()
        .float()
        .cpu()
    )
    for gt_id in context["missing_gt_ids"]:
        center = context["all_gt_centers"][
            gt_id_to_row[int(gt_id)] : gt_id_to_row[int(gt_id)] + 1
        ]
        distance = torch.cdist(refs_all, center)[:, 0]
        selected_near = (distance <= 1.0) & (scores_all >= EXIST_THRESHOLD)
        missing_stats.append(
            {
                "gt_id": int(gt_id),
                "nearest_dref": float(distance.min()) if len(distance) else float("inf"),
                "selected_within_1dref": int(selected_near.sum()),
            }
        )

    noisy_stats = []
    for source_id in context["noisy_source_ids"]:
        tied = source_ids_all == int(source_id)
        noisy_stats.append(
            {
                "source_id": int(source_id),
                "proposal_queries": int(tied.sum()),
                "selected_queries": int(
                    (tied & (scores_all >= EXIST_THRESHOLD)).sum()
                ),
                "max_exist": float(scores_all[tied].max())
                if bool(tied.any())
                else float("nan"),
            }
        )
    results["G6_missing_noise"] = {
        "missing_gt": missing_stats,
        "noisy_sources": noisy_stats,
    }

    # G7: composition. Only run after the preceding gates pass.
    oracle_qrows = pairing["query_rows"].to(trainer.device)
    near_distance = torch.cdist(
        refs_all, context["source9_gt_centers"].float()
    ).min(dim=1).values
    learned_selected = (
        (near_distance <= SOURCE_NEIGHBORHOOD_DREF)
        & (scores_all >= EXIST_THRESHOLD)
    )
    learned_qrows = qrows_all[
        learned_selected.to(qrows_all.device)
    ]

    g7_oracle = compose_fragmentation_stats(
        state,
        trainer,
        outputs,
        context,
        oracle_qrows,
        use_existence=False,
    )
    g7_learned = compose_fragmentation_stats(
        state,
        trainer,
        outputs,
        context,
        learned_qrows,
        use_existence=True,
    )
    results["G7_composition"] = {
        "oracle_one_per_gt": g7_oracle,
        "learned_selection": g7_learned,
    }

    if (
        int(g7_learned["gt_with_multiple_fragments"])
        > int(g7_oracle["gt_with_multiple_fragments"])
    ):
        results["first_failure"] = "G7_learned_composition"
        return results

    # G8: trained temporal contribution only if enough time remains.
    if state.budget.remaining < 240:
        results["G8_temporal_ablation"] = {"status": "skipped_low_time"}
        results["first_failure"] = None
        return results

    full_diag = {
        "mask_hard_dice_mean": g2["hard_dice_mean"],
        "mask_hard_dice_min": g2["hard_dice_min"],
        "final_center_error_mean_dref": g2["final_center_error_mean_dref"],
        **{f"selection_{k}": v for k, v in g4.items()},
    }

    del outputs
    cleanup_cuda()

    spatial_batch = make_spatial_only_batch(batch_cpu)
    spatial_outputs = full_forward(
        state, trainer, spatial_batch, return_debug=False
    )
    spatial_pairing = gt_pairing(
        spatial_outputs,
        context["source9_gt_ids"],
        context["source9_gt_centers"],
    )
    spatial_masks = eval_local_pairing(
        trainer.model.local_mask_decoder,
        spatial_outputs,
        context,
        spatial_pairing,
        use_gt_anchors=False,
    )
    spatial_selection = selection_metrics(
        spatial_outputs,
        context["source9_gt_centers"],
        threshold=EXIST_THRESHOLD,
    )

    sqrows = spatial_pairing["query_rows"].to(spatial_outputs.centers_cellscale.device)
    sgrows = spatial_pairing["gt_rows"]
    if len(sqrows):
        sgt = context["source9_gt_centers"][sgrows]
        sfinal = (
            spatial_outputs.centers_cellscale[0, sqrows]
            .detach()
            .float()
            .cpu()
        )
        spatial_center = float(
            torch.linalg.vector_norm(sfinal - sgt, dim=-1).mean()
        )
    else:
        spatial_center = float("nan")

    results["G8_temporal_ablation"] = {
        "full_temporal": full_diag,
        "spatial_only": {
            "mask_hard_dice_mean": spatial_masks["hard_dice_mean"],
            "mask_hard_dice_min": spatial_masks["hard_dice_min"],
            "final_center_error_mean_dref": spatial_center,
            **{f"selection_{k}": v for k, v in spatial_selection.items()},
        },
    }
    results["first_failure"] = None
    return results


def critical_preflight(
    state: ExperimentState,
    trainer: Trainer,
    batch_cpu: dict[str, Any],
) -> dict[str, Any]:
    state.budget.require_gpu_window("critical preflight", estimated_seconds=30)
    trainer.model.eval()
    gpu_batch = move_batch_to_device(batch_cpu, trainer.device)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model_forward_from_batch(trainer.model, gpu_batch)

    proposals = valid_proposal_rows(outputs)
    if proposals.numel() == 0:
        raise RuntimeError("Preflight produced no spatial proposals.")

    selected = proposals[:1]
    with torch.no_grad():
        rendered = trainer.model.render_masks(outputs, [selected])
    if not bool(torch.isfinite(rendered[0]).all()):
        raise RuntimeError("Preflight render contains non-finite values.")

    torch.cuda.synchronize()
    result = {
        "seconds": time.perf_counter() - started,
        "proposal_count": int(proposals.numel()),
        "rendered_queries": int(selected.numel()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }

    del rendered, selected, proposals, outputs, gpu_batch
    cleanup_cuda()
    record_diagnostic(state, {"stage": "critical_preflight", **result})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--runtime-profile", default="cloud_48gb")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--allow-fresh-start", action="store_true")
    parser.add_argument(
        "--hard-time-limit-seconds",
        type=int,
        default=DEFAULT_INTERNAL_HARD_LIMIT_SECONDS,
    )
    parser.add_argument("--query-min-steps", type=int, default=DEFAULT_QUERY_MIN_STEPS)
    parser.add_argument("--query-max-steps", type=int, default=DEFAULT_QUERY_MAX_STEPS)
    parser.add_argument("--local-min-steps", type=int, default=DEFAULT_LOCAL_MIN_STEPS)
    parser.add_argument("--local-max-steps", type=int, default=DEFAULT_LOCAL_MAX_STEPS)
    parser.add_argument("--joint-min-steps", type=int, default=DEFAULT_JOINT_MIN_STEPS)
    parser.add_argument("--joint-max-steps", type=int, default=DEFAULT_JOINT_MAX_STEPS)
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    for low_name, high_name in (
        ("query_min_steps", "query_max_steps"),
        ("local_min_steps", "local_max_steps"),
        ("joint_min_steps", "joint_max_steps"),
    ):
        low = int(getattr(args, low_name))
        high = int(getattr(args, high_name))
        if low < 0 or high < 1 or low > high:
            parser.error(f"Invalid stage limits: {low_name}={low}, {high_name}={high}")

    if args.hard_time_limit_seconds <= DEFAULT_FINALIZATION_RESERVE_SECONDS + 120:
        parser.error("hard time limit is too small for safe finalization")

    return args


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)

    budget = Budget(float(args.hard_time_limit_seconds))
    state = ExperimentState(run_dir=args.run_dir, budget=budget)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Experiment 30 requires CUDA.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

        log(state, f"GPU: {torch.cuda.get_device_name(0)}")
        log(state, f"Run dir: {args.run_dir}")
        log(state, f"Internal time cap: {args.hard_time_limit_seconds}s")

        batch_cpu, sample = build_real_batch(args.data_dir)
        validate_scene(sample)
        context = source9_context(batch_cpu)

        scene_stats = {
            "sample": sample,
            "source9_gt_ids": context["source9_gt_ids"].tolist(),
            "missing_gt_ids": context["missing_gt_ids"],
            "noisy_source_ids": context["noisy_source_ids"],
        }
        atomic_json(state.run_dir / "scene.json", scene_stats)

        if len(context["source9_gt_ids"]) != 9:
            raise RuntimeError(
                f"Expected 9 GT cells in source {SOURCE_ID}; "
                f"found {len(context['source9_gt_ids'])}."
            )

        cfg = _reduced_config()
        cfg.proposals.enabled = True
        cfg.proposals.query_mode = "spatial_proposals"
        cfg.local_masks.enabled = True

        # Runtime profile must be applied before constructing model/criterion.
        apply_runtime_profile(cfg, args.runtime_profile)

        # Start in query bootstrap. Stage lengths are shortened after early
        # stopping so global_step remains the true optimizer-step count.
        cfg.curriculum.enabled = True
        cfg.curriculum.spatial_dense_steps = 0
        cfg.curriculum.temporal_dense_steps = 0
        cfg.curriculum.query_bootstrap_steps = args.query_max_steps
        cfg.curriculum.native_bootstrap_steps = args.local_max_steps
        cfg.curriculum.joint_spatial_lr_scale = 0.10
        cfg.curriculum.joint_dense_lr_scale = 0.50

        atomic_json(
            state.run_dir / "runtime_profile.json",
            describe_runtime_profile(cfg),
        )

        model = StirNet(cfg)

        warm_start = args.warm_start
        if warm_start is None and args.runs_root is not None:
            warm_start = discover_warm_start(args.runs_root)

        warm_info: dict[str, Any]
        if warm_start is not None:
            if not warm_start.exists():
                raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_start}")
            ckpt = load_checkpoint(
                warm_start,
                model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location="cpu",
                strict=True,
                migrate_history=True,
            )
            warm_info = {
                "path": str(warm_start),
                "loaded": True,
                "source_step": ckpt.get("step"),
                "migration_notes": ckpt.get("model_migration", []),
            }
            log(state, f"Warm-started from {warm_start}")
        elif args.allow_fresh_start:
            warm_info = {"path": None, "loaded": False}
            log(state, "WARNING: starting Experiment 30 from fresh initialization.")
        else:
            raise RuntimeError(
                "No warm-start checkpoint was supplied/discovered. "
                "Provide --warm-start, populate --runs-root with an earlier "
                "checkpoint, or explicitly pass --allow-fresh-start."
            )

        atomic_json(state.run_dir / "warm_start.json", warm_info)

        trainer = Trainer(model, cfg, device="cuda", amp_dtype="fp16")
        trainer.global_step = 0

        critical_preflight(state, trainer, batch_cpu)

        # Query + temporal refresh.
        query = run_training_phase(
            state,
            trainer,
            cfg,
            batch_cpu,
            context,
            phase_name="query_temporal_refresh",
            min_steps=args.query_min_steps,
            max_steps=args.query_max_steps,
            eval_every=QUERY_EVAL_EVERY,
            tracker=PlateauTracker(mode="min", min_delta=0.005, patience_evals=2),
            diagnostic=False,
        )
        query_steps = int(query["successful_steps"])
        cfg.curriculum.query_bootstrap_steps = query_steps
        trainer.curriculum.current = None

        model_only_checkpoint(
            state,
            state.run_dir / "checkpoint_after_query.pt",
            trainer=trainer,
            cfg=cfg,
            label="after_query",
            extra={"query_steps": query_steps},
        )

        # Local native-mask bootstrap.
        local = run_training_phase(
            state,
            trainer,
            cfg,
            batch_cpu,
            context,
            phase_name="local_mask_bootstrap",
            min_steps=args.local_min_steps,
            max_steps=args.local_max_steps,
            eval_every=LOCAL_EVAL_EVERY,
            tracker=PlateauTracker(mode="max", min_delta=0.01, patience_evals=3),
            diagnostic=True,
        )
        local_steps = int(local["successful_steps"])
        cfg.curriculum.native_bootstrap_steps = local_steps
        trainer.curriculum.current = None

        model_only_checkpoint(
            state,
            state.run_dir / "checkpoint_after_local_mask.pt",
            trainer=trainer,
            cfg=cfg,
            label="after_local_mask",
            extra={
                "query_steps": query_steps,
                "local_steps": local_steps,
            },
        )

        # Full joint integration.
        joint = run_training_phase(
            state,
            trainer,
            cfg,
            batch_cpu,
            context,
            phase_name="joint",
            min_steps=args.joint_min_steps,
            max_steps=args.joint_max_steps,
            eval_every=JOINT_EVAL_EVERY,
            tracker=PlateauTracker(mode="max", min_delta=0.005, patience_evals=3),
            diagnostic=True,
        )
        joint_steps = int(joint["successful_steps"])

        model_only_checkpoint(
            state,
            state.run_dir / "checkpoint_joint_best_or_last.pt",
            trainer=trainer,
            cfg=cfg,
            label="joint_last",
            extra={
                "query_steps": query_steps,
                "local_steps": local_steps,
                "joint_steps": joint_steps,
            },
        )

        if not args.skip_oracle:
            state.budget.require_gpu_window(
                "oracle ladder",
                estimated_seconds=120,
                extra_reserve_seconds=60,
            )
            log(state, "START cost-aware first-failure oracle ladder")
            state.oracle = run_oracle_first_failure(
                state, trainer, batch_cpu, context
            )
            log(
                state,
                "DONE oracle ladder; first_failure="
                f"{state.oracle.get('first_failure')}",
            )

        # Final completed-run checkpoint. This is omitted if we are too close
        # to the time cap.
        full_checkpoint(
            state,
            state.run_dir / "checkpoint_final.pt",
            trainer=trainer,
            cfg=cfg,
            label="completed",
            extra={
                "stages": state.stages,
                "oracle_first_failure": state.oracle.get("first_failure"),
            },
        )

        final_scalar_flush(
            state,
            status="completed",
            reason="normal_completion",
        )
        log(
            state,
            f"Experiment 30 complete in {state.budget.elapsed:.1f}s; "
            f"remaining={state.budget.remaining:.1f}s",
        )
        return 0

    except TimeBudgetExceeded as exc:
        # Critical requirement: statistics only. No new checkpoint/tensor save.
        log(state, f"TIME CAP: {exc}")
        cleanup_cuda()
        final_scalar_flush(
            state,
            status="time_cap",
            reason=str(exc),
        )
        return 0

    except BaseException as exc:
        # Failure path also prioritizes diagnostics over a potentially expensive
        # emergency tensor write.
        try:
            log(state, f"FAILURE: {type(exc).__name__}: {exc}")
            cleanup_cuda()
            final_scalar_flush(
                state,
                status="failed",
                reason=str(exc),
                error=exc,
            )
        finally:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
