from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from learned.stirnet import StirNet
from learned.stirnet.debugging.acceptance.first_overfit import (
    _reduced_config,
    build_real_batch,
)
from learned.stirnet.model.runtime_profiles import (
    apply_runtime_profile,
    describe_runtime_profile,
)
from learned.stirnet.training.trainer import Trainer


SEED = 40266

EXPECTED_CURRENT_CELLS = 36
EXPECTED_GT_CELLS = 33
EXPECTED_TEMPORAL_TRACKLETS = 52
EXPECTED_SPLIT_COMPANIONS = 44
EXPECTED_DISCOVERY_QUERIES = 8
EXPECTED_REQUIRED_QUERIES = 140

L40S_REFERENCE_SECONDS_PER_STEP = 5.0697959909999994
L40S_REFERENCE_LOCAL_MASKS_PER_SECOND = 1.5785966618905067


def repo_root() -> Path:
    """Assumes this file lives at <repo>/modal/03_benchmark_stirnet_local.py."""
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return (
        repo_root()
        / "data"
        / "learned"
        / "stirnet"
        / "first_overfit"
        / "BlastoSPIM1_F22_030_034"
    )


def gib(value: int) -> float:
    return float(value) / 1024**3


def main(data_dir: Path, warmup_steps: int, measured_steps: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this with the project's CUDA-enabled "
            "PyTorch environment."
        )
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1")
    if measured_steps < 1:
        raise ValueError("measured_steps must be at least 1")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_vram_gib = gib(props.total_memory)

    print("=" * 72, flush=True)
    print("STIR-Net local GPU benchmark", flush=True)
    print("=" * 72, flush=True)
    print(f"GPU                 : {gpu_name}", flush=True)
    print(f"CUDA-visible VRAM   : {total_vram_gib:.2f} GiB", flush=True)
    print(f"Torch               : {torch.__version__}", flush=True)
    print(f"CUDA runtime        : {torch.version.cuda}", flush=True)
    print("Runtime profile     : local_6gb", flush=True)
    print(f"Warm-up steps       : {warmup_steps}", flush=True)
    print(f"Measured steps      : {measured_steps}", flush=True)
    print(f"Dataset             : {data_dir}", flush=True)
    print("=" * 72, flush=True)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Prepared STIR-Net dataset not found:\n{data_dir}\n\n"
            "Pass --data-dir if your prepared sample is elsewhere."
        )

    load_started = time.perf_counter()
    batch_cpu, sample = build_real_batch(data_dir)
    load_seconds = time.perf_counter() - load_started

    split_companion_count = sum(
        int(value)
        for value in sample["split_companions_by_source"].values()
    )
    discovery_query_count = _reduced_config().queries.discovery_queries
    reconstructed_required_queries = (
        sample["current_count"]
        + split_companion_count
        + sample["temporal_tracklets"]
        + discovery_query_count
    )

    print("\nSample", flush=True)
    print("-" * 72, flush=True)
    print(f"ROI shape           : {sample['roi_shape']}", flush=True)
    print(f"Current cells       : {sample['current_count']}", flush=True)
    print(f"GT cells            : {sample['target_count']}", flush=True)
    print(f"Temporal tracklets  : {sample['temporal_tracklets']}", flush=True)
    print(f"Split companions    : {split_companion_count}", flush=True)
    print(f"Discovery queries   : {discovery_query_count}", flush=True)
    print(f"Required queries    : {sample['required_queries']}", flush=True)
    print(
        "Query decomposition : "
        f"{sample['current_count']} primary + "
        f"{split_companion_count} split + "
        f"{sample['temporal_tracklets']} temporal + "
        f"{discovery_query_count} discovery = "
        f"{reconstructed_required_queries}",
        flush=True,
    )
    print(f"Batch build time    : {load_seconds:.2f} s", flush=True)

    assert sample["current_count"] == EXPECTED_CURRENT_CELLS, sample
    assert sample["target_count"] == EXPECTED_GT_CELLS, sample
    assert sample["temporal_tracklets"] == EXPECTED_TEMPORAL_TRACKLETS, sample
    assert split_companion_count == EXPECTED_SPLIT_COMPANIONS, sample
    assert discovery_query_count == EXPECTED_DISCOVERY_QUERIES, sample
    assert sample["required_queries"] == reconstructed_required_queries, sample
    assert sample["required_queries"] == EXPECTED_REQUIRED_QUERIES, sample

    cfg = _reduced_config()

    cfg.proposals.enabled = True
    cfg.proposals.query_mode = "spatial_proposals"
    cfg.local_masks.enabled = True

    apply_runtime_profile(cfg, "local_6gb")

    cfg.curriculum.enabled = True
    cfg.curriculum.spatial_dense_steps = 0
    cfg.curriculum.temporal_dense_steps = 0
    cfg.curriculum.query_bootstrap_steps = 0
    cfg.curriculum.native_bootstrap_steps = 0
    cfg.curriculum.joint_spatial_lr_scale = 0.10
    cfg.curriculum.joint_dense_lr_scale = 0.50

    print("\nEffective runtime configuration", flush=True)
    print("-" * 72, flush=True)
    print(
        json.dumps(describe_runtime_profile(cfg), indent=2, default=str),
        flush=True,
    )
    print(
        f"Decoder max spatial tokens     : {cfg.decoder.max_spatial_tokens} "
        f"(model setting; not profile-controlled)",
        flush=True,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = StirNet(cfg)
    parameter_count = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        model,
        cfg,
        device=device,
        amp_dtype="fp16",
    )
    trainer.global_step = 0

    stage = trainer.curriculum.apply(trainer.global_step)
    if stage.name != "joint":
        raise RuntimeError(
            f"Benchmark expected joint curriculum stage, got {stage.name!r}"
        )

    trainable_parameter_count = sum(
        p.numel()
        for p in trainer.model.parameters()
        if p.requires_grad
    )

    print("\nModel", flush=True)
    print("-" * 72, flush=True)
    print(f"Curriculum stage    : {stage.name}", flush=True)
    print(f"Parameters          : {parameter_count:,}", flush=True)
    print(f"Trainable params    : {trainable_parameter_count:,}", flush=True)
    print("AMP                 : fp16", flush=True)
    print(
        f"Local-mask cap      : {cfg.local_masks.train_max_queries_per_batch}",
        flush=True,
    )

    print("\nWarm-up", flush=True)
    print("-" * 72, flush=True)

    for step in range(warmup_steps):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()

        metrics = trainer.train_step(batch_cpu)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        sampled_local = len(
            getattr(trainer.criterion, "last_local_sampled_requests", [])
        )
        original_local_counts = list(
            getattr(trainer.criterion, "last_local_original_request_counts", [])
        )

        print(
            f"warmup {step + 1}/{warmup_steps}: "
            f"{elapsed:.3f} s | "
            f"loss={metrics.get('loss', float('nan')):.6f} | "
            f"peak alloc={gib(torch.cuda.max_memory_allocated()):.2f} GiB | "
            f"peak reserved={gib(torch.cuda.max_memory_reserved()):.2f} GiB | "
            f"local sampled={sampled_local} from {original_local_counts}",
            flush=True,
        )

    print("\nMeasured steps", flush=True)
    print("-" * 72, flush=True)

    rows: list[dict] = []

    for step in range(measured_steps):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        allocated_before = gib(torch.cuda.memory_allocated())
        reserved_before = gib(torch.cuda.memory_reserved())

        started = time.perf_counter()
        metrics = trainer.train_step(batch_cpu)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        peak_allocated = gib(torch.cuda.max_memory_allocated())
        peak_reserved = gib(torch.cuda.max_memory_reserved())
        allocated_after = gib(torch.cuda.memory_allocated())
        reserved_after = gib(torch.cuda.memory_reserved())

        sampled_local = len(
            getattr(trainer.criterion, "last_local_sampled_requests", [])
        )
        original_local_counts = list(
            getattr(trainer.criterion, "last_local_original_request_counts", [])
        )

        local_masks_per_second = (
            sampled_local / elapsed if elapsed > 0 else float("nan")
        )

        row = {
            "step": step + 1,
            "global_step": trainer.global_step,
            "seconds": elapsed,
            "loss": float(metrics.get("loss", float("nan"))),
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "allocated_before_gib": allocated_before,
            "reserved_before_gib": reserved_before,
            "allocated_after_gib": allocated_after,
            "reserved_after_gib": reserved_after,
            "local_sampled": sampled_local,
            "local_original_counts": original_local_counts,
            "local_masks_per_second": local_masks_per_second,
        }
        rows.append(row)

        print(
            f"step {step + 1}/{measured_steps}: "
            f"{elapsed:.3f} s | "
            f"loss={row['loss']:.6f} | "
            f"peak alloc={peak_allocated:.2f} GiB | "
            f"peak reserved={peak_reserved:.2f} GiB | "
            f"local={sampled_local} | "
            f"local/s={local_masks_per_second:.4f}",
            flush=True,
        )

    seconds = [row["seconds"] for row in rows]
    local_rates = [
        row["local_masks_per_second"]
        for row in rows
        if np.isfinite(row["local_masks_per_second"])
    ]

    median_seconds = statistics.median(seconds)
    mean_seconds = statistics.mean(seconds)
    max_peak_allocated = max(row["peak_allocated_gib"] for row in rows)
    max_peak_reserved = max(row["peak_reserved_gib"] for row in rows)
    mean_local_rate = statistics.mean(local_rates) if local_rates else float("nan")

    l40s_step_speedup = (
        median_seconds / L40S_REFERENCE_SECONDS_PER_STEP
    )
    l40s_mask_throughput_speedup = (
        L40S_REFERENCE_LOCAL_MASKS_PER_SECOND / mean_local_rate
        if mean_local_rate > 0
        else float("inf")
    )

    report = {
        "gpu": gpu_name,
        "total_vram_gib": total_vram_gib,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "runtime_profile": "local_6gb",
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "batch_build_seconds": load_seconds,
        "sample": sample,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "median_seconds_per_step": median_seconds,
        "mean_seconds_per_step": mean_seconds,
        "max_peak_allocated_gib": max_peak_allocated,
        "max_peak_reserved_gib": max_peak_reserved,
        "mean_local_masks_per_second": mean_local_rate,
        "l40s_reference_seconds_per_step": L40S_REFERENCE_SECONDS_PER_STEP,
        "l40s_reference_local_masks_per_second": (
            L40S_REFERENCE_LOCAL_MASKS_PER_SECOND
        ),
        "l40s_step_speedup_over_local": l40s_step_speedup,
        "l40s_useful_mask_throughput_speedup_over_local": (
            l40s_mask_throughput_speedup
        ),
        "steps": rows,
    }

    print("\n" + "=" * 72, flush=True)
    print("LOCAL BENCHMARK SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"GPU                   : {gpu_name}", flush=True)
    print("Runtime profile       : local_6gb", flush=True)
    print(f"Median step time      : {median_seconds:.3f} s", flush=True)
    print(f"Mean step time        : {mean_seconds:.3f} s", flush=True)
    print(f"Max peak allocated    : {max_peak_allocated:.2f} GiB", flush=True)
    print(f"Max peak reserved     : {max_peak_reserved:.2f} GiB", flush=True)
    print(f"Mean local masks/s    : {mean_local_rate:.4f}", flush=True)
    print("-" * 72, flush=True)
    print(f"L40S step speedup     : {l40s_step_speedup:.2f}x", flush=True)
    print(
        "L40S useful-mask "
        f"throughput speedup    : {l40s_mask_throughput_speedup:.2f}x",
        flush=True,
    )
    print("=" * 72, flush=True)

    print("\nReturned benchmark report:")
    print(json.dumps(report, indent=2, default=str))

    del trainer, model, batch_cpu
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real STIR-Net joint-training step locally with "
            "the local_6gb runtime profile."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help=(
            "Prepared first-overfit dataset directory. Defaults to the "
            "repository data/learned/stirnet path."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir.resolve(),
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
