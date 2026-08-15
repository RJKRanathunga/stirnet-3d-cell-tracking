from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Modal / repository layout
# ---------------------------------------------------------------------------

app = modal.App("stirnet-benchmark")
data_volume = modal.Volume.from_name("stirnet-data")

# Save this file as:
#   <repo>/modal/02_benchmark_stirnet.py
LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep remote container paths as POSIX strings. This file is imported on
# Windows before Modal sends it to Linux.
REMOTE_REPO_ROOT = "/workspace/cell-tracking"
DATA_MOUNT = "/workspace/cell-tracking/data"
DATA_DIR = (
    "/workspace/cell-tracking/data/learned/stirnet/first_overfit/"
    "BlastoSPIM1_F22_030_034"
)

# Match the current local preprocessing/runtime environment as closely as
# practical. If your local torch version changes, update this pin too so the
# local-vs-cloud timing comparison remains fair.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.13.0",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
        "networkx>=3.0",
    )
    .workdir(REMOTE_REPO_ROOT)
    .add_local_dir(
        LOCAL_REPO_ROOT / "learned",
        remote_path="/workspace/cell-tracking/learned",
    )
)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

SEED = 40266
EXPECTED_CURRENT_CELLS = 36
EXPECTED_GT_CELLS = 33
EXPECTED_TEMPORAL_TRACKLETS = 52
EXPECTED_REQUIRED_QUERIES = 140
EXPECTED_SPLIT_COMPANIONS = 44
EXPECTED_DISCOVERY_QUERIES = 8

VALID_VARIANTS = {
    "baseline",
    "no_spatial_checkpoint",
}


@app.function(
    image=image,
    gpu="L40S",
    cpu=2.0,
    memory=8192,
    timeout=60 * 60,
    volumes={
        DATA_MOUNT: data_volume,
    },
)
def benchmark(
    variant: str = "baseline",
    warmup_steps: int = 2,
    measured_steps: int = 3,
) -> dict:
    """
    Benchmark the real all-cell STIR-Net joint-training step on an NVIDIA L4.

    Variants
    --------
    baseline:
        Use cloud_48gb exactly as configured in runtime_profiles.py.

    no_spatial_checkpoint:
        Start from cloud_48gb, then disable spatial activation checkpointing.
        Use this only as the A/B follow-up after the baseline measurement.
    """
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, REMOTE_REPO_ROOT)

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

    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown benchmark variant {variant!r}. "
            f"Expected one of {sorted(VALID_VARIANTS)}."
        )
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1.")
    if measured_steps < 1:
        raise ValueError("measured_steps must be at least 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal container.")

    # ------------------------------------------------------------------
    # Reproducibility / CUDA setup
    # ------------------------------------------------------------------

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    total_vram_gib = gpu_props.total_memory / 1024**3

    print("=" * 72, flush=True)
    print("STIR-Net Modal GPU benchmark", flush=True)
    print("=" * 72, flush=True)
    print(f"GPU                 : {gpu_name}", flush=True)
    print(f"Total VRAM          : {total_vram_gib:.2f} GiB", flush=True)
    print(f"Torch               : {torch.__version__}", flush=True)
    print(f"CUDA runtime        : {torch.version.cuda}", flush=True)
    print(f"Variant             : {variant}", flush=True)
    print(f"Warm-up steps       : {warmup_steps}", flush=True)
    print(f"Measured steps      : {measured_steps}", flush=True)
    print(f"Dataset             : {DATA_DIR}", flush=True)
    print("=" * 72, flush=True)

    # ------------------------------------------------------------------
    # Load the exact real BlastoSPIM first-overfit sample.
    # GT target maps intentionally remain CPU-backed in Trainer.train_step().
    # ------------------------------------------------------------------

    data_path = Path(DATA_DIR)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared STIR-Net dataset is missing from the Volume: {data_path}"
        )

    load_started = time.perf_counter()
    batch_cpu, sample = build_real_batch(data_path)
    load_seconds = time.perf_counter() - load_started

    print("\nSample", flush=True)
    print("-" * 72, flush=True)
    print(f"ROI shape           : {sample['roi_shape']}", flush=True)
    print(f"Current cells       : {sample['current_count']}", flush=True)
    print(f"GT cells            : {sample['target_count']}", flush=True)
    print(f"Temporal tracklets  : {sample['temporal_tracklets']}", flush=True)
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

    # Established invariants for the current first-overfit scene.
    assert sample["current_count"] == EXPECTED_CURRENT_CELLS, sample
    assert sample["target_count"] == EXPECTED_GT_CELLS, sample
    assert sample["temporal_tracklets"] == EXPECTED_TEMPORAL_TRACKLETS, sample
    assert split_companion_count == EXPECTED_SPLIT_COMPANIONS, sample
    assert discovery_query_count == EXPECTED_DISCOVERY_QUERIES, sample
    assert (
        sample["required_queries"] == reconstructed_required_queries
    ), sample
    assert sample["required_queries"] == EXPECTED_REQUIRED_QUERIES, sample

    # ------------------------------------------------------------------
    # Build the same model architecture used by the current first-overfit
    # notebooks, then change only execution/memory behavior via cloud_48gb.
    #
    # Important:
    #   _reduced_config() is the model/experiment architecture.
    #   cloud_48gb is the runtime/execution profile.
    # ------------------------------------------------------------------

    cfg = _reduced_config()

    # Match the current proposal/local-mask training path.
    cfg.proposals.enabled = True
    cfg.proposals.query_mode = "spatial_proposals"
    cfg.local_masks.enabled = True

    # Select the cloud execution profile implemented in runtime_profiles.py.
    apply_runtime_profile(cfg, "cloud_48gb")

    # Benchmark the most representative/high-memory training stage: joint.
    # Zero-length preceding stages make step 0 enter "joint" immediately.
    cfg.curriculum.enabled = True
    cfg.curriculum.spatial_dense_steps = 0
    cfg.curriculum.temporal_dense_steps = 0
    cfg.curriculum.query_bootstrap_steps = 0
    cfg.curriculum.native_bootstrap_steps = 0
    cfg.curriculum.joint_spatial_lr_scale = 0.10
    cfg.curriculum.joint_dense_lr_scale = 0.50

    if variant == "no_spatial_checkpoint":
        # A/B experiment only. Do not edit runtime_profiles.py yet.
        cfg.training.checkpoint_spatial = False

    profile_description = describe_runtime_profile(cfg)

    print("\nEffective runtime configuration", flush=True)
    print("-" * 72, flush=True)
    if isinstance(profile_description, str):
        print(profile_description, flush=True)
    else:
        print(
            json.dumps(profile_description, indent=2, default=str),
            flush=True,
        )

    print(
        f"Effective spatial checkpointing: "
        f"{cfg.training.checkpoint_spatial}",
        flush=True,
    )
    print(
        f"Local-mask train cap           : "
        f"{cfg.local_masks.train_max_queries_per_batch}",
        flush=True,
    )
    print(
        f"Decoder max spatial tokens     : "
        f"{cfg.decoder.max_spatial_tokens} "
        f"(model setting; not profile-controlled)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Model / trainer
    # ------------------------------------------------------------------

    gc.collect()
    torch.cuda.empty_cache()

    model = StirNet(cfg)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    trainer = Trainer(
        model,
        cfg,
        device=device,
        amp_dtype="fp16",
    )
    trainer.global_step = 0

    # Confirm that this benchmark is really exercising the full joint stage.
    stage = trainer.curriculum.apply(trainer.global_step)
    if stage.name != "joint":
        raise RuntimeError(
            f"Benchmark expected joint curriculum stage, got {stage.name!r}."
        )

    # Trainer initialization applies stage 0 before we overwrite/confirm the
    # stage above, so compute the final trainable count after joint is applied.
    trainable_parameter_count = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )

    print("\nModel", flush=True)
    print("-" * 72, flush=True)
    print(f"Curriculum stage    : {stage.name}", flush=True)
    print(f"Parameters          : {parameter_count:,}", flush=True)
    print(f"Trainable params    : {trainable_parameter_count:,}", flush=True)
    print(f"AMP                 : fp16", flush=True)

    # ------------------------------------------------------------------
    # Warm-up
    #
    # This allocates AdamW state, initializes CUDA kernels/caches, and ensures
    # measured iterations represent steady-state training rather than startup.
    # ------------------------------------------------------------------

    print("\nWarm-up", flush=True)
    print("-" * 72, flush=True)

    for step in range(warmup_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()

        metrics = trainer.train_step(batch_cpu)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        sampled_local = len(
            getattr(
                trainer.criterion,
                "last_local_sampled_requests",
                [],
            )
        )
        original_local_counts = list(
            getattr(
                trainer.criterion,
                "last_local_original_request_counts",
                [],
            )
        )

        print(
            f"warmup {step + 1}/{warmup_steps}: "
            f"{elapsed:.3f} s | "
            f"loss={metrics.get('loss', float('nan')):.6f} | "
            f"local sampled={sampled_local} "
            f"from {original_local_counts}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Measured steady-state steps
    # ------------------------------------------------------------------

    print("\nMeasured steps", flush=True)
    print("-" * 72, flush=True)

    rows: list[dict] = []

    for step in range(measured_steps):
        # Do not call empty_cache() here. Real training reuses the CUDA
        # allocator/cache between iterations, so the benchmark should too.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3

        started = time.perf_counter()
        metrics = trainer.train_step(batch_cpu)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3

        sampled_local = len(
            getattr(
                trainer.criterion,
                "last_local_sampled_requests",
                [],
            )
        )
        original_local_counts = list(
            getattr(
                trainer.criterion,
                "last_local_original_request_counts",
                [],
            )
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
            f"local/s={local_masks_per_second:.3f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    seconds = [row["seconds"] for row in rows]
    peak_allocated_values = [row["peak_allocated_gib"] for row in rows]
    peak_reserved_values = [row["peak_reserved_gib"] for row in rows]
    local_rates = [
        row["local_masks_per_second"]
        for row in rows
        if row["local_masks_per_second"] == row["local_masks_per_second"]
    ]

    median_seconds = statistics.median(seconds)
    mean_seconds = statistics.mean(seconds)
    max_peak_allocated = max(peak_allocated_values)
    max_peak_reserved = max(peak_reserved_values)
    mean_local_rate = statistics.mean(local_rates) if local_rates else float("nan")

    headroom_from_reserved = total_vram_gib - max_peak_reserved

    report = {
        "gpu": gpu_name,
        "total_vram_gib": total_vram_gib,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "variant": variant,
        "runtime_profile": "cloud_48gb",
        "checkpoint_spatial": bool(cfg.training.checkpoint_spatial),
        "local_mask_train_cap": int(
            cfg.local_masks.train_max_queries_per_batch
        ),
        "decoder_max_spatial_tokens": int(
            cfg.decoder.max_spatial_tokens
        ),
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
        "reserved_headroom_gib": headroom_from_reserved,
        "mean_local_masks_per_second": mean_local_rate,
        "steps": rows,
    }

    print("\n" + "=" * 72, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"Variant              : {variant}", flush=True)
    print(
        f"Spatial checkpoint   : {cfg.training.checkpoint_spatial}",
        flush=True,
    )
    print(
        f"Median step time     : {median_seconds:.3f} s",
        flush=True,
    )
    print(
        f"Mean step time       : {mean_seconds:.3f} s",
        flush=True,
    )
    print(
        f"Max peak allocated   : {max_peak_allocated:.2f} GiB",
        flush=True,
    )
    print(
        f"Max peak reserved    : {max_peak_reserved:.2f} GiB",
        flush=True,
    )
    print(
        f"Reserved headroom    : {headroom_from_reserved:.2f} GiB",
        flush=True,
    )
    print(
        f"Mean local masks/s   : {mean_local_rate:.3f}",
        flush=True,
    )

    if max_peak_reserved < 14:
        recommendation = (
            "Large VRAM headroom remains. Next test: disable spatial "
            "checkpointing."
        )
    elif max_peak_reserved < 17:
        recommendation = (
            "Good headroom remains. Test spatial checkpointing OFF."
        )
    elif max_peak_reserved < 20:
        recommendation = (
            "Healthy 24-GB utilization. Spatial checkpointing OFF may still "
            "be worth an A/B test."
        )
    elif max_peak_reserved <= 21:
        recommendation = (
            "Near the intended 18-21 GiB operating range. Do not increase "
            "memory pressure aggressively."
        )
    elif max_peak_reserved <= 22:
        recommendation = (
            "Close to the safety boundary. Keep the current memory-saving "
            "settings unless the speed gain is compelling."
        )
    else:
        recommendation = (
            "Too close to 24-GB OOM territory. Reduce memory pressure before "
            "long training."
        )

    report["recommendation"] = recommendation
    print(f"Recommendation        : {recommendation}", flush=True)
    print("=" * 72, flush=True)

    # Free GPU state before the Modal container exits.
    del trainer, model, batch_cpu
    gc.collect()
    torch.cuda.empty_cache()

    return report


@app.local_entrypoint()
def main(
    variant: str = "baseline",
    warmup_steps: int = 2,
    measured_steps: int = 3,
):
    """
    Examples:

        modal run .\\modal\\02_benchmark_stirnet.py

        modal run .\\modal\\02_benchmark_stirnet.py \
            --variant no_spatial_checkpoint

        modal run .\\modal\\02_benchmark_stirnet.py \
            --warmup-steps 2 --measured-steps 5
    """
    result = benchmark.remote(
        variant=variant,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )

    print("\nReturned benchmark report:")
    print(json.dumps(result, indent=2, default=str))
