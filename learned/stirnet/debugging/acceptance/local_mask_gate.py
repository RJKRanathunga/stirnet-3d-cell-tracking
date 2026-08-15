"""Reduced real-sample CUDA gate for bounded STIR-Net local-mask training."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import time
from pathlib import Path

import torch

from learned.stirnet import StirNet
from learned.stirnet.model.query_builder import QUERY_SPATIAL_PROPOSAL
from learned.stirnet.training.trainer import (
    Trainer,
    model_forward_from_batch,
    move_batch_to_device,
)

from .first_overfit import _reduced_config, _repo_root, build_real_batch


GIB = 1024**3


def _start_phase() -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _finish_phase(name: str, started: float) -> tuple[float, float]:
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / GIB
    reserved = torch.cuda.max_memory_reserved() / GIB
    print(
        f"{name}: success; seconds={time.perf_counter() - started:.2f}; "
        f"peak_allocated_gib={allocated:.3f}; peak_reserved_gib={reserved:.3f}",
        flush=True,
    )
    return allocated, reserved


def _finite_gradients(model: StirNet) -> tuple[bool, int, int]:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    nonzero = sum(bool(torch.count_nonzero(value)) for value in gradients)
    return finite, nonzero, len(gradients)


def _train_step(trainer: Trainer, batch: dict, label: str) -> dict[str, float]:
    started = _start_phase()
    losses = trainer.train_step(batch)
    _finish_phase(label, started)
    finite, nonzero, total = _finite_gradients(trainer.model)
    sampled = trainer.criterion.last_local_sampled_requests
    original = trainer.criterion.last_local_original_request_counts
    sampled_by_batch = Counter(batch_index for batch_index, _ in sampled)
    print(
        f"{label}: stage={trainer.curriculum_stage.name}; "
        f"sampled_local_queries={len(sampled)}; original_local_matches={original}; "
        f"finite_gradients={finite}; nonzero_gradients={nonzero}/{total}; "
        f"loss={losses['loss']:.7g}; dice_hi={losses['dice_hi']:.7g}; "
        f"focal_hi={losses['focal_hi']:.7g}",
        flush=True,
    )
    if any(
        count > trainer.cfg.local_masks.train_max_queries_per_batch
        for count in sampled_by_batch.values()
    ):
        raise RuntimeError("local sampled-query cap was exceeded")
    if not finite:
        raise RuntimeError(f"{label} produced non-finite gradients")
    return losses


def run(data_dir: Path, *, local_steps: int = 3) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The local-mask acceptance gate requires CUDA")
    batch, sample = build_real_batch(data_dir)
    cfg = _reduced_config()
    cfg.local_masks.train_max_queries_per_batch = 2
    cfg.curriculum.enabled = True
    cfg.curriculum.spatial_dense_steps = 0
    cfg.curriculum.temporal_dense_steps = 0
    cfg.curriculum.query_bootstrap_steps = 0
    cfg.curriculum.native_bootstrap_steps = local_steps
    trainer = Trainer(StirNet(cfg), cfg, device="cuda", amp_dtype="fp16")
    device_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / GIB
    print(
        f"device={device_name}; total_memory_gib={total_memory:.3f}; "
        f"roi_shape={sample['roi_shape']}; targets={sample['target_count']}; "
        f"local_train_cap={cfg.local_masks.train_max_queries_per_batch}",
        flush=True,
    )

    # Reproduce the Notebook-29 rendering sequence: an FP16-autocast forward,
    # followed by local rendering after the autocast context has ended.
    gpu_batch = move_batch_to_device(batch, torch.device("cuda"))
    trainer.model.eval()
    started = _start_phase()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model_forward_from_batch(trainer.model, gpu_batch)
    _finish_phase("fp16_forward", started)
    proposals = torch.nonzero(
        outputs.query_types[0] == QUERY_SPATIAL_PROPOSAL, as_tuple=False
    ).flatten()
    if proposals.numel() == 0:
        raise RuntimeError("real acceptance forward produced no spatial proposals")
    selected = proposals[: min(2, proposals.numel())]
    started = _start_phase()
    with torch.no_grad():
        rendered = trainer.model.render_masks(outputs, [selected])
    _finish_phase("render_outside_autocast", started)
    if rendered[0].shape[0] != selected.numel():
        raise RuntimeError("rendered proposal count is incorrect")
    if not bool(torch.isfinite(rendered[0]).all()):
        raise RuntimeError("rendered masks contain non-finite values")
    print(
        f"render_outside_autocast: selected={selected.tolist()}; "
        f"shape={tuple(rendered[0].shape)}; dtype={rendered[0].dtype}",
        flush=True,
    )
    del rendered, selected, proposals, outputs, gpu_batch
    gc.collect()
    torch.cuda.empty_cache()

    for index in range(local_steps):
        _train_step(trainer, batch, f"local_bootstrap_{index + 1}")

    # The next curriculum step is joint. It is informative but not the hard
    # acceptance condition requested for the 6-GiB local bootstrap path.
    try:
        _train_step(trainer, batch, "short_joint_1")
        spatial_gradient = trainer.model.decoder.stage_e0.fuse.weight.grad
        spatial_gradient_finite = (
            spatial_gradient is not None
            and bool(torch.isfinite(spatial_gradient).all())
            and bool(torch.count_nonzero(spatial_gradient))
        )
        print(
            f"short_joint_1: d0_spatial_gradient_finite_nonzero="
            f"{spatial_gradient_finite}",
            flush=True,
        )
        if not spatial_gradient_finite:
            raise RuntimeError("joint local loss did not reach the D0 spatial path")
    except torch.OutOfMemoryError:
        print(
            "short_joint_1: CUDA OOM (local bootstrap acceptance already passed); "
            f"allocated_gib={torch.cuda.memory_allocated() / GIB:.3f}; "
            f"reserved_gib={torch.cuda.memory_reserved() / GIB:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    default = (
        _repo_root(Path.cwd())
        / "data"
        / "learned"
        / "stirnet"
        / "first_overfit"
        / "BlastoSPIM1_F22_030_034"
    )
    parser.add_argument("--data-dir", type=Path, default=default)
    parser.add_argument("--local-steps", type=int, default=3)
    args = parser.parse_args()
    if args.local_steps < 1:
        parser.error("--local-steps must be at least 1")
    run(args.data_dir, local_steps=args.local_steps)


if __name__ == "__main__":
    main()
