from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import runpy
import time

import torch

from .model import StirNet
from .training import TrainingConfig
from .training.trainer import (
    Trainer,
    gt_labels_from_batch,
    move_batch_to_device,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device) -> float:
    return (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )


def benchmark_split_refinement(
    data_dir: Path,
    crop_shape: tuple[int, int, int],
    *,
    device: str = "cuda",
    amp_dtype: str = "bf16",
) -> dict[str, float | list[int]]:
    experiment = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "experiments"
            / "stirnet"
            / "31_spatial_first_overfit.py"
        )
    )
    batch, scene = experiment["build_real_batch"](data_dir)
    batch, scene = experiment["crop_batch_for_smoke"](
        batch, scene, crop_shape
    )
    config = experiment["reduced_config"]()
    config.refinement.partition_update = "full"
    training = TrainingConfig()
    training.amp_dtype = amp_dtype
    training.curriculum.fixed_stage = "refinement_joint"
    labels = gt_labels_from_batch(batch)

    torch.manual_seed(1234)
    source = StirNet(config)
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in source.state_dict().items()
    }
    del source

    monolithic = Trainer(
        StirNet(config), copy.deepcopy(training), device=device
    )
    monolithic.model.load_state_dict(initial_state)
    targets = monolithic.prepare_geometry_targets(batch, gt_labels=labels)
    moved = move_batch_to_device(batch, monolithic.device)
    monolithic.model.train()
    monolithic.optimizer.zero_grad(set_to_none=True)
    if monolithic.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(monolithic.device)
    _sync(monolithic.device)
    started = time.perf_counter()
    with monolithic._autocast():
        _, monolithic_losses, _ = monolithic._forward_and_loss(
            moved,
            gt_labels=labels,
            precomputed_geometry_targets=targets,
        )
        monolithic_loss = monolithic_losses["loss"]
    monolithic.scaler.scale(monolithic_loss).backward()
    _sync(monolithic.device)
    monolithic_seconds = time.perf_counter() - started
    monolithic_peak = _peak_mb(monolithic.device)
    del monolithic, monolithic_losses, monolithic_loss, moved, targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    split = Trainer(StirNet(config), copy.deepcopy(training), device=device)
    split.model.load_state_dict(initial_state)
    split_targets = split.prepare_geometry_targets(batch, gt_labels=labels)
    split_moved = move_batch_to_device(batch, split.device)
    split.model.train()
    split.optimizer.zero_grad(set_to_none=True)
    if split.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(split.device)
    _sync(split.device)
    started = time.perf_counter()
    split_losses, _ = split._split_refinement_backward(
        split_moved,
        gt_labels=labels,
        precomputed_geometry_targets=split_targets,
    )
    _sync(split.device)
    split_seconds = time.perf_counter() - started
    split_peak = _peak_mb(split.device)
    return {
        "shape_zyx": list(scene["shape"]),
        "monolithic_forward_backward_seconds": monolithic_seconds,
        "split_forward_backward_seconds": split_seconds,
        "monolithic_peak_allocated_mb": monolithic_peak,
        "split_peak_allocated_mb": split_peak,
        "split_objective_abs_error": float(
            split_losses["split_objective_abs_error"].detach().cpu()
        ),
    }


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            repository
            / "data"
            / "learned"
            / "stirnet"
            / "first_overfit"
            / "BlastoSPIM1_F22_030_034"
        ),
    )
    parser.add_argument("--shape", type=int, nargs=3, default=(12, 96, 96))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = benchmark_split_refinement(
        args.data_dir,
        tuple(args.shape),
        device=args.device,
        amp_dtype=args.amp_dtype,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

