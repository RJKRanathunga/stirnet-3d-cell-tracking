"""Run exactly one real all-cell STIR-Net optimizer-step acceptance gate."""
from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch

from learned.stirnet import RefinementCriterion, StirNet
from learned.stirnet.training.trainer import model_forward_from_batch, move_to_device

from .first_overfit_acceptance import (
    _reduced_config,
    _repo_root,
    build_real_batch,
)


GIB = 1024**3


def _memory() -> str:
    return (
        f"allocated={torch.cuda.memory_allocated() / GIB:.3f} GiB, "
        f"reserved={torch.cuda.memory_reserved() / GIB:.3f} GiB, "
        f"peak={torch.cuda.max_memory_allocated() / GIB:.3f} GiB"
    )


def _begin_phase() -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _end_phase(name: str, started: float) -> None:
    torch.cuda.synchronize()
    print(f"{name}: seconds={time.perf_counter() - started:.2f}; {_memory()}", flush=True)


def _finite_primary_outputs(outputs) -> None:
    tensors = {
        "exist_logits": outputs.exist_logits,
        "centers_cellscale": outputs.centers_cellscale,
        "coarse_mask_logits": outputs.coarse_mask_logits,
        "query_embeddings": outputs.query_embeddings,
    }
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Non-finite model output: {name}")


def run(data_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The real backward gate requires CUDA")

    total_started = time.perf_counter()
    batch, sample = build_real_batch(data_dir)
    print(
        f"ROI: {sample['roi_shape']}; current={sample['current_count']}; "
        f"GT={sample['target_count']}",
        flush=True,
    )
    print(
        f"graph nodes={sample['graph_nodes']}; "
        f"temporal tracklets={sample['temporal_tracklets']}; "
        f"required queries={sample['required_queries']}",
        flush=True,
    )

    cfg = _reduced_config()
    properties = torch.cuda.get_device_properties(0)
    print(
        f"device={properties.name}; physical_cuda_gib="
        f"{properties.total_memory / GIB:.3f}",
        flush=True,
    )
    print(
        "checkpointing: "
        f"master={cfg.training.activation_checkpointing}, "
        f"spatial={cfg.training.checkpoint_spatial}, "
        f"coreasoning={cfg.training.checkpoint_coreasoning}, "
        f"losses={cfg.training.checkpoint_losses}",
        flush=True,
    )
    device = torch.device("cuda")
    model = StirNet(cfg).to(device).train()
    criterion = RefinementCriterion(
        cfg.losses, cfg.queries, cfg.training
    ).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    batch_device = {}
    for key, value in batch.items():
        if key == "targets":
            batch_device[key] = value
        elif key == "spatial_inputs":
            batch_device[key] = value.to(device=device, dtype=torch.float16)
        elif key == "instance_labels":
            batch_device[key] = value.to(device=device, dtype=torch.int32)
        else:
            batch_device[key] = move_to_device(value, device)
    del batch

    gc.collect()
    torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    phase = "model forward"
    try:
        started = _begin_phase()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model_forward_from_batch(model, batch_device)
        _end_phase(phase, started)
        _finite_primary_outputs(outputs)

        phase = "matching and loss forward"
        started = _begin_phase()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            losses = criterion(outputs, batch_device["targets"])
            loss = losses["loss"]
        _end_phase(phase, started)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite total loss: {float(loss)}")
        print(
            f"loss={float(loss.detach()):.7f}; scale={scaler.get_scale():.1f}",
            flush=True,
        )

        phase = "scaled backward"
        started = _begin_phase()
        scaler.scale(loss).backward()
        _end_phase(phase, started)

        phase = "unscale and gradient clip"
        started = _begin_phase()
        scaler.unscale_(optimizer)
        finite_gradient_tensors = 0
        nonzero_gradient_tensors = 0
        parameter_tensor_count = 0
        changed_parameter_name = None
        changed_parameter_before = None
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            parameter_tensor_count += 1
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"Non-finite gradient: {name}")
            finite_gradient_tensors += 1
            if bool(torch.count_nonzero(parameter.grad)):
                nonzero_gradient_tensors += 1
                if changed_parameter_name is None:
                    changed_parameter_name = name
                    changed_parameter_before = parameter.detach().clone()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.max_grad_norm, error_if_nonfinite=True
        )
        _end_phase(phase, started)
        grad_norm_value = float(grad_norm)
        if not math.isfinite(grad_norm_value) or grad_norm_value <= 0:
            raise RuntimeError(f"Invalid gradient norm: {grad_norm_value}")
        if nonzero_gradient_tensors == 0 or changed_parameter_name is None:
            raise RuntimeError("No nonzero parameter gradients")
        print(
            f"gradients: finite={finite_gradient_tensors}/{parameter_tensor_count}; "
            f"nonzero={nonzero_gradient_tensors}; preclip_norm={grad_norm_value:.7f}; "
            f"clip_limit={cfg.training.max_grad_norm:.3f}",
            flush=True,
        )

        phase = "optimizer step"
        started = _begin_phase()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        _end_phase(phase, started)
        parameter_after = dict(model.named_parameters())[changed_parameter_name]
        parameter_delta = float(
            (parameter_after.detach() - changed_parameter_before).abs().max()
        )
        if parameter_delta <= 0:
            raise RuntimeError(
                f"Optimizer did not change parameter: {changed_parameter_name}"
            )
        print(
            f"parameter_update: name={changed_parameter_name}; "
            f"max_abs_delta={parameter_delta:.9g}; "
            f"scale={scale_before:.1f}->{scaler.get_scale():.1f}",
            flush=True,
        )

        optimizer.zero_grad(set_to_none=True)
        del outputs, losses, loss, changed_parameter_before
        gc.collect()
        torch.cuda.empty_cache()

        phase = "post-step model forward"
        model.eval()
        criterion.eval()
        started = _begin_phase()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            post_outputs = model_forward_from_batch(model, batch_device)
        _end_phase(phase, started)
        _finite_primary_outputs(post_outputs)
        print("post_step_outputs_finite=True", flush=True)
    except torch.OutOfMemoryError:
        print(f"OOM during {phase}: {_memory()}", flush=True)
        raise

    print(f"total_seconds={time.perf_counter() - total_started:.2f}", flush=True)


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
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
