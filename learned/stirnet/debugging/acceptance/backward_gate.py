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
from .first_overfit import _reduced_config, _repo_root, build_real_batch

GIB = 1024**3


def _memory():
    return f"allocated={torch.cuda.memory_allocated()/GIB:.3f} GiB, reserved={torch.cuda.memory_reserved()/GIB:.3f} GiB, peak={torch.cuda.max_memory_allocated()/GIB:.3f} GiB"


def _begin_phase():
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); return time.perf_counter()


def _end_phase(name, started):
    torch.cuda.synchronize(); print(f"{name}: seconds={time.perf_counter()-started:.2f}; {_memory()}", flush=True)


def run(
    data_dir: Path,
    *,
    temporal_memory_ablation: str = "full",
    detection_graph_ablation: str = "full",
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The real backward gate requires CUDA")
    batch, sample = build_real_batch(data_dir)
    cfg = _reduced_config(); device = torch.device("cuda")
    model = StirNet(cfg).to(device).train()
    criterion = RefinementCriterion(cfg.losses, cfg.queries, cfg.training).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    b = {}
    for key, value in batch.items():
        if key == "targets": b[key] = value
        elif key == "spatial_inputs": b[key] = value.to(device=device, dtype=torch.float16)
        elif key == "instance_labels": b[key] = value.to(device=device, dtype=torch.int32)
        else: b[key] = move_to_device(value, device)
    del batch
    gc.collect(); torch.cuda.empty_cache(); optimizer.zero_grad(set_to_none=True)
    print(
        f"ablation: temporal_memory={temporal_memory_ablation}; "
        f"detection_graph={detection_graph_ablation}; "
        f"nodes={len(b['graph_x'])}; candidate_edges={b['graph_edge_index'].shape[1]}; "
        f"accepted_edges={int((b['graph_edge_attr'][:,14]>0.5).sum())}",
        flush=True,
    )

    phase = "model forward"
    try:
        started = _begin_phase()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model_forward_from_batch(
                model,
                b,
                temporal_memory_ablation=temporal_memory_ablation,
                detection_graph_ablation=detection_graph_ablation,
            )
        _end_phase(phase, started)

        phase = "matching and loss forward"; started = _begin_phase()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            losses = criterion(outputs, b["targets"]); loss = losses["loss"]
        _end_phase(phase, started)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite total loss: {float(loss)}")

        phase = "scaled backward"; started = _begin_phase(); scaler.scale(loss).backward(); _end_phase(phase, started)
        phase = "unscale and gradient clip"; started = _begin_phase(); scaler.unscale_(optimizer)
        finite = nonzero = total = 0; changed_name = None; before = None
        for name, parameter in model.named_parameters():
            if parameter.grad is None: continue
            total += 1
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"Non-finite gradient: {name}")
            finite += 1
            if bool(torch.count_nonzero(parameter.grad)):
                nonzero += 1
                if changed_name is None:
                    changed_name = name; before = parameter.detach().clone()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm, error_if_nonfinite=True)
        _end_phase(phase, started)
        if not math.isfinite(float(grad_norm)) or float(grad_norm) <= 0:
            raise RuntimeError(f"Invalid gradient norm: {float(grad_norm)}")
        print(f"gradients: finite={finite}/{total}; nonzero={nonzero}; preclip_norm={float(grad_norm):.7f}")
        path_prefixes = {
            "query_temporal_memory":"query_decoder.layers.0.temporal_fusion.node_attention.",
            "detection_graph":"graph_encoder.",
            "history_encoder":"history_encoder.",
            "history_fusion":"history_fusion.",
        }
        for label,prefix in path_prefixes.items():
            values=[
                parameter.grad.detach().float().abs().sum()
                for name,parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            gradient_sum=float(torch.stack(values).sum()) if values else 0.0
            print(f"gradient_path/{label}_abs_sum={gradient_sum:.9g}")
            if temporal_memory_ablation=="full" and gradient_sum<=0:
                raise RuntimeError(f"No gradient reached required path: {label}")

        phase = "optimizer step"; started = _begin_phase(); scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update(); _end_phase(phase, started)
        if changed_name is not None:
            after = dict(model.named_parameters())[changed_name]
            delta = float((after.detach() - before).abs().max())
            print(f"parameter_update: name={changed_name}; max_abs_delta={delta:.9g}; scale={scale_before:.1f}->{scaler.get_scale():.1f}")
    except torch.OutOfMemoryError:
        print(f"OOM during {phase}: {_memory()}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    default = _repo_root(Path.cwd()) / "data" / "learned" / "stirnet" / "first_overfit" / "BlastoSPIM1_F22_030_034"
    parser.add_argument("--data-dir", type=Path, default=default)
    parser.add_argument(
        "--temporal-memory-ablation",
        choices=("full","zero_node","shuffle_node","tracklet_only","node_only"),
        default="full",
    )
    parser.add_argument(
        "--detection-graph-ablation",choices=("full","accepted_only"),default="full"
    )
    args = parser.parse_args()
    run(
        args.data_dir,
        temporal_memory_ablation=args.temporal_memory_ablation,
        detection_graph_ablation=args.detection_graph_ablation,
    )


if __name__ == "__main__":
    main()
