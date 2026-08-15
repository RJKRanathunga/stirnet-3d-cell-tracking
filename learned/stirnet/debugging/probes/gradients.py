from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from ...training.trainer import model_forward_from_batch, move_to_device


MODULE_GROUPS = (
    ("spatial.encoder", "encoder."),
    ("temporal.graph_encoder", "graph_encoder."),
    ("temporal.tracklet_pooler", "tracklet_pooler."),
    ("temporal.builder", "temporal_builder."),
    ("coreasoning.cr1", "cr1."),
    ("spatial.decoder", "decoder."),
    ("coreasoning.cr2", "cr2."),
    ("queries.builder", "query_builder."),
    ("queries.decoder", "query_decoder."),
    ("heads.native_mask", "native_mask_head."),
    ("heads.dense", "dense_heads."),
)


def _group(name):
    for group, prefix in MODULE_GROUPS:
        if name.startswith(prefix):
            return group
    return "other"


def summarize_gradients(model) -> list[dict[str, Any]]:
    acc = {}
    for name, parameter in model.named_parameters():
        group = _group(name)
        row = acc.setdefault(group, {
            "parameter_tensors": 0,
            "gradient_tensors": 0,
            "parameter_count": 0,
            "gradient_nonzero_count": 0,
            "parameter_sq_sum": 0.0,
            "gradient_sq_sum": 0.0,
            "gradient_abs_max": 0.0,
            "all_gradients_finite": True,
        })
        row["parameter_tensors"] += 1
        row["parameter_count"] += int(parameter.numel())
        row["parameter_sq_sum"] += float(parameter.detach().float().square().sum().cpu())
        grad = parameter.grad
        if grad is None:
            continue
        row["gradient_tensors"] += 1
        finite = bool(torch.isfinite(grad).all())
        row["all_gradients_finite"] = bool(row["all_gradients_finite"] and finite)
        if finite:
            work = grad.detach().float()
            row["gradient_sq_sum"] += float(work.square().sum().cpu())
            row["gradient_abs_max"] = max(float(row["gradient_abs_max"]), float(work.abs().max().cpu()) if work.numel() else 0.0)
            row["gradient_nonzero_count"] += int(torch.count_nonzero(work).cpu())

    result = []
    for group, row in acc.items():
        pnorm = row.pop("parameter_sq_sum") ** 0.5
        gnorm = row.pop("gradient_sq_sum") ** 0.5
        result.append({
            "module": group,
            **row,
            "parameter_norm": pnorm,
            "gradient_norm": gnorm,
            "gradient_to_parameter_ratio": gnorm / max(pnorm, 1e-12),
        })
    return sorted(result, key=lambda r: r["module"])


def run_total_backward_probe(model, criterion, batch, *, device=None, amp_dtype="fp16"):
    """Forward + criterion + backward, with no optimizer step."""
    device = torch.device(device or next(model.parameters()).device)
    b = {}
    for key, value in batch.items():
        if key == "targets":
            b[key] = value
        elif key == "spatial_inputs":
            dtype = torch.float16 if amp_dtype == "fp16" and device.type == "cuda" else torch.bfloat16 if amp_dtype == "bf16" and device.type == "cuda" else torch.float32
            b[key] = value.to(device=device, dtype=dtype)
        elif key == "instance_labels":
            b[key] = value.to(device=device, dtype=torch.int32)
        else:
            b[key] = move_to_device(value, device)

    model.train(); criterion.train(); model.zero_grad(set_to_none=True)
    autocast = nullcontext()
    if device.type == "cuda" and amp_dtype != "none":
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16 if amp_dtype == "fp16" else torch.bfloat16)
    with autocast:
        outputs = model_forward_from_batch(model, b)
        losses = criterion(
            outputs,
            b["targets"],
            local_mask_decoder=model.local_mask_decoder,
        )
        loss = losses["loss"]
    # Mirror the training-time protection against FP16 gradient underflow
    # without taking an optimizer step. The probe does not own an optimizer,
    # so unscale gradients explicitly before summarizing them.
    scale = 1024.0 if device.type == "cuda" and amp_dtype == "fp16" else 1.0
    (loss * scale).backward()
    if scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(scale)

    return {
        "losses": {k: float(v.detach().float().cpu()) for k, v in losses.items()},
        "gradients": summarize_gradients(model),
    }
