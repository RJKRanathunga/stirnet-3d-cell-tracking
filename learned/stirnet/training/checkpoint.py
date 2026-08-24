from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


ARCHITECTURE_ID = "spatial_first_v2"
CHECKPOINT_VERSION = 3


def _serialize(value: Any) -> Any:
    return value.to_dict() if hasattr(value, "to_dict") else value


def save_checkpoint(
    path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    step: int = 0,
    epoch: int = 0,
    model_config=None,
    training_config=None,
    extra=None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": ARCHITECTURE_ID,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "global_step": int(step),
        "epoch": int(epoch),
        "model_config": _serialize(
            model_config if model_config is not None else getattr(model, "cfg", None)
        ),
        "training_config": _serialize(training_config),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, destination)


def _optimizer_structure_matches(optimizer, state: dict) -> bool:
    current = optimizer.state_dict()
    old_groups = state.get("param_groups", [])
    new_groups = current.get("param_groups", [])
    return len(old_groups) == len(new_groups) and all(
        len(old.get("params", [])) == len(new.get("params", []))
        for old, new in zip(old_groups, new_groups)
    )


def _load_exact_compatible(model, state: dict) -> list[str]:
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    strict: bool = True,
    *,
    transfer_compatible: bool = False,
):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE_ID:
        if not transfer_compatible:
            raise ValueError(
                "Checkpoint architecture is not spatial_first_v2. V1 query/proposal "
                "checkpoints are intentionally incompatible; pass "
                "transfer_compatible=True only for explicit exact-name/exact-shape transfer."
            )
        checkpoint["transferred_parameters"] = _load_exact_compatible(
            model, checkpoint.get("model", {})
        )
        return checkpoint

    if transfer_compatible:
        # Explicit architecture transfer also applies when the architecture ID
        # is unchanged but optional modules (for example morphology-aware RAG)
        # have been added. Optimizer state is intentionally not transferred.
        checkpoint["transferred_parameters"] = _load_exact_compatible(
            model, checkpoint.get("model", {})
        )
        return checkpoint

    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        if not _optimizer_structure_matches(optimizer, checkpoint["optimizer"]):
            raise ValueError(
                "Checkpoint optimizer state does not match the V2 parameter groups; "
                "load without an optimizer and create fresh optimizer state."
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


__all__ = [
    "ARCHITECTURE_ID",
    "CHECKPOINT_VERSION",
    "load_checkpoint",
    "save_checkpoint",
]
