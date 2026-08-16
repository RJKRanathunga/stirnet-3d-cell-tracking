from __future__ import annotations

import torch

from .trainer import (
    gt_labels_from_batch,
    model_forward_from_batch,
    move_batch_to_device,
)


@torch.no_grad()
def evaluate_losses(
    model,
    criterion,
    loader,
    device,
    stage: str = "refinement_joint",
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        moved = move_batch_to_device(batch, torch.device(device))
        output = model_forward_from_batch(
            model,
            moved,
            execution_stage={
                "geometry_bootstrap": "geometry",
                "spatial_partition": "spatial",
                "instance_temporal": "temporal",
                "refinement_joint": "refinement",
            }[stage],
            apply_existence_filter=False,
        )
        losses = criterion(
            output,
            gt_labels_from_batch(moved),
            moved["spacing_um"],
            moved["dref_um"],
            stage=stage,
        )
        count += 1
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
    return {key: value / max(count, 1) for key, value in sums.items()}


__all__ = ["evaluate_losses"]
