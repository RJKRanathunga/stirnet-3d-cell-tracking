from __future__ import annotations

import torch

from .trainer import move_to_device, model_forward_from_batch


@torch.no_grad()
def evaluate_losses(model,criterion,loader,device):
    model.eval();sums={};n=0
    for batch in loader:
        b=move_to_device(batch,device);out=model_forward_from_batch(model,b);losses=criterion(
            out,
            b["targets"],
            local_mask_decoder=(
                model.local_mask_decoder
                if model.cfg.local_masks.enabled
                else None
            ),
        );n+=1
        for k,v in losses.items():sums[k]=sums.get(k,0.0)+float(v.detach().cpu())
    return {k:v/max(n,1) for k,v in sums.items()}
