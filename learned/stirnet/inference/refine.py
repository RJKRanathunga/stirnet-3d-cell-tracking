from __future__ import annotations

import torch

from ..training.trainer import move_to_device, model_forward_from_batch
from .postprocess import postprocess_batch


class StirNetRefiner:
    """Inference wrapper for already-prepared STIR-Net batches.

    Trackastra pass 1 / patch-specific graph preparation intentionally remain external.
    This wrapper begins once a batch satisfies the data contract.
    """
    def __init__(self,model,device=None):
        self.model=model
        self.device=torch.device(device or next(model.parameters()).device)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def refine_batch(self,batch:dict):
        b=move_to_device(batch,self.device)
        outputs=model_forward_from_batch(self.model,b)
        cfg=self.model.cfg.inference
        return postprocess_batch(
            self.model, outputs, cfg.render_exist_threshold, cfg.final_exist_threshold,
            cfg.mask_threshold, cfg.min_mask_voxels,
            batch.get("valid_min_rel_um"), batch.get("valid_max_rel_um"),
        ), outputs
