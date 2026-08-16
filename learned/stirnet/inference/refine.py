from __future__ import annotations

import torch

from ..training.trainer import move_batch_to_device, model_forward_from_batch
from .postprocess import PostprocessConfig, postprocess_batch


class StirNetRefiner:
    """Inference wrapper returning the V2 connected partition and in-mask centers."""

    def __init__(self, model, device=None, postprocess_config=None):
        self.model = model
        self.device = torch.device(device or next(model.parameters()).device)
        self.postprocess_config = postprocess_config or PostprocessConfig()
        self.model.to(self.device).eval()

    @torch.no_grad()
    def refine_batch(self, batch: dict):
        moved = move_batch_to_device(batch, self.device)
        outputs = model_forward_from_batch(
            self.model,
            moved,
            use_temporal=True,
            run_refinement=self.model.cfg.refinement.enabled,
            apply_existence_filter=self.model.cfg.instances.apply_existence_filter,
        )
        results = postprocess_batch(
            outputs,
            moved["spacing_um"],
            config=self.postprocess_config,
            valid_min_rel_um=batch.get("valid_min_rel_um"),
            valid_max_rel_um=batch.get("valid_max_rel_um"),
        )
        return results, outputs


__all__ = ["StirNetRefiner"]
