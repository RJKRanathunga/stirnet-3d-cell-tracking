from __future__ import annotations

import torch

from .model import StirNet, StirNetConfig
from .model.smoke_test import main as model_smoke_main
from .training import LossConfig, StirNetCriterion


def _small_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.evidence.stem_channels = 8
    cfg.evidence.prior_gate_hidden = 8
    cfg.spatial.channels = (8, 12, 16, 24)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.acquisition_dim = 16
    cfg.geometry.hidden_channels = 16
    cfg.geometry.residual_blocks = 1
    cfg.partition.node_feature_channels = 8
    cfg.partition.rag_hidden_dim = 16
    cfg.partition.rag_layers = 1
    cfg.partition.max_supervoxels = 128
    cfg.instances.d_model = 32
    cfg.instances.pooled_feature_dim = 8
    cfg.history.hidden_channels = 8
    cfg.temporal.d_model = 32
    cfg.temporal.graph_hidden_dim = 48
    cfg.temporal.graph_layers = 1
    cfg.temporal.cross_heads = 4
    cfg.refinement.hidden_channels = 16
    cfg.refinement.query_channels = 8
    cfg.refinement.enabled = False
    cfg.validate()
    return cfg


def criterion_backward_smoke() -> None:
    torch.manual_seed(11)
    cfg = _small_config()
    model = StirNet(cfg).train()
    spatial = torch.rand(1, 5, 6, 12, 12)
    spacing = torch.tensor([[1.6, 0.4, 0.4]])
    dref = torch.tensor([4.0])
    gt = torch.zeros((1, 6, 12, 12), dtype=torch.long)
    gt[:, 1:5, 3:9, 3:9] = 1
    output = model(
        spatial,
        spacing,
        dref,
        run_refinement=False,
        apply_existence_filter=False,
    )
    criterion = StirNetCriterion(cfg, LossConfig())
    losses = criterion(
        output, gt, spacing, dref, stage="geometry_bootstrap"
    )
    losses["loss"].backward()
    groups = {
        "evidence_stem": model.evidence_stem,
        "spatial_backbone": model.spatial_backbone,
        "geometry_decoder": model.geometry_decoder,
    }
    for name, module in groups.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not any(
            bool(torch.isfinite(grad).all() and grad.abs().sum() > 0)
            for grad in gradients
        ):
            raise AssertionError(f"No finite nonzero gradient reached {name}")
    print("STIR-Net V2 criterion/backward smoke test OK")


def main() -> None:
    model_smoke_main()
    criterion_backward_smoke()


if __name__ == "__main__":
    main()
