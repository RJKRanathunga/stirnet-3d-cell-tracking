from __future__ import annotations

import torch

from learned.stirnet import StirNetConfig
from learned.stirnet.data.targets import build_gt_targets
from learned.stirnet.training import TrainingConfig


def small_model_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.evidence.stem_channels = 8
    cfg.evidence.prior_gate_hidden = 8
    cfg.spatial.channels = (8, 12, 16, 24)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.acquisition_dim = 16
    cfg.spatial.activation_checkpointing = False
    cfg.geometry.hidden_channels = 16
    cfg.geometry.residual_blocks = 1
    cfg.partition.foreground_threshold = 0.40
    cfg.partition.seed_threshold = 0.20
    cfg.partition.node_feature_channels = 8
    cfg.partition.rag_hidden_dim = 16
    cfg.partition.rag_layers = 1
    cfg.partition.max_supervoxels = 128
    cfg.instances.d_model = 32
    cfg.instances.pooled_feature_dim = 8
    cfg.instances.apply_existence_filter = False
    cfg.history.hidden_channels = 8
    cfg.history.activation_checkpointing = False
    cfg.temporal.d_model = 32
    cfg.temporal.graph_hidden_dim = 48
    cfg.temporal.graph_layers = 1
    cfg.temporal.cross_heads = 4
    cfg.refinement.hidden_channels = 16
    cfg.refinement.query_channels = 8
    cfg.refinement.max_rois_per_batch = 2
    cfg.refinement.max_roi_voxels = 4096
    cfg.validate()
    return cfg


def synthetic_batch(*, temporal: bool = True) -> dict:
    torch.manual_seed(17)
    shape = (6, 12, 12)
    spacing = torch.tensor([[1.6, 0.4, 0.4]])
    dref = torch.tensor([4.0])
    gt = torch.zeros(shape, dtype=torch.long)
    gt[1:5, 2:6, 2:6] = 1
    gt[1:5, 7:11, 7:11] = 2
    noisy = torch.zeros_like(gt)
    noisy[1:5, 2:11, 2:11] = 1
    raw = torch.rand(shape)
    spatial = torch.stack(
        [
            raw,
            (noisy > 0).float(),
            (noisy > 0).float() * 0.5,
            torch.zeros(shape),
            torch.zeros(shape),
        ]
    )[None]
    batch = {
        "spatial_inputs": spatial,
        "instance_labels": noisy[None],
        "spacing_um": spacing,
        "dref_um": dref,
        "targets": [build_gt_targets(gt.numpy(), tuple(spacing[0].tolist()), 4.0, current_labels=noisy.numpy())],
    }
    if temporal:
        graph_x = torch.randn(2, 32)
        batch.update(
            {
                "graph_x": graph_x,
                "graph_edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "graph_edge_attr": torch.randn(2, 15),
                "tracklet_id": torch.tensor([0, 1], dtype=torch.long),
                "temporal_ref_um": torch.tensor([[0.0, -0.8, -0.8], [0.0, 0.8, 0.8]]),
                "temporal_status": torch.zeros(2, 10),
                "temporal_batch": torch.zeros(2, dtype=torch.long),
                "node_instance_grid": torch.randn(2, 4, 8, 8, 8).half(),
                "node_history_valid": torch.tensor([True, False]),
            }
        )
    return batch


def fixed_stage_training(stage: str) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.amp_dtype = "fp32"
    cfg.curriculum.fixed_stage = stage
    return cfg
