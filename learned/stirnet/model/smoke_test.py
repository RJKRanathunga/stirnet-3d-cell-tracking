from __future__ import annotations

import torch

from .config import ModelConfig
from .stir_net import StirNet
from .types import TemporalInput


def main() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig()
    cfg.instances.apply_existence_filter = False
    cfg.refinement.max_rois_per_batch = 2
    cfg.partition.max_supervoxels = 512
    model = StirNet(cfg).eval()

    spatial = torch.randn(1, 5, 10, 20, 20)
    spatial[:, 1:] = spatial[:, 1:].sigmoid()
    spacing = torch.tensor([[1.625, 0.40625, 0.40625]])
    dref = torch.tensor([5.0])

    node_count = 4
    track_count = 2
    graph_x = torch.randn(node_count, cfg.temporal.node_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], cfg.temporal.edge_dim)
    temporal = TemporalInput(
        graph_x=graph_x,
        graph_edge_index=edge_index,
        graph_edge_attr=edge_attr,
        tracklet_id=torch.tensor([0, 0, 1, 1]),
        temporal_ref_um=torch.tensor([[0.0, -2.0, -2.0], [0.0, 2.0, 2.0]]),
        temporal_status=torch.randn(track_count, cfg.temporal.status_dim),
        temporal_batch=torch.zeros(track_count, dtype=torch.long),
    )
    history = torch.randn(node_count, cfg.history.input_channels, 8, 8, 8)
    history_valid = torch.tensor([True, True, False, True])

    with torch.no_grad():
        out = model(
            spatial,
            spacing,
            dref,
            temporal_input=temporal,
            node_instance_grid=history,
            node_history_valid=history_valid,
            run_refinement=True,
            apply_existence_filter=False,
            return_debug=True,
        )

    assert out.geometry.flow.shape == (1, 3, 10, 20, 20)
    assert out.final_labels[0].shape == (10, 20, 20)
    assert out.centers_um[0].shape[-1] == 3
    assert out.temporal.tokens.shape == (track_count, cfg.temporal.d_model)
    assert out.rag.edge_index.shape[0] == 2
    print(
        "STIR-Net spatial-first smoke test OK |",
        "supervoxels=", int(out.rag.supervoxel_labels[0].max()),
        "instances=", int(out.final_labels[0].max()),
        "temporal_tokens=", out.temporal.tokens.shape[0],
        "refinements=", 0 if out.refinement is None else out.refinement.applied_count,
    )


if __name__ == "__main__":
    main()
