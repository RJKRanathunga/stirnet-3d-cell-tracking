"""Same-weight synthetic causal diagnostic for hierarchical temporal memory.

The scene contains two cells at T-2/T-1, one current merged source, and two
cells again at T+1. Accepted associations explain only one branch; the complete
candidate graph and fine node memory still retain every observation.
"""
from __future__ import annotations

import argparse
import json

import torch

from .history_overfit import (
    _task_metrics,
    _tiny_config,
    build_converging_merge_batch,
)
from ...model import RefinementCriterion, StirNet
from ...model.query_builder import QUERY_SPLIT
from ...training.trainer import model_forward_from_batch, move_batch_to_device


def _split_attention(output) -> list[dict[str, float | int]]:
    debug = output.debug or {}
    layers = debug.get("query_temporal_attention") or []
    node = debug.get("node_memory") or {}
    node_ids = node.get("node_ids")
    result = []
    for layer_index, layer in enumerate(layers):
        if not layer or "node" not in layer:
            continue
        query_type = layer["query_type"]
        slots = layer["query_slot_index"]
        top_indices = layer["node"]["top_indices"]
        top_weights = layer["node"]["top_weights"]
        for row in torch.nonzero(query_type == QUERY_SPLIT, as_tuple=False).flatten():
            if top_indices.shape[1] == 0:
                continue
            memory_index = int(top_indices[row, 0])
            result.append(
                {
                    "layer": layer_index,
                    "query": int(slots[row]),
                    "node_index": memory_index,
                    "node_id": (
                        int(node_ids[memory_index]) if node_ids is not None else memory_index
                    ),
                    "weight": float(top_weights[row, 0]),
                    "entropy": float(layer["node"]["entropy"][row]),
                }
            )
    return result


def _sibling_center_separation(output) -> list[float]:
    debug = output.debug or {}
    references = debug.get("query_layer_references_cellscale")
    if references is None:
        return []
    query_types = output.query_types[0]
    source_ids = output.source_instance_ids[0]
    split = query_types == QUERY_SPLIT
    separations = []
    for layer in references:
        distances = []
        for source_id in torch.unique(source_ids[split]).tolist():
            indices = torch.nonzero(
                split & (source_ids == source_id), as_tuple=False
            ).flatten()
            if len(indices) > 1:
                pairwise = torch.pdist(layer[0, indices].float())
                distances.append(pairwise.mean())
        separations.append(float(torch.stack(distances).mean()) if distances else 0.0)
    return separations


def run(steps: int = 100, device: str | None = None) -> dict:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(23)
    batch = move_batch_to_device(
        build_converging_merge_batch(accepted_second_branch=False), selected_device
    )
    cfg = _tiny_config()
    # Two unconstrained split slots make specialization directly observable;
    # neither slot is assigned to a historical branch by the architecture.
    cfg.queries.split_companions_per_instance = 2
    cfg.proposals.query_mode = "legacy"
    model = StirNet(cfg).to(selected_device)
    criterion = RefinementCriterion(
        cfg.losses, cfg.queries, cfg.training, cfg.proposals
    ).to(
        selected_device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    initial_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model_forward_from_batch(model, batch)
        losses = criterion(
            output,
            batch["targets"],
            local_mask_decoder=model.local_mask_decoder,
        )
        losses["loss"].backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = float(losses["loss"].detach())

    trials = {
        "full": ("full", "full"),
        "zero_node": ("zero_node", "full"),
        "shuffle_node": ("shuffle_node", "full"),
        "tracklet_only": ("tracklet_only", "full"),
        "node_only": ("node_only", "full"),
        "accepted_graph_only": ("full", "accepted_only"),
    }
    report: dict[str, object] = {
        "initial_loss": initial_loss,
        "N_nodes": int(batch["graph_x"].shape[0]),
        "candidate_edges": int(batch["graph_edge_index"].shape[1]),
        "accepted_edges": int((batch["graph_edge_attr"][:, 14] > 0.5).sum()),
        "trials": {},
    }
    model.eval()
    for name, (memory_ablation, graph_ablation) in trials.items():
        with torch.no_grad():
            output = model_forward_from_batch(
                model,
                batch,
                return_debug=True,
                return_full_temporal_attention=name == "full",
                temporal_memory_ablation=memory_ablation,
                detection_graph_ablation=graph_ablation,
            )
            losses = criterion(
                output,
                batch["targets"],
                local_mask_decoder=model.local_mask_decoder,
            )
        metrics = _task_metrics(model, output, batch["targets"][0])
        metrics.update(
            {
                "loss": float(losses["loss"]),
                "split_attention": _split_attention(output),
                "decoder_layer_sibling_center_separation_dref": (
                    _sibling_center_separation(output)
                ),
            }
        )
        report["trials"][name] = metrics
    full_loss = report["trials"]["full"]["loss"]
    report["same_weight_loss_delta"] = {
        name: values["loss"] - full_loss
        for name, values in report["trials"].items()
        if name != "full"
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.steps, args.device), indent=2))


if __name__ == "__main__":
    main()
