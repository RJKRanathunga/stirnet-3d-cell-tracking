from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...model.query_builder import (
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPATIAL_PROPOSAL,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
)
from .matching import MatchingProbeResult, coarse_dice_for_matches


QUERY_TYPE_NAMES = {
    int(QUERY_PRIMARY): "primary",
    int(QUERY_SPLIT): "split",
    int(QUERY_TEMPORAL): "temporal",
    int(QUERY_DISCOVERY): "discovery",
    int(QUERY_SPATIAL_PROPOSAL): "spatial_proposal",
}


def _layers(outputs, hook_capture):
    captured = [] if hook_capture is None else [x for x in hook_capture.decoder_layers if x is not None]
    if len(captured) == 3:
        return captured

    result = []
    for out in outputs.aux_outputs:
        result.append(
            {
                "exist_logits": out["exist_logits"].detach().float().cpu(),
                "centers_cellscale": out["centers_cellscale"].detach().float().cpu(),
                "coarse_mask_logits": out["coarse_mask_logits"].detach().float().cpu(),
                "query_embeddings": out["query_embeddings"].detach().float().cpu(),
            }
        )
    result.append(
        {
            "exist_logits": outputs.exist_logits.detach().float().cpu(),
            "centers_cellscale": outputs.centers_cellscale.detach().float().cpu(),
            "coarse_mask_logits": outputs.coarse_mask_logits.detach().float().cpu(),
            "query_embeddings": outputs.query_embeddings.detach().float().cpu(),
        }
    )
    return result


@torch.no_grad()
def build_query_table(outputs, targets, matching: MatchingProbeResult, hook_capture, final_exist_threshold: float):
    layers = _layers(outputs, hook_capture)
    B, Q = outputs.exist_logits.shape
    rows: list[dict[str, Any]] = []
    initial = None if hook_capture is None else hook_capture.query_initial

    for b in range(B):
        match = matching.matches[b]
        layer_dice = []
        for layer in layers:
            logits = layer["coarse_mask_logits"]
            layer_logits = logits[b] if logits.ndim == 5 else logits
            layer_dice.append(coarse_dice_for_matches(layer_logits, targets[b], match))

        gt_centers = torch.as_tensor(targets[b]["centers_cellscale"], dtype=torch.float32)
        gt_ids = torch.as_tensor(targets[b]["ids"], dtype=torch.long)
        dref = float(outputs.dref_um[b].detach().cpu())
        qtypes = outputs.query_types[b].detach().cpu().long()
        source_ids = outputs.source_instance_ids[b].detach().cpu().long()
        padding = outputs.query_padding_mask[b].detach().cpu().bool()
        salience = outputs.temporal_salience[b].detach().float().cpu()
        reliability = outputs.temporal_reliability[b].detach().float().cpu()

        temporal_counter = 0
        for q in range(Q):
            qtype = int(qtypes[q])
            temporal_index = temporal_counter if qtype == int(QUERY_TEMPORAL) else -1
            if qtype == int(QUERY_TEMPORAL):
                temporal_counter += 1

            row: dict[str, Any] = {
                "batch": b,
                "query": q,
                "valid_query": not bool(padding[q]),
                "query_type": QUERY_TYPE_NAMES.get(qtype, f"type_{qtype}"),
                "query_type_id": qtype,
                "source_instance_id": int(source_ids[q]),
                "temporal_index": temporal_index,
                "temporal_salience": float(salience[q, 0]),
                "temporal_reliability": float(reliability[q, 0]),
                "matched": False,
                "target_index": -1,
                "gt_id": -1,
                "center_error_um": float("nan"),
            }

            if initial is not None:
                ref_um = initial["references_cellscale"][b, q].numpy() * dref
            else:
                ref_um = (
                    outputs.query_initial_references_cellscale[b, q]
                    .detach().float().cpu().numpy() * dref
                )
            row.update(initial_z_um=float(ref_um[0]), initial_y_um=float(ref_um[1]), initial_x_um=float(ref_um[2]))

            target_idx = matching.query_to_target.get((b, q))
            if target_idx is not None:
                row["matched"] = True
                row["target_index"] = int(target_idx)
                row["gt_id"] = int(gt_ids[target_idx])

            for li, layer in enumerate(layers, start=1):
                logits = layer["exist_logits"]
                centers = layer["centers_cellscale"]
                logit = logits[b, q] if logits.ndim == 2 else logits[q]
                center = centers[b, q] if centers.ndim == 3 else centers[q]
                center_um = center.numpy() * dref
                row[f"layer{li}_exist_prob"] = float(torch.sigmoid(logit))
                row[f"layer{li}_center_z_um"] = float(center_um[0])
                row[f"layer{li}_center_y_um"] = float(center_um[1])
                row[f"layer{li}_center_x_um"] = float(center_um[2])
                row[f"layer{li}_coarse_dice"] = float(layer_dice[li - 1].get(q, float("nan")))

                if target_idx is not None:
                    delta_um = (center - gt_centers[target_idx]).numpy() * dref
                    err = float(np.linalg.norm(delta_um))
                    row[f"layer{li}_center_error_um"] = err
                    if li == len(layers):
                        row["center_error_um"] = err

            final_prob = row.get(f"layer{len(layers)}_exist_prob", 0.0)
            row["survives_final_exist"] = bool(row["valid_query"] and final_prob >= final_exist_threshold)
            rows.append(row)

    return rows


def select_queries_for_deep_probe(
    rows,
    *,
    explicit: tuple[int, ...],
    max_queries: int,
    worst_center: int,
    worst_coarse_dice: int,
    top_temporal: int,
    top_split: int,
):
    if max_queries <= 0:
        return []
    selected: list[int] = []

    def add(q):
        q = int(q)
        if q not in selected and len(selected) < max_queries:
            selected.append(q)

    for q in explicit:
        add(q)

    matched = [r for r in rows if r.get("matched")]
    center_sorted = [r for r in matched if np.isfinite(float(r.get("center_error_um", np.nan)))]
    center_sorted.sort(key=lambda r: -float(r["center_error_um"]))
    for r in center_sorted[:worst_center]:
        add(r["query"])

    def last_dice(row):
        keys = sorted(k for k in row if k.startswith("layer") and k.endswith("_coarse_dice"))
        if not keys:
            return float("inf")
        value = float(row[keys[-1]])
        return value if np.isfinite(value) else float("inf")

    for r in sorted(matched, key=last_dice)[:worst_coarse_dice]:
        add(r["query"])

    temporal = [r for r in rows if r.get("survives_final_exist") and r.get("query_type") == "temporal"]
    temporal.sort(key=lambda r: -float(r.get("layer3_exist_prob", 0.0)))
    for r in temporal[:top_temporal]:
        add(r["query"])

    split = [r for r in rows if r.get("query_type") == "split"]
    split.sort(key=lambda r: -float(r.get("layer3_exist_prob", 0.0)))
    for r in split[:top_split]:
        add(r["query"])

    return selected
