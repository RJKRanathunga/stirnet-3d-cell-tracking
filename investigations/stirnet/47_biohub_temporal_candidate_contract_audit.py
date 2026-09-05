from __future__ import annotations

"""Investigation 47: audit the temporal-candidate action contract.

This investigation is intentionally diagnostic.  It reuses the curated spatial
cache, Investigation-42 temporal checkpoint, and Investigation-46 V4 physical
evidence/cutter, then compares the raw temporal candidate with the only logits
that can legally be written by the split-only policy::

    legal = where(case.editable, candidate, spatial)

It also evaluates candidate-choice and target-edge oracles under that same
contract, records every bad development component, and runs several frozen
input ablations.  No validation-derived threshold is fitted here.
"""

import copy
import dataclasses
import importlib.util
import json
import math
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (
            (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "dataset_curation").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV46 = load_module(
    ROOT
    / "investigations"
    / "stirnet"
    / "46_biohub_hard_cutter_temporal_training_v4.py",
    "_inv47_inv46_v4",
)
INV42 = INV46.INV42
INV45 = INV46.INV45
INV35 = INV46.INV35

SCRIPT_NAME = "47_biohub_temporal_candidate_contract_audit"


class Inv46DiscreteWriteSelector(torch.nn.Module):
    """Checkpoint-compatible Investigation-46 V3 edge selector."""

    SCALAR_DIM = 16

    def __init__(self, edge_embedding_dim: int) -> None:
        super().__init__()
        self.edge_embedding_dim = int(edge_embedding_dim)
        self.edge_norm = torch.nn.LayerNorm(self.edge_embedding_dim)
        self.scalar_norm = torch.nn.LayerNorm(self.SCALAR_DIM)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.edge_embedding_dim + self.SCALAR_DIM, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, *, edge_embedding, scalar_features):
        return self.net(
            torch.cat(
                [
                    self.edge_norm(edge_embedding.float()),
                    self.scalar_norm(scalar_features.float()),
                ],
                dim=-1,
            )
        ).squeeze(-1)


def atomic_json(path: Path, payload: Any) -> None:
    INV42.atomic_json(path, payload)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def metric_with_counts(acc: dict[str, float]) -> dict[str, Any]:
    result = INV42.finalize_metrics(acc)
    result.update(
        {
            "cut_correct": int(acc["cut_correct"]),
            "keep_correct": int(acc["keep_correct"]),
            "bad_components_exact": int(acc["bad_exact"]),
            "clean_components_split": int(acc["clean_split"]),
        }
    )
    return result


def predicted_partition(model, case, logits, *, enforce: bool = True):
    result = model.partitioner(
        case.rag,
        logits,
        model.cfg.partition.final_merge_threshold,
        stage="final",
    )
    return INV42.enforce_split_only(case, result) if enforce else result


def constrained_split_only_partition(
    model, case, logits, *, enforce: bool = True
):
    """Solve each frozen spatial component independently.

    The current implementation solves the full-frame multicut and intersects
    its answer with the current partition afterward.  Cross-component edges can
    therefore change within-component decisions through multicut cycle costs.
    Giving every cross-component interface a decisive CUT removes those paths
    from the optimization itself, which is the intended split-only contract.
    """
    result = model.partitioner(
        case.rag,
        logits,
        model.cfg.partition.final_merge_threshold,
        stage="final",
        node_parent_component=case.node_current_component,
    )
    return INV42.enforce_split_only(case, result) if enforce else result


def update_from_partition(
    acc: dict[str, float],
    *,
    model,
    runtime,
    case,
    logits,
    partition,
) -> None:
    threshold = float(model.cfg.partition.final_merge_threshold)
    keep_prediction = torch.sigmoid(logits) >= threshold
    acc["cut_total"] += int(case.cut_mask.sum().item())
    acc["cut_correct"] += int(
        (~keep_prediction[case.cut_mask]).sum().item()
    )
    acc["keep_total"] += int(case.keep_mask.sum().item())
    acc["keep_correct"] += int(
        keep_prediction[case.keep_mask].sum().item()
    )
    bad_total, bad_exact, clean_total, clean_split = (
        INV42.exact_component_counts(runtime, case, partition)
    )
    acc["bad_total"] += bad_total
    acc["bad_exact"] += bad_exact
    acc["clean_total"] += clean_total
    acc["clean_split"] += clean_split
    acc["violations"] += INV35.split_only_violation_count(case, partition)


def legal_candidate_logits(case, candidate_logits):
    return torch.where(
        case.editable,
        candidate_logits,
        case.rag.spatial_edge_logits,
    )


def component_exact(runtime, case, partition, component: int) -> bool:
    return INV46.component_exact_v4(
        runtime,
        case,
        partition.node_component_global,
        int(component),
    )


def component_flags(runtime, case, partition) -> torch.Tensor:
    flags = torch.zeros_like(case.split_valid)
    for component in range(int(case.split_valid.numel())):
        if bool(case.split_valid[component] & case.metric_component_valid[component]):
            flags[component] = component_exact(
                runtime, case, partition, component
            )
    return flags


def component_choice_logits(
    case,
    candidate_logits,
    spatial_exact: torch.Tensor,
    candidate_exact: torch.Tensor,
):
    use_component = (
        case.split_valid
        & case.metric_component_valid
        & candidate_exact
        & ~spatial_exact
    )
    src, _ = case.rag.edge_index
    edge_component = case.node_current_component[src]
    use_edge = case.editable & use_component[edge_component]
    return torch.where(use_edge, candidate_logits, case.rag.spatial_edge_logits)


def edge_choice_oracle_logits(model, case, candidate_logits):
    threshold = float(model.cfg.partition.final_merge_threshold)
    target = case.target_keep.bool()
    spatial_prediction = (
        torch.sigmoid(case.rag.spatial_edge_logits) >= threshold
    )
    candidate_prediction = torch.sigmoid(candidate_logits) >= threshold
    candidate_helps = (
        case.editable
        & (candidate_prediction == target)
        & (spatial_prediction != target)
    )
    return torch.where(
        candidate_helps,
        candidate_logits,
        case.rag.spatial_edge_logits,
    )


def target_edge_oracle_logits(case):
    target_logits = torch.where(
        case.target_keep,
        torch.full_like(case.rag.spatial_edge_logits, 20.0),
        torch.full_like(case.rag.spatial_edge_logits, -20.0),
    )
    return torch.where(
        case.editable,
        target_logits,
        case.rag.spatial_edge_logits,
    )


def current_v4_final(
    model,
    selector,
    *,
    runtime,
    case,
    base_reasoning,
    candidate_logits,
    raw_candidate_partition,
    use_threshold: float,
):
    pooled, scalar = INV46.component_feature_tables_v4(
        model,
        runtime=runtime,
        case=case,
        base_reasoning=base_reasoning,
        candidate_logits=candidate_logits,
        candidate_partition=raw_candidate_partition,
    )
    probability = torch.sigmoid(
        selector(
            pooled_edge_embedding=pooled.detach(),
            scalar_features=scalar.detach(),
        )
    )
    use_component = probability >= float(use_threshold)
    src, _ = case.rag.edge_index
    edge_component = case.node_current_component[src]
    use_edge = case.editable & use_component[edge_component]
    logits = torch.where(
        use_edge,
        candidate_logits,
        case.rag.spatial_edge_logits,
    )
    return logits, probability, use_component


def current_v3_final(
    model,
    selector,
    *,
    runtime,
    case,
    base_reasoning,
    candidate_logits,
    write_threshold: float,
):
    scalar = INV46.selective_gate_scalar_features_46(
        model,
        runtime=runtime,
        case=case,
        base_reasoning=base_reasoning,
        candidate_logits=candidate_logits,
    )
    probability = torch.sigmoid(
        selector(
            edge_embedding=case.rag.edge_embeddings.detach(),
            scalar_features=scalar.detach(),
        )
    )
    write = case.editable & (probability >= float(write_threshold))
    logits = torch.where(
        write, candidate_logits, case.rag.spatial_edge_logits
    )
    return logits, probability, write


def stabilized_base_input(
    graph,
    *,
    runtime,
    paths,
    temporal_radius: int,
    spacing: Sequence[float],
    device: torch.device,
):
    """Stabilized coordinates with Investigation-35 feature semantics."""
    node_ids, _ = INV35.selected_nodes(
        graph, runtime.t, paths.frame_count, temporal_radius
    )
    if not node_ids:
        return INV35.direct_temporal_input(
            graph,
            target_t=runtime.t,
            frame_count=paths.frame_count,
            temporal_radius=temporal_radius,
            spacing=spacing,
            dref_um=float(runtime.dref_um),
            shape_zyx=runtime.target.shape,
            device=device,
        )

    cumulative = np.asarray(
        INV46.STATE.motion.cumulative_float_zyx, dtype=np.float64
    )
    target_motion = cumulative[int(runtime.t)]
    saved: dict[int, Any] = {}
    try:
        for node_id in node_ids:
            data = graph.nodes[int(node_id)]
            source = np.asarray(data["inv35_coords_zyx"], dtype=np.float64)
            saved[int(node_id)] = data["inv35_coords_zyx"]
            data["inv35_coords_zyx"] = (
                source - cumulative[int(data["time"])] + target_motion
            ).astype(np.float32)
        return INV35.direct_temporal_input(
            graph,
            target_t=runtime.t,
            frame_count=paths.frame_count,
            temporal_radius=temporal_radius,
            spacing=spacing,
            dref_um=float(runtime.dref_um),
            shape_zyx=runtime.target.shape,
            device=device,
        )
    finally:
        for node_id, value in saved.items():
            graph.nodes[int(node_id)]["inv35_coords_zyx"] = value


def raw_coordinate_enriched_input(
    graph,
    *,
    runtime,
    paths,
    temporal_radius: int,
    spacing: Sequence[float],
    device: torch.device,
):
    """Raw positions/velocities plus the same explicit Inv46 evidence."""
    raw = INV35.direct_temporal_input(
        graph,
        target_t=runtime.t,
        frame_count=paths.frame_count,
        temporal_radius=temporal_radius,
        spacing=spacing,
        dref_um=float(runtime.dref_um),
        shape_zyx=runtime.target.shape,
        device=device,
    )
    enriched = INV46.make_temporal_input_46(
        graph,
        runtime=runtime,
        paths=paths,
        temporal_radius=temporal_radius,
        spacing=spacing,
        device=device,
    )
    if raw.graph_x.shape != enriched.graph_x.shape:
        raise RuntimeError("Raw/enriched temporal rows do not align")
    result = dataclasses.replace(raw, graph_x=raw.graph_x.clone())
    columns = [
        INV35.GX_LOG_VOLUME,
        *range(INV35.GX_BBOX.start, INV35.GX_BBOX.stop),
        *range(INV35.GX_PCA.start, INV35.GX_PCA.stop),
        INV35.GX_ELONGATION,
        INV35.GX_FLATNESS,
        INV35.GX_INTENSITY_MEAN,
        INV35.GX_INTENSITY_STD,
    ]
    result.graph_x[:, columns] = enriched.graph_x[:, columns]
    return result


def without_rejected_feature(temporal_input):
    result = dataclasses.replace(
        temporal_input, graph_x=temporal_input.graph_x.clone()
    )
    result.graph_x[:, INV35.GX_ELONGATION] = 0.0
    return result


def dominant_cell_evidence(runtime, case, component: int):
    base_labels = np.asarray(INV46._base_instance_movie()[int(runtime.t)])
    supervoxels = (
        runtime.rag.supervoxel_labels[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    node_sv = (
        runtime.rag.node_supervoxel_id.detach().cpu().numpy().astype(np.int64)
    )
    lookup = INV35.sv_label_lookup(
        supervoxels,
        base_labels,
        name=f"inv47 current identity t={runtime.t}",
    )
    node_cell = lookup[node_sv]
    current = case.node_current_component.detach().cpu().numpy()
    ids = node_cell[current == int(component)]
    ids = ids[ids > 0]
    if not ids.size:
        return None, None
    values, counts = np.unique(ids, return_counts=True)
    cell_id = int(values[int(np.argmax(counts))])
    return cell_id, INV46._cell_evidence_index().get(
        (int(runtime.t), cell_id)
    )


def classify_failure(
    *,
    raw_exact: bool,
    legal_exact: bool,
    missing_cuts: int,
    false_cuts: int,
) -> str:
    if raw_exact and not legal_exact:
        return "raw_depends_on_forbidden_logits"
    if legal_exact:
        return "legal_candidate_success"
    if not raw_exact and legal_exact:
        return "raw_harmed_by_forbidden_logits"
    if missing_cuts and false_cuts:
        return "missing_required_and_false_cuts"
    if missing_cuts:
        return "missing_required_cuts"
    if false_cuts:
        return "false_internal_cuts"
    return "partition_coordination_or_uneditable_bridge"


@torch.no_grad()
def evaluate_mode(
    name: str,
    *,
    model,
    selector,
    selector_threshold: float,
    edge_selector,
    edge_selector_threshold: float,
    frames: Sequence[int],
    loader,
    graph,
    paths,
    spacing: Sequence[float],
    temporal_radius: int,
    device: torch.device,
    input_factory: Callable[..., Any],
    collect_rows: bool,
):
    accumulators = OrderedDict(
        (
            metric_name,
            INV42.metric_accumulator(),
        )
        for metric_name in (
            "spatial",
            "raw_candidate",
            "legal_candidate",
            "current_v4_final",
            "current_v3_edge_selector_final",
            "candidate_choice_edge_oracle",
            "legal_component_oracle",
            "editable_target_edge_oracle",
            "constrained_spatial",
            "constrained_legal_candidate",
            "constrained_current_v4_final",
            "constrained_current_v3_edge_selector_final",
            "constrained_candidate_choice_edge_oracle",
            "constrained_legal_component_oracle",
            "constrained_editable_target_edge_oracle",
        )
    )
    diagnostics: list[dict[str, Any]] = []
    illegal_prediction_changes = 0
    illegal_logit_changes = 0
    pre_enforcement_violations: dict[str, int] = {
        key: 0 for key in accumulators
    }

    threshold = float(model.cfg.partition.final_merge_threshold)
    start = time.perf_counter()

    for frame in frames:
        runtime = loader.load(int(frame))
        case = INV42.build_real_case(runtime)
        temporal_input = input_factory(
            graph,
            runtime=runtime,
            paths=paths,
            temporal_radius=int(temporal_radius),
            spacing=spacing,
            device=device,
        )
        encoded = INV42.encode_case(
            model,
            runtime=runtime,
            case=case,
            temporal_input=temporal_input,
            spacing=spacing,
            device=device,
        )
        base_reasoning, raw_logits = INV46.frozen_candidate_bundle_v4(
            model,
            runtime=runtime,
            case=case,
            encoded=encoded,
        )
        legal_logits = legal_candidate_logits(case, raw_logits)

        spatial_partition = predicted_partition(
            model, case, case.rag.spatial_edge_logits
        )
        raw_partition = predicted_partition(model, case, raw_logits)
        legal_partition = predicted_partition(model, case, legal_logits)

        constrained_spatial_partition = constrained_split_only_partition(
            model, case, case.rag.spatial_edge_logits
        )
        constrained_legal_partition = constrained_split_only_partition(
            model, case, legal_logits
        )

        spatial_exact = component_flags(runtime, case, spatial_partition)
        legal_exact = component_flags(runtime, case, legal_partition)
        constrained_spatial_exact = component_flags(
            runtime, case, constrained_spatial_partition
        )
        constrained_legal_exact = component_flags(
            runtime, case, constrained_legal_partition
        )

        final_logits, selector_probability, selector_use = current_v4_final(
            model,
            selector,
            runtime=runtime,
            case=case,
            base_reasoning=base_reasoning,
            candidate_logits=raw_logits,
            raw_candidate_partition=raw_partition,
            use_threshold=selector_threshold,
        )
        v3_final_logits, v3_probability, v3_write = current_v3_final(
            model,
            edge_selector,
            runtime=runtime,
            case=case,
            base_reasoning=base_reasoning,
            candidate_logits=raw_logits,
            write_threshold=edge_selector_threshold,
        )
        edge_oracle = edge_choice_oracle_logits(model, case, legal_logits)
        component_oracle = component_choice_logits(
            case, legal_logits, spatial_exact, legal_exact
        )
        target_oracle = target_edge_oracle_logits(case)
        constrained_component_oracle = component_choice_logits(
            case,
            legal_logits,
            constrained_spatial_exact,
            constrained_legal_exact,
        )

        logits_by_name = {
            "spatial": case.rag.spatial_edge_logits,
            "raw_candidate": raw_logits,
            "legal_candidate": legal_logits,
            "current_v4_final": final_logits,
            "current_v3_edge_selector_final": v3_final_logits,
            "candidate_choice_edge_oracle": edge_oracle,
            "legal_component_oracle": component_oracle,
            "editable_target_edge_oracle": target_oracle,
        }

        for metric_name, logits in logits_by_name.items():
            INV42.update_metric_accumulator(
                accumulators[metric_name],
                model=model,
                runtime=runtime,
                case=case,
                logits=logits,
            )
            unconstrained = predicted_partition(
                model, case, logits, enforce=False
            )
            pre_enforcement_violations[metric_name] += int(
                INV35.split_only_violation_count(case, unconstrained)
            )

        constrained_logits_by_name = {
            "constrained_spatial": case.rag.spatial_edge_logits,
            "constrained_legal_candidate": legal_logits,
            "constrained_current_v4_final": final_logits,
            "constrained_current_v3_edge_selector_final": v3_final_logits,
            "constrained_candidate_choice_edge_oracle": edge_oracle,
            "constrained_legal_component_oracle": constrained_component_oracle,
            "constrained_editable_target_edge_oracle": target_oracle,
        }
        for metric_name, logits in constrained_logits_by_name.items():
            partition = constrained_split_only_partition(
                model, case, logits
            )
            update_from_partition(
                accumulators[metric_name],
                model=model,
                runtime=runtime,
                case=case,
                logits=logits,
                partition=partition,
            )
            unconstrained = constrained_split_only_partition(
                model, case, logits, enforce=False
            )
            pre_enforcement_violations[metric_name] += int(
                INV35.split_only_violation_count(case, unconstrained)
            )

        noneditable = ~case.editable
        raw_prediction = torch.sigmoid(raw_logits) >= threshold
        spatial_prediction = (
            torch.sigmoid(case.rag.spatial_edge_logits) >= threshold
        )
        illegal_prediction_changes += int(
            (noneditable & (raw_prediction != spatial_prediction)).sum().item()
        )
        illegal_logit_changes += int(
            (
                noneditable
                & ~torch.isclose(raw_logits, case.rag.spatial_edge_logits)
            ).sum().item()
        )

        if not collect_rows:
            continue

        raw_exact = component_flags(runtime, case, raw_partition)
        final_partition = predicted_partition(model, case, final_logits)
        edge_oracle_partition = predicted_partition(model, case, edge_oracle)
        component_oracle_partition = predicted_partition(
            model, case, component_oracle
        )
        target_oracle_partition = predicted_partition(model, case, target_oracle)
        constrained_final_partition = constrained_split_only_partition(
            model, case, final_logits
        )
        constrained_v3_final_partition = constrained_split_only_partition(
            model, case, v3_final_logits
        )
        v3_final_partition = predicted_partition(
            model, case, v3_final_logits
        )
        constrained_edge_oracle_partition = constrained_split_only_partition(
            model, case, edge_oracle
        )
        constrained_component_oracle_partition = (
            constrained_split_only_partition(
                model, case, constrained_component_oracle
            )
        )
        constrained_target_oracle_partition = constrained_split_only_partition(
            model, case, target_oracle
        )
        src, dst = case.rag.edge_index
        current = case.node_current_component

        for component in case.bad_components.tolist():
            component = int(component)
            nodes = torch.nonzero(current == component, as_tuple=False).flatten()
            internal = (current[src] == component) & (current[dst] == component)
            incident = (current[src] == component) | (current[dst] == component)
            editable_internal = internal & case.editable
            required_cut = editable_internal & case.cut_mask
            required_keep = editable_internal & case.keep_mask
            candidate_cut = ~raw_prediction
            missing_cuts = int((required_cut & ~candidate_cut).sum().item())
            false_cuts = int((required_keep & candidate_cut).sum().item())
            forbidden_changed = noneditable & incident & (
                raw_prediction != spatial_prediction
            )
            target_ids = torch.unique(runtime.node_target[nodes])
            target_ids = [int(v) for v in target_ids.tolist() if int(v) > 0]
            cell_id, evidence = dominant_cell_evidence(
                runtime, case, component
            )

            raw_ok = bool(raw_exact[component])
            legal_ok = bool(legal_exact[component])
            row: dict[str, Any] = {
                "mode": name,
                "frame": int(frame),
                "current_component": component,
                "dominant_base_cell_id": cell_id,
                "node_count": int(nodes.numel()),
                "target_cell_count": len(target_ids),
                "target_ids": "|".join(map(str, target_ids)),
                "spatial_exact": bool(spatial_exact[component]),
                "raw_candidate_exact": raw_ok,
                "legal_candidate_exact": legal_ok,
                "current_v4_final_exact": component_exact(
                    runtime, case, final_partition, component
                ),
                "current_v3_edge_selector_final_exact": component_exact(
                    runtime, case, v3_final_partition, component
                ),
                "candidate_choice_edge_oracle_exact": component_exact(
                    runtime, case, edge_oracle_partition, component
                ),
                "legal_component_oracle_exact": component_exact(
                    runtime, case, component_oracle_partition, component
                ),
                "editable_target_edge_oracle_exact": component_exact(
                    runtime, case, target_oracle_partition, component
                ),
                "constrained_spatial_exact": bool(
                    constrained_spatial_exact[component]
                ),
                "constrained_legal_candidate_exact": bool(
                    constrained_legal_exact[component]
                ),
                "constrained_current_v4_final_exact": component_exact(
                    runtime, case, constrained_final_partition, component
                ),
                "constrained_current_v3_edge_selector_final_exact": component_exact(
                    runtime,
                    case,
                    constrained_v3_final_partition,
                    component,
                ),
                "constrained_candidate_choice_edge_oracle_exact": component_exact(
                    runtime,
                    case,
                    constrained_edge_oracle_partition,
                    component,
                ),
                "constrained_legal_component_oracle_exact": component_exact(
                    runtime,
                    case,
                    constrained_component_oracle_partition,
                    component,
                ),
                "constrained_editable_target_edge_oracle_exact": component_exact(
                    runtime,
                    case,
                    constrained_target_oracle_partition,
                    component,
                ),
                "required_cut_edges": int(required_cut.sum().item()),
                "candidate_required_cuts_correct": int(
                    (required_cut & candidate_cut).sum().item()
                ),
                "required_keep_edges": int(required_keep.sum().item()),
                "candidate_false_cut_edges": false_cuts,
                "internal_noneditable_edges": int(
                    (internal & noneditable).sum().item()
                ),
                "incident_noneditable_prediction_changes": int(
                    forbidden_changed.sum().item()
                ),
                "incident_noneditable_logit_changes": int(
                    (
                        noneditable
                        & incident
                        & ~torch.isclose(
                            raw_logits, case.rag.spatial_edge_logits
                        )
                    ).sum().item()
                ),
                "candidate_split_probability": float(
                    torch.sigmoid(base_reasoning.split_logits[component]).item()
                ),
                "selector_use_probability": float(
                    selector_probability[component].item()
                ),
                "selector_used_candidate": bool(selector_use[component]),
                "v3_selected_writes_in_component": int(
                    (v3_write & internal).sum().item()
                ),
                "v3_mean_write_probability_in_component": float(
                    v3_probability[internal].mean().item()
                    if bool(internal.any())
                    else 0.0
                ),
                "failure_category": classify_failure(
                    raw_exact=raw_ok,
                    legal_exact=legal_ok,
                    missing_cuts=missing_cuts,
                    false_cuts=false_cuts,
                ),
            }
            if evidence is not None:
                row.update(
                    {
                        "local_volume_ratio": float(
                            math.exp(evidence.local_log_ratio)
                        ),
                        "parent_volume_ratio": float(
                            math.exp(evidence.parent_log_ratio)
                        ),
                        "prediction_error_um": float(
                            evidence.prediction_error_um
                        ),
                        "nearby_broken_tracks": int(
                            evidence.nearby_broken_count
                        ),
                        "plausible_predecessors": int(
                            evidence.plausible_predecessor_count
                        ),
                        "rejected_plausible_predecessors": int(
                            evidence.rejected_plausible_count
                        ),
                        "hard_cut_incoming": bool(evidence.hard_cut_incoming),
                        "hard_cut_outgoing": bool(evidence.hard_cut_outgoing),
                        "cutter_score": float(evidence.cutter_score),
                    }
                )
            diagnostics.append(row)

    metrics = {
        key: metric_with_counts(value) for key, value in accumulators.items()
    }
    for key, count in pre_enforcement_violations.items():
        metrics[key]["pre_enforcement_split_only_violations"] = int(count)
    return {
        "mode": name,
        "runtime_seconds": float(time.perf_counter() - start),
        "metrics": metrics,
        "raw_candidate_noneditable_logit_changes": int(illegal_logit_changes),
        "raw_candidate_noneditable_prediction_changes": int(
            illegal_prediction_changes
        ),
        "bad_components": diagnostics,
    }


def prepare(args):
    INV46.validate_inv46_args(args)
    INV46.auto_defaults(args)
    reviewed = INV42.parse_frame_spec(args.reviewed_frames)
    train_frames = INV42.parse_frame_spec(args.train_frames)
    val_frames = INV42.parse_frame_spec(args.val_frames)
    INV42.validate_args(
        args,
        reviewed=reviewed,
        train_frames=train_frames,
        val_frames=val_frames,
    )
    spacing = INV42.parse_spacing(args.spacing)
    paths = INV42.make_paths(args)
    paths.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    reviewed_set = set(reviewed)
    ignored_ids, _ = INV42.load_ignored_ids(paths, reviewed_set)
    target_store = INV42.TargetFrameStore(paths, reviewed, ignored_ids)
    requested = tuple(sorted(set(train_frames) | set(val_frames)))
    INV42.prepare_spatial_cache(
        paths,
        frames=requested,
        spacing=spacing,
        device=device,
        args=args,
    )

    graph_original = INV42.load_track_graph(paths)
    INV42.audit_and_enrich_track_graph(
        graph_original,
        paths=paths,
        target_frames=requested,
        spacing=spacing,
        temporal_radius=int(args.temporal_radius),
        max_error_um=float(args.max_coordinate_error_um),
    )
    movie = np.load(paths.base_instances, mmap_mode="r", allow_pickle=False)
    shape_zyx = tuple(int(v) for v in movie.shape[1:])
    inv45_paths = INV46.inv45_paths_from_inv42(
        paths, output=paths.output / "motion"
    )
    motion, motion_metrics = INV45.load_motion(inv45_paths, shape_zyx, spacing)
    classification = INV45.classify_components(
        inv45_paths,
        movie,
        reviewed,
        INV45.ignored_ids(inv45_paths, reviewed_set),
    )
    hypothesis_index, hypothesis_path = INV46.load_hypothesis_index(
        paths.sample,
        override=args.hypothesis_cache,
        plausible_score=float(args.cutter_plausible_score),
    )
    cutter_rows, node_evidence = INV46.build_cutter_rows(
        graph_original,
        classification=classification,
        motion=motion,
        spacing=spacing,
        shape_zyx=shape_zyx,
        near_radius_dref=float(args.cutter_near_radius_dref),
        boundary_volume_scale=float(args.cutter_boundary_volume_scale),
        hypothesis_index=hypothesis_index,
    )
    cutter_threshold, calibration = INV46.choose_threshold(
        cutter_rows,
        train_frames=set(train_frames),
        target_clean_break_rate=float(args.cutter_target_clean_break_rate),
        minimum_threshold=float(args.cutter_min_threshold),
        override=args.cutter_threshold,
    )
    graph_sanitized, cutter_decisions = INV46.apply_hard_cutter(
        graph_original,
        frame=cutter_rows,
        node_evidence=node_evidence,
        threshold=cutter_threshold,
    )
    cutter_metrics = {
        "threshold_calibration": calibration,
        "train": INV46.split_metrics(
            cutter_decisions,
            target_frames=set(train_frames),
            threshold=cutter_threshold,
        ),
        "validation": INV46.split_metrics(
            cutter_decisions,
            target_frames=set(val_frames),
            threshold=cutter_threshold,
        ),
    }
    INV46.STATE = INV46.Inv46State(
        args=args,
        paths=paths,
        spacing=tuple(float(v) for v in spacing),
        shape_zyx=shape_zyx,
        motion=motion,
        motion_metrics=motion_metrics,
        node_evidence=node_evidence,
        cutter_rows=cutter_decisions,
        cutter_threshold=float(cutter_threshold),
        cutter_metrics=cutter_metrics,
        hypothesis_cache=hypothesis_path,
        hypothesis_used=bool(hypothesis_path is not None),
    )
    INV46.COMPONENT_PHYSICAL_CACHE.clear()
    INV46.CELL_EVIDENCE_BY_KEY = None
    INV46.BASE_INSTANCE_MOVIE = None

    loader = INV42.RuntimeLoader(paths, target_store, device)
    initializer = INV46.resolve(args.resume) if args.resume else paths.checkpoint
    _, model = INV42.load_model(initializer, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    selector = INV46.Inv46ComponentSelector(
        int(model.cfg.partition.rag_hidden_dim)
    ).to(device)
    selector_payload = INV46.restore_v4(
        Path(args.v4_checkpoint).resolve(), model=model, selector=selector
    )
    selector.eval()
    edge_payload = INV46.torch_load(
        Path(args.v3_checkpoint).resolve(), map_location="cpu"
    )
    if edge_payload.get("format") != "inv46_discrete_write_selector_v3":
        raise RuntimeError("Unexpected Investigation-46 V3 checkpoint format")
    edge_selector = Inv46DiscreteWriteSelector(
        int(model.cfg.partition.rag_hidden_dim)
    ).to(device)
    edge_selector.load_state_dict(edge_payload["selector_state_dict"], strict=True)
    edge_selector.eval()
    return {
        "reviewed": reviewed,
        "train_frames": train_frames,
        "val_frames": val_frames,
        "spacing": spacing,
        "paths": paths,
        "device": device,
        "graph_original": graph_original,
        "graph_sanitized": graph_sanitized,
        "loader": loader,
        "model": model,
        "selector": selector,
        "selector_payload": selector_payload,
        "edge_selector": edge_selector,
        "edge_selector_payload": edge_payload,
        "cutter_metrics": cutter_metrics,
        "motion_metrics": motion_metrics,
    }


def main() -> int:
    parser = INV46.build_parser()
    default_output = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / SCRIPT_NAME
        / str(INV42.DEFAULT_SAMPLE)
    ).resolve()
    default_v4 = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "46_biohub_hard_cutter_temporal_training_v4"
        / str(INV42.DEFAULT_SAMPLE)
        / "best_component_selector_v4.pt"
    ).resolve()
    default_v3 = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "46_biohub_hard_cutter_temporal_training_v3"
        / str(INV42.DEFAULT_SAMPLE)
        / "best_selector_v3.pt"
    ).resolve()
    parser.set_defaults(output=default_output)
    parser.add_argument("--v4-checkpoint", type=Path, default=default_v4)
    parser.add_argument("--v3-checkpoint", type=Path, default=default_v3)
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Run only the canonical full-evidence contract audit.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    context = prepare(args)
    paths = context["paths"]
    selector_threshold = float(
        context["selector_payload"]["use_threshold"]
    )
    edge_selector_threshold = float(
        context["edge_selector_payload"]["write_threshold"]
    )

    def full_input(graph, **kwargs):
        return INV46.make_temporal_input_46(graph, **kwargs)

    def no_rejected_input(graph, **kwargs):
        return without_rejected_feature(full_input(graph, **kwargs))

    modes: list[tuple[str, Any, Callable[..., Any]]] = [
        (
            "full_hard_cutter_stabilized_physical",
            context["graph_sanitized"],
            full_input,
        )
    ]
    if not args.skip_ablations:
        modes.extend(
            [
                (
                    "no_hard_cutter_topology",
                    context["graph_original"],
                    full_input,
                ),
                (
                    "raw_coordinates_with_physical",
                    context["graph_sanitized"],
                    raw_coordinate_enriched_input,
                ),
                (
                    "stabilized_without_explicit_physical",
                    context["graph_sanitized"],
                    stabilized_base_input,
                ),
                (
                    "stabilized_physical_without_rejected_count",
                    context["graph_sanitized"],
                    no_rejected_input,
                ),
            ]
        )

    results: dict[str, Any] = {}
    prior_summary_path = paths.output / "contract_audit_summary.json"
    if args.skip_ablations and prior_summary_path.is_file():
        with prior_summary_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if isinstance(prior.get("ablations"), dict):
            results.update(prior["ablations"])
    rows: list[dict[str, Any]] = []
    for index, (name, graph, factory) in enumerate(modes):
        print(f"[Inv47] evaluating {name}", flush=True)
        result = evaluate_mode(
            name,
            model=context["model"],
            selector=context["selector"],
            selector_threshold=selector_threshold,
            edge_selector=context["edge_selector"],
            edge_selector_threshold=edge_selector_threshold,
            frames=context["val_frames"],
            loader=context["loader"],
            graph=graph,
            paths=paths,
            spacing=context["spacing"],
            temporal_radius=int(args.temporal_radius),
            device=context["device"],
            input_factory=factory,
            collect_rows=index == 0,
        )
        rows.extend(result.pop("bad_components"))
        results[name] = result

    canonical = results["full_hard_cutter_stabilized_physical"]
    summary = {
        "investigation": SCRIPT_NAME,
        "development_validation_frames": list(map(int, context["val_frames"])),
        "validation_is_untouched_holdout": False,
        "initializer": str(args.resume),
        "v4_checkpoint": str(Path(args.v4_checkpoint).resolve()),
        "v3_checkpoint": str(Path(args.v3_checkpoint).resolve()),
        "selector_threshold_calibrated_on_train": selector_threshold,
        "edge_selector_threshold_calibrated_on_train": edge_selector_threshold,
        "candidate_contract": (
            "legal logits equal temporal candidate only on case.editable; "
            "all other logits equal frozen spatial"
        ),
        "cutter": context["cutter_metrics"],
        "motion": context["motion_metrics"],
        "canonical": canonical,
        "ablations": results,
        "total_runtime_seconds": float(time.perf_counter() - started),
    }
    output = paths.output
    atomic_json(output / "contract_audit_summary.json", summary)
    atomic_csv(output / "bad_validation_components.csv", pd.DataFrame(rows))

    table_rows = []
    for metric_name, metric in canonical["metrics"].items():
        table_rows.append({"method": metric_name, **metric})
    atomic_csv(output / "canonical_metrics.csv", pd.DataFrame(table_rows))

    ablation_rows = []
    for mode_name, result in results.items():
        for candidate_name in (
            "raw_candidate",
            "legal_candidate",
            "constrained_legal_candidate",
            "current_v3_edge_selector_final",
            "constrained_current_v3_edge_selector_final",
        ):
            if candidate_name not in result["metrics"]:
                continue
            ablation_rows.append(
                {
                    "mode": mode_name,
                    "candidate_contract": candidate_name,
                    **result["metrics"][candidate_name],
                    "runtime_seconds": result["runtime_seconds"],
                }
            )
    atomic_csv(output / "candidate_input_ablations.csv", pd.DataFrame(ablation_rows))

    print(json.dumps(summary["canonical"], indent=2), flush=True)
    print(f"[Inv47] outputs: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
