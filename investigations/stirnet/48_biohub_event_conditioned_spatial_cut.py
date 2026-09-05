from __future__ import annotations

r"""
Investigation 48 — event-conditioned spatial CUT reasoning.

Research question
-----------------
The temporal problem is decomposed into two explicit decisions:

    1. WHICH current spatial instance is suspicious?

       This should be largely determined by temporal/statistical evidence:
       - a newborn track after the hard cutter,
       - nearby predecessor tracks that broke at the same stabilized location,
       - abnormal local/current-parent volume,
       - hard-cut status,
       - stabilized prediction error,
       - plausible/rejected Trackastra predecessor evidence,
       - boundary status.

    2. WHERE inside that suspicious instance should the split occur?

       This should use the rich frozen spatial representation that STIR-Net
       already computes:
       - RAG edge embedding and raw edge features,
       - left/right RAG node embeddings and raw node features,
       - morphology embeddings when present,
       - the rich InstanceTokenizer spatial token,
       - frozen spatial merge logit.

This experiment does NOT train the temporal transformer.  It freezes the
Investigation-42 temporal/spatial initializer, preserves Investigation-46 hard
cutter + global-motion compensation, and trains two small experiment-local
heads:

    EventDetector:
        explicit temporal/statistical component features -> merge-event score

    EventConditionedSpatialCutHead:
        rich frozen spatial edge representation
        + broadcast event embedding
        -> CUT probability for each editable RAG edge

At inference, the temporal model is split-only by construction:

    if component_event_score < threshold:
        preserve every spatial edge exactly

    if component_event_score >= threshold:
        only edges whose spatial CUT head exceeds its threshold are forced CUT

The final partition is solved with Investigation-47's in-solver
node_parent_component constraint.  There is no post-hoc whole-frame shortcut.

Important diagnostics
---------------------
The script evaluates:

    spatial
    existing constrained V3 selector
    constrained legal Inv46 candidate

    Inv48 rule-event + learned spatial CUT head
    Inv48 learned-event + learned spatial CUT head
    Inv48 oracle-event + learned spatial CUT head

    Inv48 learned-event + target-edge oracle
    Inv48 oracle-event + target-edge oracle

The oracle-event + learned-spatial-head row answers:

    "If temporal localization tells us the correct component, is the rich
     spatial representation sufficient to find the right RAG CUT edges?"

The learned-event + target-edge-oracle row answers:

    "If edge localization were perfect, how much does event detection alone
     limit the final result?"

Thresholds are calibrated on TRAIN frames only.  Frames 30-39 remain the
already-used development validation range and are NOT an untouched holdout.

Typical command
---------------

    python .\investigations\stirnet\48_biohub_event_conditioned_spatial_cut.py `
        --event-steps 500 `
        --cut-steps 700 `
        --event-lr 3e-4 `
        --cut-lr 3e-4 `
        --max-clean-event-rate 0.05 `
        --max-keep-cut-rate 0.05

This is an experiment-local architecture.  It does not modify production
STIR-Net weights.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =============================================================================
# Repository / prior investigations
# =============================================================================


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
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INV47 = load_module(
    ROOT
    / "investigations"
    / "stirnet"
    / "47_biohub_temporal_candidate_contract_audit.py",
    "_inv48_inv47",
)
INV46 = INV47.INV46
INV42 = INV47.INV42
INV35 = INV47.INV35

SCRIPT_NAME = "48_biohub_event_conditioned_spatial_cut"
OBJECTIVE_VERSION = 2

EVENT_FEATURE_NAMES = [
    "local_volume_anomaly",
    "positive_parent_growth",
    "hard_cut_incoming",
    "hard_cut_outgoing",
    "is_newborn_after_sanitize",
    "nearby_broken_count",
    "newborn_x_nearby_broken",
    "newborn_x_hard_cut",
    "newborn_x_hard_cut_x_broken",
    "stabilized_prediction_error",
    "cutter_score",
    "plausible_predecessors",
    "rejected_plausible_predecessors",
    "incoming_association_score",
    "boundary",
    "accepted_predecessor_count",
    "accepted_successor_count",
]

EVENT_DIM = len(EVENT_FEATURE_NAMES)
EVENT_EMBED_DIM = 16


# =============================================================================
# Small experiment-local models
# =============================================================================


class EventDetector(nn.Module):
    """Explicit temporal/statistical event detector at current-component level."""

    def __init__(self, input_dim: int, embed_dim: int = EVENT_EMBED_DIM) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 48),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 32),
            nn.SiLU(),
            nn.Linear(32, self.embed_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(self.embed_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.encoder(self.norm(x.float()))
        logit = self.classifier(embedding).squeeze(-1)
        return embedding, logit


class EventConditionedSpatialCutHead(nn.Module):
    """Find the exact RAG edge to CUT inside a suspicious current instance."""

    def __init__(
        self,
        spatial_dim: int,
        event_embed_dim: int = EVENT_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        self.event_embed_dim = int(event_embed_dim)

        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.event_norm = nn.LayerNorm(self.event_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.spatial_dim + self.event_embed_dim + 1, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        spatial_features: Tensor,
        event_embedding: Tensor,
        event_probability: Tensor,
    ) -> Tensor:
        if spatial_features.ndim != 2:
            raise ValueError("spatial_features must be [E,D]")
        if event_embedding.ndim != 2:
            raise ValueError("event_embedding must be [E,D_event]")
        if event_probability.ndim != 1:
            raise ValueError("event_probability must be [E]")
        if not (
            spatial_features.shape[0]
            == event_embedding.shape[0]
            == event_probability.shape[0]
        ):
            raise ValueError("edge feature row mismatch")

        x = torch.cat(
            [
                self.spatial_norm(spatial_features.float()),
                self.event_norm(event_embedding.float()),
                event_probability.float()[:, None],
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


# =============================================================================
# Feature containers
# =============================================================================


@dataclass
class ComponentDataset:
    x: Tensor
    y: Tensor
    metadata: pd.DataFrame

    @property
    def positive_count(self) -> int:
        return int((self.y > 0.5).sum().item())

    @property
    def negative_count(self) -> int:
        return int((self.y <= 0.5).sum().item())


@dataclass
class EdgeDataset:
    spatial_x: Tensor
    event_x: Tensor
    y_cut: Tensor
    metadata: pd.DataFrame

    @property
    def cut_count(self) -> int:
        return int((self.y_cut > 0.5).sum().item())

    @property
    def keep_count(self) -> int:
        return int((self.y_cut <= 0.5).sum().item())


@dataclass
class ExtractionBundle:
    components: ComponentDataset
    edges: EdgeDataset
    spatial_dim: int


# =============================================================================
# Generic utilities
# =============================================================================


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return result if math.isfinite(result) else 0.0


def safe_log_ratio(numerator: float, denominator: float) -> float:
    numerator = max(float(numerator), 1.0e-8)
    denominator = max(float(denominator), 1.0e-8)
    return float(math.log(numerator / denominator))


def clipped(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), float(lo)), float(hi)))


def normalized_count(value: int, cap: int = 3) -> float:
    return min(max(int(value), 0), int(cap)) / float(max(int(cap), 1))


# =============================================================================
# Explicit temporal-event evidence
# =============================================================================


def graph_status_index(graph) -> dict[tuple[int, int], dict[str, Any]]:
    """Accepted sanitized topology indexed by canonical (frame, cell_id)."""

    node_key: dict[int, tuple[int, int]] = {}
    node_time: dict[int, int] = {}
    result: dict[tuple[int, int], dict[str, Any]] = {}

    for node_id, data in graph.nodes(data=True):
        if "time" not in data or "inv42_cell_id" not in data:
            continue
        node_id = int(node_id)
        t = int(data["time"])
        cell_id = int(data["inv42_cell_id"])
        key = (t, cell_id)
        node_key[node_id] = key
        node_time[node_id] = t
        result[key] = {
            "node_id": node_id,
            "accepted_predecessor_count": 0,
            "accepted_successor_count": 0,
        }

    for u, v in graph.edges():
        u = int(u)
        v = int(v)
        if u not in node_key or v not in node_key:
            continue
        tu = node_time[u]
        tv = node_time[v]
        if tu == tv:
            continue
        if tv < tu:
            u, v = v, u
            tu, tv = tv, tu

        ku = node_key[u]
        kv = node_key[v]
        result[ku]["accepted_successor_count"] += 1
        result[kv]["accepted_predecessor_count"] += 1

    for value in result.values():
        value["is_newborn_after_sanitize"] = (
            int(value["accepted_predecessor_count"]) == 0
        )

    return result


COMPONENT_CELL_ID_CACHE: dict[int, np.ndarray] = {}


def component_cell_ids(runtime, case) -> np.ndarray:
    """Map current component index -> authoritative persisted current cell ID."""

    t = int(runtime.t)
    cached = COMPONENT_CELL_ID_CACHE.get(t)
    if cached is not None:
        return cached.copy()

    base_movie = INV46._base_instance_movie()
    base_labels = np.asarray(base_movie[t])

    supervoxels = (
        runtime.rag.supervoxel_labels[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    node_sv = (
        runtime.rag.node_supervoxel_id
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    lookup = INV35.sv_label_lookup(
        supervoxels,
        base_labels,
        name=f"inv48 current identity t={t}",
    )
    node_cell = lookup[node_sv]

    current = (
        case.node_current_component
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    component_count = int(case.split_target.numel())
    result = np.zeros(component_count, dtype=np.int64)

    for component in range(component_count):
        rows = np.flatnonzero(current == component)
        if rows.size == 0:
            continue
        ids = node_cell[rows]
        ids = ids[ids > 0]
        if ids.size == 0:
            continue
        unique, counts = np.unique(ids, return_counts=True)
        result[component] = int(unique[int(np.argmax(counts))])

    COMPONENT_CELL_ID_CACHE[t] = result.copy()
    return result


def component_event_features(
    runtime,
    case,
    *,
    status_index: dict[tuple[int, int], dict[str, Any]],
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Build explicit event features for every current spatial component."""

    device = case.rag.spatial_edge_logits.device
    component_count = int(case.split_target.numel())
    rows = np.zeros((component_count, EVENT_DIM), dtype=np.float32)
    metadata: list[dict[str, Any]] = []

    cell_ids = component_cell_ids(runtime, case)
    evidence_index = INV46._cell_evidence_index()
    dref = max(float(runtime.dref_um), 1.0e-6)
    legacy_volume_scale = max(
        float(getattr(INV46, "LEGACY_VOLUME_LOG_SCALE", math.log(1.25))),
        1.0e-6,
    )
    cutter_threshold = max(
        float(getattr(INV46.STATE, "cutter_threshold", 1.0)),
        1.0e-6,
    )

    for component in range(component_count):
        cell_id = int(cell_ids[component])
        evidence = evidence_index.get((int(runtime.t), cell_id))
        status = status_index.get((int(runtime.t), cell_id), {})

        if evidence is None:
            local_log = 0.0
            positive_growth = 0.0
            hard_in = False
            hard_out = False
            prediction_error = 0.0
            nearby_broken = 0
            cutter_score = 0.0
            plausible = 0
            rejected = 0
            association = 0.0
            boundary = False
            volume = 0.0
            local_median = 0.0
            parent_log = 0.0
        else:
            local_log = finite_or_zero(evidence.local_log_ratio)
            positive_growth = finite_or_zero(
                evidence.positive_parent_growth_units
            )
            hard_in = bool(evidence.hard_cut_incoming)
            hard_out = bool(evidence.hard_cut_outgoing)
            prediction_error = finite_or_zero(evidence.prediction_error_um)
            nearby_broken = int(evidence.nearby_broken_count)
            cutter_score = finite_or_zero(evidence.cutter_score)
            plausible = int(evidence.plausible_predecessor_count)
            rejected = int(evidence.rejected_plausible_count)
            association = finite_or_zero(evidence.incoming_association_score)
            boundary = bool(evidence.boundary)
            volume = finite_or_zero(evidence.volume_um3)
            local_median = finite_or_zero(evidence.local_median_volume_um3)
            parent_log = finite_or_zero(evidence.parent_log_ratio)

        pred_count = int(status.get("accepted_predecessor_count", 0))
        succ_count = int(status.get("accepted_successor_count", 0))
        is_newborn = bool(
            status.get(
                "is_newborn_after_sanitize",
                False,
            )
        )

        local_units = local_log / legacy_volume_scale
        local_feature = clipped(local_units / 5.0, -1.0, 1.0)
        growth_feature = clipped(positive_growth / 5.0, 0.0, 1.0)
        broken_feature = normalized_count(nearby_broken, 3)
        prediction_feature = clipped(
            prediction_error / (5.0 * dref),
            0.0,
            1.0,
        )
        cutter_feature = math.tanh(cutter_score / cutter_threshold)
        plausible_feature = normalized_count(plausible, 3)
        rejected_feature = normalized_count(rejected, 3)
        association_feature = clipped(association, 0.0, 1.0)

        rows[component] = np.asarray(
            [
                local_feature,
                growth_feature,
                float(hard_in),
                float(hard_out),
                float(is_newborn),
                broken_feature,
                float(is_newborn) * broken_feature,
                float(is_newborn and hard_in),
                float(is_newborn and hard_in) * broken_feature,
                prediction_feature,
                float(cutter_feature),
                plausible_feature,
                rejected_feature,
                association_feature,
                float(boundary),
                normalized_count(pred_count, 3),
                normalized_count(succ_count, 3),
            ],
            dtype=np.float32,
        )

        local_ratio = (
            math.exp(local_log)
            if math.isfinite(local_log)
            else 1.0
        )
        parent_ratio = (
            math.exp(parent_log)
            if math.isfinite(parent_log)
            else 1.0
        )

        metadata.append(
            {
                "frame": int(runtime.t),
                "component": int(component),
                "cell_id": int(cell_id),
                "target_merge": bool(
                    case.split_target[component].item() > 0.5
                ),
                "trusted_component": bool(
                    case.split_valid[component].item()
                    and case.metric_component_valid[component].item()
                ),
                "is_newborn_after_sanitize": bool(is_newborn),
                "hard_cut_incoming": bool(hard_in),
                "hard_cut_outgoing": bool(hard_out),
                "nearby_broken_count": int(nearby_broken),
                "newborn_and_broken": bool(
                    is_newborn and nearby_broken >= 1
                ),
                "newborn_hardcut_and_broken": bool(
                    is_newborn and hard_in and nearby_broken >= 1
                ),
                "local_volume_ratio": float(local_ratio),
                "parent_volume_ratio": float(parent_ratio),
                "prediction_error_um": float(prediction_error),
                "dref_um": float(dref),
                "cutter_score": float(cutter_score),
                "plausible_predecessors": int(plausible),
                "rejected_plausible_predecessors": int(rejected),
                "incoming_association_score": float(association),
                "boundary": bool(boundary),
                "accepted_predecessor_count": int(pred_count),
                "accepted_successor_count": int(succ_count),
                "volume_um3": float(volume),
                "local_median_volume_um3": float(local_median),
            }
        )

    return (
        torch.from_numpy(rows).to(device=device, dtype=torch.float32),
        metadata,
    )


# =============================================================================
# Rich frozen spatial representation
# =============================================================================


def component_spatial_tokens(encoded, case) -> Tensor:
    """Pool the rich InstanceTokenizer spatial token onto current components."""

    instances = encoded.instances
    source_tokens = (
        instances.spatial_tokens
        if instances.spatial_tokens is not None
        else instances.tokens
    )
    source_tokens = source_tokens.detach().float()

    component_count = int(case.split_target.numel())
    token_dim = int(source_tokens.shape[1]) if source_tokens.ndim == 2 else 0
    result = source_tokens.new_zeros((component_count, token_dim))

    node_to_instance = instances.node_to_instance.long()
    current = case.node_current_component.long()

    for component in range(component_count):
        node_rows = torch.nonzero(
            current == component,
            as_tuple=False,
        ).flatten()
        if node_rows.numel() == 0:
            continue

        instance_ids = torch.unique(node_to_instance[node_rows])
        instance_ids = instance_ids[
            (instance_ids >= 0)
            & (instance_ids < int(source_tokens.shape[0]))
        ]
        if instance_ids.numel() == 0:
            continue
        result[component] = source_tokens[instance_ids].mean(dim=0)

    return result


def rich_spatial_edge_features(encoded, case) -> Tensor:
    """Expose the full frozen spatial evidence to the new CUT head."""

    rag = case.rag
    edge_count = int(rag.edge_index.shape[1])
    if edge_count == 0:
        return rag.spatial_edge_logits.new_zeros((0, 1))

    src, dst = rag.edge_index
    current = case.node_current_component.long()

    parts: list[Tensor] = []

    def append_feature(value: Tensor | None) -> None:
        if value is None:
            return
        if value.ndim != 2 or int(value.shape[0]) != edge_count:
            raise RuntimeError(
                f"Invalid edge feature shape {tuple(value.shape)} "
                f"for E={edge_count}"
            )
        parts.append(value.detach().float())

    append_feature(rag.edge_embeddings)

    node_emb = rag.node_embeddings.detach().float()
    append_feature(node_emb[src])
    append_feature(node_emb[dst])
    append_feature((node_emb[src] - node_emb[dst]).abs())

    if rag.edge_features is not None:
        append_feature(rag.edge_features)

    if rag.node_features is not None and rag.node_features.ndim == 2:
        raw_node = rag.node_features.detach().float()
        append_feature(raw_node[src])
        append_feature(raw_node[dst])
        append_feature((raw_node[src] - raw_node[dst]).abs())

    if rag.edge_morphology_embeddings is not None:
        append_feature(rag.edge_morphology_embeddings)

    if rag.node_morphology_embeddings is not None:
        morph = rag.node_morphology_embeddings.detach().float()
        append_feature(morph[src])
        append_feature(morph[dst])
        append_feature((morph[src] - morph[dst]).abs())

    component_token = component_spatial_tokens(encoded, case)
    append_feature(component_token[current[src]])

    spatial_logit = rag.spatial_edge_logits.detach().float()
    spatial_prob = spatial_logit.sigmoid()
    uncertainty = torch.exp(-spatial_logit.abs())
    append_feature(
        torch.stack(
            [
                spatial_logit,
                spatial_prob,
                uncertainty,
            ],
            dim=-1,
        )
    )

    result = torch.cat(parts, dim=-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite rich spatial edge feature")
    return result


# =============================================================================
# Dataset extraction
# =============================================================================


@torch.inference_mode()
def extract_frames(
    *,
    context,
    frames: Sequence[int],
    args,
    purpose: str,
) -> ExtractionBundle:
    model = context["model"]
    loader = context["loader"]
    graph = context["graph_sanitized"]
    paths = context["paths"]
    spacing = context["spacing"]
    device = context["device"]
    status_index = graph_status_index(graph)

    component_x_chunks: list[Tensor] = []
    component_y_chunks: list[Tensor] = []
    component_rows: list[dict[str, Any]] = []

    edge_x_chunks: list[Tensor] = []
    edge_event_chunks: list[Tensor] = []
    edge_y_chunks: list[Tensor] = []
    edge_rows: list[dict[str, Any]] = []

    spatial_dim: int | None = None

    for ordinal, t in enumerate(frames, 1):
        started = time.perf_counter()
        runtime = loader.load(int(t))
        case = INV42.build_real_case(runtime)

        temporal_input = INV46.make_temporal_input_46(
            graph,
            runtime=runtime,
            paths=paths,
            temporal_radius=int(args.temporal_radius),
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

        event_x, event_meta = component_event_features(
            runtime,
            case,
            status_index=status_index,
        )
        spatial_edge_x = rich_spatial_edge_features(encoded, case)

        if spatial_dim is None:
            spatial_dim = int(spatial_edge_x.shape[1])
        elif int(spatial_edge_x.shape[1]) != spatial_dim:
            raise RuntimeError(
                f"Spatial feature dimension changed at t={t}: "
                f"{spatial_edge_x.shape[1]} vs {spatial_dim}"
            )

        component_valid = (
            case.split_valid
            & case.metric_component_valid
        )
        component_indices = torch.nonzero(
            component_valid,
            as_tuple=False,
        ).flatten()
        if component_indices.numel():
            component_x_chunks.append(
                event_x[component_indices]
                .detach()
                .float()
                .cpu()
            )
            component_y_chunks.append(
                case.split_target[component_indices]
                .detach()
                .float()
                .cpu()
            )
            for component in component_indices.tolist():
                row = dict(event_meta[int(component)])
                row["purpose"] = str(purpose)
                component_rows.append(row)

        editable = case.editable
        edge_indices = torch.nonzero(
            editable,
            as_tuple=False,
        ).flatten()
        if edge_indices.numel():
            src, _dst = case.rag.edge_index
            edge_component = case.node_current_component[src]
            edge_x_chunks.append(
                spatial_edge_x[edge_indices]
                .detach()
                .float()
                .cpu()
            )
            edge_event_chunks.append(
                event_x[
                    edge_component[edge_indices]
                ]
                .detach()
                .float()
                .cpu()
            )
            edge_target_cut = (
                ~case.target_keep[edge_indices].bool()
            ).float()
            edge_y_chunks.append(
                edge_target_cut.detach().cpu()
            )

            spatial_prob = torch.sigmoid(
                case.rag.spatial_edge_logits
            )
            for edge_row in edge_indices.tolist():
                component = int(
                    edge_component[edge_row].item()
                )
                row = {
                    "purpose": str(purpose),
                    "frame": int(t),
                    "edge_row": int(edge_row),
                    "component": component,
                    "target_cut": bool(
                        not case.target_keep[edge_row].item()
                    ),
                    "component_target_merge": bool(
                        case.split_target[component].item() > 0.5
                    ),
                    "spatial_keep_probability": float(
                        spatial_prob[edge_row].item()
                    ),
                }
                row.update(
                    {
                        f"event_{name}": float(
                            event_x[component, idx].item()
                        )
                        for idx, name in enumerate(
                            EVENT_FEATURE_NAMES
                        )
                    }
                )
                edge_rows.append(row)

        print(
            f"[Inv48 extract {purpose}] "
            f"t={int(t):03d} "
            f"{ordinal}/{len(frames)} "
            f"components={int(component_indices.numel())} "
            f"editable_edges={int(edge_indices.numel())} "
            f"time={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    if not component_x_chunks:
        raise RuntimeError(
            f"No trusted component examples extracted for {purpose}"
        )
    if not edge_x_chunks:
        raise RuntimeError(
            f"No editable edge examples extracted for {purpose}"
        )
    if spatial_dim is None:
        raise RuntimeError("Could not determine spatial feature dimension")

    components = ComponentDataset(
        x=torch.cat(component_x_chunks, dim=0),
        y=torch.cat(component_y_chunks, dim=0),
        metadata=pd.DataFrame(component_rows),
    )
    edges = EdgeDataset(
        spatial_x=torch.cat(edge_x_chunks, dim=0),
        event_x=torch.cat(edge_event_chunks, dim=0),
        y_cut=torch.cat(edge_y_chunks, dim=0),
        metadata=pd.DataFrame(edge_rows),
    )

    if components.positive_count <= 0 or components.negative_count <= 0:
        raise RuntimeError(
            f"{purpose}: component data needs both classes; "
            f"merge={components.positive_count} "
            f"clean={components.negative_count}"
        )
    if edges.cut_count <= 0 or edges.keep_count <= 0:
        raise RuntimeError(
            f"{purpose}: edge data needs CUT and KEEP; "
            f"CUT={edges.cut_count} KEEP={edges.keep_count}"
        )

    return ExtractionBundle(
        components=components,
        edges=edges,
        spatial_dim=int(spatial_dim),
    )


# =============================================================================
# Threshold / classification metrics
# =============================================================================


def pairwise_auc(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    pos = probability[target == 1]
    neg = probability[target == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    greater = float((pos[:, None] > neg[None, :]).sum())
    equal = float((pos[:, None] == neg[None, :]).sum())
    return (greater + 0.5 * equal) / float(pos.size * neg.size)


def binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    prediction = probability >= float(threshold)

    positive = target == 1
    negative = target == 0

    tp = int((prediction & positive).sum())
    fn = int((~prediction & positive).sum())
    fp = int((prediction & negative).sum())
    tn = int((~prediction & negative).sum())

    recall = tp / max(tp + fn, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    return {
        "threshold": float(threshold),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "precision": float(precision),
        "auc": float(pairwise_auc(probability, target)),
    }


def choose_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    max_false_positive_rate: float,
) -> tuple[float, dict[str, Any]]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    candidates = (
        [1.0 + 1.0e-7]
        + sorted(
            set(float(v) for v in probability.tolist()),
            reverse=True,
        )
        + [0.0]
    )

    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(
            probability,
            target,
            threshold,
        )
        if (
            metrics["false_positive_rate"]
            <= float(max_false_positive_rate) + 1.0e-12
        ):
            feasible.append(
                (
                    metrics["recall"],
                    metrics["precision"],
                    -metrics["false_positive_rate"],
                    float(threshold),
                    metrics,
                )
            )

    if not feasible:
        threshold = 1.0 + 1.0e-7
        return threshold, binary_metrics(
            probability,
            target,
            threshold,
        )

    feasible.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
        ),
        reverse=True,
    )
    return float(feasible[0][3]), dict(feasible[0][4])


# =============================================================================
# Interpretable rule baseline
# =============================================================================


def rule_mask(
    frame: pd.DataFrame,
    *,
    hard_cut_required: bool,
    broken_min: int,
    local_ratio_min: float,
    allow_boundary: bool,
) -> np.ndarray:
    # Pandas may expose read-only NumPy views under newer NumPy/Pandas
    # combinations.  Build an owned boolean array and avoid in-place boolean
    # updates so this diagnostic remains version-robust.
    result = np.asarray(
        frame["is_newborn_after_sanitize"].to_numpy(),
        dtype=bool,
    ).copy()

    if hard_cut_required:
        result = np.logical_and(
            result,
            np.asarray(
                frame["hard_cut_incoming"].to_numpy(),
                dtype=bool,
            ),
        )

    result = np.logical_and(
        result,
        frame["nearby_broken_count"].to_numpy(dtype=np.int64)
        >= int(broken_min),
    )
    result = np.logical_and(
        result,
        frame["local_volume_ratio"].to_numpy(dtype=np.float64)
        >= float(local_ratio_min),
    )

    if not allow_boundary:
        result = np.logical_and(
            result,
            ~np.asarray(
                frame["boundary"].to_numpy(),
                dtype=bool,
            ),
        )

    return np.asarray(result, dtype=bool)


def search_rule(
    train_components: ComponentDataset,
    *,
    max_clean_event_rate: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = train_components.metadata.reset_index(drop=True)
    target = train_components.y.numpy().astype(np.int64, copy=False)

    records: list[dict[str, Any]] = []

    for hard_cut_required in (False, True):
        for broken_min in (0, 1, 2, 3):
            for local_ratio_min in (
                1.0,
                1.10,
                1.25,
                1.50,
                2.0,
            ):
                for allow_boundary in (False, True):
                    prediction = rule_mask(
                        frame,
                        hard_cut_required=hard_cut_required,
                        broken_min=broken_min,
                        local_ratio_min=local_ratio_min,
                        allow_boundary=allow_boundary,
                    )
                    positive = target == 1
                    negative = target == 0
                    tp = int((prediction & positive).sum())
                    fp = int((prediction & negative).sum())
                    fn = int((~prediction & positive).sum())
                    tn = int((~prediction & negative).sum())

                    recall = tp / max(tp + fn, 1)
                    fpr = fp / max(fp + tn, 1)
                    precision = tp / max(tp + fp, 1)

                    records.append(
                        {
                            "hard_cut_required": bool(hard_cut_required),
                            "broken_min": int(broken_min),
                            "local_ratio_min": float(local_ratio_min),
                            "allow_boundary": bool(allow_boundary),
                            "tp": tp,
                            "fn": fn,
                            "fp": fp,
                            "tn": tn,
                            "recall": float(recall),
                            "clean_event_rate": float(fpr),
                            "precision": float(precision),
                            "feasible": bool(
                                fpr
                                <= float(max_clean_event_rate)
                                + 1.0e-12
                            ),
                        }
                    )

    table = pd.DataFrame(records)
    feasible = table[table["feasible"]].copy()
    if feasible.empty:
        feasible = table.copy()

    feasible = feasible.sort_values(
        [
            "recall",
            "precision",
            "clean_event_rate",
            "local_ratio_min",
        ],
        ascending=[False, False, True, False],
    )
    best = feasible.iloc[0].to_dict()
    return best, table


def rule_metrics(
    dataset: ComponentDataset,
    rule: dict[str, Any],
) -> dict[str, Any]:
    frame = dataset.metadata.reset_index(drop=True)
    target = dataset.y.numpy().astype(np.int64, copy=False)
    prediction = rule_mask(
        frame,
        hard_cut_required=bool(rule["hard_cut_required"]),
        broken_min=int(rule["broken_min"]),
        local_ratio_min=float(rule["local_ratio_min"]),
        allow_boundary=bool(rule["allow_boundary"]),
    )

    positive = target == 1
    negative = target == 0
    tp = int((prediction & positive).sum())
    fn = int((~prediction & positive).sum())
    fp = int((prediction & negative).sum())
    tn = int((~prediction & negative).sum())

    return {
        "merge_count": int(positive.sum()),
        "clean_count": int(negative.sum()),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "merge_recall": float(tp / max(tp + fn, 1)),
        "clean_event_rate": float(fp / max(fp + tn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
    }


# =============================================================================
# Training
# =============================================================================


def balanced_batch_indices(
    target: Tensor,
    *,
    per_class: int,
    generator: torch.Generator,
) -> Tensor:
    positive = torch.nonzero(
        target > 0.5,
        as_tuple=False,
    ).flatten()
    negative = torch.nonzero(
        target <= 0.5,
        as_tuple=False,
    ).flatten()

    if positive.numel() == 0 or negative.numel() == 0:
        raise RuntimeError("Balanced sampling needs both classes")

    pos = positive[
        torch.randint(
            0,
            int(positive.numel()),
            (int(per_class),),
            generator=generator,
        )
    ]
    neg = negative[
        torch.randint(
            0,
            int(negative.numel()),
            (int(per_class),),
            generator=generator,
        )
    ]
    batch = torch.cat([pos, neg], dim=0)
    order = torch.randperm(
        int(batch.numel()),
        generator=generator,
    )
    return batch[order]


@torch.inference_mode()
def event_probabilities(
    model: EventDetector,
    x: Tensor,
    device: torch.device,
) -> tuple[np.ndarray, Tensor]:
    model.eval()
    embedding, logits = model(x.to(device))
    probability = torch.sigmoid(logits)
    return (
        probability.detach().float().cpu().numpy().astype(np.float64),
        embedding.detach().float().cpu(),
    )


def train_event_detector(
    dataset: ComponentDataset,
    *,
    device: torch.device,
    args,
) -> tuple[EventDetector, list[dict[str, Any]]]:
    model = EventDetector(EVENT_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.event_lr),
        weight_decay=float(args.event_weight_decay),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 48_101)
    history: list[dict[str, Any]] = []

    for step in range(1, int(args.event_steps) + 1):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)

        batch = balanced_batch_indices(
            dataset.y,
            per_class=int(args.event_batch_per_class),
            generator=generator,
        )
        x = dataset.x[batch].to(device)
        y = dataset.y[batch].to(device)

        _embedding, logits = model(x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite event loss at step {step}"
            )

        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(args.grad_clip),
        )
        optimizer.step()

        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(
                torch.as_tensor(grad).detach().cpu()
            ),
        }

        if (
            step == 1
            or step % int(args.print_every) == 0
            or step == int(args.event_steps)
        ):
            probability, _ = event_probabilities(
                model,
                dataset.x,
                device,
            )
            metrics = binary_metrics(
                probability,
                dataset.y.numpy().astype(np.int64),
                0.5,
            )
            row.update(
                {
                    "train_auc": metrics["auc"],
                    "train_recall_at_0.5": metrics["recall"],
                    "train_fpr_at_0.5": metrics[
                        "false_positive_rate"
                    ],
                }
            )
            print(
                f"[event {step:04d}/{int(args.event_steps)}] "
                f"loss={row['loss']:.5f} "
                f"grad={row['grad_norm']:.3f} "
                f"auc={metrics['auc']:.4f} "
                f"recall@.5={metrics['recall']:.4f} "
                f"clean@.5={metrics['false_positive_rate']:.4f}",
                flush=True,
            )

        history.append(row)

    return model, history


@torch.inference_mode()
def cut_probabilities(
    event_model: EventDetector,
    cut_model: EventConditionedSpatialCutHead,
    *,
    spatial_x: Tensor,
    event_x: Tensor,
    device: torch.device,
) -> np.ndarray:
    event_model.eval()
    cut_model.eval()

    event_embedding, event_logits = event_model(
        event_x.to(device)
    )
    event_probability = torch.sigmoid(event_logits)
    logits = cut_model(
        spatial_features=spatial_x.to(device),
        event_embedding=event_embedding,
        event_probability=event_probability,
    )
    return (
        torch.sigmoid(logits)
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def train_cut_head(
    dataset: EdgeDataset,
    *,
    event_model: EventDetector,
    spatial_dim: int,
    device: torch.device,
    args,
) -> tuple[EventConditionedSpatialCutHead, list[dict[str, Any]]]:
    for parameter in event_model.parameters():
        parameter.requires_grad_(False)
    event_model.eval()

    model = EventConditionedSpatialCutHead(
        spatial_dim=spatial_dim,
        event_embed_dim=EVENT_EMBED_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.cut_lr),
        weight_decay=float(args.cut_weight_decay),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 48_202)
    history: list[dict[str, Any]] = []

    for step in range(1, int(args.cut_steps) + 1):
        model.train(True)
        optimizer.zero_grad(set_to_none=True)

        batch = balanced_batch_indices(
            dataset.y_cut,
            per_class=int(args.cut_batch_per_class),
            generator=generator,
        )

        spatial_x = dataset.spatial_x[batch].to(device)
        event_x = dataset.event_x[batch].to(device)
        target = dataset.y_cut[batch].to(device)

        with torch.no_grad():
            event_embedding, event_logits = event_model(event_x)
            event_probability = torch.sigmoid(event_logits)

        logits = model(
            spatial_features=spatial_x,
            event_embedding=event_embedding,
            event_probability=event_probability,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite CUT loss at step {step}"
            )

        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(args.grad_clip),
        )
        optimizer.step()

        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(
                torch.as_tensor(grad).detach().cpu()
            ),
        }

        if (
            step == 1
            or step % int(args.print_every) == 0
            or step == int(args.cut_steps)
        ):
            probability = cut_probabilities(
                event_model,
                model,
                spatial_x=dataset.spatial_x,
                event_x=dataset.event_x,
                device=device,
            )
            metrics = binary_metrics(
                probability,
                dataset.y_cut.numpy().astype(np.int64),
                0.5,
            )
            row.update(
                {
                    "train_auc": metrics["auc"],
                    "train_cut_recall_at_0.5": metrics[
                        "recall"
                    ],
                    "train_keep_cut_rate_at_0.5": metrics[
                        "false_positive_rate"
                    ],
                }
            )
            print(
                f"[cut   {step:04d}/{int(args.cut_steps)}] "
                f"loss={row['loss']:.5f} "
                f"grad={row['grad_norm']:.3f} "
                f"auc={metrics['auc']:.4f} "
                f"CUT@.5={metrics['recall']:.4f} "
                f"KEEP-cut@.5={metrics['false_positive_rate']:.4f}",
                flush=True,
            )

        history.append(row)

    return model, history


# =============================================================================
# Final partition evaluation
# =============================================================================


def action_logits(
    case,
    *,
    event_use: Tensor,
    cut_use: Tensor,
) -> Tensor:
    """Apply only explicit CUT actions; preserve every other spatial logit."""

    if event_use.ndim != 1:
        raise ValueError("event_use must be [component]")
    if cut_use.shape != case.rag.spatial_edge_logits.shape:
        raise ValueError("cut_use must align with RAG edges")

    src, dst = case.rag.edge_index
    current = case.node_current_component

    if bool(
        (
            case.editable
            & (current[src] != current[dst])
        ).any()
    ):
        raise RuntimeError(
            "Editable edge crosses current spatial components"
        )

    edge_component = current[src]
    active_cut = (
        case.editable
        & event_use[edge_component]
        & cut_use
    )

    forced_cut = torch.full_like(
        case.rag.spatial_edge_logits,
        -20.0,
    )
    return torch.where(
        active_cut,
        forced_cut,
        case.rag.spatial_edge_logits,
    )


def rule_mask_tensor(
    metadata: list[dict[str, Any]],
    *,
    rule: dict[str, Any],
    device: torch.device,
) -> Tensor:
    frame = pd.DataFrame(metadata)
    values = rule_mask(
        frame,
        hard_cut_required=bool(rule["hard_cut_required"]),
        broken_min=int(rule["broken_min"]),
        local_ratio_min=float(rule["local_ratio_min"]),
        allow_boundary=bool(rule["allow_boundary"]),
    )
    return torch.as_tensor(
        values,
        device=device,
        dtype=torch.bool,
    )


def metric_accumulators(names: Sequence[str]):
    return OrderedDict(
        (str(name), INV42.metric_accumulator())
        for name in names
    )


@torch.inference_mode()
def evaluate_final(
    *,
    context,
    frames,
    args,
    event_model: EventDetector,
    cut_model: EventConditionedSpatialCutHead,
    event_threshold: float,
    cut_threshold: float,
    rule: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model = context["model"]
    loader = context["loader"]
    graph = context["graph_sanitized"]
    paths = context["paths"]
    spacing = context["spacing"]
    device = context["device"]
    status_index = graph_status_index(graph)

    edge_selector = context["edge_selector"]
    edge_selector_threshold = float(
        context["edge_selector_payload"]["write_threshold"]
    )

    names = [
        "spatial",
        "constrained_legal_inv46_candidate",
        "current_v3_constrained",
        "inv48_rule_event_spatial_cut",
        "inv48_learned_event_spatial_cut",
        "inv48_oracle_event_spatial_cut",
        "inv48_all_components_spatial_cut",
        "inv48_learned_event_target_edge_oracle",
        "inv48_oracle_event_target_edge_oracle",
    ]
    acc = metric_accumulators(names)

    component_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for ordinal, t in enumerate(frames, 1):
        started = time.perf_counter()
        runtime = loader.load(int(t))
        case = INV42.build_real_case(runtime)

        temporal_input = INV46.make_temporal_input_46(
            graph,
            runtime=runtime,
            paths=paths,
            temporal_radius=int(args.temporal_radius),
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
        base_reasoning = model.instance_temporal(
            encoded.instances,
            case.rag,
            encoded.temporal,
            encoded.dref_t,
        )

        raw_candidate_logits = (
            case.rag.spatial_edge_logits
            + base_reasoning.edge_temporal_delta
        )
        legal_candidate_logits = INV47.legal_candidate_logits(
            case,
            raw_candidate_logits,
        )
        v3_logits, _v3_prob, _v3_write = INV47.current_v3_final(
            model,
            edge_selector,
            runtime=runtime,
            case=case,
            base_reasoning=base_reasoning,
            candidate_logits=raw_candidate_logits,
            write_threshold=edge_selector_threshold,
        )

        event_x, event_meta = component_event_features(
            runtime,
            case,
            status_index=status_index,
        )
        spatial_x = rich_spatial_edge_features(
            encoded,
            case,
        )

        event_embedding, event_logits = event_model(
            event_x.to(device)
        )
        event_probability = torch.sigmoid(event_logits)
        learned_event_use = (
            event_probability >= float(event_threshold)
        )

        rule_event_use = rule_mask_tensor(
            event_meta,
            rule=rule,
            device=device,
        )
        oracle_event_use = (
            case.split_valid
            & case.metric_component_valid
            & (case.split_target > 0.5)
        )
        all_event_use = (
            case.split_valid
            & case.metric_component_valid
        )

        src, _dst = case.rag.edge_index
        edge_component = case.node_current_component[src]

        edge_event_embedding = event_embedding[
            edge_component
        ]
        edge_event_probability = event_probability[
            edge_component
        ]
        cut_logits = cut_model(
            spatial_features=spatial_x.to(device),
            event_embedding=edge_event_embedding,
            event_probability=edge_event_probability,
        )
        cut_probability = torch.sigmoid(cut_logits)
        learned_cut_use = (
            cut_probability >= float(cut_threshold)
        )
        target_cut_use = ~case.target_keep.bool()

        logits_by_name = {
            "spatial": case.rag.spatial_edge_logits,
            "constrained_legal_inv46_candidate": legal_candidate_logits,
            "current_v3_constrained": v3_logits,
            "inv48_rule_event_spatial_cut": action_logits(
                case,
                event_use=rule_event_use,
                cut_use=learned_cut_use,
            ),
            "inv48_learned_event_spatial_cut": action_logits(
                case,
                event_use=learned_event_use,
                cut_use=learned_cut_use,
            ),
            "inv48_oracle_event_spatial_cut": action_logits(
                case,
                event_use=oracle_event_use,
                cut_use=learned_cut_use,
            ),
            "inv48_all_components_spatial_cut": action_logits(
                case,
                event_use=all_event_use,
                cut_use=learned_cut_use,
            ),
            "inv48_learned_event_target_edge_oracle": action_logits(
                case,
                event_use=learned_event_use,
                cut_use=target_cut_use,
            ),
            "inv48_oracle_event_target_edge_oracle": action_logits(
                case,
                event_use=oracle_event_use,
                cut_use=target_cut_use,
            ),
        }

        partitions: dict[str, Any] = {}
        for name, logits in logits_by_name.items():
            partition = INV47.constrained_split_only_partition(
                model,
                case,
                logits,
            )
            partitions[name] = partition
            INV47.update_from_partition(
                acc[name],
                model=model,
                runtime=runtime,
                case=case,
                logits=logits,
                partition=partition,
            )

        for component in torch.nonzero(
            case.split_valid & case.metric_component_valid,
            as_tuple=False,
        ).flatten().tolist():
            component = int(component)
            row = dict(event_meta[component])
            row.update(
                {
                    "event_probability": float(
                        event_probability[component].item()
                    ),
                    "event_used": bool(
                        learned_event_use[component].item()
                    ),
                    "rule_event_used": bool(
                        rule_event_use[component].item()
                    ),
                    "oracle_event": bool(
                        oracle_event_use[component].item()
                    ),
                }
            )
            for name, partition in partitions.items():
                row[f"{name}_exact"] = bool(
                    INV47.component_exact(
                        runtime,
                        case,
                        partition,
                        component,
                    )
                )
            component_rows.append(row)

        editable_rows = torch.nonzero(
            case.editable,
            as_tuple=False,
        ).flatten()
        for edge_row in editable_rows.tolist():
            component = int(
                edge_component[edge_row].item()
            )
            edge_rows.append(
                {
                    "frame": int(t),
                    "edge_row": int(edge_row),
                    "component": component,
                    "target_cut": bool(
                        target_cut_use[edge_row].item()
                    ),
                    "cut_probability": float(
                        cut_probability[edge_row].item()
                    ),
                    "cut_used": bool(
                        learned_cut_use[edge_row].item()
                    ),
                    "event_probability": float(
                        event_probability[component].item()
                    ),
                    "event_used": bool(
                        learned_event_use[component].item()
                    ),
                    "rule_event_used": bool(
                        rule_event_use[component].item()
                    ),
                    "component_target_merge": bool(
                        case.split_target[component].item() > 0.5
                    ),
                    "spatial_keep_probability": float(
                        torch.sigmoid(
                            case.rag.spatial_edge_logits[edge_row]
                        ).item()
                    ),
                }
            )

        print(
            f"[Inv48 eval] t={int(t):03d} "
            f"{ordinal}/{len(frames)} "
            f"time={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    metrics = {
        name: INV47.metric_with_counts(value)
        for name, value in acc.items()
    }
    return (
        metrics,
        pd.DataFrame(component_rows),
        pd.DataFrame(edge_rows),
    )


# =============================================================================
# Reporting / checkpointing
# =============================================================================


def print_metric_table(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 126, flush=True)
    print(
        "INVESTIGATION 48 — FINAL DEVELOPMENT METRICS",
        flush=True,
    )
    print("=" * 126, flush=True)
    print(
        f"{'method':46s} "
        f"{'CUT':>11s} {'KEEP':>13s} {'exact':>12s} {'clean split':>14s}",
        flush=True,
    )
    print("-" * 126, flush=True)

    for name, m in metrics.items():
        cut = f"{m['cut_correct']}/{m['cut_edges']}"
        keep = f"{m['keep_correct']}/{m['keep_edges']}"
        exact = (
            f"{m['bad_components_exact']}/{m['bad_components']}"
        )
        clean = (
            f"{m['clean_components_split']}/{m['clean_components']}"
        )
        print(
            f"{name:46s} "
            f"{cut:>11s} {keep:>13s} {exact:>12s} {clean:>14s}",
            flush=True,
        )

    print("=" * 126, flush=True)


def save_checkpoint(
    path: Path,
    *,
    event_model: EventDetector,
    cut_model: EventConditionedSpatialCutHead,
    event_threshold: float,
    cut_threshold: float,
    rule: dict[str, Any],
    train_event_metrics: dict[str, Any],
    train_cut_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    args,
    spatial_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "inv48_event_conditioned_spatial_cut_v1",
            "investigation": SCRIPT_NAME,
            "objective_version": OBJECTIVE_VERSION,
            "event_feature_names": list(EVENT_FEATURE_NAMES),
            "event_dim": int(EVENT_DIM),
            "event_embed_dim": int(EVENT_EMBED_DIM),
            "spatial_dim": int(spatial_dim),
            "event_model_state_dict": event_model.state_dict(),
            "cut_model_state_dict": cut_model.state_dict(),
            "event_threshold": float(event_threshold),
            "cut_threshold": float(cut_threshold),
            "rule": dict(rule),
            "train_event_metrics": train_event_metrics,
            "train_cut_metrics": train_cut_metrics,
            "validation_metrics": val_metrics,
            "args": vars(args),
            "notes": {
                "base_stirnet_frozen": True,
                "hard_cutter_kept": True,
                "global_motion_stabilization_kept": True,
                "constrained_split_only_solver": True,
                "event_detector_component_level": True,
                "cut_head_uses_rich_frozen_spatial_features": True,
                "final_actions_cut_only": True,
                "thresholds_train_only": True,
                "validation_30_39_is_development_not_holdout": True,
            },
        },
        path,
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = INV46.build_parser()

    default_output = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / SCRIPT_NAME
        / str(INV42.DEFAULT_SAMPLE)
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
    default_v4 = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "46_biohub_hard_cutter_temporal_training_v4"
        / str(INV42.DEFAULT_SAMPLE)
        / "best_component_selector_v4.pt"
    ).resolve()

    parser.set_defaults(output=default_output)
    parser.description = (
        "Investigation 48: explicit temporal-event detector + "
        "rich spatial RAG CUT head."
    )

    parser.add_argument(
        "--v3-checkpoint",
        type=Path,
        default=default_v3,
    )
    parser.add_argument(
        "--v4-checkpoint",
        type=Path,
        default=default_v4,
    )

    parser.add_argument(
        "--event-steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--event-lr",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--event-weight-decay",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--event-batch-per-class",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--cut-steps",
        type=int,
        default=700,
    )
    parser.add_argument(
        "--cut-lr",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--cut-weight-decay",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--cut-batch-per-class",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-clean-event-rate",
        type=float,
        default=0.05,
        help=(
            "TRAIN-only maximum fraction of trusted clean components "
            "allowed to activate the learned/rule event detector."
        ),
    )
    parser.add_argument(
        "--max-keep-cut-rate",
        type=float,
        default=0.05,
        help=(
            "TRAIN-only maximum false CUT rate on trusted KEEP edges "
            "for spatial CUT threshold calibration."
        ),
    )
    # --grad-clip and --print-every come from the inherited Inv42/46 parser.
    parser.set_defaults(
        grad_clip=5.0,
        print_every=50,
    )
    return parser


def validate_args_48(args) -> None:
    for name in (
        "event_steps",
        "event_batch_per_class",
        "cut_steps",
        "cut_batch_per_class",
        "print_every",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_','-')} must be >=1")

    for name in (
        "event_lr",
        "cut_lr",
        "grad_clip",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be >0")

    for name in (
        "event_weight_decay",
        "cut_weight_decay",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be >=0")

    for name in (
        "max_clean_event_rate",
        "max_keep_cut_rate",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--{name.replace('_','-')} must be in [0,1]"
            )


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args_48(args)
    seed_everything(int(args.seed))

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 126, flush=True)
    print(
        "INVESTIGATION 48 — EVENT-CONDITIONED SPATIAL CUT",
        flush=True,
    )
    print("=" * 126, flush=True)
    print(f"repository           : {ROOT}", flush=True)
    print(f"output               : {output}", flush=True)
    print(
        "hypothesis           : temporal/statistical evidence localizes "
        "the suspicious instance; rich spatial features localize the CUT",
        flush=True,
    )
    print(
        "production weights   : FROZEN",
        flush=True,
    )
    print(
        "partition contract    : constrained split-only inside solver",
        flush=True,
    )
    print("=" * 126, flush=True)

    started = time.perf_counter()

    # Investigation 47 prepares the exact corrected state:
    # persisted RAG atoms, curated masks, canonical Trackastra identities,
    # global-motion stabilization, hard cutter, frozen initializer.
    context = INV47.prepare(args)
    device = context["device"]

    train_frames = tuple(map(int, context["train_frames"]))
    val_frames = tuple(map(int, context["val_frames"]))

    train = extract_frames(
        context=context,
        frames=train_frames,
        args=args,
        purpose="train",
    )
    val = extract_frames(
        context=context,
        frames=val_frames,
        args=args,
        purpose="validation",
    )

    if train.spatial_dim != val.spatial_dim:
        raise RuntimeError(
            "Train/validation spatial feature dimensions differ: "
            f"{train.spatial_dim} vs {val.spatial_dim}"
        )

    print("\n" + "=" * 126, flush=True)
    print("INV48 DATASET AUDIT", flush=True)
    print("=" * 126, flush=True)
    print(
        f"train components      : {len(train.components.y):,} "
        f"(merge={train.components.positive_count}, "
        f"clean={train.components.negative_count})",
        flush=True,
    )
    print(
        f"validation components : {len(val.components.y):,} "
        f"(merge={val.components.positive_count}, "
        f"clean={val.components.negative_count})",
        flush=True,
    )
    print(
        f"train editable edges  : {len(train.edges.y_cut):,} "
        f"(CUT={train.edges.cut_count}, KEEP={train.edges.keep_count})",
        flush=True,
    )
    print(
        f"validation edges      : {len(val.edges.y_cut):,} "
        f"(CUT={val.edges.cut_count}, KEEP={val.edges.keep_count})",
        flush=True,
    )
    print(
        f"rich spatial dim      : {train.spatial_dim:,}",
        flush=True,
    )
    print(
        f"explicit event dim    : {EVENT_DIM}",
        flush=True,
    )
    print("=" * 126, flush=True)

    # ------------------------------------------------------------------
    # Interpretable rule first: if this works, the event is intrinsically
    # simple and a learned model should not be allowed to obscure it.
    # ------------------------------------------------------------------
    best_rule, rule_search = search_rule(
        train.components,
        max_clean_event_rate=float(
            args.max_clean_event_rate
        ),
    )
    train_rule_metrics = rule_metrics(
        train.components,
        best_rule,
    )
    val_rule_metrics = rule_metrics(
        val.components,
        best_rule,
    )

    print("\n[rule baseline]", flush=True)
    print(
        "  "
        f"newborn=True "
        f"hard_cut_required={bool(best_rule['hard_cut_required'])} "
        f"broken>={int(best_rule['broken_min'])} "
        f"local_volume_ratio>={float(best_rule['local_ratio_min']):.2f} "
        f"allow_boundary={bool(best_rule['allow_boundary'])}",
        flush=True,
    )
    print(
        "  TRAIN "
        f"merge_recall={train_rule_metrics['merge_recall']:.4f} "
        f"clean_event={train_rule_metrics['clean_event_rate']:.4f} "
        f"precision={train_rule_metrics['precision']:.4f}",
        flush=True,
    )
    print(
        "  DEV   "
        f"merge_recall={val_rule_metrics['merge_recall']:.4f} "
        f"clean_event={val_rule_metrics['clean_event_rate']:.4f} "
        f"precision={val_rule_metrics['precision']:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Learned event detector.
    # ------------------------------------------------------------------
    event_model, event_history = train_event_detector(
        train.components,
        device=device,
        args=args,
    )

    train_event_prob, _ = event_probabilities(
        event_model,
        train.components.x,
        device,
    )
    event_threshold, train_event_metrics = choose_threshold(
        train_event_prob,
        train.components.y.numpy().astype(np.int64),
        max_false_positive_rate=float(
            args.max_clean_event_rate
        ),
    )

    val_event_prob, _ = event_probabilities(
        event_model,
        val.components.x,
        device,
    )
    val_event_metrics = binary_metrics(
        val_event_prob,
        val.components.y.numpy().astype(np.int64),
        event_threshold,
    )

    print("\n[learned event detector]", flush=True)
    print(
        f"  threshold (TRAIN only)={event_threshold:.6f}",
        flush=True,
    )
    print(
        "  TRAIN "
        f"recall={train_event_metrics['recall']:.4f} "
        f"clean_event={train_event_metrics['false_positive_rate']:.4f} "
        f"precision={train_event_metrics['precision']:.4f} "
        f"AUC={train_event_metrics['auc']:.4f}",
        flush=True,
    )
    print(
        "  DEV   "
        f"recall={val_event_metrics['recall']:.4f} "
        f"clean_event={val_event_metrics['false_positive_rate']:.4f} "
        f"precision={val_event_metrics['precision']:.4f} "
        f"AUC={val_event_metrics['auc']:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Rich spatial CUT head, with the event representation broadcast to
    # each edge in the same current component.
    # ------------------------------------------------------------------
    cut_model, cut_history = train_cut_head(
        train.edges,
        event_model=event_model,
        spatial_dim=train.spatial_dim,
        device=device,
        args=args,
    )

    train_cut_prob = cut_probabilities(
        event_model,
        cut_model,
        spatial_x=train.edges.spatial_x,
        event_x=train.edges.event_x,
        device=device,
    )
    cut_threshold, train_cut_metrics = choose_threshold(
        train_cut_prob,
        train.edges.y_cut.numpy().astype(np.int64),
        max_false_positive_rate=float(
            args.max_keep_cut_rate
        ),
    )

    val_cut_prob = cut_probabilities(
        event_model,
        cut_model,
        spatial_x=val.edges.spatial_x,
        event_x=val.edges.event_x,
        device=device,
    )
    val_cut_metrics = binary_metrics(
        val_cut_prob,
        val.edges.y_cut.numpy().astype(np.int64),
        cut_threshold,
    )

    print("\n[rich spatial CUT head]", flush=True)
    print(
        f"  threshold (TRAIN only)={cut_threshold:.6f}",
        flush=True,
    )
    print(
        "  TRAIN "
        f"CUT_recall={train_cut_metrics['recall']:.4f} "
        f"KEEP_cut={train_cut_metrics['false_positive_rate']:.4f} "
        f"precision={train_cut_metrics['precision']:.4f} "
        f"AUC={train_cut_metrics['auc']:.4f}",
        flush=True,
    )
    print(
        "  DEV   "
        f"CUT_recall={val_cut_metrics['recall']:.4f} "
        f"KEEP_cut={val_cut_metrics['false_positive_rate']:.4f} "
        f"precision={val_cut_metrics['precision']:.4f} "
        f"AUC={val_cut_metrics['auc']:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Whole partition evaluation: this is the scientifically important part.
    # ------------------------------------------------------------------
    final_metrics, component_predictions, edge_predictions = evaluate_final(
        context=context,
        frames=val_frames,
        args=args,
        event_model=event_model,
        cut_model=cut_model,
        event_threshold=event_threshold,
        cut_threshold=cut_threshold,
        rule=best_rule,
    )
    print_metric_table(final_metrics)

    # Attach offline classifier predictions to extracted tables.
    train_components_csv = train.components.metadata.copy()
    train_components_csv["learned_event_probability"] = train_event_prob
    train_components_csv["learned_event_used"] = (
        train_event_prob >= event_threshold
    )
    train_components_csv["rule_event_used"] = rule_mask(
        train_components_csv,
        hard_cut_required=bool(best_rule["hard_cut_required"]),
        broken_min=int(best_rule["broken_min"]),
        local_ratio_min=float(best_rule["local_ratio_min"]),
        allow_boundary=bool(best_rule["allow_boundary"]),
    )

    val_components_csv = val.components.metadata.copy()
    val_components_csv["learned_event_probability"] = val_event_prob
    val_components_csv["learned_event_used"] = (
        val_event_prob >= event_threshold
    )
    val_components_csv["rule_event_used"] = rule_mask(
        val_components_csv,
        hard_cut_required=bool(best_rule["hard_cut_required"]),
        broken_min=int(best_rule["broken_min"]),
        local_ratio_min=float(best_rule["local_ratio_min"]),
        allow_boundary=bool(best_rule["allow_boundary"]),
    )

    train_edges_csv = train.edges.metadata.copy()
    train_edges_csv["learned_cut_probability"] = train_cut_prob
    train_edges_csv["learned_cut_used"] = (
        train_cut_prob >= cut_threshold
    )

    val_edges_csv = val.edges.metadata.copy()
    val_edges_csv["learned_cut_probability"] = val_cut_prob
    val_edges_csv["learned_cut_used"] = (
        val_cut_prob >= cut_threshold
    )

    atomic_csv(
        output / "rule_search_train.csv",
        rule_search,
    )
    atomic_csv(
        output / "train_component_events.csv",
        train_components_csv,
    )
    atomic_csv(
        output / "validation_component_events.csv",
        val_components_csv,
    )
    atomic_csv(
        output / "train_edge_cut_predictions.csv",
        train_edges_csv,
    )
    atomic_csv(
        output / "validation_edge_cut_predictions.csv",
        val_edges_csv,
    )
    atomic_csv(
        output / "validation_component_partition_results.csv",
        component_predictions,
    )
    atomic_csv(
        output / "validation_edge_partition_results.csv",
        edge_predictions,
    )
    atomic_json(
        output / "event_training_history.json",
        event_history,
    )
    atomic_json(
        output / "cut_training_history.json",
        cut_history,
    )

    summary = {
        "investigation": SCRIPT_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "train_frames": list(map(int, train_frames)),
        "development_validation_frames": list(
            map(int, val_frames)
        ),
        "validation_is_untouched_holdout": False,
        "event_feature_names": list(EVENT_FEATURE_NAMES),
        "event_feature_dim": int(EVENT_DIM),
        "rich_spatial_feature_dim": int(train.spatial_dim),
        "rule": dict(best_rule),
        "rule_train": train_rule_metrics,
        "rule_validation": val_rule_metrics,
        "event_threshold_train_only": float(event_threshold),
        "event_train_metrics": train_event_metrics,
        "event_validation_metrics": val_event_metrics,
        "cut_threshold_train_only": float(cut_threshold),
        "cut_train_metrics": train_cut_metrics,
        "cut_validation_metrics": val_cut_metrics,
        "final_partition_metrics": final_metrics,
        "runtime_seconds": float(
            time.perf_counter() - started
        ),
        "architecture": {
            "base_stirnet_frozen": True,
            "hard_cutter": True,
            "global_motion_compensated": True,
            "event_detector": (
                "explicit temporal/statistical current-component evidence"
            ),
            "cut_head": (
                "rich frozen RAG/node/raw/morphology/instance spatial "
                "representation + event embedding"
            ),
            "final_action": (
                "CUT-only on editable edges inside event-positive component"
            ),
            "partition": (
                "Investigation-47 constrained split-only solver"
            ),
        },
    }
    atomic_json(
        output / "summary.json",
        summary,
    )

    save_checkpoint(
        output / "best_inv48_event_conditioned_spatial_cut.pt",
        event_model=event_model,
        cut_model=cut_model,
        event_threshold=event_threshold,
        cut_threshold=cut_threshold,
        rule=best_rule,
        train_event_metrics=train_event_metrics,
        train_cut_metrics=train_cut_metrics,
        val_metrics=final_metrics,
        args=args,
        spatial_dim=train.spatial_dim,
    )

    print("\n" + "=" * 126, flush=True)
    print("INVESTIGATION 48 COMPLETE", flush=True)
    print("=" * 126, flush=True)
    print(f"output                 : {output}", flush=True)
    print(
        f"checkpoint             : "
        f"{output / 'best_inv48_event_conditioned_spatial_cut.pt'}",
        flush=True,
    )
    print(
        f"component diagnostics  : "
        f"{output / 'validation_component_partition_results.csv'}",
        flush=True,
    )
    print(
        f"edge diagnostics       : "
        f"{output / 'validation_edge_partition_results.csv'}",
        flush=True,
    )
    print(
        f"summary                : {output / 'summary.json'}",
        flush=True,
    )
    print(
        f"runtime                : "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print("=" * 126, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
