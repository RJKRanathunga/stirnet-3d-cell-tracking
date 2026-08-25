from __future__ import annotations

r"""
STIR-Net Investigation 27 — edge-relational group reasoning training.

Purpose
-------
Test the new hypothesis WITHOUT modifying production STIR-Net:

    local node/contact evidence
            -> frozen h100 spatial RAG/GNN
            -> initial merge logits
            -> DIRECT edge-to-edge relational GNN
            -> refined signed merge logits
            -> signed multicut group consistency (validation only)

The current production RAG already performs node message passing, so neighboring
edges can influence one another only indirectly (AB -> B -> BC).  Investigation
27 adds an experiment-local line-graph reasoner where original RAG edges become
relation nodes and edges that share a supervoxel communicate directly:

    original RAG:       A --eAB-- B --eBC-- C
    relation graph:          eAB <----> eBC

This directly tests the "people forming a group" idea: the decision for one
relationship should depend on the other relationships incident on the same
objects before a group is committed.

Isolation policy
----------------
The h100 morphology-v2 model is completely FROZEN:
    dense geometry
    watershed
    morphology-v2 encoders
    existing node-GNN / RAG message passing
    existing edge classifier

TRAINABLE:
    only EdgeRelationalReasoner defined in THIS investigation file.

The production separator barrier is intentionally NOT enabled here.  This is an
alternative experiment so its gain can be measured independently from the
separator-veto branch in Investigation 26.

Training losses
---------------
1. Balanced GT edge BCE on refined logits.
2. Mixed-wedge ranking loss.  If two edges share node B and GT says one is
   SAME-cell while the other is DIFFERENT-cell, the positive relationship must
   outrank the negative relationship by a configurable logit margin.
3. Triangle consistency loss.  For a triangle AB/BC/AC, two very high merge
   probabilities should not coexist with a low third probability.
4. Conservative distillation on h100 edges that were already confidently
   correct, preventing a broad global merge/separate bias.
5. Small residual-magnitude penalty.

Group consistency
-----------------
Multicut is discrete and is therefore NOT differentiated through.  During
held-out validation, both the frozen h100 logits and the refined relational
logits are passed through the CURRENT production GraphPartitioner configured as
signed multicut at q=0.845.  We report:
    * GT-negative adjacency edges trapped inside a component
    * GT-positive adjacency edges cut between components
    * total partition disagreement rate

Checkpoint policy
-----------------
The frozen h100 baseline is explicitly treated as candidate step 0.  A trained
reasoner is saved as best_reasoner.pt ONLY if it beats the baseline ranking while
respecting the positive-merge safety guard.  This avoids the Investigation-23
failure mode where "best trained" could still be worse than the starting model.

Default real run
----------------
    python .\investigations\stirnet\27_edge_relational_group_reasoning_training.py

Quick smoke
-----------
    python .\investigations\stirnet\27_edge_relational_group_reasoning_training.py `
        --max-steps 4 `
        --validation-every 2 `
        --checkpoint-every 2 `
        --validation-crops-per-sample 2 `
        --mining-max-crops-per-sample 24 `
        --run-name edge_relational_27_smoke

The default real run uses 600 optimizer steps, validation every 50 steps, full
training-manifest relational crop mining, and bounded best/latest checkpoints.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm


EXPERIMENT_NAME = "27_edge_relational_group_reasoning_training"
DEFAULT_SAMPLES = "Drosophila_1,Drosophila_2"
DEFAULT_SPACING_XYZ = "0.20312639,0.20312639,0.79099447"
DEFAULT_CHECKPOINT = (
    "runs/stirnet/investigations/"
    "19_morphology_rag_v2_headroom_training/"
    "recovery/"
    "drosophila_12_morphology_rag_v2_headroom_h100/"
    "checkpoint_step_000600.pt"
)


# ======================================================================================
# Repository / support
# ======================================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
        ):
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "learned" / "stirnet").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not resolve the cell-tracking repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_inv17_support():
    path = ROOT / "investigations" / "stirnet" / "17_morphology_rag_multicrop_training.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_stirnet_inv17_support_for_inv27", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import support module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def resolve_checkpoint(value: str | Path) -> Path:
    path = resolve(value)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    for candidate in (
        path / "best_checkpoint.pt",
        path / "best_reasoner.pt",
        path / "latest_reasoner.pt",
    ):
        if candidate.is_file():
            return candidate
    rows = sorted(path.glob("checkpoint_step_*.pt"))
    if rows:
        return rows[-1]
    rows = sorted(path.glob("**/best_checkpoint.pt"))
    if rows:
        return rows[-1]
    raise FileNotFoundError(f"No checkpoint found below {path}")


def torch_load(path: Path, *, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_samples(text: str) -> tuple[str, ...]:
    result = tuple(v.strip() for v in text.split(",") if v.strip())
    if not result:
        raise ValueError("--samples cannot be empty")
    return result


def parse_triplet(text: str, *, cast=float) -> tuple:
    values = tuple(cast(v.strip()) for v in text.split(","))
    if len(values) != 3:
        raise ValueError(f"Expected exactly 3 comma-separated values, got {text!r}")
    return values


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


# ======================================================================================
# Direct edge-to-edge relational GNN
# ======================================================================================


@dataclass
class RelationGraph:
    # Directed line-graph messages: source_edge -> target_edge through shared_node.
    target_edge: Tensor
    source_edge: Tensor
    shared_node: Tensor
    # Unordered pairs are useful for ranking supervision / diagnostics.
    pair_edge_a: Tensor
    pair_edge_b: Tensor
    pair_shared_node: Tensor
    # Valid RAG triangles, represented by three original edge rows.
    triangles: Tensor

    @property
    def directed_count(self) -> int:
        return int(self.target_edge.numel())

    @property
    def pair_count(self) -> int:
        return int(self.pair_edge_a.numel())

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


def _spread_python_rows(rows: list[tuple[int, int]], maximum: int) -> list[tuple[int, int]]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    if maximum == 1:
        return [rows[len(rows) // 2]]
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(maximum):
        position = round(i * (len(rows) - 1) / (maximum - 1))
        if position in used:
            continue
        used.add(position)
        selected.append(rows[position])
    return selected


def build_relation_graph(
    edge_index: Tensor,
    *,
    node_count: int,
    max_pairs_per_node: int,
) -> RelationGraph:
    """Build the line graph and local triangles from the atomic RAG topology."""
    device = edge_index.device
    edge_cpu = edge_index.detach().long().cpu()
    edge_count = int(edge_cpu.shape[1])

    incident: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    pair_to_edge: dict[tuple[int, int], int] = {}
    for edge_row in range(edge_count):
        u = int(edge_cpu[0, edge_row].item())
        v = int(edge_cpu[1, edge_row].item())
        if u == v:
            continue
        incident[u].append((edge_row, v))
        incident[v].append((edge_row, u))
        key = (u, v) if u < v else (v, u)
        pair_to_edge[key] = edge_row

    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_shared: list[int] = []
    directed_target: list[int] = []
    directed_source: list[int] = []
    directed_shared: list[int] = []

    triangle_set: set[tuple[int, int, int]] = set()

    for shared_node, rows in enumerate(incident):
        rows = sorted(rows, key=lambda item: item[0])
        combinations: list[tuple[int, int]] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                combinations.append((i, j))
        combinations = _spread_python_rows(combinations, max_pairs_per_node)

        for i, j in combinations:
            edge_a, outer_a = rows[i]
            edge_b, outer_b = rows[j]
            if edge_a == edge_b:
                continue

            pair_a.append(edge_a)
            pair_b.append(edge_b)
            pair_shared.append(shared_node)

            # Direct information flow in BOTH directions.
            directed_target.extend((edge_a, edge_b))
            directed_source.extend((edge_b, edge_a))
            directed_shared.extend((shared_node, shared_node))

            outer_key = (
                (outer_a, outer_b)
                if outer_a < outer_b
                else (outer_b, outer_a)
            )
            closing_edge = pair_to_edge.get(outer_key)
            if closing_edge is not None:
                triangle_set.add(tuple(sorted((edge_a, edge_b, closing_edge))))

    def tensor1(rows: list[int]) -> Tensor:
        return torch.as_tensor(rows, device=device, dtype=torch.long)

    if triangle_set:
        triangles = torch.as_tensor(
            sorted(triangle_set), device=device, dtype=torch.long
        ).reshape(-1, 3)
    else:
        triangles = torch.empty((0, 3), device=device, dtype=torch.long)

    return RelationGraph(
        target_edge=tensor1(directed_target),
        source_edge=tensor1(directed_source),
        shared_node=tensor1(directed_shared),
        pair_edge_a=tensor1(pair_a),
        pair_edge_b=tensor1(pair_b),
        pair_shared_node=tensor1(pair_shared),
        triangles=triangles,
    )


class EdgeRelationBlock(nn.Module):
    """One direct edge-to-edge message-passing block on the RAG line graph."""

    def __init__(self, hidden: int, node_dim: int, dropout: float):
        super().__init__()
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(node_dim)
        pair_dim = 2 * hidden + node_dim + 2
        self.message_mlp = nn.Sequential(
            nn.Linear(pair_dim, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(
        self,
        edge_state: Tensor,
        node_state: Tensor,
        base_signed: Tensor,
        relation: RelationGraph,
    ) -> Tensor:
        if relation.directed_count == 0:
            return edge_state

        edge_norm = self.edge_norm(edge_state)
        node_norm = self.node_norm(node_state)
        target = relation.target_edge
        source = relation.source_edge
        shared = relation.shared_node

        pair = torch.cat(
            [
                edge_norm[target],
                edge_norm[source],
                node_norm[shared],
                base_signed[target, None],
                base_signed[source, None],
            ],
            dim=-1,
        )
        message = self.message_mlp(pair)
        gate = torch.sigmoid(self.gate_mlp(pair))
        weighted = message * gate

        aggregate = torch.zeros_like(edge_state)
        denominator = edge_state.new_zeros((edge_state.shape[0], 1))
        aggregate.index_add_(0, target, weighted)
        denominator.index_add_(0, target, gate)
        aggregate = aggregate / denominator.clamp_min(1e-4)

        update = self.update_mlp(torch.cat([edge_norm, aggregate], dim=-1))
        return edge_state + update


class EdgeRelationalReasoner(nn.Module):
    """Experiment-local signed residual reasoner over relationships themselves."""

    def __init__(
        self,
        *,
        edge_dim: int,
        node_dim: int,
        hidden: int = 96,
        layers: int = 2,
        dropout: float = 0.05,
        max_logit_delta: float = 6.0,
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.node_dim = int(node_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.max_logit_delta = float(max_logit_delta)

        self.edge_seed = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [EdgeRelationBlock(hidden, hidden, dropout) for _ in range(layers)]
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # Near-exact h100 behavior at initialization while still allowing
        # gradients to reach the relational blocks on the first step.
        final = self.delta_head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(final.bias)

    def config_dict(self) -> dict[str, Any]:
        return {
            "edge_dim": self.edge_dim,
            "node_dim": self.node_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "max_logit_delta": self.max_logit_delta,
        }

    def forward(
        self,
        *,
        edge_embeddings: Tensor,
        node_embeddings: Tensor,
        base_logits: Tensor,
        relation: RelationGraph,
    ) -> tuple[Tensor, Tensor]:
        base_logits = base_logits.float()
        # bounded signed confidence in [-1,1], stable even for very certain base logits
        base_signed = torch.tanh(base_logits / 4.0)
        edge_state = self.edge_seed(
            torch.cat([edge_embeddings.float(), base_signed[:, None]], dim=-1)
        )
        node_state = self.node_projection(node_embeddings.float())
        for block in self.blocks:
            edge_state = block(edge_state, node_state, base_signed, relation)

        raw_delta = self.delta_head(edge_state).squeeze(-1).float()
        delta = self.max_logit_delta * torch.tanh(
            raw_delta / self.max_logit_delta
        )
        return base_logits + delta, delta


# ======================================================================================
# Relational supervision helpers
# ======================================================================================


def mixed_wedge_rows(
    relation: RelationGraph,
    targets,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (positive_edge, negative_edge, shared_node) for mixed wedges."""
    if relation.pair_count == 0:
        empty = relation.pair_edge_a[:0]
        return empty, empty, relation.pair_shared_node[:0]

    a = relation.pair_edge_a
    b = relation.pair_edge_b
    valid = targets.valid.bool()
    pair_valid = valid[a] & valid[b]
    label_a = targets.target[a] > 0.5
    label_b = targets.target[b] > 0.5
    mixed = pair_valid & (label_a != label_b)
    if not bool(mixed.any()):
        empty = a[:0]
        return empty, empty, relation.pair_shared_node[:0]

    a = a[mixed]
    b = b[mixed]
    shared = relation.pair_shared_node[mixed]
    a_positive = targets.target[a] > 0.5
    positive = torch.where(a_positive, a, b)
    negative = torch.where(a_positive, b, a)
    return positive, negative, shared


def select_balanced_edges(
    targets,
    base_logits: Tensor,
    *,
    max_per_class: int,
) -> Tensor:
    valid = targets.valid.bool()
    positive = torch.nonzero(valid & (targets.target > 0.5), as_tuple=False).flatten()
    negative = torch.nonzero(valid & (targets.target <= 0.5), as_tuple=False).flatten()

    if positive.numel() > max_per_class:
        positions = torch.linspace(
            0, positive.numel() - 1, max_per_class, device=positive.device
        ).round().long()
        positive = positive[positions]

    if negative.numel() > max_per_class:
        # Prefer the negatives the frozen h100 considers most merge-like.
        order = torch.argsort(base_logits.detach()[negative], descending=True)
        negative = negative[order[:max_per_class]]

    if positive.numel() and negative.numel():
        quota = min(int(positive.numel()), int(negative.numel()), max_per_class)
        positive = positive[:quota]
        negative = negative[:quota]

    if not positive.numel():
        return negative
    if not negative.numel():
        return positive
    return torch.cat([positive, negative])


def select_ranking_pairs(
    positive: Tensor,
    negative: Tensor,
    base_logits: Tensor,
    *,
    max_pairs: int,
) -> tuple[Tensor, Tensor]:
    if positive.numel() == 0:
        return positive, negative
    # Hardest first: smallest base positive-minus-negative gap.
    gap = base_logits.detach()[positive] - base_logits.detach()[negative]
    order = torch.argsort(gap, descending=False)
    if max_pairs > 0:
        order = order[:max_pairs]
    return positive[order], negative[order]


def triangle_consistency_loss(
    refined_logits: Tensor,
    relation: RelationGraph,
    targets,
    *,
    max_triangles: int,
) -> tuple[Tensor, int]:
    triangles = relation.triangles
    if triangles.numel() == 0:
        return refined_logits.sum() * 0.0, 0
    valid = targets.valid.bool()
    keep = valid[triangles].all(dim=1)
    triangles = triangles[keep]
    if triangles.shape[0] == 0:
        return refined_logits.sum() * 0.0, 0
    if max_triangles > 0 and triangles.shape[0] > max_triangles:
        triangles = triangles[:max_triangles]

    p = refined_logits.float().sigmoid()
    p1 = p[triangles[:, 0]]
    p2 = p[triangles[:, 1]]
    p3 = p[triangles[:, 2]]
    # Merge transitivity: if two sides are simultaneously high, the third side
    # cannot be arbitrarily low. BCE/ranking decide WHICH edge should move.
    penalty = (
        F.relu(p1 + p2 - p3 - 1.0).square()
        + F.relu(p1 + p3 - p2 - 1.0).square()
        + F.relu(p2 + p3 - p1 - 1.0).square()
    ) / 3.0
    return penalty.mean(), int(triangles.shape[0])


def training_losses(
    *,
    refined_logits: Tensor,
    delta: Tensor,
    base_logits: Tensor,
    targets,
    relation: RelationGraph,
    merge_threshold: float,
    max_edges_per_class: int,
    max_ranking_pairs: int,
    ranking_margin: float,
    max_triangles: int,
    ranking_weight: float,
    triangle_weight: float,
    preservation_weight: float,
    residual_weight: float,
    preserve_negative_probability: float,
) -> tuple[Tensor, dict[str, Any]]:
    selected = select_balanced_edges(
        targets, base_logits, max_per_class=max_edges_per_class
    )
    if selected.numel():
        bce = F.binary_cross_entropy_with_logits(
            refined_logits[selected].float(),
            targets.target[selected].float(),
        )
    else:
        bce = refined_logits.sum() * 0.0

    positive, negative, _ = mixed_wedge_rows(relation, targets)
    positive, negative = select_ranking_pairs(
        positive,
        negative,
        base_logits,
        max_pairs=max_ranking_pairs,
    )
    if positive.numel():
        gap = refined_logits[positive].float() - refined_logits[negative].float()
        ranking = F.softplus(float(ranking_margin) - gap).mean()
    else:
        ranking = refined_logits.sum() * 0.0

    triangle, triangle_count = triangle_consistency_loss(
        refined_logits,
        relation,
        targets,
        max_triangles=max_triangles,
    )

    valid = targets.valid.bool()
    target_positive = targets.target > 0.5
    base_p = base_logits.detach().float().sigmoid()
    preserve = valid & (
        (target_positive & (base_p >= float(merge_threshold)))
        | ((~target_positive) & (base_p <= float(preserve_negative_probability)))
    )
    if bool(preserve.any()):
        preservation = F.smooth_l1_loss(
            refined_logits[preserve].float(),
            base_logits.detach()[preserve].float(),
            beta=1.0,
        )
    else:
        preservation = refined_logits.sum() * 0.0

    residual = delta.float().square().mean() if delta.numel() else delta.sum() * 0.0

    total = (
        bce
        + float(ranking_weight) * ranking
        + float(triangle_weight) * triangle
        + float(preservation_weight) * preservation
        + float(residual_weight) * residual
    )
    return total, {
        "selected_edge_count": int(selected.numel()),
        "ranking_pair_count": int(positive.numel()),
        "triangle_count": int(triangle_count),
        "preserve_edge_count": int(preserve.sum().item()),
        "bce": float(bce.detach().cpu()),
        "ranking": float(ranking.detach().cpu()),
        "triangle": float(triangle.detach().cpu()),
        "preservation": float(preservation.detach().cpu()),
        "residual": float(residual.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


# ======================================================================================
# Frozen base forward
# ======================================================================================


def forward_base_crop(
    *,
    model,
    crop,
    criterion,
    support,
    amp_dtype: str,
):
    with torch.no_grad(), support._autocast_context(amp_dtype):
        geometry = model(
            crop["spatial_inputs"],
            crop["spacing_um"],
            crop["dref_um"],
            spatial_padding_mask=crop.get("spatial_padding_mask"),
            execution_stage="geometry",
        )
        output = model(
            crop["spatial_inputs"],
            crop["spacing_um"],
            crop["dref_um"],
            spatial_padding_mask=crop.get("spatial_padding_mask"),
            execution_stage="spatial",
            precomputed_geometry=geometry,
        )
    rag = output.rag
    if rag.node_embeddings is None or rag.edge_embeddings is None:
        raise RuntimeError("Frozen h100 RAG did not expose node/edge embeddings")
    targets = criterion.build_targets(
        rag,
        crop["gt_labels"],
        valid_mask=crop.get("supervision_valid_mask"),
    )
    return rag, targets


# ======================================================================================
# Relational crop mining
# ======================================================================================


def _spread_indices(rows: list[int], maximum: int) -> list[int]:
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    if maximum == 1:
        return [rows[len(rows) // 2]]
    result: list[int] = []
    used: set[int] = set()
    for i in range(maximum):
        position = round(i * (len(rows) - 1) / (maximum - 1))
        value = int(rows[position])
        if value not in used:
            used.add(value)
            result.append(value)
    return result


def mining_fingerprint(
    *,
    checkpoint: Path,
    splits,
    samples,
    max_pairs_per_node: int,
    ranking_margin: float,
    mining_max_crops_per_sample: int,
) -> str:
    payload = {
        "version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "samples": list(samples),
        "max_pairs_per_node": int(max_pairs_per_node),
        "ranking_margin": float(ranking_margin),
        "mining_max_crops_per_sample": int(mining_max_crops_per_sample),
        "train_indices": {
            sample: [int(v) for v in splits[sample]["train_indices"]]
            for sample in samples
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def mine_relational_crops(
    *,
    checkpoint: Path,
    model,
    criterion,
    support,
    source_batches,
    splits,
    samples,
    amp_dtype: str,
    partial_ignore_margin_um: float,
    max_pairs_per_node: int,
    ranking_margin: float,
    mining_max_crops_per_sample: int,
    cache_root: Path,
):
    fingerprint = mining_fingerprint(
        checkpoint=checkpoint,
        splits=splits,
        samples=samples,
        max_pairs_per_node=max_pairs_per_node,
        ranking_margin=ranking_margin,
        mining_max_crops_per_sample=mining_max_crops_per_sample,
    )
    cache_path = cache_root / f"{fingerprint}.json"
    if cache_path.is_file():
        print(f"[mining] cache HIT: {cache_path}", flush=True)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"[mining] cache MISS: {cache_path}", flush=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    profile = {"version": 1, "fingerprint": fingerprint, "samples": {}}
    total_candidate = total_productive = total_hard = 0
    total_mixed = total_hard_mixed = total_pairs = 0
    started = time.perf_counter()

    for sample in samples:
        candidates = _spread_indices(
            [int(v) for v in splits[sample]["train_indices"]],
            int(mining_max_crops_per_sample),
        )
        rows = []
        productive = []
        hard = []
        progress = tqdm(
            total=len(candidates),
            desc=f"Mine relational {sample}",
            unit="crop",
            dynamic_ncols=True,
            leave=False,
            colour="green",
            file=sys.stdout,
        )
        for manifest_index in candidates:
            record = splits[sample]["records"][manifest_index]
            crop_cpu, _ = support._materialize_crop(
                source_batches[sample],
                record,
                partial_ignore_margin_um=partial_ignore_margin_um,
            )
            crop = support._move_crop_to_cuda(crop_cpu)
            rag, targets = forward_base_crop(
                model=model,
                crop=crop,
                criterion=criterion,
                support=support,
                amp_dtype=amp_dtype,
            )
            relation = build_relation_graph(
                rag.edge_index,
                node_count=int(rag.node_features.shape[0]),
                max_pairs_per_node=max_pairs_per_node,
            )
            pos, neg, _ = mixed_wedge_rows(relation, targets)
            base_logits = rag.spatial_edge_logits.detach().float()
            base_p = base_logits.sigmoid()
            if pos.numel():
                gap = base_logits[pos] - base_logits[neg]
                hard_mask = (
                    (base_p[neg] >= 0.5)
                    | (gap < float(ranking_margin))
                )
                hard_count = int(hard_mask.sum().item())
                mean_gap = float(gap.mean().cpu())
                ranking_violation = int((gap <= 0.0).sum().item())
            else:
                hard_count = 0
                mean_gap = 0.0
                ranking_violation = 0

            row = {
                "manifest_index": int(manifest_index),
                "valid_edge_count": int(targets.valid.sum().item()),
                "line_pair_count": int(relation.pair_count),
                "mixed_wedge_count": int(pos.numel()),
                "hard_mixed_wedge_count": int(hard_count),
                "base_ranking_violation_count": int(ranking_violation),
                "base_mean_mixed_logit_gap": float(mean_gap),
                "triangle_count": int(relation.triangle_count),
            }
            rows.append(row)
            if pos.numel():
                productive.append(int(manifest_index))
            if hard_count:
                hard.append(int(manifest_index))

            total_pairs += relation.pair_count
            total_mixed += int(pos.numel())
            total_hard_mixed += int(hard_count)
            progress.update(1)
            progress.set_postfix(
                {
                    "prod": len(productive),
                    "hard": len(hard),
                    "mixed": total_mixed,
                },
                refresh=False,
            )
            del rag, targets, relation, crop, crop_cpu
            torch.cuda.empty_cache()
            gc.collect()
        progress.close()

        sample_summary = {
            "candidate_indices": candidates,
            "candidate_count": len(candidates),
            "productive_indices": sorted(set(productive)),
            "productive_count": len(set(productive)),
            "hard_indices": sorted(set(hard)),
            "hard_count": len(set(hard)),
            "rows": rows,
        }
        profile["samples"][sample] = sample_summary
        total_candidate += len(candidates)
        total_productive += sample_summary["productive_count"]
        total_hard += sample_summary["hard_count"]
        print(
            f"[mining] {sample}: candidate={len(candidates)} "
            f"productive={sample_summary['productive_count']} "
            f"hard={sample_summary['hard_count']} "
            f"mixed={sum(r['mixed_wedge_count'] for r in rows)}",
            flush=True,
        )

    profile["summary"] = {
        "candidate_crop_count": int(total_candidate),
        "productive_crop_count": int(total_productive),
        "hard_crop_count": int(total_hard),
        "line_pair_count": int(total_pairs),
        "mixed_wedge_count": int(total_mixed),
        "hard_mixed_wedge_count": int(total_hard_mixed),
        "productive_fraction": safe_ratio(total_productive, total_candidate),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    if total_productive == 0:
        raise RuntimeError(
            "Relational mining found zero crops with mixed same/different incident edges"
        )
    atomic_json(cache_path, profile)
    print(
        "[mining] COMPLETE: "
        f"productive={total_productive}/{total_candidate} "
        f"hard={total_hard}/{total_candidate} "
        f"mixed={total_mixed} hard-mixed={total_hard_mixed} "
        f"time={duration(profile['summary']['elapsed_seconds'])}",
        flush=True,
    )
    return profile


def choose_training_job(
    *,
    support,
    splits,
    mining,
    samples,
    attempt: int,
    sample_steps: dict,
    productive_fraction: float,
    hard_fraction: float,
    seed: int,
):
    rng = random.Random(seed * 1000003 + attempt * 9176)
    base_sample = samples[attempt % len(samples)]
    want_productive = rng.random() < float(productive_fraction)

    if want_productive:
        sample_order = [base_sample] + [s for s in samples if s != base_sample]
        sample = next(
            (
                s for s in sample_order
                if mining["samples"][s]["productive_indices"]
            ),
            None,
        )
        if sample is not None:
            row = mining["samples"][sample]
            use_hard = bool(row["hard_indices"]) and rng.random() < float(hard_fraction)
            pool_name = "hard" if use_hard else "productive"
            pool = row["hard_indices"] if use_hard else row["productive_indices"]
            ordinal = sample_steps[sample][pool_name]
            sample_steps[sample][pool_name] += 1
            return {
                "sample": sample,
                "manifest_index": int(pool[ordinal % len(pool)]),
                "pool": pool_name,
                "provenance": f"mined_{pool_name}",
            }

    sample = base_sample
    ordinal = sample_steps[sample]["ordinary"]
    sample_steps[sample]["ordinary"] += 1
    manifest_index, provenance = support._training_manifest_index(
        splits[sample], sample_local_step=ordinal
    )
    return {
        "sample": sample,
        "manifest_index": int(manifest_index),
        "pool": "ordinary",
        "provenance": provenance,
    }


# ======================================================================================
# Evaluation
# ======================================================================================


def new_edge_accumulator() -> dict[str, float]:
    return {
        "edges": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "bce_sum": 0.0,
        "positive_probability_sum": 0.0,
        "negative_probability_sum": 0.0,
        "false_merge": 0.0,
        "positive_accept": 0.0,
    }


def add_edge_metrics(acc, logits: Tensor, targets, *, merge_threshold: float) -> None:
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    negative = valid & ~positive
    p = logits.detach().float().sigmoid()
    if bool(valid.any()):
        acc["bce_sum"] += float(
            F.binary_cross_entropy_with_logits(
                logits[valid].float(), targets.target[valid].float(), reduction="sum"
            ).cpu()
        )
    acc["edges"] += float(valid.sum().item())
    acc["positive"] += float(positive.sum().item())
    acc["negative"] += float(negative.sum().item())
    acc["positive_probability_sum"] += float(p[positive].sum().cpu()) if bool(positive.any()) else 0.0
    acc["negative_probability_sum"] += float(p[negative].sum().cpu()) if bool(negative.any()) else 0.0
    acc["false_merge"] += float((negative & (p >= merge_threshold)).sum().item())
    acc["positive_accept"] += float((positive & (p >= merge_threshold)).sum().item())


def finalize_edge_metrics(acc) -> dict[str, Any]:
    return {
        "edge_count": int(acc["edges"]),
        "positive_edge_count": int(acc["positive"]),
        "negative_edge_count": int(acc["negative"]),
        "bce": safe_ratio(acc["bce_sum"], acc["edges"]),
        "mean_positive_probability": safe_ratio(
            acc["positive_probability_sum"], acc["positive"]
        ),
        "mean_negative_probability": safe_ratio(
            acc["negative_probability_sum"], acc["negative"]
        ),
        "false_merge_count": int(acc["false_merge"]),
        "false_merge_rate": safe_ratio(acc["false_merge"], acc["negative"]),
        "positive_accept_count": int(acc["positive_accept"]),
        "positive_accept_rate": safe_ratio(acc["positive_accept"], acc["positive"]),
    }


def new_relational_accumulator() -> dict[str, float]:
    return {
        "pair_count": 0.0,
        "mixed_count": 0.0,
        "ranking_violation": 0.0,
        "ranking_margin_violation": 0.0,
        "gap_sum": 0.0,
        "triangle_count": 0.0,
        "triangle_hard_inconsistent": 0.0,
        "triangle_soft_penalty_sum": 0.0,
    }


def add_relational_metrics(
    acc,
    logits: Tensor,
    relation: RelationGraph,
    targets,
    *,
    merge_threshold: float,
    ranking_margin: float,
) -> None:
    acc["pair_count"] += relation.pair_count
    positive, negative, _ = mixed_wedge_rows(relation, targets)
    acc["mixed_count"] += int(positive.numel())
    if positive.numel():
        gap = logits.detach().float()[positive] - logits.detach().float()[negative]
        acc["ranking_violation"] += int((gap <= 0.0).sum().item())
        acc["ranking_margin_violation"] += int((gap < float(ranking_margin)).sum().item())
        acc["gap_sum"] += float(gap.sum().cpu())

    triangles = relation.triangles
    if triangles.numel():
        valid = targets.valid.bool()[triangles].all(dim=1)
        triangles = triangles[valid]
    if triangles.numel():
        p = logits.detach().float().sigmoid()
        tp = p[triangles]
        accepted = tp >= float(merge_threshold)
        accepted_count = accepted.sum(dim=1)
        hard_inconsistent = accepted_count == 2
        p1, p2, p3 = tp[:, 0], tp[:, 1], tp[:, 2]
        soft = (
            F.relu(p1 + p2 - p3 - 1.0).square()
            + F.relu(p1 + p3 - p2 - 1.0).square()
            + F.relu(p2 + p3 - p1 - 1.0).square()
        ) / 3.0
        acc["triangle_count"] += int(triangles.shape[0])
        acc["triangle_hard_inconsistent"] += int(hard_inconsistent.sum().item())
        acc["triangle_soft_penalty_sum"] += float(soft.sum().cpu())


def finalize_relational_metrics(acc) -> dict[str, Any]:
    return {
        "line_pair_count": int(acc["pair_count"]),
        "mixed_wedge_count": int(acc["mixed_count"]),
        "ranking_violation_count": int(acc["ranking_violation"]),
        "ranking_violation_rate": safe_ratio(acc["ranking_violation"], acc["mixed_count"]),
        "ranking_margin_violation_count": int(acc["ranking_margin_violation"]),
        "ranking_margin_violation_rate": safe_ratio(
            acc["ranking_margin_violation"], acc["mixed_count"]
        ),
        "mean_positive_minus_negative_logit_gap": safe_ratio(
            acc["gap_sum"], acc["mixed_count"]
        ),
        "triangle_count": int(acc["triangle_count"]),
        "triangle_hard_inconsistent_count": int(acc["triangle_hard_inconsistent"]),
        "triangle_hard_inconsistent_rate": safe_ratio(
            acc["triangle_hard_inconsistent"], acc["triangle_count"]
        ),
        "triangle_soft_penalty_mean": safe_ratio(
            acc["triangle_soft_penalty_sum"], acc["triangle_count"]
        ),
    }


def new_partition_accumulator() -> dict[str, float]:
    return {
        "valid": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "negative_inside": 0.0,
        "positive_cut": 0.0,
        "solve_failures": 0.0,
    }


def add_partition_metrics(acc, partition, rag, targets) -> None:
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    negative = valid & ~positive
    src, dst = rag.edge_index
    same_component = partition.node_component[src] == partition.node_component[dst]
    acc["valid"] += int(valid.sum().item())
    acc["positive"] += int(positive.sum().item())
    acc["negative"] += int(negative.sum().item())
    acc["negative_inside"] += int((negative & same_component).sum().item())
    acc["positive_cut"] += int((positive & ~same_component).sum().item())


def finalize_partition_metrics(acc) -> dict[str, Any]:
    disagreements = acc["negative_inside"] + acc["positive_cut"]
    return {
        "valid_edge_count": int(acc["valid"]),
        "negative_inside_count": int(acc["negative_inside"]),
        "negative_inside_rate": safe_ratio(acc["negative_inside"], acc["negative"]),
        "positive_cut_count": int(acc["positive_cut"]),
        "positive_cut_rate": safe_ratio(acc["positive_cut"], acc["positive"]),
        "total_disagreement_count": int(disagreements),
        "total_disagreement_rate": safe_ratio(disagreements, acc["valid"]),
        "solve_failure_count": int(acc["solve_failures"]),
    }


def evaluate(
    *,
    model,
    reasoner,
    criterion,
    support,
    source_batches,
    splits,
    amp_dtype: str,
    partial_ignore_margin_um: float,
    merge_threshold: float,
    max_pairs_per_node: int,
    ranking_margin: float,
    use_multicut: bool,
    multicut_time_limit_seconds: float,
):
    from learned.stirnet.model.partition.partitioner import GraphPartitioner

    base_edge = new_edge_accumulator()
    refined_edge = new_edge_accumulator()
    base_rel = new_relational_accumulator()
    refined_rel = new_relational_accumulator()
    base_partition = new_partition_accumulator()
    refined_partition = new_partition_accumulator()

    partitioner = None
    if use_multicut:
        partition_cfg = deepcopy(model.cfg.partition)
        partition_cfg.spatial_partition_backend = "multicut"
        partition_cfg.multicut_time_limit_seconds = float(multicut_time_limit_seconds)
        partitioner = GraphPartitioner(partition_cfg)

    reasoner.eval()
    progress_total = sum(len(split["validation_indices"]) for split in splits.values())
    progress = tqdm(
        total=progress_total,
        desc="Relational validation",
        unit="crop",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    for sample, source_batch in source_batches.items():
        for manifest_index in splits[sample]["validation_indices"]:
            record = splits[sample]["records"][manifest_index]
            crop_cpu, _ = support._materialize_crop(
                source_batch,
                record,
                partial_ignore_margin_um=partial_ignore_margin_um,
            )
            crop = support._move_crop_to_cuda(crop_cpu)
            rag, targets = forward_base_crop(
                model=model,
                crop=crop,
                criterion=criterion,
                support=support,
                amp_dtype=amp_dtype,
            )
            relation = build_relation_graph(
                rag.edge_index,
                node_count=int(rag.node_features.shape[0]),
                max_pairs_per_node=max_pairs_per_node,
            )
            base_logits = rag.spatial_edge_logits.detach().float()
            with torch.no_grad(), support._autocast_context(amp_dtype):
                refined_logits, _ = reasoner(
                    edge_embeddings=rag.edge_embeddings.detach(),
                    node_embeddings=rag.node_embeddings.detach(),
                    base_logits=base_logits,
                    relation=relation,
                )
            refined_logits = refined_logits.float()

            add_edge_metrics(base_edge, base_logits, targets, merge_threshold=merge_threshold)
            add_edge_metrics(refined_edge, refined_logits, targets, merge_threshold=merge_threshold)
            add_relational_metrics(
                base_rel,
                base_logits,
                relation,
                targets,
                merge_threshold=merge_threshold,
                ranking_margin=ranking_margin,
            )
            add_relational_metrics(
                refined_rel,
                refined_logits,
                relation,
                targets,
                merge_threshold=merge_threshold,
                ranking_margin=ranking_margin,
            )

            if partitioner is not None:
                try:
                    base_part = partitioner(
                        rag, base_logits, merge_threshold, stage="spatial"
                    )
                    add_partition_metrics(base_partition, base_part, rag, targets)
                except Exception as exc:
                    base_partition["solve_failures"] += 1
                    print(f"\n[warn] base multicut failed {sample}/{manifest_index}: {exc}", flush=True)
                try:
                    refined_part = partitioner(
                        rag, refined_logits, merge_threshold, stage="spatial"
                    )
                    add_partition_metrics(refined_partition, refined_part, rag, targets)
                except Exception as exc:
                    refined_partition["solve_failures"] += 1
                    print(f"\n[warn] refined multicut failed {sample}/{manifest_index}: {exc}", flush=True)

            progress.update(1)
            del rag, targets, relation, crop, crop_cpu
            torch.cuda.empty_cache()
            gc.collect()

    progress.close()
    reasoner.train()
    return {
        "base": {
            "edge": finalize_edge_metrics(base_edge),
            "relational": finalize_relational_metrics(base_rel),
            "multicut": finalize_partition_metrics(base_partition) if use_multicut else None,
        },
        "refined": {
            "edge": finalize_edge_metrics(refined_edge),
            "relational": finalize_relational_metrics(refined_rel),
            "multicut": finalize_partition_metrics(refined_partition) if use_multicut else None,
        },
    }


def validation_rank(
    candidate: dict,
    baseline: dict,
    *,
    allowed_positive_drop: float,
    use_multicut: bool,
):
    baseline_pa = float(baseline["edge"]["positive_accept_rate"])
    candidate_pa = float(candidate["edge"]["positive_accept_rate"])
    violation = max(0.0, baseline_pa - float(allowed_positive_drop) - candidate_pa)

    if use_multicut and candidate["multicut"] is not None:
        mc = candidate["multicut"]
        multicut_failure = int(mc["solve_failure_count"] > 0)
        negative_inside = float(mc["negative_inside_rate"])
        positive_cut = float(mc["positive_cut_rate"])
    else:
        multicut_failure = 0
        negative_inside = float(candidate["edge"]["false_merge_rate"])
        positive_cut = 0.0

    return (
        int(violation > 0.0),
        float(violation),
        int(multicut_failure),
        float(negative_inside),
        float(candidate["edge"]["false_merge_rate"]),
        float(candidate["relational"]["ranking_violation_rate"]),
        float(positive_cut),
        float(candidate["edge"]["bce"]),
        -float(candidate["edge"]["positive_accept_rate"]),
    )


# ======================================================================================
# Checkpointing
# ======================================================================================


def save_reasoner_checkpoint(
    path: Path,
    *,
    reasoner,
    optimizer,
    scaler,
    step: int,
    base_checkpoint: Path,
    run_dir: Path,
    baseline_validation: dict,
    best_validation: dict | None,
    best_rank,
    args,
) -> None:
    payload = {
        "checkpoint_version": 1,
        "experiment": EXPERIMENT_NAME,
        "global_step": int(step),
        "reasoner": reasoner.state_dict(),
        "reasoner_config": reasoner.config_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "extra": {
            "experiment": EXPERIMENT_NAME,
            "base_checkpoint": str(base_checkpoint),
            "run_dir": str(run_dir),
            "baseline_validation": baseline_validation,
            "best_validation": best_validation,
            "best_rank": None if best_rank is None else list(best_rank),
            "config": vars(args),
        },
    }
    atomic_torch_save(path, payload)


# ======================================================================================
# Main
# ======================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigation 27: direct edge-to-edge relational GNN with signed group consistency"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--spacing-xyz", default=DEFAULT_SPACING_XYZ)
    parser.add_argument("--crop-shape-zyx", default="32,192,192")
    parser.add_argument("--validation-crops-per-sample", type=int, default=6)
    parser.add_argument("--confidence-ignore-margin-um", type=float, default=1.0)
    parser.add_argument("--partial-ignore-margin-um", type=float, default=1.0)

    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--rel-hidden-dim", type=int, default=96)
    parser.add_argument("--rel-layers", type=int, default=2)
    parser.add_argument("--rel-dropout", type=float, default=0.05)
    parser.add_argument("--max-logit-delta", type=float, default=6.0)
    parser.add_argument("--max-line-pairs-per-node", type=int, default=256)

    parser.add_argument("--max-edges-per-class", type=int, default=64)
    parser.add_argument("--max-ranking-pairs", type=int, default=128)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--max-triangles", type=int, default=128)
    parser.add_argument("--ranking-weight", type=float, default=0.50)
    parser.add_argument("--triangle-weight", type=float, default=0.10)
    parser.add_argument("--preservation-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--preserve-negative-probability", type=float, default=0.10)
    parser.add_argument("--allowed-positive-drop", type=float, default=0.02)

    parser.add_argument("--productive-crop-fraction", type=float, default=0.75)
    parser.add_argument("--hard-relational-fraction", type=float, default=0.25)
    parser.add_argument(
        "--mining-max-crops-per-sample",
        type=int,
        default=0,
        help="0 mines the full train split once; result is cached",
    )

    parser.add_argument(
        "--multicut-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--multicut-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=27001)
    parser.add_argument("--run-name", default="drosophila_12_edge_relational_group_reasoning_v1")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Investigation 27 requires CUDA")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >=1")
    if args.rel_layers < 1 or args.rel_hidden_dim < 8:
        raise ValueError("Relational network must have >=1 layer and hidden dim >=8")
    if args.max_logit_delta <= 0:
        raise ValueError("--max-logit-delta must be >0")
    if args.ranking_margin < 0:
        raise ValueError("--ranking-margin must be >=0")
    for name in ("productive_crop_fraction", "hard_relational_fraction"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_','-')} must be in [0,1]")

    seed_everything(args.seed)
    support = load_inv17_support()

    resume_payload = None
    if args.resume_from:
        resume_path = resolve_checkpoint(args.resume_from)
        resume_payload = torch_load(resume_path, map_location="cpu")
        if resume_payload.get("experiment") != EXPERIMENT_NAME:
            raise ValueError("--resume-from must be an Investigation-27 reasoner checkpoint")
        base_checkpoint = resolve_checkpoint(
            resume_payload["extra"]["base_checkpoint"]
        )
    else:
        resume_path = None
        base_checkpoint = resolve_checkpoint(args.checkpoint)

    # Current Investigation-17 helper is the authoritative h100 loader.
    model, model_cfg, _, transfer = support._build_morphology_model(
        base_checkpoint, device="cuda"
    )
    if bool(getattr(model_cfg.partition, "rag_separator_barrier_enabled", False)):
        raise RuntimeError(
            "Investigation 27 intentionally isolates relational reasoning from the "
            "separator barrier. Start from the morphology-v2 h100 checkpoint, not Inv26."
        )

    # Avoid paying for a discrete multicut during every frozen base forward.
    # Validation below invokes a separate explicit multicut on base/refined logits.
    model.cfg.partition.spatial_partition_backend = "union_find"
    model.partitioner.cfg.spatial_partition_backend = "union_find"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    hidden_base = int(model_cfg.partition.rag_hidden_dim)
    reasoner = EdgeRelationalReasoner(
        edge_dim=hidden_base,
        node_dim=hidden_base,
        hidden=args.rel_hidden_dim,
        layers=args.rel_layers,
        dropout=args.rel_dropout,
        max_logit_delta=args.max_logit_delta,
    ).cuda()

    if resume_payload is not None:
        expected = reasoner.config_dict()
        actual = resume_payload.get("reasoner_config", {})
        if expected != actual:
            raise ValueError(
                "Reasoner architecture arguments do not match resume checkpoint.\n"
                f"expected={expected}\ncheckpoint={actual}"
            )
        reasoner.load_state_dict(resume_payload["reasoner"], strict=True)

    trainable = list(reasoner.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    scaler = make_scaler(amp_dtype == "fp16")
    start_step = 0
    resume_extra = {}
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["scaler"])
        start_step = int(resume_payload.get("global_step", 0))
        resume_extra = dict(resume_payload.get("extra", {}))

    from learned.stirnet.model.partition.rag import RAGCriterion
    criterion = RAGCriterion(model_cfg.partition)
    merge_threshold = float(model_cfg.partition.spatial_merge_threshold)

    samples = parse_samples(args.samples)
    xyz = parse_triplet(args.spacing_xyz, cast=float)
    spacing_zyx = (float(xyz[2]), float(xyz[1]), float(xyz[0]))
    crop_shape = tuple(int(v) for v in parse_triplet(args.crop_shape_zyx, cast=int))
    nis3d_root = support._discover_nis3d_root(
        samples, data_dir=args.data_dir, execution_mode="local"
    )

    # Reuse the exact large full-volume source cache from Investigation 17.
    source_cache_root = (
        ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / "17_morphology_rag_multicrop_training"
        / "cache"
    )
    print(f"[cache] shared NIS3D source cache: {source_cache_root}", flush=True)
    source_batches = {}
    source_reports = {}
    for sample in samples:
        namespace = support._data_signature(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_zyx,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
        )
        expected_cache = source_cache_root / "source" / namespace / f"{sample}.pt"
        print(
            f"[cache] {sample}: {'HIT' if expected_cache.is_file() else 'MISS'} {expected_cache}",
            flush=True,
        )
        batch, report = support._prepare_sample_batch(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=spacing_zyx,
            confidence_ignore_margin_um=args.confidence_ignore_margin_um,
            cache_root=source_cache_root,
            cache_namespace=namespace,
        )
        source_batches[sample] = batch
        source_reports[sample] = report

    splits = support._build_split(
        source_batches,
        crop_shape_zyx=crop_shape,
        validation_crops_per_sample=args.validation_crops_per_sample,
    )

    experiment_root = ROOT / "runs" / "stirnet" / "investigations" / EXPERIMENT_NAME
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_root / "attempts" / f"{timestamp}_{args.run_name}"
    recovery_dir = experiment_root / "recovery" / args.run_name
    mining_cache_root = experiment_root / "mining_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)

    mining = mine_relational_crops(
        checkpoint=base_checkpoint,
        model=model,
        criterion=criterion,
        support=support,
        source_batches=source_batches,
        splits=splits,
        samples=samples,
        amp_dtype=amp_dtype,
        partial_ignore_margin_um=args.partial_ignore_margin_um,
        max_pairs_per_node=args.max_line_pairs_per_node,
        ranking_margin=args.ranking_margin,
        mining_max_crops_per_sample=args.mining_max_crops_per_sample,
        cache_root=mining_cache_root,
    )
    atomic_json(run_dir / "relational_mining.json", mining)

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "run_name": args.run_name,
        "base_checkpoint": str(base_checkpoint),
        "resume_from": None if resume_path is None else str(resume_path),
        "start_step": start_step,
        "max_steps": args.max_steps,
        "samples": list(samples),
        "spacing_zyx_um": list(spacing_zyx),
        "crop_shape_zyx": list(crop_shape),
        "merge_threshold": merge_threshold,
        "amp_dtype": amp_dtype,
        "reasoner_config": reasoner.config_dict(),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "base_transfer": transfer,
        "source_reports": source_reports,
        "mining_summary": mining["summary"],
        "args": vars(args),
    }
    atomic_json(run_dir / "manifest.json", manifest)

    props = torch.cuda.get_device_properties(0)
    print("=" * 118)
    print("STIR-Net Investigation 27 — direct edge-to-edge relational group reasoning")
    print("=" * 118)
    print(f"GPU                       : {props.name}")
    print(f"VRAM                      : {props.total_memory / 2**30:.2f} GiB")
    print(f"Base checkpoint           : {base_checkpoint}")
    print("Production separator      : OFF / isolated from Inv26")
    print("Frozen h100               : ALL parameters")
    print(f"Relational hidden/layers  : {args.rel_hidden_dim} / {args.rel_layers}")
    print(f"Trainable parameters      : {sum(p.numel() for p in trainable):,}")
    print(f"Merge / multicut q        : {merge_threshold:.3f}")
    print(f"Training steps            : {start_step} -> {args.max_steps}")
    print(
        "Loss weights              : "
        f"BCE=1 ranking={args.ranking_weight} triangle={args.triangle_weight} "
        f"preserve={args.preservation_weight} residual={args.residual_weight}"
    )
    print(
        "Training crop mixture     : "
        f"productive={args.productive_crop_fraction:.2f}, "
        f"hard-within-productive={args.hard_relational_fraction:.2f}"
    )
    print(
        "Mined relation crops      : "
        f"productive={mining['summary']['productive_crop_count']}/"
        f"{mining['summary']['candidate_crop_count']} "
        f"hard={mining['summary']['hard_crop_count']} "
        f"mixed-wedges={mining['summary']['mixed_wedge_count']}"
    )
    print(f"Multicut validation       : {args.multicut_validation}")
    print(f"Run directory             : {run_dir}")
    print("=" * 118, flush=True)

    baseline_validation = resume_extra.get("baseline_validation")
    if baseline_validation is None:
        print("[validation] computing frozen h100 / initial reasoner baseline ...", flush=True)
        baseline_validation = evaluate(
            model=model,
            reasoner=reasoner,
            criterion=criterion,
            support=support,
            source_batches=source_batches,
            splits=splits,
            amp_dtype=amp_dtype,
            partial_ignore_margin_um=args.partial_ignore_margin_um,
            merge_threshold=merge_threshold,
            max_pairs_per_node=args.max_line_pairs_per_node,
            ranking_margin=args.ranking_margin,
            use_multicut=args.multicut_validation,
            multicut_time_limit_seconds=args.multicut_time_limit_seconds,
        )
        atomic_json(run_dir / "baseline_validation.json", baseline_validation)
        print(json.dumps(jsonable(baseline_validation), indent=2), flush=True)

    # The BASE model, not the randomly initialized reasoner, owns the step-0 rank.
    base_metrics = baseline_validation["base"]
    baseline_rank = validation_rank(
        base_metrics,
        base_metrics,
        allowed_positive_drop=args.allowed_positive_drop,
        use_multicut=args.multicut_validation,
    )
    best_rank = tuple(resume_extra.get("best_rank", baseline_rank))
    best_validation = resume_extra.get("best_validation")
    best_source = "resume" if resume_payload is not None and best_validation is not None else "frozen_h100_baseline"

    sample_steps = {
        sample: {"ordinary": 0, "productive": 0, "hard": 0}
        for sample in samples
    }
    sampling_counts = {"ordinary": 0, "productive": 0, "hard": 0}
    optimizer_step = int(start_step)
    attempt = 0
    skipped = 0
    started = time.perf_counter()
    progress = tqdm(
        total=args.max_steps,
        initial=optimizer_step,
        desc="Edge-relational RAG",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    while optimizer_step < args.max_steps:
        job = choose_training_job(
            support=support,
            splits=splits,
            mining=mining,
            samples=samples,
            attempt=attempt,
            sample_steps=sample_steps,
            productive_fraction=args.productive_crop_fraction,
            hard_fraction=args.hard_relational_fraction,
            seed=args.seed,
        )
        attempt += 1
        sample = job["sample"]
        manifest_index = int(job["manifest_index"])
        record = splits[sample]["records"][manifest_index]
        crop_cpu, _ = support._materialize_crop(
            source_batches[sample],
            record,
            partial_ignore_margin_um=args.partial_ignore_margin_um,
        )
        crop = support._move_crop_to_cuda(crop_cpu)
        rag, targets = forward_base_crop(
            model=model,
            crop=crop,
            criterion=criterion,
            support=support,
            amp_dtype=amp_dtype,
        )
        relation = build_relation_graph(
            rag.edge_index,
            node_count=int(rag.node_features.shape[0]),
            max_pairs_per_node=args.max_line_pairs_per_node,
        )
        base_logits = rag.spatial_edge_logits.detach().float()

        optimizer.zero_grad(set_to_none=True)
        with support._autocast_context(amp_dtype):
            refined_logits, delta = reasoner(
                edge_embeddings=rag.edge_embeddings.detach(),
                node_embeddings=rag.node_embeddings.detach(),
                base_logits=base_logits,
                relation=relation,
            )
            loss, loss_report = training_losses(
                refined_logits=refined_logits,
                delta=delta,
                base_logits=base_logits,
                targets=targets,
                relation=relation,
                merge_threshold=merge_threshold,
                max_edges_per_class=args.max_edges_per_class,
                max_ranking_pairs=args.max_ranking_pairs,
                ranking_margin=args.ranking_margin,
                max_triangles=args.max_triangles,
                ranking_weight=args.ranking_weight,
                triangle_weight=args.triangle_weight,
                preservation_weight=args.preservation_weight,
                residual_weight=args.residual_weight,
                preserve_negative_probability=args.preserve_negative_probability,
            )

        if loss_report["selected_edge_count"] == 0 and loss_report["ranking_pair_count"] == 0:
            skipped += 1
            del rag, targets, relation, crop, crop_cpu
            torch.cuda.empty_cache()
            gc.collect()
            if skipped > max(1000, args.max_steps * 20):
                raise RuntimeError("Too many crops contain no usable relational supervision")
            continue

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer_step += 1
        sampling_counts[job["pool"]] += 1
        progress.update(1)
        progress.set_postfix(
            {
                "loss": f"{loss_report['total']:.4f}",
                "bce": f"{loss_report['bce']:.3f}",
                "rank": f"{loss_report['ranking']:.3f}",
                "pairs": loss_report["ranking_pair_count"],
                "tri": loss_report["triangle_count"],
                "pool": job["pool"][0].upper(),
                "skip": skipped,
            },
            refresh=False,
        )

        append_jsonl(
            run_dir / "training.jsonl",
            {
                "step": optimizer_step,
                "sample": sample,
                "manifest_index": manifest_index,
                "pool": job["pool"],
                "provenance": job["provenance"],
                "line_pair_count": relation.pair_count,
                **loss_report,
            },
        )

        should_validate = optimizer_step % args.validation_every == 0 or optimizer_step == args.max_steps
        should_checkpoint = optimizer_step % args.checkpoint_every == 0 or optimizer_step == args.max_steps

        if should_validate:
            validation = evaluate(
                model=model,
                reasoner=reasoner,
                criterion=criterion,
                support=support,
                source_batches=source_batches,
                splits=splits,
                amp_dtype=amp_dtype,
                partial_ignore_margin_um=args.partial_ignore_margin_um,
                merge_threshold=merge_threshold,
                max_pairs_per_node=args.max_line_pairs_per_node,
                ranking_margin=args.ranking_margin,
                use_multicut=args.multicut_validation,
                multicut_time_limit_seconds=args.multicut_time_limit_seconds,
            )
            refined = validation["refined"]
            rank = validation_rank(
                refined,
                base_metrics,
                allowed_positive_drop=args.allowed_positive_drop,
                use_multicut=args.multicut_validation,
            )
            append_jsonl(
                run_dir / "validation.jsonl",
                {"step": optimizer_step, "rank": list(rank), **validation},
            )
            mc_text = ""
            if refined["multicut"] is not None:
                mc_text = (
                    f" | MC neg-in={refined['multicut']['negative_inside_rate']:.5f} "
                    f"pos-cut={refined['multicut']['positive_cut_rate']:.5f}"
                )
            print(
                f"\n[val {optimizer_step}] "
                f"edge FM={refined['edge']['false_merge_rate']:.5f} "
                f"PA={refined['edge']['positive_accept_rate']:.5f} "
                f"BCE={refined['edge']['bce']:.5f} | "
                f"wedge-viol={refined['relational']['ranking_violation_rate']:.5f}"
                f"{mc_text}",
                flush=True,
            )

            # Baseline is an actual contender. Only save BEST when trained
            # relational reasoning really beats it under the safety-first rank.
            if tuple(rank) < tuple(best_rank):
                best_rank = tuple(rank)
                best_validation = validation
                best_source = f"trained_step_{optimizer_step}"
                save_reasoner_checkpoint(
                    recovery_dir / "best_reasoner.pt",
                    reasoner=reasoner,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=optimizer_step,
                    base_checkpoint=base_checkpoint,
                    run_dir=run_dir,
                    baseline_validation=baseline_validation,
                    best_validation=best_validation,
                    best_rank=best_rank,
                    args=args,
                )
                print(f"[best] trained reasoner beats previous best at step {optimizer_step}: {best_rank}", flush=True)

        if should_checkpoint:
            save_reasoner_checkpoint(
                recovery_dir / "latest_reasoner.pt",
                reasoner=reasoner,
                optimizer=optimizer,
                scaler=scaler,
                step=optimizer_step,
                base_checkpoint=base_checkpoint,
                run_dir=run_dir,
                baseline_validation=baseline_validation,
                best_validation=best_validation,
                best_rank=best_rank,
                args=args,
            )

        del rag, targets, relation, crop, crop_cpu
        torch.cuda.empty_cache()
        gc.collect()

    progress.close()
    elapsed = time.perf_counter() - started
    summary = {
        "status": "success",
        "experiment": EXPERIMENT_NAME,
        "run_name": args.run_name,
        "base_checkpoint": str(base_checkpoint),
        "final_step": optimizer_step,
        "skipped_attempts": skipped,
        "sampling_counts": sampling_counts,
        "mining_summary": mining["summary"],
        "baseline_validation": baseline_validation,
        "baseline_rank": list(baseline_rank),
        "best_source": best_source,
        "best_rank": list(best_rank),
        "best_validation": best_validation,
        "best_reasoner_path": (
            str(recovery_dir / "best_reasoner.pt")
            if (recovery_dir / "best_reasoner.pt").is_file()
            else None
        ),
        "latest_reasoner_path": str(recovery_dir / "latest_reasoner.pt"),
        "elapsed_seconds": float(elapsed),
    }
    atomic_json(run_dir / "summary.json", summary)

    print("=" * 118)
    print("Investigation 27 complete")
    print(f"Elapsed                   : {duration(elapsed)}")
    print(f"Best source               : {best_source}")
    if (recovery_dir / "best_reasoner.pt").is_file():
        print(f"Best reasoner             : {recovery_dir / 'best_reasoner.pt'}")
    else:
        print("Best reasoner             : NONE — frozen h100 remained better")
    print(f"Latest reasoner           : {recovery_dir / 'latest_reasoner.pt'}")
    print(f"Summary                   : {run_dir / 'summary.json'}")
    print("=" * 118)


if __name__ == "__main__":
    main()
