from __future__ import annotations

"""
Stage 09: isolated temporal TWO->ONE overfit for STIR-Net (candidate-0 near-pass geometry).

Scientific question
-------------------
Given a current-frame spatial RAG that is deliberately WRONG for one target
cell (GT 6 is represented by two provisional spatial instances), can the real
STIR-Net temporal pathway use the controlled Trackastra/history context created
by Stage 08 to override that strong spatial SEPARATE decision and restore ONE
final instance?

This investigation deliberately stops before LocalGeometryRefiner.  The target
case is graph-recoverable: Stage 08 constructed the two pieces from safe
supervoxels, so temporal reasoning only has to alter RAG grouping.

Production modules exercised
----------------------------
Frozen / prepared once:
    candidate-0 explicit geometry from Stage-01 last_attempt_predictions.pt
    one frozen feature-donor Stage-01 forward pass for D0/D1/D2/hidden features
    production safe-supervoxel topology from Stage 08
    RAGBuilder + SpatialRAGNetwork after a tiny internal memorization warm-up
    GraphPartitioner

Trainable:
    InstanceTokenizer
    HistoricalInstanceEncoder
    TemporalGraphEncoder
        - detection message passing
        - pooled tracklet hypothesis message passing
    TemporalSpatialObserver
    InstanceTemporalReasoner

The training loop uses the REAL controlled Stage-08 temporal graph, including
Trackastra associations, historical instance grids, temporal status, and the
22-D tracklet-hypothesis graph.

Important coordinate contract
-----------------------------
Stage-08 temporal records are expressed in all-cell-ROI-centred physical
coordinates.  The candidate-0 spatial crop is crop-centred.  Stage 09 rebases
the temporal reference positions and the absolute detection-position columns in
graph_x into the candidate-crop frame before temporal/spatial cross reasoning.
Relative temporal edge features are translation invariant and are left intact.

Controlled spatial error
------------------------
The Stage-08 split artifact contains:
    safe_supervoxels_crop
    controlled_partition_bbox

Stage 09 maps each target supervoxel to controlled piece A/B and forces every
RAG edge crossing A<->B to a strong spatial logit of -6.0.  It then verifies
that the production GraphPartitioner produces exactly TWO target components
before temporal training.

Strict PASS
-----------
A PASS requires:
    - internal candidate-0 RAG warm-up is GT-perfect;
    - forced spatial target merge probability <= 0.01;
    - spatial partition contains exactly TWO target components;
    - final temporal target merge probability >= 0.95;
    - final valid-edge accuracy == 1.0;
    - final GT-backed pairwise grouping accuracy == 1.0;
    - no false merge / false split pairs;
    - target is ONE final component;
    - existence and split auxiliary accuracies == 1.0;
    - ZERO temporal input preserves the forced spatial logits exactly and leaves
      the target split into TWO components;
    - the strict PASS remains true for three consecutive evaluations.

The no-hypothesis and no-history variants are reported as diagnostics after
training; they are not PASS requirements.

Prerequisite
------------
Stage 08 controlled preparation must already exist.

Stage 09 intentionally DOES NOT require candidate 0 to have a formal Stage-01
PASS checkpoint.  The completed 700-step candidate-0 run is already suitable
for this temporal experiment, even though its centroid endpoint median was
2.111 um versus the Stage-01 research gate of 2.000 um.

The explicit candidate-0 geometry is loaded from:

    data/learned/stirnet/dense_geometry_temporal_merge/
        last_attempt_predictions.pt

That artifact contains the actual candidate-0 foreground/surface/separator/SDF/
flow/centroid-offset/seed predictions.  Stage 09 applies its own suitability
gate to those predictions; centroid endpoint error is reported but is not a
blocking criterion because Stage 08 has already frozen the safe-supervoxel
topology used by this experiment.

D0/D1/D2 and hidden geometry features are obtained with ONE frozen forward pass
from an already-successful Stage-01 checkpoint, by default:

    data/learned/stirnet/dense_geometry_debug/latest_success.pt

Those weights are a FEATURE DONOR ONLY.  Their explicit geometry predictions
are discarded and replaced by the candidate-0 last-attempt predictions above.

Run
---
    python investigations/stirnet/09_temporal_reasoning_overfit.py

Meaningful temporal-evidence dependence test
--------------------------------------------
After Stage 09 has produced latest_success.pt, run the trained model WITHOUT
further optimization under stronger counterfactual temporal controls:

    python investigations/stirnet/09_temporal_reasoning_overfit.py \
        --evidence-dependence

The command evaluates:
    1. correct temporal context;
    2. empty temporal context;
    3. non-empty context shifted away from the target (position/support control);
    4. WRONG nearby-cell encoded tokens placed at the exact target-track refs;
    5. a global deterministic token derangement with every temporal ref/support
       value left unchanged.

The last two are the important content controls.  They preserve temporal
support/positions while corrupting the encoded temporal evidence.  Therefore a
model that still merges with high confidence has not demonstrated meaningful
temporal-content dependence on this one memorized example.

Visualize latest successful/last attempt
----------------------------------------
    python investigations/stirnet/09_temporal_reasoning_overfit.py --visualize
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# =============================================================================
# REPOSITORY / ADJACENT INVESTIGATION IMPORTS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _import_adjacent(module_name: str, filename: str):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing required investigation: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stage01 = _import_adjacent(
    "_stirnet_stage01_for_stage09",
    "01_dense_geometry_overfit.py",
)

from learned.stirnet.model.geometry.derived import build_geometry_derived_cache
from learned.stirnet.model.instances.tokenizer import InstanceTokenizer
from learned.stirnet.model.partition.graph_net import SpatialRAGNetwork
from learned.stirnet.model.partition.partitioner import GraphPartitioner
from learned.stirnet.model.partition.rag import (
    RAGBuilder,
    RAGCriterion,
    RAGTargets,
)
from learned.stirnet.model.partition.statistics import (
    build_supervoxel_statistics,
)
from learned.stirnet.model.temporal.fusion import InstanceTemporalReasoner
from learned.stirnet.model.temporal.graph_encoder import TemporalGraphEncoder
from learned.stirnet.model.temporal.history import HistoricalInstanceEncoder
from learned.stirnet.model.temporal.observer import TemporalSpatialObserver
from learned.stirnet.model.types import (
    GeometryState,
    InstanceState,
    PartitionState,
    RAGState,
    ReasoningState,
    TemporalInput,
    TemporalState,
)
from learned.stirnet.training.config import LossConfig
from learned.stirnet.training.criterion import build_instance_targets

try:
    from tqdm.auto import trange as tqdm_trange
except ImportError:
    tqdm_trange = None


# =============================================================================
# DEFAULTS / PASS THRESHOLDS
# =============================================================================

EXPECTED_CANDIDATE_INDEX = 0
EXPECTED_TARGET_GT_ID = 6

DEFAULT_SEED = 230525
DEFAULT_STEPS = 1000
DEFAULT_LR = 1e-3
DEFAULT_EVAL_EVERY = 20
DEFAULT_RAG_WARMUP_STEPS = 500
DEFAULT_RAG_WARMUP_LR = 1e-3
DEFAULT_RAG_EVAL_EVERY = 20

GRAD_CLIP_NORM = 5.0
PASS_STREAK = 3

FORCED_SPATIAL_LOGIT = -6.0

LOSS_TARGET_EDGE_WEIGHT = 1.0
LOSS_OTHER_EDGE_WEIGHT = 0.25
LOSS_EXISTENCE_WEIGHT = 0.10
LOSS_SPLIT_WEIGHT = 0.10

PASS = {
    "rag_warmup_edge_accuracy_min": 1.0,
    "rag_warmup_min_merge_probability_min": 0.95,
    "rag_warmup_max_separate_probability_max": 0.05,
    "rag_warmup_pairwise_accuracy_min": 1.0,
    "forced_spatial_merge_probability_max": 0.01,
    "target_final_merge_probability_min": 0.95,
    "final_edge_accuracy_min": 1.0,
    "final_pairwise_accuracy_min": 1.0,
    "existence_accuracy_min": 1.0,
    "split_accuracy_min": 1.0,
    "zero_temporal_max_logit_change_max": 0.0,
}

EVIDENCE_DEPENDENCE = {
    # The trained Stage-09 baseline must still solve the intended correction.
    "correct_merge_probability_min": 0.95,
    # Counterfactual content should no longer trigger the merge.
    "counterfactual_merge_probability_max": 0.50,
    # Require a material confidence drop, not a threshold-edge coincidence.
    "correct_vs_counterfactual_margin_min": 0.25,
    # A non-empty but spatially irrelevant context should behave like no context.
    "shift_distance_dref": 8.0,
}

DEFAULT_SAMPLE = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_merge_debug_crop.pt"
)
DEFAULT_GEOMETRY_PREDICTIONS = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "dense_geometry_temporal_merge"
    / "last_attempt_predictions.pt"
)
DEFAULT_FEATURE_DONOR_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "dense_geometry_debug"
    / "latest_success.pt"
)

# Stage-09-specific suitability gate.  This intentionally differs from the
# Stage-01 formal research gate: centroid endpoint quality is diagnostic only
# here because Stage 08 has already fixed the safe-supervoxel topology.
STAGE09_GEOMETRY_SUITABILITY = {
    "foreground_dice_min": 0.95,
    "surface_dice_min": 0.78,
    "separator_dice_min": 0.72,
    "separator_recall_min": 0.80,
    "sdf_mae_max": 0.13,
    "sdf_corr_min": 0.92,
    "flow_cosine_min": 0.85,
    "flow_angle_median_deg_max": 32.0,
    "seed_mae_foreground_max": 0.15,
}
DEFAULT_CONTROLLED_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_merge_case"
    / "controlled_error"
)
DEFAULT_CONTROLLED_CONTEXT = (
    DEFAULT_CONTROLLED_DIR / "controlled_temporal_context.pt"
)
DEFAULT_CONTROLLED_SPLIT = (
    DEFAULT_CONTROLLED_DIR / "controlled_split.pt"
)
DEFAULT_CASE_JSON = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_merge_case_selection.json"
)
DEFAULT_RESULTS = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "temporal_reasoning_overfit"
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_dict_cpu(module: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def gpu_peak_gib() -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated() / 1024**3,
        torch.cuda.max_memory_reserved() / 1024**3,
    )


def binary_accuracy(logits: Tensor, target: Tensor) -> float:
    if target.numel() == 0:
        return 1.0
    return float(
        ((logits >= 0) == (target > 0.5))
        .float()
        .mean()
        .item()
    )


def _clone_partition(partition: PartitionState) -> PartitionState:
    return replace(
        partition,
        labels=[row.detach().clone() for row in partition.labels],
        node_component=partition.node_component.detach().clone(),
        node_component_global=partition.node_component_global.detach().clone(),
        component_count_per_batch=(
            partition.component_count_per_batch.detach().clone()
        ),
        edge_logits=partition.edge_logits.detach().clone(),
    )


def detach_rag(rag: RAGState) -> RAGState:
    return replace(
        rag,
        node_features=rag.node_features.detach(),
        node_embeddings=rag.node_embeddings.detach(),
        node_batch=rag.node_batch.detach(),
        node_supervoxel_id=rag.node_supervoxel_id.detach(),
        node_centroid_um=rag.node_centroid_um.detach(),
        node_volume_voxels=rag.node_volume_voxels.detach(),
        edge_index=rag.edge_index.detach(),
        edge_features=rag.edge_features.detach(),
        edge_embeddings=rag.edge_embeddings.detach(),
        spatial_edge_logits=rag.spatial_edge_logits.detach(),
        edge_batch=rag.edge_batch.detach(),
        supervoxel_labels=[
            labels.detach() for labels in rag.supervoxel_labels
        ],
        node_offsets=rag.node_offsets.detach(),
    )


# =============================================================================
# CANDIDATE-0 GEOMETRY + FROZEN FEATURE DONOR
# =============================================================================


def _prediction_probability_to_logit(
    value: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    value = value.float().clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def validate_candidate0_geometry_predictions(
    payload: dict[str, Any],
    batch,
) -> dict[str, Any]:
    kind = str(payload.get("kind", ""))
    if kind != "stirnet_dense_geometry_stage01_attempt_predictions":
        raise RuntimeError(
            "Candidate-0 geometry artifact has unexpected kind: "
            f"{kind!r}."
        )

    stage = str(payload.get("stage", ""))
    if stage not in {"joint", "full"}:
        raise RuntimeError(
            "Stage 09 requires candidate-0 joint/full geometry predictions; "
            f"found stage={stage!r}."
        )

    selection = dict(payload.get("sample_selection", {}))
    candidate_index = selection.get("candidate_index")
    if candidate_index is None:
        candidate_index = dict(
            payload.get("attempt", {})
        ).get("sample_selection", {}).get("candidate_index")

    if candidate_index is not None and int(candidate_index) != EXPECTED_CANDIDATE_INDEX:
        raise RuntimeError(
            "Candidate-0 geometry predictions belong to a different candidate: "
            f"{candidate_index}."
        )

    artifact_shape = payload.get("sample_shape_zyx")
    expected_shape = list(batch.spatial_inputs.shape[-3:])
    if artifact_shape is not None and list(artifact_shape) != expected_shape:
        raise RuntimeError(
            "Candidate-0 geometry prediction/sample shape mismatch: "
            f"artifact={artifact_shape}, sample={expected_shape}."
        )

    predictions = dict(payload.get("predictions", {}))
    required_predictions = {
        "foreground",
        "surface",
        "separator",
        "sdf",
        "flow",
        "centroid_offset",
        "seed",
    }
    missing_predictions = required_predictions - set(predictions)
    if missing_predictions:
        raise RuntimeError(
            "Candidate-0 prediction artifact is missing fields: "
            f"{sorted(missing_predictions)}"
        )

    metrics = dict(payload.get("metrics", {}))
    if not metrics:
        metrics = dict(payload.get("attempt", {}).get("metrics", {}))

    rules = STAGE09_GEOMETRY_SUITABILITY
    failures: list[str] = []

    minimum_rules = {
        "foreground_dice": rules["foreground_dice_min"],
        "surface_dice": rules["surface_dice_min"],
        "separator_dice": rules["separator_dice_min"],
        "separator_recall": rules["separator_recall_min"],
        "sdf_corr": rules["sdf_corr_min"],
        "flow_cosine": rules["flow_cosine_min"],
    }
    maximum_rules = {
        "sdf_mae": rules["sdf_mae_max"],
        "flow_angle_median_deg": rules["flow_angle_median_deg_max"],
        "seed_mae_foreground": rules["seed_mae_foreground_max"],
    }

    for name, threshold in minimum_rules.items():
        value = float(metrics.get(name, float("nan")))
        if not np.isfinite(value) or value < threshold:
            failures.append(f"{name}={value:.6f} < {threshold}")

    for name, threshold in maximum_rules.items():
        value = float(metrics.get(name, float("nan")))
        if not np.isfinite(value) or value > threshold:
            failures.append(f"{name}={value:.6f} > {threshold}")

    centroid_endpoint = float(
        metrics.get("centroid_endpoint_median_um", float("nan"))
    )

    if failures:
        raise RuntimeError(
            "Candidate-0 last-attempt geometry is not suitable for Stage 09. "
            "Failures: " + "; ".join(failures)
        )

    return {
        "stage": stage,
        "formal_stage01_passed": bool(
            payload.get("passed", payload.get("attempt", {}).get("passed", False))
        ),
        "metrics": metrics,
        "centroid_endpoint_median_um_diagnostic": centroid_endpoint,
        "suitability_gate": dict(STAGE09_GEOMETRY_SUITABILITY),
        "suitable_for_stage09": True,
    }


@torch.no_grad()
def run_feature_donor(
    batch,
    checkpoint_path: Path,
    cfg,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing frozen feature-donor Stage-01 checkpoint:\n  "
            f"{checkpoint_path}\n"
            "Use --feature-donor-checkpoint to point at any successful "
            "joint/full Stage-01 checkpoint."
        )

    checkpoint = torch_load(checkpoint_path)
    stage = str(checkpoint.get("stage", ""))
    if stage not in {"joint", "full"}:
        raise RuntimeError(
            "Feature donor must be a successful Stage-01 joint/full checkpoint; "
            f"found stage={stage!r}."
        )
    if "model_state" not in checkpoint:
        raise RuntimeError(
            "Feature-donor checkpoint does not contain model_state."
        )

    # Deliberately DO NOT require the donor checkpoint to come from candidate 0.
    # It supplies learned feature representations only; its geometry outputs are
    # replaced below by the actual candidate-0 700-step predictions.
    model = stage01.DenseGeometryOnlyModel(cfg).to(device)
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    acquisition = model.acquisition(
        batch.spacing_um,
        batch.dref_um,
    )
    stem = model.evidence_stem(
        batch.spatial_inputs,
        acquisition,
    )
    pyramid, decoded = model.spatial_backbone(
        stem,
        batch.spacing_um,
        acquisition,
        padding_mask=None,
    )
    donor_geometry = model.geometry_decoder(
        decoded.d0,
        acquisition,
    )

    donor_selection = dict(checkpoint.get("sample_selection", {}))
    donor_info = {
        "stage": stage,
        "source_candidate_index": donor_selection.get("candidate_index"),
        "checkpoint_kind": checkpoint.get("kind"),
    }

    del model, acquisition, stem
    return donor_geometry, pyramid, decoded, donor_info


def build_candidate0_geometry(
    prediction_payload: dict[str, Any],
    donor_geometry: GeometryState,
    batch,
    device: torch.device,
) -> GeometryState:
    predictions = dict(prediction_payload["predictions"])

    def scalar_probability(name: str) -> Tensor:
        value = (
            torch.as_tensor(predictions[name])
            .float()
            .to(device)
        )
        if tuple(value.shape) != tuple(batch.gt_labels.shape):
            raise RuntimeError(
                f"Prediction {name!r} shape {tuple(value.shape)} does not "
                f"match candidate crop {tuple(batch.gt_labels.shape)}."
            )
        return _prediction_probability_to_logit(
            value
        )[None, None]

    def scalar_value(name: str) -> Tensor:
        value = (
            torch.as_tensor(predictions[name])
            .float()
            .to(device)
        )
        if tuple(value.shape) != tuple(batch.gt_labels.shape):
            raise RuntimeError(
                f"Prediction {name!r} shape {tuple(value.shape)} does not "
                f"match candidate crop {tuple(batch.gt_labels.shape)}."
            )
        return value[None, None]

    def vector_value(name: str) -> Tensor:
        value = (
            torch.as_tensor(predictions[name])
            .float()
            .to(device)
        )
        expected = (3, *tuple(batch.gt_labels.shape))
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"Prediction {name!r} shape {tuple(value.shape)} does not "
                f"match expected {expected}."
            )
        return value[None]

    return GeometryState(
        foreground_logits=scalar_probability("foreground"),
        surface_logits=scalar_probability("surface"),
        separator_logits=scalar_probability("separator"),
        sdf=scalar_value("sdf"),
        flow=vector_value("flow"),
        centroid_offset=vector_value("centroid_offset"),
        seed_logits=scalar_probability("seed"),
        # Hidden decoder geometry is unavailable in last_attempt_predictions.pt.
        # Retain only this hidden representation from the frozen feature donor.
        features=donor_geometry.features,
        feature_spacing_um=batch.spacing_um,
    )


@torch.no_grad()
def prepare_candidate0_spatial_evidence(
    batch,
    geometry_predictions_path: Path,
    feature_donor_checkpoint: Path,
    cfg,
    device: torch.device,
):
    if not geometry_predictions_path.exists():
        raise FileNotFoundError(
            "Missing candidate-0 last-attempt geometry predictions:\n  "
            f"{geometry_predictions_path}\n"
            "This should be the completed 700-step Stage-01 artifact."
        )

    prediction_payload = torch_load(
        geometry_predictions_path
    )
    suitability = validate_candidate0_geometry_predictions(
        prediction_payload,
        batch,
    )

    donor_geometry, pyramid, decoded, donor_info = run_feature_donor(
        batch,
        feature_donor_checkpoint,
        cfg,
        device,
    )
    geometry = build_candidate0_geometry(
        prediction_payload,
        donor_geometry,
        batch,
        device,
    )

    return (
        geometry,
        pyramid,
        decoded,
        suitability,
        donor_info,
    )


# =============================================================================
# STAGE-08 ARTIFACTS / CONTROLLED PIECE MAP
# =============================================================================


@dataclass
class Stage08Artifacts:
    context: dict[str, Any]
    split: dict[str, Any]
    case_report: dict[str, Any]
    safe_supervoxels: Tensor
    piece_map_crop: Tensor
    seed_supervoxels: tuple[int, int]


def _slices_from_pairs(
    pairs: list[list[int]],
) -> tuple[slice, slice, slice]:
    if len(pairs) != 3:
        raise ValueError("Expected three z/y/x slice pairs.")
    return tuple(
        slice(int(pair[0]), int(pair[1]))
        for pair in pairs
    )  # type: ignore[return-value]


def reconstruct_piece_map_crop(
    safe_supervoxels: Tensor,
    split_artifact: dict[str, Any],
) -> Tensor:
    crop_pairs = split_artifact.get("crop_slices_zyx")
    bbox_pairs = split_artifact.get("bbox_zyx")
    if crop_pairs is None or bbox_pairs is None:
        raise RuntimeError(
            "Stage-08 controlled_split.pt is missing crop/bbox coordinates."
        )

    crop = _slices_from_pairs(crop_pairs)
    bbox = _slices_from_pairs(bbox_pairs)
    local = (
        torch.as_tensor(
            split_artifact["controlled_partition_bbox"]
        )
        .long()
        .cpu()
    )

    expected_bbox_shape = tuple(
        int(axis.stop) - int(axis.start)
        for axis in bbox
    )
    if tuple(local.shape) != expected_bbox_shape:
        raise RuntimeError(
            "controlled_partition_bbox shape does not match bbox_zyx: "
            f"{tuple(local.shape)} vs {expected_bbox_shape}."
        )

    result = torch.zeros_like(
        safe_supervoxels,
        dtype=torch.long,
        device="cpu",
    )

    full_low = [
        max(int(crop[a].start), int(bbox[a].start))
        for a in range(3)
    ]
    full_high = [
        min(int(crop[a].stop), int(bbox[a].stop))
        for a in range(3)
    ]
    if any(
        high <= low
        for low, high in zip(full_low, full_high)
    ):
        raise RuntimeError(
            "Controlled split bbox does not intersect candidate-0 crop."
        )

    destination = tuple(
        slice(
            full_low[a] - int(crop[a].start),
            full_high[a] - int(crop[a].start),
        )
        for a in range(3)
    )
    source = tuple(
        slice(
            full_low[a] - int(bbox[a].start),
            full_high[a] - int(bbox[a].start),
        )
        for a in range(3)
    )
    result[destination] = local[source]
    return result


def load_stage08_artifacts(
    context_path: Path,
    split_path: Path,
    case_json: Path,
) -> Stage08Artifacts:
    if not context_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-08 controlled temporal context:\n  "
            f"{context_path}\nRun Stage 08 --prepare-temporal-overfit first."
        )
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-08 controlled split:\n  {split_path}"
        )
    if not case_json.exists():
        raise FileNotFoundError(
            f"Missing Stage-08 case report:\n  {case_json}"
        )

    context = torch_load(context_path)
    split = torch_load(split_path)
    case_report = json.loads(
        case_json.read_text(encoding="utf-8")
    )

    if context.get("kind") != "stirnet_stage08_controlled_temporal_context":
        raise RuntimeError(
            "controlled_temporal_context.pt has an unexpected artifact kind."
        )
    if not bool(
        context.get("report", {}).get(
            "controlled_temporal_ready",
            False,
        )
    ):
        raise RuntimeError(
            "Stage-08 controlled temporal context is not marked ready."
        )

    target_gt = int(context.get("target_gt_id", -1))
    if target_gt != EXPECTED_TARGET_GT_ID:
        raise RuntimeError(
            f"Stage-08 target GT={target_gt}; "
            f"expected {EXPECTED_TARGET_GT_ID}."
        )

    safe = (
        torch.as_tensor(split["safe_supervoxels_crop"])
        .long()
        .cpu()
    )
    piece_map = reconstruct_piece_map_crop(
        safe,
        split,
    )

    split_report = dict(split.get("report", {}))
    seed_supervoxels = split_report.get("seed_supervoxels")
    if (
        not isinstance(seed_supervoxels, list)
        or len(seed_supervoxels) != 2
    ):
        raise RuntimeError(
            "Stage-08 split report does not contain two seed_supervoxels."
        )

    return Stage08Artifacts(
        context=context,
        split=split,
        case_report=case_report,
        safe_supervoxels=safe,
        piece_map_crop=piece_map,
        seed_supervoxels=(
            int(seed_supervoxels[0]),
            int(seed_supervoxels[1]),
        ),
    )


# =============================================================================
# CANDIDATE-0 SPATIAL RAG WARM-UP
# =============================================================================


@dataclass
class RAGBundle:
    statistics: list[Any]
    template: RAGState
    targets: RAGTargets


class CachedRAGModel(nn.Module):
    """Production RAGBuilder projection + SpatialRAGNetwork over cached stats."""

    def __init__(
        self,
        builder: RAGBuilder,
        network: SpatialRAGNetwork,
        bundle: RAGBundle,
        dref_um: Tensor,
    ):
        super().__init__()
        self.builder = builder
        self.network = network
        self.bundle = bundle
        self.register_buffer(
            "_dref_um",
            dref_um.detach().clone(),
            persistent=False,
        )

    def current_rag(self) -> RAGState:
        template = self.bundle.template

        node_chunks: list[Tensor] = []
        centroid_chunks: list[Tensor] = []
        count_chunks: list[Tensor] = []

        for batch_index, statistics in enumerate(
            self.bundle.statistics
        ):
            start = int(
                template.node_offsets[batch_index].item()
            )
            stop = int(
                template.node_offsets[batch_index + 1].item()
            )
            reference = template.node_features[start:stop]

            nodes, centroids, counts = (
                self.builder._node_rows_from_statistics(
                    statistics,
                    reference,
                    self._dref_um[batch_index],
                )
            )
            node_chunks.append(nodes)
            centroid_chunks.append(centroids)
            count_chunks.append(counts)

        node_features = torch.cat(node_chunks, dim=0)
        hidden = self.network.cfg.rag_hidden_dim

        return replace(
            template,
            node_features=node_features,
            node_centroid_um=torch.cat(
                centroid_chunks,
                dim=0,
            ),
            node_volume_voxels=torch.cat(
                count_chunks,
                dim=0,
            ),
            node_embeddings=node_features.new_zeros(
                (node_features.shape[0], hidden)
            ),
            edge_embeddings=template.edge_features.new_zeros(
                (template.edge_features.shape[0], hidden)
            ),
            spatial_edge_logits=template.edge_features.new_zeros(
                (template.edge_features.shape[0],)
            ),
        )

    def forward(self) -> RAGState:
        return self.network(self.current_rag())


def grouping_metrics(
    partition: PartitionState,
    targets: RAGTargets,
    cfg,
) -> dict[str, float | int]:
    valid_nodes = (
        (targets.dominant_gt > 0)
        & (
            targets.node_purity
            >= cfg.partition.rag_min_node_purity
        )
        & (
            targets.node_gt_support
            >= cfg.partition.rag_min_node_gt_support
        )
    )
    rows = torch.nonzero(
        valid_nodes,
        as_tuple=False,
    ).flatten()

    if rows.numel() < 2:
        return {
            "valid_node_count": int(rows.numel()),
            "pair_count": 0,
            "pairwise_accuracy": 1.0,
            "false_merge_pairs": 0,
            "false_split_pairs": 0,
        }

    i, j = torch.triu_indices(
        rows.numel(),
        rows.numel(),
        offset=1,
        device=rows.device,
    )
    left = rows[i]
    right = rows[j]
    expected = (
        targets.dominant_gt[left]
        == targets.dominant_gt[right]
    )
    predicted = (
        partition.node_component[left]
        == partition.node_component[right]
    )
    return {
        "valid_node_count": int(rows.numel()),
        "pair_count": int(expected.numel()),
        "pairwise_accuracy": float(
            (expected == predicted).float().mean().item()
        ),
        "false_merge_pairs": int(
            (predicted & ~expected).sum().item()
        ),
        "false_split_pairs": int(
            (~predicted & expected).sum().item()
        ),
    }


@torch.no_grad()
def evaluate_spatial_rag(
    model: CachedRAGModel,
    criterion: RAGCriterion,
    partitioner: GraphPartitioner,
    targets: RAGTargets,
    gt_batched: Tensor,
    cfg,
) -> tuple[dict[str, float | int], RAGState, PartitionState]:
    was_training = model.training
    model.eval()

    rag = model()
    loss_metrics = criterion(
        rag,
        gt_batched,
        targets=targets,
    )
    probabilities = rag.spatial_edge_logits.sigmoid()
    valid = targets.valid
    positive = valid & (targets.target > 0.5)
    negative = valid & (targets.target <= 0.5)
    predicted = probabilities >= 0.5
    truth = targets.target > 0.5

    partition = partitioner(
        rag,
        rag.spatial_edge_logits,
        cfg.partition.spatial_merge_threshold,
    )
    grouping = grouping_metrics(
        partition,
        targets,
        cfg,
    )

    metrics = {
        "rag_bce": float(
            loss_metrics["rag_bce"].item()
        ),
        "edge_accuracy": float(
            (
                predicted[valid]
                == truth[valid]
            )
            .float()
            .mean()
            .item()
        ),
        "min_merge_probability": (
            float(probabilities[positive].min().item())
            if positive.any()
            else 1.0
        ),
        "max_separate_probability": (
            float(probabilities[negative].max().item())
            if negative.any()
            else 0.0
        ),
        "valid_edge_count": int(valid.sum().item()),
        "invalid_edge_count": int((~valid).sum().item()),
        **{
            f"partition_{key}": value
            for key, value in grouping.items()
        },
    }

    if was_training:
        model.train()
    return metrics, rag, partition


def spatial_rag_passed(
    metrics: dict[str, float | int],
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    checks = (
        (
            float(metrics["edge_accuracy"])
            >= PASS["rag_warmup_edge_accuracy_min"],
            f"edge_accuracy={metrics['edge_accuracy']:.4f}",
        ),
        (
            float(metrics["min_merge_probability"])
            >= PASS["rag_warmup_min_merge_probability_min"],
            (
                "min_merge_probability="
                f"{metrics['min_merge_probability']:.4f}"
            ),
        ),
        (
            float(metrics["max_separate_probability"])
            <= PASS["rag_warmup_max_separate_probability_max"],
            (
                "max_separate_probability="
                f"{metrics['max_separate_probability']:.4f}"
            ),
        ),
        (
            float(metrics["partition_pairwise_accuracy"])
            >= PASS["rag_warmup_pairwise_accuracy_min"],
            (
                "partition_pairwise_accuracy="
                f"{metrics['partition_pairwise_accuracy']:.4f}"
            ),
        ),
        (
            int(metrics["partition_false_merge_pairs"]) == 0,
            (
                "partition_false_merge_pairs="
                f"{metrics['partition_false_merge_pairs']}"
            ),
        ),
        (
            int(metrics["partition_false_split_pairs"]) == 0,
            (
                "partition_false_split_pairs="
                f"{metrics['partition_false_split_pairs']}"
            ),
        ),
    )

    for passed, message in checks:
        if not passed:
            failures.append(message)
    return not failures, failures


def warmup_spatial_rag(
    labels: Tensor,
    geometry: GeometryState,
    derived_cache,
    decoded,
    batch,
    cfg,
    *,
    steps: int,
    lr: float,
    eval_every: int,
) -> tuple[
    RAGState,
    RAGTargets,
    dict[str, Any],
    dict[str, Tensor],
    dict[str, Tensor],
]:
    print(
        "\n[setup 2/7] Building and memorizing candidate-0 spatial RAG ...",
        flush=True,
    )

    builder = RAGBuilder(
        cfg.partition,
        cfg.spatial,
    ).to(labels.device)
    network = SpatialRAGNetwork(
        cfg.partition,
        builder.node_feature_dim,
        builder.edge_feature_dim,
    ).to(labels.device)
    criterion = RAGCriterion(
        cfg.partition,
    ).to(labels.device)
    partitioner = GraphPartitioner().to(labels.device)

    with torch.no_grad():
        statistics = build_supervoxel_statistics(
            [labels],
            batch.spatial_inputs,
            geometry,
            batch.spacing_um,
            (
                decoded.d0,
                decoded.d1,
                decoded.d2,
            ),
            derived=derived_cache,
        )
        template = builder(
            [labels],
            decoded.d0,
            batch.spatial_inputs,
            geometry,
            batch.spacing_um,
            batch.dref_um,
            statistics_by_batch=statistics,
            derived_cache=derived_cache,
        )
        targets = criterion.build_targets(
            template,
            batch.gt_labels.unsqueeze(0).to(labels.device),
        )

    valid = targets.valid
    positive = valid & (targets.target > 0.5)
    negative = valid & (targets.target <= 0.5)

    print(
        f"[RAG] nodes={template.node_features.shape[0]} "
        f"edges={template.edge_index.shape[1]} "
        f"validMerge={int(positive.sum())} "
        f"validSeparate={int(negative.sum())} "
        f"invalid={int((~valid).sum())}"
    )
    if not positive.any():
        raise RuntimeError(
            "Candidate-0 RAG has no valid MERGE edge; "
            "cannot prepare the temporal correction."
        )
    if not negative.any():
        raise RuntimeError(
            "Candidate-0 RAG has no valid SEPARATE edge; "
            "the spatial warm-up is not a useful two-class check."
        )

    bundle = RAGBundle(
        statistics=statistics,
        template=template,
        targets=targets,
    )
    model = CachedRAGModel(
        builder,
        network,
        bundle,
        batch.dref_um,
    ).to(labels.device)

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=0.0,
    )

    history: list[dict[str, Any]] = []
    pass_streak = 0
    stopped = False

    iterator = (
        tqdm_trange(
            1,
            steps + 1,
            desc="stirnet:stage09-rag",
            unit="step",
            dynamic_ncols=True,
        )
        if tqdm_trange is not None
        else range(1, steps + 1)
    )
    has_tqdm = tqdm_trange is not None

    for step in iterator:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rag = model()

        loss = criterion(
            rag,
            batch.gt_labels.unsqueeze(0).to(labels.device),
            targets=targets,
        )["rag_bce"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite spatial RAG warm-up loss at step {step}."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters,
            GRAD_CLIP_NORM,
        )
        optimizer.step()

        if has_tqdm:
            iterator.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                refresh=(step % 5 == 0),
            )

        if (
            step == 1
            or step % eval_every == 0
            or step == steps
        ):
            metrics, _, _ = evaluate_spatial_rag(
                model,
                criterion,
                partitioner,
                targets,
                batch.gt_labels.unsqueeze(0),
                cfg,
            )
            passed, failures = spatial_rag_passed(
                metrics
            )
            pass_streak = (
                pass_streak + 1 if passed else 0
            )
            history.append(
                {
                    "step": step,
                    "metrics": metrics,
                    "passed": passed,
                    "failures": failures,
                }
            )

            message = (
                f"[RAG eval] step={step:04d} "
                f"bce={metrics['rag_bce']:.5f} "
                f"edgeAcc={metrics['edge_accuracy']:.3f} "
                f"pMergeMin={metrics['min_merge_probability']:.3f} "
                f"pSepMax={metrics['max_separate_probability']:.3f} "
                f"groupAcc={metrics['partition_pairwise_accuracy']:.3f} "
                f"pass={passed} streak={pass_streak}/2"
            )
            if has_tqdm:
                iterator.write(message)
            else:
                print(message)

            if passed and pass_streak >= 2:
                stopped = True
                break

    metrics, rag, partition = evaluate_spatial_rag(
        model,
        criterion,
        partitioner,
        targets,
        batch.gt_labels.unsqueeze(0).to(labels.device),
        cfg,
    )
    passed, failures = spatial_rag_passed(metrics)
    if not passed:
        raise RuntimeError(
            "Candidate-0 spatial RAG did not reach a GT-perfect frozen "
            f"baseline. Failures: {failures}"
        )

    builder_state = state_dict_cpu(builder)
    network_state = state_dict_cpu(network)
    rag = detach_rag(rag)

    summary = {
        "passed": True,
        "stopped_early": stopped,
        "metrics": metrics,
        "history": history,
    }

    del model, optimizer, parameters
    return (
        rag,
        targets,
        summary,
        builder_state,
        network_state,
    )


# =============================================================================
# CONTROLLED RAG CUT
# =============================================================================


@dataclass
class ControlledCut:
    rag: RAGState
    partition: PartitionState
    cut_edge_mask: Tensor
    cut_edge_rows: list[int]
    node_piece: Tensor
    target_node_rows: Tensor
    spatial_metrics: dict[str, Any]


def map_rag_nodes_to_controlled_pieces(
    rag: RAGState,
    targets: RAGTargets,
    safe_supervoxels: Tensor,
    piece_map_crop: Tensor,
    gt_labels: Tensor,
    *,
    target_gt_id: int,
) -> tuple[Tensor, list[dict[str, Any]]]:
    safe = safe_supervoxels.to(
        device=rag.node_supervoxel_id.device,
    )
    piece_map = piece_map_crop.to(
        device=rag.node_supervoxel_id.device,
    )
    gt = gt_labels.to(
        device=rag.node_supervoxel_id.device,
    )

    node_piece = torch.zeros(
        (rag.node_features.shape[0],),
        device=rag.node_features.device,
        dtype=torch.long,
    )
    rows: list[dict[str, Any]] = []

    for node_row in range(
        rag.node_features.shape[0]
    ):
        if int(targets.dominant_gt[node_row]) != target_gt_id:
            continue

        supervoxel_id = int(
            rag.node_supervoxel_id[node_row].item()
        )
        mask = (
            (safe == supervoxel_id)
            & (gt == target_gt_id)
        )
        piece_values = piece_map[mask]
        count_a = int(
            (piece_values == 1).sum().item()
        )
        count_b = int(
            (piece_values == 2).sum().item()
        )
        covered = count_a + count_b

        if covered == 0:
            rows.append(
                {
                    "node_row": node_row,
                    "supervoxel_id": supervoxel_id,
                    "piece": 0,
                    "piece_purity": 0.0,
                    "covered_voxels": 0,
                }
            )
            continue

        piece = 1 if count_a >= count_b else 2
        purity = max(count_a, count_b) / covered

        node_piece[node_row] = piece
        rows.append(
            {
                "node_row": node_row,
                "supervoxel_id": supervoxel_id,
                "piece": piece,
                "piece_purity": float(purity),
                "covered_voxels": covered,
                "piece_a_voxels": count_a,
                "piece_b_voxels": count_b,
            }
        )

    return node_piece, rows


def _expected_controlled_pairwise(
    left: Tensor,
    right: Tensor,
    targets: RAGTargets,
    node_piece: Tensor,
    target_gt_id: int,
) -> Tensor:
    same_gt = (
        targets.dominant_gt[left]
        == targets.dominant_gt[right]
    )
    target_pair = (
        (targets.dominant_gt[left] == target_gt_id)
        & (targets.dominant_gt[right] == target_gt_id)
    )
    same_piece = (
        (node_piece[left] > 0)
        & (node_piece[right] > 0)
        & (node_piece[left] == node_piece[right])
    )
    return torch.where(
        target_pair,
        same_piece,
        same_gt,
    )


def controlled_grouping_metrics(
    partition: PartitionState,
    targets: RAGTargets,
    node_piece: Tensor,
    cfg,
    *,
    target_gt_id: int,
) -> dict[str, float | int]:
    valid_nodes = (
        (targets.dominant_gt > 0)
        & (
            targets.node_purity
            >= cfg.partition.rag_min_node_purity
        )
        & (
            targets.node_gt_support
            >= cfg.partition.rag_min_node_gt_support
        )
    )
    rows = torch.nonzero(
        valid_nodes,
        as_tuple=False,
    ).flatten()
    if rows.numel() < 2:
        return {
            "pairwise_accuracy": 1.0,
            "false_merge_pairs": 0,
            "false_split_pairs": 0,
        }

    i, j = torch.triu_indices(
        rows.numel(),
        rows.numel(),
        offset=1,
        device=rows.device,
    )
    left = rows[i]
    right = rows[j]
    expected = _expected_controlled_pairwise(
        left,
        right,
        targets,
        node_piece,
        target_gt_id,
    )
    predicted = (
        partition.node_component[left]
        == partition.node_component[right]
    )
    return {
        "pairwise_accuracy": float(
            (expected == predicted)
            .float()
            .mean()
            .item()
        ),
        "false_merge_pairs": int(
            (predicted & ~expected).sum().item()
        ),
        "false_split_pairs": int(
            (~predicted & expected).sum().item()
        ),
    }


def make_controlled_rag_cut(
    rag: RAGState,
    targets: RAGTargets,
    safe_supervoxels: Tensor,
    piece_map_crop: Tensor,
    gt_labels: Tensor,
    cfg,
    partitioner: GraphPartitioner,
) -> ControlledCut:
    node_piece, mapping_rows = (
        map_rag_nodes_to_controlled_pieces(
            rag,
            targets,
            safe_supervoxels,
            piece_map_crop,
            gt_labels,
            target_gt_id=EXPECTED_TARGET_GT_ID,
        )
    )

    target_nodes = torch.nonzero(
        targets.dominant_gt
        == EXPECTED_TARGET_GT_ID,
        as_tuple=False,
    ).flatten()

    print("\nControlled target supervoxel mapping")
    print(
        f"{'node':>5s} {'SV':>5s} {'piece':>6s} "
        f"{'purity':>8s} {'covered':>8s}"
    )
    print("-" * 44)
    for row in mapping_rows:
        print(
            f"{row['node_row']:5d} "
            f"{row['supervoxel_id']:5d} "
            f"{row['piece']:6d} "
            f"{row['piece_purity']:8.3f} "
            f"{row['covered_voxels']:8d}"
        )

    if target_nodes.numel() < 2:
        raise RuntimeError(
            "GT 6 is represented by fewer than two RAG nodes."
        )

    target_pieces = set(
        int(value)
        for value in node_piece[target_nodes].tolist()
        if int(value) > 0
    )
    if target_pieces != {1, 2}:
        raise RuntimeError(
            "The Stage-08 controlled split is not expressible as two "
            f"target RAG sides. Observed pieces={sorted(target_pieces)}."
        )

    for row in mapping_rows:
        if (
            row["piece"] > 0
            and row["piece_purity"] < 0.90
        ):
            raise RuntimeError(
                "A target safe supervoxel crosses the controlled A/B split "
                f"too strongly: SV {row['supervoxel_id']} has piece purity "
                f"{row['piece_purity']:.3f}. The controlled error is not "
                "cleanly expressible at RAG level."
            )

    src, dst = rag.edge_index
    cross = (
        (targets.dominant_gt[src] == EXPECTED_TARGET_GT_ID)
        & (targets.dominant_gt[dst] == EXPECTED_TARGET_GT_ID)
        & (node_piece[src] > 0)
        & (node_piece[dst] > 0)
        & (node_piece[src] != node_piece[dst])
    )

    if not cross.any():
        raise RuntimeError(
            "No RAG adjacency edge crosses controlled pieces A/B."
        )

    if not bool(
        torch.all(
            targets.valid[cross]
            & (targets.target[cross] > 0.5)
        )
    ):
        raise RuntimeError(
            "At least one controlled A/B cut edge is not a valid GT MERGE "
            "edge. Do not train temporal reasoning on an ambiguous edge."
        )

    cut_rows = torch.nonzero(
        cross,
        as_tuple=False,
    ).flatten()
    logits = rag.spatial_edge_logits.detach().clone()
    logits[cross] = FORCED_SPATIAL_LOGIT

    controlled_rag = replace(
        rag,
        spatial_edge_logits=logits,
    )
    partition = partitioner(
        controlled_rag,
        logits,
        cfg.partition.spatial_merge_threshold,
    )

    target_components = torch.unique(
        partition.node_component[target_nodes]
    )
    controlled_group = controlled_grouping_metrics(
        partition,
        targets,
        node_piece,
        cfg,
        target_gt_id=EXPECTED_TARGET_GT_ID,
    )

    forced_probability = float(
        torch.sigmoid(
            torch.tensor(FORCED_SPATIAL_LOGIT)
        ).item()
    )

    print("\nControlled spatial cut")
    print(
        f"  cut edge rows               : "
        f"{[int(v) for v in cut_rows.tolist()]}"
    )
    print(
        f"  forced spatial logit        : "
        f"{FORCED_SPATIAL_LOGIT:.3f}"
    )
    print(
        f"  forced spatial P(merge)     : "
        f"{forced_probability:.6f}"
    )
    print(
        f"  target spatial components   : "
        f"{int(target_components.numel())}"
    )
    print(
        f"  controlled pairwise accuracy: "
        f"{controlled_group['pairwise_accuracy']:.4f}"
    )

    if (
        forced_probability
        > PASS["forced_spatial_merge_probability_max"]
    ):
        raise RuntimeError(
            "Forced spatial logit is not sufficiently strong."
        )
    if target_components.numel() != 2:
        raise RuntimeError(
            "Controlled RAG cut does not produce exactly TWO target "
            f"components; got {int(target_components.numel())}."
        )
    if (
        controlled_group["pairwise_accuracy"] < 1.0
        or controlled_group["false_merge_pairs"] != 0
        or controlled_group["false_split_pairs"] != 0
    ):
        raise RuntimeError(
            "The controlled spatial partition introduces errors beyond the "
            "intended GT-6 A/B split."
        )

    return ControlledCut(
        rag=controlled_rag,
        partition=partition,
        cut_edge_mask=cross,
        cut_edge_rows=[
            int(v) for v in cut_rows.tolist()
        ],
        node_piece=node_piece,
        target_node_rows=target_nodes,
        spatial_metrics={
            "forced_spatial_logit": FORCED_SPATIAL_LOGIT,
            "forced_spatial_merge_probability": forced_probability,
            "target_spatial_component_count": int(
                target_components.numel()
            ),
            **{
                f"controlled_{key}": value
                for key, value in controlled_group.items()
            },
        },
    )


# =============================================================================
# TEMPORAL INPUT REBASE
# =============================================================================


@dataclass
class TemporalBundle:
    input: TemporalInput
    node_instance_grid: Tensor
    node_history_valid: Tensor
    selected_original_tracklet_ids: Tensor
    target_local_tracklets: list[int]
    observer_tracklet_mask: Tensor
    diagnostics: dict[str, Any]


def rebase_temporal_context_to_crop(
    context: dict[str, Any],
    split_artifact: dict[str, Any],
    *,
    spacing_um: Tensor,
    dref_um: Tensor,
    crop_shape_zyx: tuple[int, int, int],
    device: torch.device,
) -> TemporalBundle:
    target = dict(context["target"])

    roi_low = torch.tensor(
        target["temporal_roi_low_zyx"],
        dtype=torch.float32,
        device=device,
    )
    roi_center_um = torch.tensor(
        target["temporal_roi_center_um_zyx"],
        dtype=torch.float32,
        device=device,
    )

    crop_pairs = split_artifact.get("crop_slices_zyx")
    if crop_pairs is None:
        raise RuntimeError(
            "controlled_split.pt is missing crop_slices_zyx."
        )
    crop_low = torch.tensor(
        [float(pair[0]) for pair in crop_pairs],
        dtype=torch.float32,
        device=device,
    )
    crop_center_um = (
        0.5
        * (
            torch.tensor(
                crop_shape_zyx,
                device=device,
                dtype=torch.float32,
            )
            - 1.0
        )
        * spacing_um.float()
    )

    # Convert a position p from all-cell-ROI-centred microns to crop-centred
    # microns:
    #
    # full_um = roi_low*spacing + roi_center_um + p
    # crop_um = full_um - crop_low*spacing - crop_center_um
    translation_um = (
        roi_low * spacing_um.float()
        + roi_center_um
        - crop_low * spacing_um.float()
        - crop_center_um
    )

    graph_x = (
        torch.as_tensor(context["graph_x"])
        .float()
        .to(device)
        .clone()
    )
    if graph_x.ndim != 2 or graph_x.shape[1] != 32:
        raise RuntimeError(
            f"Stage-08 graph_x must have shape [N,32]; got "
            f"{tuple(graph_x.shape)}."
        )
    graph_x[:, 1:4] += (
        translation_um[None]
        / dref_um.float().clamp_min(1e-6)
    )

    temporal_ref = (
        torch.as_tensor(context["temporal_ref_um"])
        .float()
        .to(device)
        + translation_um[None]
    )

    data = TemporalInput(
        graph_x=graph_x,
        graph_edge_index=(
            torch.as_tensor(context["graph_edge_index"])
            .long()
            .to(device)
        ),
        graph_edge_attr=(
            torch.as_tensor(context["graph_edge_attr"])
            .float()
            .to(device)
        ),
        hypothesis_edge_index=(
            torch.as_tensor(
                context["hypothesis_edge_index"]
            )
            .long()
            .to(device)
        ),
        hypothesis_edge_attr=(
            torch.as_tensor(
                context["hypothesis_edge_attr"]
            )
            .float()
            .to(device)
        ),
        tracklet_id=(
            torch.as_tensor(context["tracklet_id"])
            .long()
            .to(device)
        ),
        temporal_ref_um=temporal_ref,
        temporal_status=(
            torch.as_tensor(context["temporal_status"])
            .float()
            .to(device)
        ),
        temporal_batch=(
            torch.as_tensor(context["temporal_batch"])
            .long()
            .to(device)
        ),
    )

    grids = (
        torch.as_tensor(context["node_instance_grid"])
        .to(device)
    )
    valid = (
        torch.as_tensor(context["node_history_valid"])
        .bool()
        .to(device)
    )

    original_tracklets = (
        torch.as_tensor(
            context["selected_original_tracklet_ids"]
        )
        .long()
        .to(device)
    )

    # Map Stage-08 piece-associated ORIGINAL full-cache tracklet IDs into the
    # local ego-graph indices consumed by TemporalGraphEncoder.
    report = dict(context.get("report", {}))
    controlled = dict(
        report.get("controlled_error", {})
    )
    piece_tracklets = dict(
        controlled.get("piece_tracklets", {})
    )
    requested_original: set[int] = set()
    for rows in piece_tracklets.values():
        for row in rows:
            requested_original.add(
                int(row["tracklet_id"])
            )

    target_local_tracklets: list[int] = []
    for local_index, original_id in enumerate(
        original_tracklets.tolist()
    ):
        if int(original_id) in requested_original:
            target_local_tracklets.append(local_index)

    half_extent_um = (
        0.5
        * (
            torch.tensor(
                crop_shape_zyx,
                device=device,
                dtype=torch.float32,
            )
            - 1.0
        )
        * spacing_um.float()
    )
    observer_mask = (
        temporal_ref.abs()
        <= half_extent_um[None]
    ).all(dim=-1)

    if not target_local_tracklets:
        raise RuntimeError(
            "None of the Stage-08 target-piece tracklets survived into the "
            "controlled ego graph."
        )

    target_full_vox = torch.tensor(
        target["target_centroid_vox_zyx"],
        dtype=torch.float32,
        device=device,
    )
    target_crop_um = (
        target_full_vox
        - crop_low
        - 0.5
        * (
            torch.tensor(
                crop_shape_zyx,
                dtype=torch.float32,
                device=device,
            )
            - 1.0
        )
    ) * spacing_um.float()

    target_distances = []
    for local_index in target_local_tracklets:
        target_distances.append(
            float(
                torch.linalg.vector_norm(
                    temporal_ref[local_index]
                    - target_crop_um
                )
                .div(
                    dref_um.float().clamp_min(1e-6)
                )
                .item()
            )
        )

    target_observed = [
        bool(observer_mask[index].item())
        for index in target_local_tracklets
    ]
    if not any(target_observed):
        raise RuntimeError(
            "No target-piece temporal tracklet lies inside the candidate-0 "
            "spatial crop after coordinate rebasing."
        )

    diagnostics = {
        "translation_um_zyx": [
            float(v)
            for v in translation_um.detach().cpu().tolist()
        ],
        "tracklet_count": int(
            temporal_ref.shape[0]
        ),
        "detection_node_count": int(
            graph_x.shape[0]
        ),
        "detection_edge_count": int(
            data.graph_edge_index.shape[1]
        ),
        "hypothesis_edge_count": int(
            data.hypothesis_edge_index.shape[1]
        ),
        "observer_in_crop_tracklet_count": int(
            observer_mask.sum().item()
        ),
        "target_local_tracklets": [
            int(v) for v in target_local_tracklets
        ],
        "target_original_tracklets": [
            int(original_tracklets[v].item())
            for v in target_local_tracklets
        ],
        "target_gt_ref_um_zyx": [
            float(v)
            for v in target_crop_um.detach().cpu().tolist()
        ],
        "target_tracklet_distance_to_target_dref": (
            target_distances
        ),
        "target_tracklet_observer_in_crop": (
            target_observed
        ),
    }

    return TemporalBundle(
        input=data,
        node_instance_grid=grids,
        node_history_valid=valid,
        selected_original_tracklet_ids=original_tracklets,
        target_local_tracklets=target_local_tracklets,
        observer_tracklet_mask=observer_mask,
        diagnostics=diagnostics,
    )


# =============================================================================
# TEMPORAL OVERFIT MODEL
# =============================================================================


def _subset_temporal_state(
    state: TemporalState,
    indices: Tensor,
) -> TemporalState:
    return TemporalState(
        tokens=state.tokens[indices],
        ref_um=state.ref_um[indices],
        batch_index=state.batch_index[indices],
        salience=state.salience[indices],
        reliability=state.reliability[indices],
        status=state.status[indices],
        node_tokens=None,
    )


def _merge_observed_tokens(
    base: TemporalState,
    indices: Tensor,
    observed: TemporalState,
) -> TemporalState:
    tokens = base.tokens.index_copy(
        0,
        indices,
        observed.tokens,
    )
    return replace(
        base,
        tokens=tokens,
    )


class TemporalOverfitModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.instance_tokenizer = InstanceTokenizer(
            cfg.instances,
            cfg.spatial,
        )
        self.history_encoder = HistoricalInstanceEncoder(
            cfg.history,
            cfg.temporal,
        )
        self.temporal_encoder = TemporalGraphEncoder(
            cfg.temporal,
        )
        self.temporal_observer = TemporalSpatialObserver(
            cfg.temporal,
            cfg.spatial,
            cfg.geometry,
        )
        self.instance_temporal = InstanceTemporalReasoner(
            cfg.temporal,
            cfg.instances,
            cfg.partition,
            cfg.refinement,
        )

    def encode_temporal(
        self,
        temporal_bundle: TemporalBundle,
        decoded,
        geometry,
        pyramid,
        spacing_um: Tensor,
        dref_um: Tensor,
        *,
        use_history: bool = True,
        use_hypothesis: bool = True,
        use_observer: bool = True,
    ) -> TemporalState:
        data = temporal_bundle.input

        if not use_hypothesis:
            data = replace(
                data,
                hypothesis_edge_index=torch.zeros(
                    (2, 0),
                    device=data.graph_x.device,
                    dtype=torch.long,
                ),
                hypothesis_edge_attr=torch.zeros(
                    (
                        0,
                        data.hypothesis_edge_attr.shape[-1],
                    ),
                    device=data.graph_x.device,
                    dtype=data.graph_x.dtype,
                ),
            )

        if use_history:
            history = self.history_encoder(
                temporal_bundle.node_instance_grid,
                temporal_bundle.node_history_valid,
            )
            data = replace(
                data,
                node_history_embedding=history,
            )
        else:
            data = replace(
                data,
                node_history_embedding=None,
            )

        temporal = self.temporal_encoder(data)

        if not use_observer or temporal.is_empty:
            return temporal

        indices = torch.nonzero(
            temporal_bundle.observer_tracklet_mask,
            as_tuple=False,
        ).flatten()
        if indices.numel() == 0:
            return temporal

        subset = _subset_temporal_state(
            temporal,
            indices,
        )
        observed = self.temporal_observer(
            subset,
            decoded,
            geometry,
            pyramid.spacings_um,
            spacing_um,
            dref_um,
        )
        return _merge_observed_tokens(
            temporal,
            indices,
            observed,
        )

    def forward(
        self,
        partition: PartitionState,
        rag: RAGState,
        decoded,
        geometry,
        pyramid,
        spacing_um: Tensor,
        dref_um: Tensor,
        temporal_bundle: TemporalBundle,
        *,
        use_history: bool = True,
        use_hypothesis: bool = True,
        use_observer: bool = True,
        zero_temporal: bool = False,
    ) -> tuple[
        InstanceState,
        TemporalState,
        ReasoningState,
    ]:
        instances = self.instance_tokenizer(
            partition,
            rag,
            decoded,
            geometry,
            spacing_um,
            dref_um,
        )

        if zero_temporal:
            temporal = self.temporal_encoder.empty(
                instances.tokens.device,
                instances.tokens.dtype,
            )
        else:
            temporal = self.encode_temporal(
                temporal_bundle,
                decoded,
                geometry,
                pyramid,
                spacing_um,
                dref_um,
                use_history=use_history,
                use_hypothesis=use_hypothesis,
                use_observer=use_observer,
            )

        reasoning = self.instance_temporal(
            instances,
            rag,
            temporal,
            dref_um,
        )
        return instances, temporal, reasoning


# =============================================================================
# TEMPORAL LOSS / METRICS
# =============================================================================


@dataclass
class TemporalTargets:
    rag: RAGTargets
    existence: Tensor
    split: Tensor


def build_temporal_targets(
    cut: ControlledCut,
    rag_targets: RAGTargets,
    gt_batched: Tensor,
    loss_cfg: LossConfig,
    device: torch.device,
) -> TemporalTargets:
    instance_targets = build_instance_targets(
        cut.partition.labels,
        gt_batched,
        existence_min_precision=(
            loss_cfg.existence_min_precision
        ),
        existence_min_gt_coverage=(
            loss_cfg.existence_min_gt_coverage
        ),
        split_min_pred_fraction=(
            loss_cfg.split_min_pred_fraction
        ),
        split_min_gt_coverage=(
            loss_cfg.split_min_gt_coverage
        ),
        device=device,
    )
    return TemporalTargets(
        rag=rag_targets,
        existence=instance_targets.existence,
        split=instance_targets.split,
    )


def temporal_loss(
    reasoning: ReasoningState,
    cut: ControlledCut,
    targets: TemporalTargets,
) -> tuple[Tensor, dict[str, float]]:
    final_logits = reasoning.final_edge_logits

    target_mask = cut.cut_edge_mask
    other_valid = (
        targets.rag.valid
        & ~target_mask
    )

    target_loss = F.binary_cross_entropy_with_logits(
        final_logits[target_mask],
        targets.rag.target[target_mask].float(),
    )

    other_loss = (
        F.binary_cross_entropy_with_logits(
            final_logits[other_valid],
            targets.rag.target[other_valid].float(),
        )
        if other_valid.any()
        else final_logits.sum() * 0
    )

    existence_loss = (
        F.binary_cross_entropy_with_logits(
            reasoning.instance_exist_logits,
            targets.existence,
        )
        if targets.existence.numel()
        else final_logits.sum() * 0
    )
    split_loss = (
        F.binary_cross_entropy_with_logits(
            reasoning.split_logits,
            targets.split,
        )
        if targets.split.numel()
        else final_logits.sum() * 0
    )

    total = (
        LOSS_TARGET_EDGE_WEIGHT * target_loss
        + LOSS_OTHER_EDGE_WEIGHT * other_loss
        + LOSS_EXISTENCE_WEIGHT * existence_loss
        + LOSS_SPLIT_WEIGHT * split_loss
    )

    components = {
        "target_edge_bce": float(
            target_loss.detach().item()
        ),
        "other_edge_bce": float(
            other_loss.detach().item()
        ),
        "existence_bce": float(
            existence_loss.detach().item()
        ),
        "split_bce": float(
            split_loss.detach().item()
        ),
        "total": float(total.detach().item()),
    }
    return total, components


@torch.no_grad()
def target_component_count(
    partition: PartitionState,
    target_nodes: Tensor,
) -> int:
    if target_nodes.numel() == 0:
        return 0
    return int(
        torch.unique(
            partition.node_component[target_nodes]
        ).numel()
    )


@torch.no_grad()
def evaluate_temporal(
    model: TemporalOverfitModel,
    cut: ControlledCut,
    targets: TemporalTargets,
    temporal_bundle: TemporalBundle,
    decoded,
    geometry,
    pyramid,
    spacing_um: Tensor,
    dref_um: Tensor,
    partitioner: GraphPartitioner,
    cfg,
    *,
    use_history: bool = True,
    use_hypothesis: bool = True,
    use_observer: bool = True,
    zero_temporal: bool = False,
) -> tuple[
    dict[str, float | int],
    InstanceState,
    TemporalState,
    ReasoningState,
    PartitionState,
]:
    was_training = model.training
    model.eval()

    instances, temporal, reasoning = model(
        cut.partition,
        cut.rag,
        decoded,
        geometry,
        pyramid,
        spacing_um,
        dref_um,
        temporal_bundle,
        use_history=use_history,
        use_hypothesis=use_hypothesis,
        use_observer=use_observer,
        zero_temporal=zero_temporal,
    )

    final_partition = partitioner(
        cut.rag,
        reasoning.final_edge_logits,
        cfg.partition.final_merge_threshold,
    )

    probabilities = reasoning.final_edge_logits.sigmoid()
    valid = targets.rag.valid
    truth = targets.rag.target > 0.5
    predicted = probabilities >= 0.5

    grouping = grouping_metrics(
        final_partition,
        targets.rag,
        cfg,
    )

    cut_probabilities = probabilities[
        cut.cut_edge_mask
    ]
    cut_gates = reasoning.edge_temporal_gate[
        cut.cut_edge_mask
    ]

    metrics: dict[str, float | int] = {
        "target_min_merge_probability": float(
            cut_probabilities.min().item()
        ),
        "target_mean_merge_probability": float(
            cut_probabilities.mean().item()
        ),
        "target_min_temporal_gate": float(
            cut_gates.min().item()
        ),
        "target_mean_temporal_gate": float(
            cut_gates.mean().item()
        ),
        "final_edge_accuracy": float(
            (
                predicted[valid]
                == truth[valid]
            )
            .float()
            .mean()
            .item()
        ),
        "target_final_component_count": (
            target_component_count(
                final_partition,
                cut.target_node_rows,
            )
        ),
        "existence_accuracy": binary_accuracy(
            reasoning.instance_exist_logits,
            targets.existence,
        ),
        "split_accuracy": binary_accuracy(
            reasoning.split_logits,
            targets.split,
        ),
        "mean_instance_temporal_support": (
            float(
                reasoning.temporal_support.mean().item()
            )
            if reasoning.temporal_support.numel()
            else 0.0
        ),
        "max_instance_temporal_support": (
            float(
                reasoning.temporal_support.max().item()
            )
            if reasoning.temporal_support.numel()
            else 0.0
        ),
        **{
            f"partition_{key}": value
            for key, value in grouping.items()
        },
    }

    if zero_temporal:
        delta = (
            reasoning.final_edge_logits
            - cut.rag.spatial_edge_logits
        ).abs()
        metrics[
            "zero_temporal_max_logit_change"
        ] = (
            float(delta.max().item())
            if delta.numel()
            else 0.0
        )

    if was_training:
        model.train()

    return (
        metrics,
        instances,
        temporal,
        reasoning,
        final_partition,
    )


def strict_pass(
    full: dict[str, float | int],
    zero: dict[str, float | int],
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    checks = (
        (
            float(
                full["target_min_merge_probability"]
            )
            >= PASS["target_final_merge_probability_min"],
            (
                "target_min_merge_probability="
                f"{full['target_min_merge_probability']:.4f}"
            ),
        ),
        (
            float(full["final_edge_accuracy"])
            >= PASS["final_edge_accuracy_min"],
            (
                "final_edge_accuracy="
                f"{full['final_edge_accuracy']:.4f}"
            ),
        ),
        (
            float(
                full["partition_pairwise_accuracy"]
            )
            >= PASS["final_pairwise_accuracy_min"],
            (
                "partition_pairwise_accuracy="
                f"{full['partition_pairwise_accuracy']:.4f}"
            ),
        ),
        (
            int(
                full["partition_false_merge_pairs"]
            )
            == 0,
            (
                "partition_false_merge_pairs="
                f"{full['partition_false_merge_pairs']}"
            ),
        ),
        (
            int(
                full["partition_false_split_pairs"]
            )
            == 0,
            (
                "partition_false_split_pairs="
                f"{full['partition_false_split_pairs']}"
            ),
        ),
        (
            int(
                full["target_final_component_count"]
            )
            == 1,
            (
                "target_final_component_count="
                f"{full['target_final_component_count']}"
            ),
        ),
        (
            float(full["existence_accuracy"])
            >= PASS["existence_accuracy_min"],
            (
                "existence_accuracy="
                f"{full['existence_accuracy']:.4f}"
            ),
        ),
        (
            float(full["split_accuracy"])
            >= PASS["split_accuracy_min"],
            (
                "split_accuracy="
                f"{full['split_accuracy']:.4f}"
            ),
        ),
        (
            float(
                zero["zero_temporal_max_logit_change"]
            )
            <= PASS["zero_temporal_max_logit_change_max"],
            (
                "zero_temporal_max_logit_change="
                f"{zero['zero_temporal_max_logit_change']:.8f}"
            ),
        ),
        (
            int(
                zero["target_final_component_count"]
            )
            == 2,
            (
                "zero_target_component_count="
                f"{zero['target_final_component_count']}"
            ),
        ),
    )

    for passed, message in checks:
        if not passed:
            failures.append(message)
    return not failures, failures


def print_temporal_metrics(
    step: int,
    full: dict[str, float | int],
    zero: dict[str, float | int],
    *,
    loss: dict[str, float] | None = None,
) -> None:
    prefix = (
        f"step={step:04d} "
        + (
            f"loss={loss['total']:.5f} "
            if loss is not None
            else ""
        )
    )
    print(
        prefix
        + (
            f"targetP={full['target_min_merge_probability']:.3f} "
            f"gate={full['target_min_temporal_gate']:.3f} "
            f"edgeAcc={full['final_edge_accuracy']:.3f} "
            f"groupAcc={full['partition_pairwise_accuracy']:.3f} "
            f"targetComp={full['target_final_component_count']} "
            f"existAcc={full['existence_accuracy']:.3f} "
            f"splitAcc={full['split_accuracy']:.3f} "
            f"zeroΔ={zero['zero_temporal_max_logit_change']:.2e} "
            f"zeroComp={zero['target_final_component_count']}"
        )
    )


# =============================================================================
# TRAIN / SAVE
# =============================================================================


@dataclass
class Options:
    steps: int
    lr: float
    eval_every: int
    rag_warmup_steps: int
    rag_warmup_lr: float
    rag_eval_every: int
    seed: int
    sample: Path
    geometry_predictions: Path
    feature_donor_checkpoint: Path
    controlled_context: Path
    controlled_split: Path
    case_json: Path
    results: Path


def train(opts: Options) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 09 requires the local CUDA GPU."
        )

    device = torch.device("cuda")
    set_seed(opts.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = stage01.build_debug_config()
    batch = stage01.load_debug_batch(
        opts.sample,
        device,
    )
    if int(
        batch.selection.get("candidate_index", -1)
    ) != EXPECTED_CANDIDATE_INDEX:
        raise RuntimeError(
            "Stage-09 spatial sample must be candidate 0."
        )

    artifacts = load_stage08_artifacts(
        opts.controlled_context,
        opts.controlled_split,
        opts.case_json,
    )

    print("\n" + "=" * 118)
    print("STIR-Net Stage 09 — isolated temporal TWO->ONE overfit")
    print("=" * 118)
    print(
        f"GPU                              : "
        f"{torch.cuda.get_device_name(device)}"
    )
    print(
        f"Candidate / target               : "
        f"{EXPECTED_CANDIDATE_INDEX} / GT {EXPECTED_TARGET_GT_ID}"
    )
    print(
        f"Crop                             : "
        f"{tuple(batch.gt_labels.shape)}"
    )
    print(
        f"Stage-08 split seed SVs          : "
        f"{artifacts.seed_supervoxels}"
    )
    print(
        f"Controlled temporal artifact     : "
        f"{opts.controlled_context}"
    )
    print(
        "Trainable                        : "
        "InstanceTokenizer + history/temporal encoder + "
        "TemporalSpatialObserver + InstanceTemporalReasoner"
    )
    print(
        "Frozen                           : "
        "candidate-0 explicit geometry + donor spatial features + "
        "safe SV topology + trained spatial RAG"
    )
    print(
        f"Temporal LR                      : {opts.lr:.3g}"
    )
    print(
        f"Temporal max steps               : {opts.steps}"
    )
    print("=" * 118)

    setup_started = time.perf_counter()

    print(
        "\n[setup 1/7] Loading candidate-0 700-step geometry + running one frozen feature-donor pass ...",
        flush=True,
    )
    (
        geometry,
        pyramid,
        decoded,
        geometry_suitability,
        feature_donor_info,
    ) = prepare_candidate0_spatial_evidence(
        batch,
        opts.geometry_predictions,
        opts.feature_donor_checkpoint,
        cfg,
        device,
    )
    metrics = geometry_suitability["metrics"]
    print(
        f"[geometry] source                    : {opts.geometry_predictions}"
    )
    print(
        f"[geometry] formal Stage-01 PASS       : "
        f"{geometry_suitability['formal_stage01_passed']}"
    )
    print(
        f"[geometry] Stage-09 suitability       : PASS"
    )
    print(
        f"[geometry] fg/surface/separator Dice  : "
        f"{metrics['foreground_dice']:.4f} / "
        f"{metrics['surface_dice']:.4f} / "
        f"{metrics['separator_dice']:.4f}"
    )
    print(
        f"[geometry] SDF corr / flow cosine     : "
        f"{metrics['sdf_corr']:.4f} / {metrics['flow_cosine']:.4f}"
    )
    print(
        f"[geometry] centroid endpoint (diag)   : "
        f"{geometry_suitability['centroid_endpoint_median_um_diagnostic']:.4f} um"
    )
    print(
        f"[features] donor checkpoint           : {opts.feature_donor_checkpoint}"
    )
    print(
        f"[features] donor stage / candidate    : "
        f"{feature_donor_info['stage']} / "
        f"{feature_donor_info['source_candidate_index']}"
    )

    # Freeze all dense tensors. They are read-only evidence during Stage 09.
    for tensor in (
        geometry.foreground_logits,
        geometry.surface_logits,
        geometry.separator_logits,
        geometry.sdf,
        geometry.flow,
        geometry.centroid_offset,
        geometry.seed_logits,
        geometry.features,
        decoded.d0,
        decoded.d1,
        decoded.d2,
    ):
        if tensor is not None:
            tensor.requires_grad_(False)

    derived = build_geometry_derived_cache(
        geometry,
        cfg.partition,
        padding_mask=None,
    )

    safe_supervoxels = (
        artifacts.safe_supervoxels
        .to(device)
        .long()
    )
    if tuple(safe_supervoxels.shape) != tuple(
        batch.gt_labels.shape
    ):
        raise RuntimeError(
            "Stage-08 safe-supervoxel crop shape does not match Stage-09 "
            f"sample: {tuple(safe_supervoxels.shape)} vs "
            f"{tuple(batch.gt_labels.shape)}."
        )

    (
        learned_rag,
        rag_targets,
        rag_warmup_summary,
        rag_builder_state,
        rag_network_state,
    ) = warmup_spatial_rag(
        safe_supervoxels,
        geometry,
        derived,
        decoded,
        batch,
        cfg,
        steps=opts.rag_warmup_steps,
        lr=opts.rag_warmup_lr,
        eval_every=opts.rag_eval_every,
    )

    partitioner = GraphPartitioner().to(device)

    print(
        "\n[setup 3/7] Injecting the controlled same-GT RAG cut ...",
        flush=True,
    )
    cut = make_controlled_rag_cut(
        learned_rag,
        rag_targets,
        artifacts.safe_supervoxels,
        artifacts.piece_map_crop,
        batch.gt_labels,
        cfg,
        partitioner,
    )

    print(
        "\n[setup 4/7] Rebasing full-volume temporal context into candidate crop coordinates ...",
        flush=True,
    )
    temporal_bundle = rebase_temporal_context_to_crop(
        artifacts.context,
        artifacts.split,
        spacing_um=batch.spacing_um[0],
        dref_um=batch.dref_um[0],
        crop_shape_zyx=tuple(
            int(v) for v in batch.gt_labels.shape
        ),
        device=device,
    )
    diagnostics = temporal_bundle.diagnostics
    print(
        f"[temporal] translation zyx um     : "
        f"{diagnostics['translation_um_zyx']}"
    )
    print(
        f"[temporal] graph tracklets/nodes   : "
        f"{diagnostics['tracklet_count']} / "
        f"{diagnostics['detection_node_count']}"
    )
    print(
        f"[temporal] detection/hyp edges     : "
        f"{diagnostics['detection_edge_count']} / "
        f"{diagnostics['hypothesis_edge_count']}"
    )
    print(
        f"[temporal] observer in-crop tracks : "
        f"{diagnostics['observer_in_crop_tracklet_count']}"
    )
    print(
        f"[temporal] target original tracks  : "
        f"{diagnostics['target_original_tracklets']}"
    )
    print(
        f"[temporal] target distances dref    : "
        f"{[round(v, 3) for v in diagnostics['target_tracklet_distance_to_target_dref']]}"
    )
    print(
        f"[temporal] target observed locally  : "
        f"{diagnostics['target_tracklet_observer_in_crop']}"
    )

    print(
        "\n[setup 5/7] Building fixed object/RAG supervision ...",
        flush=True,
    )
    loss_cfg = LossConfig()
    gt_batched = batch.gt_labels.unsqueeze(0).to(device)
    temporal_targets = build_temporal_targets(
        cut,
        rag_targets,
        gt_batched,
        loss_cfg,
        device,
    )
    print(
        f"[targets] provisional instances     : "
        f"{temporal_targets.existence.numel()}"
    )
    print(
        f"[targets] existence positives       : "
        f"{int((temporal_targets.existence > 0.5).sum())}"
    )
    print(
        f"[targets] split positives           : "
        f"{int((temporal_targets.split > 0.5).sum())}"
    )
    print(
        f"[targets] controlled MERGE edges    : "
        f"{cut.cut_edge_rows}"
    )

    print(
        "\n[setup 6/7] Initializing fresh tokenizer + temporal modules ...",
        flush=True,
    )
    model = TemporalOverfitModel(cfg).to(device)
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel()
        for parameter in parameters
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=opts.lr,
        weight_decay=0.0,
    )
    print(
        f"[setup] Trainable temporal parameters: "
        f"{trainable_count:,}"
    )

    print(
        "\n[setup 7/7] Evaluating fresh random temporal model + zero-context invariant ...",
        flush=True,
    )
    initial_full, _, _, _, _ = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
    )
    initial_zero, _, _, _, _ = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
        zero_temporal=True,
    )
    print_temporal_metrics(
        0,
        initial_full,
        initial_zero,
    )

    if (
        float(
            initial_zero[
                "zero_temporal_max_logit_change"
            ]
        )
        != 0.0
    ):
        raise RuntimeError(
            "Production zero-temporal invariant failed before training."
        )
    if int(
        initial_zero["target_final_component_count"]
    ) != 2:
        raise RuntimeError(
            "Zero temporal context unexpectedly repaired the controlled split "
            "before training."
        )

    setup_seconds = (
        time.perf_counter() - setup_started
    )
    print(
        f"[setup] complete in {duration(setup_seconds)}"
    )

    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "full": initial_full,
            "zero": initial_zero,
        }
    ]
    pass_streak = 0
    stopped_early = False
    train_started = time.perf_counter()
    latest_loss_components: dict[str, float] | None = None

    iterator = (
        tqdm_trange(
            1,
            opts.steps + 1,
            desc="stirnet:temporal",
            unit="step",
            dynamic_ncols=True,
        )
        if tqdm_trange is not None
        else range(1, opts.steps + 1)
    )
    has_tqdm = tqdm_trange is not None

    for step in iterator:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        _, _, reasoning = model(
            cut.partition,
            cut.rag,
            decoded,
            geometry,
            pyramid,
            batch.spacing_um,
            batch.dref_um,
            temporal_bundle,
        )

        loss, components = temporal_loss(
            reasoning,
            cut,
            temporal_targets,
        )
        latest_loss_components = components

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite Stage-09 loss at step {step}."
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            GRAD_CLIP_NORM,
        )
        optimizer.step()

        if has_tqdm:
            allocated, _ = gpu_peak_gib()
            iterator.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                mem=f"{allocated:.2f}G",
                refresh=(step % 5 == 0),
            )

        if (
            step == 1
            or step % opts.eval_every == 0
            or step == opts.steps
        ):
            full, _, _, _, _ = evaluate_temporal(
                model,
                cut,
                temporal_targets,
                temporal_bundle,
                decoded,
                geometry,
                pyramid,
                batch.spacing_um,
                batch.dref_um,
                partitioner,
                cfg,
            )
            zero, _, _, _, _ = evaluate_temporal(
                model,
                cut,
                temporal_targets,
                temporal_bundle,
                decoded,
                geometry,
                pyramid,
                batch.spacing_um,
                batch.dref_um,
                partitioner,
                cfg,
                zero_temporal=True,
            )

            passed_now, failures = strict_pass(
                full,
                zero,
            )
            pass_streak = (
                pass_streak + 1
                if passed_now
                else 0
            )

            history.append(
                {
                    "step": step,
                    "grad_norm": float(
                        torch.as_tensor(grad_norm)
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    ),
                    "loss": components,
                    "full": full,
                    "zero": zero,
                    "passed": passed_now,
                    "pass_streak": pass_streak,
                    "failures": failures,
                    "elapsed_seconds": (
                        time.perf_counter()
                        - train_started
                    ),
                }
            )

            message = (
                f"[eval] step={step:04d} "
                f"loss={components['total']:.5f} "
                f"targetP={full['target_min_merge_probability']:.3f} "
                f"gate={full['target_min_temporal_gate']:.3f} "
                f"edgeAcc={full['final_edge_accuracy']:.3f} "
                f"groupAcc={full['partition_pairwise_accuracy']:.3f} "
                f"targetComp={full['target_final_component_count']} "
                f"zeroComp={zero['target_final_component_count']} "
                f"pass={passed_now} "
                f"streak={pass_streak}/{PASS_STREAK}"
            )
            if has_tqdm:
                iterator.write(message)
            else:
                print(message)

            if (
                passed_now
                and pass_streak >= PASS_STREAK
            ):
                stopped_early = True
                if has_tqdm:
                    iterator.write(
                        "[EARLY STOP] strict temporal correction is stable."
                    )
                else:
                    print(
                        "[EARLY STOP] strict temporal correction is stable."
                    )
                break

    (
        final_full,
        final_instances,
        final_temporal,
        final_reasoning,
        final_partition,
    ) = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
    )
    (
        final_zero,
        _,
        _,
        _,
        zero_partition,
    ) = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
        zero_temporal=True,
    )
    final_pass, final_failures = strict_pass(
        final_full,
        final_zero,
    )

    # Diagnostics only: remove one temporal evidence source at a time.
    no_hypothesis, _, _, _, _ = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
        use_hypothesis=False,
    )
    no_history, _, _, _, _ = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
        use_history=False,
    )
    no_observer, _, _, _, _ = evaluate_temporal(
        model,
        cut,
        temporal_targets,
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        partitioner,
        cfg,
        use_observer=False,
    )

    training_seconds = (
        time.perf_counter() - train_started
    )
    allocated_gib, reserved_gib = gpu_peak_gib()

    print("\n" + "=" * 118)
    print("Stage 09 final evaluation")
    print("=" * 118)
    print_temporal_metrics(
        history[-1]["step"]
        if history
        else opts.steps,
        final_full,
        final_zero,
        loss=latest_loss_components,
    )
    print(
        f"Spatial forced P(merge)          : "
        f"{cut.spatial_metrics['forced_spatial_merge_probability']:.6f}"
    )
    print(
        f"Target temporal P(merge), min    : "
        f"{final_full['target_min_merge_probability']:.6f}"
    )
    print(
        f"Target temporal gate, min        : "
        f"{final_full['target_min_temporal_gate']:.6f}"
    )
    print(
        f"Spatial target components        : "
        f"{cut.spatial_metrics['target_spatial_component_count']}"
    )
    print(
        f"Temporal target components       : "
        f"{final_full['target_final_component_count']}"
    )
    print(
        f"Zero-context target components   : "
        f"{final_zero['target_final_component_count']}"
    )
    print(
        f"Zero-context max logit change    : "
        f"{final_zero['zero_temporal_max_logit_change']:.8f}"
    )
    print(
        f"Final valid-edge accuracy        : "
        f"{final_full['final_edge_accuracy']:.4f}"
    )
    print(
        f"Final grouping accuracy          : "
        f"{final_full['partition_pairwise_accuracy']:.4f}"
    )
    print(
        f"False merge / split pairs        : "
        f"{final_full['partition_false_merge_pairs']} / "
        f"{final_full['partition_false_split_pairs']}"
    )
    print(
        f"Existence / split accuracy       : "
        f"{final_full['existence_accuracy']:.4f} / "
        f"{final_full['split_accuracy']:.4f}"
    )
    print("\nAblation diagnostics")
    print(
        f"  no hypothesis targetP          : "
        f"{no_hypothesis['target_min_merge_probability']:.4f}"
    )
    print(
        f"  no history targetP             : "
        f"{no_history['target_min_merge_probability']:.4f}"
    )
    print(
        f"  no observer targetP            : "
        f"{no_observer['target_min_merge_probability']:.4f}"
    )
    print(
        f"\nAcceptance                       : "
        f"{'PASS' if final_pass else 'FAIL'}"
    )
    if final_failures:
        print(
            "Failures                         : "
            + "; ".join(final_failures)
        )
    print(
        f"Setup time                       : "
        f"{duration(setup_seconds)}"
    )
    print(
        f"Training time                    : "
        f"{duration(training_seconds)}"
    )
    print(
        f"Peak CUDA                        : "
        f"{allocated_gib:.3f} GiB allocated / "
        f"{reserved_gib:.3f} GiB reserved"
    )
    print("=" * 118)

    # Compact visualization arrays before releasing dense tensors.
    raw_cpu = (
        batch.spatial_inputs[0, 0]
        .detach()
        .float()
        .cpu()
    )
    gt_cpu = (
        batch.gt_labels
        .detach()
        .long()
        .cpu()
    )
    safe_cpu = (
        safe_supervoxels
        .detach()
        .long()
        .cpu()
    )
    spatial_partition_cpu = (
        cut.partition.labels[0]
        .detach()
        .long()
        .cpu()
    )
    final_partition_cpu = (
        final_partition.labels[0]
        .detach()
        .long()
        .cpu()
    )
    zero_partition_cpu = (
        zero_partition.labels[0]
        .detach()
        .long()
        .cpu()
    )

    final_summary = {
        "format_version": 1,
        "stage": "09_temporal_reasoning_overfit",
        "candidate_index": EXPECTED_CANDIDATE_INDEX,
        "target_gt_id": EXPECTED_TARGET_GT_ID,
        "accepted": final_pass,
        "failures": final_failures,
        "completed_steps": int(
            history[-1]["step"]
            if history
            else opts.steps
        ),
        "stopped_early": stopped_early,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "trainable_parameter_count": trainable_count,
        "peak_cuda_allocated_gib": allocated_gib,
        "peak_cuda_reserved_gib": reserved_gib,
        "geometry_suitability": geometry_suitability,
        "feature_donor": feature_donor_info,
        "spatial_rag_warmup": rag_warmup_summary,
        "controlled_cut": {
            "edge_rows": cut.cut_edge_rows,
            **cut.spatial_metrics,
        },
        "temporal_rebase": temporal_bundle.diagnostics,
        "final_full": final_full,
        "final_zero": final_zero,
        "ablations": {
            "no_hypothesis": no_hypothesis,
            "no_history": no_history,
            "no_observer": no_observer,
        },
        "history": history,
        "paths": {
            "sample": str(opts.sample),
            "geometry_predictions": str(
                opts.geometry_predictions
            ),
            "feature_donor_checkpoint": str(
                opts.feature_donor_checkpoint
            ),
            "controlled_context": str(
                opts.controlled_context
            ),
            "controlled_split": str(
                opts.controlled_split
            ),
        },
    }

    artifact = {
        "format_version": 1,
        "stage": "09_temporal_reasoning_overfit",
        "summary": final_summary,
        "model_state": {
            "instance_tokenizer": state_dict_cpu(
                model.instance_tokenizer
            ),
            "history_encoder": state_dict_cpu(
                model.history_encoder
            ),
            "temporal_encoder": state_dict_cpu(
                model.temporal_encoder
            ),
            "temporal_observer": state_dict_cpu(
                model.temporal_observer
            ),
            "instance_temporal": state_dict_cpu(
                model.instance_temporal
            ),
        },
        "spatial_rag_state": {
            "rag_builder": rag_builder_state,
            "rag_network": rag_network_state,
        },
        "controlled_cut_edge_rows": torch.tensor(
            cut.cut_edge_rows,
            dtype=torch.long,
        ),
        "controlled_node_piece": (
            cut.node_piece.detach().cpu()
        ),
        "raw": raw_cpu,
        "gt_labels": gt_cpu,
        "safe_supervoxels": safe_cpu,
        "controlled_spatial_partition": (
            spatial_partition_cpu
        ),
        "temporal_final_partition": (
            final_partition_cpu
        ),
        "zero_temporal_partition": (
            zero_partition_cpu
        ),
        "spacing_um": (
            batch.spacing_um[0]
            .detach()
            .float()
            .cpu()
        ),
        "dref_um": (
            batch.dref_um[0]
            .detach()
            .float()
            .cpu()
        ),
    }

    opts.results.mkdir(
        parents=True,
        exist_ok=True,
    )
    atomic_json(
        opts.results / "last_attempt.json",
        final_summary,
    )
    atomic_torch_save(
        opts.results / "last_attempt.pt",
        artifact,
    )

    if final_pass:
        atomic_json(
            opts.results / "latest_success.json",
            final_summary,
        )
        atomic_torch_save(
            opts.results / "latest_success.pt",
            artifact,
        )
        print(
            "\nSaved latest successful Stage-09 checkpoint."
        )
    else:
        print(
            "\nStage 09 did not satisfy strict acceptance; "
            "latest_success was not replaced."
        )

    print(
        f"Attempt summary : "
        f"{opts.results / 'last_attempt.json'}"
    )
    print(
        f"Attempt artifact: "
        f"{opts.results / 'last_attempt.pt'}"
    )

    return final_summary



# =============================================================================
# MEANINGFUL TEMPORAL-EVIDENCE DEPENDENCE
# =============================================================================


@torch.no_grad()
def restore_saved_spatial_rag(
    checkpoint: dict[str, Any],
    safe_supervoxels: Tensor,
    geometry: GeometryState,
    derived_cache,
    decoded,
    batch,
    cfg,
) -> tuple[RAGState, RAGTargets]:
    """Rebuild the exact Stage-09 spatial RAG using saved learned weights.

    This does NOT repeat the RAG warm-up.  Region statistics are deterministic;
    the trainable builder projection and RAG network are restored from the
    successful Stage-09 artifact.
    """
    saved = dict(checkpoint.get("spatial_rag_state", {}))
    if "rag_builder" not in saved or "rag_network" not in saved:
        raise RuntimeError(
            "Stage-09 success artifact does not contain saved spatial RAG state."
        )

    builder = RAGBuilder(
        cfg.partition,
        cfg.spatial,
    ).to(safe_supervoxels.device)
    network = SpatialRAGNetwork(
        cfg.partition,
        builder.node_feature_dim,
        builder.edge_feature_dim,
    ).to(safe_supervoxels.device)
    builder.load_state_dict(
        saved["rag_builder"],
        strict=True,
    )
    network.load_state_dict(
        saved["rag_network"],
        strict=True,
    )
    builder.eval()
    network.eval()

    statistics = build_supervoxel_statistics(
        [safe_supervoxels],
        batch.spatial_inputs,
        geometry,
        batch.spacing_um,
        (
            decoded.d0,
            decoded.d1,
            decoded.d2,
        ),
        derived=derived_cache,
    )
    rag = builder(
        [safe_supervoxels],
        decoded.d0,
        batch.spatial_inputs,
        geometry,
        batch.spacing_um,
        batch.dref_um,
        statistics_by_batch=statistics,
        derived_cache=derived_cache,
    )
    rag = network(rag)

    criterion = RAGCriterion(
        cfg.partition,
    ).to(safe_supervoxels.device)
    targets = criterion.build_targets(
        rag,
        batch.gt_labels.unsqueeze(0).to(
            safe_supervoxels.device
        ),
    )
    return detach_rag(rag), targets


def load_saved_temporal_model(
    checkpoint: dict[str, Any],
    cfg,
    device: torch.device,
) -> TemporalOverfitModel:
    saved = dict(checkpoint.get("model_state", {}))
    required = {
        "instance_tokenizer",
        "history_encoder",
        "temporal_encoder",
        "temporal_observer",
        "instance_temporal",
    }
    missing = required - set(saved)
    if missing:
        raise RuntimeError(
            "Stage-09 success artifact is missing temporal module states: "
            f"{sorted(missing)}"
        )

    model = TemporalOverfitModel(cfg).to(device)
    model.instance_tokenizer.load_state_dict(
        saved["instance_tokenizer"],
        strict=True,
    )
    model.history_encoder.load_state_dict(
        saved["history_encoder"],
        strict=True,
    )
    model.temporal_encoder.load_state_dict(
        saved["temporal_encoder"],
        strict=True,
    )
    model.temporal_observer.load_state_dict(
        saved["temporal_observer"],
        strict=True,
    )
    model.instance_temporal.load_state_dict(
        saved["instance_temporal"],
        strict=True,
    )
    model.eval()
    return model


def replace_temporal_tokens(
    temporal: TemporalState,
    new_tokens: Tensor,
) -> TemporalState:
    if new_tokens.shape != temporal.tokens.shape:
        raise ValueError(
            "Counterfactual temporal tokens must preserve [tracklets,d_model]."
        )
    return replace(
        temporal,
        tokens=new_tokens,
    )


def choose_nearby_wrong_donors(
    temporal: TemporalState,
    temporal_bundle: TemporalBundle,
    dref_um: Tensor,
    cfg,
) -> tuple[list[int], list[int], list[float]]:
    """Choose non-target tracklets closest to the target-track references.

    Prefer non-target tracks that are themselves inside the spatial observation
    crop and within the production instance-match radius.  Fall back to the
    nearest available non-target tracks if necessary, but report the distances.
    """
    target = [
        int(value)
        for value in temporal_bundle.target_local_tracklets
    ]
    target_set = set(target)
    candidates = [
        index
        for index in range(temporal.tokens.shape[0])
        if index not in target_set
    ]
    if len(candidates) < len(target):
        raise RuntimeError(
            "Not enough non-target temporal tracklets to build a wrong-context "
            "counterfactual."
        )

    preferred = [
        index
        for index in candidates
        if bool(
            temporal_bundle.observer_tracklet_mask[index].item()
        )
    ]
    pool = preferred if len(preferred) >= len(target) else candidates

    selected: list[int] = []
    distances: list[float] = []
    used: set[int] = set()
    dref = float(
        dref_um.float().clamp_min(1e-6).item()
    )

    for target_index in target:
        ranked = sorted(
            (
                (
                    float(
                        torch.linalg.vector_norm(
                            temporal.ref_um[candidate]
                            - temporal.ref_um[target_index]
                        ).item()
                    )
                    / dref,
                    candidate,
                )
                for candidate in pool
                if candidate not in used
            ),
            key=lambda row: (row[0], row[1]),
        )
        if not ranked:
            raise RuntimeError(
                "Could not assign a unique nearby wrong-context donor."
            )
        distance, donor = ranked[0]
        used.add(donor)
        selected.append(int(donor))
        distances.append(float(distance))

    return target, selected, distances


def wrong_nearby_token_context(
    temporal: TemporalState,
    temporal_bundle: TemporalBundle,
    dref_um: Tensor,
    cfg,
) -> tuple[TemporalState, dict[str, Any]]:
    """Place nearby non-target token CONTENT at the exact target refs.

    Only encoded `tokens` are exchanged.  ref_um, salience, reliability, status,
    batch assignment, and therefore the local temporal-support geometry remain
    exactly unchanged.  This is the strongest single test for "content matters"
    in the current one-case overfit.
    """
    target, donors, distances = choose_nearby_wrong_donors(
        temporal,
        temporal_bundle,
        dref_um,
        cfg,
    )
    tokens = temporal.tokens.clone()
    original = tokens.clone()

    # Pairwise swap preserves the global token multiset exactly while ensuring
    # that every target reference now carries non-target encoded evidence.
    for target_index, donor_index in zip(
        target,
        donors,
    ):
        tokens[target_index] = original[donor_index]
        tokens[donor_index] = original[target_index]

    original_ids = (
        temporal_bundle.selected_original_tracklet_ids
        .detach()
        .cpu()
    )
    diagnostics = {
        "target_local_tracklets": target,
        "target_original_tracklets": [
            int(original_ids[index])
            for index in target
        ],
        "donor_local_tracklets": donors,
        "donor_original_tracklets": [
            int(original_ids[index])
            for index in donors
        ],
        "target_to_donor_distance_dref": distances,
        "all_donors_inside_observer_crop": all(
            bool(
                temporal_bundle.observer_tracklet_mask[index].item()
            )
            for index in donors
        ),
    }
    return (
        replace_temporal_tokens(
            temporal,
            tokens,
        ),
        diagnostics,
    )


def globally_deranged_token_context(
    temporal: TemporalState,
) -> tuple[TemporalState, dict[str, Any]]:
    """Deterministically derange token content while preserving every ref.

    Only the encoded token row is moved; every support/location scalar stays at
    its original tracklet reference.  A half-cycle rotation avoids fixed points
    for the 51-tracklet Stage-08 graph.
    """
    count = int(temporal.tokens.shape[0])
    if count < 2:
        raise RuntimeError(
            "Need at least two temporal tracklets for token derangement."
        )

    shift = max(1, count // 2)
    permutation = (
        torch.arange(
            count,
            device=temporal.tokens.device,
        )
        + shift
    ) % count

    fixed_points = int(
        (
            permutation
            == torch.arange(
                count,
                device=permutation.device,
            )
        )
        .sum()
        .item()
    )
    if fixed_points:
        raise RuntimeError(
            "Deterministic token permutation unexpectedly contains fixed points."
        )

    return (
        replace_temporal_tokens(
            temporal,
            temporal.tokens[permutation],
        ),
        {
            "shift": int(shift),
            "fixed_points": fixed_points,
            "tracklet_count": count,
        },
    )


def spatially_irrelevant_nonempty_context(
    temporal: TemporalState,
    dref_um: Tensor,
) -> TemporalState:
    """Keep all temporal content but move every ref far from the spatial crop."""
    shift_um = (
        float(EVIDENCE_DEPENDENCE["shift_distance_dref"])
        * float(
            dref_um.float().clamp_min(1e-6).item()
        )
    )
    displacement = temporal.ref_um.new_tensor(
        [shift_um, 0.0, 0.0]
    )
    return replace(
        temporal,
        ref_um=temporal.ref_um + displacement[None],
    )


@torch.no_grad()
def evaluate_given_temporal_state(
    model: TemporalOverfitModel,
    instances: InstanceState,
    temporal: TemporalState,
    cut: ControlledCut,
    targets: TemporalTargets,
    partitioner: GraphPartitioner,
    cfg,
    dref_um: Tensor,
) -> tuple[
    dict[str, float | int],
    ReasoningState,
    PartitionState,
]:
    reasoning = model.instance_temporal(
        instances,
        cut.rag,
        temporal,
        dref_um,
    )
    final_partition = partitioner(
        cut.rag,
        reasoning.final_edge_logits,
        cfg.partition.final_merge_threshold,
    )

    probabilities = reasoning.final_edge_logits.sigmoid()
    valid = targets.rag.valid
    truth = targets.rag.target > 0.5
    predicted = probabilities >= 0.5

    grouping = grouping_metrics(
        final_partition,
        targets.rag,
        cfg,
    )
    target_probability = probabilities[
        cut.cut_edge_mask
    ]
    target_gate = reasoning.edge_temporal_gate[
        cut.cut_edge_mask
    ]

    metrics: dict[str, float | int] = {
        "target_min_merge_probability": float(
            target_probability.min().item()
        ),
        "target_mean_merge_probability": float(
            target_probability.mean().item()
        ),
        "target_min_temporal_gate": float(
            target_gate.min().item()
        ),
        "target_mean_temporal_gate": float(
            target_gate.mean().item()
        ),
        "target_final_component_count": (
            target_component_count(
                final_partition,
                cut.target_node_rows,
            )
        ),
        "final_edge_accuracy": float(
            (
                predicted[valid]
                == truth[valid]
            )
            .float()
            .mean()
            .item()
        ),
        "existence_accuracy": binary_accuracy(
            reasoning.instance_exist_logits,
            targets.existence,
        ),
        "split_accuracy": binary_accuracy(
            reasoning.split_logits,
            targets.split,
        ),
        "mean_instance_temporal_support": (
            float(
                reasoning.temporal_support.mean().item()
            )
            if reasoning.temporal_support.numel()
            else 0.0
        ),
        "max_instance_temporal_support": (
            float(
                reasoning.temporal_support.max().item()
            )
            if reasoning.temporal_support.numel()
            else 0.0
        ),
        **{
            f"partition_{key}": value
            for key, value in grouping.items()
        },
    }
    return metrics, reasoning, final_partition


def evidence_dependence_verdict(
    results: dict[str, dict[str, float | int]],
) -> tuple[bool, str, list[str]]:
    correct = results["correct"]
    zero = results["empty"]
    shifted = results["shifted_nonempty"]
    wrong = results["wrong_nearby_tokens"]
    shuffled = results["global_token_derangement"]

    correct_p = float(
        correct["target_min_merge_probability"]
    )
    wrong_p = float(
        wrong["target_min_merge_probability"]
    )
    shuffled_p = float(
        shuffled["target_min_merge_probability"]
    )

    failures: list[str] = []

    if (
        correct_p
        < EVIDENCE_DEPENDENCE[
            "correct_merge_probability_min"
        ]
        or int(
            correct["target_final_component_count"]
        )
        != 1
    ):
        failures.append(
            "trained correct-context baseline no longer solves the target"
        )

    if int(
        zero["target_final_component_count"]
    ) != 2:
        failures.append(
            "empty temporal control does not preserve the TWO-piece spatial error"
        )

    if int(
        shifted["target_final_component_count"]
    ) != 2:
        failures.append(
            "non-empty but spatially irrelevant temporal context still merges"
        )

    counterfactual_limit = float(
        EVIDENCE_DEPENDENCE[
            "counterfactual_merge_probability_max"
        ]
    )
    if (
        wrong_p > counterfactual_limit
        or int(
            wrong["target_final_component_count"]
        )
        != 2
    ):
        failures.append(
            "wrong nearby-cell token content still triggers the merge"
        )

    if (
        shuffled_p > counterfactual_limit
        or int(
            shuffled["target_final_component_count"]
        )
        != 2
    ):
        failures.append(
            "globally deranged token content still triggers the merge"
        )

    worst_counterfactual = max(
        wrong_p,
        shuffled_p,
    )
    margin = correct_p - worst_counterfactual
    if (
        margin
        < EVIDENCE_DEPENDENCE[
            "correct_vs_counterfactual_margin_min"
        ]
    ):
        failures.append(
            "correct-vs-wrong token confidence margin is too small "
            f"({margin:.3f})"
        )

    passed = not failures
    if passed:
        classification = (
            "MEANINGFUL_TEMPORAL_CONTENT_DEPENDENCE"
        )
    else:
        # More informative failure classification.
        if (
            int(
                shifted[
                    "target_final_component_count"
                ]
            )
            == 2
            and wrong_p >= 0.90
            and shuffled_p >= 0.90
        ):
            classification = (
                "TEMPORAL_SUPPORT_OR_POSITION_DEPENDENCE_WITH_CONTENT_MEMORIZATION"
            )
        elif (
            margin >= 0.10
            and (
                wrong_p < correct_p
                or shuffled_p < correct_p
            )
        ):
            classification = (
                "PARTIAL_TEMPORAL_CONTENT_DEPENDENCE"
            )
        else:
            classification = (
                "MEANINGFUL_TEMPORAL_CONTENT_DEPENDENCE_NOT_DEMONSTRATED"
            )

    return passed, classification, failures


def print_evidence_row(
    name: str,
    metrics: dict[str, float | int],
) -> None:
    print(
        f"{name:28s} "
        f"Pmerge={metrics['target_min_merge_probability']:.4f}  "
        f"gate={metrics['target_min_temporal_gate']:.4f}  "
        f"targetComp={metrics['target_final_component_count']}  "
        f"support={metrics['mean_instance_temporal_support']:.4f}  "
        f"edgeAcc={metrics['final_edge_accuracy']:.3f}"
    )


def run_evidence_dependence(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run strong post-overfit temporal-content counterfactuals.

    No optimization is performed.  The saved Stage-09 temporal weights and
    saved learned RAG weights are restored exactly.
    """
    success_path = args.result_dir / "latest_success.pt"
    if not success_path.exists():
        raise FileNotFoundError(
            f"Missing successful Stage-09 artifact:\n  {success_path}\n"
            "Run normal Stage 09 training first."
        )

    checkpoint = torch_load(
        success_path
    )
    summary = dict(
        checkpoint.get("summary", {})
    )
    if not bool(summary.get("accepted", False)):
        raise RuntimeError(
            "latest_success.pt is not marked as an accepted Stage-09 run."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "--evidence-dependence requires CUDA for the frozen donor/observer "
            "forward pass."
        )

    device = torch.device("cuda")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = stage01.build_debug_config()
    batch = stage01.load_debug_batch(
        args.sample_path,
        device,
    )
    artifacts = load_stage08_artifacts(
        args.controlled_context,
        args.controlled_split,
        args.case_json,
    )

    print("\n" + "=" * 120)
    print("STIR-Net Stage 09 — meaningful temporal-evidence dependence")
    print("=" * 120)
    print(
        "Mode                             : evaluation only; NO training"
    )
    print(
        f"Stage-09 checkpoint              : {success_path}"
    )
    print(
        f"Target                           : GT {EXPECTED_TARGET_GT_ID}"
    )
    print(
        "Core question                    : does encoded temporal CONTENT matter,"
    )
    print(
        "                                   or is any non-empty local support enough?"
    )
    print("=" * 120)

    started = time.perf_counter()

    (
        geometry,
        pyramid,
        decoded,
        geometry_suitability,
        feature_donor_info,
    ) = prepare_candidate0_spatial_evidence(
        batch,
        args.geometry_predictions,
        args.feature_donor_checkpoint,
        cfg,
        device,
    )
    derived = build_geometry_derived_cache(
        geometry,
        cfg.partition,
        padding_mask=None,
    )

    safe_supervoxels = (
        artifacts.safe_supervoxels
        .to(device)
        .long()
    )
    learned_rag, rag_targets = restore_saved_spatial_rag(
        checkpoint,
        safe_supervoxels,
        geometry,
        derived,
        decoded,
        batch,
        cfg,
    )
    partitioner = GraphPartitioner().to(device)
    cut = make_controlled_rag_cut(
        learned_rag,
        rag_targets,
        artifacts.safe_supervoxels,
        artifacts.piece_map_crop,
        batch.gt_labels,
        cfg,
        partitioner,
    )

    temporal_bundle = rebase_temporal_context_to_crop(
        artifacts.context,
        artifacts.split,
        spacing_um=batch.spacing_um[0],
        dref_um=batch.dref_um[0],
        crop_shape_zyx=tuple(
            int(v)
            for v in batch.gt_labels.shape
        ),
        device=device,
    )
    targets = build_temporal_targets(
        cut,
        rag_targets,
        batch.gt_labels.unsqueeze(0).to(device),
        LossConfig(),
        device,
    )

    model = load_saved_temporal_model(
        checkpoint,
        cfg,
        device,
    )

    # Build the exact trained tokenizer state and exact full encoded temporal
    # state once. Counterfactuals below change ONLY the TemporalState presented
    # to InstanceTemporalReasoner.
    instances = model.instance_tokenizer(
        cut.partition,
        cut.rag,
        decoded,
        geometry,
        batch.spacing_um,
        batch.dref_um,
    )
    correct_temporal = model.encode_temporal(
        temporal_bundle,
        decoded,
        geometry,
        pyramid,
        batch.spacing_um,
        batch.dref_um,
        use_history=True,
        use_hypothesis=True,
        use_observer=True,
    )
    empty_temporal = model.temporal_encoder.empty(
        instances.tokens.device,
        instances.tokens.dtype,
    )
    shifted_temporal = spatially_irrelevant_nonempty_context(
        correct_temporal,
        batch.dref_um[0],
    )
    wrong_temporal, wrong_diagnostics = (
        wrong_nearby_token_context(
            correct_temporal,
            temporal_bundle,
            batch.dref_um[0],
            cfg,
        )
    )
    shuffled_temporal, shuffle_diagnostics = (
        globally_deranged_token_context(
            correct_temporal,
        )
    )

    variants = {
        "correct": correct_temporal,
        "empty": empty_temporal,
        "shifted_nonempty": shifted_temporal,
        "wrong_nearby_tokens": wrong_temporal,
        "global_token_derangement": (
            shuffled_temporal
        ),
    }
    results: dict[
        str,
        dict[str, float | int],
    ] = {}

    for name, temporal in variants.items():
        metrics, _, _ = evaluate_given_temporal_state(
            model,
            instances,
            temporal,
            cut,
            targets,
            partitioner,
            cfg,
            batch.dref_um,
        )
        results[name] = metrics

    passed, classification, failures = (
        evidence_dependence_verdict(
            results
        )
    )

    print("\nCounterfactual temporal-evidence results")
    print("-" * 120)
    print_evidence_row(
        "correct context",
        results["correct"],
    )
    print_evidence_row(
        "empty context",
        results["empty"],
    )
    print_evidence_row(
        "shifted non-empty",
        results["shifted_nonempty"],
    )
    print_evidence_row(
        "WRONG nearby tokens",
        results["wrong_nearby_tokens"],
    )
    print_evidence_row(
        "global token derangement",
        results["global_token_derangement"],
    )
    print("-" * 120)

    correct_p = float(
        results["correct"][
            "target_min_merge_probability"
        ]
    )
    wrong_p = float(
        results["wrong_nearby_tokens"][
            "target_min_merge_probability"
        ]
    )
    shuffled_p = float(
        results["global_token_derangement"][
            "target_min_merge_probability"
        ]
    )
    margin = correct_p - max(
        wrong_p,
        shuffled_p,
    )

    print("\nWrong-nearby donor assignment")
    print(
        f"  target original tracklets      : "
        f"{wrong_diagnostics['target_original_tracklets']}"
    )
    print(
        f"  donor original tracklets       : "
        f"{wrong_diagnostics['donor_original_tracklets']}"
    )
    print(
        f"  target→donor distances dref    : "
        f"{[round(v, 3) for v in wrong_diagnostics['target_to_donor_distance_dref']]}"
    )
    print(
        f"  donors inside observer crop    : "
        f"{wrong_diagnostics['all_donors_inside_observer_crop']}"
    )

    print("\nDependence verdict")
    print(
        f"  correct-vs-worst-wrong margin  : {margin:.4f}"
    )
    print(
        f"  classification                 : {classification}"
    )
    print(
        f"  strict meaningful dependence   : "
        f"{'PASS' if passed else 'NOT DEMONSTRATED'}"
    )
    for failure in failures:
        print(
            f"    - {failure}"
        )

    elapsed = time.perf_counter() - started
    allocated, reserved = gpu_peak_gib()
    print(
        f"  elapsed                        : {duration(elapsed)}"
    )
    print(
        f"  peak CUDA                      : "
        f"{allocated:.3f} / {reserved:.3f} GiB"
    )
    print("=" * 120)

    report = {
        "format_version": 1,
        "stage": "09_temporal_evidence_dependence",
        "source_checkpoint": str(
            success_path
        ),
        "passed": passed,
        "classification": classification,
        "failures": failures,
        "thresholds": dict(
            EVIDENCE_DEPENDENCE
        ),
        "results": results,
        "correct_vs_worst_wrong_margin": (
            margin
        ),
        "wrong_nearby_diagnostics": (
            wrong_diagnostics
        ),
        "global_derangement_diagnostics": (
            shuffle_diagnostics
        ),
        "temporal_rebase": (
            temporal_bundle.diagnostics
        ),
        "geometry_suitability": (
            geometry_suitability
        ),
        "feature_donor": (
            feature_donor_info
        ),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_gib": allocated,
        "peak_cuda_reserved_gib": reserved,
    }
    atomic_json(
        args.result_dir
        / "evidence_dependence.json",
        report,
    )
    print(
        f"Saved evidence report: "
        f"{args.result_dir / 'evidence_dependence.json'}"
    )
    return report



# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize(results: Path) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for --visualize."
        ) from exc

    success = results / "latest_success.pt"
    attempt = results / "last_attempt.pt"
    path = (
        success
        if success.exists()
        else attempt
    )
    if not path.exists():
        raise FileNotFoundError(
            f"No Stage-09 artifact found under {results}."
        )

    payload = torch_load(path)
    summary = dict(payload["summary"])
    spacing = tuple(
        float(v)
        for v in torch.as_tensor(
            payload["spacing_um"]
        ).tolist()
    )

    raw = torch.as_tensor(
        payload["raw"]
    ).float().numpy()
    gt = torch.as_tensor(
        payload["gt_labels"]
    ).long().numpy()
    controlled = torch.as_tensor(
        payload["controlled_spatial_partition"]
    ).long().numpy()
    final = torch.as_tensor(
        payload["temporal_final_partition"]
    ).long().numpy()

    viewer = napari.Viewer(
        title=(
            "STIR-Net Stage 09 — temporal TWO→ONE overfit "
            f"({'PASS' if summary.get('accepted') else 'ATTEMPT'})"
        ),
        ndisplay=3,
    )
    viewer.add_image(
        raw,
        name="Raw",
        colormap="gray",
        scale=spacing,
        visible=True,
    )
    viewer.add_labels(
        gt,
        name="GT labels",
        scale=spacing,
        visible=False,
    )
    viewer.add_labels(
        controlled,
        name="CONTROLLED spatial partition — TWO",
        scale=spacing,
        visible=True,
    )
    viewer.add_labels(
        final,
        name="TEMPORAL final partition — should be ONE",
        scale=spacing,
        visible=True,
    )

    print("\n" + "=" * 96)
    print(f"Loaded Stage-09 artifact: {path}")
    print(
        f"Acceptance                  : "
        f"{'PASS' if summary.get('accepted') else 'FAIL'}"
    )
    full = summary.get("final_full", {})
    zero = summary.get("final_zero", {})
    print(
        f"Target temporal P(merge)    : "
        f"{full.get('target_min_merge_probability', float('nan')):.4f}"
    )
    print(
        f"Spatial target components   : "
        f"{summary.get('controlled_cut', {}).get('target_spatial_component_count')}"
    )
    print(
        f"Temporal target components  : "
        f"{full.get('target_final_component_count')}"
    )
    print(
        f"Zero-context components     : "
        f"{zero.get('target_final_component_count')}"
    )
    print("=" * 96)
    napari.run()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated STIR-Net temporal TWO->ONE memorization on the "
            "Stage-08 controlled candidate-0 / GT-6 case."
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="lr",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    parser.add_argument(
        "--rag-warmup-steps",
        type=int,
        default=DEFAULT_RAG_WARMUP_STEPS,
    )
    parser.add_argument(
        "--rag-warmup-learning-rate",
        type=float,
        default=DEFAULT_RAG_WARMUP_LR,
    )
    parser.add_argument(
        "--rag-eval-every",
        type=int,
        default=DEFAULT_RAG_EVAL_EVERY,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=DEFAULT_SAMPLE,
    )
    parser.add_argument(
        "--geometry-predictions",
        type=Path,
        default=DEFAULT_GEOMETRY_PREDICTIONS,
        help=(
            "Candidate-0 Stage-01 last_attempt_predictions.pt. Its explicit "
            "geometry is used after the Stage-09 suitability gate."
        ),
    )
    parser.add_argument(
        "--feature-donor-checkpoint",
        type=Path,
        default=DEFAULT_FEATURE_DONOR_CHECKPOINT,
        help=(
            "Any successful Stage-01 joint/full checkpoint used only for one "
            "frozen D0/D1/D2/hidden-feature forward pass."
        ),
    )
    parser.add_argument(
        "--controlled-context",
        type=Path,
        default=DEFAULT_CONTROLLED_CONTEXT,
    )
    parser.add_argument(
        "--controlled-split",
        type=Path,
        default=DEFAULT_CONTROLLED_SPLIT,
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        default=DEFAULT_CASE_JSON,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
    )
    parser.add_argument(
        "--evidence-dependence",
        action="store_true",
        help=(
            "Load latest_success.pt and run counterfactual temporal-content "
            "dependence controls without further training."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.sample_path = args.sample_path.resolve()
    args.geometry_predictions = (
        args.geometry_predictions.resolve()
    )
    args.feature_donor_checkpoint = (
        args.feature_donor_checkpoint.resolve()
    )
    args.controlled_context = (
        args.controlled_context.resolve()
    )
    args.controlled_split = (
        args.controlled_split.resolve()
    )
    args.case_json = args.case_json.resolve()
    args.result_dir = args.result_dir.resolve()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.lr <= 0:
        raise ValueError(
            "--learning-rate must be positive"
        )
    if args.eval_every < 1:
        raise ValueError(
            "--eval-every must be >= 1"
        )
    if args.rag_warmup_steps < 1:
        raise ValueError(
            "--rag-warmup-steps must be >= 1"
        )
    if args.rag_warmup_learning_rate <= 0:
        raise ValueError(
            "--rag-warmup-learning-rate must be positive"
        )
    if args.rag_eval_every < 1:
        raise ValueError(
            "--rag-eval-every must be >= 1"
        )

    if args.visualize:
        visualize(args.result_dir)
        return

    if args.evidence_dependence:
        run_evidence_dependence(args)
        return

    train(
        Options(
            steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
            rag_warmup_steps=(
                args.rag_warmup_steps
            ),
            rag_warmup_lr=(
                args.rag_warmup_learning_rate
            ),
            rag_eval_every=(
                args.rag_eval_every
            ),
            seed=args.seed,
            sample=args.sample_path,
            geometry_predictions=(
                args.geometry_predictions
            ),
            feature_donor_checkpoint=(
                args.feature_donor_checkpoint
            ),
            controlled_context=(
                args.controlled_context
            ),
            controlled_split=(
                args.controlled_split
            ),
            case_json=args.case_json,
            results=args.result_dir,
        )
    )


if __name__ == "__main__":
    main()
