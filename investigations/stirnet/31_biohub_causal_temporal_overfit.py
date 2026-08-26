from __future__ import annotations

r"""
Investigation 31 — causal temporal overfit on the annotated BioHub movie.

Why Investigation 31 exists
----------------------------
Investigation 30 established three useful facts:

1. EMPTY temporal input is an exact spatial no-op.
2. The temporal pathway has enough capacity to reproduce most manual splits.
3. The first good solutions did NOT depend on the actual temporal content:
   FULL and SHUFFLED temporal states behaved almost identically.
4. Later, FULL began to outperform SHUFFLED, but optimization became unstable
   and the temporal gate eventually collapsed to ~0 everywhere.

Investigation 31 attacks those two failure modes directly:

    A. static "temporal support exists" shortcut
    B. unstable single-frame high-learning-rate optimization

It does NOT change the production temporal architecture.

Core causal objective
---------------------
For a normal FULL temporal state:

    manual CUT edge  -> final logit should be negative
    manual KEEP edge -> final logit should be positive

For a CORRUPTED temporal state:

    shuffled:
        tracklet content is assigned to the wrong physical tracklet positions

    contentless:
        tracklet tokens are zeroed while physical positions, salience,
        reliability and support remain present

the final edge logits are explicitly trained to return to the frozen spatial
answer.

That makes the following shortcut expensive:

    "some temporal support exists"
            -> use static RAG geometry to split

because the contentless negative still has temporal support but must NOT split.

A causal margin is also imposed on manual CUT edges:

    FULL cut logit + margin <= corrupted cut logit

so the model is rewarded for making the correct temporal evidence specifically
more split-inducing than corrupted temporal evidence.

Optimization changes vs Investigation 30
----------------------------------------
- default LR: 3e-4 instead of 2e-3
- no keep-gate penalty on FULL temporal examples
- gradient accumulation over multiple frames per optimizer update
- every optimizer update contains both correction and clean-frame exposure
- corrupted-temporal no-op supervision
- causal margin on manual CUT edges
- best-checkpoint selection favors BOTH:
      manual split recovery
      and FULL > SHUFFLED/CONTENTLESS

Spatial network policy
----------------------
Exactly like Investigation 30:

    spatial CNN       : NOT instantiated
    dense geometry    : NOT rerun
    watershed         : NOT rerun
    spatial RAG net   : NOT rerun

The script reuses:
    Investigation 25 current instances
    Investigation 24 atomic supervoxels
    Investigation 12 saved compact RAG
    Investigation 30 Trackastra + temporal-v4 caches

If Investigation-30 preprocessing caches are missing, this script can rebuild
them using Investigation-30's preparation functions, still without spatial
network inference.

Interpretation
--------------
This remains an OVERFIT experiment. A successful result means the production
temporal pathway can learn the annotated corrections *because of temporal
content*. It does not yet establish generalization to another BioHub movie.

Typical run
-----------
From repository root:

    python .\investigations\stirnet\31_biohub_causal_temporal_overfit.py

Short smoke run:

    python .\investigations\stirnet\31_biohub_causal_temporal_overfit.py ^
        --steps 200 --eval-every 50

Outputs:
    runs/stirnet/evaluation/
        31_biohub_causal_temporal_overfit/<sample>/
            latest.pt
            best.pt
            best_metrics.json
            final.pt
            final_metrics.json
            training_history.json
            predictions/
"""

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# REPOSITORY / INVESTIGATION-30 IMPORT
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (
            (candidate / "learned").is_dir()
            and (candidate / "investigations").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate

    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_inv30():
    path = (
        ROOT
        / "investigations"
        / "stirnet"
        / "30_biohub_temporal_partition_overfit.py"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Investigation 31 intentionally reuses Investigation 30's "
            f"artifact/preparation code, but this file is missing:\n  {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "stirnet_investigation_30",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Investigation 30: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inv30 = load_inv30()

from learned.stirnet.data.historical_instances import (
    TEMPORAL_CACHE_CONTRACT_VERSION,
)
from learned.stirnet.data.trackastra_cache import load_cache
from learned.stirnet.model.config import ModelConfig
from learned.stirnet.model.types import TemporalState


# mmap-backed arrays are intentionally read-only in the frozen artifact path.
# Investigation 30 emitted PyTorch's one-time "non-writable NumPy" warning even
# though these tensors are never mutated. Silence only that known warning.
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
)


# =============================================================================
# DEFAULTS
# =============================================================================

SCRIPT_NAME = "31_biohub_causal_temporal_overfit"
DEFAULT_SAMPLE_ID = inv30.DEFAULT_SAMPLE_ID
DEFAULT_FRAME_COUNT = inv30.DEFAULT_FRAME_COUNT
DEFAULT_SPACING_ZYX_UM = inv30.DEFAULT_SPACING_ZYX_UM
DEFAULT_TEMPORAL_RADIUS = inv30.DEFAULT_TEMPORAL_RADIUS

DEFAULT_STEPS = 1500
DEFAULT_LR = 3.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_EVAL_EVERY = 100
DEFAULT_PRINT_EVERY = 20
DEFAULT_ACCUMULATE_FRAMES = 4
DEFAULT_CORRECTION_FRAME_PROB = 0.70
DEFAULT_CLEAN_KEEP_EDGES = 512
DEFAULT_KEEP_TO_CUT_RATIO = 12
DEFAULT_SPLIT_LOSS_WEIGHT = 0.05
DEFAULT_GRAD_CLIP = 3.0

# Causal terms.
DEFAULT_NOOP_WEIGHT = 0.50
DEFAULT_CORRUPTED_GATE_WEIGHT = 0.05
DEFAULT_CAUSAL_MARGIN_WEIGHT = 0.50
DEFAULT_CAUSAL_MARGIN = 1.0

# Alternate the two corruption types by default.
CORRUPTION_TYPES = ("contentless", "shuffled")


# =============================================================================
# HELPERS
# =============================================================================


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return _jsonable(value.detach().cpu().item())
        return [_jsonable(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(
                _jsonable(payload),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# =============================================================================
# CORRUPTED TEMPORAL STATES
# =============================================================================


def contentless_temporal_state(temporal: TemporalState) -> TemporalState:
    """Keep temporal support/positions present but erase learned content.

    This is the key anti-shortcut negative:
        local temporal support still exists,
        but tracklet content carries no useful temporal identity/history.
    """
    return TemporalState(
        tokens=torch.zeros_like(temporal.tokens),
        ref_um=temporal.ref_um,
        batch_index=temporal.batch_index,
        salience=temporal.salience,
        reliability=temporal.reliability,
        status=temporal.status,
        node_tokens=torch.zeros_like(temporal.node_tokens),
    )


def shuffled_temporal_state(
    temporal: TemporalState,
    *,
    seed: int,
) -> TemporalState:
    return inv30.shuffled_temporal_state(
        temporal,
        seed=seed,
    )


def corrupt_temporal_state(
    temporal: TemporalState,
    *,
    corruption: str,
    seed: int,
) -> TemporalState:
    if corruption == "contentless":
        return contentless_temporal_state(temporal)
    if corruption == "shuffled":
        return shuffled_temporal_state(
            temporal,
            seed=seed,
        )
    raise ValueError(f"Unknown temporal corruption: {corruption}")


# =============================================================================
# FORWARD
# =============================================================================


def build_spatial_states(
    frozen_spatial,
    case,
    *,
    device: torch.device,
    dref_um: float,
):
    return frozen_spatial(
        case,
        device=device,
        dref_um_value=dref_um,
    )


def reason_with_temporal(
    temporal_model,
    instances,
    rag,
    temporal: TemporalState,
    *,
    device: torch.device,
    dref_um: float,
):
    return temporal_model.reasoner(
        instances,
        rag,
        temporal,
        torch.tensor(
            [float(dref_um)],
            device=device,
            dtype=torch.float32,
        ),
    )


def full_and_corrupted_forward(
    temporal_model,
    frozen_spatial,
    case,
    temporal_payload: dict[str, Any],
    *,
    device: torch.device,
    dref_um: float,
    corruption: str,
    corruption_seed: int,
):
    rag, instances = build_spatial_states(
        frozen_spatial,
        case,
        device=device,
        dref_um=dref_um,
    )

    temporal = temporal_model.encode_temporal(
        temporal_payload,
        device=device,
    )
    full = reason_with_temporal(
        temporal_model,
        instances,
        rag,
        temporal,
        device=device,
        dref_um=dref_um,
    )

    corrupted_state = corrupt_temporal_state(
        temporal,
        corruption=corruption,
        seed=corruption_seed,
    )
    corrupted = reason_with_temporal(
        temporal_model,
        instances,
        rag,
        corrupted_state,
        device=device,
        dref_um=dref_um,
    )

    return rag, instances, temporal, full, corrupted


# =============================================================================
# LOSS
# =============================================================================


@dataclasses.dataclass
class CausalLoss:
    total: Tensor

    full_edge: Tensor
    full_cut: Tensor
    full_keep: Tensor
    split: Tensor

    corrupted_noop: Tensor
    corrupted_gate: Tensor
    causal_margin: Tensor

    selected_cut_edges: int
    selected_keep_edges: int
    corruption: str


def choose_keep_indices(
    case,
    *,
    clean_keep_edges: int,
    keep_to_cut_ratio: int,
    rng: random.Random,
) -> Tensor:
    # Reuse Investigation 30's balanced KEEP sampler. It preserves all KEEP
    # edges in changed components and samples ordinary clean components.
    return inv30.choose_keep_indices(
        case,
        clean_keep_edges=clean_keep_edges,
        keep_to_cut_ratio=keep_to_cut_ratio,
        rng=rng,
    )


def balanced_split_indices(
    split_target: Tensor,
    *,
    rng: random.Random,
) -> Tensor:
    return inv30.balanced_split_indices(
        split_target,
        negative_ratio=8,
        rng=rng,
    )


def causal_temporal_loss(
    case,
    rag,
    full,
    corrupted,
    *,
    device: torch.device,
    corruption: str,
    clean_keep_edges: int,
    keep_to_cut_ratio: int,
    split_loss_weight: float,
    noop_weight: float,
    corrupted_gate_weight: float,
    causal_margin_weight: float,
    causal_margin: float,
    rng: random.Random,
) -> CausalLoss:
    cut_index = torch.nonzero(
        case.cut_edge_mask,
        as_tuple=False,
    ).flatten().to(device)

    keep_index = choose_keep_indices(
        case,
        clean_keep_edges=clean_keep_edges,
        keep_to_cut_ratio=keep_to_cut_ratio,
        rng=rng,
    ).to(device)

    zero = full.final_edge_logits.sum() * 0.0

    # ------------------------------------------------------------------
    # FULL temporal supervision.
    # ------------------------------------------------------------------
    if cut_index.numel():
        cut_loss = F.binary_cross_entropy_with_logits(
            full.final_edge_logits[cut_index],
            torch.zeros(
                cut_index.numel(),
                device=device,
                dtype=full.final_edge_logits.dtype,
            ),
        )
    else:
        cut_loss = zero

    if keep_index.numel():
        keep_loss = F.binary_cross_entropy_with_logits(
            full.final_edge_logits[keep_index],
            torch.ones(
                keep_index.numel(),
                device=device,
                dtype=full.final_edge_logits.dtype,
            ),
        )
    else:
        keep_loss = zero

    if cut_index.numel() and keep_index.numel():
        full_edge = cut_loss + keep_loss
    elif cut_index.numel():
        full_edge = cut_loss
    else:
        full_edge = keep_loss

    # Auxiliary current-instance split head. Kept deliberately weak; the
    # experiment is decided by actual final RAG partitioning.
    split_index = balanced_split_indices(
        case.instance_split_target,
        rng=rng,
    ).to(device)

    if split_index.numel() and split_loss_weight > 0:
        split_loss = F.binary_cross_entropy_with_logits(
            full.split_logits[split_index],
            case.instance_split_target.to(device)[split_index],
        )
    else:
        split_loss = zero

    # ------------------------------------------------------------------
    # CORRUPTED temporal state must be a spatial no-op.
    # ------------------------------------------------------------------
    eligible_index = torch.nonzero(
        case.eligible_edge_mask,
        as_tuple=False,
    ).flatten().to(device)

    if eligible_index.numel():
        # Smooth-L1 on logits directly targets the frozen spatial decision.
        # Unlike BCE, this says "return to exactly the spatial baseline",
        # not merely "remain on the same side of zero".
        corrupted_noop = F.smooth_l1_loss(
            corrupted.final_edge_logits[eligible_index],
            rag.spatial_edge_logits[eligible_index].detach(),
            beta=0.5,
        )

        # Gate-closing penalty is applied ONLY to corrupted temporal content.
        # Investigation 30 penalized FULL keep gates and eventually encouraged
        # complete gate collapse. We deliberately do not do that here.
        corrupted_gate = (
            corrupted.edge_temporal_gate[eligible_index]
            .square()
            .mean()
        )
    else:
        corrupted_noop = zero
        corrupted_gate = zero

    # ------------------------------------------------------------------
    # Causal contrast on actual manually required CUT edges.
    #
    # Want:
    #   full_logit + margin <= corrupted_logit
    #
    # If FULL and corrupted are identical, this term is positive.
    # ------------------------------------------------------------------
    if cut_index.numel():
        causal_margin_loss = F.relu(
            full.final_edge_logits[cut_index]
            - corrupted.final_edge_logits[cut_index].detach()
            + float(causal_margin)
        ).mean()
    else:
        causal_margin_loss = zero

    total = (
        full_edge
        + float(split_loss_weight) * split_loss
        + float(noop_weight) * corrupted_noop
        + float(corrupted_gate_weight) * corrupted_gate
        + float(causal_margin_weight) * causal_margin_loss
    )

    return CausalLoss(
        total=total,
        full_edge=full_edge,
        full_cut=cut_loss,
        full_keep=keep_loss,
        split=split_loss,
        corrupted_noop=corrupted_noop,
        corrupted_gate=corrupted_gate,
        causal_margin=causal_margin_loss,
        selected_cut_edges=int(cut_index.numel()),
        selected_keep_edges=int(keep_index.numel()),
        corruption=corruption,
    )


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def forward_eval_ablation(
    temporal_model,
    frozen_spatial,
    case,
    temporal_payload: dict[str, Any],
    *,
    device: torch.device,
    dref_um: float,
    ablation: str,
):
    rag, instances = build_spatial_states(
        frozen_spatial,
        case,
        device=device,
        dref_um=dref_um,
    )

    if ablation == "empty":
        temporal = temporal_model.empty_temporal(device)
    else:
        temporal = temporal_model.encode_temporal(
            temporal_payload,
            device=device,
        )

        if ablation == "shuffled":
            temporal = shuffled_temporal_state(
                temporal,
                seed=inv30.FIXED_PROJECTION_SEED + case.t,
            )
        elif ablation == "contentless":
            temporal = contentless_temporal_state(temporal)
        elif ablation != "full":
            raise ValueError(ablation)

    reasoning = reason_with_temporal(
        temporal_model,
        instances,
        rag,
        temporal,
        device=device,
        dref_um=dref_um,
    )
    return rag, reasoning


@torch.no_grad()
def evaluate(
    temporal_model,
    frozen_spatial,
    cases,
    temporal_payloads,
    *,
    device: torch.device,
    dref_um: float,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    temporal_model.eval()
    frozen_spatial.eval()

    results: dict[str, Any] = {}

    for ablation in (
        "full",
        "empty",
        "shuffled",
        "contentless",
    ):
        accumulator = inv30.EvalAccumulator()

        for case, payload in zip(cases, temporal_payloads):
            rag, reasoning = forward_eval_ablation(
                temporal_model,
                frozen_spatial,
                case,
                payload,
                device=device,
                dref_um=dref_um,
                ablation=ablation,
            )

            if ablation == "empty":
                error = float(
                    (
                        reasoning.final_edge_logits
                        - rag.spatial_edge_logits
                    )
                    .abs()
                    .max()
                    .item()
                    if rag.spatial_edge_logits.numel()
                    else 0.0
                )
                accumulator.max_empty_noop_error = max(
                    accumulator.max_empty_noop_error,
                    error,
                )

            predicted = inv30.accumulate_frame_metrics(
                accumulator,
                case,
                reasoning,
            )

            if (
                prediction_dir is not None
                and ablation == "full"
            ):
                prediction_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                labels = inv30.rasterize_prediction(
                    case,
                    predicted,
                )
                np.save(
                    prediction_dir
                    / f"temporal_partition_t{case.t:03d}.npy",
                    labels.astype(np.int32, copy=False),
                    allow_pickle=False,
                )

        results[ablation] = accumulator.as_dict()

    full = results["full"]
    empty = results["empty"]
    shuffled = results["shuffled"]
    contentless = results["contentless"]

    shuffled_cut_gain = (
        full["cut_edge_recall"]
        - shuffled["cut_edge_recall"]
    )
    contentless_cut_gain = (
        full["cut_edge_recall"]
        - contentless["cut_edge_recall"]
    )
    shuffled_exact_gain = (
        full["changed_component_exact_rate"]
        - shuffled["changed_component_exact_rate"]
    )
    contentless_exact_gain = (
        full["changed_component_exact_rate"]
        - contentless["changed_component_exact_rate"]
    )

    min_cut_gain = min(
        shuffled_cut_gain,
        contentless_cut_gain,
    )
    min_exact_gain = min(
        shuffled_exact_gain,
        contentless_exact_gain,
    )

    empty_exact_noop = (
        float(empty["max_empty_noop_error"]) == 0.0
    )

    override_pass = bool(
        empty_exact_noop
        and full["cut_edge_recall"] >= 0.90
        and full["changed_component_exact_rate"] >= 0.75
        and full["unchanged_component_split_rate"] <= 0.02
    )

    # Require FULL to beat BOTH corruptions, not merely one.
    causal_pass = bool(
        (
            min_cut_gain >= 0.10
            or min_exact_gain >= 0.10
        )
        and full["cut_edge_recall"] >= 0.75
    )

    strict_pass = bool(
        override_pass and causal_pass
    )

    # Best-checkpoint score strongly rewards actual split recovery, but static
    # shortcuts lose points because the two causal gains enter explicitly.
    score = (
        3.0 * full["changed_component_exact_rate"]
        + 1.0 * full["cut_edge_recall"]
        + 0.5 * full["keep_edge_accuracy"]
        - 5.0 * full["unchanged_component_split_rate"]
        + 2.0 * min_cut_gain
        + 2.0 * min_exact_gain
    )

    results["verdict"] = {
        "empty_temporal_exact_noop": empty_exact_noop,
        "temporal_override_mechanism": (
            "PASS" if override_pass else "FAIL"
        ),
        "causal_temporal_dependence": (
            "PASS" if causal_pass else "FAIL"
        ),
        "strict_investigation31_pass": strict_pass,
        "full_minus_shuffled_cut_recall": float(
            shuffled_cut_gain
        ),
        "full_minus_contentless_cut_recall": float(
            contentless_cut_gain
        ),
        "full_minus_shuffled_changed_exact_rate": float(
            shuffled_exact_gain
        ),
        "full_minus_contentless_changed_exact_rate": float(
            contentless_exact_gain
        ),
        "minimum_corruption_cut_gain": float(
            min_cut_gain
        ),
        "minimum_corruption_exact_gain": float(
            min_exact_gain
        ),
        "checkpoint_score": float(score),
    }
    return results


def print_eval(
    title: str,
    metrics: dict[str, Any],
) -> None:
    print()
    print("=" * 132)
    print(title)
    print("=" * 132)

    for name in (
        "full",
        "empty",
        "shuffled",
        "contentless",
    ):
        row = metrics[name]
        print(
            f"{name.upper():11s} | "
            f"cut={row['cut_edge_recall']:.4f} "
            f"keep={row['keep_edge_accuracy']:.4f} | "
            f"changed={row['changed_components_exact']}/"
            f"{row['changed_components']} "
            f"({row['changed_component_exact_rate']:.4f}) | "
            f"clean_split="
            f"{row['unchanged_components_accidentally_split']}/"
            f"{row['unchanged_components']} "
            f"({row['unchanged_component_split_rate']:.4f}) | "
            f"gate cut/keep="
            f"{row['mean_cut_gate']:.3f}/"
            f"{row['mean_keep_gate']:.3f}"
        )

    verdict = metrics["verdict"]
    print("-" * 132)
    print(
        f"EMPTY EXACT NO-OP        : "
        f"{'PASS' if verdict['empty_temporal_exact_noop'] else 'FAIL'}"
    )
    print(
        f"OVERRIDE MECHANISM       : "
        f"{verdict['temporal_override_mechanism']}"
    )
    print(
        f"CAUSAL TEMPORAL CONTENT  : "
        f"{verdict['causal_temporal_dependence']}"
    )
    print(
        f"STRICT INVESTIGATION 31  : "
        f"{'PASS' if verdict['strict_investigation31_pass'] else 'FAIL'}"
    )
    print(
        "FULL - SHUFFLED         : "
        f"cut={verdict['full_minus_shuffled_cut_recall']:+.4f}, "
        f"exact={verdict['full_minus_shuffled_changed_exact_rate']:+.4f}"
    )
    print(
        "FULL - CONTENTLESS      : "
        f"cut={verdict['full_minus_contentless_cut_recall']:+.4f}, "
        f"exact={verdict['full_minus_contentless_changed_exact_rate']:+.4f}"
    )
    print(
        f"CHECKPOINT SCORE          : "
        f"{verdict['checkpoint_score']:.5f}"
    )
    print("=" * 132)


# =============================================================================
# OPTIMIZER-BATCH SAMPLING
# =============================================================================


def optimizer_batch_cases(
    cases,
    correction_cases,
    clean_cases,
    *,
    count: int,
    correction_probability: float,
    rng: random.Random,
):
    """Return a diverse microbatch for one optimizer update.

    Whenever possible:
      microbatch[0] = correction frame
      microbatch[1] = clean/no-CUT frame

    Remaining frames follow the requested correction sampling probability.
    """
    if count < 1:
        raise ValueError("accumulate_frames must be >= 1")

    rows = []

    if correction_cases:
        rows.append(rng.choice(correction_cases))

    if len(rows) < count and clean_cases:
        rows.append(rng.choice(clean_cases))

    while len(rows) < count:
        if (
            correction_cases
            and rng.random() < correction_probability
        ):
            rows.append(rng.choice(correction_cases))
        else:
            rows.append(rng.choice(cases))

    return rows[:count]


# =============================================================================
# CHECKPOINTING
# =============================================================================


def checkpoint_payload(
    *,
    temporal_model,
    cfg: ModelConfig,
    step: int,
    dref_um: float,
    spacing: tuple[float, float, float],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    best_score: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "investigation": SCRIPT_NAME,
        "step": int(step),
        "sample_id": args.sample_id,
        "spacing_zyx_um": tuple(float(v) for v in spacing),
        "dref_um": float(dref_um),
        "model_config": dataclasses.asdict(cfg),
        "temporal_model_state_dict": temporal_model.state_dict(),
        "metrics": metrics,
        "best_score": float(best_score),
        "training_args": vars(args),
        "notes": {
            "spatial_model_ran": False,
            "objective": (
                "FULL manual partition supervision + corrupted temporal "
                "no-op + causal CUT margin"
            ),
            "corruptions": list(CORRUPTION_TYPES),
            "full_keep_gate_penalty": False,
        },
    }


def maybe_save_best(
    *,
    output: Path,
    temporal_model,
    cfg: ModelConfig,
    step: int,
    dref_um: float,
    spacing: tuple[float, float, float],
    metrics: dict[str, Any],
    args: argparse.Namespace,
    best_score: float,
    best_is_strict: bool,
) -> tuple[float, bool, bool]:
    verdict = metrics["verdict"]
    score = float(verdict["checkpoint_score"])
    strict = bool(verdict["strict_investigation31_pass"])

    # A strict PASS always outranks a non-strict checkpoint.
    improved = (
        (strict and not best_is_strict)
        or (
            strict == best_is_strict
            and score > best_score
        )
    )

    if not improved:
        return best_score, best_is_strict, False

    payload = checkpoint_payload(
        temporal_model=temporal_model,
        cfg=cfg,
        step=step,
        dref_um=dref_um,
        spacing=spacing,
        metrics=metrics,
        args=args,
        best_score=score,
    )
    atomic_torch_save(output / "best.pt", payload)
    atomic_json(output / "best_metrics.json", metrics)

    return score, strict, True


# =============================================================================
# TRAINING
# =============================================================================


def train(
    *,
    temporal_model,
    frozen_spatial,
    cases,
    temporal_payloads,
    cfg: ModelConfig,
    device: torch.device,
    dref_um: float,
    spacing: tuple[float, float, float],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    correction_cases = [
        case
        for case in cases
        if case.cut_count > 0
    ]
    clean_cases = [
        case
        for case in cases
        if case.cut_count == 0
    ]

    optimizer = torch.optim.AdamW(
        temporal_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        payload = torch_load(resolve(args.resume))
        temporal_model.load_state_dict(
            payload["temporal_model_state_dict"],
            strict=True,
        )
        start_step = int(payload.get("step", 0))
        print(f"[resume] temporal model step={start_step}")

    temporal_model.to(device)
    frozen_spatial.to(device)

    # Sanity: Investigation 30's frozen representation is buffer-only.
    if any(True for _ in frozen_spatial.parameters()):
        raise RuntimeError(
            "Frozen cached-spatial representation unexpectedly has parameters."
        )

    print()
    print("=" * 132)
    print("INVESTIGATION 31 — CAUSAL TEMPORAL OVERFIT")
    print("=" * 132)
    print(f"device                  : {device}")
    print(f"frames                  : {len(cases)}")
    print(f"correction frames       : {len(correction_cases)}")
    print(f"clean frames            : {len(clean_cases)}")
    print(
        f"manual changed comps    : "
        f"{sum(case.changed_component_count for case in cases)}"
    )
    print(
        f"manual CUT edges        : "
        f"{sum(case.cut_count for case in cases)}"
    )
    print(f"optimizer steps         : {args.steps}")
    print(f"frames / optimizer step : {args.accumulate_frames}")
    print(f"learning rate           : {args.lr:g}")
    print(f"no-op weight            : {args.noop_weight:g}")
    print(
        f"corrupted gate weight   : "
        f"{args.corrupted_gate_weight:g}"
    )
    print(
        f"causal margin           : "
        f"{args.causal_margin:g} "
        f"(weight={args.causal_margin_weight:g})"
    )
    print(
        "FULL keep-gate penalty  : NONE "
        "(removed from Investigation 30)"
    )
    print(
        "spatial CNN/watershed/RAG network: NOT INSTANTIATED / NOT RUN"
    )
    print("=" * 132)

    initial = evaluate(
        temporal_model,
        frozen_spatial,
        cases,
        temporal_payloads,
        device=device,
        dref_um=dref_um,
    )
    print_eval("BEFORE INVESTIGATION-31 TRAINING", initial)

    history = []
    rng = random.Random(args.seed + 3100)
    started = time.perf_counter()

    best_score = -float("inf")
    best_is_strict = False

    temporal_model.train()

    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        micro_cases = optimizer_batch_cases(
            cases,
            correction_cases,
            clean_cases,
            count=args.accumulate_frames,
            correction_probability=args.correction_frame_probability,
            rng=rng,
        )

        aggregate = {
            "total": 0.0,
            "full_edge": 0.0,
            "cut": 0.0,
            "keep": 0.0,
            "split": 0.0,
            "noop": 0.0,
            "corrupt_gate": 0.0,
            "margin": 0.0,
            "cut_edges": 0,
            "keep_edges": 0,
        }
        micro_description = []

        for micro_index, case in enumerate(micro_cases):
            corruption = CORRUPTION_TYPES[
                (step + micro_index) % len(CORRUPTION_TYPES)
            ]

            payload = temporal_payloads[case.t]

            rag, _, _, full, corrupted = (
                full_and_corrupted_forward(
                    temporal_model,
                    frozen_spatial,
                    case,
                    payload,
                    device=device,
                    dref_um=dref_um,
                    corruption=corruption,
                    corruption_seed=(
                        args.seed * 1_000_003
                        + step * 97
                        + micro_index * 13
                        + case.t
                    ),
                )
            )

            losses = causal_temporal_loss(
                case,
                rag,
                full,
                corrupted,
                device=device,
                corruption=corruption,
                clean_keep_edges=args.clean_keep_edges,
                keep_to_cut_ratio=args.keep_to_cut_ratio,
                split_loss_weight=args.split_loss_weight,
                noop_weight=args.noop_weight,
                corrupted_gate_weight=args.corrupted_gate_weight,
                causal_margin_weight=args.causal_margin_weight,
                causal_margin=args.causal_margin,
                rng=rng,
            )

            # True gradient accumulation: normalize each micro-loss by the
            # number of frames participating in this optimizer step.
            (
                losses.total
                / float(len(micro_cases))
            ).backward()

            aggregate["total"] += float(
                losses.total.detach()
            )
            aggregate["full_edge"] += float(
                losses.full_edge.detach()
            )
            aggregate["cut"] += float(
                losses.full_cut.detach()
            )
            aggregate["keep"] += float(
                losses.full_keep.detach()
            )
            aggregate["split"] += float(
                losses.split.detach()
            )
            aggregate["noop"] += float(
                losses.corrupted_noop.detach()
            )
            aggregate["corrupt_gate"] += float(
                losses.corrupted_gate.detach()
            )
            aggregate["margin"] += float(
                losses.causal_margin.detach()
            )
            aggregate["cut_edges"] += (
                losses.selected_cut_edges
            )
            aggregate["keep_edges"] += (
                losses.selected_keep_edges
            )
            micro_description.append(
                f"t{case.t:02d}:{corruption[0].upper()}"
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            temporal_model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

        denominator = float(len(micro_cases))
        if (
            step == 1
            or step % args.print_every == 0
        ):
            elapsed = time.perf_counter() - started
            print(
                f"[step {step:05d}/{args.steps}] "
                f"frames={','.join(micro_description)} "
                f"loss={aggregate['total']/denominator:.5f} "
                f"edge={aggregate['full_edge']/denominator:.5f} "
                f"cut={aggregate['cut']/denominator:.5f} "
                f"keep={aggregate['keep']/denominator:.5f} "
                f"noop={aggregate['noop']/denominator:.5f} "
                f"margin={aggregate['margin']/denominator:.5f} "
                f"negGate={aggregate['corrupt_gate']/denominator:.5f} "
                f"grad={float(torch.as_tensor(grad_norm)):.3f} "
                f"elapsed={format_seconds(elapsed)}",
                flush=True,
            )

        if (
            step % args.eval_every == 0
            or step == args.steps
        ):
            metrics = evaluate(
                temporal_model,
                frozen_spatial,
                cases,
                temporal_payloads,
                device=device,
                dref_um=dref_um,
            )
            print_eval(
                f"INVESTIGATION-31 EVALUATION @ STEP {step}",
                metrics,
            )

            history.append(
                {
                    "step": int(step),
                    "metrics": metrics,
                }
            )
            atomic_json(
                output / "training_history.json",
                history,
            )

            current_best_for_payload = max(
                best_score,
                float(
                    metrics["verdict"][
                        "checkpoint_score"
                    ]
                ),
            )
            atomic_torch_save(
                output / "latest.pt",
                checkpoint_payload(
                    temporal_model=temporal_model,
                    cfg=cfg,
                    step=step,
                    dref_um=dref_um,
                    spacing=spacing,
                    metrics=metrics,
                    args=args,
                    best_score=current_best_for_payload,
                ),
            )

            (
                best_score,
                best_is_strict,
                saved,
            ) = maybe_save_best(
                output=output,
                temporal_model=temporal_model,
                cfg=cfg,
                step=step,
                dref_um=dref_um,
                spacing=spacing,
                metrics=metrics,
                args=args,
                best_score=best_score,
                best_is_strict=best_is_strict,
            )
            if saved:
                print(
                    f"[best] saved step={step} "
                    f"score={best_score:.5f} "
                    f"strict={best_is_strict}"
                )

            temporal_model.train()

    # ------------------------------------------------------------------
    # Final checkpoint.
    # ------------------------------------------------------------------
    final_metrics = evaluate(
        temporal_model,
        frozen_spatial,
        cases,
        temporal_payloads,
        device=device,
        dref_um=dref_um,
        prediction_dir=output / "predictions",
    )
    print_eval(
        "FINAL INVESTIGATION-31 CHECKPOINT",
        final_metrics,
    )

    atomic_json(
        output / "final_metrics.json",
        final_metrics,
    )
    atomic_torch_save(
        output / "final.pt",
        checkpoint_payload(
            temporal_model=temporal_model,
            cfg=cfg,
            step=args.steps,
            dref_um=dref_um,
            spacing=spacing,
            metrics=final_metrics,
            args=args,
            best_score=best_score,
        ),
    )

    return final_metrics


# =============================================================================
# PREPARATION — REUSE INVESTIGATION 30
# =============================================================================


def make_inv30_data_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the argument namespace expected by Investigation 30 helpers."""

    # Investigation-30 output is intentionally the data-cache location.
    inv30_output = (
        resolve(args.inv30_cache)
        if args.inv30_cache is not None
        else (
            ROOT
            / "runs"
            / "stirnet"
            / "evaluation"
            / inv30.SCRIPT_NAME
            / args.sample_id
        ).resolve()
    )

    return argparse.Namespace(
        sample_id=args.sample_id,
        frame_count=args.frame_count,
        inv12=args.inv12,
        inv24=args.inv24,
        inv25=args.inv25,
        annotations=args.annotations,
        zarr=args.zarr,
        output=inv30_output,
        spacing_zyx_um=args.spacing_zyx_um,
        dref_um=args.dref_um,
        temporal_radius=args.temporal_radius,
        trackastra_model=args.trackastra_model,
        trackastra_mode=args.trackastra_mode,
        trackastra_device=args.trackastra_device,
        rebuild_trackastra=args.rebuild_trackastra,
        rebuild_temporal=args.rebuild_temporal,
        rebuild_movies=args.rebuild_movies,
        complete_candidate_graph=args.complete_candidate_graph,
        spatial_prior_logit=args.spatial_prior_logit,
    )


def prepare_data(
    args: argparse.Namespace,
):
    data_args = make_inv30_data_args(args)
    paths = inv30.make_paths(data_args)

    inv30.validate_required_artifacts(
        paths,
        args.frame_count,
    )

    spacing = tuple(
        float(value)
        for value in args.spacing_zyx_um
    )

    print()
    print("=" * 118)
    print("INVESTIGATION 31 — FROZEN ARTIFACT / TEMPORAL CACHE PREPARATION")
    print("=" * 118)
    print(f"Investigation-30 cache root: {paths.output}")
    print(f"Investigation-31 output    : {args.output_resolved}")
    print("=" * 118)

    cases = [
        inv30.load_spatial_case(
            paths,
            t,
            spacing,
            args.spatial_prior_logit,
        )
        for t in range(args.frame_count)
    ]

    audit = inv30.audit_cases(cases)
    atomic_json(
        args.output_resolved / "annotation_rag_audit.json",
        audit,
    )

    base_movie_path = inv30.assemble_base_instance_movie(
        paths,
        args.frame_count,
        rebuild=args.rebuild_movies,
    )
    base_movie = np.load(
        base_movie_path,
        mmap_mode="r",
    )

    raw_movie_path = inv30.assemble_raw_movie(
        paths,
        args.frame_count,
        cases[0].shape_zyx,
        rebuild=args.rebuild_movies,
    )
    raw_movie = np.load(
        raw_movie_path,
        mmap_mode="r",
    )

    dref_um = (
        float(args.dref_um)
        if args.dref_um is not None
        else inv30.resolve_movie_dref_um(
            base_movie,
            spacing,
        )
    )

    # Reuse Trackastra from Investigation 30 whenever available.
    graph_path, tracked_masks_path = inv30.prepare_trackastra(
        paths,
        raw_movie_path,
        base_movie_path,
        model_name=args.trackastra_model,
        mode=args.trackastra_mode,
        device=args.trackastra_device,
        rebuild=args.rebuild_trackastra,
    )

    with graph_path.open("rb") as handle:
        track_graph = pickle.load(handle)

    tracked_movie = np.load(
        tracked_masks_path,
        mmap_mode="r",
    )

    inv30.build_temporal_caches(
        paths,
        track_graph=track_graph,
        tracked_movie=tracked_movie,
        raw_movie=raw_movie,
        frame_count=args.frame_count,
        spacing=spacing,
        dref_um=dref_um,
        temporal_radius=args.temporal_radius,
        complete_candidate_graph=args.complete_candidate_graph,
        rebuild=args.rebuild_temporal,
    )

    temporal_payloads = inv30.load_all_temporal_payloads(
        paths,
        args.frame_count,
    )

    return (
        paths,
        cases,
        temporal_payloads,
        spacing,
        dref_um,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Investigation 31: causally supervised temporal-only overfit."
        )
    )

    parser.add_argument(
        "--sample-id",
        default=DEFAULT_SAMPLE_ID,
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
    )

    parser.add_argument("--inv12", type=Path, default=None)
    parser.add_argument("--inv24", type=Path, default=None)
    parser.add_argument("--inv25", type=Path, default=None)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
    )
    parser.add_argument("--zarr", type=Path, default=None)

    parser.add_argument(
        "--inv30-cache",
        type=Path,
        default=None,
        help=(
            "Investigation-30 sample output containing raw/base movies, "
            "Trackastra and temporal_v4 caches. Default resolves automatically."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--spacing-zyx-um",
        type=float,
        nargs=3,
        default=DEFAULT_SPACING_ZYX_UM,
    )
    parser.add_argument(
        "--dref-um",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=DEFAULT_TEMPORAL_RADIUS,
    )

    parser.add_argument(
        "--trackastra-model",
        default="ctc",
    )
    parser.add_argument(
        "--trackastra-mode",
        default="greedy",
    )
    parser.add_argument(
        "--trackastra-device",
        default="cuda",
    )
    parser.add_argument(
        "--rebuild-trackastra",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-temporal",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-movies",
        action="store_true",
    )
    parser.add_argument(
        "--complete-candidate-graph",
        action="store_true",
    )

    parser.add_argument(
        "--spatial-prior-logit",
        type=float,
        default=inv30.DEFAULT_SPATIAL_PRIOR_LOGIT,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
    )
    parser.add_argument(
        "--accumulate-frames",
        type=int,
        default=DEFAULT_ACCUMULATE_FRAMES,
    )
    parser.add_argument(
        "--correction-frame-probability",
        type=float,
        default=DEFAULT_CORRECTION_FRAME_PROB,
    )
    parser.add_argument(
        "--clean-keep-edges",
        type=int,
        default=DEFAULT_CLEAN_KEEP_EDGES,
    )
    parser.add_argument(
        "--keep-to-cut-ratio",
        type=int,
        default=DEFAULT_KEEP_TO_CUT_RATIO,
    )
    parser.add_argument(
        "--split-loss-weight",
        type=float,
        default=DEFAULT_SPLIT_LOSS_WEIGHT,
    )

    parser.add_argument(
        "--noop-weight",
        type=float,
        default=DEFAULT_NOOP_WEIGHT,
    )
    parser.add_argument(
        "--corrupted-gate-weight",
        type=float,
        default=DEFAULT_CORRUPTED_GATE_WEIGHT,
    )
    parser.add_argument(
        "--causal-margin-weight",
        type=float,
        default=DEFAULT_CAUSAL_MARGIN_WEIGHT,
    )
    parser.add_argument(
        "--causal-margin",
        type=float,
        default=DEFAULT_CAUSAL_MARGIN,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=31,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = parser.parse_args()

    args.output_resolved = (
        resolve(args.output)
        if args.output is not None
        else (
            ROOT
            / "runs"
            / "stirnet"
            / "evaluation"
            / SCRIPT_NAME
            / args.sample_id
        ).resolve()
    )

    return args


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if TEMPORAL_CACHE_CONTRACT_VERSION < 4:
        raise RuntimeError(
            "Investigation 31 requires temporal cache contract v4+."
        )
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.accumulate_frames < 1:
        raise ValueError("--accumulate-frames must be >= 1")
    if args.eval_every < 1 or args.print_every < 1:
        raise ValueError("eval/print cadence must be positive")
    if not 0.0 <= args.correction_frame_probability <= 1.0:
        raise ValueError(
            "--correction-frame-probability must be in [0,1]"
        )
    for name in (
        "noop_weight",
        "corrupted_gate_weight",
        "causal_margin_weight",
        "causal_margin",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")

    args.output_resolved.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 132)
    print("INVESTIGATION 31 — BIOHUB CAUSAL TEMPORAL OVERFIT")
    print("=" * 132)
    print(f"repository : {ROOT}")
    print(f"sample     : {args.sample_id}")
    print(f"output     : {args.output_resolved}")
    print(
        f"cache contract: temporal-v{TEMPORAL_CACHE_CONTRACT_VERSION}"
    )
    print("=" * 132)

    (
        data_paths,
        cases,
        temporal_payloads,
        spacing,
        dref_um,
    ) = prepare_data(args)

    print(f"[scale] dref = {dref_um:.5f} um")

    if args.prepare_only:
        print("Preparation complete (--prepare-only).")
        return

    cfg = ModelConfig()

    # Deterministic overfit setup.
    cfg.history.dropout = 0.0
    cfg.history.activation_checkpointing = False
    cfg.temporal.dropout = 0.0
    cfg.instances.dropout = 0.0
    cfg.validate()

    temporal_model = inv30.TemporalOnlyModel(cfg)
    frozen_spatial = inv30.FrozenCachedSpatialRepresentation(cfg)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    final_metrics = train(
        temporal_model=temporal_model,
        frozen_spatial=frozen_spatial,
        cases=cases,
        temporal_payloads=temporal_payloads,
        cfg=cfg,
        device=device,
        dref_um=dref_um,
        spacing=spacing,
        output=args.output_resolved,
        args=args,
    )

    print()
    print("=" * 132)
    print("INVESTIGATION 31 COMPLETE")
    print("=" * 132)
    print(f"latest : {args.output_resolved / 'latest.pt'}")
    print(f"best   : {args.output_resolved / 'best.pt'}")
    print(f"final  : {args.output_resolved / 'final.pt'}")
    print(
        f"best metrics : "
        f"{args.output_resolved / 'best_metrics.json'}"
    )
    print(
        f"final metrics: "
        f"{args.output_resolved / 'final_metrics.json'}"
    )
    print("-" * 132)

    verdict = final_metrics["verdict"]
    print(
        "FINAL checkpoint: "
        f"override={verdict['temporal_override_mechanism']}, "
        f"causal={verdict['causal_temporal_dependence']}, "
        f"strict={verdict['strict_investigation31_pass']}"
    )
    print()
    print(
        "Use best.pt / best_metrics.json for the scientific conclusion; "
        "the final optimizer step is not automatically the best checkpoint."
    )
    print("=" * 132)


if __name__ == "__main__":
    main()
