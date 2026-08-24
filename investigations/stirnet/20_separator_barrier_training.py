from __future__ import annotations

r"""
STIR-Net Investigation 20 v2 — separator-aware barrier training.

Purpose
-------
Prototype a separator-aware late RAG barrier BEFORE permanently modifying the
production STIR-Net source files.

The current production model already provides:
    * the legacy 8-D RAG edge statistics;
    * contact-aware local+broad morphology embeddings;
    * explicit A/B/union/interface/proximity topology inside the morphology CNN.

The missing experiment is a late, identifiable separator pathway.  This script
therefore leaves learned/stirnet untouched and patches the *training computation*
as:

    base_merge_logit = current SpatialRAGNetwork logit

    explicit separator/contact features
              +
    current edge morphology embedding
              |
              v
        SeparatorBarrier
              |
              v
        nonnegative barrier

    final_merge_logit = base_merge_logit - barrier

The barrier is mathematically separation-only: it can lower merge confidence,
but can never increase it.

V2 also gives the barrier a direct balanced semantic loss:
    ON  = GT-different-cell edge with strong separator coverage;
    OFF = every GT-same-cell edge;
    weak-separator GT-negative edges are ignored by this auxiliary loss.

This prevents already-correct low-p_merge negatives from starving the barrier
of learning signal.

Live near-zero initialization (v2 fix)
--------------------------------------
Investigation-20 v1 used an exact-zero clamped-softplus barrier. Positive-edge
updates could push its raw value below zero, after which the clamp made both the
barrier and its gradient exactly zero. The observed logs showed that failure.

V2 learns a directly supervised barrier score and maps it to the actual
correction with:

    gate_logit = -8 + score_scale * barrier_score
    barrier = max_barrier_logit * sigmoid(gate_logit)

The score starts near zero and receives full BCE gradients, while the -8 gate
bias keeps the initial correction around ~0.00268 logit. This preserves
production decisions without starving optimization.

Explicit barrier features
-------------------------
Features are measured on the *exact six-neighbour A<->B contact faces* and are
weighted by physical face area, so anisotropic Z/Y/X spacing is respected:

    0  area-weighted separator mean
    1  separator maximum
    2  area-weighted separator standard deviation
    3  physical contact coverage with separator >= 0.50
    4  physical contact coverage with separator >= 0.70
    5  physical contact coverage with separator >= 0.85
    6  physical contact coverage with separator >= 0.95
    7  area-weighted surface mean
    8  surface maximum
    9  area-weighted separator*surface mean
   10  area-weighted |SDF| mean / dref
   11  log(1 + physical_contact_area / dref^2)

The barrier network sees these 12 explicit values *and* the existing 64-D edge
morphology embedding.  Separator identity therefore survives until the final
edge reasoning stage instead of being hidden only inside a generic CNN latent.

Training policy
---------------
Dense geometry and the legacy RAG are always frozen.

--train-scope auto (default):
    * if the starting checkpoint already contains all current morphology-v2
      tensors, train only SeparatorBarrier;
    * if only the current edge-morphology-v2 tensors are new/incomplete, train
      edge morphology + edge residual projection + SeparatorBarrier;
    * if morphology is absent more broadly, train morphology + residual
      projections + SeparatorBarrier.

This makes the script safe both for a current morphology-v2 checkpoint and for
an older morphology milestone.

If a transferred checkpoint does not contain the current edge morphology
implementation, the edge morphology projection is reset to ZERO before
training.  Random/new edge embeddings can therefore not corrupt the inherited
RAG logits at step 0.

Data/crop semantics
-------------------
The script intentionally reuses Investigation 17's data/crop helpers, which were
built to match the production data semantics of:

    experiments/stirnet/training/01_nis3d_spatial_training.py

This preserves:
    * NIS3D ConfidenceScore validity handling;
    * the production raw-source preprocessing/cache;
    * Drosophila effective spacing override;
    * merge-aware model-independent crop manifests;
    * fixed held-out validation;
    * no source-dropout and no XY flip during this targeted RAG experiment.

The support script must exist at:
    investigations/stirnet/17_morphology_rag_multicrop_training.py

Resume semantics
----------------
Investigation 20 always starts from a checkpoint and therefore requires:

    --resume --checkpoint <path>

Two cases are handled automatically:

1. A pre-Investigation-20 checkpoint:
       compatible STIR-Net tensors are transferred;
       new barrier/morphology-v2 tensors are initialized as described above;
       Investigation-20 optimizer step starts at 0.

2. An Investigation-20 checkpoint:
       model, barrier, optimizer, scaler, baseline validation and experiment
       step are restored exactly.

The checkpoint argument may point either to a .pt file or to a directory
containing best_checkpoint.pt / checkpoint_step_*.pt.

Recommended smoke
-----------------
From the repository root on the local GPU:

    python .\investigations\stirnet\20_separator_barrier_training.py `
        --resume `
        --checkpoint .\runs\stirnet\milestones\drosophila_12_morphology_rag_multicrop_v1 `
        --max-steps 4 `
        --validation-every 2 `
        --validation-crops-per-sample 2 `
        --run-name separator_barrier_smoke

Recommended first real run:

    python .\investigations\stirnet\20_separator_barrier_training.py `
        --resume `
        --checkpoint .\runs\stirnet\milestones\drosophila_12_morphology_rag_multicrop_v1 `
        --max-steps 600 `
        --run-name drosophila_12_separator_barrier_v1

This experiment intentionally does NOT implement the original-mask mutex guard.
That should be evaluated only after the separator barrier has been trained and
validated.
"""

import argparse
import copy
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


EXPERIMENT_NAME = "20_separator_barrier_training"
BARRIER_VERSION = 2
BARRIER_PARAMETERIZATION = "scaled_sigmoid_score_v2"
BARRIER_FEATURE_DIM = 12
DEFAULT_SPACING_XYZ = "0.20312639,0.20312639,0.79099447"
DEFAULT_SAMPLES = "Drosophila_1,Drosophila_2"


# ======================================================================================
# Repository / support loading
# ======================================================================================

def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "learned").is_dir() and (candidate / "src").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    raise RuntimeError(
        "Could not resolve repository root. Run this file from inside the "
        "cell-tracking repository."
    )


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_inv17_support():
    """Load the already-tested NIS3D/crop helpers from Investigation 17."""
    path = (
        REPO_ROOT
        / "investigations"
        / "stirnet"
        / "17_morphology_rag_multicrop_training.py"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Investigation-17 support script is required but missing: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "_stirnet_inv17_support_for_inv20",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load support module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# Small helpers
# ======================================================================================

def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _torch_load(path: Path, *, map_location="cpu") -> dict:
    import torch

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_context(amp_dtype: str):
    import torch

    if amp_dtype == "bf16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    if amp_dtype == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    from contextlib import nullcontext
    return nullcontext()


def _hydrate_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    """Hydrate only fields that still exist in the current repository config."""
    if not is_dataclass(instance):
        raise TypeError("Expected dataclass instance")

    names = {field.name for field in fields(instance)}
    for key, value in payload.items():
        if key not in names:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _hydrate_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, (tuple, list)):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def _resolve_checkpoint(value: str) -> Path:
    if not value.strip():
        raise ValueError(
            "Investigation 20 requires --checkpoint together with --resume"
        )

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()

    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    best = path / "best_checkpoint.pt"
    if best.is_file():
        return best

    checkpoints = sorted(path.glob("checkpoint_step_*.pt"))
    if checkpoints:
        return checkpoints[-1]

    # Permit a milestone directory with one nested recovery stream.
    nested_best = sorted(path.glob("**/best_checkpoint.pt"))
    if nested_best:
        return nested_best[-1]
    nested = sorted(path.glob("**/checkpoint_step_*.pt"))
    if nested:
        return nested[-1]

    raise FileNotFoundError(
        f"No checkpoint found inside directory: {path}"
    )


def _gradient_norm(parameters) -> float:
    import torch

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(
            torch.linalg.vector_norm(
                parameter.grad.detach().float()
            ).cpu()
        )
        total = math.hypot(total, value)
    return total


# ======================================================================================
# Separator-aware late barrier
# ======================================================================================

class SeparatorAwareBarrier:
    """Late separation-only barrier with an always-differentiable gate.

    Investigation-20 v1 used::

        clamp_min(softplus(raw) - log(2), 0)

    That preserved exact zero at transfer initialization, but once optimization
    pushed ``raw`` below zero the clamp could make the barrier and its gradient
    exactly zero. Positive/same-cell edges naturally exert that downward
    pressure, while already-correct negative edges provide only a weak final-BCE
    gradient. The barrier therefore died before learning separator semantics.

    V2 learns a directly supervised ``barrier_score`` and maps it to a bounded
    separation-only correction::

        gate_logit = initial_gate_bias + score_scale * barrier_score
        barrier = max_barrier_logit * sigmoid(gate_logit)

    ``barrier_score`` starts near zero and receives ordinary BCE gradients,
    while the default gate bias -8 keeps the actual initial barrier negligible.
    This decouples optimization strength from the near-zero inherited correction.
    """

    @staticmethod
    def build(
        *,
        morphology_dim: int,
        hidden_dim: int = 64,
        max_barrier_logit: float = 8.0,
        initial_gate_bias: float = -8.0,
        barrier_score_scale: float = 4.0,
    ):
        import torch
        from torch import nn

        if max_barrier_logit <= 0:
            raise ValueError("max_barrier_logit must be > 0")

        class _Barrier(nn.Module):
            def __init__(self):
                super().__init__()
                self.explicit_norm = nn.LayerNorm(BARRIER_FEATURE_DIM)
                self.explicit_encoder = nn.Sequential(
                    nn.Linear(BARRIER_FEATURE_DIM, 32),
                    nn.SiLU(),
                    nn.Linear(32, 32),
                    nn.SiLU(),
                )
                self.morphology_norm = nn.LayerNorm(morphology_dim)
                self.morphology_encoder = nn.Sequential(
                    nn.Linear(morphology_dim, 32),
                    nn.SiLU(),
                    nn.Linear(32, 32),
                    nn.SiLU(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(64, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 32),
                    nn.SiLU(),
                )
                self.final = nn.Linear(32, 1)
                # Tiny random weights preserve an almost-constant initial
                # barrier while allowing auxiliary gradients to reach the
                # barrier encoders on the very first optimizer step.
                nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.final.bias)
                self.max_barrier_logit = float(max_barrier_logit)
                self.initial_gate_bias = float(initial_gate_bias)
                self.barrier_score_scale = float(barrier_score_scale)

            def forward(self, explicit_features, edge_morphology):
                explicit = explicit_features.float()
                morphology = edge_morphology.float()
                explicit = self.explicit_encoder(self.explicit_norm(explicit))
                morphology = self.morphology_encoder(self.morphology_norm(morphology))
                hidden = self.fusion(torch.cat([explicit, morphology], dim=-1))
                # ``barrier_score`` is the directly supervised ON/OFF score.
                # The strongly negative gate bias suppresses the actual barrier
                # at initialization without suppressing the auxiliary gradient.
                barrier_score = self.final(hidden).squeeze(-1).float()
                gate_logit = (
                    self.initial_gate_bias
                    + self.barrier_score_scale * barrier_score
                )
                gate_probability = torch.sigmoid(gate_logit)
                barrier = self.max_barrier_logit * gate_probability
                return barrier, barrier_score, gate_logit

        return _Barrier()


# ======================================================================================
# Exact physical A<->B separator/contact features
# ======================================================================================

def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _separator_contact_features(
    *,
    rag,
    geometry,
    spacing_um,
    dref_um,
):
    """Return [E,12] exact contact-face barrier features.

    Geometry is intentionally detached/frozen.  Each six-neighbour contact face
    contributes with its physical face area, so a Z-normal face and an X-normal
    face are not treated as equal when acquisition spacing is anisotropic.
    """
    import torch

    from learned.stirnet.model.types import (
        geometry_field,
        geometry_probability,
    )

    edge_count = int(rag.edge_index.shape[1])
    if edge_count == 0:
        return rag.node_features.new_zeros(
            (0, BARRIER_FEATURE_DIM),
            dtype=torch.float32,
        )

    with torch.no_grad():
        separator_all = geometry_probability(
            geometry,
            "separator",
        )[:, 0].float()
        surface_all = geometry_probability(
            geometry,
            "surface",
        )[:, 0].float()
        sdf_all = geometry_field(
            geometry,
            "sdf",
        )[:, 0].float().abs()

        result = torch.zeros(
            (edge_count, BARRIER_FEATURE_DIM),
            device=rag.edge_index.device,
            dtype=torch.float32,
        )

        for edge_row in range(edge_count):
            batch_index = int(
                rag.edge_batch[edge_row].item()
            )
            ga = int(rag.edge_index[0, edge_row].item())
            gb = int(rag.edge_index[1, edge_row].item())

            label_a = int(
                rag.node_supervoxel_id[ga].item()
            )
            label_b = int(
                rag.node_supervoxel_id[gb].item()
            )
            labels = rag.supervoxel_labels[batch_index]

            stats = (
                None
                if rag.statistics is None
                else rag.statistics[batch_index]
            )
            if (
                stats is not None
                and label_a - 1 < stats.min_voxel.shape[0]
                and label_b - 1 < stats.min_voxel.shape[0]
            ):
                lower = torch.minimum(
                    stats.min_voxel[label_a - 1],
                    stats.min_voxel[label_b - 1],
                ).long() - 1
                upper = torch.maximum(
                    stats.max_voxel[label_a - 1],
                    stats.max_voxel[label_b - 1],
                ).long() + 2
                lower = lower.clamp_min(0)
                upper = torch.minimum(
                    upper,
                    torch.as_tensor(
                        labels.shape,
                        device=labels.device,
                        dtype=torch.long,
                    ),
                )
                bbox = tuple(
                    slice(int(lo), int(hi))
                    for lo, hi in zip(
                        lower.tolist(),
                        upper.tolist(),
                    )
                )
            else:
                # Crop-sized fallback.  This is slower but exact.
                bbox = tuple(
                    slice(0, int(v))
                    for v in labels.shape
                )

            local_labels = labels[bbox]
            sep = separator_all[
                batch_index,
                bbox[0],
                bbox[1],
                bbox[2],
            ]
            surf = surface_all[
                batch_index,
                bbox[0],
                bbox[1],
                bbox[2],
            ]
            sdf_abs = sdf_all[
                batch_index,
                bbox[0],
                bbox[1],
                bbox[2],
            ]

            spacing = spacing_um[batch_index].float()
            dref = dref_um[batch_index].float().clamp_min(
                1e-6
            )

            sep_rows = []
            surf_rows = []
            sdf_rows = []
            weight_rows = []

            for axis in range(3):
                left_slice = [slice(None)] * 3
                right_slice = [slice(None)] * 3
                left_slice[axis] = slice(0, -1)
                right_slice[axis] = slice(1, None)
                left_slice = tuple(left_slice)
                right_slice = tuple(right_slice)

                left = local_labels[left_slice]
                right = local_labels[right_slice]
                touching = (
                    ((left == label_a) & (right == label_b))
                    | ((left == label_b) & (right == label_a))
                )
                if not bool(touching.any()):
                    continue

                pair_sep = 0.5 * (
                    sep[left_slice][touching]
                    + sep[right_slice][touching]
                )
                pair_surf = 0.5 * (
                    surf[left_slice][touching]
                    + surf[right_slice][touching]
                )
                pair_sdf = 0.5 * (
                    sdf_abs[left_slice][touching]
                    + sdf_abs[right_slice][touching]
                )

                other_axes = [i for i in range(3) if i != axis]
                face_area = (
                    spacing[other_axes[0]]
                    * spacing[other_axes[1]]
                )
                weights = torch.full_like(
                    pair_sep,
                    float(face_area.item()),
                    dtype=torch.float32,
                )

                sep_rows.append(pair_sep.float())
                surf_rows.append(pair_surf.float())
                sdf_rows.append(pair_sdf.float())
                weight_rows.append(weights)

            if not sep_rows:
                # A RAG edge should always correspond to direct adjacency.
                # Keep a deterministic zero row instead of inventing evidence.
                continue

            sep_values = torch.cat(sep_rows)
            surf_values = torch.cat(surf_rows)
            sdf_values = torch.cat(sdf_rows)
            weights = torch.cat(weight_rows)
            total_area = weights.sum().clamp_min(1e-8)

            sep_mean = _weighted_mean(
                sep_values,
                weights,
            )
            sep_max = sep_values.max()
            sep_var = _weighted_mean(
                (sep_values - sep_mean).square(),
                weights,
            )
            sep_std = sep_var.clamp_min(0).sqrt()

            coverage = [
                (
                    weights[sep_values >= threshold].sum()
                    / total_area
                )
                for threshold in (0.50, 0.70, 0.85, 0.95)
            ]

            surface_mean = _weighted_mean(
                surf_values,
                weights,
            )
            surface_max = surf_values.max()
            separator_surface_mean = _weighted_mean(
                sep_values * surf_values,
                weights,
            )
            sdf_mean_normalized = (
                _weighted_mean(
                    sdf_values,
                    weights,
                )
                / dref
            )
            normalized_area = torch.log1p(
                total_area / (dref * dref)
            )

            result[edge_row] = torch.stack(
                [
                    sep_mean,
                    sep_max,
                    sep_std,
                    *coverage,
                    surface_mean,
                    surface_max,
                    separator_surface_mean,
                    sdf_mean_normalized,
                    normalized_area,
                ]
            )

        return result


# ======================================================================================
# Model transfer / resume
# ======================================================================================

def _is_inv20_checkpoint(payload: dict) -> bool:
    """True only for checkpoints produced by the fixed v2 barrier script."""
    extra = payload.get("extra", {})
    barrier_cfg = payload.get("separator_barrier_config", {})
    return (
        isinstance(extra, dict)
        and extra.get("experiment") == EXPERIMENT_NAME
        and "separator_barrier" in payload
        and isinstance(barrier_cfg, dict)
        and int(barrier_cfg.get("version", 0)) == BARRIER_VERSION
        and barrier_cfg.get("parameterization") == BARRIER_PARAMETERIZATION
    )


def _is_legacy_inv20_checkpoint(payload: dict) -> bool:
    extra = payload.get("extra", {})
    return (
        isinstance(extra, dict)
        and extra.get("experiment") == EXPERIMENT_NAME
        and "separator_barrier" in payload
        and not _is_inv20_checkpoint(payload)
    )


def _build_current_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
):
    """Hydrate current repo architecture and transfer all shape-compatible tensors."""
    import torch

    from learned.stirnet import StirNet, StirNetConfig

    payload = _torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    model_cfg = StirNetConfig()
    checkpoint_cfg = payload.get("model_config")
    if isinstance(checkpoint_cfg, dict):
        _hydrate_dataclass(
            model_cfg,
            checkpoint_cfg,
        )

    # The current contact-aware morphology branch is required by the barrier.
    model_cfg.partition.rag_morphology_enabled = True
    model_cfg.partition.rag_morphology_detach_geometry = True
    model_cfg.validate()

    model = StirNet(model_cfg)
    historical = payload.get("model", {})
    if not isinstance(historical, dict):
        raise ValueError(
            f"Checkpoint does not contain a model state_dict: {checkpoint_path}"
        )

    current = model.state_dict()
    compatible = {}
    mismatched = {}
    for name, value in historical.items():
        if name not in current:
            continue
        if tuple(current[name].shape) == tuple(value.shape):
            compatible[name] = value
        else:
            mismatched[name] = {
                "checkpoint": tuple(value.shape),
                "current": tuple(current[name].shape),
            }

    model.load_state_dict(
        compatible,
        strict=False,
    )

    missing = sorted(set(current) - set(compatible))
    core_missing = [
        name
        for name in missing
        if (
            "morphology_builder" not in name
            and "morphology_projection" not in name
        )
    ]
    if core_missing:
        preview = "\n".join(
            f"  - {name}"
            for name in core_missing[:25]
        )
        raise RuntimeError(
            "Checkpoint transfer omitted non-morphology model tensors. "
            "This experiment refuses to silently change the inherited dense/"
            "legacy-RAG architecture:\n"
            + preview
        )

    node_morph_missing = any(
        (
            "rag_builder.morphology_builder.node_encoder"
            in name
        )
        for name in missing
    )
    edge_morph_missing = any(
        (
            "rag_builder.morphology_builder.edge_encoder"
            in name
            or "rag_builder.morphology_builder.edge_scale_fusion"
            in name
        )
        for name in missing
    )

    # If a morphology encoder is new/partially incompatible, a previously
    # trained non-zero residual projection must not multiply random embeddings.
    # Reset only the affected residual projection(s), restoring a safe inherited
    # baseline at transfer initialization.
    reset = []
    if node_morph_missing:
        module = model.rag_network.node_morphology_projection
        if module is not None:
            torch.nn.init.zeros_(module.weight)
            reset.append(
                "rag_network.node_morphology_projection.weight"
            )

    if edge_morph_missing:
        module = model.rag_network.edge_morphology_projection
        if module is not None:
            torch.nn.init.zeros_(module.weight)
            reset.append(
                "rag_network.edge_morphology_projection.weight"
            )

    model = model.to(device)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(
            payload.get("global_step", -1)
        ),
        "historical_parameter_tensors": len(historical),
        "transferred_parameter_tensors": len(compatible),
        "missing_current_tensors": missing,
        "mismatched_tensors": mismatched,
        "node_morphology_incomplete": bool(node_morph_missing),
        "edge_morphology_incomplete": bool(edge_morph_missing),
        "reset_projection_tensors": reset,
        "exact_inv20_checkpoint": _is_inv20_checkpoint(
            payload
        ),
        "legacy_inv20_checkpoint": _is_legacy_inv20_checkpoint(
            payload
        ),
    }
    return model, model_cfg, payload, report


def _resolve_train_scope(
    requested: str,
    transfer_report: dict,
) -> str:
    if requested != "auto":
        return requested
    if transfer_report["node_morphology_incomplete"]:
        return "morphology+barrier"
    if transfer_report["edge_morphology_incomplete"]:
        return "edge+barrier"
    return "barrier"


def _configure_trainability(
    model,
    barrier,
    *,
    scope: str,
):
    """Freeze dense/legacy RAG.  Only requested morphology pieces + barrier train."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in barrier.parameters():
        parameter.requires_grad_(True)

    morphology = model.rag_builder.morphology_builder
    if morphology is None:
        raise RuntimeError(
            "Morphology-enabled current model has no morphology_builder"
        )

    if scope == "edge+barrier":
        for parameter in morphology.edge_encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in morphology.edge_scale_fusion.parameters():
            parameter.requires_grad_(True)
        edge_projection = (
            model.rag_network.edge_morphology_projection
        )
        if edge_projection is None:
            raise RuntimeError(
                "edge_morphology_projection is missing"
            )
        for parameter in edge_projection.parameters():
            parameter.requires_grad_(True)

    elif scope == "morphology+barrier":
        for parameter in morphology.parameters():
            parameter.requires_grad_(True)
        for module in (
            model.rag_network.node_morphology_projection,
            model.rag_network.edge_morphology_projection,
        ):
            if module is None:
                raise RuntimeError(
                    "Morphology residual projection is missing"
                )
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    elif scope != "barrier":
        raise ValueError(
            f"Unsupported train scope: {scope}"
        )

    # eval() makes the frozen legacy dropout deterministic while preserving
    # gradients through explicitly trainable modules.
    model.eval()
    barrier.train()
    if scope == "edge+barrier":
        morphology.edge_encoder.train()
        morphology.edge_scale_fusion.train()
        model.rag_network.edge_morphology_projection.train()
    elif scope == "morphology+barrier":
        morphology.train()
        model.rag_network.node_morphology_projection.train()
        model.rag_network.edge_morphology_projection.train()

    barrier_parameters = [
        p for p in barrier.parameters() if p.requires_grad
    ]
    morphology_parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    return barrier_parameters, morphology_parameters


def _trainable_breakdown(
    model,
    barrier,
) -> dict[str, int]:
    return {
        "separator_barrier": sum(
            p.numel()
            for p in barrier.parameters()
            if p.requires_grad
        ),
        "model_trainable": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "legacy_rag_trainable": sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
            and (
                "morphology" not in name
            )
        ),
        "total_trainable": (
            sum(
                p.numel()
                for p in barrier.parameters()
                if p.requires_grad
            )
            + sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
        ),
    }


# ======================================================================================
# Edge selection / loss
# ======================================================================================

def _spread_indices(indices, count: int):
    import torch

    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        count,
        device=indices.device,
    ).round().long()
    return indices[positions]


def _select_training_edges(
    *,
    targets,
    base_logits,
    barrier_features,
    max_edges_per_class: int,
    separator_hard_fraction: float,
    hard_negative_probability: float,
    strong_separator_coverage: float,
):
    """Balanced selection with quota reserved for separator-backed negatives.

    V1 required a separator-backed edge to also have base p_merge>=0.5 before
    it could enter the reserved quota. Most training negatives were already easy,
    so that quota was often empty. V2 reserves quota for all strong-separator
    negatives and ranks them by current base p_merge.
    """
    import torch

    valid = targets.valid.bool()
    positive = torch.nonzero(valid & (targets.target > 0.5), as_tuple=False).flatten()
    negative = torch.nonzero(valid & (targets.target <= 0.5), as_tuple=False).flatten()
    selected_positive = _spread_indices(positive, min(max_edges_per_class, int(positive.numel())))

    if selected_positive.numel() and negative.numel():
        negative_quota = min(int(selected_positive.numel()), max_edges_per_class, int(negative.numel()))
    else:
        negative_quota = min(max_edges_per_class, int(negative.numel()))

    base_prob = base_logits.detach().sigmoid()
    coverage70 = barrier_features[:, 4].detach()
    separator_strong = negative[coverage70[negative] >= strong_separator_coverage]
    separator_hard = separator_strong[base_prob[separator_strong] >= hard_negative_probability]

    reserved = min(int(round(negative_quota * separator_hard_fraction)), int(separator_strong.numel()))
    if reserved:
        priority = base_prob[separator_strong] + 0.05 * coverage70[separator_strong]
        order = torch.argsort(priority, descending=True)
        selected_separator = separator_strong[order[:reserved]]
    else:
        selected_separator = separator_strong[:0]

    used = set(int(v) for v in selected_separator.detach().cpu().tolist())
    ordinary = torch.as_tensor([int(v) for v in negative.detach().cpu().tolist() if int(v) not in used], device=negative.device, dtype=negative.dtype)
    selected_ordinary = _spread_indices(ordinary, negative_quota - int(selected_separator.numel()))
    selected_negative = torch.cat([selected_separator, selected_ordinary], dim=0)

    mask = torch.zeros_like(valid)
    mask[selected_positive] = True
    mask[selected_negative] = True
    return {
        "mask": mask,
        "positive_indices": selected_positive,
        "negative_indices": selected_negative,
        "separator_priority_indices": selected_separator,
        "all_positive_count": int(positive.numel()),
        "all_negative_count": int(negative.numel()),
        "all_separator_strong_count": int(separator_strong.numel()),
        "all_separator_hard_count": int(separator_hard.numel()),
    }


def _class_balanced_bce(
    logits,
    selected,
):
    import torch
    import torch.nn.functional as F

    pieces = []
    pos_loss = logits.new_zeros(())
    neg_loss = logits.new_zeros(())

    pos = selected["positive_indices"]
    neg = selected["negative_indices"]

    if pos.numel():
        pos_loss = F.binary_cross_entropy_with_logits(
            logits[pos],
            torch.ones_like(logits[pos]),
        )
        pieces.append(pos_loss)

    if neg.numel():
        neg_loss = F.binary_cross_entropy_with_logits(
            logits[neg],
            torch.zeros_like(logits[neg]),
        )
        pieces.append(neg_loss)

    if not pieces:
        return (
            logits.sum() * 0,
            pos_loss.detach(),
            neg_loss.detach(),
        )

    return (
        torch.stack(pieces).mean(),
        pos_loss.detach(),
        neg_loss.detach(),
    )


def _separator_barrier_aux_loss(
    *,
    barrier_scores,
    targets,
    barrier_features,
    strong_separator_coverage: float,
):
    """Balanced semantic barrier supervision.

    ON: valid GT-different-cell edge with strong separator coverage.
    OFF: every valid GT-same-cell edge.
    Weak-separator GT-negative edges are ignored by this auxiliary loss.
    """
    import torch
    import torch.nn.functional as F

    valid = targets.valid.bool()
    same_cell = valid & (targets.target > 0.5)
    different_cell = valid & ~same_cell
    strong_different = different_cell & (barrier_features[:, 4] >= strong_separator_coverage)

    pieces = []
    off_loss = barrier_scores.sum() * 0
    on_loss = barrier_scores.sum() * 0
    if bool(same_cell.any()):
        off_loss = F.binary_cross_entropy_with_logits(barrier_scores[same_cell], torch.zeros_like(barrier_scores[same_cell]))
        pieces.append(off_loss)
    if bool(strong_different.any()):
        on_loss = F.binary_cross_entropy_with_logits(barrier_scores[strong_different], torch.ones_like(barrier_scores[strong_different]))
        pieces.append(on_loss)
    loss = torch.stack(pieces).mean() if pieces else barrier_scores.sum() * 0
    return loss, on_loss.detach(), off_loss.detach(), strong_different, same_cell


def _positive_preservation_loss(
    *,
    final_logits,
    base_logits,
    barrier,
    targets,
    merge_threshold: float,
):
    import torch.nn.functional as F

    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    baseline_prob = base_logits.detach().sigmoid()
    preserve = (
        positive
        & (baseline_prob >= merge_threshold)
    )

    if not bool(preserve.any()):
        zero = final_logits.sum() * 0
        return zero, zero, preserve

    logit_preservation = F.smooth_l1_loss(
        final_logits[preserve],
        base_logits.detach()[preserve],
        beta=1.0,
    )
    # Since the barrier is separation-only, a direct penalty on known-good
    # same-cell edges is semantically clean.
    positive_barrier = barrier[preserve].mean()
    return (
        logit_preservation,
        positive_barrier,
        preserve,
    )


# ======================================================================================
# Forward helper
# ======================================================================================

def _forward_crop(
    *,
    model,
    barrier_model,
    crop,
    rag_criterion,
    amp_dtype: str,
    need_model_grad: bool,
):
    import torch

    # Dense geometry is always frozen and computed once.
    with torch.no_grad(), _autocast_context(amp_dtype):
        geometry_output = model(
            crop["spatial_inputs"],
            crop["spacing_um"],
            crop["dref_um"],
            spatial_padding_mask=crop.get(
                "spatial_padding_mask"
            ),
            execution_stage="geometry",
        )

    if need_model_grad:
        with _autocast_context(amp_dtype):
            output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get(
                    "spatial_padding_mask"
                ),
                execution_stage="spatial",
                precomputed_geometry=geometry_output,
            )
    else:
        with torch.no_grad(), _autocast_context(amp_dtype):
            output = model(
                crop["spatial_inputs"],
                crop["spacing_um"],
                crop["dref_um"],
                spatial_padding_mask=crop.get(
                    "spatial_padding_mask"
                ),
                execution_stage="spatial",
                precomputed_geometry=geometry_output,
            )

    targets = rag_criterion.build_targets(
        output.rag,
        crop["gt_labels"],
        valid_mask=crop.get(
            "supervision_valid_mask"
        ),
    )

    explicit = _separator_contact_features(
        rag=output.rag,
        geometry=geometry_output.geometry,
        spacing_um=crop["spacing_um"],
        dref_um=crop["dref_um"],
    )

    if output.rag.edge_morphology_embeddings is None:
        raise RuntimeError(
            "Current RAG does not contain edge morphology embeddings"
        )

    morphology = (
        output.rag.edge_morphology_embeddings
        if need_model_grad
        else output.rag.edge_morphology_embeddings.detach()
    )

    with _autocast_context(amp_dtype):
        (
            barrier,
            barrier_score,
            barrier_gate_logit,
        ) = barrier_model(
            explicit,
            morphology,
        )

    base_logits = output.rag.spatial_edge_logits
    final_logits = base_logits - barrier.to(
        dtype=base_logits.dtype
    )

    return {
        "geometry_output": geometry_output,
        "output": output,
        "targets": targets,
        "barrier_features": explicit,
        "base_logits": base_logits,
        "barrier": barrier,
        # Kept for compatibility with older diagnostics: raw_barrier is now
        # the directly supervised barrier score, not the suppressed gate logit.
        "raw_barrier": barrier_score,
        "barrier_score": barrier_score,
        "barrier_gate_logit": barrier_gate_logit,
        "final_logits": final_logits,
    }


# ======================================================================================
# Validation
# ======================================================================================

def _new_accumulator():
    return {
        "crop_count": 0.0,
        "valid_edge_count": 0.0,
        "positive_edge_count": 0.0,
        "negative_edge_count": 0.0,
        "bce_sum": 0.0,
        "correct_count": 0.0,
        "positive_probability_sum": 0.0,
        "negative_probability_sum": 0.0,
        "false_merge_count": 0.0,
        "positive_accept_count": 0.0,
        "strong_separator_positive_count": 0.0,
        "strong_separator_negative_count": 0.0,
        "strong_separator_false_merge_count": 0.0,
        "barrier_positive_sum": 0.0,
        "barrier_negative_sum": 0.0,
        "barrier_strong_separator_negative_sum": 0.0,
        "gate_probability_positive_sum": 0.0,
        "gate_probability_negative_sum": 0.0,
        "gate_probability_strong_separator_negative_sum": 0.0,
    }


def _validation_contribution(
    *,
    logits,
    targets,
    barrier_features,
    barrier,
    gate_logits,
    merge_threshold: float,
    strong_separator_coverage: float,
):
    import torch.nn.functional as F

    p = logits.detach().sigmoid()
    gate_probability = gate_logits.detach().float().sigmoid()
    valid = targets.valid.bool()
    positive = valid & (targets.target > 0.5)
    negative = valid & ~positive
    strong = barrier_features[:, 4] >= strong_separator_coverage
    strong_separator_positive = positive & strong
    strong_separator_negative = negative & strong

    bce_sum = float(F.binary_cross_entropy_with_logits(logits[valid], targets.target[valid], reduction="sum").detach().float().cpu()) if bool(valid.any()) else 0.0
    def count(mask): return float(mask.sum().item())
    def sum_values(mask, values):
        return 0.0 if not bool(mask.any()) else float(values[mask].detach().float().sum().cpu())

    return {
        "crop_count": 1.0,
        "valid_edge_count": count(valid),
        "positive_edge_count": count(positive),
        "negative_edge_count": count(negative),
        "bce_sum": bce_sum,
        "correct_count": count(valid & ((p >= 0.5) == (targets.target > 0.5))),
        "positive_probability_sum": sum_values(positive, p),
        "negative_probability_sum": sum_values(negative, p),
        "false_merge_count": count(negative & (p >= merge_threshold)),
        "positive_accept_count": count(positive & (p >= merge_threshold)),
        "strong_separator_positive_count": count(strong_separator_positive),
        "strong_separator_negative_count": count(strong_separator_negative),
        "strong_separator_false_merge_count": count(strong_separator_negative & (p >= merge_threshold)),
        "barrier_positive_sum": sum_values(positive, barrier),
        "barrier_negative_sum": sum_values(negative, barrier),
        "barrier_strong_separator_negative_sum": sum_values(strong_separator_negative, barrier),
        "gate_probability_positive_sum": sum_values(positive, gate_probability),
        "gate_probability_negative_sum": sum_values(negative, gate_probability),
        "gate_probability_strong_separator_negative_sum": sum_values(strong_separator_negative, gate_probability),
    }


def _add_accumulator(acc, row):
    for key, value in row.items():
        acc[key] = (
            acc.get(key, 0.0)
            + float(value)
        )


def _ratio(a, b):
    return float(a / b) if b > 0 else 0.0


def _finalize_accumulator(acc):
    valid = acc["valid_edge_count"]
    pos = acc["positive_edge_count"]
    neg = acc["negative_edge_count"]
    strong_pos = acc["strong_separator_positive_count"]
    strong_neg = acc["strong_separator_negative_count"]
    return {
        "crop_count": int(acc["crop_count"]),
        "valid_edge_count": int(valid),
        "positive_edge_count": int(pos),
        "negative_edge_count": int(neg),
        "bce": _ratio(acc["bce_sum"], valid),
        "accuracy_05": _ratio(acc["correct_count"], valid),
        "mean_positive_probability": _ratio(acc["positive_probability_sum"], pos),
        "mean_negative_probability": _ratio(acc["negative_probability_sum"], neg),
        "false_merge_count": int(acc["false_merge_count"]),
        "false_merge_rate": _ratio(acc["false_merge_count"], neg),
        "positive_accept_count": int(acc["positive_accept_count"]),
        "positive_accept_rate": _ratio(acc["positive_accept_count"], pos),
        "strong_separator_positive_count": int(strong_pos),
        "strong_separator_positive_rate": _ratio(strong_pos, pos),
        "strong_separator_negative_count": int(strong_neg),
        "strong_separator_negative_rate": _ratio(strong_neg, neg),
        "strong_separator_false_merge_count": int(acc["strong_separator_false_merge_count"]),
        "strong_separator_false_merge_rate": _ratio(acc["strong_separator_false_merge_count"], strong_neg),
        "mean_barrier_positive": _ratio(acc["barrier_positive_sum"], pos),
        "mean_barrier_negative": _ratio(acc["barrier_negative_sum"], neg),
        "mean_barrier_strong_separator_negative": _ratio(acc["barrier_strong_separator_negative_sum"], strong_neg),
        "mean_gate_probability_positive": _ratio(acc["gate_probability_positive_sum"], pos),
        "mean_gate_probability_negative": _ratio(acc["gate_probability_negative_sum"], neg),
        "mean_gate_probability_strong_separator_negative": _ratio(acc["gate_probability_strong_separator_negative_sum"], strong_neg),
    }


def _validation_rank(
    candidate,
    initial_baseline,
    *,
    allowed_positive_accept_drop: float,
):
    """Lower tuple is better; positive-merge safety is the hard first gate."""
    floor = (
        initial_baseline["positive_accept_rate"]
        - allowed_positive_accept_drop
    )
    violation = max(
        0.0,
        floor
        - candidate["positive_accept_rate"],
    )
    return (
        int(violation > 0),
        float(violation),
        float(
            candidate[
                "strong_separator_false_merge_rate"
            ]
        ),
        float(candidate["false_merge_rate"]),
        float(candidate["bce"]),
    )



def _distribution_summary(values):
    import torch
    if not values:
        return {"count":0,"mean":0.0,"p25":0.0,"median":0.0,"p75":0.0,"p90":0.0,"min":0.0,"max":0.0}
    t=torch.cat(values).float()
    if t.numel()==0:
        return {"count":0,"mean":0.0,"p25":0.0,"median":0.0,"p75":0.0,"p90":0.0,"min":0.0,"max":0.0}
    q=torch.quantile(t,torch.tensor([0.25,0.50,0.75,0.90]))
    return {"count":int(t.numel()),"mean":float(t.mean()),"p25":float(q[0]),"median":float(q[1]),"p75":float(q[2]),"p90":float(q[3]),"min":float(t.min()),"max":float(t.max())}


def _new_distribution_store():
    features=("separator_mean","coverage_ge_0.50","coverage_ge_0.70","coverage_ge_0.85","coverage_ge_0.95")
    classes=("positive","negative","false_merge_negative")
    return {c:{**{n:[] for n in features},"barrier":[],"barrier_score":[],"gate_probability":[],"gate_logit":[]} for c in classes}


def _accumulate_distributions(store, *, result, merge_threshold: float):
    targets=result["targets"]
    valid=targets.valid.bool()
    positive=valid & (targets.target>0.5)
    negative=valid & ~positive
    p=result["final_logits"].detach().sigmoid()
    masks={"positive":positive,"negative":negative,"false_merge_negative":negative & (p>=merge_threshold)}
    cols={"separator_mean":0,"coverage_ge_0.50":3,"coverage_ge_0.70":4,"coverage_ge_0.85":5,"coverage_ge_0.95":6}
    for cname,mask in masks.items():
        if not bool(mask.any()): continue
        for name,col in cols.items():
            store[cname][name].append(result["barrier_features"][mask,col].detach().float().cpu())
        store[cname]["barrier"].append(result["barrier"][mask].detach().float().cpu())
        store[cname]["barrier_score"].append(result["barrier_score"][mask].detach().float().cpu())
        store[cname]["gate_logit"].append(result["barrier_gate_logit"][mask].detach().float().cpu())
        store[cname]["gate_probability"].append(result["barrier_gate_logit"][mask].detach().float().sigmoid().cpu())


def _finalize_distribution_store(store):
    return {c:{name:_distribution_summary(vals) for name,vals in rows.items()} for c,rows in store.items()}

def _run_validation(
    *,
    support,
    model,
    barrier_model,
    source_batches,
    splits,
    rag_criterion,
    amp_dtype: str,
    partial_ignore_margin_um: float,
    merge_threshold: float,
    strong_separator_coverage: float,
    description: str,
):
    import torch

    model.eval()
    barrier_model.eval()

    candidate_acc = _new_accumulator()
    base_acc = _new_accumulator()
    distribution_store = _new_distribution_store()
    per_sample = {}

    total = sum(
        len(
            splits[sample][
                "validation_indices"
            ]
        )
        for sample in source_batches
    )
    progress = tqdm(
        total=total,
        desc=description,
        unit="crop",
        dynamic_ncols=True,
        leave=False,
        colour="green",
        file=sys.stdout,
    )

    for sample, source_batch in (
        source_batches.items()
    ):
        sample_candidate = _new_accumulator()
        sample_base = _new_accumulator()

        for manifest_index in (
            splits[sample][
                "validation_indices"
            ]
        ):
            record = splits[sample][
                "records"
            ][manifest_index]
            crop_cpu, _ = support._materialize_crop(
                source_batch,
                record,
                partial_ignore_margin_um=(
                    partial_ignore_margin_um
                ),
            )
            crop = support._move_crop_to_cuda(
                crop_cpu
            )

            with torch.no_grad():
                result = _forward_crop(
                    model=model,
                    barrier_model=barrier_model,
                    crop=crop,
                    rag_criterion=rag_criterion,
                    amp_dtype=amp_dtype,
                    need_model_grad=False,
                )

            zero_barrier = torch.zeros_like(
                result["barrier"]
            )

            candidate_row = (
                _validation_contribution(
                    logits=result["final_logits"],
                    targets=result["targets"],
                    barrier_features=(
                        result["barrier_features"]
                    ),
                    barrier=result["barrier"],
                    gate_logits=result["barrier_gate_logit"],
                    merge_threshold=(
                        merge_threshold
                    ),
                    strong_separator_coverage=(
                        strong_separator_coverage
                    ),
                )
            )
            base_row = _validation_contribution(
                logits=result["base_logits"],
                targets=result["targets"],
                barrier_features=(
                    result["barrier_features"]
                ),
                barrier=zero_barrier,
                gate_logits=torch.full_like(result["raw_barrier"], -40.0),
                merge_threshold=merge_threshold,
                strong_separator_coverage=(
                    strong_separator_coverage
                ),
            )

            _add_accumulator(
                candidate_acc,
                candidate_row,
            )
            _add_accumulator(
                base_acc,
                base_row,
            )
            _add_accumulator(
                sample_candidate,
                candidate_row,
            )
            _add_accumulator(
                sample_base,
                base_row,
            )
            _accumulate_distributions(
                distribution_store,
                result=result,
                merge_threshold=merge_threshold,
            )

            progress.set_postfix(
                {
                    "sample": sample.replace(
                        "Drosophila_",
                        "D",
                    ),
                    "idx": manifest_index,
                    "FM": int(
                        candidate_row[
                            "false_merge_count"
                        ]
                    ),
                    "SFM": int(
                        candidate_row[
                            "strong_separator_false_merge_count"
                        ]
                    ),
                },
                refresh=False,
            )
            progress.update(1)

            del result, crop, crop_cpu
            torch.cuda.empty_cache()

        per_sample[sample] = {
            "candidate": _finalize_accumulator(
                sample_candidate
            ),
            "base_without_barrier": (
                _finalize_accumulator(
                    sample_base
                )
            ),
        }

    progress.close()
    return {
        "candidate": _finalize_accumulator(
            candidate_acc
        ),
        "base_without_barrier": (
            _finalize_accumulator(base_acc)
        ),
        "separator_distributions": _finalize_distribution_store(
            distribution_store
        ),
        "per_sample": per_sample,
    }


# ======================================================================================
# Checkpointing
# ======================================================================================

def _atomic_torch_save(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_checkpoint(
    *,
    path: Path,
    model,
    model_cfg,
    barrier_model,
    optimizer,
    scaler,
    global_step: int,
    run_dir: Path,
    starting_checkpoint: Path,
    train_scope: str,
    split_summary: dict,
    initial_baseline: dict,
    validation: dict | None,
    config_payload: dict,
):
    payload = {
        "architecture": "STIR-Net",
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "separator_barrier": (
            barrier_model.state_dict()
        ),
        "separator_barrier_config": {
            "version": BARRIER_VERSION,
            "parameterization": BARRIER_PARAMETERIZATION,
            "feature_dim": BARRIER_FEATURE_DIM,
            "morphology_dim": int(
                model_cfg.partition.rag_edge_morphology_dim
            ),
            "final_logit_rule": (
                "base_merge_logit - nonnegative_barrier"
            ),
            "initial_gate_bias": float(config_payload["barrier_initial_gate_bias"]),
            "barrier_score_scale": float(config_payload["barrier_score_scale"]),
            "max_barrier_logit": float(config_payload["max_barrier_logit"]),
        },
        "training_config": {
            "experiment": EXPERIMENT_NAME,
            "train_scope": train_scope,
        },
        "extra": {
            "experiment": EXPERIMENT_NAME,
            "run_dir": str(run_dir),
            "starting_checkpoint": str(
                starting_checkpoint
            ),
            "train_scope": train_scope,
            "split": split_summary,
            "initial_baseline": initial_baseline,
            "validation": validation,
            "config": config_payload,
        },
    }
    _atomic_torch_save(path, payload)


# ======================================================================================
# Main training
# ======================================================================================

def _training_impl(args) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from learned.stirnet.model.partition.rag import (
        RAGCriterion,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Investigation 20"
        )
    if not args.resume:
        raise ValueError(
            "Investigation 20 intentionally starts from a checkpoint. "
            "Pass --resume --checkpoint <path>."
        )

    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.validation_every < 1:
        raise ValueError(
            "--validation-every must be positive"
        )
    if args.checkpoint_every < 1:
        raise ValueError(
            "--checkpoint-every must be positive"
        )
    if not 0.0 <= args.separator_hard_fraction <= 1.0:
        raise ValueError(
            "--separator-hard-fraction must be in [0,1]"
        )
    if not 0.0 <= args.strong_separator_coverage <= 1.0:
        raise ValueError(
            "--strong-separator-coverage must be in [0,1]"
        )
    if args.barrier_aux_weight < 0:
        raise ValueError("--barrier-aux-weight must be >= 0")
    if args.max_barrier_logit <= 0:
        raise ValueError("--max-barrier-logit must be > 0")
    if args.barrier_score_scale <= 0:
        raise ValueError("--barrier-score-scale must be > 0")

    support = _load_inv17_support()
    _seed_everything(args.seed)

    starting_checkpoint = _resolve_checkpoint(
        args.checkpoint
    )
    model, model_cfg, source_payload, transfer_report = (
        _build_current_model_from_checkpoint(
            starting_checkpoint,
            device="cuda",
        )
    )

    barrier_model = SeparatorAwareBarrier.build(
        morphology_dim=(
            model_cfg.partition.rag_edge_morphology_dim
        ),
        hidden_dim=args.barrier_hidden_dim,
        max_barrier_logit=args.max_barrier_logit,
        initial_gate_bias=args.barrier_initial_gate_bias,
        barrier_score_scale=args.barrier_score_scale,
    ).to("cuda")

    exact_resume = _is_inv20_checkpoint(
        source_payload
    )
    if exact_resume:
        checkpoint_scope = (
            source_payload.get("extra", {}).get("train_scope")
            or source_payload.get("training_config", {}).get("train_scope")
        )
        if checkpoint_scope is None:
            raise RuntimeError(
                "Investigation-20 checkpoint does not record train_scope; "
                "cannot safely restore optimizer parameter groups."
            )
        if args.train_scope == "auto":
            train_scope = str(checkpoint_scope)
        elif args.train_scope != str(checkpoint_scope):
            raise ValueError(
                "Exact Investigation-20 resume requires the original train "
                f"scope {checkpoint_scope!r}; got --train-scope "
                f"{args.train_scope!r}."
            )
        else:
            train_scope = args.train_scope
    else:
        train_scope = _resolve_train_scope(
            args.train_scope,
            transfer_report,
        )

    barrier_parameters, morphology_parameters = (
        _configure_trainability(
            model,
            barrier_model,
            scope=train_scope,
        )
    )

    parameter_groups = [
        {
            "params": barrier_parameters,
            "lr": args.barrier_lr,
            "weight_decay": args.weight_decay,
            "name": "separator_barrier",
        }
    ]
    if morphology_parameters:
        parameter_groups.append(
            {
                "params": morphology_parameters,
                "lr": args.morphology_lr,
                "weight_decay": args.weight_decay,
                "name": "morphology",
            }
        )

    optimizer = torch.optim.AdamW(
        parameter_groups
    )
    amp_dtype = (
        "bf16"
        if torch.cuda.is_bf16_supported()
        else "fp16"
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_dtype == "fp16"),
    )

    global_step = 0
    resumed_from = None
    restored_initial_baseline = None

    if exact_resume:
        print(
            f"[resume] exact Investigation-20 v2 checkpoint: "
            f"{starting_checkpoint}",
            flush=True,
        )
        model.load_state_dict(
            source_payload["model"],
            strict=True,
        )
        barrier_model.load_state_dict(
            source_payload["separator_barrier"],
            strict=True,
        )
        if "optimizer" in source_payload:
            optimizer.load_state_dict(
                source_payload["optimizer"]
            )
        if "scaler" in source_payload:
            scaler.load_state_dict(
                source_payload["scaler"]
            )
        global_step = int(
            source_payload.get(
                "global_step",
                0,
            )
        )
        restored_initial_baseline = (
            source_payload.get(
                "extra",
                {},
            ).get(
                "initial_baseline"
            )
        )
        resumed_from = str(starting_checkpoint)
        # Reassert requires_grad flags after strict state restoration.
        barrier_parameters, morphology_parameters = (
            _configure_trainability(
                model,
                barrier_model,
                scope=train_scope,
            )
        )
    else:
        print(
            "[resume] architecture-transfer checkpoint: "
            f"{starting_checkpoint}",
            flush=True,
        )
        print(
            "[resume] Investigation-20 optimizer step starts at 0; "
            f"source checkpoint global_step="
            f"{source_payload.get('global_step', -1)}",
            flush=True,
        )

    breakdown = _trainable_breakdown(
        model,
        barrier_model,
    )
    if breakdown["legacy_rag_trainable"] != 0:
        raise RuntimeError(
            "Legacy RAG unexpectedly has trainable parameters"
        )

    samples = tuple(
        token.strip()
        for token in args.samples.split(",")
        if token.strip()
    )
    if not samples:
        raise ValueError(
            "At least one NIS3D sample is required"
        )

    crop_shape = support._parse_shape_zyx(
        args.crop_shape_zyx
    )
    spacing_override_zyx_um = (
        support._parse_spacing_xyz_override(
            args.spacing_xyz
        )
    )

    nis3d_root = support._discover_nis3d_root(
        samples,
        data_dir=args.data_dir,
        execution_mode="local",
    )

    experiment_root = (
        REPO_ROOT
        / "runs"
        / "stirnet"
        / "investigations"
        / EXPERIMENT_NAME
    )
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        experiment_root
        / "attempts"
        / f"{timestamp}_{args.run_name}"
    )
    recovery_dir = (
        experiment_root
        / "recovery"
        / args.run_name
    )
    cache_root = (
        experiment_root / "cache"
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    recovery_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 122, flush=True)
    print(
        "STIR-Net Investigation 20 v2 — separator-aware barrier training",
        flush=True,
    )
    print("=" * 122, flush=True)
    props = torch.cuda.get_device_properties(0)
    print(
        f"GPU                       : {props.name}",
        flush=True,
    )
    print(
        f"VRAM                      : {props.total_memory/2**30:.2f} GiB",
        flush=True,
    )
    print(
        f"Starting checkpoint       : {starting_checkpoint}",
        flush=True,
    )
    print(
        f"Exact Inv20-v2 resume     : {exact_resume}",
        flush=True,
    )
    print(
        f"Legacy Inv20-v1 source    : {transfer_report['legacy_inv20_checkpoint']}",
        flush=True,
    )
    print(
        f"Train scope               : {train_scope}",
        flush=True,
    )
    print(
        f"Transferred tensors       : "
        f"{transfer_report['transferred_parameter_tensors']}/"
        f"{transfer_report['historical_parameter_tensors']}",
        flush=True,
    )
    print(
        f"Node morphology incomplete: "
        f"{transfer_report['node_morphology_incomplete']}",
        flush=True,
    )
    print(
        f"Edge morphology incomplete: "
        f"{transfer_report['edge_morphology_incomplete']}",
        flush=True,
    )
    print(
        f"Reset projections         : "
        f"{transfer_report['reset_projection_tensors']}",
        flush=True,
    )
    print(
        f"Trainable barrier         : "
        f"{breakdown['separator_barrier']/1e6:.3f}M",
        flush=True,
    )
    print(
        f"Trainable model/morphology: "
        f"{breakdown['model_trainable']/1e6:.3f}M",
        flush=True,
    )
    print(
        f"Legacy RAG trainable      : "
        f"{breakdown['legacy_rag_trainable']/1e6:.3f}M",
        flush=True,
    )
    print(
        f"Samples                   : {list(samples)}",
        flush=True,
    )
    print(
        f"NIS3D root                : {nis3d_root}",
        flush=True,
    )
    print(
        f"Crop shape ZYX            : {crop_shape}",
        flush=True,
    )
    print(
        f"Barrier LR                : {args.barrier_lr:g}",
        flush=True,
    )
    print(
        f"Barrier aux weight        : {args.barrier_aux_weight:g}",
        flush=True,
    )
    print(
        f"Initial gate bias         : {args.barrier_initial_gate_bias:g}",
        flush=True,
    )
    print(
        f"Barrier score scale       : {args.barrier_score_scale:g}",
        flush=True,
    )
    print(
        f"Morphology LR             : {args.morphology_lr:g}",
        flush=True,
    )
    print(
        f"Merge threshold           : {args.merge_threshold:.3f}",
        flush=True,
    )
    print(
        "Strong separator metric   : "
        f"physical coverage(sep>=0.70) >= "
        f"{args.strong_separator_coverage:.2f}",
        flush=True,
    )
    print(
        f"Separator-hard quota      : "
        f"{args.separator_hard_fraction:.2f}",
        flush=True,
    )
    print(
        "Barrier rule              : final_logit = "
        "base_logit - nonnegative_barrier",
        flush=True,
    )
    print(
        "Original-mask mutex       : NOT IMPLEMENTED",
        flush=True,
    )
    print("=" * 122, flush=True)

    # ------------------------------------------------------------------
    # Data preparation: use the same source/crop semantics as Inv17,
    # which in turn mirrors production Training01.
    # ------------------------------------------------------------------
    source_batches = {}
    sample_reports = {}

    for sample in samples:
        print(
            f"[data] preparing {sample} ...",
            flush=True,
        )
        signature = support._data_signature(
            nis3d_root=nis3d_root,
            sample=sample,
            spacing_override_zyx_um=(
                spacing_override_zyx_um
            ),
            confidence_ignore_margin_um=(
                args.confidence_ignore_margin_um
            ),
        )
        source_batch, report = (
            support._prepare_sample_batch(
                nis3d_root=nis3d_root,
                sample=sample,
                spacing_override_zyx_um=(
                    spacing_override_zyx_um
                ),
                confidence_ignore_margin_um=(
                    args.confidence_ignore_margin_um
                ),
                cache_root=cache_root,
                cache_namespace=signature,
            )
        )
        source_batches[sample] = source_batch
        sample_reports[sample] = report
        print(
            f"[data] {sample}: "
            f"shape={tuple(report['shape_zyx'])} "
            f"GT={report['gt_ids_kept']} "
            f"source={report['source_instance_count']} "
            f"dref={report['model_dref_um']:.4f}um "
            f"RAM-cache="
            f"{report['source_ram_cache_gross_gib']:.2f}GiB",
            flush=True,
        )

    splits = support._build_split(
        source_batches,
        crop_shape_zyx=crop_shape,
        validation_crops_per_sample=(
            args.validation_crops_per_sample
        ),
    )

    split_summary = {}
    for sample in samples:
        row = splits[sample]
        split_summary[sample] = {
            "manifest_record_count": len(
                row["records"]
            ),
            "train_indices": row[
                "train_indices"
            ],
            "validation_indices": row[
                "validation_indices"
            ],
            "train_merge_count": len(
                row["train_merge_indices"]
            ),
            "train_nonmerge_count": len(
                row["train_nonmerge_indices"]
            ),
        }
        print(
            f"[split] {sample}: "
            f"manifest={len(row['records'])} "
            f"train={len(row['train_indices'])} "
            f"(merge={len(row['train_merge_indices'])}, "
            f"nonmerge={len(row['train_nonmerge_indices'])}) "
            f"val={row['validation_indices']}",
            flush=True,
        )

    rag_criterion = RAGCriterion(
        model_cfg.partition
    ).to("cuda")
    rag_criterion.eval()

    config_payload = {
        "experiment": EXPERIMENT_NAME,
        "starting_checkpoint": str(
            starting_checkpoint
        ),
        "source_checkpoint_global_step": int(
            source_payload.get(
                "global_step",
                -1,
            )
        ),
        "exact_inv20_resume": exact_resume,
        "resumed_from": resumed_from,
        "transfer_report": transfer_report,
        "train_scope": train_scope,
        "trainable_breakdown": breakdown,
        "model": model_cfg.to_dict(),
        "samples": list(samples),
        "sample_reports": sample_reports,
        "split": split_summary,
        "max_steps": int(args.max_steps),
        "barrier_lr": float(
            args.barrier_lr
        ),
        "morphology_lr": float(
            args.morphology_lr
        ),
        "weight_decay": float(
            args.weight_decay
        ),
        "max_grad_norm": float(
            args.max_grad_norm
        ),
        "max_edges_per_class": int(
            args.max_edges_per_class
        ),
        "separator_hard_fraction": float(
            args.separator_hard_fraction
        ),
        "hard_negative_probability": float(
            args.hard_negative_probability
        ),
        "strong_separator_coverage": float(
            args.strong_separator_coverage
        ),
        "merge_threshold": float(
            args.merge_threshold
        ),
        "positive_preservation_weight": float(
            args.positive_preservation_weight
        ),
        "positive_barrier_weight": float(
            args.positive_barrier_weight
        ),
        "barrier_aux_weight": float(args.barrier_aux_weight),
        "barrier_initial_gate_bias": float(args.barrier_initial_gate_bias),
        "barrier_score_scale": float(args.barrier_score_scale),
        "barrier_parameterization": BARRIER_PARAMETERIZATION,
        "barrier_version": BARRIER_VERSION,
        "allowed_positive_accept_drop": float(
            args.allowed_positive_accept_drop
        ),
        "barrier_feature_dim": (
            BARRIER_FEATURE_DIM
        ),
        "max_barrier_logit": float(args.max_barrier_logit),
        "barrier_feature_definition": [
            "separator_mean_area_weighted",
            "separator_max",
            "separator_std_area_weighted",
            "separator_coverage_ge_0.50",
            "separator_coverage_ge_0.70",
            "separator_coverage_ge_0.85",
            "separator_coverage_ge_0.95",
            "surface_mean_area_weighted",
            "surface_max",
            "separator_surface_mean_area_weighted",
            "abs_sdf_mean_over_dref",
            "log1p_contact_area_over_dref2",
        ],
        "original_mask_mutex_enabled": False,
        "dense_geometry_frozen": True,
        "legacy_rag_frozen": True,
        "source_dropout_probability": 0.0,
        "xy_flip_probability": 0.0,
        "amp_dtype": amp_dtype,
        "seed": int(args.seed),
    }
    _atomic_json(
        run_dir / "config.json",
        config_payload,
    )
    _atomic_json(
        run_dir / "samples.json",
        sample_reports,
    )
    _atomic_json(
        run_dir / "split.json",
        split_summary,
    )

    # ------------------------------------------------------------------
    # Fixed held-out baseline.  For exact resume, preserve the baseline
    # from the original Inv20 run so the safety floor does not drift.
    # ------------------------------------------------------------------
    if restored_initial_baseline is None:
        print(
            "[validation] evaluating fixed step-0 baseline ...",
            flush=True,
        )
        # V2 starts with a tiny live sigmoid barrier; inherited production
        # decisions must remain unchanged at step 0.
        validation0 = _run_validation(
            support=support,
            model=model,
            barrier_model=barrier_model,
            source_batches=source_batches,
            splits=splits,
            rag_criterion=rag_criterion,
            amp_dtype=amp_dtype,
            partial_ignore_margin_um=(
                args.partial_ignore_margin_um
            ),
            merge_threshold=(
                args.merge_threshold
            ),
            strong_separator_coverage=(
                args.strong_separator_coverage
            ),
            description="Inv20 baseline",
        )
        initial_baseline = validation0[
            "base_without_barrier"
        ]
        if not exact_resume:
            # The tiny live gate may shift logits slightly, but must not
            # change inherited merge/separate decisions at the production threshold.
            candidate0 = validation0[
                "candidate"
            ]
            if (
                candidate0[
                    "false_merge_count"
                ]
                != initial_baseline[
                    "false_merge_count"
                ]
                or abs(
                    candidate0[
                        "positive_accept_rate"
                    ]
                    - initial_baseline[
                        "positive_accept_rate"
                    ]
                )
                > 1e-8
            ):
                raise RuntimeError(
                    "Separator barrier changed step-0 production decisions. "
                    "Use a more negative --barrier-initial-gate-bias."
                )
    else:
        initial_baseline = (
            restored_initial_baseline
        )
        print(
            "[validation] restored original Inv20 baseline from checkpoint",
            flush=True,
        )

    _atomic_json(
        run_dir
        / "initial_baseline_validation.json",
        initial_baseline,
    )

    if global_step >= args.max_steps:
        summary = {
            "status": "already_complete",
            "global_step": global_step,
            "max_steps": args.max_steps,
            "train_scope": train_scope,
            "starting_checkpoint": str(starting_checkpoint),
            "run_dir": str(run_dir),
            "recovery_dir": str(recovery_dir),
            "initial_baseline": initial_baseline,
        }
        _atomic_json(run_dir / "summary.json", summary)
        print(
            f"[done] checkpoint step={global_step} already satisfies "
            f"--max-steps {args.max_steps}",
            flush=True,
        )
        return summary

    print(
        "[baseline] "
        f"BCE={initial_baseline['bce']:.5f} "
        f"FM={initial_baseline['false_merge_count']}/"
        f"{initial_baseline['negative_edge_count']} "
        f"SFM="
        f"{initial_baseline['strong_separator_false_merge_count']}/"
        f"{initial_baseline['strong_separator_negative_count']} "
        f"positive_accept="
        f"{initial_baseline['positive_accept_rate']:.4f}",
        flush=True,
    )

    best_rank = None
    best_step = None
    best_path = recovery_dir / "best_checkpoint.pt"

    if exact_resume:
        previous_validation = source_payload.get(
            "extra",
            {},
        ).get(
            "validation"
        )
        if isinstance(previous_validation, dict):
            candidate = previous_validation.get(
                "candidate"
            )
            if isinstance(candidate, dict):
                best_rank = _validation_rank(
                    candidate,
                    initial_baseline,
                    allowed_positive_accept_drop=(
                        args.allowed_positive_accept_drop
                    ),
                )
                best_step = global_step

    history_path = run_dir / "history.jsonl"
    validation_path = (
        run_dir / "validation.jsonl"
    )
    skipped_path = (
        run_dir / "skipped_crops.jsonl"
    )

    sample_local_attempts = {
        sample: 0
        for sample in samples
    }
    consecutive_skips = 0
    skipped_crops = 0

    started = time.perf_counter()
    progress = tqdm(
        total=args.max_steps,
        initial=global_step,
        desc="STIR-Net barrier",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.10,
        mininterval=0.5,
        leave=True,
        colour="green",
        file=sys.stdout,
    )

    try:
        while global_step < args.max_steps:
            sample = samples[
                global_step % len(samples)
            ]
            local_attempt = (
                sample_local_attempts[sample]
            )
            split_row = splits[sample]
            manifest_index, provenance = (
                support._training_manifest_index(
                    split_row,
                    sample_local_step=(
                        local_attempt
                    ),
                )
            )
            # Advance on attempts, including skips, so one empty crop can
            # never deadlock the run.
            sample_local_attempts[sample] += 1

            record = split_row["records"][
                manifest_index
            ]
            crop_cpu, _ = (
                support._materialize_crop(
                    source_batches[sample],
                    record,
                    partial_ignore_margin_um=(
                        args.partial_ignore_margin_um
                    ),
                )
            )
            crop = support._move_crop_to_cuda(
                crop_cpu
            )

            need_model_grad = (
                train_scope != "barrier"
            )

            optimizer.zero_grad(
                set_to_none=True
            )
            torch.cuda.reset_peak_memory_stats()
            step_started = time.perf_counter()

            result = _forward_crop(
                model=model,
                barrier_model=barrier_model,
                crop=crop,
                rag_criterion=rag_criterion,
                amp_dtype=amp_dtype,
                need_model_grad=need_model_grad,
            )

            selected = _select_training_edges(
                targets=result["targets"],
                base_logits=result[
                    "base_logits"
                ],
                barrier_features=result[
                    "barrier_features"
                ],
                max_edges_per_class=(
                    args.max_edges_per_class
                ),
                separator_hard_fraction=(
                    args.separator_hard_fraction
                ),
                hard_negative_probability=(
                    args.hard_negative_probability
                ),
                strong_separator_coverage=(
                    args.strong_separator_coverage
                ),
            )

            if not bool(selected["mask"].any()):
                skipped_crops += 1
                consecutive_skips += 1
                _append_jsonl(
                    skipped_path,
                    {
                        "timestamp_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "sample": sample,
                        "manifest_index": (
                            manifest_index
                        ),
                        "provenance": provenance,
                        "edge_count": int(
                            result[
                                "base_logits"
                            ].numel()
                        ),
                        "valid_edge_count": int(
                            result[
                                "targets"
                            ].valid.sum().item()
                        ),
                    },
                )
                progress.write(
                    f"[skip] {sample} idx={manifest_index} "
                    f"edges={result['base_logits'].numel()} "
                    f"valid={result['targets'].valid.sum().item()}"
                )
                del result, selected, crop, crop_cpu
                torch.cuda.empty_cache()
                if consecutive_skips >= 256:
                    raise RuntimeError(
                        "256 consecutive crops had no selected supervised "
                        "RAG edge; aborting instead of looping indefinitely."
                    )
                continue

            consecutive_skips = 0

            bce, pos_bce, neg_bce = (
                _class_balanced_bce(
                    result["final_logits"],
                    selected,
                )
            )
            preserve_loss, positive_barrier, preserve_mask = (
                _positive_preservation_loss(
                    final_logits=result[
                        "final_logits"
                    ],
                    base_logits=result[
                        "base_logits"
                    ],
                    barrier=result[
                        "barrier"
                    ],
                    targets=result["targets"],
                    merge_threshold=(
                        args.merge_threshold
                    ),
                )
            )
            (
                barrier_aux_loss,
                barrier_aux_on_loss,
                barrier_aux_off_loss,
                barrier_aux_on_mask,
                barrier_aux_off_mask,
            ) = _separator_barrier_aux_loss(
                barrier_scores=result["barrier_score"],
                targets=result["targets"],
                barrier_features=result["barrier_features"],
                strong_separator_coverage=args.strong_separator_coverage,
            )

            total_loss = (
                bce
                + args.barrier_aux_weight * barrier_aux_loss
                + args.positive_preservation_weight * preserve_loss
                + args.positive_barrier_weight * positive_barrier
            )

            scaler.scale(
                total_loss
            ).backward()
            scaler.unscale_(optimizer)

            trainable_parameters = (
                barrier_parameters
                + morphology_parameters
            )
            barrier_grad_norm = _gradient_norm(barrier_parameters)
            morphology_grad_norm = _gradient_norm(morphology_parameters)
            grad_norm = _gradient_norm(trainable_parameters)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress.update(1)

            torch.cuda.synchronize()
            step_seconds = (
                time.perf_counter()
                - step_started
            )

            p = (
                result[
                    "final_logits"
                ].detach().sigmoid()
            )
            base_p = (
                result[
                    "base_logits"
                ].detach().sigmoid()
            )
            valid = result[
                "targets"
            ].valid.bool()
            positive = (
                valid
                & (
                    result[
                        "targets"
                    ].target
                    > 0.5
                )
            )
            negative = valid & ~positive
            strong_negative = (
                negative
                & (
                    result[
                        "barrier_features"
                    ][:, 4]
                    >= args.strong_separator_coverage
                )
            )

            def mean(mask, values):
                if not bool(mask.any()):
                    return 0.0
                return float(
                    values[mask]
                    .float()
                    .mean()
                    .cpu()
                )

            row = {
                "step": global_step,
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "sample": sample,
                "manifest_index": (
                    manifest_index
                ),
                "provenance": provenance,
                "train_scope": train_scope,
                "step_seconds": step_seconds,
                "loss": float(
                    total_loss.detach().float().cpu()
                ),
                "bce": float(
                    bce.detach().float().cpu()
                ),
                "positive_bce": float(
                    pos_bce.float().cpu()
                ),
                "negative_bce": float(
                    neg_bce.float().cpu()
                ),
                "positive_preservation_loss": float(
                    preserve_loss.detach().float().cpu()
                ),
                "positive_barrier_penalty": float(
                    positive_barrier.detach().float().cpu()
                ),
                "barrier_aux_loss": float(barrier_aux_loss.detach().float().cpu()),
                "barrier_aux_on_loss": float(barrier_aux_on_loss.float().cpu()),
                "barrier_aux_off_loss": float(barrier_aux_off_loss.float().cpu()),
                "barrier_aux_on_edges": int(barrier_aux_on_mask.sum().item()),
                "barrier_aux_off_edges": int(barrier_aux_off_mask.sum().item()),
                "barrier_gradient_norm": barrier_grad_norm,
                "morphology_gradient_norm": morphology_grad_norm,
                "gradient_norm": grad_norm,
                "valid_edges": int(
                    valid.sum().item()
                ),
                "positive_edges": int(
                    positive.sum().item()
                ),
                "negative_edges": int(
                    negative.sum().item()
                ),
                "selected_edges": int(
                    selected[
                        "mask"
                    ].sum().item()
                ),
                "selected_positive_edges": int(
                    selected[
                        "positive_indices"
                    ].numel()
                ),
                "selected_negative_edges": int(
                    selected[
                        "negative_indices"
                    ].numel()
                ),
                "selected_separator_priority_edges": int(
                    selected["separator_priority_indices"].numel()
                ),
                "all_separator_strong_edges": int(
                    selected["all_separator_strong_count"]
                ),
                "all_separator_hard_edges": int(
                    selected["all_separator_hard_count"]
                ),
                "preserved_positive_edges": int(
                    preserve_mask.sum().item()
                ),
                "mean_positive_probability": mean(
                    positive,
                    p,
                ),
                "mean_negative_probability": mean(
                    negative,
                    p,
                ),
                "mean_base_negative_probability": mean(
                    negative,
                    base_p,
                ),
                "mean_barrier_positive": mean(
                    positive,
                    result["barrier"].detach(),
                ),
                "mean_barrier_negative": mean(
                    negative,
                    result["barrier"].detach(),
                ),
                "mean_barrier_strong_separator_negative": mean(
                    strong_negative,
                    result["barrier"].detach(),
                ),
                "mean_gate_probability_positive": mean(
                    positive, result["barrier_gate_logit"].detach().float().sigmoid()
                ),
                "mean_gate_probability_negative": mean(
                    negative, result["barrier_gate_logit"].detach().float().sigmoid()
                ),
                "mean_gate_probability_strong_separator_negative": mean(
                    strong_negative, result["barrier_gate_logit"].detach().float().sigmoid()
                ),
                "false_merge_count": int(
                    (
                        negative
                        & (
                            p
                            >= args.merge_threshold
                        )
                    ).sum().item()
                ),
                "strong_separator_false_merge_count": int(
                    (
                        strong_negative
                        & (
                            p
                            >= args.merge_threshold
                        )
                    ).sum().item()
                ),
                "positive_accept_count": int(
                    (
                        positive
                        & (
                            p
                            >= args.merge_threshold
                        )
                    ).sum().item()
                ),
                "cuda_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated()
                    / 2**30
                ),
                "cuda_peak_reserved_gib": (
                    torch.cuda.max_memory_reserved()
                    / 2**30
                ),
            }
            _append_jsonl(
                history_path,
                row,
            )

            progress.set_postfix(
                {
                    "loss": f"{row['loss']:.3f}",
                    "sample": sample.replace(
                        "Drosophila_",
                        "D",
                    ),
                    "FM": row[
                        "false_merge_count"
                    ],
                    "SFM": row[
                        "strong_separator_false_merge_count"
                    ],
                    "bar": f"{row['mean_barrier_negative']:.5f}",
                    "scope": train_scope,
                },
                refresh=False,
            )

            validation = None
            if (
                global_step
                % args.validation_every
                == 0
                or global_step
                == args.max_steps
            ):
                validation = _run_validation(
                    support=support,
                    model=model,
                    barrier_model=barrier_model,
                    source_batches=source_batches,
                    splits=splits,
                    rag_criterion=rag_criterion,
                    amp_dtype=amp_dtype,
                    partial_ignore_margin_um=(
                        args.partial_ignore_margin_um
                    ),
                    merge_threshold=(
                        args.merge_threshold
                    ),
                    strong_separator_coverage=(
                        args.strong_separator_coverage
                    ),
                    description=(
                        f"Inv20 val {global_step}"
                    ),
                )
                candidate = validation[
                    "candidate"
                ]
                rank = _validation_rank(
                    candidate,
                    initial_baseline,
                    allowed_positive_accept_drop=(
                        args.allowed_positive_accept_drop
                    ),
                )

                validation_row = {
                    "step": global_step,
                    "candidate": candidate,
                    "base_without_barrier": (
                        validation[
                            "base_without_barrier"
                        ]
                    ),
                    "initial_baseline": (
                        initial_baseline
                    ),
                    "rank": list(rank),
                }
                _append_jsonl(
                    validation_path,
                    validation_row,
                )

                progress.write(
                    "[validation] "
                    f"step={global_step} "
                    f"BCE={candidate['bce']:.5f} "
                    f"FM={candidate['false_merge_count']}/"
                    f"{candidate['negative_edge_count']} "
                    f"SFM="
                    f"{candidate['strong_separator_false_merge_count']}/"
                    f"{candidate['strong_separator_negative_count']} "
                    f"PA={candidate['positive_accept_rate']:.4f} "
                    f"strong+={candidate['strong_separator_positive_count']}/"
                    f"{candidate['positive_edge_count']} "
                    f"bar_neg={candidate['mean_barrier_negative']:.6f} "
                    f"bar_pos={candidate['mean_barrier_positive']:.6f} "
                    f"gate_neg={candidate['mean_gate_probability_negative']:.6f}"
                )

                if (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_step = global_step
                    _save_checkpoint(
                        path=best_path,
                        model=model,
                        model_cfg=model_cfg,
                        barrier_model=barrier_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        global_step=global_step,
                        run_dir=run_dir,
                        starting_checkpoint=(
                            starting_checkpoint
                        ),
                        train_scope=train_scope,
                        split_summary=(
                            split_summary
                        ),
                        initial_baseline=(
                            initial_baseline
                        ),
                        validation=validation,
                        config_payload=(
                            config_payload
                        ),
                    )
                    progress.write(
                        f"[best] step={global_step} "
                        f"rank={rank} -> {best_path}"
                    )

            if (
                global_step
                % args.checkpoint_every
                == 0
                or global_step
                == args.max_steps
            ):
                checkpoint_path = (
                    recovery_dir
                    / (
                        f"checkpoint_step_"
                        f"{global_step:06d}.pt"
                    )
                )
                _save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    model_cfg=model_cfg,
                    barrier_model=barrier_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    run_dir=run_dir,
                    starting_checkpoint=(
                        starting_checkpoint
                    ),
                    train_scope=train_scope,
                    split_summary=(
                        split_summary
                    ),
                    initial_baseline=(
                        initial_baseline
                    ),
                    validation=validation,
                    config_payload=(
                        config_payload
                    ),
                )
                progress.write(
                    f"[checkpoint] {checkpoint_path}"
                )

            del (
                result,
                selected,
                crop,
                crop_cpu,
                total_loss,
                bce,
            )
            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

        elapsed = (
            time.perf_counter() - started
        )
        summary = {
            "status": "success",
            "global_step": global_step,
            "max_steps": args.max_steps,
            "train_scope": train_scope,
            "skipped_crops": skipped_crops,
            "elapsed_seconds": elapsed,
            "elapsed_human": _duration(elapsed),
            "starting_checkpoint": str(
                starting_checkpoint
            ),
            "best_step": best_step,
            "best_checkpoint": (
                str(best_path)
                if best_path.is_file()
                else None
            ),
            "recovery_dir": str(
                recovery_dir
            ),
            "run_dir": str(run_dir),
            "initial_baseline": (
                initial_baseline
            ),
        }
        _atomic_json(
            run_dir / "summary.json",
            summary,
        )

        print("=" * 122, flush=True)
        print(
            "INVESTIGATION 20 COMPLETE",
            flush=True,
        )
        print("=" * 122, flush=True)
        print(
            f"Steps             : {global_step}",
            flush=True,
        )
        print(
            f"Skipped crops     : {skipped_crops}",
            flush=True,
        )
        print(
            f"Elapsed           : {_duration(elapsed)}",
            flush=True,
        )
        print(
            f"Best step         : {best_step}",
            flush=True,
        )
        print(
            f"Best checkpoint   : "
            f"{summary['best_checkpoint']}",
            flush=True,
        )
        print(
            f"Recovery dir      : {recovery_dir}",
            flush=True,
        )
        print("=" * 122, flush=True)
        return summary

    except BaseException as exc:
        progress.close()
        failure = {
            "status": "failed",
            "global_step": global_step,
            "max_steps": args.max_steps,
            "skipped_crops": skipped_crops,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "run_dir": str(run_dir),
            "recovery_dir": str(
                recovery_dir
            ),
        }
        _atomic_json(
            run_dir / "failure.json",
            failure,
        )
        raise


# ======================================================================================
# CLI
# ======================================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a separator-aware, separation-only late RAG barrier "
            "on Drosophila NIS3D crops using the local CUDA GPU."
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Required. Resume/transfer from --checkpoint. "
            "Pre-Inv20 checkpoints start Inv20 optimizer step at 0; "
            "Inv20 checkpoints resume exactly."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Starting .pt checkpoint or directory containing "
            "best_checkpoint.pt/checkpoint_step_*.pt."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="drosophila_12_separator_barrier_v1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--validation-crops-per-sample",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--samples",
        default=DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--data-dir",
        default="external/NIS3D/NIS3D",
    )
    parser.add_argument(
        "--spacing-xyz",
        default=DEFAULT_SPACING_XYZ,
    )
    parser.add_argument(
        "--crop-shape-zyx",
        default="32,192,192",
    )
    parser.add_argument(
        "--confidence-ignore-margin-um",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--partial-ignore-margin-um",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--train-scope",
        choices=(
            "auto",
            "barrier",
            "edge+barrier",
            "morphology+barrier",
        ),
        default="auto",
        help=(
            "auto trains only newly required morphology pieces plus the "
            "barrier. Legacy RAG and dense geometry always remain frozen."
        ),
    )
    parser.add_argument(
        "--barrier-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the small directly-supervised separator barrier.",
    )
    parser.add_argument(
        "--morphology-lr",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-edges-per-class",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--separator-hard-fraction",
        type=float,
        default=0.75,
        help=(
            "Fraction of the selected negative quota reserved for strong-separator "
            "negative edges, ranked by current base p_merge."
        ),
    )
    parser.add_argument(
        "--hard-negative-probability",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--strong-separator-coverage",
        type=float,
        default=0.50,
        help=(
            "An edge is 'strong separator' when >= this fraction of its "
            "physical A-B contact area has separator probability >= 0.70."
        ),
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.845,
    )

    parser.add_argument(
        "--barrier-aux-weight",
        type=float,
        default=1.0,
        help=(
            "Direct barrier ON/OFF BCE weight. ON=strong-separator GT-negative; "
            "OFF=GT-positive."
        ),
    )
    parser.add_argument(
        "--positive-preservation-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--positive-barrier-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--allowed-positive-accept-drop",
        type=float,
        default=0.02,
        help=(
            "Best-checkpoint safety guard relative to the fixed step-0 "
            "positive acceptance rate at p_merge >= merge threshold."
        ),
    )

    parser.add_argument(
        "--barrier-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--barrier-initial-gate-bias",
        type=float,
        default=-8.0,
        help=(
            "Initial sigmoid gate logit. -8 with max barrier 8 gives ~0.00268 "
            "logit correction while keeping gradients live."
        ),
    )
    parser.add_argument(
        "--barrier-score-scale",
        type=float,
        default=4.0,
        help=(
            "Maps the directly supervised barrier score into the suppressed "
            "sigmoid gate: gate_logit=bias+scale*score."
        ),
    )
    parser.add_argument(
        "--max-barrier-logit",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=230525,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    result = _training_impl(args)
    print(
        json.dumps(
            _jsonable(result),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
