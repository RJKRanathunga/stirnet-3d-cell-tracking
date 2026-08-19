from __future__ import annotations

"""Stage 02: diagnose STIR-Net flow/SDF regularizer failure.

This investigation does NOT modify production losses or Stage-01 checkpoints.

It answers three questions:

1. Oracle compatibility:
   Do the exact GT SDF and GT flow satisfy the current flow-SDF consistency
   and Eikonal assumptions under the same discrete physical-gradient operator?

2. Gradient forensics:
   At the known-good Stage-01 JOINT checkpoint, how large are the gradients
   from the direct objective, flow-SDF consistency, and Eikonal terms, and
   where do they conflict in the shared network?

3. Causal continuation probes:
   Starting from the exact same JOINT checkpoint, what happens when we continue
   training with:
       control      = direct JOINT losses only
       consistency  = direct + flow_sdf_consistency
       eikonal      = direct + eikonal
       both         = direct + both regularizers

Typical usage:

    python investigations/stirnet/02_geometry_regularizer_diagnostics.py

Run only one diagnostic section:

    python investigations/stirnet/02_geometry_regularizer_diagnostics.py --mode oracle
    python investigations/stirnet/02_geometry_regularizer_diagnostics.py --mode gradients
    python investigations/stirnet/02_geometry_regularizer_diagnostics.py --mode probes

Visualize the saved diagnostic artifact in Napari 3-D:

    python investigations/stirnet/02_geometry_regularizer_diagnostics.py --visualize

The script deliberately reuses the Stage-01 model, target loading, losses, and
metrics through a dynamic import of 01_dense_geometry_overfit.py so the
diagnostic cannot silently drift away from the objective that actually failed.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


# =============================================================================
# PATHS / LOAD STAGE-01 DEFINITIONS
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STAGE01_PATH = Path(__file__).with_name("01_dense_geometry_overfit.py")
if not STAGE01_PATH.exists():
    raise FileNotFoundError(f"Missing Stage-01 investigation: {STAGE01_PATH}")

_spec = importlib.util.spec_from_file_location(
    "_stirnet_stage01_dense_geometry_overfit",
    STAGE01_PATH,
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import Stage-01 investigation from {STAGE01_PATH}")

stage01 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stage01
_spec.loader.exec_module(stage01)

from learned.stirnet.model.utils.physical import physical_gradient3d


DEFAULT_SAMPLE_PATH = stage01.DEFAULT_SAMPLE_PATH
DEFAULT_JOINT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "dense_geometry_debug"
    / "latest_success.pt"
)
DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "learned"
    / "stirnet"
    / "regularizer_diagnostics"
)
DEFAULT_PROBE_STEPS = 100
DEFAULT_EVAL_EVERY = 20
DEFAULT_LEARNING_RATE = stage01.DEFAULT_LEARNING_RATE
DEFAULT_AMP_DTYPE = stage01.DEFAULT_AMP_DTYPE
GRAD_CLIP_NORM = stage01.GRAD_CLIP_NORM
WEIGHT_DECAY = stage01.WEIGHT_DECAY

# The diagnostic uses these thresholds only to define interpretable regions.
SURFACE_SOFT_THRESHOLD = 0.05
SEPARATOR_SOFT_THRESHOLD = 0.05
MEDIAL_SEED_THRESHOLD = 0.85
MIN_DEFINED_GRAD_NORM = 0.10
MIN_DEFINED_FLOW_NORM = 0.10

# Napari vector sampling. These are display-only.
VECTOR_STRIDE_ZYX = (2, 8, 8)
FLOW_VECTOR_DISPLAY_LENGTH_UM = 3.0

DIRECT_NAMES = tuple(stage01._STAGE_LOSS_NAMES["joint"])
FULL_NAMES = tuple(stage01._STAGE_LOSS_NAMES["full"])

BOUNDARY_NAMES = (
    "foreground_bce",
    "foreground_dice",
    "surface_bce",
    "surface_dice",
    "separator_bce",
    "separator_dice",
)
SDF_NAMES = ("sdf",)
FLOW_NAMES = (
    "flow_direction",
    "flow_l1",
    "flow_magnitude",
    "flow_background",
)
CENTER_NAMES = ("centroid_offset", "seed")
CONSISTENCY_NAMES = ("flow_sdf_consistency",)
EIKONAL_NAMES = ("eikonal",)

PROBE_VARIANTS = {
    "control": DIRECT_NAMES,
    "consistency": DIRECT_NAMES + CONSISTENCY_NAMES,
    "eikonal": DIRECT_NAMES + EIKONAL_NAMES,
    "both": FULL_NAMES,
}


# =============================================================================
# SMALL IO / NUMERIC HELPERS
# =============================================================================

def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def scalar(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def tensor_quantiles(values: Tensor, quantiles: Iterable[float]) -> dict[str, float | None]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {f"q{int(round(q * 100)):02d}": None for q in quantiles}
    q_values = list(quantiles)
    q = torch.as_tensor(q_values, device=values.device, dtype=torch.float32)
    out = torch.quantile(values, q)
    return {
        f"q{int(round(frac * 100)):02d}": float(v.item())
        for frac, v in zip(q_values, out)
    }


def masked_values(value: Tensor, mask: Tensor) -> Tensor:
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return value[mask]


def compact_map(value: Tensor) -> Tensor:
    value = value.detach().cpu()
    if value.dtype == torch.bool:
        return value.to(torch.uint8)
    return value.half() if value.is_floating_point() else value


# =============================================================================
# MODEL / CHECKPOINT
# =============================================================================

def load_joint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-01 successful checkpoint: {checkpoint_path}\n"
            "Run the successful Stage-01 joint experiment first."
        )

    payload = torch_load(checkpoint_path)
    checkpoint_stage = payload.get("stage")
    if checkpoint_stage != "joint":
        raise RuntimeError(
            "Stage-02 requires the known-good Stage-01 JOINT checkpoint, but "
            f"{checkpoint_path} contains stage={checkpoint_stage!r}. "
            "Do not diagnose from a different successful stage."
        )
    if "model_state" not in payload:
        raise RuntimeError(f"{checkpoint_path} has no model_state")

    cfg = stage01.build_debug_config()
    model = stage01.DenseGeometryOnlyModel(cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Joint checkpoint does not match the current Stage-01 model.\n"
            f"missing={missing}\nunexpected={unexpected}"
        )
    return model, payload, cfg


# =============================================================================
# REGULARIZER MAPS / REGION DEFINITIONS
# =============================================================================

def regularizer_maps(
    sdf: Tensor,
    flow: Tensor,
    spacing_um: Tensor,
    dref_um: Tensor,
) -> dict[str, Tensor]:
    """Compute the exact geometric quantities behind the two regularizers."""
    sdf_um = sdf.float() * dref_um[:, None, None, None, None].float()
    grad = physical_gradient3d(sdf_um, spacing_um.float()).float()
    grad_norm = torch.linalg.vector_norm(grad, dim=1, keepdim=True)

    flow_float = flow.float()
    flow_norm = torch.linalg.vector_norm(flow_float, dim=1, keepdim=True)

    # Mirrors the current production formulation. Where grad_norm is tiny,
    # the 1e-6 denominator is exactly the conditioning issue we want to expose.
    grad_dir = grad / grad_norm.clamp_min(1e-6)
    flow_dir = flow_float / flow_norm.clamp_min(1e-6)
    cosine = (grad_dir * flow_dir).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cosine))

    return {
        "sdf_grad": grad,
        "sdf_grad_norm": grad_norm,
        "flow_norm": flow_norm,
        "flow_sdf_cosine": cosine,
        "flow_sdf_angle_deg": angle_deg,
        "consistency_error": 1.0 - cosine,
        "eikonal_error": (grad_norm - 1.0).abs(),
        # Approximate local amplification scale introduced by normalization.
        # 0 => norm~1, +2 => norm~0.01, +6 => clamped at 1e-6.
        "normalization_log10_gain": torch.log10(
            1.0 / grad_norm.clamp_min(1e-6)
        ),
    }


def build_region_masks(
    target: dict[str, Tensor],
    target_maps: dict[str, Tensor],
) -> dict[str, Tensor]:
    fg = target["foreground"] > 0.5
    surface = target["surface"] > SURFACE_SOFT_THRESHOLD
    separator = target["separator"] > SEPARATOR_SOFT_THRESHOLD
    medial = target["seed"] >= MEDIAL_SEED_THRESHOLD
    target_flow_norm = torch.linalg.vector_norm(
        target["flow"].float(), dim=1, keepdim=True
    )
    target_grad_norm = target_maps["sdf_grad_norm"]

    grad_defined = target_grad_norm > MIN_DEFINED_GRAD_NORM
    flow_defined = target_flow_norm > MIN_DEFINED_FLOW_NORM

    current_eikonal = target["sdf"] > 0.15
    reliable_common = (
        fg
        & (~surface)
        & (~separator)
        & (~medial)
        & grad_defined
        & flow_defined
    )

    return {
        "foreground": fg,
        "current_consistency_mask": fg,
        "current_eikonal_mask": current_eikonal,
        "near_surface_fg": fg & surface,
        "near_separator_fg": fg & separator,
        "medial_core": fg & medial,
        "low_gt_sdf_gradient": fg & (~grad_defined),
        "defined_gt_flow_gradient": fg & grad_defined & flow_defined,
        "reliable_common": reliable_common,
        "reliable_eikonal": current_eikonal & reliable_common,
    }


def regularizer_region_summary(
    maps: dict[str, Tensor],
    masks: dict[str, Tensor],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    fg_count = int(masks["foreground"].sum().item())

    for name, mask in masks.items():
        count = int(mask.sum().item())
        row: dict[str, object] = {
            "count": count,
            "fraction_of_foreground": (
                float(count / max(fg_count, 1)) if fg_count else 0.0
            ),
        }
        if count:
            grad_norm = masked_values(maps["sdf_grad_norm"], mask)
            eik = masked_values(maps["eikonal_error"], mask)
            cos = masked_values(maps["flow_sdf_cosine"], mask)
            angle = masked_values(maps["flow_sdf_angle_deg"], mask)
            consistency = masked_values(maps["consistency_error"], mask)
            gain = masked_values(maps["normalization_log10_gain"], mask)

            row.update(
                {
                    "grad_norm_mean": float(grad_norm.mean().item()),
                    "grad_norm_quantiles": tensor_quantiles(
                        grad_norm, (0.05, 0.50, 0.90, 0.95)
                    ),
                    "eikonal_error_mean": float(eik.mean().item()),
                    "eikonal_error_quantiles": tensor_quantiles(
                        eik, (0.50, 0.90, 0.95)
                    ),
                    "flow_sdf_cosine_mean": float(cos.mean().item()),
                    "flow_sdf_angle_quantiles_deg": tensor_quantiles(
                        angle, (0.50, 0.90, 0.95)
                    ),
                    "consistency_error_mean": float(consistency.mean().item()),
                    "normalization_log10_gain_quantiles": tensor_quantiles(
                        gain, (0.50, 0.90, 0.95, 0.99)
                    ),
                    "fraction_grad_norm_lt_0p1": float(
                        (grad_norm < MIN_DEFINED_GRAD_NORM).float().mean().item()
                    ),
                    "fraction_grad_norm_gt_2": float(
                        (grad_norm > 2.0).float().mean().item()
                    ),
                }
            )
        result[name] = row
    return result


def print_region_summary(title: str, summary: dict[str, dict]) -> None:
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    print(
        f"{'region':28s} {'voxels':>9s} {'|g| mean':>10s} "
        f"{'Eik mean':>10s} {'cos mean':>10s} {'angle50':>10s} "
        f"{'angle90':>10s} {'low-g%':>9s}"
    )
    print("-" * 112)
    for name, row in summary.items():
        if not row.get("count"):
            print(f"{name:28s} {0:9d}")
            continue
        angle_q = row["flow_sdf_angle_quantiles_deg"]
        low_pct = 100.0 * float(row["fraction_grad_norm_lt_0p1"])
        print(
            f"{name:28s} "
            f"{int(row['count']):9d} "
            f"{float(row['grad_norm_mean']):10.4f} "
            f"{float(row['eikonal_error_mean']):10.4f} "
            f"{float(row['flow_sdf_cosine_mean']):10.4f} "
            f"{float(angle_q['q50']):10.2f} "
            f"{float(angle_q['q90']):10.2f} "
            f"{low_pct:8.2f}%"
        )


def compact_regularizer_maps(maps: dict[str, Tensor]) -> dict[str, Tensor]:
    keep = (
        "sdf_grad_norm",
        "flow_norm",
        "flow_sdf_cosine",
        "flow_sdf_angle_deg",
        "consistency_error",
        "eikonal_error",
        "normalization_log10_gain",
    )
    return {name: compact_map(maps[name][0, 0]) for name in keep}


# =============================================================================
# DIAGNOSTIC A — GT ORACLE COMPATIBILITY
# =============================================================================

@torch.no_grad()
def run_oracle_diagnostic(
    batch,
    cfg,
) -> tuple[dict, dict[str, Tensor], dict[str, Tensor]]:
    target_maps = regularizer_maps(
        batch.targets["sdf"],
        batch.targets["flow"],
        batch.spacing_um,
        batch.dref_um,
    )
    masks = build_region_masks(batch.targets, target_maps)
    regions = regularizer_region_summary(target_maps, masks)

    fg = masks["current_consistency_mask"]
    eik = masks["current_eikonal_mask"]

    exact_consistency_unweighted = float(
        target_maps["consistency_error"][fg].mean().item()
    )
    exact_eikonal_unweighted = float(
        target_maps["eikonal_error"][eik].mean().item()
    )

    summary = {
        "purpose": (
            "Evaluate current regularizer assumptions on exact GT targets using "
            "the same discrete physical-gradient operator as production."
        ),
        "current_loss_oracle_values": {
            "flow_sdf_consistency_unweighted": exact_consistency_unweighted,
            "flow_sdf_consistency_weighted": (
                exact_consistency_unweighted * cfg.geometry.consistency_weight
            ),
            "eikonal_unweighted": exact_eikonal_unweighted,
            "eikonal_weighted": (
                exact_eikonal_unweighted * cfg.geometry.eikonal_weight
            ),
        },
        "regions": regions,
    }

    print_region_summary("Diagnostic A — GT oracle compatibility", regions)
    print("\nCurrent production masks applied to perfect GT:")
    print(
        "  flow_sdf_consistency "
        f"unweighted={exact_consistency_unweighted:.6f} "
        f"weighted={summary['current_loss_oracle_values']['flow_sdf_consistency_weighted']:.6f}"
    )
    print(
        "  eikonal             "
        f"unweighted={exact_eikonal_unweighted:.6f} "
        f"weighted={summary['current_loss_oracle_values']['eikonal_weighted']:.6f}"
    )

    compact_masks = {
        name: compact_map(mask[0, 0])
        for name, mask in masks.items()
    }
    return summary, compact_regularizer_maps(target_maps), compact_masks


# =============================================================================
# DIAGNOSTIC B — GRADIENT FORENSICS AT JOINT CHECKPOINT
# =============================================================================

def parameter_group_indices(model: torch.nn.Module) -> tuple[list[str], dict[str, list[int]]]:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    groups: dict[str, list[int]] = {
        "all": list(range(len(names))),
        "shared_all": [],
        "acquisition": [],
        "evidence_stem": [],
        "spatial_backbone": [],
        "geometry_shared": [],
        "head_foreground": [],
        "head_surface": [],
        "head_separator": [],
        "head_sdf": [],
        "head_flow": [],
        "head_centroid_offset": [],
        "head_seed": [],
    }

    for i, name in enumerate(names):
        if name.startswith("acquisition."):
            groups["acquisition"].append(i)
            groups["shared_all"].append(i)
        elif name.startswith("evidence_stem."):
            groups["evidence_stem"].append(i)
            groups["shared_all"].append(i)
        elif name.startswith("spatial_backbone."):
            groups["spatial_backbone"].append(i)
            groups["shared_all"].append(i)
        elif name.startswith("geometry_decoder.input_proj.") or name.startswith(
            "geometry_decoder.blocks."
        ):
            groups["geometry_shared"].append(i)
            groups["shared_all"].append(i)
        elif name.startswith("geometry_decoder.foreground."):
            groups["head_foreground"].append(i)
        elif name.startswith("geometry_decoder.surface."):
            groups["head_surface"].append(i)
        elif name.startswith("geometry_decoder.separator."):
            groups["head_separator"].append(i)
        elif name.startswith("geometry_decoder.sdf."):
            groups["head_sdf"].append(i)
        elif name.startswith("geometry_decoder.flow."):
            groups["head_flow"].append(i)
        elif name.startswith("geometry_decoder.centroid_offset."):
            groups["head_centroid_offset"].append(i)
        elif name.startswith("geometry_decoder.seed."):
            groups["head_seed"].append(i)

    return names, groups


def cpu_gradient_snapshot(
    model: torch.nn.Module,
    batch,
    cfg,
    loss_names: tuple[str, ...],
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[float, tuple[Tensor | None, ...]]:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)

    with stage01.autocast_context(
        batch.spatial_inputs.device,
        amp_dtype,
        amp_enabled,
    ):
        pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
        terms = stage01.geometry_loss_terms(
            pred,
            batch.targets,
            batch.spacing_um,
            batch.dref_um,
            cfg,
            requested=set(loss_names),
        )
        loss = sum(terms[name] for name in loss_names)

    gradients = torch.autograd.grad(
        loss,
        params,
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    cpu = tuple(
        None if grad is None else grad.detach().float().cpu()
        for grad in gradients
    )
    value = float(loss.detach().float().cpu().item())

    del pred, terms, loss, gradients
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value, cpu


def grad_norm(grads: tuple[Tensor | None, ...], indices: list[int]) -> float:
    total = 0.0
    for i in indices:
        grad = grads[i]
        if grad is not None:
            total += float(grad.double().square().sum().item())
    return math.sqrt(total)


def grad_pair(
    a: tuple[Tensor | None, ...],
    b: tuple[Tensor | None, ...],
    indices: list[int],
) -> dict[str, float | None]:
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for i in indices:
        ga = a[i]
        gb = b[i]
        if ga is None or gb is None:
            continue
        ga64 = ga.double()
        gb64 = gb.double()
        dot += float((ga64 * gb64).sum().item())
        aa += float(ga64.square().sum().item())
        bb += float(gb64.square().sum().item())

    norm_a = math.sqrt(aa)
    norm_b = math.sqrt(bb)
    cosine = (
        dot / max(norm_a * norm_b, 1e-30)
        if norm_a > 0 and norm_b > 0
        else float("nan")
    )
    return {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "dot": dot,
        "cosine": finite_or_none(cosine),
    }


def combined_grad_norm(
    gradient_sets: list[tuple[Tensor | None, ...]],
    indices: list[int],
) -> float:
    total = 0.0
    for i in indices:
        active = [g[i] for g in gradient_sets if g[i] is not None]
        if not active:
            continue
        value = active[0].double().clone()
        for grad in active[1:]:
            value.add_(grad.double())
        total += float(value.square().sum().item())
    return math.sqrt(total)


def run_gradient_forensics(
    model: torch.nn.Module,
    batch,
    cfg,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict:
    _, groups = parameter_group_indices(model)

    loss_groups = {
        "direct_all": DIRECT_NAMES,
        "boundary_direct": BOUNDARY_NAMES,
        "sdf_direct": SDF_NAMES,
        "flow_direct": FLOW_NAMES,
        "center_direct": CENTER_NAMES,
        "consistency": CONSISTENCY_NAMES,
        "eikonal": EIKONAL_NAMES,
    }

    loss_values: dict[str, float] = {}
    snapshots: dict[str, tuple[Tensor | None, ...]] = {}

    print("\n" + "=" * 112)
    print("Diagnostic B — gradient forensics at the successful JOINT checkpoint")
    print("=" * 112)

    for label, names in loss_groups.items():
        value, grads = cpu_gradient_snapshot(
            model,
            batch,
            cfg,
            names,
            amp_dtype,
            amp_enabled,
        )
        loss_values[label] = value
        snapshots[label] = grads
        print(
            f"{label:20s} loss={value:11.6f} "
            f"grad_norm(all)={grad_norm(grads, groups['all']):11.6f} "
            f"grad_norm(shared)={grad_norm(grads, groups['shared_all']):11.6f}"
        )

    norms: dict[str, dict[str, float]] = {}
    for loss_name, grads in snapshots.items():
        norms[loss_name] = {
            group_name: grad_norm(grads, indices)
            for group_name, indices in groups.items()
        }

    pairs: dict[str, dict[str, dict[str, float | None]]] = {}
    for reg in ("consistency", "eikonal"):
        pairs[reg] = {}
        for direct in (
            "direct_all",
            "boundary_direct",
            "sdf_direct",
            "flow_direct",
            "center_direct",
        ):
            key = f"{reg}_vs_{direct}"
            pairs[reg][key] = {
                group_name: grad_pair(
                    snapshots[reg],
                    snapshots[direct],
                    indices,
                )
                for group_name, indices in groups.items()
            }

    full_preclip_norm = combined_grad_norm(
        [
            snapshots["direct_all"],
            snapshots["consistency"],
            snapshots["eikonal"],
        ],
        groups["all"],
    )
    direct_preclip_norm = norms["direct_all"]["all"]

    print("\nKey gradient conflicts in shared parameters:")
    for reg in ("consistency", "eikonal"):
        row = pairs[reg][f"{reg}_vs_direct_all"]["shared_all"]
        print(
            f"  {reg:12s} vs direct: "
            f"cos={row['cosine']!s:>10s} "
            f"|reg|={row['norm_a']:.6f} "
            f"|direct|={row['norm_b']:.6f}"
        )

    print("\nGlobal clipping context:")
    print(f"  direct-only preclip norm : {direct_preclip_norm:.6f}")
    print(f"  full combined preclip    : {full_preclip_norm:.6f}")
    print(f"  Stage-01 clip threshold  : {GRAD_CLIP_NORM:.6f}")

    result = {
        "loss_values": loss_values,
        "gradient_norms": norms,
        "gradient_pairs": pairs,
        "combined_preclip_norms": {
            "direct_only": direct_preclip_norm,
            "direct_plus_consistency_plus_eikonal": full_preclip_norm,
            "stage01_clip_threshold": GRAD_CLIP_NORM,
        },
    }

    del snapshots
    return result


# =============================================================================
# BASE JOINT STATE / PROBE MAPS
# =============================================================================

@torch.no_grad()
def evaluate_model(model, batch, cfg) -> tuple[dict, dict[str, Tensor], dict]:
    model.eval()
    pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
    metrics = stage01.collect_metrics(pred, batch.targets, batch.dref_um)

    maps = regularizer_maps(
        pred.sdf,
        pred.flow,
        batch.spacing_um,
        batch.dref_um,
    )
    full_terms = stage01.geometry_loss_terms(
        pred,
        batch.targets,
        batch.spacing_um,
        batch.dref_um,
        cfg,
        requested=set(FULL_NAMES),
    )
    direct_total = sum(full_terms[name] for name in DIRECT_NAMES)

    loss_summary = {
        "direct_total": scalar(direct_total),
        "flow_sdf_consistency": scalar(full_terms["flow_sdf_consistency"]),
        "eikonal": scalar(full_terms["eikonal"]),
        "full_total": scalar(sum(full_terms[name] for name in FULL_NAMES)),
    }
    return metrics, compact_regularizer_maps(maps), {
        "predictions": stage01.compact_predictions(pred),
        "loss_summary": loss_summary,
    }


# =============================================================================
# DIAGNOSTIC C — CAUSAL CONTINUATION PROBES
# =============================================================================

def run_one_probe(
    name: str,
    loss_names: tuple[str, ...],
    checkpoint_path: Path,
    batch,
    probe_steps: int,
    eval_every: int,
    learning_rate: float,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[dict, dict]:
    device = batch.spatial_inputs.device

    # Every probe starts from the exact same model weights and a fresh AdamW
    # optimizer. The fresh optimizer state is intentional and identical across
    # all four causal conditions.
    model, _, cfg = load_joint_model(checkpoint_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    history: list[dict] = []

    @torch.no_grad()
    def snapshot(step: int, last_grad_norm: float | None) -> None:
        model.eval()
        with stage01.autocast_context(device, amp_dtype, amp_enabled):
            pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)

        metrics = stage01.collect_metrics(pred, batch.targets, batch.dref_um)
        all_terms = stage01.geometry_loss_terms(
            pred,
            batch.targets,
            batch.spacing_um,
            batch.dref_um,
            cfg,
            requested=set(FULL_NAMES),
        )
        direct_loss = sum(all_terms[n] for n in DIRECT_NAMES)
        active_loss = sum(all_terms[n] for n in loss_names)
        joint_pass, joint_failures = stage01.pass_report("joint", metrics)

        row = {
            "step": step,
            "active_loss": scalar(active_loss),
            "direct_loss": scalar(direct_loss),
            "flow_sdf_consistency": scalar(all_terms["flow_sdf_consistency"]),
            "eikonal": scalar(all_terms["eikonal"]),
            "last_preclip_grad_norm": last_grad_norm,
            "joint_acceptance": bool(joint_pass),
            "joint_failures": joint_failures,
            "metrics": metrics,
        }
        history.append(row)

        print(
            f"[{name:11s}] step={step:03d} "
            f"active={row['active_loss']:.4f} "
            f"direct={row['direct_loss']:.4f} "
            f"cons={row['flow_sdf_consistency']:.4f} "
            f"eik={row['eikonal']:.4f} "
            f"fg={metrics['foreground_dice']:.4f} "
            f"surf={metrics['surface_dice']:.4f} "
            f"sep={metrics['separator_dice']:.4f} "
            f"sdfCorr={metrics['sdf_corr']:.4f} "
            f"flowCos={metrics['flow_cosine']:.4f} "
            f"cent={metrics['centroid_endpoint_median_um']:.3f}um "
            f"jointPASS={joint_pass}"
        )

    snapshot(0, None)

    start = time.perf_counter()
    last_preclip_grad_norm: float | None = None

    for step in range(1, probe_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with stage01.autocast_context(device, amp_dtype, amp_enabled):
            pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
            terms = stage01.geometry_loss_terms(
                pred,
                batch.targets,
                batch.spacing_um,
                batch.dref_um,
                cfg,
                requested=set(loss_names),
            )
            loss = sum(terms[n] for n in loss_names)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Probe {name} produced non-finite loss at step {step}: "
                f"{float(loss.detach().float().cpu())}"
            )

        loss.backward()
        preclip = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM,
        )
        last_preclip_grad_norm = float(preclip.detach().float().cpu().item())
        optimizer.step()

        if step % eval_every == 0 or step == probe_steps:
            snapshot(step, last_preclip_grad_norm)

        del pred, terms, loss

    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad(), stage01.autocast_context(device, amp_dtype, amp_enabled):
        final_pred = model(batch.spatial_inputs, batch.spacing_um, batch.dref_um)
    final_metrics = stage01.collect_metrics(
        final_pred,
        batch.targets,
        batch.dref_um,
    )
    final_maps = regularizer_maps(
        final_pred.sdf,
        final_pred.flow,
        batch.spacing_um,
        batch.dref_um,
    )

    artifact = {
        "predictions": stage01.compact_predictions(final_pred),
        "regularizer_maps": compact_regularizer_maps(final_maps),
    }
    summary = {
        "name": name,
        "active_losses": list(loss_names),
        "probe_steps": probe_steps,
        "elapsed_seconds": elapsed,
        "history": history,
        "final_metrics": final_metrics,
    }

    del model, optimizer, final_pred, final_maps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, artifact


def run_probe_suite(
    checkpoint_path: Path,
    batch,
    probe_steps: int,
    eval_every: int,
    learning_rate: float,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[dict, dict]:
    print("\n" + "=" * 112)
    print("Diagnostic C — causal continuation probes from the same JOINT checkpoint")
    print("=" * 112)
    print(
        f"Probe steps={probe_steps}, eval_every={eval_every}, "
        f"lr={learning_rate:.3g}, clip={GRAD_CLIP_NORM}"
    )

    summaries: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}

    for name, names in PROBE_VARIANTS.items():
        print("\n" + "-" * 112)
        print(f"Starting probe: {name}")
        print("Active losses: " + ", ".join(names))
        print("-" * 112)

        summary, artifact = run_one_probe(
            name,
            names,
            checkpoint_path,
            batch,
            probe_steps,
            eval_every,
            learning_rate,
            amp_dtype,
            amp_enabled,
        )
        summaries[name] = summary
        artifacts[name] = artifact

    return summaries, artifacts


# =============================================================================
# NAPARI VISUALIZATION
# =============================================================================

def _as_numpy(value) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prediction_numpy(predictions: dict, name: str) -> np.ndarray:
    return _as_numpy(predictions[name]).astype(np.float32, copy=False)


def _add_probability(viewer, value, name: str, scale, visible=False) -> None:
    viewer.add_image(
        _as_numpy(value).astype(np.float32, copy=False),
        name=name,
        scale=scale,
        colormap="viridis",
        contrast_limits=(0.0, 1.0),
        visible=visible,
    )


def _add_scalar(viewer, value, name: str, scale, visible=False, colormap="viridis") -> None:
    array = _as_numpy(value).astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    if finite.size:
        lo = float(np.quantile(finite, 0.01))
        hi = float(np.quantile(finite, 0.99))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        contrast = (lo, hi)
    else:
        contrast = None

    viewer.add_image(
        array,
        name=name,
        scale=scale,
        colormap=colormap,
        contrast_limits=contrast,
        visible=visible,
    )


def _vector_samples(
    field_3zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    kind: str,
    dref_um: float,
) -> np.ndarray:
    z_step, y_step, x_step = VECTOR_STRIDE_ZYX
    zz = np.arange(0, field_3zyx.shape[1], z_step)
    yy = np.arange(0, field_3zyx.shape[2], y_step)
    xx = np.arange(0, field_3zyx.shape[3], x_step)
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    starts = np.stack([z.reshape(-1), y.reshape(-1), x.reshape(-1)], axis=1)

    keep = mask_zyx[
        starts[:, 0],
        starts[:, 1],
        starts[:, 2],
    ]
    starts = starts[keep]
    if starts.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    vectors = field_3zyx[
        :,
        starts[:, 0],
        starts[:, 1],
        starts[:, 2],
    ].T.astype(np.float32, copy=False)

    if kind == "flow":
        physical_delta_um = vectors * FLOW_VECTOR_DISPLAY_LENGTH_UM
    elif kind == "centroid":
        physical_delta_um = vectors * float(dref_um)
    else:
        raise ValueError(kind)

    voxel_delta = physical_delta_um / spacing_zyx[None, :]
    return np.stack(
        [starts.astype(np.float32), voxel_delta.astype(np.float32)],
        axis=1,
    )


def _add_prediction_bundle(
    viewer,
    prefix: str,
    predictions: dict,
    regularizer_map_payload: dict | None,
    fg_mask: np.ndarray,
    scale,
    spacing_np: np.ndarray,
    dref_um: float,
) -> None:
    foreground = _prediction_numpy(predictions, "foreground")
    surface = _prediction_numpy(predictions, "surface")
    separator = _prediction_numpy(predictions, "separator")
    sdf = _prediction_numpy(predictions, "sdf")
    flow = _prediction_numpy(predictions, "flow")
    centroid = _prediction_numpy(predictions, "centroid_offset")
    seed = _prediction_numpy(predictions, "seed")

    _add_probability(viewer, foreground, f"{prefix} — foreground", scale)
    _add_probability(viewer, surface, f"{prefix} — surface", scale)
    _add_probability(viewer, separator, f"{prefix} — separator", scale)
    _add_scalar(viewer, sdf, f"{prefix} — SDF", scale)
    _add_probability(viewer, seed, f"{prefix} — seed", scale)

    flow_mag = np.linalg.norm(flow, axis=0)
    centroid_mag_um = np.linalg.norm(centroid, axis=0) * dref_um
    _add_scalar(viewer, flow_mag, f"{prefix} — flow magnitude", scale)
    _add_scalar(
        viewer,
        centroid_mag_um,
        f"{prefix} — centroid offset magnitude (um, raw)",
        scale,
    )

    flow_vectors = _vector_samples(
        flow,
        fg_mask,
        spacing_np,
        kind="flow",
        dref_um=dref_um,
    )
    if len(flow_vectors):
        viewer.add_vectors(
            flow_vectors,
            name=f"{prefix} — flow vectors (GT foreground sample)",
            scale=scale,
            visible=False,
        )

    centroid_vectors = _vector_samples(
        centroid,
        fg_mask,
        spacing_np,
        kind="centroid",
        dref_um=dref_um,
    )
    if len(centroid_vectors):
        viewer.add_vectors(
            centroid_vectors,
            name=f"{prefix} — centroid vectors (GT foreground sample)",
            scale=scale,
            visible=False,
        )

    if regularizer_map_payload:
        _add_scalar(
            viewer,
            regularizer_map_payload["sdf_grad_norm"],
            f"{prefix} — |grad SDF|",
            scale,
        )
        _add_scalar(
            viewer,
            regularizer_map_payload["eikonal_error"],
            f"{prefix} — Eikonal abs error",
            scale,
        )
        _add_scalar(
            viewer,
            regularizer_map_payload["flow_sdf_angle_deg"],
            f"{prefix} — flow/SDF angle (deg)",
            scale,
        )
        _add_scalar(
            viewer,
            regularizer_map_payload["normalization_log10_gain"],
            f"{prefix} — log10 normalization gain",
            scale,
        )


def visualize_diagnostics(
    sample_path: Path,
    diagnostic_path: Path,
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for --visualize. Install it in the local environment."
        ) from exc

    if not diagnostic_path.exists():
        raise FileNotFoundError(
            f"Missing diagnostic artifact: {diagnostic_path}\n"
            "Run Stage-02 diagnostics first."
        )
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing Stage-00 sample: {sample_path}")

    payload = torch_load(diagnostic_path)
    sample = torch_load(sample_path)

    spatial = _as_numpy(sample["spatial_inputs"]).astype(np.float32, copy=False)
    current = _as_numpy(sample["current_labels"])
    gt = _as_numpy(sample["gt_labels"])
    targets = sample["geometry_targets"]

    spacing_np = _as_numpy(sample["spacing_um"]).astype(np.float32).reshape(3)
    scale = tuple(float(v) for v in spacing_np)
    dref_um = float(_as_numpy(sample["dref_um"]).reshape(()))
    fg_mask = _as_numpy(targets["foreground"])[0] > 0.5

    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(
        spatial[0],
        name="00 Raw",
        scale=scale,
        colormap="gray",
        visible=True,
    )
    viewer.add_labels(
        current.astype(np.int32),
        name="01 Current labels",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        gt.astype(np.int32),
        name="02 GT labels",
        scale=scale,
        visible=False,
    )

    _add_probability(
        viewer,
        _as_numpy(targets["foreground"])[0],
        "03 Target — foreground",
        scale,
    )
    _add_probability(
        viewer,
        _as_numpy(targets["surface"])[0],
        "04 Target — surface",
        scale,
    )
    _add_probability(
        viewer,
        _as_numpy(targets["separator"])[0],
        "05 Target — separator",
        scale,
    )
    _add_scalar(
        viewer,
        _as_numpy(targets["sdf"])[0],
        "06 Target — SDF",
        scale,
    )
    _add_probability(
        viewer,
        _as_numpy(targets["seed"])[0],
        "07 Target — seed",
        scale,
    )

    target_flow = _as_numpy(targets["flow"]).astype(np.float32, copy=False)
    target_centroid = _as_numpy(targets["centroid_offset"]).astype(np.float32, copy=False)
    _add_scalar(
        viewer,
        np.linalg.norm(target_flow, axis=0),
        "08 Target — flow magnitude",
        scale,
    )
    _add_scalar(
        viewer,
        np.linalg.norm(target_centroid, axis=0) * dref_um,
        "09 Target — centroid offset magnitude (um)",
        scale,
    )

    oracle = payload.get("oracle_artifact")
    if oracle:
        for name, value in oracle.get("maps", {}).items():
            _add_scalar(
                viewer,
                value,
                f"10 Oracle GT — {name}",
                scale,
            )
        for name, value in oracle.get("masks", {}).items():
            viewer.add_image(
                _as_numpy(value).astype(np.uint8),
                name=f"11 Oracle mask — {name}",
                scale=scale,
                colormap="gray",
                contrast_limits=(0, 1),
                visible=False,
            )

    base = payload.get("base_joint")
    if base:
        _add_prediction_bundle(
            viewer,
            "20 Base JOINT",
            base["predictions"],
            base.get("regularizer_maps"),
            fg_mask,
            scale,
            spacing_np,
            dref_um,
        )

    probes = payload.get("probe_artifacts", {})
    base_pred = None if base is None else base["predictions"]
    order = ["control", "consistency", "eikonal", "both"]
    for i, name in enumerate(order):
        if name not in probes:
            continue
        artifact = probes[name]
        prefix = f"{30 + i * 10:02d} Probe {name}"
        _add_prediction_bundle(
            viewer,
            prefix,
            artifact["predictions"],
            artifact.get("regularizer_maps"),
            fg_mask,
            scale,
            spacing_np,
            dref_um,
        )

        if base_pred is not None:
            for field in ("foreground", "separator", "sdf"):
                probe_field = _prediction_numpy(artifact["predictions"], field)
                base_field = _prediction_numpy(base_pred, field)
                _add_scalar(
                    viewer,
                    np.abs(probe_field - base_field),
                    f"{prefix} — abs delta {field} vs base",
                    scale,
                )

            probe_flow = _prediction_numpy(artifact["predictions"], "flow")
            base_flow = _prediction_numpy(base_pred, "flow")
            flow_delta = np.linalg.norm(probe_flow - base_flow, axis=0)
            _add_scalar(
                viewer,
                flow_delta,
                f"{prefix} — flow vector delta magnitude vs base",
                scale,
            )

            probe_cent = _prediction_numpy(
                artifact["predictions"], "centroid_offset"
            )
            base_cent = _prediction_numpy(base_pred, "centroid_offset")
            cent_delta_um = (
                np.linalg.norm(probe_cent - base_cent, axis=0) * dref_um
            )
            _add_scalar(
                viewer,
                cent_delta_um,
                f"{prefix} — centroid vector delta (um) vs base",
                scale,
            )

    print("\nNapari diagnostic view")
    print("  All scalar prediction layers show raw behavior; no prediction field is masked.")
    print("  Vector layers are spatially subsampled only to keep the 3-D view readable.")
    print("  Scalar centroid magnitude is the raw full-volume prediction.")
    napari.run()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose STIR-Net flow/SDF consistency and Eikonal regularizers."
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "gradients", "probes", "all"),
        default="all",
        help="Diagnostic section to run. Default: all.",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=DEFAULT_JOINT_CHECKPOINT,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=DEFAULT_PROBE_STEPS,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default=DEFAULT_AMP_DTYPE,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the saved Stage-02 artifact in Napari 3-D and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = args.result_dir / "regularizer_diagnostics.pt"
    json_path = args.result_dir / "regularizer_diagnostics.json"

    if args.visualize:
        visualize_diagnostics(args.sample_path, artifact_path)
        return

    if args.probe_steps < 1:
        raise ValueError("--probe-steps must be >= 1")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage-02 training/gradient diagnostics require CUDA. "
            "Napari visualization itself does not."
        )

    device = torch.device("cuda")
    stage01.set_seed(stage01.DEFAULT_SEED)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    amp_dtype, amp_enabled = stage01.resolve_amp(args.amp_dtype, device)
    cfg = stage01.build_debug_config()
    batch = stage01.load_debug_batch(args.sample_path, device)

    model, checkpoint_payload, _ = load_joint_model(
        args.joint_checkpoint,
        device,
    )

    print("\n" + "=" * 112)
    print("STIR-Net Stage 02 — flow/SDF regularizer diagnostics")
    print("=" * 112)
    print(f"Mode                   : {args.mode}")
    print(f"Sample                 : {args.sample_path}")
    print(f"Joint checkpoint       : {args.joint_checkpoint}")
    print(f"Checkpoint stage       : {checkpoint_payload.get('stage')}")
    print(f"Crop shape             : {tuple(batch.spatial_inputs.shape[-3:])}")
    print(
        "Spacing (z,y,x) um    : "
        + ", ".join(f"{float(v):.6g}" for v in batch.spacing_um[0])
    )
    print(f"dref um                : {float(batch.dref_um[0]):.6f}")
    print(f"AMP                    : {args.amp_dtype}")
    print(
        f"Regularizer weights    : consistency={cfg.geometry.consistency_weight}, "
        f"eikonal={cfg.geometry.eikonal_weight}"
    )
    print(f"Stage-01 grad clip     : {GRAD_CLIP_NORM}")
    if args.mode in {"probes", "all"}:
        print(
            f"Probe plan             : 4 x {args.probe_steps} steps "
            f"(eval every {args.eval_every})"
        )
    print("=" * 112)

    summary: dict[str, object] = {
        "format_version": 1,
        "kind": "stirnet_geometry_regularizer_diagnostics",
        "mode": args.mode,
        "sample_path": str(args.sample_path),
        "joint_checkpoint": str(args.joint_checkpoint),
        "joint_checkpoint_stage": checkpoint_payload.get("stage"),
        "probe_steps": args.probe_steps,
        "eval_every": args.eval_every,
        "learning_rate": args.learning_rate,
        "amp_dtype": args.amp_dtype,
        "regularizer_weights": {
            "consistency": cfg.geometry.consistency_weight,
            "eikonal": cfg.geometry.eikonal_weight,
        },
        "thresholds": {
            "surface_soft": SURFACE_SOFT_THRESHOLD,
            "separator_soft": SEPARATOR_SOFT_THRESHOLD,
            "medial_seed": MEDIAL_SEED_THRESHOLD,
            "defined_grad_norm": MIN_DEFINED_GRAD_NORM,
            "defined_flow_norm": MIN_DEFINED_FLOW_NORM,
        },
    }

    artifact: dict[str, object] = {
        "format_version": 1,
        "kind": "stirnet_geometry_regularizer_diagnostics",
        "spacing_um": batch.spacing_um[0].detach().cpu(),
        "dref_um": batch.dref_um[0].detach().cpu(),
        "sample_shape_zyx": list(batch.spatial_inputs.shape[-3:]),
        "sample_selection": batch.selection,
    }

    # Always save the known-good base JOINT prediction so every probe can be
    # inspected against the exact starting point.
    with stage01.autocast_context(device, amp_dtype, amp_enabled):
        base_metrics, base_maps, base_extra = evaluate_model(model, batch, cfg)
    summary["base_joint"] = {
        "metrics": base_metrics,
        "loss_summary": base_extra["loss_summary"],
    }
    artifact["base_joint"] = {
        "predictions": base_extra["predictions"],
        "regularizer_maps": base_maps,
    }

    if args.mode in {"oracle", "all"}:
        oracle_summary, oracle_maps, oracle_masks = run_oracle_diagnostic(
            batch,
            cfg,
        )
        summary["oracle"] = oracle_summary
        artifact["oracle_artifact"] = {
            "maps": oracle_maps,
            "masks": oracle_masks,
        }

    if args.mode in {"gradients", "all"}:
        summary["gradient_forensics"] = run_gradient_forensics(
            model,
            batch,
            cfg,
            amp_dtype,
            amp_enabled,
        )

    # Free the base model before the sequential training probes.
    del model
    torch.cuda.empty_cache()

    if args.mode in {"probes", "all"}:
        probe_summaries, probe_artifacts = run_probe_suite(
            args.joint_checkpoint,
            batch,
            args.probe_steps,
            args.eval_every,
            args.learning_rate,
            amp_dtype,
            amp_enabled,
        )
        summary["probes"] = probe_summaries
        artifact["probe_artifacts"] = probe_artifacts

    allocated, reserved = stage01.gpu_peak_gib()
    summary["peak_cuda_allocated_gib"] = allocated
    summary["peak_cuda_reserved_gib"] = reserved
    summary["timestamp_unix"] = time.time()

    atomic_json(json_path, summary)
    atomic_torch_save(artifact_path, artifact)

    print("\n" + "=" * 112)
    print("Stage-02 diagnostics complete")
    print("=" * 112)
    print(f"JSON summary            : {json_path}")
    print(f"Napari artifact         : {artifact_path}")
    print(
        f"Peak CUDA memory        : {allocated:.3f} GiB allocated / "
        f"{reserved:.3f} GiB reserved"
    )
    print("\nVisualize with:")
    print(
        "  python investigations/stirnet/"
        "02_geometry_regularizer_diagnostics.py --visualize"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
