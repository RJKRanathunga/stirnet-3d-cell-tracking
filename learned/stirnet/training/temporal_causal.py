from __future__ import annotations

"""Causal temporal negatives for production STIR-Net training.

Investigation 31 established the training invariant:

    correct temporal content
        may override the spatial RAG

    corrupted temporal content
        must regress to the spatial RAG

This module is training-only. It does not alter inference behavior or model
parameter shapes.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F

from ..model.types import ReasoningState, StirNetOutput, TemporalState

if TYPE_CHECKING:
    from ..model import StirNet
    from ..model.partition import RAGCriterion
    from .config import LossConfig


CAUSAL_TEMPORAL_CORRUPTIONS = frozenset({"contentless", "shuffled"})


def contentless_temporal_state(temporal: TemporalState) -> TemporalState:
    """Erase track/history content while retaining temporal support geometry."""
    return TemporalState(
        tokens=torch.zeros_like(temporal.tokens),
        ref_um=temporal.ref_um,
        batch_index=temporal.batch_index,
        salience=temporal.salience,
        reliability=temporal.reliability,
        status=temporal.status,
        node_tokens=(
            None
            if temporal.node_tokens is None
            else torch.zeros_like(temporal.node_tokens)
        ),
    )


def _batchwise_shuffle_permutation(
    batch_index: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Shuffle only within each batch row; never mix unrelated samples."""
    count = int(batch_index.numel())
    permutation = torch.arange(
        count,
        device=batch_index.device,
        dtype=torch.long,
    )
    if count <= 1:
        return permutation

    batch_cpu = batch_index.detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for batch_id in torch.unique(batch_cpu, sorted=True).tolist():
        rows_cpu = torch.nonzero(
            batch_cpu == int(batch_id),
            as_tuple=False,
        ).flatten()
        if rows_cpu.numel() <= 1:
            continue
        local = torch.randperm(
            int(rows_cpu.numel()),
            generator=generator,
        )
        target_rows = rows_cpu.to(batch_index.device)
        source_rows = rows_cpu[local].to(batch_index.device)
        permutation[target_rows] = source_rows

    return permutation


def shuffled_temporal_state(
    temporal: TemporalState,
    *,
    seed: int,
) -> TemporalState:
    """Break tracklet-content/location correspondence inside each batch row.

    Physical reference positions remain fixed. Detection-level ``node_tokens``
    cannot safely use the tracklet permutation and therefore remain unchanged,
    matching the successful Investigation-31 ablation.
    """
    permutation = _batchwise_shuffle_permutation(
        temporal.batch_index,
        seed=seed,
    )
    return TemporalState(
        tokens=temporal.tokens[permutation],
        ref_um=temporal.ref_um,
        batch_index=temporal.batch_index,
        salience=temporal.salience[permutation],
        reliability=temporal.reliability[permutation],
        status=temporal.status[permutation],
        node_tokens=temporal.node_tokens,
    )


def corrupt_temporal_state(
    temporal: TemporalState,
    *,
    corruption: str,
    seed: int,
) -> TemporalState:
    corruption = str(corruption).lower()
    if corruption == "contentless":
        return contentless_temporal_state(temporal)
    if corruption == "shuffled":
        return shuffled_temporal_state(temporal, seed=seed)
    raise ValueError(
        f"Unknown temporal corruption {corruption!r}; "
        f"expected one of {sorted(CAUSAL_TEMPORAL_CORRUPTIONS)}"
    )


@dataclass(frozen=True)
class TemporalCausalTerms:
    total: Tensor
    noop: Tensor
    corrupted_gate: Tensor
    margin: Tensor
    valid_fraction: Tensor
    correction_fraction: Tensor


def causal_temporal_loss_terms(
    *,
    spatial_edge_logits: Tensor,
    spatial_same_component: Tensor,
    full_reasoning: ReasoningState,
    corrupted_reasoning: ReasoningState,
    target: Tensor,
    valid: Tensor,
    noop_weight: float,
    corrupted_gate_weight: float,
    margin_weight: float,
    margin: float,
) -> TemporalCausalTerms:
    """Compute the causal objective on aligned production RAG edges.

    ``target`` follows the RAG target convention:
        1 -> same GT instance / KEEP
        0 -> different GT instances / CUT

    The causal margin is applied only where the ACTUAL frozen spatial
    partition disagrees with GT. This is important for multicut: partition
    membership is a global graph result and cannot be reconstructed from one
    edge threshold in isolation.
    """
    full_logits = full_reasoning.final_edge_logits
    corrupted_logits = corrupted_reasoning.final_edge_logits

    if not (
        spatial_edge_logits.shape
        == spatial_same_component.shape
        == full_logits.shape
        == corrupted_logits.shape
        == target.shape
        == valid.shape
    ):
        raise ValueError(
            "Causal temporal edge tensors must have identical 1-D shapes"
        )

    zero = full_logits.sum() * 0.0
    valid = valid.bool()
    target = target.to(
        device=full_logits.device,
        dtype=full_logits.dtype,
    )
    spatial = spatial_edge_logits.detach().to(full_logits)
    spatial_same_component = spatial_same_component.to(
        device=full_logits.device,
        dtype=torch.bool,
    )

    if bool(valid.any()):
        noop = F.smooth_l1_loss(
            corrupted_logits[valid],
            spatial[valid],
            beta=0.5,
        )
        corrupted_gate = (
            corrupted_reasoning.edge_temporal_gate[valid]
            .square()
            .mean()
        )
        valid_fraction = valid.float().mean()
    else:
        noop = zero
        corrupted_gate = zero
        valid_fraction = zero.detach()

    target_keep = target >= 0.5
    correction = valid & (spatial_same_component != target_keep)

    if bool(correction.any()):
        direction = target.mul(2.0).sub(1.0)
        improvement = direction * (
            full_logits - corrupted_logits.detach()
        )
        margin_loss = F.relu(
            float(margin) - improvement[correction]
        ).mean()
        correction_fraction = (
            correction.float().sum()
            / valid.float().sum().clamp_min(1.0)
        )
    else:
        margin_loss = zero
        correction_fraction = zero.detach()

    total = (
        float(noop_weight) * noop
        + float(corrupted_gate_weight) * corrupted_gate
        + float(margin_weight) * margin_loss
    )
    return TemporalCausalTerms(
        total=total,
        noop=noop,
        corrupted_gate=corrupted_gate,
        margin=margin_loss,
        valid_fraction=valid_fraction.detach(),
        correction_fraction=correction_fraction.detach(),
    )


def temporal_causal_objective(
    model: "StirNet",
    output: StirNetOutput,
    gt_labels: Tensor,
    dref_um: Tensor,
    *,
    rag_criterion: "RAGCriterion",
    loss_config: "LossConfig",
    corruption: str,
    seed: int,
) -> dict[str, Tensor]:
    """Re-run only temporal reasoning with corrupted content.

    Spatial CNN, watershed, RAG construction and the spatial RAG network are
    NOT repeated. The corrupted pass reuses the FULL forward's
    ``output.rag``, ``output.provisional_instances`` and observed temporal
    geometry.
    """
    reference = output.reasoning.final_edge_logits
    zero = reference.sum() * 0.0

    corruption_code = {
        "contentless": 0.0,
        "shuffled": 1.0,
    }.get(str(corruption).lower())
    if corruption_code is None:
        raise ValueError(f"Unsupported temporal corruption: {corruption!r}")

    if output.temporal.is_empty or output.rag.edge_index.shape[1] == 0:
        return {
            "temporal_causal_loss": zero,
            "temporal_causal_noop": zero.detach(),
            "temporal_causal_corrupted_gate": zero.detach(),
            "temporal_causal_margin": zero.detach(),
            "temporal_causal_valid_fraction": zero.detach(),
            "temporal_causal_correction_fraction": zero.detach(),
            "temporal_causal_corruption_code": reference.new_tensor(
                corruption_code
            ),
        }

    corrupted_temporal = corrupt_temporal_state(
        output.temporal,
        corruption=corruption,
        seed=seed,
    )
    corrupted_reasoning = model.instance_temporal(
        output.provisional_instances,
        output.rag,
        corrupted_temporal,
        dref_um,
    )

    targets = rag_criterion.build_targets(
        output.rag,
        gt_labels,
    )

    src, dst = output.rag.edge_index
    components = output.spatial_partition.node_component_global
    spatial_same_component = components[src] == components[dst]

    terms = causal_temporal_loss_terms(
        spatial_edge_logits=output.rag.spatial_edge_logits,
        spatial_same_component=spatial_same_component,
        full_reasoning=output.reasoning,
        corrupted_reasoning=corrupted_reasoning,
        target=targets.target,
        valid=targets.valid,
        noop_weight=loss_config.temporal_causal_noop_weight,
        corrupted_gate_weight=(
            loss_config.temporal_causal_corrupted_gate_weight
        ),
        margin_weight=loss_config.temporal_causal_margin_weight,
        margin=loss_config.temporal_causal_margin,
    )

    return {
        "temporal_causal_loss": terms.total,
        "temporal_causal_noop": terms.noop.detach(),
        "temporal_causal_corrupted_gate": terms.corrupted_gate.detach(),
        "temporal_causal_margin": terms.margin.detach(),
        "temporal_causal_valid_fraction": terms.valid_fraction,
        "temporal_causal_correction_fraction": terms.correction_fraction,
        "temporal_causal_corruption_code": reference.new_tensor(
            corruption_code
        ),
    }


__all__ = [
    "CAUSAL_TEMPORAL_CORRUPTIONS",
    "TemporalCausalTerms",
    "causal_temporal_loss_terms",
    "contentless_temporal_state",
    "corrupt_temporal_state",
    "shuffled_temporal_state",
    "temporal_causal_objective",
]
