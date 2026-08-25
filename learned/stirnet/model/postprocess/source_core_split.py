# STIRNET_SOURCE_CORE_SPLIT_ONLY_FILTER_V1
from __future__ import annotations

"""Asymmetric inference-only post-graph split filter.

Fallible source segmentation is used in one direction only:

    source evidence may request a SPLIT;
    source evidence can never request or enforce a MERGE.

The filter runs after learned graph reasoning and existence filtering.  It does
not alter RAG logits, RAG components, temporal reasoning, or refinement state.
A merged source mask is therefore harmless: at worst it provides no split cue;
it can never undo a graph separation.
"""

import math
from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch import Tensor

from ..config import InferenceConfig
from ..types import SplitOnlyPostprocessState


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _internal_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        a = labels[left]
        b = labels[right]
        changed = (a > 0) & (b > 0) & (a != b)
        boundary[left] |= changed
        boundary[right] |= changed
    return boundary


def _compact(labels: np.ndarray) -> np.ndarray:
    positive = np.unique(labels[labels > 0])
    if positive.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    mapping = np.zeros(int(positive.max()) + 1, dtype=np.int32)
    mapping[positive] = np.arange(1, len(positive) + 1, dtype=np.int32)
    return mapping[labels]


def _bbox(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty component")
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def _source_cores(
    source_foreground: np.ndarray,
    final_labels: np.ndarray,
    *,
    min_voxels: int,
    min_containment: float,
) -> tuple[np.ndarray, dict[int, list[dict[str, Any]]]]:
    """Find 6-connected bright source cores and assign only safe containments.

    A source core that substantially spans multiple already-separated final
    components is discarded.  That is the key anti-poison property: source
    undersegmentation never becomes a must-link.
    """
    cores, count = ndi.label(
        source_foreground,
        structure=ndi.generate_binary_structure(3, 1),
    )
    by_final: dict[int, list[dict[str, Any]]] = {}
    if count == 0:
        return cores.astype(np.int32, copy=False), by_final

    for core_id, box in enumerate(ndi.find_objects(cores), 1):
        if box is None:
            continue
        local = cores[box] == core_id
        voxels = int(local.sum())
        if voxels < min_voxels:
            continue
        final_values = final_labels[box][local]
        positive = final_values[final_values > 0]
        if positive.size == 0:
            continue
        ids, counts = np.unique(positive, return_counts=True)
        best = int(np.argmax(counts))
        final_id = int(ids[best])
        contained = int(counts[best]) / max(voxels, 1)
        if contained < min_containment:
            continue

        local_coords = np.argwhere(local).astype(np.float64)
        starts = np.asarray([axis.start for axis in box], dtype=np.float64)
        centroid_voxel = local_coords.mean(axis=0) + starts
        by_final.setdefault(final_id, []).append(
            {
                "core_id": int(core_id),
                "voxels": voxels,
                "containment": float(contained),
                "centroid_voxel": centroid_voxel,
            }
        )
    return cores.astype(np.int32, copy=False), by_final


def _single_core_reference_volume(
    final_labels: np.ndarray,
    cores_by_final: dict[int, list[dict[str, Any]]],
    min_reference: int,
) -> float | None:
    ids, counts = np.unique(final_labels[final_labels > 0], return_counts=True)
    volume = {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())}
    rows = [
        volume[component]
        for component, cores in cores_by_final.items()
        if len(cores) == 1 and component in volume
    ]
    if len(rows) < min_reference:
        return None
    return float(np.median(np.asarray(rows, dtype=np.float64)))


def _min_core_separation_dref(
    cores: list[dict[str, Any]],
    spacing_zyx_um: np.ndarray,
    dref_um: float,
) -> float:
    points = np.stack([row["centroid_voxel"] for row in cores], axis=0)
    points = points * spacing_zyx_um[None]
    minimum = float("inf")
    for i in range(len(points) - 1):
        distance = np.linalg.norm(points[i + 1 :] - points[i], axis=1)
        if distance.size:
            minimum = min(minimum, float(distance.min()))
    return 0.0 if not math.isfinite(minimum) else minimum / max(dref_um, 1e-6)


def _verify_split_only(old: np.ndarray, new: np.ndarray) -> None:
    positive = new > 0
    if not positive.any():
        return
    pairs = np.unique(np.stack([new[positive], old[positive]], axis=1), axis=0)
    owner: dict[int, int] = {}
    for new_id, old_id in pairs.tolist():
        new_id = int(new_id)
        old_id = int(old_id)
        if new_id in owner and owner[new_id] != old_id:
            raise RuntimeError(
                "split-only filter attempted to merge distinct input components"
            )
        owner[new_id] = old_id


class SourceCoreSplitOnlyFilter:
    """Transparent probability-like split requester; no learned parameters."""

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg

    def __call__(
        self,
        final_labels: list[Tensor],
        source_foreground_prior: Tensor,
        separator_probability: Tensor,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> SplitOnlyPostprocessState:
        if source_foreground_prior.ndim != 4:
            raise ValueError("source foreground must be [B,Z,Y,X]")
        if separator_probability.shape != source_foreground_prior.shape:
            raise ValueError("separator/source shapes must match")
        if len(final_labels) != source_foreground_prior.shape[0]:
            raise ValueError("label batch does not match source batch")

        source_cpu = source_foreground_prior.detach().float().cpu().numpy()
        separator_cpu = separator_probability.detach().float().cpu().numpy()
        spacing_cpu = spacing_um.detach().float().cpu().numpy()
        dref_cpu = dref_um.detach().float().cpu().numpy()

        output: list[Tensor] = []
        records: list[dict[str, Any]] = []
        candidate_count = 0
        applied_count = 0
        skipped_too_many = 0

        for batch_index, labels_tensor in enumerate(final_labels):
            old = labels_tensor.detach().long().cpu().numpy()
            source_binary = source_cpu[batch_index] >= float(
                self.cfg.source_core_split_foreground_threshold
            )
            separator = np.clip(separator_cpu[batch_index], 0.0, 1.0)
            core_labels, by_final = _source_cores(
                source_binary,
                old,
                min_voxels=int(self.cfg.source_core_split_min_core_voxels),
                min_containment=float(
                    self.cfg.source_core_split_min_core_containment
                ),
            )
            reference_volume = _single_core_reference_volume(
                old,
                by_final,
                int(self.cfg.source_core_split_min_reference_components),
            )

            result = old.astype(np.int32, copy=True)
            next_id = int(result.max()) + 1
            spacing = np.asarray(spacing_cpu[batch_index], dtype=np.float64)
            dref = float(dref_cpu[batch_index])

            for final_id in sorted(by_final):
                cores = by_final[final_id]
                if len(cores) < 2:
                    continue
                if len(cores) > int(
                    self.cfg.source_core_split_max_cores_per_component
                ):
                    skipped_too_many += 1
                    records.append(
                        {
                            "batch_index": batch_index,
                            "final_component_id": int(final_id),
                            "source_core_count": len(cores),
                            "status": "skipped_too_many_cores",
                            "applied": False,
                        }
                    )
                    continue

                candidate_count += 1
                component = old == int(final_id)
                box = _bbox(component)
                local_mask = component[box]
                local_separator = separator[box]
                local_cores = core_labels[box]
                markers = np.zeros(local_mask.shape, dtype=np.int32)
                for marker_id, core in enumerate(cores, 1):
                    marker = (
                        (local_cores == int(core["core_id"]))
                        & local_mask
                    )
                    if marker.any():
                        markers[marker] = marker_id

                present_markers = np.unique(markers[markers > 0])
                if len(present_markers) != len(cores):
                    records.append(
                        {
                            "batch_index": batch_index,
                            "final_component_id": int(final_id),
                            "source_core_count": len(cores),
                            "status": "marker_lost",
                            "applied": False,
                        }
                    )
                    continue

                # High separator probability is high elevation.  Flooding from
                # source cores therefore prefers to place the split along the
                # learned separator ridge.
                territories = watershed(
                    local_separator,
                    markers=markers,
                    mask=local_mask,
                    connectivity=ndi.generate_binary_structure(3, 1),
                    watershed_line=False,
                ).astype(np.int32, copy=False)

                child_ids, child_counts = np.unique(
                    territories[territories > 0], return_counts=True
                )
                component_voxels = int(local_mask.sum())
                child_fraction = (
                    child_counts.astype(np.float64) / max(component_voxels, 1)
                )
                min_child_fraction = float(child_fraction.min())
                if min_child_fraction < float(
                    self.cfg.source_core_split_min_child_fraction
                ):
                    records.append(
                        {
                            "batch_index": batch_index,
                            "final_component_id": int(final_id),
                            "source_core_count": len(cores),
                            "min_child_fraction": min_child_fraction,
                            "status": "child_too_small",
                            "applied": False,
                        }
                    )
                    continue

                boundary = _internal_boundary(territories)
                sep_values = local_separator[boundary]
                if sep_values.size:
                    sep_mean = float(sep_values.mean())
                    sep_max = float(sep_values.max())
                    sep_coverage = float(
                        np.mean(
                            sep_values
                            >= float(
                                self.cfg.source_core_split_separator_support_threshold
                            )
                        )
                    )
                else:
                    sep_mean = sep_max = sep_coverage = 0.0

                count_score = 1.0 - math.exp(-1.20 * (len(cores) - 1))
                min_distance_dref = _min_core_separation_dref(
                    cores, spacing, dref
                )
                distance_score = _sigmoid(
                    (
                        min_distance_dref
                        - float(
                            self.cfg.source_core_split_min_core_separation_dref
                        )
                    )
                    / max(
                        float(
                            self.cfg.source_core_split_separation_softness_dref
                        ),
                        1e-6,
                    )
                )
                containment_score = float(
                    np.mean([core["containment"] for core in cores])
                )
                balance_reference = max(
                    0.5 / len(cores),
                    float(self.cfg.source_core_split_min_child_fraction),
                )
                balance_score = float(
                    np.clip(
                        min_child_fraction / max(balance_reference, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                if reference_volume is None:
                    volume_ratio = None
                    volume_score = 0.5
                else:
                    volume_ratio = component_voxels / max(reference_volume, 1e-6)
                    volume_score = _sigmoid(
                        (
                            volume_ratio
                            - float(
                                self.cfg.source_core_split_volume_ratio_center
                            )
                        )
                        / max(
                            float(
                                self.cfg.source_core_split_volume_ratio_softness
                            ),
                            1e-6,
                        )
                    )

                # Source evidence is deliberately dominant. Separator is an
                # independent boost, not a prerequisite and never a merge cue.
                source_score = float(
                    0.25 * count_score
                    + 0.25 * distance_score
                    + 0.15 * containment_score
                    + 0.15 * balance_score
                    + 0.20 * volume_score
                )
                separator_score = float(
                    0.50 * sep_mean
                    + 0.25 * sep_max
                    + 0.25 * sep_coverage
                )
                confidence = float(
                    np.clip(
                        source_score
                        + (1.0 - source_score)
                        * float(self.cfg.source_core_split_separator_boost)
                        * separator_score,
                        0.0,
                        1.0,
                    )
                )
                separation_gate = min_distance_dref >= float(
                    self.cfg.source_core_split_min_core_separation_dref
                )
                apply = separation_gate and confidence >= float(
                    self.cfg.source_core_split_confidence_threshold
                )

                records.append(
                    {
                        "batch_index": batch_index,
                        "final_component_id": int(final_id),
                        "source_core_ids": [int(c["core_id"]) for c in cores],
                        "source_core_count": len(cores),
                        "component_voxels": component_voxels,
                        "reference_single_core_component_voxels": reference_volume,
                        "component_volume_ratio_to_single_core_median": volume_ratio,
                        "minimum_core_separation_dref": min_distance_dref,
                        "mean_core_containment": containment_score,
                        "child_fractions": child_fraction.tolist(),
                        "separator_boundary_voxels": int(boundary.sum()),
                        "separator_mean": sep_mean,
                        "separator_max": sep_max,
                        "separator_coverage": sep_coverage,
                        "source_score": source_score,
                        "separator_score": separator_score,
                        "split_confidence": confidence,
                        "separation_gate": bool(separation_gate),
                        "status": "applied" if apply else "below_threshold",
                        "applied": bool(apply),
                    }
                )
                if not apply:
                    continue

                local_result = result[box]
                for ordinal, child_id in enumerate(
                    sorted(int(v) for v in child_ids.tolist())
                ):
                    output_id = int(final_id) if ordinal == 0 else next_id
                    if ordinal > 0:
                        next_id += 1
                    local_result[
                        (territories == child_id) & local_mask
                    ] = output_id
                result[box] = local_result
                applied_count += 1

            result = _compact(result)
            _verify_split_only(old, result)
            output.append(
                torch.as_tensor(
                    result,
                    device=labels_tensor.device,
                    dtype=torch.long,
                )
            )

        return SplitOnlyPostprocessState(
            labels=output,
            candidate_count=candidate_count,
            applied_count=applied_count,
            skipped_too_many_cores=skipped_too_many,
            records=records,
        )
