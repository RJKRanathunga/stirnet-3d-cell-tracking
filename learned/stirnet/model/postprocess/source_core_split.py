# STIRNET_SOURCE_CORE_SPLIT_ONLY_FILTER_V2
from __future__ import annotations

"""Asymmetric inference-only post-graph split filter.

Two implementations are intentionally kept:

``supervoxel_graph`` (DEFAULT)
    Uses disconnected source-foreground components only as split anchors, then
    partitions the already-existing watershed supervoxels with a marker-
    controlled graph watershed.  Learned separator evidence defines the
    supervoxel-interface barrier and boosts the final split confidence.

``voxel_watershed`` (LEGACY)
    Preserves the original v1 behavior: source components become voxel markers
    for a separator-guided watershed inside the final component.

Both implementations are strictly SPLIT ONLY.  Source evidence can never merge
two graph-separated components and never feeds back into RAG/temporal logits.
"""

from dataclasses import dataclass
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
    """Find 6-connected bright source masks and assign safe containments.

    The connected components here are exactly the "initial binary masks" used
    as one-way split evidence.  A source component that substantially spans
    multiple already-separated final components is discarded; it is never
    interpreted as must-link evidence.
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
        contained_voxels = int(counts[best])
        containment = contained_voxels / max(voxels, 1)
        if containment < min_containment:
            continue

        local_coords = np.argwhere(local).astype(np.float64)
        starts = np.asarray([axis.start for axis in box], dtype=np.float64)
        centroid_voxel = local_coords.mean(axis=0) + starts

        by_final.setdefault(final_id, []).append(
            {
                "core_id": int(core_id),
                "voxels": voxels,
                "contained_voxels": contained_voxels,
                "containment": float(containment),
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


def _separator_statistics(
    territories: np.ndarray,
    separator: np.ndarray,
    support_threshold: float,
) -> tuple[float, float, float, np.ndarray]:
    boundary = _internal_boundary(territories)
    values = separator[boundary]
    if values.size == 0:
        return 0.0, 0.0, 0.0, boundary
    return (
        float(values.mean()),
        float(values.max()),
        float(np.mean(values >= support_threshold)),
        boundary,
    )


@dataclass
class _GraphSplitProposal:
    territories: np.ndarray
    source_core_count: int
    supervoxel_count: int
    anchored_supervoxel_count: int
    cut_supervoxel_edge_count: int
    mean_cut_barrier: float
    max_cut_barrier: float


class _SeededUnionFind:
    """Kruskal-style marker-controlled minimum spanning forest."""

    def __init__(self, markers: np.ndarray):
        n = int(len(markers))
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)
        self.marker = np.asarray(markers, dtype=np.int32).copy()

    def find(self, x: int) -> int:
        x = int(x)
        root = x
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[x]) != x:
            nxt = int(self.parent[x])
            self.parent[x] = root
            x = nxt
        return root

    def try_union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True

        ma = int(self.marker[ra])
        mb = int(self.marker[rb])
        # Two different source markers are a hard split request.  The graph
        # watershed can grow each territory but can never merge those markers.
        if ma > 0 and mb > 0 and ma != mb:
            return False

        if int(self.rank[ra]) < int(self.rank[rb]):
            ra, rb = rb, ra
            ma, mb = mb, ma
        self.parent[rb] = ra
        if int(self.rank[ra]) == int(self.rank[rb]):
            self.rank[ra] += 1
        self.marker[ra] = ma if ma > 0 else mb
        return True


def _aggregate_sv_interfaces(
    *,
    supervoxels: np.ndarray,
    component_mask: np.ndarray,
    separator: np.ndarray,
    support_threshold: float,
) -> list[tuple[int, int, float, float, float]]:
    """Return (sv_a, sv_b, mean, max, coverage) for 6-neighbor interfaces."""
    pair_rows: dict[tuple[int, int], list[np.ndarray]] = {}

    for axis in range(3):
        lo_slice = [slice(None)] * 3
        hi_slice = [slice(None)] * 3
        lo_slice[axis] = slice(0, -1)
        hi_slice[axis] = slice(1, None)
        lo_slice = tuple(lo_slice)
        hi_slice = tuple(hi_slice)

        a_sv = supervoxels[lo_slice]
        b_sv = supervoxels[hi_slice]
        a_mask = component_mask[lo_slice]
        b_mask = component_mask[hi_slice]
        valid = (
            a_mask
            & b_mask
            & (a_sv > 0)
            & (b_sv > 0)
            & (a_sv != b_sv)
        )
        if not bool(valid.any()):
            continue

        a = a_sv[valid].astype(np.int64, copy=False)
        b = b_sv[valid].astype(np.int64, copy=False)
        sep_face = 0.5 * (
            separator[lo_slice][valid].astype(np.float64, copy=False)
            + separator[hi_slice][valid].astype(np.float64, copy=False)
        )

        pair_lo = np.minimum(a, b)
        pair_hi = np.maximum(a, b)
        keys = np.stack([pair_lo, pair_hi], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)

        for row_index, pair in enumerate(unique.tolist()):
            values = sep_face[inverse == row_index]
            key = (int(pair[0]), int(pair[1]))
            pair_rows.setdefault(key, []).append(values)

    output: list[tuple[int, int, float, float, float]] = []
    for (a, b), chunks in pair_rows.items():
        values = np.concatenate(chunks, axis=0)
        output.append(
            (
                int(a),
                int(b),
                float(values.mean()),
                float(values.max()),
                float(np.mean(values >= support_threshold)),
            )
        )
    return output


def _supervoxel_graph_proposal(
    *,
    final_component_mask: np.ndarray,
    source_core_labels: np.ndarray,
    cores: list[dict[str, Any]],
    supervoxels: np.ndarray,
    separator: np.ndarray,
    cfg: InferenceConfig,
) -> tuple[_GraphSplitProposal | None, str, dict[str, Any]]:
    """Split one final component as a graph of immutable atomic supervoxels."""
    sv_ids = np.unique(supervoxels[final_component_mask])
    sv_ids = sv_ids[sv_ids > 0].astype(np.int64, copy=False)
    if sv_ids.size < len(cores):
        return None, "too_few_supervoxels", {
            "supervoxel_count": int(sv_ids.size),
        }

    sv_to_node = {int(sv): i for i, sv in enumerate(sv_ids.tolist())}
    core_ids = [int(core["core_id"]) for core in cores]
    core_to_marker = {core_id: i + 1 for i, core_id in enumerate(core_ids)}

    markers = np.zeros(len(sv_ids), dtype=np.int32)
    anchored_count = 0
    ambiguous_supervoxels: list[int] = []

    for sv_id in sv_ids.tolist():
        node = sv_to_node[int(sv_id)]
        mask = final_component_mask & (supervoxels == int(sv_id))
        values = source_core_labels[mask]
        values = values[values > 0]
        if values.size == 0:
            continue

        ids, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts)
        ids = ids[order]
        counts = counts[order]
        total = int(counts.sum())
        best_core = int(ids[0])
        best_count = int(counts[0])
        purity = best_count / max(total, 1)

        if best_core not in core_to_marker:
            # This source component was not safely assigned to this final
            # component, so it must not become an anchor here.
            continue

        if (
            best_count
            < int(cfg.source_core_split_supervoxel_min_anchor_voxels)
        ):
            continue

        if (
            len(ids) > 1
            and purity
            < float(cfg.source_core_split_supervoxel_anchor_purity)
        ):
            ambiguous_supervoxels.append(int(sv_id))
            continue

        markers[node] = int(core_to_marker[best_core])
        anchored_count += 1

    if ambiguous_supervoxels:
        # A single atomic SV materially overlaps multiple source masks.  The
        # default method refuses to invent a boundary through that SV.  Users
        # can select legacy voxel_watershed explicitly if desired.
        return None, "ambiguous_source_cores_share_supervoxel", {
            "ambiguous_supervoxel_ids": ambiguous_supervoxels,
            "supervoxel_count": int(len(sv_ids)),
            "anchored_supervoxel_count": int(anchored_count),
        }

    present_markers = set(int(value) for value in markers.tolist() if value > 0)
    expected_markers = set(range(1, len(cores) + 1))
    if present_markers != expected_markers:
        return None, "missing_source_core_supervoxel_anchor", {
            "present_marker_count": int(len(present_markers)),
            "expected_marker_count": int(len(expected_markers)),
            "supervoxel_count": int(len(sv_ids)),
            "anchored_supervoxel_count": int(anchored_count),
        }

    interfaces = _aggregate_sv_interfaces(
        supervoxels=supervoxels,
        component_mask=final_component_mask,
        separator=separator,
        support_threshold=float(
            cfg.source_core_split_separator_support_threshold
        ),
    )
    if not interfaces:
        return None, "no_supervoxel_interfaces", {
            "supervoxel_count": int(len(sv_ids)),
            "anchored_supervoxel_count": int(anchored_count),
        }

    # Barrier is intentionally transparent and bounded.  Low-separator
    # interfaces are joined first.  High-separator interfaces are delayed and
    # become the natural marker-controlled cut when two source territories meet.
    edge_rows: list[tuple[float, int, int, float, float, float]] = []
    for sv_a, sv_b, sep_mean, sep_max, sep_cov in interfaces:
        if sv_a not in sv_to_node or sv_b not in sv_to_node:
            continue
        barrier = (
            0.50 * sep_mean
            + 0.25 * sep_max
            + 0.25 * sep_cov
        )
        edge_rows.append(
            (
                float(barrier),
                int(sv_to_node[sv_a]),
                int(sv_to_node[sv_b]),
                float(sep_mean),
                float(sep_max),
                float(sep_cov),
            )
        )

    edge_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    uf = _SeededUnionFind(markers)
    rejected_edges: list[tuple[float, int, int]] = []

    for barrier, node_a, node_b, _mean, _max, _cov in edge_rows:
        if not uf.try_union(node_a, node_b):
            rejected_edges.append((float(barrier), node_a, node_b))

    node_territory = np.zeros(len(sv_ids), dtype=np.int32)
    for node in range(len(sv_ids)):
        root = uf.find(node)
        marker = int(uf.marker[root])
        if marker <= 0:
            # With a connected final component and at least one marker this
            # should be impossible because unseeded components can always join.
            return None, "unassigned_supervoxel_territory", {
                "supervoxel_count": int(len(sv_ids)),
                "anchored_supervoxel_count": int(anchored_count),
            }
        node_territory[node] = marker

    if len(np.unique(node_territory)) < 2:
        return None, "graph_proposal_collapsed", {
            "supervoxel_count": int(len(sv_ids)),
            "anchored_supervoxel_count": int(anchored_count),
        }

    max_sv = int(max(int(supervoxels.max()), int(sv_ids.max())))
    sv_to_territory = np.zeros(max_sv + 1, dtype=np.int32)
    for sv_id, node in sv_to_node.items():
        sv_to_territory[int(sv_id)] = int(node_territory[node])

    territories = np.zeros(supervoxels.shape, dtype=np.int32)
    territories[final_component_mask] = sv_to_territory[
        supervoxels[final_component_mask].astype(np.int64, copy=False)
    ]

    cut_barriers = np.asarray(
        [row[0] for row in rejected_edges],
        dtype=np.float64,
    )
    return (
        _GraphSplitProposal(
            territories=territories,
            source_core_count=int(len(cores)),
            supervoxel_count=int(len(sv_ids)),
            anchored_supervoxel_count=int(anchored_count),
            cut_supervoxel_edge_count=int(len(rejected_edges)),
            mean_cut_barrier=(
                float(cut_barriers.mean()) if cut_barriers.size else 0.0
            ),
            max_cut_barrier=(
                float(cut_barriers.max()) if cut_barriers.size else 0.0
            ),
        ),
        "ok",
        {},
    )


def _voxel_watershed_proposal(
    *,
    final_component_mask: np.ndarray,
    source_core_labels: np.ndarray,
    cores: list[dict[str, Any]],
    separator: np.ndarray,
) -> np.ndarray | None:
    """Legacy v1 proposal retained as an explicit configuration."""
    box = _bbox(final_component_mask)
    local_mask = final_component_mask[box]
    local_separator = separator[box]
    local_cores = source_core_labels[box]
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
        return None

    local_territories = watershed(
        local_separator,
        markers=markers,
        mask=local_mask,
        connectivity=ndi.generate_binary_structure(3, 1),
        watershed_line=False,
    ).astype(np.int32, copy=False)

    territories = np.zeros(final_component_mask.shape, dtype=np.int32)
    territories[box] = local_territories
    return territories


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
        *,
        supervoxel_labels: list[Tensor] | None = None,
    ) -> SplitOnlyPostprocessState:
        if source_foreground_prior.ndim != 4:
            raise ValueError("source foreground must be [B,Z,Y,X]")
        if separator_probability.shape != source_foreground_prior.shape:
            raise ValueError("separator/source shapes must match")
        if len(final_labels) != source_foreground_prior.shape[0]:
            raise ValueError("label batch does not match source batch")

        method = str(self.cfg.source_core_split_method)
        if method not in {"supervoxel_graph", "voxel_watershed"}:
            raise ValueError(
                "source_core_split_method must be 'supervoxel_graph' "
                "or 'voxel_watershed'"
            )
        if method == "supervoxel_graph":
            if supervoxel_labels is None:
                raise ValueError(
                    "supervoxel_graph split method requires supervoxel_labels"
                )
            if len(supervoxel_labels) != len(final_labels):
                raise ValueError(
                    "supervoxel_labels batch does not match final labels"
                )

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
            supervoxels = (
                None
                if supervoxel_labels is None
                else supervoxel_labels[batch_index]
                .detach()
                .long()
                .cpu()
                .numpy()
            )
            if supervoxels is not None and supervoxels.shape != old.shape:
                raise ValueError(
                    "supervoxel/final label shapes must match"
                )

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
                            "method": method,
                            "status": "skipped_too_many_cores",
                            "applied": False,
                        }
                    )
                    continue

                candidate_count += 1
                component = old == int(final_id)

                proposal_metadata: dict[str, Any] = {}
                if method == "supervoxel_graph":
                    assert supervoxels is not None
                    graph_proposal, proposal_status, proposal_metadata = (
                        _supervoxel_graph_proposal(
                            final_component_mask=component,
                            source_core_labels=core_labels,
                            cores=cores,
                            supervoxels=supervoxels,
                            separator=separator,
                            cfg=self.cfg,
                        )
                    )
                    if graph_proposal is None:
                        records.append(
                            {
                                "batch_index": batch_index,
                                "final_component_id": int(final_id),
                                "source_core_ids": [
                                    int(core["core_id"]) for core in cores
                                ],
                                "source_core_count": len(cores),
                                "method": method,
                                "status": proposal_status,
                                "applied": False,
                                **proposal_metadata,
                            }
                        )
                        continue
                    territories = graph_proposal.territories
                    proposal_metadata = {
                        "supervoxel_count": graph_proposal.supervoxel_count,
                        "anchored_supervoxel_count": (
                            graph_proposal.anchored_supervoxel_count
                        ),
                        "cut_supervoxel_edge_count": (
                            graph_proposal.cut_supervoxel_edge_count
                        ),
                        "mean_cut_supervoxel_barrier": (
                            graph_proposal.mean_cut_barrier
                        ),
                        "max_cut_supervoxel_barrier": (
                            graph_proposal.max_cut_barrier
                        ),
                    }
                else:
                    territories = _voxel_watershed_proposal(
                        final_component_mask=component,
                        source_core_labels=core_labels,
                        cores=cores,
                        separator=separator,
                    )
                    if territories is None:
                        records.append(
                            {
                                "batch_index": batch_index,
                                "final_component_id": int(final_id),
                                "source_core_count": len(cores),
                                "method": method,
                                "status": "marker_lost",
                                "applied": False,
                            }
                        )
                        continue

                child_ids, child_counts = np.unique(
                    territories[component & (territories > 0)],
                    return_counts=True,
                )
                component_voxels = int(component.sum())
                if len(child_ids) < 2:
                    records.append(
                        {
                            "batch_index": batch_index,
                            "final_component_id": int(final_id),
                            "source_core_count": len(cores),
                            "method": method,
                            "status": "proposal_collapsed",
                            "applied": False,
                            **proposal_metadata,
                        }
                    )
                    continue

                child_fraction = (
                    child_counts.astype(np.float64)
                    / max(component_voxels, 1)
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
                            "method": method,
                            "min_child_fraction": min_child_fraction,
                            "status": "child_too_small",
                            "applied": False,
                            **proposal_metadata,
                        }
                    )
                    continue

                (
                    sep_mean,
                    sep_max,
                    sep_coverage,
                    boundary,
                ) = _separator_statistics(
                    territories,
                    separator,
                    float(
                        self.cfg.source_core_split_separator_support_threshold
                    ),
                )

                count_score = 1.0 - math.exp(-1.20 * (len(cores) - 1))
                min_distance_dref = _min_core_separation_dref(
                    cores,
                    spacing,
                    dref,
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
                    volume_ratio = (
                        component_voxels / max(reference_volume, 1e-6)
                    )
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
                        "source_core_ids": [
                            int(core["core_id"]) for core in cores
                        ],
                        "source_core_count": len(cores),
                        "method": method,
                        "component_voxels": component_voxels,
                        "reference_single_core_component_voxels": (
                            reference_volume
                        ),
                        "component_volume_ratio_to_single_core_median": (
                            volume_ratio
                        ),
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
                        **proposal_metadata,
                    }
                )
                if not apply:
                    continue

                local_result = result[component]
                local_territory = territories[component]
                sorted_children = sorted(
                    int(value) for value in child_ids.tolist()
                )
                child_to_output: dict[int, int] = {}
                for ordinal, child_id in enumerate(sorted_children):
                    if ordinal == 0:
                        child_to_output[child_id] = int(final_id)
                    else:
                        child_to_output[child_id] = next_id
                        next_id += 1

                for child_id, output_id in child_to_output.items():
                    local_result[local_territory == child_id] = int(output_id)
                result[component] = local_result
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
