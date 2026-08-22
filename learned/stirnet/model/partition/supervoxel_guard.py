from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy import ndimage as ndi
from torch import Tensor, nn

from ..config import PartitionConfig
from ..types import GeometryDerivedCache, GeometryLike, geometry_field


FaceArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SupervoxelGuardDiagnostics:
    preliminary_count: int
    final_count: int
    split_supervoxel_count: int
    added_supervoxel_count: int
    cut_face_count: int
    suppressed_pathological_split_count: int
    # Compatibility only: one-voxel lower-endpoint proxy, not a real barrier.
    barrier_voxel_count: int


def _axis_scalar_faces(field: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    lower = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower[axis] = slice(0, -1)
    upper[axis] = slice(1, None)
    return field[tuple(lower)], field[tuple(upper)]


def _axis_vector_faces(field: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    lower = [slice(None)] * 3
    upper = [slice(None)] * 3
    lower[axis] = slice(0, -1)
    upper[axis] = slice(1, None)
    return field[(slice(None), *lower)], field[(slice(None), *upper)]


def _absolute_centroid_votes(
    centroid_offset: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
) -> np.ndarray:
    if centroid_offset.ndim != 4 or centroid_offset.shape[0] != 3:
        raise ValueError("centroid_offset must have shape [3,Z,Y,X]")
    shape = centroid_offset.shape[1:]
    votes = np.empty_like(centroid_offset, dtype=np.float32)
    for axis in range(3):
        coordinate = np.arange(shape[axis], dtype=np.float32) * float(spacing_um[axis])
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        votes[axis] = (
            coordinate.reshape(reshape)
            + centroid_offset[axis].astype(np.float32, copy=False) * float(dref_um)
        )
    return votes


def _separator_face_score(
    separator: np.ndarray,
    spacing_um: np.ndarray,
    separator_sigma_um: float,
    axis: int,
) -> np.ndarray:
    """Undo native-center attenuation of a face-centered Gaussian target."""
    lower, upper = _axis_scalar_faces(separator, axis)
    raw = 0.5 * (lower + upper)
    sigma = max(float(separator_sigma_um), 1e-6)
    half_pitch = 0.5 * float(spacing_um[axis])
    expected = max(float(np.exp(-0.5 * (half_pitch / sigma) ** 2)), 1e-4)
    return np.clip(raw / expected, 0.0, 1.0).astype(np.float32, copy=False)


def _ridge_faces(score: np.ndarray, axis: int, tolerance: float) -> np.ndarray:
    """Keep local maxima along the face normal instead of a thick band."""
    if score.shape[axis] == 0:
        return np.zeros(score.shape, dtype=bool)
    previous = np.full(score.shape, -np.inf, dtype=np.float32)
    following = np.full(score.shape, -np.inf, dtype=np.float32)
    current = [slice(None)] * 3
    shifted = [slice(None)] * 3
    current[axis] = slice(1, None)
    shifted[axis] = slice(0, -1)
    previous[tuple(current)] = score[tuple(shifted)]
    current[axis] = slice(0, -1)
    shifted[axis] = slice(1, None)
    following[tuple(current)] = score[tuple(shifted)]
    return (
        (score > previous + float(tolerance))
        & (score >= following - float(tolerance))
    )


def _centroid_face_disagreement(
    centroid_votes: np.ndarray,
    axis: int,
    dref_um: float,
) -> np.ndarray:
    lower, upper = _axis_vector_faces(centroid_votes, axis)
    return (
        np.sqrt(np.sum((lower - upper) ** 2, axis=0))
        / max(float(dref_um), 1e-6)
    ).astype(np.float32, copy=False)


def _flow_face_disagreement(
    flow: np.ndarray,
    axis: int,
    minimum_vector_norm: float,
) -> np.ndarray:
    lower, upper = _axis_vector_faces(flow, axis)
    lower_norm = np.sqrt(np.sum(lower * lower, axis=0))
    upper_norm = np.sqrt(np.sum(upper * upper, axis=0))
    valid = (
        (lower_norm >= float(minimum_vector_norm))
        & (upper_norm >= float(minimum_vector_norm))
    )
    dot = np.sum(lower * upper, axis=0)
    denom = np.maximum(lower_norm * upper_norm, 1e-6)
    result = np.zeros(dot.shape, dtype=np.float32)
    result[valid] = np.clip(1.0 - dot[valid] / denom[valid], 0.0, 2.0)
    return result


def _valley_face_support(
    seed: np.ndarray,
    sdf_normalized: np.ndarray,
    axis: int,
    cfg: PartitionConfig,
) -> np.ndarray:
    seed_lower, seed_upper = _axis_scalar_faces(seed, axis)
    sdf_lower, sdf_upper = _axis_scalar_faces(sdf_normalized, axis)
    return (
        np.maximum(seed_lower, seed_upper) <= cfg.supervoxel_guard_seed_valley_max
    ) | (
        np.maximum(sdf_lower, sdf_upper) <= cfg.supervoxel_guard_sdf_valley_max
    )


def build_supervoxel_face_cuts(
    separator: np.ndarray,
    centroid_offset: np.ndarray,
    flow: np.ndarray,
    seed: np.ndarray,
    sdf_normalized: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    cfg: PartitionConfig,
    *,
    separator_sigma_um: float = 0.50,
) -> tuple[FaceArrays, dict[str, FaceArrays]]:
    """Construct per-face cannot-link evidence from dense geometry."""
    separator = np.asarray(separator, dtype=np.float32)
    centroid_offset = np.asarray(centroid_offset, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    seed = np.asarray(seed, dtype=np.float32)
    sdf_normalized = np.asarray(sdf_normalized, dtype=np.float32)
    spacing_um = np.asarray(spacing_um, dtype=np.float32)

    if separator.ndim != 3:
        raise ValueError("separator must be [Z,Y,X]")
    if centroid_offset.shape != (3, *separator.shape):
        raise ValueError("centroid_offset must be [3,Z,Y,X]")
    if flow.shape != (3, *separator.shape):
        raise ValueError("flow must be [3,Z,Y,X]")
    if seed.shape != separator.shape or sdf_normalized.shape != separator.shape:
        raise ValueError("seed and sdf_normalized must match separator shape")

    centroid_votes = _absolute_centroid_votes(centroid_offset, spacing_um, dref_um)
    buckets: dict[str, list[np.ndarray]] = {
        "separator_face_score": [],
        "separator_ridge": [],
        "centroid_vote_disagreement": [],
        "flow_disagreement": [],
        "valley_support": [],
        "strong_separator": [],
        "separator_corroborated": [],
        "geometry_only": [],
    }
    cuts: list[np.ndarray] = []

    for axis in range(3):
        sep_score = _separator_face_score(
            separator, spacing_um, separator_sigma_um, axis
        )
        ridge = _ridge_faces(
            sep_score, axis, cfg.supervoxel_guard_face_ridge_tolerance
        )
        centroid_face = _centroid_face_disagreement(centroid_votes, axis, dref_um)
        flow_face = _flow_face_disagreement(
            flow, axis, cfg.supervoxel_guard_flow_min_norm
        )
        valley_face = _valley_face_support(seed, sdf_normalized, axis, cfg)

        centroid_support = (
            centroid_face >= cfg.supervoxel_guard_centroid_disagreement_dref
        )
        strong_centroid_support = (
            centroid_face >= cfg.supervoxel_guard_centroid_strong_disagreement_dref
        )
        flow_support = flow_face >= cfg.supervoxel_guard_flow_disagreement
        strong_sep = ridge & (
            sep_score >= cfg.supervoxel_guard_face_separator_high
        )
        corroborated = (
            ridge
            & (sep_score >= cfg.supervoxel_guard_face_separator_low)
            & centroid_support
            & (flow_support | valley_face)
        )
        geometry = (
            bool(cfg.supervoxel_guard_geometry_only_enabled)
            & strong_centroid_support
            & flow_support
            & valley_face
        )
        cut = strong_sep | corroborated | geometry

        cuts.append(cut.astype(bool, copy=False))
        for key, value in (
            ("separator_face_score", sep_score),
            ("separator_ridge", ridge),
            ("centroid_vote_disagreement", centroid_face),
            ("flow_disagreement", flow_face),
            ("valley_support", valley_face),
            ("strong_separator", strong_sep),
            ("separator_corroborated", corroborated),
            ("geometry_only", geometry),
        ):
            buckets[key].append(np.asarray(value))

    return (
        tuple(cuts),  # type: ignore[return-value]
        {key: tuple(values) for key, values in buckets.items()},  # type: ignore[return-value]
    )


def face_cuts_to_voxel_proxy(
    cut_faces: FaceArrays,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """One-voxel lower-endpoint proxy for visualization only."""
    proxy = np.zeros(shape, dtype=np.uint8)
    for axis, faces in enumerate(cut_faces):
        target = [slice(None)] * 3
        target[axis] = slice(0, -1)
        view = proxy[tuple(target)]
        np.maximum(view, faces.astype(np.uint8), out=view)
    return proxy


def _face_values_to_voxel_max(face_values: FaceArrays, shape: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for axis, values in enumerate(face_values):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        np.maximum(out[tuple(lower)], values, out=out[tuple(lower)])
        np.maximum(out[tuple(upper)], values, out=out[tuple(upper)])
    return out


def build_supervoxel_barrier(
    separator: np.ndarray,
    centroid_offset: np.ndarray,
    flow: np.ndarray,
    seed: np.ndarray,
    sdf_normalized: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    cfg: PartitionConfig,
    *,
    separator_sigma_um: float = 0.50,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compatibility wrapper; production splitting is face-based."""
    cuts, evidence = build_supervoxel_face_cuts(
        separator, centroid_offset, flow, seed, sdf_normalized,
        spacing_um, dref_um, cfg, separator_sigma_um=separator_sigma_um,
    )
    shape = tuple(int(v) for v in separator.shape)
    proxy = face_cuts_to_voxel_proxy(cuts, shape)
    compat = {
        "centroid_vote_disagreement": _face_values_to_voxel_max(
            evidence["centroid_vote_disagreement"], shape
        ),
        "flow_disagreement": _face_values_to_voxel_max(
            evidence["flow_disagreement"], shape
        ),
        "strong_separator": face_cuts_to_voxel_proxy(
            evidence["strong_separator"], shape
        ).astype(bool),
        "separator_corroborated": face_cuts_to_voxel_proxy(
            evidence["separator_corroborated"], shape
        ).astype(bool),
        "geometry_only": face_cuts_to_voxel_proxy(
            evidence["geometry_only"], shape
        ).astype(bool),
        "valley_support": face_cuts_to_voxel_proxy(
            evidence["valley_support"], shape
        ).astype(bool),
    }
    return proxy.astype(bool), compat


def _face_slice_for_box(box: tuple[slice, slice, slice], axis: int) -> tuple[slice, slice, slice]:
    result = list(box)
    start = int(box[axis].start)
    stop = int(box[axis].stop)
    result[axis] = slice(start, max(start, stop - 1))
    return tuple(result)  # type: ignore[return-value]


def _local_components_with_face_cuts(
    local_mask: np.ndarray,
    local_cuts: FaceArrays,
) -> tuple[np.ndarray, int]:
    points = np.flatnonzero(local_mask.ravel())
    node_count = int(points.size)
    components = np.zeros(local_mask.shape, dtype=np.int32)
    if node_count == 0:
        return components, 0
    if node_count == 1:
        components.ravel()[points[0]] = 1
        return components, 1

    node_index = np.full(local_mask.shape, -1, dtype=np.int32)
    node_index.ravel()[points] = np.arange(node_count, dtype=np.int32)
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []

    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        allowed = (
            local_mask[tuple(lower)]
            & local_mask[tuple(upper)]
            & ~np.asarray(local_cuts[axis], dtype=bool)
        )
        if not allowed.any():
            continue
        a = node_index[tuple(lower)][allowed]
        b = node_index[tuple(upper)][allowed]
        row_chunks.extend([a, b])
        col_chunks.extend([b, a])

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        graph = sparse.coo_matrix(
            (np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
            shape=(node_count, node_count),
        ).tocsr()
    else:
        graph = sparse.csr_matrix((node_count, node_count), dtype=np.uint8)

    count, node_components = connected_components(
        graph, directed=False, return_labels=True
    )
    components.ravel()[points] = node_components.astype(np.int32) + 1
    return components, int(count)


def _internal_cut_face_count(labels: np.ndarray, cut_faces: FaceArrays) -> int:
    total = 0
    for axis, cuts in enumerate(cut_faces):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        a = labels[tuple(lower)]
        b = labels[tuple(upper)]
        total += int(np.sum(cuts & (a > 0) & (a == b)))
    return total


def split_preliminary_supervoxels(
    labels: np.ndarray,
    separator: np.ndarray,
    centroid_offset: np.ndarray,
    flow: np.ndarray,
    seed: np.ndarray,
    sdf_normalized: np.ndarray,
    spacing_um: np.ndarray,
    dref_um: float,
    cfg: PartitionConfig,
    *,
    separator_sigma_um: float = 0.50,
) -> tuple[np.ndarray, SupervoxelGuardDiagnostics]:
    """Split preliminary SVs by deleting only selected 6-neighbor graph edges."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 3:
        raise ValueError("labels must be [Z,Y,X]")
    preliminary_count = int(labels.max())
    if not cfg.supervoxel_guard_enabled or preliminary_count <= 0:
        return labels.copy(), SupervoxelGuardDiagnostics(
            preliminary_count, preliminary_count, 0, 0, 0, 0, 0
        )

    cut_faces, _ = build_supervoxel_face_cuts(
        separator, centroid_offset, flow, seed, sdf_normalized,
        spacing_um, dref_um, cfg, separator_sigma_um=separator_sigma_um,
    )
    internal_cut_count = _internal_cut_face_count(labels, cut_faces)
    proxy_count = int(np.sum(
        face_cuts_to_voxel_proxy(cut_faces, labels.shape) & (labels > 0)
    ))

    output = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    split_count = 0
    suppressed_count = 0

    for old_id, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        local_labels = labels[box]
        local_mask = local_labels == old_id
        if not local_mask.any():
            continue

        local_cuts: list[np.ndarray] = []
        has_internal_cut = False
        for axis in range(3):
            cuts = np.asarray(cut_faces[axis][_face_slice_for_box(box, axis)], dtype=bool)
            local_cuts.append(cuts)
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            if cuts.size and np.any(
                cuts & local_mask[tuple(lower)] & local_mask[tuple(upper)]
            ):
                has_internal_cut = True

        if not has_internal_cut:
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue

        components, component_count = _local_components_with_face_cuts(
            local_mask, tuple(local_cuts)  # type: ignore[arg-type]
        )
        if component_count < 2:
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue

        sizes = np.bincount(
            components[local_mask].ravel(), minlength=component_count + 1
        )[1:]
        minimum = max(
            int(cfg.supervoxel_guard_min_fragment_voxels),
            int(np.ceil(
                float(local_mask.sum())
                * float(cfg.supervoxel_guard_min_fragment_fraction)
            )),
        )
        meaningful_count = int(np.sum(sizes >= minimum))
        if meaningful_count < 2:
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue
        if component_count > cfg.supervoxel_guard_max_fragments:
            suppressed_count += 1
            target = output[box]
            target[local_mask] = next_id
            output[box] = target
            next_id += 1
            continue

        split_count += 1
        target = output[box]
        for component_id in range(1, component_count + 1):
            target[components == component_id] = next_id
            next_id += 1
        output[box] = target

    final_count = next_id - 1
    return output, SupervoxelGuardDiagnostics(
        preliminary_count=preliminary_count,
        final_count=final_count,
        split_supervoxel_count=split_count,
        added_supervoxel_count=max(final_count - preliminary_count, 0),
        cut_face_count=internal_cut_count,
        suppressed_pathological_split_count=suppressed_count,
        barrier_voxel_count=proxy_count,
    )


class SupervoxelSafetyGuard(nn.Module):
    def __init__(
        self,
        cfg: PartitionConfig,
        *,
        separator_sigma_um: float = 0.50,
    ):
        super().__init__()
        self.cfg = cfg
        self.separator_sigma_um = float(separator_sigma_um)

    @torch.no_grad()
    def forward(
        self,
        labels: np.ndarray,
        geometry: GeometryLike,
        derived_cache: GeometryDerivedCache,
        batch_index: int,
        spacing_um: Tensor,
        dref_um: Tensor,
    ) -> np.ndarray:
        if not self.cfg.supervoxel_guard_enabled:
            return np.asarray(labels, dtype=np.int32)
        separator = derived_cache.separator_prob[batch_index, 0].detach().float().cpu().numpy()
        seed = derived_cache.seed_prob[batch_index, 0].detach().float().cpu().numpy()
        sdf_normalized = derived_cache.sdf_normalized[batch_index, 0].detach().float().cpu().numpy()
        centroid_offset = geometry_field(geometry, "centroid_offset")[batch_index].detach().float().cpu().numpy()
        flow = geometry_field(geometry, "flow")[batch_index].detach().float().cpu().numpy()
        spacing = spacing_um.detach().float().cpu().numpy().astype(np.float32)
        dref = float(dref_um.detach().float().cpu().item())
        guarded, _ = split_preliminary_supervoxels(
            np.asarray(labels, dtype=np.int32), separator, centroid_offset, flow,
            seed, sdf_normalized, spacing, dref, self.cfg,
            separator_sigma_um=self.separator_sigma_um,
        )
        return guarded


__all__ = [
    "FaceArrays",
    "SupervoxelGuardDiagnostics",
    "SupervoxelSafetyGuard",
    "build_supervoxel_face_cuts",
    "face_cuts_to_voxel_proxy",
    "build_supervoxel_barrier",
    "split_preliminary_supervoxels",
]
