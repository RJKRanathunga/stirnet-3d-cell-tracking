from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch import Tensor


def _profile(profiler, name: str):
    return nullcontext() if profiler is None else profiler.profile(name)


def ellipsoid_footprint(spacing_um: np.ndarray, radius_um: float) -> np.ndarray:
    radii = np.maximum(1, np.ceil(radius_um / np.maximum(spacing_um, 1e-6)).astype(int))
    axes = [np.arange(-r, r + 1, dtype=np.float32) * spacing_um[i] for i, r in enumerate(radii)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return (zz**2 + yy**2 + xx**2) <= (radius_um + 1e-6) ** 2


def build_markers(
    score: np.ndarray,
    foreground: np.ndarray,
    spacing_um: np.ndarray,
    radius_um: float,
    threshold: float,
    max_markers: int,
) -> np.ndarray:
    footprint = ellipsoid_footprint(spacing_um, radius_um)
    local_max = score >= ndi.maximum_filter(score, footprint=footprint, mode="nearest") - 1e-7
    candidates = local_max & foreground & (score >= threshold)

    # Plateau maxima become one marker. This is intentionally more tolerant
    # than assigning every local-maximum voxel its own object identity.
    markers, count = ndi.label(candidates)
    if count > max_markers:
        values = []
        for marker_id in range(1, count + 1):
            mask = markers == marker_id
            values.append((float(score[mask].max()), marker_id))
        keep = {m for _, m in sorted(values, reverse=True)[:max_markers]}
        markers = np.where(np.isin(markers, list(keep)), markers, 0)
        markers, count = ndi.label(markers > 0)

    # Every disconnected foreground component needs at least one seed, but a
    # component can have many candidate seeds; the later RAG decides whether
    # those watershed basins should merge.
    fg_cc, fg_count = ndi.label(foreground)
    next_id = int(markers.max()) + 1
    for cc_id in range(1, fg_count + 1):
        region = fg_cc == cc_id
        if np.any(markers[region] > 0):
            continue
        points = np.argwhere(region)
        if len(points) == 0:
            continue
        best = points[np.argmax(score[region])]
        markers[tuple(best)] = next_id
        next_id += 1
    return markers.astype(np.int32, copy=False)


@torch.no_grad()
def build_markers_fast(
    score: Tensor,
    foreground: Tensor,
    spacing_um: Tensor,
    radius_um: float,
    threshold: float,
    max_markers: int,
    *,
    stage_profiler=None,
    profile_prefix: str = "watershed",
) -> np.ndarray:
    """GPU-friendly deterministic rectangular physical NMS plus CPU CCL.

    The exact ellipsoidal SciPy filter remains the reference implementation.
    This fast backend uses a conservative axis-aligned physical window, then
    preserves the same plateau/component repair semantics.
    """
    radii = torch.ceil(
        torch.as_tensor(radius_um, device=spacing_um.device)
        / spacing_um.float().clamp_min(1e-6)
    ).long().clamp_min(1)
    kernel = tuple(int(2 * value.item() + 1) for value in radii)
    with _profile(stage_profiler, f"{profile_prefix}_marker_max_filter"):
        pooled = F.max_pool3d(
            score[None, None].float(), kernel_size=kernel,
            stride=1, padding=tuple(int(value.item()) for value in radii),
        )[0, 0]
        candidates = (
            (score.float() >= pooled - 1e-7)
            & foreground.bool()
            & (score.float() >= threshold)
        ).cpu().numpy()
    score_np = score.float().cpu().numpy()
    foreground_np = foreground.bool().cpu().numpy()
    with _profile(stage_profiler, f"{profile_prefix}_candidate_connected_components"):
        markers, count = ndi.label(candidates)
    if count > max_markers:
        maximum = ndi.maximum(score_np, labels=markers, index=np.arange(1, count + 1))
        keep_ids = np.argsort(maximum, kind="stable")[-max_markers:] + 1
        markers = np.where(np.isin(markers, keep_ids), markers, 0)
        markers, count = ndi.label(markers > 0)
    with _profile(stage_profiler, f"{profile_prefix}_foreground_connected_components"):
        foreground_cc, foreground_count = ndi.label(foreground_np)
    with _profile(stage_profiler, f"{profile_prefix}_missing_component_seed_repair"):
        present = np.unique(foreground_cc[markers > 0])
        missing = np.setdiff1d(np.arange(1, foreground_count + 1), present, assume_unique=False)
        next_id = int(markers.max()) + 1
        for component_id in missing.tolist():
            region = foreground_cc == component_id
            if not region.any():
                continue
            flat = np.flatnonzero(region)
            best_flat = flat[int(np.argmax(score_np.ravel()[flat]))]
            markers.ravel()[best_flat] = next_id
            next_id += 1
    return markers.astype(np.int32, copy=False)


__all__ = ["build_markers", "build_markers_fast", "ellipsoid_footprint"]
