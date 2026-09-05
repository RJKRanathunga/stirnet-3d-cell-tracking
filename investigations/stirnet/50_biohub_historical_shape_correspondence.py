r"""
Investigation 50: learned historical-shape correspondence for split-only repair.

Place in investigations/stirnet/ and run from the repository root:
    python investigations/stirnet/50_biohub_historical_shape_correspondence.py
    python investigations/stirnet/50_biohub_historical_shape_correspondence.py --features-only
    python investigations/stirnet/50_biohub_historical_shape_correspondence.py --self-test

EXPERIMENT PLAN
==============
Question: does spatially located historical shape information improve WHERE to
cut, beyond current spatial features and a whole-component history summary?

All three variants use the same frozen current-frame RAG/tokenizer features,
the same continuous Inv48 event features, supervision, optimizer and seeds:
  spatial:         current spatial + event features; no history.
  pooled_history:  shared history encoder, but every edge gets only component
                   averages (no left/right or interface correspondence).
  correspondence:  shared candidate encoder reads aligned occupancy and signed
                   distance on each supervoxel AND each side of its interface.

This is the first learned geometry version of the proposed architecture, not a
new dense 3-D appearance CNN. Masks retain identity and location until sampling.
A permutation-invariant candidate encoder is trained jointly with a CUT head.
History at t-1 and t-2 is pooled separately so identities are not counted twice.
There is no required newborn/broken flag and no mandatory event gate. Track
status is continuous context; candidate retrieval uses ALL nearby detections.

Historical inputs are ONLY persisted preprocessed instances, never curated
masks. Coordinates are inverse-sampled using G[t]-G[t-lag] in voxel ZYX, then
SDFs use physical ZYX spacing. No independent recentering of predecessor cells.
The first experiment uses global translation, not learned local registration.
Soft distances tolerate some residual motion; candidate/mask coverage is
reported so bad alignment is observable rather than assumed to be solved.

TRAINING / CALIBRATION
  Default fit targets: existing 2-25; development: existing 30-39.
  Default thresholds: fit frames only, matching the prior investigations. This
  is in-sample calibration and its clean rate is NOT an independent estimate.
  Optional --calibration-frames 20-25 removes those frames and overlapping
  temporal windows from fitting (default event radius 2 => fit 2-15).
  No validation-based checkpoint/threshold/seed selection. Fixed update count.
  The 12 familiar merge cases are DEVELOPMENT, not an untouched test set.
  Temporal windows are disjoint; related biological episodes may still span
  windows. A separate volume/episode holdout is required before promotion.

Loss: balanced CUT/KEEP BCE, sampling components uniformly, plus within-merge
CUT-vs-KEEP ranking. All variants receive identical losses. Normalization is
fit on fitting examples only. History dropout is applied to both history models.
Threshold maximizes recall subject to both KEEP-edge false CUT and clean
COMPONENT ACTION rates on calibration. The latter is a conservative bound on
new false splits because inactive components are exactly preserved.

INFERENCE / METRICS
  Predict on ALL internal RAG edges, including unannotated edges. Annotation
  validity is used only for losses and metrics, never inference eligibility.
  Solve with node_parent_component INSIDE the existing partitioner. No-active-
  action components are kept exactly as the frozen spatial result.
  Report CUT/KEEP (logit and actual partition), exact merge repair, clean false
  split, within-component AUC, per-component records and split-only violations.
  Matched no-history and shuffled-history evaluations retain the learned
  threshold. A target-edge oracle diagnoses the representation/solver ceiling.
  Neither oracle labels nor validation labels enter automatic predictions.

OUTPUTS under runs/stirnet/investigations/50_biohub_historical_shape_correspondence/
  <sample>/feature_cache/<fingerprint>/ : numeric NPZ frames, candidate audits
  <sample>/run_<timestamp>/            : plan, manifest, checkpoints, loss CSVs,
  calibration.csv, metrics.csv, component_metrics.csv, edge_predictions.csv,
     history_ranking.csv, history_geometry_audit.csv, summary.json, candidate_audit.csv

No production source or checkpoint is edited. Inv42's existing cache preparation
may populate a missing spatial cache; use --rebuild-spatial-cache only deliberately.
Requires repository investigations 35, 42, 45, 46, 47, 48 and their dependencies.
Does NOT require Inv46 V3/V4 selector checkpoints or Inv49 result CSVs.
Reviewed against repository commit 73ea380a8591cb217930888fcf9a7c0abafde3b6.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

SCRIPT = "50_biohub_historical_shape_correspondence"
VERSION = 1
PLAN = __doc__
# Interleaved left/right values, followed by component/candidate scalars.
FEATURE_NAMES = [
    "occupancy_u", "occupancy_v", "soft_support_u", "soft_support_v",
    "sdf_u", "sdf_v", "interface_occupancy_u", "interface_occupancy_v",
    "interface_sdf_u", "interface_sdf_v", "component_occupancy",
    "log_predecessor_over_current_volume", "component_soft_support", "in_fov_fraction",
]
LAGS = (1, 2)


def deps():
    global np, pd, ndi, torch, nn, F
    import numpy as np
    import pandas as pd
    from scipy import ndimage as ndi
    import torch
    from torch import nn
    import torch.nn.functional as F


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_repo():
    for candidate in (Path.cwd().resolve(), *Path(__file__).resolve().parents):
        if (candidate / "learned/stirnet").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Put this file in the repository and run from its root.")


def import_repository(root):
    global I48, I47, I46, I45, I42, I35
    sys.path.insert(0, str(root))
    path = root / "investigations/stirnet/48_biohub_event_conditioned_spatial_cut.py"
    if not path.is_file():
        raise FileNotFoundError(f"Required investigation missing: {path}")
    spec = importlib.util.spec_from_file_location("_inv50_inv48", path)
    I48 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = I48
    spec.loader.exec_module(I48)
    I47, I46, I42, I35 = I48.INV47, I48.INV46, I48.INV42, I48.INV35
    I45 = I47.INV45


def parser_for_repo(root):
    # Keep upstream data/cache/cutter arguments, avoiding upstream main()/training.
    p = I46.build_parser()
    p.description = PLAN
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.set_defaults(output=None, print_every=50)
    p.add_argument("--inv50-steps", type=int, default=700)
    p.add_argument("--inv50-lr", type=float, default=3e-4)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--variants", default="spatial,pooled_history,correspondence")
    p.add_argument("--calibration-frames", default="")
    p.add_argument("--history-candidates", type=int, default=8)
    p.add_argument("--candidate-pool", type=int, default=24)
    p.add_argument("--history-radius-dref", type=float, default=1.0)
    p.add_argument("--sdf-scale-dref", type=float, default=0.15)
    p.add_argument("--points-per-region", type=int, default=64)
    p.add_argument("--interface-points", type=int, default=32)
    p.add_argument("--batch-per-class", type=int, default=64)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--history-dropout", type=float, default=0.10)
    p.add_argument("--max-clean-action-rate", type=float, default=0.0)
    p.add_argument("--max-keep-action-rate", type=float, default=0.05)
    p.add_argument("--features-only", action="store_true")
    p.add_argument("--rebuild-history-cache", action="store_true")
    return p


def split_frames(args):
    train = tuple(I42.parse_frame_spec(args.train_frames))
    dev = tuple(I42.parse_frame_spec(args.val_frames))
    radius = int(args.temporal_radius)
    # Event features include a +/- temporal window; history itself is past-only.
    def evidence(frames):
        return {f + d for f in frames for d in range(-max(radius, 2), radius + 1) if f+d >= 0}
    if evidence(train) & evidence(dev):
        raise ValueError("Fit/development temporal windows overlap; widen the frame gap.")
    if args.calibration_frames:
        cal = tuple(I42.parse_frame_spec(args.calibration_frames))
        if not set(cal) <= set(train):
            raise ValueError("--calibration-frames must be a subset of --train-frames.")
        cal_evidence = evidence(cal)
        fit = tuple(t for t in train if not (evidence([t]) & cal_evidence))
        if not fit:
            raise ValueError("No fit frames remain after calibration window guard.")
    else:
        fit, cal = train, train
    if not fit or not dev or not cal:
        raise ValueError("Each requested split must contain frames.")
    return fit, cal, dev


def prepare_context(args, fit, cal, dev):
    """Verified Inv47 preparation, without its unrelated selector checkpoints."""
    I46.validate_inv46_args(args)
    I46.auto_defaults(args)
    reviewed = I42.parse_frame_spec(args.reviewed_frames)
    # Existing validation includes data arguments and window guards.
    I42.validate_args(args, reviewed=reviewed,
                     train_frames=tuple(I42.parse_frame_spec(args.train_frames)), val_frames=dev)
    spacing = I42.parse_spacing(args.spacing)
    paths = I42.make_paths(args)
    device = torch.device(args.device)
    paths.output.mkdir(parents=True, exist_ok=True)
    requested = tuple(sorted(set(fit) | set(cal) | set(dev)))
    ignored, _ = I42.load_ignored_ids(paths, set(reviewed))
    targets = I42.TargetFrameStore(paths, reviewed, ignored)
    I42.prepare_spatial_cache(paths, frames=requested, spacing=spacing, device=device, args=args)
    graph = I42.load_track_graph(paths)
    I42.audit_and_enrich_track_graph(graph, paths=paths, target_frames=requested,
        spacing=spacing, temporal_radius=args.temporal_radius,
        max_error_um=args.max_coordinate_error_um)
    movie = np.load(paths.base_instances, mmap_mode="r", allow_pickle=False)
    if movie.ndim != 4:
        raise ValueError(f"Expected TZYX instance movie, got {movie.shape}")
    shape = tuple(map(int, movie.shape[1:]))
    p45 = I46.inv45_paths_from_inv42(paths, output=paths.output / "motion")
    motion, motion_metrics = I45.load_motion(p45, shape, spacing)
    classification = I45.classify_components(p45, movie, reviewed, I45.ignored_ids(p45, set(reviewed)))
    hyp, hyp_path = I46.load_hypothesis_index(paths.sample, override=args.hypothesis_cache,
                                           plausible_score=args.cutter_plausible_score)
    rows, node_evidence = I46.build_cutter_rows(graph, classification=classification,
        motion=motion, spacing=spacing, shape_zyx=shape,
        near_radius_dref=args.cutter_near_radius_dref,
        boundary_volume_scale=args.cutter_boundary_volume_scale, hypothesis_index=hyp)
    # Cutter fit uses only fitting frames, even when a calibration block is given.
    cutoff, calibration = I46.choose_threshold(rows, train_frames=set(fit),
        target_clean_break_rate=args.cutter_target_clean_break_rate,
        minimum_threshold=args.cutter_min_threshold, override=args.cutter_threshold)
    sanitized, decisions = I46.apply_hard_cutter(graph, frame=rows,
        node_evidence=node_evidence, threshold=cutoff)
    I46.STATE = I46.Inv46State(args=args, paths=paths, spacing=tuple(spacing),
        shape_zyx=shape, motion=motion, motion_metrics=motion_metrics,
        node_evidence=node_evidence, cutter_rows=decisions,
        cutter_threshold=float(cutoff), cutter_metrics={"threshold_calibration": calibration},
        hypothesis_cache=hyp_path, hypothesis_used=hyp_path is not None)
    I46.COMPONENT_PHYSICAL_CACHE.clear()
    I46.CELL_EVIDENCE_BY_KEY = None
    I46.BASE_INSTANCE_MOVIE = None
    I48.COMPONENT_CELL_ID_CACHE.clear()
    initializer = I46.resolve(args.resume) if args.resume else paths.checkpoint
    _, model = I42.load_model(initializer, device)
    model.eval().requires_grad_(False)
    if "node_parent_component" not in inspect.signature(model.partitioner.forward).parameters:
        raise RuntimeError("Partitioner lacks the corrected node_parent_component contract. Update the repository.")
    cumulative = np.asarray(motion.cumulative_float_zyx, dtype=np.float64)
    if cumulative.shape != (movie.shape[0], 3) or not np.isfinite(cumulative).all():
        raise ValueError("Global motion must be finite [T,3] voxel ZYX translations.")
    return dict(paths=paths, device=device, spacing=tuple(spacing), movie=movie,
        cumulative=cumulative, graph=sanitized, model=model, initializer=initializer,
        loader=I42.RuntimeLoader(paths, targets, device), motion_metrics=motion_metrics,
        cutter_threshold=float(cutoff))


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def evenly(points, count):
    if len(points) <= count:
        return points
    return points[np.linspace(0, len(points)-1, count, dtype=np.int64)]


def interface_samples(sv, origin, edge_pairs, count):
    """Exact face-adjacent side voxel centers, in the RAG edge's endpoint order."""
    pairs = {tuple(sorted(map(int, pair))): [] for pair in edge_pairs}
    for axis in range(3):
        a, b = [slice(None)]*3, [slice(None)]*3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        left, right = sv[tuple(a)], sv[tuple(b)]
        locations = np.argwhere((left > 0) & (right > 0) & (left != right))
        if not len(locations):
            continue
        lids, rids = left[tuple(locations.T)], right[tuple(locations.T)]
        # Group faces before Python iteration; avoids looping over every face.
        keys = np.stack([np.minimum(lids, rids), np.maximum(lids, rids)], 1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for group, key in enumerate(unique):
            key = tuple(map(int, key))
            if key not in pairs:
                continue
            rows = np.flatnonzero(inverse == group)
            p = locations[rows].astype(np.float64) + origin
            q = p.copy()
            q[:, axis] += 1
            reverse = lids[rows] != key[0]
            a_points = np.where(reverse[:, None], q, p)
            b_points = np.where(reverse[:, None], p, q)
            pairs[key].append(np.concatenate([a_points, b_points], 1))
    result = []
    for a, b in edge_pairs:
        chunks = pairs[tuple(sorted((int(a), int(b))))]
        if not chunks:
            # Empty explicitly represents a non-face interface, never a false sample.
            result.append((np.empty((0, 3)), np.empty((0, 3))))
            continue
        samples = evenly(np.concatenate(chunks), count)
        result.append((samples[:, :3], samples[:, 3:]) if a < b
                      else (samples[:, 3:], samples[:, :3]))
    return result


def sample_fields(mask, sdf, points, low, shift, spacing, scale):
    """Inverse warp: a point in frame t samples history at point - shift."""
    if not len(points):
        return np.array([0., 0., 1.], np.float32)
    coords = (points - shift - low).T
    occ = ndi.map_coordinates(mask.astype(np.float32, copy=False), coords, order=1,
                              mode="constant", cval=0, prefilter=False)
    dist = ndi.map_coordinates(sdf, coords, order=1, mode="constant",
                               cval=float(scale * 4), prefilter=False)
    inside = np.all((coords >= 0) & (coords <= np.asarray(mask.shape)[:, None]-1), axis=0)
    near = np.exp(-np.maximum(dist, 0) / scale) * inside
    return np.array([occ.mean(), near.mean(), np.clip(dist / scale, -4, 4).mean()/4], np.float32)


def build_component_history(previous, counts, bbox, comp_points, node_points,
                            faces, edge_node_rows, shift, spacing, dref, volume_vox, args):
    """Returns candidate-specific edge and pooled features without any target input."""
    k, ne = args.history_candidates, len(edge_node_rows)
    out = np.zeros((ne, k, len(FEATURE_NAMES)), np.float32)
    pooled = np.zeros((k, len(FEATURE_NAMES)), np.float32)
    valid = np.zeros(k, bool)
    radius = max(args.history_radius_dref * dref, 1e-3)
    scale = max(args.sdf_scale_dref * dref, 1e-3)
    # ROI includes a margin outside the clipped SDF range to avoid crop-edge SDF artifacts.
    margin = (radius + 4*scale) / spacing
    low = np.maximum(np.floor(bbox[0] - shift - margin).astype(int), 0)
    high = np.minimum(np.ceil(bbox[1] - shift + margin).astype(int), previous.shape)
    if np.any(high <= low):
        return out, pooled, valid, [dict(candidate_id=0, reason="outside_history_fov")]
    roi = previous[tuple(slice(int(a), int(b)) for a,b in zip(low, high))]
    ids = np.unique(roi)
    ids = ids[ids > 0]
    boxes = ndi.find_objects(roi)  # One bounded ROI, not a full-volume EDT per object.
    candidates = []
    center_prev = comp_points.mean(0) - shift
    for cell in ids:
        sl = boxes[int(cell)-1]
        if sl is None:
            continue
        lo = low + np.array([s.start for s in sl])
        hi = low + np.array([s.stop-1 for s in sl])
        sep = np.maximum(np.maximum(lo - (bbox[1]-shift), (bbox[0]-shift)-hi), 0)*spacing
        if np.linalg.norm(sep) > radius:
            continue
        center_dist = np.linalg.norm((np.clip(center_prev, lo, hi)-center_prev)*spacing)
        candidates.append((float(center_dist), int(cell)))
    candidates.sort()
    retrieved = len(candidates)
    candidates = candidates[:args.candidate_pool]
    qualified = []
    fov = np.mean(np.all((comp_points-shift >= 0) & (comp_points-shift <= np.asarray(previous.shape)-1), 1))
    for _, cell in candidates:
        mask = roi == cell
        # Positive outside, negative inside; use anisotropic physical spacing.
        sdf = (ndi.distance_transform_edt(~mask, sampling=spacing)
               - ndi.distance_transform_edt(mask, sampling=spacing)).astype(np.float32)
        stats = sample_fields(mask, sdf, comp_points, low, shift, spacing, scale)
        # Prefer actual aligned support. Rank is label-free and deterministic.
        score = float(stats[0] + 0.25*stats[1])
        qualified.append((score, cell, stats, mask, sdf))
        # Bound retained ROI memory even if many plausible cells surround a component.
        qualified.sort(key=lambda x: (-x[0], x[1]))
        if len(qualified) > k:
            qualified.pop()
    qualified.sort(key=lambda x: (-x[0], x[1]))
    logs = []
    for slot, (score, cell, cs, mask, sdf) in enumerate(qualified[:k]):
        mask = mask.astype(np.float32)
        valid[slot] = True
        ratio = float(counts[cell] / max(volume_vox, 1))
        common = np.array([cs[0], np.clip(np.log(max(ratio, 1e-8)), -4, 4)/4, cs[1], fov], np.float32)
        # Component-only control preserves temporal candidate volume/support but
        # destroys all within-component correspondence and interface location.
        pooled[slot] = np.r_[cs[0], cs[0], cs[1], cs[1], cs[2], cs[2], cs[0], cs[0], cs[2], cs[2], common]
        pernode = [sample_fields(mask, sdf, pts, low, shift, spacing, scale) for pts in node_points]
        for e, (u, v) in enumerate(edge_node_rows):
            su, sv = pernode[u], pernode[v]
            iu = sample_fields(mask, sdf, faces[e][0], low, shift, spacing, scale)
            iv = sample_fields(mask, sdf, faces[e][1], low, shift, spacing, scale)
            out[e, slot] = np.r_[su[0], sv[0], su[1], sv[1], su[2], sv[2], iu[0], iv[0], iu[2], iv[2], common]
        logs.append(dict(candidate_id=cell, candidate_slot=slot, score=score,
            occupancy=float(cs[0]), soft_support=float(cs[1]), predecessor_volume_vox=int(counts[cell]),
            current_volume_vox=int(volume_vox), volume_ratio=ratio,
            fov_fraction=float(fov), retrieved_candidates=retrieved,
            candidate_pool_truncated=max(0, retrieved-args.candidate_pool),
            candidate_slots_truncated=max(0, len(candidates)-k)))
    if not logs:
        logs = [dict(candidate_id=0, reason="no_candidate_in_search_radius", retrieved_candidates=retrieved)]
    return out, pooled, valid, logs


def fingerprint(ctx, args, frames):
    h = hashlib.sha256(Path(__file__).read_bytes())
    for module in (I48, I47, I46, I45, I42, I35):
        h.update(Path(module.__file__).read_bytes())
    p = ctx['paths']
    # Full initializer hash; large immutable movies use stat identity, never hash 80GB.
    h.update(I42.sha256(Path(ctx['initializer'])).encode())
    files = [p.base_instances, p.supervoxels, p.track_graph, p.inference_manifest,
             p.spatial_operations, p.annotation_manifest, p.track_state]
    if I46.STATE.hypothesis_cache is not None:
        files.append(Path(I46.STATE.hypothesis_cache))
    for path in sorted((locate_repo() / 'learned/stirnet/model').rglob('*.py')):
        h.update(path.read_bytes())
    files += [p.graph_cache(t) for t in frames]
    files += [p.spatial_meta(t) for t in frames]
    files += list(p.instance_annotations.rglob('*.json')) + list(p.instance_annotations.rglob('*.npy'))
    for path in sorted(set(map(Path, files))):
        if path.exists():
            s = path.stat()
            h.update(f'{path.resolve()}:{s.st_size}:{s.st_mtime_ns}'.encode())
    h.update(ctx['cumulative'].tobytes())
    h.update(str(ctx['cutter_threshold']).encode())
    for key in ('history_candidates','candidate_pool','history_radius_dref','sdf_scale_dref',
                'points_per_region','interface_points','temporal_radius','spacing',
                'train_frames','calibration_frames','reviewed_frames','cutter_plausible_score'):
        h.update(f'{key}:{getattr(args,key)}'.encode())
    return h.hexdigest()[:24]


def extract_frame(ctx, t, args):
    runtime = ctx['loader'].load(t)
    case = I42.build_real_case(runtime)
    with torch.inference_mode():
        # Only current spatial tokens are needed; do not run the unused temporal encoder.
        model = ctx['model']
        decoded, geometry = I35.dummy_geometry_and_decode(model, case.rag.node_features)
        instances = model.instance_tokenizer(case.partition, case.rag, decoded, geometry,
            torch.tensor([ctx['spacing']], device=ctx['device'], dtype=torch.float32),
            torch.tensor([runtime.dref_um], device=ctx['device'], dtype=torch.float32),
            profile_prefix='inv50_frozen_tokenizer')
        encoded = SimpleNamespace(instances=instances)
        spatial = numpy(I48.rich_spatial_edge_features(encoded, case)).astype(np.float32)
        event, _ = I48.component_event_features(runtime, case,
            status_index=I48.graph_status_index(ctx['graph']))
    current = numpy(case.node_current_component).astype(np.int64)
    edge = numpy(case.rag.edge_index).astype(np.int64)
    internal = current[edge[0]] == current[edge[1]]
    # Do NOT use case.editable here: it includes label validity.
    edge_ids = np.flatnonzero(internal)
    edge_comp = current[edge[0, edge_ids]]
    sv = numpy(runtime.rag.supervoxel_labels[0]).astype(np.int64)
    node_sv = numpy(runtime.rag.node_supervoxel_id).astype(np.int64)
    lookup = np.zeros(int(sv.max())+1, np.int32)
    lookup[node_sv] = current+1
    component_volume = lookup[sv]
    boxes = ndi.find_objects(component_volume)
    history = np.zeros((len(edge_ids), len(LAGS), args.history_candidates, len(FEATURE_NAMES)), np.float32)
    pooled = np.zeros((len(boxes), len(LAGS), args.history_candidates, len(FEATURE_NAMES)), np.float32)
    valid = np.zeros((len(boxes), len(LAGS), args.history_candidates), bool)
    spacing = np.asarray(ctx['spacing'], np.float64)
    counts = {lag: np.bincount(np.asarray(ctx['movie'][t-lag]).ravel().astype(np.int64))
              for lag in LAGS if t-lag >= 0}
    audit = []
    component_cells = I48.component_cell_ids(runtime, case)
    for component, box in enumerate(boxes):
        rows = np.flatnonzero(edge_comp == component)
        if box is None or not len(rows):
            continue
        low = np.array([s.start for s in box])
        high = np.array([s.stop for s in box])
        sv_roi = sv[box]
        comp_mask = component_volume[box] == component+1
        coords = np.argwhere(comp_mask).astype(np.float64) + low
        node_rows = np.flatnonzero(current == component)
        local_node = {int(n): i for i,n in enumerate(node_rows)}
        node_points = [evenly(np.argwhere(sv_roi == node_sv[n]).astype(np.float64)+low,
                              args.points_per_region) for n in node_rows]
        rag_rows = edge_ids[rows]
        endpoints = edge[:, rag_rows].T
        edge_nodes = [(local_node[int(u)], local_node[int(v)]) for u,v in endpoints]
        faces = interface_samples(sv_roi, low, node_sv[endpoints], args.interface_points)
        for li, lag in enumerate(LAGS):
            if t-lag < 0:
                continue
            shift = ctx['cumulative'][t] - ctx['cumulative'][t-lag]
            h, p, v, log = build_component_history(np.asarray(ctx['movie'][t-lag]), counts[lag],
                (low, high), evenly(coords, max(args.points_per_region, 256)), node_points,
                faces, edge_nodes, shift, spacing, runtime.dref_um, len(coords), args)
            history[rows, li], pooled[component, li], valid[component, li] = h, p, v
            for r in log:
                r.update(frame=t, component=component, current_cell_id=int(component_cells[component]),
                    lag=lag, source_frame=t-lag, shift_z=float(shift[0]),
                    shift_y=float(shift[1]), shift_x=float(shift[2]))
                audit.append(r)
    arrays = dict(t=np.array(t), edge_ids=edge_ids, edge_comp=edge_comp,
        edge_index=edge, current=current, spatial=spatial[edge_ids],
        event=numpy(event).astype(np.float32), history=history, pooled=pooled, valid=valid,
        ycut=(~numpy(case.target_keep)[edge_ids]).astype(np.float32),
        supervised=numpy(case.edge_valid)[edge_ids].astype(bool),
        trusted=numpy(case.split_valid & case.metric_component_valid).astype(bool),
        merge=numpy(case.split_target).astype(bool),
        node_target=numpy(runtime.node_target).astype(np.int64),
        node_valid=numpy(runtime.node_valid).astype(bool),
        base_logits=numpy(case.rag.spatial_edge_logits).astype(np.float32),
        component_cells=component_cells)
    for name in ('spatial','event','history','pooled'):
        if not np.isfinite(arrays[name]).all():
            raise FloatingPointError(f'Nonfinite {name} at frame {t}')
    return arrays, audit


def extract_all(ctx, frames, args, cache):
    data, audits = {}, []
    cache.mkdir(parents=True, exist_ok=True)
    for i,t in enumerate(frames):
        begin = time.perf_counter()
        path = cache / f't{t:03d}.npz'
        logpath = cache / f't{t:03d}_candidates.json'
        if path.exists() and logpath.exists() and not args.rebuild_history_cache:
            with np.load(path, allow_pickle=False) as z:
                data[t] = {key:z[key] for key in z.files}
            rows = json.loads(logpath.read_text(encoding='utf-8'))
            mode = 'cache'
        else:
            data[t], rows = extract_frame(ctx, t, args)
            temp = path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, **data[t])
            os.replace(temp, path)
            atomic_json(logpath, rows)
            mode = 'built'
        audits.extend(rows)
        print(f'[history {i+1}/{len(frames)}] t={t:03d} {mode} internal={len(data[t]["edge_ids"])} '
              f'merges={int((data[t]["trusted"] & data[t]["merge"]).sum())} '
              f'{time.perf_counter()-begin:.1f}s', flush=True)
    return data, pd.DataFrame(audits)


def make_model(spatial_dim, event_dim, lags, variant, center, scale):
    class HistoryCutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant = variant
            self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32))
            self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
            self.spatial = nn.Sequential(nn.Linear(spatial_dim,128),nn.SiLU(),nn.Dropout(.1))
            self.event = nn.Sequential(nn.Linear(event_dim,16),nn.SiLU())
            if variant != 'spatial':
                self.candidate = nn.Sequential(nn.Linear(24,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
            self.head = nn.Sequential(nn.Linear(128+16+(lags*40 if variant != 'spatial' else 0),64),
                                      nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))

        def forward(self, spatial, event, hist, valid):
            x = torch.cat([spatial,event], -1)
            x = ((x-self.center)/self.scale).clamp(-8,8)
            parts = [self.spatial(x[:,:spatial_dim]), self.event(x[:,spatial_dim:])]
            if self.variant != 'spatial':
                # Exact invariance to swapping u/v and permuting predecessor IDs.
                a,b = hist[...,:10:2],hist[...,1:10:2]
                symmetric = torch.cat([(a+b)*.5,(a-b).abs(),a*b,torch.maximum(a,b),hist[...,10:]],-1)
                c = self.candidate(symmetric)
                m = valid.unsqueeze(-1).float()
                mean = (c*m).sum(-2)/m.sum(-2).clamp_min(1)
                maximum = c.masked_fill(~valid.unsqueeze(-1),-1e4).max(-2).values
                maximum = torch.where(valid.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
                # Soft unknown assignment prevents low overlap implying a mandatory cut.
                su = (.75*hist[...,0]+.25*hist[...,2])*valid
                sv = (.75*hist[...,1]+.25*hist[...,3])*valid
                pu,pv = su/(su.sum(-1,keepdim=True)+.1),sv/(sv.sum(-1,keepdim=True)+.1)
                ku,kv = pu.sum(-1),pv.sum(-1)
                same = (pu*pv).sum(-1)
                different = (ku*kv-same).clamp_min(0)
                weights = torch.maximum(su,sv)
                volume = ((hist[...,11]*4).exp()*weights).sum(-1).clamp_max(8)/8
                scalars = torch.stack([same,different,torch.minimum(ku,kv),torch.maximum(ku,kv),
                    valid.float().mean(-1),volume,(weights*hist[...,10]).sum(-1)/(weights.sum(-1)+.1),
                    (weights*hist[...,13]).sum(-1)/(weights.sum(-1)+.1)],-1)
                parts.append(torch.cat([mean,maximum,scalars],-1).flatten(1))
            return self.head(torch.cat(parts,-1)).squeeze(-1)
    return HistoryCutNet()


def dataset_for_frames(data, frames, variant, supervised=False, control=None, seed=0):
    chunks = {k:[] for k in ('spatial','event','hist','valid','y','group','merge','edge_id','frame')}
    group_offset = 0
    for t in frames:
        d = data[t]
        rows = np.flatnonzero(d['supervised']) if supervised else np.arange(len(d['edge_ids']))
        c = d['edge_comp'][rows]
        hist = d['pooled'][c] if variant=='pooled_history' else d['history'][rows]
        valid = d['valid'][c]
        if control == 'zero':
            hist,valid = np.zeros_like(hist),np.zeros_like(valid)
        elif control == 'shuffle':
            # Corrupt full history bundles among edges WITHOUT reading targets.
            rng = np.random.default_rng(seed+50000+t)
            order = rng.permutation(len(rows))
            hist,valid = hist[order],valid[order]
        chunks['spatial'].append(d['spatial'][rows]); chunks['event'].append(d['event'][c])
        chunks['hist'].append(hist); chunks['valid'].append(valid)
        chunks['y'].append(d['ycut'][rows]); chunks['group'].append(c+group_offset)
        chunks['merge'].append(d['merge'][c]); chunks['edge_id'].append(d['edge_ids'][rows])
        chunks['frame'].append(np.full(len(rows),t))
        group_offset += len(d['trusted'])
    return {k:np.concatenate(v) for k,v in chunks.items()}


def train_model(dataset, variant, args, device, seed, out):
    seed_all(seed)
    y = dataset['y']
    if not (np.any(y==1) and np.any(y==0)):
        raise RuntimeError('Fitting requires trusted CUT and KEEP labels; adjust fit frames.')
    raw = np.concatenate([dataset['spatial'],dataset['event']],1)
    center = raw.mean(0); scale = raw.std(0).clip(.05)
    model = make_model(dataset['spatial'].shape[1],dataset['event'].shape[1],len(LAGS),variant,center,scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.inv50_lr,weight_decay=1e-4)
    # Keep cached inputs in CPU RAM, copying only small minibatches to the GPU.
    tensors = {k:torch.from_numpy(np.array(dataset[k],copy=True)) for k in ('spatial','event','hist','valid','y')}
    rng = np.random.default_rng(seed)
    groups = dataset['group']
    pos_groups,neg_groups,rank_groups = [],[],[]
    for g in np.unique(groups):
        p = np.flatnonzero((groups==g)&(y==1)); n = np.flatnonzero((groups==g)&(y==0))
        if len(p): pos_groups.append(p)
        if len(n): neg_groups.append(n)
        if len(p) and len(n): rank_groups.append((p,n))
    losses=[]
    for step in range(1,args.inv50_steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        def pick(pool,n):
            return np.array([rng.choice(pool[int(rng.integers(len(pool)))]) for _ in range(n)])
        batch = np.r_[pick(pos_groups,args.batch_per_class),pick(neg_groups,args.batch_per_class)]
        # Ranking pairs sampled independently from mixed components, preserving edge groups.
        rp,rn=[],[]
        if rank_groups and args.ranking_weight:
            for _ in range(min(32,len(rank_groups)*2)):
                p,n = rank_groups[int(rng.integers(len(rank_groups)))]; rp.append(rng.choice(p));rn.append(rng.choice(n))
        idx=np.r_[batch,rp,rn].astype(np.int64)
        inputs={k:tensors[k][idx].to(device) for k in ('spatial','event','hist','valid')}
        if variant!='spatial' and args.history_dropout:
            retain=(torch.rand((len(idx),len(LAGS),1),device=device)>=args.history_dropout)
            inputs['valid']=inputs['valid'] & retain
        logits=model(**inputs)
        bce=F.binary_cross_entropy_with_logits(logits[:len(batch)],tensors['y'][batch].to(device))
        rank=F.softplus(-(logits[len(batch):len(batch)+len(rp)]-logits[len(batch)+len(rp):])).mean() if rp else logits.new_zeros(())
        loss=bce+args.ranking_weight*rank
        if not torch.isfinite(loss): raise FloatingPointError(f'{variant}: nonfinite loss at {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        losses.append(dict(step=step,bce=float(bce.detach()),rank=float(rank.detach()),total=float(loss.detach())))
        if step==1 or step%args.print_every==0 or step==args.inv50_steps:
            print(f'[train {variant} seed={seed} {step}/{args.inv50_steps}] loss={float(loss.detach()):.5f}',flush=True)
    pd.DataFrame(losses).to_csv(out/f'{variant}_seed{seed}_loss.csv',index=False)
    return model


def probabilities(model, dataset, device):
    model.eval(); result=[]
    with torch.inference_mode():
        for start in range(0,len(dataset['y']),1024):
            stop=start+1024
            inputs={k:torch.as_tensor(dataset[k][start:stop],device=device) for k in ('spatial','event','hist','valid')}
            result.append(model(**inputs).sigmoid().cpu().numpy())
    return np.concatenate(result) if result else np.empty(0,np.float32)


def calibrate(data, frames, probability, args):
    keep_scores=[];clean_max=[];cut_scores=[];cursor=0
    for t in frames:
        d=data[t]; p=probability[cursor:cursor+len(d['edge_ids'])];cursor+=len(p)
        keep_scores.extend(p[d['supervised'] & (d['ycut']==0)].tolist())
        cut_scores.extend(p[d['supervised'] & (d['ycut']==1)].tolist())
        for c in np.flatnonzero(d['trusted'] & ~d['merge']):
            s=p[d['edge_comp']==c]
            clean_max.append(float(s.max()) if len(s) else -1.)
    if not keep_scores or not clean_max:
        raise RuntimeError('Calibration requires trusted KEEP edges and clean components.')
    # Exact changes in the threshold decision, including an explicit no-action choice.
    candidates=np.unique(np.r_[0., np.nextafter(np.asarray(keep_scores+clean_max,np.float64),np.inf),1.0000001])
    table=[];chosen=None
    for threshold in candidates:
        kr=float(np.mean(np.asarray(keep_scores)>=threshold)); cr=float(np.mean(np.asarray(clean_max)>=threshold))
        recall=float(np.mean(np.asarray(cut_scores)>=threshold)) if cut_scores else None
        row=dict(threshold=float(threshold),keep_action_rate=kr,clean_action_rate=cr,cut_recall=recall,
                 clean_count=len(clean_max),keep_count=len(keep_scores),cut_count=len(cut_scores))
        table.append(row)
        if chosen is None and kr<=args.max_keep_action_rate+1e-12 and cr<=args.max_clean_action_rate+1e-12:
            chosen=row
    if chosen is None: raise RuntimeError('No legal calibration threshold, including no-action.')
    return chosen,table


def auc_pair(scores, targets):
    pos=scores[targets==1];neg=scores[targets==0]
    if not len(pos) or not len(neg):return None
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))


def geometry_audit(data, frames):
    """Descriptive label-based scoring AFTER label-free feature construction."""
    rows=[]
    for t in frames:
        d=data[t]
        for c in np.flatnonzero(d['trusted']):
            er=np.flatnonzero(d['edge_comp']==c)
            if not len(er):
                continue
            for li,lag in enumerate(LAGS):
                h=d['history'][er,li]
                valid=d['valid'][c,li]
                u=(.75*h[...,0]+.25*h[...,2])*valid
                v=(.75*h[...,1]+.25*h[...,3])*valid
                pu=u/(u.sum(-1,keepdims=True)+.1)
                pv=v/(v.sum(-1,keepdims=True)+.1)
                coverage=pu.sum(-1)*pv.sum(-1)
                same=(pu*pv).sum(-1)
                different=np.maximum(coverage-same,0)
                score=different-same
                sup=d['supervised'][er]
                rows.append(dict(frame=t,component=int(c),cell_id=int(d['component_cells'][c]),
                    merge=bool(d['merge'][c]),lag=lag,candidate_count=int(valid.sum()),
                    mean_joint_support=float(coverage.mean()),
                    max_different_predecessor_support=float(different.max()),
                    within_component_auc=auc_pair(score[sup],d['ycut'][er][sup])))
    return pd.DataFrame(rows)


def repair_partition(ctx, t, d, action):
    """Inference only: inputs contain no ground truth; target-free actions already fixed."""
    if not np.any(action):
        return d['current'].copy(),d['base_logits'].copy()
    runtime=ctx['loader'].load(t)
    device=ctx['device']
    # Runtime loader also owns targets, but none are accessed here.
    logits=torch.as_tensor(d['base_logits'],device=device).clone()
    action_ids=d['edge_ids'][action]
    logits[torch.as_tensor(action_ids,device=device)]=-20.
    current=torch.as_tensor(d['current'],device=device,dtype=torch.long)
    with torch.inference_mode():
        part=ctx['model'].partitioner(runtime.rag,logits,
            ctx['model'].cfg.partition.final_merge_threshold,stage='final',node_parent_component=current)
    predicted=numpy(part.node_component_global).astype(np.int64)
    # Verify the actual solve before any inactive-component normalization.
    for p in np.unique(predicted):
        if len(np.unique(d['current'][predicted==p]))>1:
            raise RuntimeError('Partitioner violated in-solver frozen-component constraint.')
    active=np.zeros(len(d['trusted']),bool);active[d['edge_comp'][action]]=True
    codes=np.stack([d['current'],np.where(active[d['current']],predicted,0)],1)
    _,labels=np.unique(codes,axis=0,return_inverse=True)
    return labels,numpy(logits)


def evaluate(ctx,data,frames,scores,threshold,name,seed,oracle=False):
    comp_rows=[];edge_rows=[];ranking=[];cursor=0
    total=dict(cut_total=0,cut_correct=0,keep_total=0,keep_correct=0,partition_cut_correct=0,
               partition_keep_correct=0,merge_total=0,exact=0,clean_total=0,clean_split=0,actions=0)
    for t in frames:
        d=data[t];n=len(d['edge_ids']);p=np.asarray(scores[cursor:cursor+n],np.float64);cursor+=n
        action=p>=threshold
        if oracle:
            action=d['supervised'] & (d['ycut']==1)
        predicted,logits=repair_partition(ctx,t,d,action)
        keep_logits=1/(1+np.exp(-np.clip(logits[d['edge_ids']],-50,50)))>=float(ctx['model'].cfg.partition.final_merge_threshold)
        edge=d['edge_index'][:,d['edge_ids']]
        partition_cut=predicted[edge[0]]!=predicted[edge[1]]
        cut=d['supervised']&(d['ycut']==1);keep=d['supervised']&(d['ycut']==0)
        for key,value in [('cut_total',cut.sum()),('cut_correct',(~keep_logits[cut]).sum()),
                          ('keep_total',keep.sum()),('keep_correct',keep_logits[keep].sum()),
                          ('partition_cut_correct',partition_cut[cut].sum()),
                          ('partition_keep_correct',(~partition_cut[keep]).sum()),('actions',action.sum())]:
            total[key]+=int(value)
        for c in np.flatnonzero(d['trusted']):
            nodes=np.flatnonzero(d['current']==c);er=np.flatnonzero(d['edge_comp']==c)
            target=d['node_target'][nodes];pred=predicted[nodes]
            # Exact iff both induced partitions refine one another.
            exact=(all(len(np.unique(pred[target==v]))==1 for v in np.unique(target))
                   and all(len(np.unique(target[pred==v]))==1 for v in np.unique(pred)))
            split=len(np.unique(pred))>1
            if d['merge'][c]: total['merge_total']+=1;total['exact']+=int(exact)
            else: total['clean_total']+=1;total['clean_split']+=int(split)
            comp_rows.append(dict(variant=name,seed=seed,frame=t,component=int(c),
                cell_id=int(d['component_cells'][c]),merge=bool(d['merge'][c]),exact=bool(exact),
                predicted_parts=len(np.unique(pred)),target_parts=len(np.unique(target)),actions=int(action[er].sum()),
                max_cut_score=float(p[er].max()) if len(er) else None))
            valid_rows=er[d['supervised'][er]]
            auc=auc_pair(p[valid_rows],d['ycut'][valid_rows])
            if auc is not None:
                ranking.append(dict(variant=name,seed=seed,frame=t,component=int(c),within_component_auc=auc,
                    top_edge_is_cut=bool(d['ycut'][valid_rows[np.argmax(p[valid_rows])]]),
                    lowest_cut=float(p[valid_rows][d['ycut'][valid_rows]==1].min()),
                    highest_keep=float(p[valid_rows][d['ycut'][valid_rows]==0].max())))
        for e in np.flatnonzero(d['supervised']):
            edge_rows.append(dict(variant=name,seed=seed,frame=t,edge_row=int(d['edge_ids'][e]),
                component=int(d['edge_comp'][e]),target_cut=bool(d['ycut'][e]),score=float(p[e]),
                action=bool(action[e]),partition_cut=bool(partition_cut[e])))
    metrics=dict(variant=name,seed=seed,threshold=float(threshold),**total,
        exact_rate=total['exact']/max(total['merge_total'],1),
        clean_false_split_rate=total['clean_split']/max(total['clean_total'],1),
        within_component_auc=float(np.mean([r['within_component_auc'] for r in ranking])) if ranking else None,
        split_only_violations=0)
    print(f'[DEV {name} seed={seed}] CUT {total["cut_correct"]}/{total["cut_total"]} '
          f'KEEP {total["keep_correct"]}/{total["keep_total"]} exact {total["exact"]}/{total["merge_total"]} '
          f'clean {total["clean_split"]}/{total["clean_total"]}',flush=True)
    return metrics,comp_rows,edge_rows,ranking


def self_test():
    """Focused functional tests, synthetic data only; no repository or dataset needed."""
    deps();torch.set_num_threads(2)
    from tempfile import TemporaryDirectory
    args=SimpleNamespace(history_candidates=4,candidate_pool=8,history_radius_dref=1.,sdf_scale_dref=.15)
    previous=np.zeros((8,12,18),np.int32)
    previous[2:6,3:9,2:7]=1;previous[2:6,3:9,9:14]=2
    shift=np.array([0.,0.,2.]);spacing=np.array([1.625,.40625,.40625])
    a=np.argwhere(previous==1).astype(float)+shift;b=np.argwhere(previous==2).astype(float)+shift
    coords=np.r_[a,b];bbox=(coords.min(0),coords.max(0)+1)
    counts=np.bincount(previous.ravel())
    h,p,v,_=build_component_history(previous,counts,bbox,coords,[a,b],[(a,b)],[(0,1)],shift,spacing,4.,len(coords),args)
    assert v.sum()==2
    assert h[0,:,0].max()>.99 and h[0,:,1].max()>.99
    assert np.all(h[0,:,0]*h[0,:,1]<1e-6), 'distinct predecessors were combined'
    unaligned=build_component_history(previous,counts,bbox,coords,[a,b],[(a,b)],[(0,1)],np.zeros(3),spacing,4.,len(coords),args)[0]
    assert h[...,0].max()>unaligned[...,0].max(), 'translation direction test failed'
    # No candidate history must be finite and neutral; no wrap at image boundary.
    empty=build_component_history(previous,counts,bbox,coords,[a,b],[(a,b)],[(0,1)],np.array([100,0,0]),spacing,4.,len(coords),args)
    assert not empty[2].any() and np.isfinite(empty[0]).all()
    seed_all(50)
    model=make_model(5,3,2,'correspondence',np.zeros(8),np.ones(8)).eval()
    spatial=torch.randn(8,5);event=torch.randn(8,3)
    hist=torch.tensor(np.broadcast_to(h,(8,2,4,14)).copy());valid=torch.tensor(np.broadcast_to(v,(8,2,4)).copy())
    expected=model(spatial,event,hist,valid)
    assert torch.allclose(expected,model(spatial,event,hist[:,:,torch.tensor([2,0,3,1])],valid[:,:,torch.tensor([2,0,3,1])]),atol=1e-6)
    swapped=hist.clone();swapped[...,:10:2]=hist[...,1:10:2];swapped[...,1:10:2]=hist[...,:10:2]
    assert torch.allclose(expected,model(spatial,event,swapped,valid),atol=1e-6)
    assert torch.isfinite(model(spatial,event,hist,torch.zeros_like(valid))).all()
    loss=F.binary_cross_entropy_with_logits(expected,torch.arange(8).float()%2);loss.backward()
    assert model.candidate[0].weight.grad.abs().sum()>0, 'history branch receives no gradient'
    # Calibration guards use all internal edges of clean components.
    d=dict(edge_ids=np.arange(4),supervised=np.array([1,1,1,0],bool),ycut=np.array([1,0,0,0]),
           trusted=np.array([1,1],bool),merge=np.array([1,0],bool),edge_comp=np.array([0,0,1,1]))
    chosen,_=calibrate({0:d},[0],np.array([.99,.1,.2,.9]),SimpleNamespace(max_keep_action_rate=0.,max_clean_action_rate=0.))
    assert chosen['threshold']>.9 and chosen['threshold']<.99
    # Exercise actual mini-training, normalization, ranking, all three forward paths.
    with TemporaryDirectory() as tmp:
        for variant in ('spatial','pooled_history','correspondence'):
            n=32
            ds=dict(spatial=np.random.randn(n,5).astype('float32'),event=np.random.randn(n,3).astype('float32'),
                hist=np.random.rand(n,2,4,14).astype('float32'),valid=np.ones((n,2,4),bool),
                y=(np.arange(n)%2).astype('float32'),group=np.arange(n)//4)
            ta=SimpleNamespace(inv50_lr=1e-3,inv50_steps=3,batch_per_class=4,ranking_weight=.25,
                               history_dropout=.1,print_every=10)
            trained=train_model(ds,variant,ta,torch.device('cpu'),7,Path(tmp))
            assert np.isfinite(probabilities(trained,ds,torch.device('cpu'))).all()
    print('SELF-TEST PASS: alignment, identity separation, empty/FOV history, candidate permutation, '
          'edge-side symmetry, gradient flow, clean-component calibration, and three-variant training.',flush=True)


def main():
    if '--plan' in sys.argv:
        print(PLAN);return 0
    if '--self-test' in sys.argv:
        self_test();return 0
    deps();root=locate_repo();import_repository(root)
    args=parser_for_repo(root).parse_args()
    if args.output is None:
        args.output=root/'runs/stirnet/investigations'/SCRIPT/args.sample_id
    args.output=Path(args.output).resolve()
    variants=[v.strip() for v in args.variants.split(',') if v.strip()]
    seeds=[int(v) for v in args.seeds.split(',')]
    if not variants or not set(variants)<= {'spatial','pooled_history','correspondence'}:
        raise ValueError('Unknown/empty --variants')
    for key in ('inv50_steps','history_candidates','candidate_pool','points_per_region','interface_points','batch_per_class','print_every'):
        if getattr(args,key)<1:raise ValueError(f'{key} must be positive')
    if args.candidate_pool<args.history_candidates:raise ValueError('candidate pool must be >= candidate count')
    for key in ('inv50_lr','history_radius_dref','sdf_scale_dref'):
        if getattr(args,key)<=0:raise ValueError(f'{key} must be positive')
    for key in ('history_dropout','max_clean_action_rate','max_keep_action_rate'):
        if not 0<=getattr(args,key)<=1:raise ValueError(f'{key} must be in [0,1]')
    if args.ranking_weight<0:raise ValueError('ranking weight must be nonnegative')
    fit,cal,dev=split_frames(args)
    out=args.output/f'run_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.txt').write_text(PLAN,encoding='utf-8')
    print(f'INVESTIGATION 50\nfit={fit}\ncalibration={cal}\ndevelopment={dev}\noutput={out}',flush=True)
    print('Calibration uses fit frames.' if not args.calibration_frames else 'Calibration uses a separate guarded block.',flush=True)
    ctx=prepare_context(args,fit,cal,dev)
    frames=tuple(sorted(set(fit)|set(cal)|set(dev)))
    key=fingerprint(ctx,args,frames)
    cache=args.output/'feature_cache'/key
    manifest=dict(version=VERSION,script=SCRIPT,reference_commit='73ea380a8591cb217930888fcf9a7c0abafde3b6',
        args=vars(args),fit_frames=fit,calibration_frames=cal,development_frames=dev,
        calibration_independent_of_fit=bool(args.calibration_frames),untouched_holdout=False,
        history_lags=LAGS,history_source=str(ctx['paths'].base_instances),feature_names=FEATURE_NAMES,
        feature_fingerprint=key,motion=ctx['motion_metrics'],cutter_threshold=ctx['cutter_threshold'],
        initializer=str(ctx['initializer']),normalization='fit only',checkpoint_selection='fixed final step',
        inference_action_mask='all internal RAG edges, independent of annotations',
        history_registration='global translation in voxel ZYX; physical SDF; no local deformation')
    atomic_json(out/'manifest.json',manifest)
    data,audit=extract_all(ctx,frames,args,cache)
    if len(audit):
        audit['trusted']=[bool(data[int(r.frame)]['trusted'][int(r.component)]) for r in audit.itertuples()]
        audit['merge']=[bool(data[int(r.frame)]['merge'][int(r.component)]) for r in audit.itertuples()]
    audit.to_csv(out/'candidate_audit.csv',index=False)
    geometry_audit(data,frames).to_csv(out/'history_geometry_audit.csv',index=False)
    for split,fs in [('fit',fit),('calibration',cal),('development',dev)]:
        cuts=sum(int((data[t]['supervised']&(data[t]['ycut']==1)).sum()) for t in fs)
        keeps=sum(int((data[t]['supervised']&(data[t]['ycut']==0)).sum()) for t in fs)
        merges=sum(int((data[t]['trusted']&data[t]['merge']).sum()) for t in fs)
        clean=sum(int((data[t]['trusted']&~data[t]['merge']).sum()) for t in fs)
        print(f'[audit {split}] CUT={cuts} KEEP={keeps} merges={merges} clean={clean}',flush=True)
    if args.features_only:
        print(f'Feature audit complete: {out}',flush=True);return 0
    results=[];components=[];edges=[];ranks=[];calibration_rows=[]
    zeros=np.zeros(sum(len(data[t]['edge_ids']) for t in dev))
    for name,oracle in [('frozen_spatial',False),('target_edge_oracle_DIAGNOSTIC',True)]:
        m,c,e,r=evaluate(ctx,data,dev,zeros,2.,name,-1,oracle=oracle)
        results.append(m);components+=c;edges+=e;ranks+=r
    for seed in seeds:
        for variant in variants:
            fitting=dataset_for_frames(data,fit,variant,supervised=True)
            model=train_model(fitting,variant,args,ctx['device'],seed,out)
            calibration_data=dataset_for_frames(data,cal,variant)
            chosen,table=calibrate(data,cal,probabilities(model,calibration_data,ctx['device']),args)
            threshold=chosen['threshold']
            calibration_rows.extend(dict(variant=variant,seed=seed,selected=row['threshold']==threshold,**row) for row in table)
            checkpoint=dict(format='inv50_historical_shape_correspondence_v1',variant=variant,seed=seed,
                model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                spatial_dim=fitting['spatial'].shape[1],event_dim=fitting['event'].shape[1],
                history_lags=LAGS,feature_names=FEATURE_NAMES,threshold=threshold,calibration=chosen,
                manifest=manifest,parameters=sum(p.numel() for p in model.parameters()),steps=args.inv50_steps)
            torch.save(checkpoint,out/f'{variant}_seed{seed}.pt')
            for control in (None,'zero','shuffle') if variant=='correspondence' else (None,):
                ds=dataset_for_frames(data,dev,variant,control=control,seed=seed)
                scores=probabilities(model,ds,ctx['device'])
                name=variant if control is None else variant+'_history_'+control
                m,c,e,r=evaluate(ctx,data,dev,scores,threshold,name,seed)
                results.append(m);components+=c;edges+=e;ranks+=r
            # Save incrementally so completed seeds survive an interruption.
            pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(components).to_csv(out/'component_metrics.csv',index=False)
            pd.DataFrame(edges).to_csv(out/'edge_predictions.csv',index=False)
            pd.DataFrame(ranks).to_csv(out/'history_ranking.csv',index=False)
            pd.DataFrame(calibration_rows).to_csv(out/'calibration.csv',index=False)
            del model,fitting,calibration_data
            if ctx['device'].type=='cuda':torch.cuda.empty_cache()
    atomic_json(out/'summary.json',dict(manifest=manifest,metrics=results,
        conclusion='Compare paired seeds at their fixed calibration limits. Development results do not establish generalization.',
        promotion_criterion='Improved exact repairs and within-component separator ranking without increased clean splits; then repeat on independent episodes/volumes.'))
    print('\nINV50 COMPLETE\n'+pd.DataFrame(results)[['variant','seed','exact','merge_total','clean_split','clean_total','within_component_auc']].to_string(index=False),flush=True)
    print(f'Outputs: {out}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
