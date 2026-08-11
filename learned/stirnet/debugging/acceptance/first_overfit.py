"""Real BlastoSPIM all-cell V1 forward/matching/loss acceptance helpers.

This module preserves the data-building API used by the existing first-overfit
notebooks while placing acceptance code under a dedicated subpackage.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from learned.stirnet import RefinementCriterion, StirNet, StirNetConfig
from learned.stirnet.data.graph_builder import AssociationRecord, DetectionRecord, build_temporal_graph
from learned.stirnet.data.sample_builder import robust_normalize
from learned.stirnet.data.targets import build_gt_targets, extract_instance_metadata
from learned.stirnet.training.trainer import model_forward_from_batch, move_to_device


def _repo_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "data").exists() and (path / "learned").exists():
            return path
    raise RuntimeError("Could not locate the repository root")


def _roi_with_all_cells(instance_movie, gt_movie, spacing, margin_um: float = 12.0):
    full_shape = np.asarray(instance_movie.shape[-3:], dtype=int)
    low = full_shape.copy()
    high = np.zeros(3, dtype=int)
    for frame in range(len(instance_movie)):
        foreground = (np.asarray(instance_movie[frame]) > 0) | (np.asarray(gt_movie[frame]) > 0)
        coords = np.where(foreground)
        if len(coords[0]):
            low = np.minimum(low, np.asarray([axis.min() for axis in coords]))
            high = np.maximum(high, np.asarray([axis.max() + 1 for axis in coords]))
    margin = np.ceil(margin_um / spacing).astype(int)
    low = np.maximum(low - margin, 0)
    high = np.minimum(high + margin, full_shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high)), low, high


def _reduced_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.spatial.channels = (4, 8, 16, 32)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.mask_dim = 8
    cfg.temporal.d_model = 32
    cfg.temporal.graph_ffn_dim = 64
    cfg.coreasoning.d_model = 32
    cfg.coreasoning.position_bias_hidden = 8
    cfg.queries.d_model = 32
    cfg.queries.max_queries = None
    cfg.decoder.d_model = 32
    cfg.decoder.ffn_dim = 128
    cfg.decoder.mask_dim = 8
    cfg.decoder.max_spatial_tokens = 2048
    return cfg


def _build_temporal_inputs(
    track_graph,
    instance_movie,
    raw_movie,
    markers_movie,
    roi,
    roi_low,
    target_local_time,
    spacing,
    dref_um,
    current_target,
):
    full_shape = np.asarray(instance_movie.shape[-3:], dtype=np.float32)
    roi_shape = np.asarray(current_target.shape, dtype=np.float32)
    roi_center_um = 0.5 * (roi_shape - 1) * spacing
    node_position_abs_um = {
        int(node_id): np.asarray(data["coords"], dtype=np.float32) * spacing
        for node_id, data in track_graph.nodes(data=True)
    }

    def mean_velocity(node_id: int, neighbours: list[int], forward: bool):
        if not neighbours:
            return np.zeros(3, dtype=np.float32)
        time0 = int(track_graph.nodes[node_id]["time"])
        position0 = node_position_abs_um[node_id]
        values = []
        for other in neighbours:
            other = int(other)
            dt = abs(int(track_graph.nodes[other]["time"]) - time0)
            if dt:
                delta = node_position_abs_um[other] - position0
                values.append((delta if forward else -delta) / dt)
        return np.mean(values, axis=0).astype(np.float32) if values else np.zeros(3, dtype=np.float32)

    records = []
    for local_time in range(len(instance_movie)):
        labels = np.asarray(instance_movie[local_time][roi]).astype(np.int32, copy=False)
        raw_normalized = robust_normalize(np.asarray(raw_movie[local_time][roi]))
        marker = (np.asarray(markers_movie[local_time][roi]) > 0).astype(np.float32)
        metadata = extract_instance_metadata(labels, raw_normalized, tuple(spacing), dref_um, marker)
        ids = metadata.ids.numpy()
        features = metadata.features.numpy()
        id_to_row = {int(instance_id): row for row, instance_id in enumerate(ids)}

        for node_id, node_data in track_graph.nodes(data=True):
            if int(node_data["time"]) != local_time:
                continue
            label_id = int(node_data["label"])
            if label_id not in id_to_row:
                continue
            row = id_to_row[label_id]
            coords_full = np.asarray(node_data["coords"], dtype=np.float32)
            coords_roi = coords_full - roi_low.astype(np.float32)
            position_relative_um = coords_roi * spacing - roi_center_um
            component_voxels = int(np.count_nonzero(labels == label_id))
            feature = features[row]
            lower_full_um = coords_full * spacing
            upper_full_um = (full_shape - 1 - coords_full) * spacing
            lower_roi_um = coords_roi * spacing
            upper_roi_um = (roi_shape - 1 - coords_roi) * spacing
            predecessors = list(track_graph.predecessors(node_id))
            successors = list(track_graph.successors(node_id))
            distance_to_volume_boundary = float(np.min(np.concatenate([lower_full_um, upper_full_um])))
            records.append(
                DetectionRecord(
                    node_id=int(node_id),
                    time_offset=local_time - target_local_time,
                    position_um=tuple(position_relative_um.tolist()),
                    physical_volume_um3=component_voxels * float(np.prod(spacing)),
                    bbox_um=tuple((feature[1:4] * dref_um).tolist()),
                    pca_axes_um=tuple((feature[4:7] * dref_um).tolist()),
                    elongation=float(feature[7]),
                    flatness=float(feature[8]),
                    solidity=float(feature[9]),
                    compactness=float(feature[10]),
                    intensity_mean=float(feature[11]),
                    intensity_std=float(feature[12]),
                    backward_velocity_um=tuple(mean_velocity(int(node_id), predecessors, False).tolist()),
                    forward_velocity_um=tuple(mean_velocity(int(node_id), successors, True).tolist()),
                    distance_to_volume_boundary_um=distance_to_volume_boundary,
                    distance_to_patch_boundary_um=float(np.min(np.concatenate([lower_roi_um, upper_roi_um]))),
                    boundary_related=distance_to_volume_boundary <= 4.0,
                )
            )

    associations = []
    for source, destination, edge_data in track_graph.edges(data=True):
        source = int(source)
        destination = int(destination)
        score = edge_data.get("weight")
        associations.append(
            AssociationRecord(
                src_node_id=source,
                dst_node_id=destination,
                score=None if score is None else float(score),
                relation="division" if track_graph.out_degree(source) > 1 else "temporal",
            )
        )

    return build_temporal_graph(
        records,
        associations,
        dref_um=dref_um,
        temporal_radius=2,
        k_spatial_neighbors=6,
        spatial_radius_dref=2.5,
        current_labels=current_target,
        spacing_um=tuple(spacing),
    )


def build_real_batch(data_dir: Path):
    """Build the current uncropped all-cell first-overfit sample on CPU."""
    data_dir = Path(data_dir)
    trackastra_dir = data_dir / "trackastra"
    source_dir = data_dir / "stirnet_source"
    raw_movie = np.load(data_dir / "raw_movie.npy", mmap_mode="r")
    instance_movie = np.load(data_dir / "instance_movie.npy", mmap_mode="r")
    markers_movie = np.load(data_dir / "markers_movie.npy", mmap_mode="r")
    gt_movie = np.load(data_dir / "gt_movie.npy", mmap_mode="r")
    with (data_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (trackastra_dir / "track_graph.pkl").open("rb") as handle:
        track_graph = pickle.load(handle)

    spacing = np.asarray(metadata["spacing_zyx_um"], dtype=np.float32)
    dref_um = float(np.load(source_dir / "dref_um.npy"))
    target_local_time = 2
    roi, roi_low, roi_high = _roi_with_all_cells(instance_movie, gt_movie, spacing)
    roi_shape = tuple((roi_high - roi_low).tolist())
    current_target = np.asarray(instance_movie[target_local_time][roi]).astype(np.int32, copy=True)
    gt_target = np.asarray(gt_movie[target_local_time][roi]).astype(np.int32, copy=True)
    current_count = int(np.count_nonzero(np.unique(current_target) > 0))
    target_count = int(np.count_nonzero(np.unique(gt_target) > 0))

    spatial_inputs = np.stack(
        [
            np.asarray(np.load(source_dir / "raw_norm_target.npy", mmap_mode="r")[roi]),
            np.asarray(np.load(source_dir / "foreground_target.npy", mmap_mode="r")[roi]),
            np.asarray(np.load(source_dir / "edt_target.npy", mmap_mode="r")[roi]),
            np.asarray(np.load(source_dir / "boundary_target.npy", mmap_mode="r")[roi]),
            np.asarray(np.load(source_dir / "marker_heatmap_target.npy", mmap_mode="r")[roi]),
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    instance_metadata = extract_instance_metadata(current_target, spatial_inputs[0], tuple(spacing), dref_um, spatial_inputs[4])
    temporal = _build_temporal_inputs(
        track_graph, instance_movie, raw_movie, markers_movie,
        roi, roi_low, target_local_time, spacing, dref_um, current_target,
    )
    target = build_gt_targets(gt_target, tuple(spacing), dref_um)
    batch = {
        "spatial_inputs": torch.as_tensor(spatial_inputs).unsqueeze(0),
        "instance_labels": torch.as_tensor(current_target).unsqueeze(0),
        "spacing_um": torch.as_tensor(spacing).unsqueeze(0),
        "dref_um": torch.tensor([dref_um]),
        "targets": [target],
        "instance_features": instance_metadata.features,
        "instance_ids": instance_metadata.ids,
        "instance_batch": torch.zeros(len(instance_metadata.ids), dtype=torch.long),
        "instance_centroids_um": instance_metadata.centroids_um,
        **temporal,
        "temporal_batch": torch.zeros(len(temporal["temporal_ref_um"]), dtype=torch.long),
    }
    required_queries = 2 * len(instance_metadata.ids) + len(temporal["temporal_ref_um"]) + _reduced_config().queries.discovery_queries
    return batch, {
        "roi_shape": roi_shape,
        "current_count": current_count,
        "target_count": target_count,
        "graph_nodes": len(temporal["graph_x"]),
        "temporal_tracklets": len(temporal["temporal_ref_um"]),
        "required_queries": required_queries,
        "instance_geometry_max": float(instance_metadata.features[:, 7:9].max()),
        "graph_geometry_max": float(temporal["graph_x"][:, 11:13].max()),
    }


def run(data_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The real acceptance gate requires CUDA")
    batch, sample = build_real_batch(data_dir)
    cfg = _reduced_config()
    device = torch.device("cuda")
    model = StirNet(cfg).to(device).eval()
    criterion = RefinementCriterion(cfg.losses, cfg.queries, cfg.training).to(device).eval()
    b = {}
    for key, value in batch.items():
        if key == "targets":
            b[key] = value
        elif key == "spatial_inputs":
            b[key] = value.to(device=device, dtype=torch.float16)
        elif key == "instance_labels":
            b[key] = value.to(device=device, dtype=torch.int32)
        else:
            b[key] = move_to_device(value, device)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model_forward_from_batch(model, b)
        losses = criterion(outputs, b["targets"])
    torch.cuda.synchronize()
    print(f"ROI: {sample['roi_shape']}; current={sample['current_count']}; GT={sample['target_count']}")
    print(f"graph nodes={sample['graph_nodes']}; temporal tracklets={sample['temporal_tracklets']}; required queries={sample['required_queries']}")
    print(f"forward_matching_loss_seconds={time.perf_counter() - started:.2f}")
    print(f"peak_cuda_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}")
    for name, value in losses.items():
        print(f"loss/{name}: {float(value):.7f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    default = _repo_root(Path.cwd()) / "data" / "learned" / "stirnet" / "first_overfit" / "BlastoSPIM1_F22_030_034"
    parser.add_argument("--data-dir", type=Path, default=default)
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
