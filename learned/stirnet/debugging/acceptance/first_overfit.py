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
from learned.stirnet.debugging.probes.matching import run_matching_probe
from learned.stirnet.model.query_builder import (
    InstanceQueryBuilder,
    QUERY_DISCOVERY,
    QUERY_PRIMARY,
    QUERY_SPLIT,
    QUERY_TEMPORAL,
)
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
    target = build_gt_targets(
        gt_target, tuple(spacing), dref_um, current_labels=current_target
    )
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
    acceptance_cfg = _reduced_config()
    split_estimator = InstanceQueryBuilder(
        acceptance_cfg.queries,
        feature_channels=acceptance_cfg.spatial.channels[2],
    )
    split_counts = split_estimator.split_companion_counts(
        torch.as_tensor(current_target).unsqueeze(0),
        instance_metadata.ids,
        torch.zeros(len(instance_metadata.ids), dtype=torch.long),
    )
    split_companions_by_source = {
        int(source_id): int(companions)
        for source_id, companions in zip(
            instance_metadata.ids.tolist(), split_counts.tolist()
        )
    }
    required_queries = (
        len(instance_metadata.ids)
        + int(split_counts.sum())
        + len(temporal["temporal_ref_um"])
        + acceptance_cfg.queries.discovery_queries
    )
    return batch, {
        "roi_shape": roi_shape,
        "current_count": current_count,
        "target_count": target_count,
        "graph_nodes": len(temporal["graph_x"]),
        "temporal_tracklets": len(temporal["temporal_ref_um"]),
        "required_queries": required_queries,
        "split_companions_by_source": split_companions_by_source,
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
    print("acceptance: starting fresh-model forward", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model_forward_from_batch(model, b)
    print("acceptance: forward complete; starting matching and streamed losses", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        losses = criterion(outputs, b["targets"])
    print("acceptance: losses complete; validating structured matches", flush=True)
    matching = run_matching_probe(outputs, b["targets"])
    target = b["targets"][0]
    source_rows = {
        int(source_id): row
        for row, source_id in enumerate(target["source_ids"].tolist())
    }
    incompatible_seeded = 0
    one_gt_split_positives = 0
    matched_by_type = {
        QUERY_PRIMARY: 0,
        QUERY_SPLIT: 0,
        QUERY_TEMPORAL: 0,
        QUERY_DISCOVERY: 0,
    }
    temporal_distances_dref = []
    discovery_distances_dref = []
    target_centers = torch.as_tensor(target["centers_cellscale"]).float()
    for pred_index, target_index in zip(
        matching.matches[0].pred_indices.detach().cpu().tolist(),
        matching.matches[0].target_indices.detach().cpu().tolist(),
    ):
        query_type = int(outputs.query_types[0, pred_index].detach().cpu())
        matched_by_type[query_type] += 1
        if query_type == QUERY_TEMPORAL:
            initial = outputs.query_initial_references_cellscale[
                0, pred_index
            ].detach().float().cpu()
            temporal_distances_dref.append(
                float(torch.linalg.vector_norm(initial - target_centers[target_index]))
            )
        if query_type == QUERY_DISCOVERY:
            center = outputs.centers_cellscale[0, pred_index].detach().float().cpu()
            discovery_distances_dref.append(
                float(torch.linalg.vector_norm(center - target_centers[target_index]))
            )
        if query_type not in (QUERY_PRIMARY, QUERY_SPLIT):
            continue
        source_id = int(outputs.source_instance_ids[0, pred_index].detach().cpu())
        source_row = source_rows.get(source_id)
        if source_row is None or int(target["source_gt_overlap"][source_row, target_index]) <= 0:
            incompatible_seeded += 1
        if (
            query_type == QUERY_SPLIT
            and source_row is not None
            and int((target["source_gt_overlap"][source_row] > 0).sum()) == 1
        ):
            one_gt_split_positives += 1
    valid_query_count = int((~outputs.query_padding_mask[0]).sum().detach().cpu())
    matched_count = len(matching.matches[0].pred_indices)
    valid = ~outputs.query_padding_mask[0]
    query_types = outputs.query_types[0, valid].detach().cpu()
    query_sources = outputs.source_instance_ids[0, valid].detach().cpu()
    query_type_counts = {
        "primary": int((query_types == QUERY_PRIMARY).sum()),
        "split": int((query_types == QUERY_SPLIT).sum()),
        "temporal": int((query_types == QUERY_TEMPORAL).sum()),
        "discovery": int((query_types == QUERY_DISCOVERY).sum()),
    }
    split_companions_by_source = {
        int(source_id): int(
            ((query_types == QUERY_SPLIT) & (query_sources == source_id)).sum()
        )
        for source_id in b["instance_ids"].detach().cpu().tolist()
    }
    source_nine_seeded = int(
        (
            ((query_types == QUERY_PRIMARY) | (query_types == QUERY_SPLIT))
            & (query_sources == 9)
        ).sum()
    )
    max_temporal_dref = max(temporal_distances_dref, default=0.0)
    max_discovery_dref = max(discovery_distances_dref, default=0.0)
    output_tensors = (
        outputs.exist_logits,
        outputs.centers_cellscale,
        outputs.coarse_mask_logits,
        outputs.native_mask_embeddings,
        outputs.mask_features,
    )
    assert all(torch.isfinite(value.float()).all() for value in output_tensors)
    assert all(torch.isfinite(value.float()) for value in losses.values())
    assert incompatible_seeded == 0
    assert one_gt_split_positives == 0
    assert max_temporal_dref <= cfg.queries.temporal_match_radius_dref + 1e-6
    assert max_discovery_dref <= cfg.queries.discovery_match_radius_dref + 1e-6
    assert float(losses["overlap"]) == 0.0
    assert valid_query_count == sample["required_queries"]
    assert split_companions_by_source == sample["split_companions_by_source"]
    assert source_nine_seeded >= 9
    assert int(losses["raw_gt_count"]) == sample["target_count"]
    assert int(losses["matched_count"]) == matched_count
    torch.cuda.synchronize()
    print(f"ROI: {sample['roi_shape']}; current={sample['current_count']}; GT={sample['target_count']}")
    print(f"graph nodes={sample['graph_nodes']}; temporal tracklets={sample['temporal_tracklets']}; total_queries={valid_query_count}")
    print(f"query_type_counts={query_type_counts}")
    print(f"split_companions_by_source={split_companions_by_source}")
    print(f"source_9_seeded_hypotheses={source_nine_seeded}")
    print(f"forward_matching_loss_seconds={time.perf_counter() - started:.2f}")
    print(f"peak_cuda_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}")
    print(f"source_incompatible_seeded_matches={incompatible_seeded}")
    print(f"one_gt_split_positive_matches={one_gt_split_positives}")
    print("auxiliary_identity_switching=0 (final assignment reused by criterion)")
    print(
        "matches: "
        f"stage_a={matched_by_type[QUERY_PRIMARY] + matched_by_type[QUERY_SPLIT]}; "
        f"temporal={matched_by_type[QUERY_TEMPORAL]}; "
        f"discovery={matched_by_type[QUERY_DISCOVERY]}; "
        f"unmatched_gt={sample['target_count'] - matched_count}"
    )
    print(
        "eligibility_max: "
        f"temporal={max_temporal_dref:.6f} dref/"
        f"{max_temporal_dref * float(b['dref_um'][0]):.3f} um; "
        f"discovery={max_discovery_dref:.6f} dref/"
        f"{max_discovery_dref * float(b['dref_um'][0]):.3f} um"
    )
    print(
        f"counts: raw_gt={int(losses['raw_gt_count'])}; "
        f"matched_count_target={int(losses['matched_count'])}"
    )
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
