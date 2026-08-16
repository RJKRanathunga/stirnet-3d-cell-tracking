from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learned.stirnet import StirNet, StirNetConfig
from learned.stirnet.data.targets import build_gt_targets
from learned.stirnet.data.trackastra_cache import load_cache
from learned.stirnet.training import TrainingConfig, instance_metrics
from learned.stirnet.training.checkpoint import load_checkpoint, save_checkpoint
from learned.stirnet.training.trainer import (
    Trainer,
    model_forward_from_batch,
    move_batch_to_device,
)


STAGES = (
    "geometry_bootstrap",
    "spatial_partition",
    "instance_temporal",
    "refinement_joint",
)


def _repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "learned").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


def _roi_with_all_cells(instance_movie, gt_movie, spacing, margin_um=12.0):
    shape = np.asarray(instance_movie.shape[-3:], dtype=int)
    low = shape.copy()
    high = np.zeros(3, dtype=int)
    for frame in range(len(instance_movie)):
        coordinates = np.where(
            (np.asarray(instance_movie[frame]) > 0)
            | (np.asarray(gt_movie[frame]) > 0)
        )
        if len(coordinates[0]):
            low = np.minimum(low, [axis.min() for axis in coordinates])
            high = np.maximum(high, [axis.max() + 1 for axis in coordinates])
    margin = np.ceil(margin_um / spacing).astype(int)
    low = np.maximum(low - margin, 0)
    high = np.minimum(high + margin, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def build_real_batch(data_dir: Path) -> tuple[dict, dict]:
    source_dir = data_dir / "stirnet_source"
    instance_movie = np.load(data_dir / "instance_movie.npy", mmap_mode="r")
    gt_movie = np.load(data_dir / "gt_movie.npy", mmap_mode="r")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    spacing = np.asarray(metadata["spacing_zyx_um"], dtype=np.float32)
    dref_um = float(np.load(source_dir / "dref_um.npy"))
    roi = _roi_with_all_cells(instance_movie, gt_movie, spacing)
    target_time = 2
    current = np.asarray(instance_movie[target_time][roi]).astype(np.int64, copy=True)
    gt = np.asarray(gt_movie[target_time][roi]).astype(np.int64, copy=True)
    channel_files = (
        "raw_norm_target.npy",
        "foreground_target.npy",
        "edt_target.npy",
        "boundary_target.npy",
        "marker_heatmap_target.npy",
    )
    spatial = np.stack(
        [np.asarray(np.load(source_dir / name, mmap_mode="r")[roi]) for name in channel_files]
    ).astype(np.float32, copy=False)
    if spatial.shape[0] != 5:
        raise RuntimeError("Prepared sample does not satisfy the V2 five-channel contract")
    temporal_path = data_dir / "temporal_v3" / "temporal_graph.pt"
    if not temporal_path.exists():
        raise FileNotFoundError(
            "Experiment 31 requires the prepared temporal_v3/temporal_graph.pt cache"
        )
    temporal = load_cache(temporal_path)
    temporal["temporal_batch"] = torch.zeros(
        len(temporal["temporal_ref_um"]), dtype=torch.long
    )
    target = build_gt_targets(gt, tuple(spacing), dref_um, current_labels=current)
    batch = {
        "spatial_inputs": torch.from_numpy(spatial).unsqueeze(0),
        "instance_labels": torch.from_numpy(current).unsqueeze(0),
        "spacing_um": torch.from_numpy(spacing).unsqueeze(0),
        "dref_um": torch.tensor([dref_um], dtype=torch.float32),
        "targets": [target],
        **temporal,
    }
    scene = {
        "shape": list(current.shape),
        "current_instance_count": int(np.unique(current[current > 0]).size),
        "gt_instance_count": int(np.unique(gt[gt > 0]).size),
        "temporal_node_count": int(len(temporal["graph_x"])),
        "temporal_tracklet_count": int(len(temporal["temporal_ref_um"])),
    }
    return batch, scene


def reduced_config() -> StirNetConfig:
    cfg = StirNetConfig()
    cfg.evidence.stem_channels = 12
    cfg.evidence.prior_gate_hidden = 16
    cfg.spatial.channels = (12, 24, 48, 96)
    cfg.spatial.blocks_per_level = 1
    cfg.spatial.acquisition_dim = 32
    cfg.geometry.hidden_channels = 32
    cfg.geometry.residual_blocks = 2
    cfg.partition.node_feature_channels = 16
    cfg.partition.rag_hidden_dim = 48
    cfg.partition.rag_layers = 2
    cfg.partition.max_supervoxels = 2048
    cfg.instances.d_model = 64
    cfg.instances.pooled_feature_dim = 16
    cfg.history.hidden_channels = 16
    cfg.temporal.d_model = 64
    cfg.temporal.graph_hidden_dim = 128
    cfg.temporal.cross_heads = 4
    cfg.refinement.hidden_channels = 32
    cfg.refinement.query_channels = 16
    cfg.refinement.max_rois_per_batch = 8
    cfg.validate()
    return cfg


def crop_batch_for_smoke(
    batch: dict, scene: dict, crop_shape: tuple[int, int, int] = (32, 128, 128)
) -> tuple[dict, dict]:
    """Create a deterministic local-validation crop while preserving physical refs."""
    full_shape = torch.tensor(batch["instance_labels"].shape[-3:], dtype=torch.long)
    requested = torch.tensor(crop_shape, dtype=torch.long)
    size = torch.minimum(full_shape, requested)
    gt = torch.as_tensor(batch["targets"][0]["label_map"])
    foreground = torch.nonzero(gt > 0, as_tuple=False)
    center = (
        torch.round(foreground.float().mean(0)).long()
        if foreground.numel()
        else torch.div(full_shape, 2, rounding_mode="floor")
    )
    lower = torch.minimum(
        torch.maximum(center - torch.div(size, 2, rounding_mode="floor"), torch.zeros(3, dtype=torch.long)),
        full_shape - size,
    )
    upper = lower + size
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    cropped = dict(batch)
    cropped["spatial_inputs"] = batch["spatial_inputs"][(slice(None), slice(None), *slices)]
    cropped["instance_labels"] = batch["instance_labels"][(slice(None), *slices)]
    cropped_gt = gt[slices]
    spacing = batch["spacing_um"][0]
    dref = float(batch["dref_um"][0])
    cropped["targets"] = [
        build_gt_targets(
            cropped_gt.numpy(),
            tuple(spacing.tolist()),
            dref,
            current_labels=cropped["instance_labels"][0].numpy(),
        )
    ]
    full_center_voxel = 0.5 * (full_shape.float() - 1)
    crop_center_voxel = lower.float() + 0.5 * (size.float() - 1)
    center_shift_um = (crop_center_voxel - full_center_voxel) * spacing
    cropped["temporal_ref_um"] = batch["temporal_ref_um"] - center_shift_um
    cropped["graph_x"] = batch["graph_x"].clone()
    if cropped["graph_x"].shape[1] >= 4:
        cropped["graph_x"][:, 1:4] -= center_shift_um / dref
    smoke_scene = dict(scene)
    smoke_scene.update(
        {
            "full_shape": scene["shape"],
            "shape": size.tolist(),
            "smoke_crop_lower_zyx": lower.tolist(),
            "current_instance_count": int(
                torch.unique(cropped["instance_labels"][cropped["instance_labels"] > 0]).numel()
            ),
            "gt_instance_count": int(torch.unique(cropped_gt[cropped_gt > 0]).numel()),
        }
    )
    return cropped, smoke_scene


def _tensor(value):
    return value.detach().float().cpu().numpy()


@torch.no_grad()
def collect_diagnostics(trainer: Trainer, batch: dict, scene: dict) -> tuple[dict, object]:
    moved = move_batch_to_device(batch, trainer.device)
    stage = trainer.curriculum_stage
    trainer.model.eval()
    output = model_forward_from_batch(
        trainer.model,
        moved,
        use_temporal=stage.use_temporal,
        run_refinement=stage.run_refinement,
        apply_existence_filter=False,
        return_debug=True,
    )
    predicted = output.final_labels[0].detach().cpu().numpy()
    gt = np.asarray(batch["targets"][0]["label_map"])
    metrics = instance_metrics(predicted, gt)
    request_kinds: dict[str, int] = {}
    if output.refinement is not None:
        for request in output.refinement.requests:
            request_kinds[request.kind] = request_kinds.get(request.kind, 0) + 1
    diagnostic = {
        "stage": stage.name,
        "supervoxel_count": int(output.rag.supervoxel_labels[0].max().item()),
        "spatial_partition_count": int(output.spatial_partition.labels[0].max().item()),
        "final_instance_count": int(output.final_labels[0].max().item()),
        "gt_instance_count": scene["gt_instance_count"],
        "temporal_edge_delta_mean_abs": float(
            output.reasoning.edge_temporal_delta.abs().mean().item()
        )
        if output.reasoning.edge_temporal_delta.numel()
        else 0.0,
        "temporal_gate_mean": float(output.reasoning.edge_temporal_gate.mean().item())
        if output.reasoning.edge_temporal_gate.numel()
        else 0.0,
        "refinement_request_count": 0
        if output.refinement is None
        else len(output.refinement.requests),
        "refinement_applied_count": 0
        if output.refinement is None
        else output.refinement.applied_count,
        "refinement_request_kinds": request_kinds,
        **metrics,
    }
    return diagnostic, output


def save_artifacts(run_dir: Path, batch: dict, output, history, scene, cfg, train_cfg):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {"model": cfg.to_dict(), "training": train_cfg.to_dict()}, indent=2
        ),
        encoding="utf-8",
    )
    geometry = output.geometry
    np.savez_compressed(
        run_dir / "partitions_and_geometry.npz",
        predicted_final_labels=output.final_labels[0].detach().cpu().numpy(),
        spatial_partition=output.spatial_partition.labels[0].detach().cpu().numpy(),
        supervoxels=output.rag.supervoxel_labels[0].detach().cpu().numpy(),
        gt_labels=np.asarray(batch["targets"][0]["label_map"]),
        current_noisy_labels=batch["instance_labels"][0].numpy(),
        centers_um=_tensor(output.centers_um[0]),
        foreground=_tensor(geometry.foreground_logits[0, 0].sigmoid()),
        surface=_tensor(geometry.surface_logits[0, 0].sigmoid()),
        separator=_tensor(geometry.separator_logits[0, 0].sigmoid()),
        sdf=_tensor(geometry.sdf[0, 0]),
        flow=_tensor(geometry.flow[0]),
        centroid_offset=_tensor(geometry.centroid_offset[0]),
        seed=_tensor(geometry.seed_logits[0, 0].sigmoid()),
    )


def parse_args():
    root = _repo_root(Path.cwd())
    default_data = (
        root
        / "data"
        / "learned"
        / "stirnet"
        / "first_overfit"
        / "BlastoSPIM1_F22_030_034"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--run-dir", type=Path, default=root / "runs" / "stirnet_v2_overfit")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--stage-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    parser.add_argument("--hard-time-limit-seconds", type=float, default=3480)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage_steps < 1:
        raise ValueError("stage-steps must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(40266)
    np.random.seed(40266)
    batch, scene = build_real_batch(args.data_dir)
    if args.smoke:
        batch, scene = crop_batch_for_smoke(batch, scene)
    cfg = reduced_config()
    train_cfg = TrainingConfig(lr=args.learning_rate)
    train_cfg.amp_dtype = args.amp_dtype
    steps_per_stage = 1 if args.smoke else args.stage_steps
    if args.stage == "all":
        train_cfg.curriculum.geometry_bootstrap_steps = steps_per_stage
        train_cfg.curriculum.spatial_partition_steps = steps_per_stage
        train_cfg.curriculum.instance_temporal_steps = steps_per_stage
        total_steps = 4 * steps_per_stage
    else:
        train_cfg.curriculum.fixed_stage = args.stage
        total_steps = steps_per_stage
    trainer = Trainer(StirNet(cfg), train_cfg, device=args.device)
    if args.warm_start is not None:
        loaded = load_checkpoint(
            args.warm_start,
            trainer.model,
            optimizer=trainer.optimizer,
            scaler=trainer.scaler,
            map_location=trainer.device,
        )
        trainer.global_step = int(loaded.get("global_step", 0))

    history: list[dict] = []
    started = time.monotonic()
    output = None
    for local_step in range(total_steps):
        if time.monotonic() - started > args.hard_time_limit_seconds - 30:
            break
        metrics = trainer.train_step(batch)
        row = {
            "step": trainer.global_step,
            "stage": trainer.curriculum_stage.name,
            **metrics,
        }
        if local_step % max(args.eval_every, 1) == 0 or local_step + 1 == total_steps:
            diagnostic, output = collect_diagnostics(trainer, batch, scene)
            row.update(diagnostic)
        history.append(row)
        print(json.dumps(row), flush=True)

    diagnostic, output = collect_diagnostics(trainer, batch, scene)
    history.append({"step": trainer.global_step, "final": True, **diagnostic})
    save_artifacts(args.run_dir, batch, output, history, scene, cfg, train_cfg)
    save_checkpoint(
        args.run_dir / "checkpoint_final.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.global_step,
        model_config=cfg,
        training_config=train_cfg,
        extra={"experiment": "31_spatial_first_overfit", "stage": trainer.curriculum_stage.name},
    )
    print(f"Saved V2 experiment artifacts to {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
