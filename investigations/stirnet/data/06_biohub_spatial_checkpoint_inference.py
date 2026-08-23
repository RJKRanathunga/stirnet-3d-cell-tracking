from __future__ import annotations

"""
Run STIR-Net checkpoint 600 on one BioHub frame for qualitative inspection.

Default:
    sample    44b6_0113de3b
    timepoint 0
    crop      one automatically selected production-sized crop
    checkpoint checkpoint_step_000600.pt

The script uses the SAME raw-source preprocessing/masking path used by STIR-Net
training, then runs execution_stage="spatial". No dense BioHub segmentation GT
is assumed, so the output is for visual/qualitative inspection rather than an
accuracy benchmark.

Recommended:
    python investigations/stirnet/data/06_biohub_spatial_checkpoint_inference.py --napari
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import zarr


DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)
DEFAULT_CHECKPOINT = (
    "runs/stirnet/training/01_nis3d_spatial_training/recovery/"
    "nis3d_zebrafish_spatial_v1/checkpoint_step_000600.pt"
)


def json_default(value):
    """Convert NumPy/PyTorch/path scalar types for JSON diagnostics."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "learned").is_dir() and (p / "src").is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "learned").is_dir() and (cwd / "src").is_dir():
        return cwd
    raise RuntimeError("Could not find repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_eval04():
    path = ROOT / "investigations/stirnet/data/04_nis3d_geometry_checkpoint_eval.py"
    spec = importlib.util.spec_from_file_location("_eval04_biohub", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_eval04_biohub"] = mod
    spec.loader.exec_module(mod)
    return mod


E04 = load_eval04()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def find_sample(sample_id: str, data_root: str | None) -> Path:
    if data_root:
        root = resolve(data_root)
        candidates = (root, root / "train" / sample_id, root / sample_id)
    else:
        candidates = (
            ROOT / "data/sample/biohub_5samples_20timepoints/train" / sample_id,
            ROOT / "data/sample/biohub_5samples_20timepoints" / sample_id,
        )
    for p in candidates:
        if (p / f"{sample_id}.zarr" / "0").is_dir():
            return p
    raise FileNotFoundError(
        "Could not locate BioHub sample.\n" + "\n".join(f"  - {p}" for p in candidates)
    )


def positive_count(labels: np.ndarray) -> int:
    ids = np.unique(labels)
    return int((ids > 0).sum())


def axis_starts(full: int, crop: int, n: int = 3) -> list[int]:
    if crop >= full:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, full - crop, n)})


def auto_crop(labels: np.ndarray, crop_shape: tuple[int, int, int]):
    """Choose a source-rich crop, preferring many source instances."""
    shape = labels.shape
    crop_shape = tuple(min(a, b) for a, b in zip(crop_shape, shape))
    best = None
    for z0 in axis_starts(shape[0], crop_shape[0]):
        for y0 in axis_starts(shape[1], crop_shape[1]):
            for x0 in axis_starts(shape[2], crop_shape[2]):
                s = (
                    slice(z0, z0 + crop_shape[0]),
                    slice(y0, y0 + crop_shape[1]),
                    slice(x0, x0 + crop_shape[2]),
                )
                local = labels[s]
                count = positive_count(local)
                fg = float((local > 0).mean())
                score = (count, fg)
                if best is None or score > best[0]:
                    best = (score, s)
    if best is None:
        raise RuntimeError("No crop candidates generated.")
    return best[1]


def center_crop(shape, crop_shape):
    out = []
    for full, crop in zip(shape, crop_shape):
        crop = min(int(crop), int(full))
        start = (int(full) - crop) // 2
        out.append(slice(start, start + crop))
    return tuple(out)


def explicit_crop(shape, crop_shape, origin):
    out = []
    for full, crop, start in zip(shape, crop_shape, origin):
        crop = min(int(crop), int(full))
        start = int(start)
        if start < 0 or start + crop > full:
            raise ValueError(f"Explicit crop does not fit: full={full}, crop={crop}, start={start}")
        out.append(slice(start, start + crop))
    return tuple(out)


def expand_crop(crop, full_shape, spacing, halo_um):
    radius = np.ceil(float(halo_um) / np.asarray(spacing)).astype(int)
    halo, core = [], []
    for axis in range(3):
        lo = max(0, crop[axis].start - radius[axis])
        hi = min(full_shape[axis], crop[axis].stop + radius[axis])
        halo.append(slice(lo, hi))
        core.append(slice(crop[axis].start - lo, crop[axis].stop - lo))
    return tuple(halo), tuple(core)


def prepare_source(raw, spacing, sample_id, timepoint, cache_path):
    from learned.stirnet.training.raw_source import prepare_raw_training_batch

    # GT is not used to build the raw-source segmentation or dref.
    dummy_gt = np.zeros(raw.shape, dtype=np.uint8)
    return prepare_raw_training_batch(
        raw,
        dummy_gt,
        spacing,
        source_id=f"BioHub/{sample_id}/t{timepoint:03d}",
        source_cache_path=cache_path,
    )


def build_crop_batch(source_batch, crop, source_halo_um):
    from learned.stirnet.data.sample_builder import build_spatial_channels, normalize_with_percentiles

    raw = torch.as_tensor(source_batch["raw_volume"])[0]
    labels = torch.as_tensor(source_batch["instance_labels"])[0]
    spacing = tuple(float(v) for v in source_batch["spacing_um"][0])
    dref = float(source_batch["dref_um"][0])
    low, high = [float(v) for v in source_batch["raw_normalization_bounds"][0]]

    halo, core = expand_crop(crop, tuple(raw.shape), spacing, source_halo_um)
    raw_halo = raw[halo].cpu().numpy()
    labels_halo = labels[halo].cpu().numpy()

    raw_norm = normalize_with_percentiles(raw_halo, low, high)
    spatial_halo = build_spatial_channels(
        raw_norm,
        labels_halo,
        spacing,
        dref,
        derive_marker=True,
    )
    spatial = np.ascontiguousarray(
        spatial_halo[:, core[0], core[1], core[2]]
    )
    current = np.ascontiguousarray(
        labels_halo[core[0], core[1], core[2]]
    )

    batch = {
        "spatial_inputs": torch.from_numpy(spatial)[None].float(),
        "spacing_um": torch.tensor([spacing], dtype=torch.float32),
        "dref_um": torch.tensor([dref], dtype=torch.float32),
    }
    return batch, spatial, current, halo


def run_model(model, train_cfg, batch, device, alternate_threshold):
    from learned.stirnet.model.partition.partitioner import GraphPartitioner
    from learned.stirnet.training.trainer import model_forward_from_batch, move_batch_to_device

    batch = move_batch_to_device(batch, device)
    amp, amp_name = E04._amp_context(device, train_cfg.amp_dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    with torch.inference_mode():
        with amp:
            output = model_forward_from_batch(
                model,
                batch,
                use_temporal=False,
                execution_stage="spatial",
                apply_existence_filter=False,
            )
            alternate = GraphPartitioner()(
                output.rag,
                output.rag.spatial_edge_logits,
                float(alternate_threshold),
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0
    )
    return output, alternate, amp_name, elapsed, float(peak)


def arr(t, dtype=None):
    """Convert a tensor to NumPy safely, including CUDA BF16 outputs."""
    x = torch.as_tensor(t).detach()

    # NumPy has no native bfloat16 dtype.  The checkpoint runs under BF16
    # autocast on CUDA, so geometry probability/SDF tensors can remain BF16
    # after the forward pass.  Convert those tensors to ordinary float32
    # before moving them into NumPy.
    if x.dtype == torch.bfloat16:
        x = x.float()

    x = x.cpu().numpy()
    return x.astype(dtype) if dtype is not None else x


def save_npz(path, spatial, current, crop, spacing, dref, output, alternate, alt_threshold):
    probs = output.geometry.probabilities()
    rag = output.rag
    np.savez_compressed(
        path,
        normalized_raw=spatial[0].astype(np.float16),
        source_foreground_prior=spatial[1].astype(np.float16),
        source_edt_prior=spatial[2].astype(np.float16),
        source_boundary_prior=spatial[3].astype(np.float16),
        source_marker_prior=spatial[4].astype(np.float16),
        source_labels=current.astype(np.int32),
        pred_foreground=arr(probs["foreground"][0, 0], np.float16),
        pred_surface=arr(probs["surface"][0, 0], np.float16),
        pred_separator=arr(probs["separator"][0, 0], np.float16),
        pred_seed=arr(probs["seed"][0, 0], np.float16),
        pred_sdf=arr(output.geometry.sdf[0, 0], np.float16),
        pred_flow=arr(output.geometry.flow[0], np.float16),
        pred_centroid_offset=arr(output.geometry.centroid_offset[0], np.float16),
        watershed_supervoxels=arr(rag.supervoxel_labels[0], np.int32),
        spatial_partition_default=arr(output.spatial_partition.labels[0], np.int32),
        spatial_partition_alternate=arr(alternate.labels[0], np.int32),
        rag_edge_probability=arr(rag.spatial_edge_logits.sigmoid(), np.float16),
        rag_edge_index=arr(rag.edge_index, np.int32),
        crop_bounds_zyx=np.asarray([[s.start, s.stop] for s in crop], np.int32),
        spacing_zyx_um=np.asarray(spacing, np.float32),
        dref_um=np.asarray([dref], np.float32),
        alternate_threshold=np.asarray([alt_threshold], np.float32),
    )


def open_napari(npz_path, title, default_threshold, alt_threshold):
    import napari

    d = np.load(npz_path, allow_pickle=False)
    scale = tuple(float(v) for v in d["spacing_zyx_um"])
    v = napari.Viewer(title=title, ndisplay=3)

    v.add_image(d["normalized_raw"].astype(np.float32), name="normalized raw", scale=scale)
    v.add_labels(d["source_labels"].astype(np.int32), name="source labels", scale=scale, visible=False)

    v.add_image(d["source_foreground_prior"].astype(np.float32), name="source foreground prior", scale=scale, visible=False)
    v.add_image(d["source_edt_prior"].astype(np.float32), name="source EDT prior", scale=scale, visible=False)
    v.add_image(d["source_boundary_prior"].astype(np.float32), name="source boundary prior", scale=scale, visible=False)
    v.add_image(d["source_marker_prior"].astype(np.float32), name="source marker prior", scale=scale, visible=False)

    v.add_image(d["pred_foreground"].astype(np.float32), name="pred foreground", scale=scale, visible=False, contrast_limits=(0, 1))
    v.add_image(d["pred_surface"].astype(np.float32), name="pred surface", scale=scale, visible=False, contrast_limits=(0, 1))
    v.add_image(d["pred_separator"].astype(np.float32), name="pred separator", scale=scale, visible=False, contrast_limits=(0, 1))
    v.add_image(d["pred_seed"].astype(np.float32), name="pred seed", scale=scale, visible=False, contrast_limits=(0, 1))
    v.add_image(d["pred_sdf"].astype(np.float32), name="pred SDF", scale=scale, visible=False)

    v.add_labels(d["watershed_supervoxels"].astype(np.int32), name="watershed supervoxels", scale=scale, visible=False)
    v.add_labels(
        d["spatial_partition_default"].astype(np.int32),
        name=f"spatial partition threshold={default_threshold:.2f}",
        scale=scale,
    )
    v.add_labels(
        d["spatial_partition_alternate"].astype(np.int32),
        name=f"spatial partition threshold={alt_threshold:.2f}",
        scale=scale,
        visible=False,
    )

    print("\nNapari: use 3-D mode and compare raw -> source -> geometry -> supervoxels -> partitions.")
    napari.run()
    d.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    p.add_argument("--data-root", default=None)
    p.add_argument("--timepoint", type=int, default=0)
    p.add_argument("--spacing", nargs=3, type=float, default=DEFAULT_SPACING_ZYX_UM)
    p.add_argument("--crop-mode", choices=("auto", "center", "explicit"), default="auto")
    p.add_argument("--z0", type=int, default=None)
    p.add_argument("--y0", type=int, default=None)
    p.add_argument("--x0", type=int, default=None)
    p.add_argument("--alternate-threshold", type=float, default=0.90)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--napari", action="store_true")
    args = p.parse_args()

    if not 0 < args.alternate_threshold < 1:
        p.error("--alternate-threshold must be in (0,1)")

    sample_dir = find_sample(args.sample_id, args.data_root)
    zarr_path = sample_dir / f"{args.sample_id}.zarr" / "0"
    volume = zarr.open_array(str(zarr_path), mode="r")
    if volume.ndim != 4:
        raise ValueError(f"Expected [T,Z,Y,X], got {volume.shape}")
    if not 0 <= args.timepoint < volume.shape[0]:
        raise ValueError(f"timepoint {args.timepoint} outside T={volume.shape[0]}")

    raw = np.asarray(volume[args.timepoint])
    spacing = tuple(float(v) for v in args.spacing)

    ckpt_path = resolve(args.checkpoint)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    checkpoint, model, model_cfg, train_cfg = E04._load_checkpoint_model(ckpt_path, device)
    step = int(checkpoint["global_step"])

    crop_shape = tuple(int(v) for v in train_cfg.curriculum.refinement_crop_shape_zyx)
    source_halo_um = float(getattr(train_cfg.curriculum, "refinement_crop_source_halo_um", 8.0))
    default_threshold = float(model_cfg.partition.spatial_merge_threshold)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (
        resolve(args.output_dir)
        if args.output_dir
        else ROOT / "runs/stirnet/evaluation/06_biohub_spatial_checkpoint_inference" /
        f"{args.sample_id}_t{args.timepoint:02d}_step{step}_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    cache = ROOT / "runs/stirnet/evaluation/06_biohub_spatial_checkpoint_inference/cache" / f"{args.sample_id}_t{args.timepoint:02d}.pt"

    print("=" * 110)
    print("STIR-Net Investigation 06 — BioHub spatial checkpoint inference")
    print("=" * 110)
    print("sample           :", args.sample_id)
    print("timepoint        :", args.timepoint)
    print("full Zarr shape  :", tuple(volume.shape))
    print("frame shape      :", raw.shape)
    print("spacing zyx um   :", spacing)
    print("checkpoint       :", ckpt_path)
    print("checkpoint step  :", step)
    print("device           :", device)
    print("production crop  :", crop_shape)
    print("source halo um   :", source_halo_um)
    print("RAG threshold    :", default_threshold)
    print("alternate thresh :", args.alternate_threshold)
    print("=" * 110)

    print("\n[source] production preprocessing + masking ...")
    source = prepare_source(raw, spacing, args.sample_id, args.timepoint, cache)
    current_full = source["instance_labels"][0].cpu().numpy()
    dref = float(source["dref_um"][0])
    meta = source["source_preprocessing_metadata"][0]
    print(
        f"[source] instances={positive_count(current_full)} dref={dref:.4f}um "
        f"cache_hit={meta.get('source_cache_hit', False)}"
    )

    if args.crop_mode == "auto":
        crop = auto_crop(current_full, crop_shape)
    elif args.crop_mode == "center":
        crop = center_crop(raw.shape, crop_shape)
    else:
        if None in (args.z0, args.y0, args.x0):
            raise ValueError("explicit crop requires --z0 --y0 --x0")
        crop = explicit_crop(raw.shape, crop_shape, (args.z0, args.y0, args.x0))

    print("[crop] bounds:", [(s.start, s.stop) for s in crop])

    batch, spatial, current, halo = build_crop_batch(source, crop, source_halo_um)
    print(
        f"[crop] source instances={positive_count(current)} "
        f"foreground_fraction={(current > 0).mean():.4f}"
    )

    print("\n[model] spatial forward ...")
    output, alt_partition, amp_name, seconds, peak_gib = run_model(
        model, train_cfg, batch, device, args.alternate_threshold
    )

    pred_fg = output.geometry.probabilities()["foreground"][0, 0].detach().float().cpu().numpy()
    supervox = output.rag.supervoxel_labels[0].detach().cpu().numpy()
    default_labels = output.spatial_partition.labels[0].detach().cpu().numpy()
    alt_labels = alt_partition.labels[0].detach().cpu().numpy()
    edge_prob = output.rag.spatial_edge_logits.detach().float().sigmoid().cpu()

    npz_path = out / "biohub_spatial_inference.npz"
    save_npz(
        npz_path, spatial, current, crop, spacing, dref,
        output, alt_partition, args.alternate_threshold
    )

    source_fg = current > 0
    pred_binary = pred_fg >= float(model_cfg.partition.foreground_threshold)
    overlap = int((source_fg & pred_binary).sum())
    dice = (
        2 * overlap / (int(source_fg.sum()) + int(pred_binary.sum()))
        if int(source_fg.sum()) + int(pred_binary.sum()) else 1.0
    )

    summary = {
        "sample_id": args.sample_id,
        "timepoint": args.timepoint,
        "checkpoint_step": step,
        "frame_shape_zyx": list(raw.shape),
        "spacing_zyx_um": list(spacing),
        "crop_bounds_zyx": [[s.start, s.stop] for s in crop],
        "source_context_bounds_zyx": [[s.start, s.stop] for s in halo],
        "dref_um": dref,
        "source_full_instance_count": positive_count(current_full),
        "source_crop_instance_count": positive_count(current),
        "source_crop_foreground_fraction": float(source_fg.mean()),
        "pred_foreground_fraction": float(pred_binary.mean()),
        "pred_vs_source_foreground_dice": float(dice),
        "watershed_supervoxel_count": positive_count(supervox),
        "rag_edge_count": int(edge_prob.numel()),
        "rag_edge_probability_mean": float(edge_prob.mean()) if edge_prob.numel() else float("nan"),
        "rag_edge_probability_q90": float(torch.quantile(edge_prob, 0.90)) if edge_prob.numel() else float("nan"),
        "rag_edge_probability_q99": float(torch.quantile(edge_prob, 0.99)) if edge_prob.numel() else float("nan"),
        "default_threshold": default_threshold,
        "default_partition_instance_count": positive_count(default_labels),
        "alternate_threshold": args.alternate_threshold,
        "alternate_partition_instance_count": positive_count(alt_labels),
        "inference_seconds": seconds,
        "peak_allocated_vram_gib": peak_gib,
        "amp_dtype": amp_name,
        "npz_path": str(npz_path),
        "note": "Qualitative inference only; no dense BioHub segmentation GT is used.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("BIOHUB INFERENCE COMPLETE")
    print("=" * 110)
    print("source instances       :", summary["source_crop_instance_count"])
    print("pred/source FG Dice    :", f"{dice:.3f}", "(agreement, not GT accuracy)")
    print("watershed supervoxels  :", summary["watershed_supervoxel_count"])
    print(f"spatial @ {default_threshold:.2f}          :", summary["default_partition_instance_count"])
    print(f"spatial @ {args.alternate_threshold:.2f}          :", summary["alternate_partition_instance_count"])
    print("RAG edges              :", summary["rag_edge_count"])
    print("inference seconds      :", f"{seconds:.2f}")
    if device.type == "cuda":
        print("peak allocated VRAM GiB:", f"{peak_gib:.2f}")
    print("output                  :", out)
    print("=" * 110)

    if args.napari:
        open_napari(
            npz_path,
            f"STIR-Net BioHub — {args.sample_id} t={args.timepoint} step={step}",
            default_threshold,
            args.alternate_threshold,
        )


if __name__ == "__main__":
    main()
