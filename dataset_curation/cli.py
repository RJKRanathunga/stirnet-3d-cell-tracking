
from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path

from dataset_curation._repo import repo_root
from dataset_curation.config import DEFAULT_DATASET
from dataset_curation.errors import ArtifactError
from dataset_curation.inference.pipeline import run_current_inference
from dataset_curation.io.atomic import atomic_json, read_json
from dataset_curation.workspace.sample import CurationSample, _now

def _extra(values):
    values = list(values)
    return values[1:] if values and values[0] == "--" else values

def _sample(args):
    return CurationSample.open(
        sample_id=args.sample_id,
        root=args.root,
        dataset=args.dataset,
    )

def _run_module(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("[dataset_curation] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root(), check=True)

def _inference_payload(sample, run_id: str) -> dict:
    path = sample.layout.inference_manifest(run_id)
    return read_json(path) if path.is_file() else {}

def _merge_artifacts(sample, run_id: str, artifacts: dict, *, backend=None) -> None:
    path = sample.layout.inference_manifest(run_id)
    old = read_json(path) if path.is_file() else {}
    merged = dict(old.get("artifacts", {}))
    merged.update(artifacts)
    atomic_json(
        path,
        {
            "schema_version": 1,
            "kind": "inference_run",
            "sample_id": sample.layout.sample_id,
            "run_id": run_id,
            "backend": backend or old.get("backend", "registered_existing_artifacts"),
            "immutable_base_prediction": True,
            "source_zarr": str(sample.source_zarr),
            "command": old.get("command"),
            "artifacts": merged,
            "spacing_zyx_um": old.get("spacing_zyx_um", list(sample.spacing_zyx_um)),
            "created_at": old.get("created_at", _now()),
            "updated_at": _now(),
        },
    )

def _existing_path(value: str | None):
    if value is None:
        return None
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (repo_root() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)

def cmd_setup(args):
    sample = CurationSample.create(
        sample_id=args.sample_id,
        source_zarr=args.source_zarr,
        root=args.root,
        dataset=args.dataset,
        spacing_zyx_um=tuple(args.spacing),
    )
    print(f"sample root : {sample.layout.sample_root}")
    print(f"manifest    : {sample.layout.sample_manifest}")
    print(f"source zarr : {sample.source_zarr}")

def cmd_infer(args):
    sample = _sample(args)
    output = run_current_inference(
        sample,
        run_id=args.run_id,
        extra_args=_extra(args.extra),
    )
    print(f"inference   : {output}")
    print(f"manifest    : {sample.layout.inference_manifest(args.run_id)}")

def cmd_register_spatial(args):
    sample = _sample(args)
    sample.ensure_inference_run(args.run_id)
    source = {
        "instances_root": _existing_path(args.instances_root),
        "supervoxels_root": _existing_path(args.supervoxels_root),
        "stage6_root": _existing_path(args.stage6_root),
        "zarr": _existing_path(args.zarr) if args.zarr else str(sample.source_zarr),
    }
    _merge_artifacts(
        sample,
        args.run_id,
        {"instance_annotation_source": source},
        backend="registered_existing_spatial",
    )
    print("registered instance-annotation source:")
    for key, value in source.items():
        print(f"  {key:18s}: {value}")

def cmd_suspects(args):
    sample = _sample(args)
    manifest = _inference_payload(sample, args.run_id)
    source = manifest.get("artifacts", {}).get("instance_annotation_source")
    if not source:
        raise ArtifactError(
            "No instance-annotation source is registered for this run. "
            "Run `python -m dataset_curation register-spatial ...` first."
        )

    output = sample.layout.inference_run(args.run_id) / "suspects"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "--sample-id", sample.layout.sample_id,
        "--output-dir", str(output),
        "--inv25", source["instances_root"],
        "--inv24", source["supervoxels_root"],
        "--zarr", source["zarr"],
        *_extra(args.extra),
    ]
    _run_module("dataset_curation._compat.merge_suspect_exporter", command)
    _merge_artifacts(sample, args.run_id, {"suspects": str(output)})
    print(f"suspects    : {output}")

def cmd_annotate_instances(args):
    sample = _sample(args)
    sample.ensure_annotation_set(
        args.annotation_set,
        base_inference_run=args.run_id,
    )
    manifest = _inference_payload(sample, args.run_id)
    artifacts = manifest.get("artifacts", {})
    source = artifacts.get("instance_annotation_source")
    if not source:
        raise ArtifactError(
            "The current instance annotator requires the exact current "
            "Inv25/Inv24/Stage-6 input contract. Register it first with "
            "`python -m dataset_curation register-spatial ...`."
        )

    command = [
        "--sample-id", sample.layout.sample_id,
        "--spatial-root", source["instances_root"],
        "--supervoxel-root", source["supervoxels_root"],
        "--stage6-root", source["stage6_root"],
        "--zarr-path", source["zarr"],
        "--output-dir", str(sample.layout.instance_annotations(args.annotation_set)),
    ]
    if artifacts.get("suspects"):
        command += ["--suspect-root", artifacts["suspects"]]
    command += _extra(args.extra)
    _run_module("dataset_curation._compat.instance_annotator", command)

def cmd_annotate_tracks(args):
    sample = _sample(args)
    sample.ensure_annotation_set(
        args.annotation_set,
        base_inference_run=args.run_id,
    )
    source = sample.layout.inference_run(args.run_id)
    required = [
        source / "movies" / "raw.npy",
        source / "movies" / "binary_mask.npy",
        source / "movies" / "final_instances.npy",
        source / "cells_all.csv",
        source / "trackastra" / "napari_graph.json",
        source / "trackastra" / "tracks.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ArtifactError(
            "Track annotation requires an Investigation-36-compatible "
            "inference run. Missing:\\n"
            + "\\n".join(f"  {path}" for path in missing)
        )
    command = [
        "--sample-id", sample.layout.sample_id,
        "--source-root", str(source),
        "--output-dir", str(sample.layout.track_annotations(args.annotation_set)),
        *_extra(args.extra),
    ]
    _run_module("dataset_curation._compat.track_annotator", command)

def cmd_annotate_points(args):
    print(
        "Running the exact historical point annotator. Its current repository "
        "version uses its existing hard-coded sample/timepoint configuration."
    )
    runpy.run_module("dataset_curation._compat.point_annotator", run_name="__main__")

def cmd_status(args):
    sample = _sample(args)
    runs = (
        sorted(path.name for path in sample.layout.inference.iterdir() if path.is_dir())
        if sample.layout.inference.is_dir()
        else []
    )
    sets = (
        sorted(path.name for path in sample.layout.annotations.iterdir() if path.is_dir())
        if sample.layout.annotations.is_dir()
        else []
    )
    print(f"sample          : {sample.layout.sample_root}")
    print(f"source          : {sample.source_zarr}")
    print(f"inference runs  : {runs}")
    print(f"annotation sets : {sets}")

def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m dataset_curation",
        description="Persistent inference-assisted cell dataset curation.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sample-id", required=True)
    common.add_argument("--dataset", default=DEFAULT_DATASET)
    common.add_argument("--root", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", parents=[common])
    p.add_argument("--source-zarr", required=True)
    p.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=(1.625, 0.40625, 0.40625),
        metavar=("Z", "Y", "X"),
    )
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("infer", parents=[common])
    p.add_argument("--run-id", default="current")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("register-spatial", parents=[common])
    p.add_argument("--run-id", default="current")
    p.add_argument("--instances-root", required=True)
    p.add_argument("--supervoxels-root", required=True)
    p.add_argument("--stage6-root", required=True)
    p.add_argument("--zarr", default=None)
    p.set_defaults(func=cmd_register_spatial)

    p = sub.add_parser("suspects", parents=[common])
    p.add_argument("--run-id", default="current")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_suspects)

    p = sub.add_parser("annotate-instances", parents=[common])
    p.add_argument("--run-id", default="current")
    p.add_argument("--annotation-set", default="main")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_annotate_instances)

    p = sub.add_parser("annotate-tracks", parents=[common])
    p.add_argument("--run-id", default="current")
    p.add_argument("--annotation-set", default="main")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_annotate_tracks)

    p = sub.add_parser("annotate-points", parents=[common])
    p.set_defaults(func=cmd_annotate_points)

    p = sub.add_parser("status", parents=[common])
    p.set_defaults(func=cmd_status)
    return parser

def main():
    args = build_parser().parse_args()
    args.func(args)
    return 0
