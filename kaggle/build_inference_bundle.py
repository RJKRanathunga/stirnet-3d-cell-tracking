from __future__ import annotations

# PROMOTE_STIRNET_PRODUCTION_INFERENCE_V1

"""Build a reproducible, private Kaggle inference bundle from the repository root.

The bundle is deliberately assembled from the committed Git tree rather than by
copying the working tree.  This prevents an untracked experiment, local data,
credentials, or a half-edited source file from silently entering a leaderboard
submission.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


EXPECTED_REPO_SHA = "ac981bf4fa79f9a5f8db2aa24f11adcd87754482"
BUNDLE_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_CANDIDATES = (
    "runs/stirnet/investigations/19_morphology_rag_v2_headroom_training/recovery/"
    "drosophila_12_morphology_rag_v2_headroom_h100/best_checkpoint.pt",
    "runs/stirnet/investigations/19_morphology_rag_v2_headroom_training/recovery/"
    "drosophila_12_morphology_rag_v2_headroom_h100/checkpoint_step_000600.pt",
)
ARCHIVE_PATHS = (
    "learned",
    "src",
    "pyproject.toml",
    "requirements.txt",
)
FORBIDDEN_REPO_ROOT_NAMES = {"data", "runs", ".stirnet_patch_backup"}
FORBIDDEN_ANYWHERE_NAMES = {".git", ".env", "kaggle.json"}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parent.parent
    if (candidate / ".git").exists() and (candidate / "learned").is_dir():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / ".git").exists() and (cwd / "learned").is_dir():
        return cwd
    raise RuntimeError(
        "Run this script from the cell-tracking repository after placing kaggle/ at its root."
    )


def run_git(root: Path, *args: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def checkpoint_step_from_name(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint_step_") and name.endswith(".pt"):
        token = name[len("checkpoint_step_") : -len(".pt")]
        if token.isdigit():
            return int(token)
    return -1


def resolve_checkpoint(root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        path = supplied.expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_dir():
            candidates = [
                item
                for item in path.glob("checkpoint_step_*.pt")
                if checkpoint_step_from_name(item) >= 0
            ]
            best = path / "best_checkpoint.pt"
            if best.is_file():
                return best.resolve()
            if candidates:
                return max(candidates, key=checkpoint_step_from_name).resolve()
            raise FileNotFoundError(f"No checkpoint found in {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    for relative in DEFAULT_CHECKPOINT_CANDIDATES:
        path = root / relative
        if path.is_file():
            return path.resolve()

    recovery = (
        root
        / "runs"
        / "stirnet"
        / "investigations"
        / "19_morphology_rag_v2_headroom_training"
        / "recovery"
        / "drosophila_12_morphology_rag_v2_headroom_h100"
    )
    if recovery.is_dir():
        candidates = [
            item
            for item in recovery.glob("checkpoint_step_*.pt")
            if checkpoint_step_from_name(item) >= 0
        ]
        if candidates:
            return max(candidates, key=checkpoint_step_from_name).resolve()

    raise FileNotFoundError(
        "Could not auto-discover the spatial checkpoint. Pass --checkpoint explicitly."
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint must contain a dictionary, got {type(value).__name__}")
    return value



def validate_checkpoint_model_compatibility(
    root: Path,
    checkpoint: Path,
) -> None:
    """Strictly load the checkpoint through the production inference API."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from learned.stirnet.inference import (
        SpatialInferenceConfig,
        load_spatial_runtime,
    )

    runtime = load_spatial_runtime(
        checkpoint,
        device=torch.device("cpu"),
        config=SpatialInferenceConfig(),
    )
    morphology_enabled = bool(
        runtime.model_cfg.partition.rag_morphology_enabled
    )
    del runtime

    print(
        "[bundle] strict load     : OK "
        f"(morphology={morphology_enabled}, production-inference-api=True)"
    )


def write_inference_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    payload = load_checkpoint(source)
    required = ("model", "global_step", "model_config")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Checkpoint is missing inference fields: {missing}")

    # Keep model/config/provenance fields only.  Optimizer, scheduler, AMP scaler,
    # RNG and experiment-local TrainingConfig metadata have no role in inference
    # and can substantially increase the private Kaggle dataset size.
    preferred = (
        "architecture",
        "model",
        "global_step",
        "model_config",
        "checkpoint_version",
        "epoch",
        "refinement_stage_step",
    )
    inference_payload = {key: payload[key] for key in preferred if key in payload}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_payload, destination)
    return {
        "global_step": int(payload["global_step"]),
        "source_size_bytes": int(source.stat().st_size),
        "bundle_size_bytes": int(destination.stat().st_size),
        "source_sha256": sha256(source),
        "bundle_sha256": sha256(destination),
        "retained_fields": sorted(inference_payload),
        "removed_fields": sorted(set(payload) - set(inference_payload)),
    }


def assert_core_tree_clean(root: Path, allow_dirty: bool) -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *ARCHIVE_PATHS],
        cwd=root,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("git diff failed")
    if result.returncode == 1 and not allow_dirty:
        raise RuntimeError(
            "Inference-critical files differ from HEAD. Commit/revert them first, "
            "or use --allow-dirty-core only if this is intentional. The bundle "
            "still archives HEAD, not the dirty working-tree versions."
        )


def archive_committed_code(root: Path, destination_repo: Path) -> None:
    destination_repo.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stirnet_git_archive_") as temp_dir:
        tar_path = Path(temp_dir) / "repo.tar"
        with tar_path.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", "HEAD", *ARCHIVE_PATHS],
                cwd=root,
                check=True,
                stdout=handle,
            )
        with tarfile.open(tar_path, "r") as archive:
            archive.extractall(destination_repo)


def copy_kaggle_runtime_files(root: Path, destination_repo: Path) -> None:
    target = destination_repo / "kaggle"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("run_submission.py", "validate_submission.py"):
        source = root / "kaggle" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)


def copy_wheels(wheel_dir: Path | None, bundle_root: Path) -> list[str]:
    if wheel_dir is None:
        return []
    source = wheel_dir.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    wheels = sorted(source.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No .whl files found in {source}")
    target = bundle_root / "wheels"
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in wheels:
        shutil.copy2(path, target / path.name)
        copied.append(path.name)
    return copied


def validate_bundle_contents(bundle_root: Path) -> None:
    for path in bundle_root.rglob("*"):
        relative = path.relative_to(bundle_root)
        parts = relative.parts
        forbidden_anywhere = set(parts) & FORBIDDEN_ANYWHERE_NAMES
        if forbidden_anywhere:
            raise RuntimeError(
                f"Forbidden path entered inference bundle: {path} "
                f"({sorted(forbidden_anywhere)})"
            )
        # The repository's legitimate helper package is
        # investigations/stirnet/data/.  Only top-level repo data/runs are
        # forbidden, not every directory whose basename happens to be 'data'.
        if len(parts) >= 2 and parts[0] == "repo" and parts[1] in FORBIDDEN_REPO_ROOT_NAMES:
            raise RuntimeError(
                f"Forbidden repository-root path entered inference bundle: {path}"
            )
    required = (
        bundle_root / "repo" / "learned" / "stirnet",
        bundle_root / "repo" / "src",
        bundle_root / "repo" / "kaggle" / "run_submission.py",
        bundle_root / "repo" / "kaggle" / "validate_submission.py",
        bundle_root / "assets" / "stirnet_spatial.pt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Bundle is incomplete: " + ", ".join(missing))


def file_manifest(bundle_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        relative = path.relative_to(bundle_root).as_posix()
        result[relative] = {
            "sha256": sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    return result


def write_zip(bundle_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{bundle_root.name}/{path.relative_to(bundle_root).as_posix()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build private Kaggle STIR-Net inference bundle")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Spatial checkpoint file/directory. Default auto-discovers the Investigation-19 morphology-v2 h100 recovery checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/dist/stirnet_kaggle_bundle"),
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="Optional directory of offline wheels to include under bundle/wheels/.",
    )
    parser.add_argument(
        "--allow-repo-sha-mismatch",
        action="store_true",
        help="Permit building from a commit other than the revision this package was generated against.",
    )
    parser.add_argument(
        "--allow-dirty-core",
        action="store_true",
        help="Permit dirty inference-critical files. The bundle still archives committed HEAD versions.",
    )
    parser.add_argument("--no-zip", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = repo_root()
    sha = run_git(root, "rev-parse", "HEAD")
    if sha != EXPECTED_REPO_SHA and not args.allow_repo_sha_mismatch:
        raise RuntimeError(
            "Repository HEAD changed since this Kaggle package was generated.\n"
            f"Expected: {EXPECTED_REPO_SHA}\nObserved: {sha}\n"
            "Re-generate/review the Kaggle package against the new HEAD, or pass "
            "--allow-repo-sha-mismatch only after reviewing compatibility."
        )
    assert_core_tree_clean(root, args.allow_dirty_core)
    checkpoint = resolve_checkpoint(root, args.checkpoint)
    print(f"[bundle] repository HEAD : {sha}")
    print(f"[bundle] checkpoint      : {checkpoint}")
    print("[bundle] preflight       : strict checkpoint/model compatibility")
    validate_checkpoint_model_compatibility(root, checkpoint)

    output = args.output_dir.expanduser()
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    print(f"[bundle] output          : {output}")

    archive_committed_code(root, output / "repo")
    copy_kaggle_runtime_files(root, output / "repo")
    checkpoint_info = write_inference_checkpoint(
        checkpoint,
        output / "assets" / "stirnet_spatial.pt",
    )
    wheels = copy_wheels(args.wheel_dir, output)
    (output / "VERSION.txt").write_text(sha + "\n", encoding="utf-8")
    validate_bundle_contents(output)

    manifest = {
        "bundle_type": "stirnet-kaggle-inference-bundle",
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo_sha": sha,
        "expected_repo_sha_at_package_generation": EXPECTED_REPO_SHA,
        "checkpoint": {
            "bundle_path": "assets/stirnet_spatial.pt",
            "source_name": checkpoint.name,
            **checkpoint_info,
        },
        "offline_wheels": wheels,
        "scientific_scope": {
            "temporal_stirnet": False,
            "spatial_partition": "current production signed multicut",
            "postfilter": "source-instance anchored split-only",
            "tracking": "current Stage 7/8/10/11 stack",
        },
        "files": {},
    }
    # Hash after all bundle files except the manifest itself are final.
    manifest["files"] = file_manifest(output)
    atomic_json(output / "bundle_manifest.json", manifest)

    total_bytes = sum(item["size_bytes"] for item in manifest["files"].values())
    print(f"[bundle] files           : {len(manifest['files'])}")
    print(f"[bundle] unpacked size   : {total_bytes / 2**30:.3f} GiB")
    print(
        "[bundle] checkpoint      : "
        f"{checkpoint_info['source_size_bytes'] / 2**20:.1f} MiB -> "
        f"{checkpoint_info['bundle_size_bytes'] / 2**20:.1f} MiB inference-only"
    )

    if not args.no_zip:
        zip_path = output.parent / f"{output.name}.zip"
        write_zip(output, zip_path)
        print(f"[bundle] zip             : {zip_path}")
        print(f"[bundle] zip size        : {zip_path.stat().st_size / 2**30:.3f} GiB")
    print("[bundle] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
