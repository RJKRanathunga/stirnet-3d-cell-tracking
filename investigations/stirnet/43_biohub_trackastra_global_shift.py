from __future__ import annotations

r"""
Investigation 43 — BioHub Trackastra global-motion compensation.

Scientific question
-------------------
On the manually curated BioHub volume, does removing the global translational
motion before Trackastra reduce known Trackastra track breaks?

This investigation deliberately uses an ORACLE global shift derived from the
annotation-owned corrected tracks. It does NOT attempt to estimate the shift
without ground-truth/curated correspondences. If the oracle experiment works,
automatic shift estimation can be investigated separately.

Experimental conditions
-----------------------
A. BASELINE
   Original BioHub raw movie + current curated instance masks -> Trackastra.

B. ORACLE GLOBAL SHIFT
   The same raw movie and same curated instance masks are translated frame by
   frame so that the robust median motion of curated one-to-one continuations
   is removed -> the exact same Trackastra model/mode.

Important invariants
--------------------
* annotations/ and preprocessed/ are READ ONLY.
* Only runs/stirnet/investigations/43_biohub_trackastra_global_shift is written.
* Instance IDs are preserved by translation.
* Translation uses a padded canvas. There is NO periodic wrap-around / np.roll.
* Birth/division edges are excluded from global-shift estimation.
* Explicit Continue / Break / Birth annotations are the primary scoring labels.
* Trackastra graph nodes are mapped directly through their `time` and `label`
  attributes; no nearest-centroid rematching is used.
* Investigation 43 defaults to CPU in the current environment because
  Trackastra transformer F.linear fails under the installed cu130 CUDA stack.
* Trackastra association prediction uses batch size 1 by default so the
  experiment is reliable on a 6 GB CUDA GPU. Baseline and shifted conditions
  always use the same batch size.

Default data
------------
data root:
    E:\Data\cell-tracking\Datasets\full_biohub_80\
    biohub-cell-tracking-during-development

sample:
    train/44b6_0113de3b

Typical command
---------------
From the repository root:

    python .\investigations\stirnet\43_biohub_trackastra_global_shift.py

Audit the annotations and oracle shifts without running Trackastra:

    python .\investigations\stirnet\43_biohub_trackastra_global_shift.py --audit-only

Re-run and replace previous Investigation-43 outputs:

    python .\investigations\stirnet\43_biohub_trackastra_global_shift.py --force

Keep the large temporary shifted raw/label movies for inspection:

    python .\investigations\stirnet\43_biohub_trackastra_global_shift.py --keep-shifted-inputs
"""

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import pickle
import shutil
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Repository bootstrap
# =============================================================================


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.parent, *here.parents, Path.cwd().resolve())
    for candidate in candidates:
        if (
            (candidate / "dataset_curation").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_curation.catalog import BioHubCatalog
from dataset_curation.config import DEFAULT_SPACING_ZYX_UM
from dataset_curation.annotation.tracks.graph import _canonical_edge
from src.io import open_sample


SCRIPT_NAME = "43_biohub_trackastra_global_shift"
DEFAULT_DATA_ROOT = Path(
    r"E:\Data\cell-tracking\Datasets\full_biohub_80"
    r"\biohub-cell-tracking-during-development"
)
DEFAULT_SAMPLE = "44b6_0113de3b"
DEFAULT_SPLIT = "train"
DEFAULT_ANNOTATION_SET = "main"
DEFAULT_MODEL = "ctc"
DEFAULT_MODE = "greedy"
DEFAULT_DEVICE = "cpu"
DEFAULT_TRACKASTRA_BATCH_SIZE = 1

Node = tuple[int, int]  # (frame, spatial_instance_id)
Edge = tuple[Node, Node]


# =============================================================================
# Generic helpers
# =============================================================================


def banner(title: str, width: int = 104) -> None:
    print("=" * width, flush=True)
    print(title, flush=True)
    print("=" * width, flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def digest_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_node(value: Any) -> Node:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid serialized node: {value!r}")
    return int(value[0]), int(value[1])


def parse_edge(value: Any) -> Edge:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid serialized edge: {value!r}")
    return _canonical_edge(parse_node(value[0]), parse_node(value[1]))


def edge_json(edge: Edge) -> list[list[int]]:
    return [
        [int(edge[0][0]), int(edge[0][1])],
        [int(edge[1][0]), int(edge[1][1])],
    ]


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def format_pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def gibibytes(nbytes: int | float) -> float:
    return float(nbytes) / float(1024**3)


def close_memmap(array: np.ndarray) -> None:
    try:
        array.flush()
    except Exception:
        pass
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        try:
            mmap.close()
        except Exception:
            pass


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# =============================================================================
# Paths / provenance
# =============================================================================


@dataclass(frozen=True)
class Paths:
    data_root: Path
    sample: str
    split: str
    annotation_set: str
    frame_count: int

    zarr: Path
    base_instances: Path
    preprocessed_root: Path
    inference_manifest: Path

    annotation_root: Path
    annotation_manifest: Path
    instance_annotations: Path
    track_annotations: Path
    spatial_operations: Path
    track_state: Path
    current_tracks: Path

    output: Path
    audit_json: Path
    shifts_csv: Path
    shift_summary_json: Path
    comparison_json: Path
    break_comparison_csv: Path

    baseline_dir: Path
    shifted_dir: Path
    temp_dir: Path

    @property
    def shifted_raw(self) -> Path:
        return self.temp_dir / "oracle_shifted_raw.npy"

    @property
    def shifted_labels(self) -> Path:
        return self.temp_dir / "oracle_shifted_instances.npy"

    def manual_frame(self, t: int) -> Path:
        return self.instance_annotations / f"manual_instances_t{int(t):03d}.npy"


def make_paths(args: argparse.Namespace) -> Paths:
    data_root = Path(args.data_root).expanduser().resolve()
    catalog = BioHubCatalog(data_root)
    record = catalog.get(str(args.sample_id), split=str(args.split))

    if record.frame_count is None:
        raise RuntimeError(
            f"Could not determine frame count for {record.split}/{record.volume_id}"
        )
    frame_count = int(record.frame_count)

    if not record.paths.spatial_complete(frame_count=frame_count):
        raise RuntimeError(
            "Canonical spatial inference is incomplete. Investigation 43 needs "
            f"{record.paths.final_instances}"
        )

    annotation_root = record.paths.annotation_set(str(args.annotation_set))
    annotation_manifest = record.paths.annotation_manifest(str(args.annotation_set))
    instance_annotations = record.paths.instance_annotations(str(args.annotation_set))
    track_annotations = record.paths.track_annotations(str(args.annotation_set))
    spatial_operations = instance_annotations / "spatial_operations.json"
    track_state = track_annotations / "track_annotations.json"
    current_tracks = track_annotations / "current_tracks.csv"

    required = (
        annotation_manifest,
        spatial_operations,
        track_state,
        current_tracks,
        record.paths.final_instances,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Investigation 43 is missing required curated artifacts:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    ann_manifest = read_json(annotation_manifest)
    current_inference_id = record.paths.inference_id()
    annotated_inference_id = str(
        ann_manifest.get("base_inference_id", "")
    ).strip() or None
    if current_inference_id is not None and annotated_inference_id != current_inference_id:
        raise RuntimeError(
            "Annotation/inference binding mismatch.\n"
            f"annotation base_inference_id = {annotated_inference_id!r}\n"
            f"current inference_id        = {current_inference_id!r}"
        )

    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else (
            ROOT
            / "runs"
            / "stirnet"
            / "investigations"
            / SCRIPT_NAME
            / record.volume_id
        ).resolve()
    )

    return Paths(
        data_root=data_root,
        sample=record.volume_id,
        split=record.split,
        annotation_set=str(args.annotation_set),
        frame_count=frame_count,
        zarr=record.paths.zarr,
        base_instances=record.paths.final_instances,
        preprocessed_root=record.paths.preprocessed_root,
        inference_manifest=record.paths.inference_manifest,
        annotation_root=annotation_root,
        annotation_manifest=annotation_manifest,
        instance_annotations=instance_annotations,
        track_annotations=track_annotations,
        spatial_operations=spatial_operations,
        track_state=track_state,
        current_tracks=current_tracks,
        output=output,
        audit_json=output / "annotation_audit.json",
        shifts_csv=output / "oracle_shifts.csv",
        shift_summary_json=output / "oracle_shift_summary.json",
        comparison_json=output / "comparison.json",
        break_comparison_csv=output / "break_comparison.csv",
        baseline_dir=output / "baseline",
        shifted_dir=output / "oracle_shift",
        temp_dir=output / "_temporary_inputs",
    )


def provenance_signature(paths: Paths) -> dict[str, Any]:
    inference_id = None
    if paths.inference_manifest.is_file():
        inference_id = read_json(paths.inference_manifest).get("inference_id")

    return {
        "script": SCRIPT_NAME,
        "sample": paths.sample,
        "split": paths.split,
        "annotation_set": paths.annotation_set,
        "frame_count": paths.frame_count,
        "inference_id": inference_id,
        "annotation_manifest_sha256": sha256(paths.annotation_manifest),
        "spatial_operations_sha256": sha256(paths.spatial_operations),
        "track_annotations_sha256": sha256(paths.track_state),
        "current_tracks_sha256": sha256(paths.current_tracks),
    }


# =============================================================================
# Curated instance movie
# =============================================================================


def manual_frame_ids(paths: Paths) -> tuple[int, ...]:
    result: list[int] = []
    for t in range(paths.frame_count):
        if paths.manual_frame(t).is_file():
            result.append(t)
    return tuple(result)


def load_curated_frame(
    paths: Paths,
    t: int,
    *,
    base_handle: np.ndarray | None = None,
) -> np.ndarray:
    t = int(t)
    manual = paths.manual_frame(t)
    if manual.is_file():
        frame = np.load(manual, allow_pickle=False)
    else:
        if base_handle is None:
            base_handle = np.load(
                paths.base_instances,
                mmap_mode="r",
                allow_pickle=False,
            )
        frame = np.asarray(base_handle[t])

    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Curated frame t={t} must be 3-D, got {frame.shape}")
    if frame.size:
        minimum = int(frame.min())
        maximum = int(frame.max())
    else:
        minimum = maximum = 0
    if minimum < 0 or maximum > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Curated frame t={t} has label range [{minimum}, {maximum}], "
            "which cannot be represented losslessly as uint16."
        )
    return np.asarray(frame, dtype=np.uint16)


def build_curated_label_dask(paths: Paths):
    """
    Build a lazy (T,Z,Y,X) Dask array without duplicating the full curated movie.

    Frames with manual_instances_tXXX.npy use the saved curated frame. All other
    frames lazily read the canonical final_instances frame.
    """
    import dask.array as da
    from dask import delayed

    base = np.load(
        paths.base_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    if base.ndim != 4 or base.dtype != np.dtype(np.uint16):
        raise ValueError(
            f"Expected canonical uint16 (T,Z,Y,X) instances, got "
            f"{base.shape} {base.dtype}"
        )
    if int(base.shape[0]) != paths.frame_count:
        raise ValueError(
            f"Frame-count mismatch: metadata={paths.frame_count}, labels={base.shape[0]}"
        )

    spatial_shape = tuple(int(v) for v in base.shape[1:])

    def load_one(t: int) -> np.ndarray:
        return load_curated_frame(paths, int(t))

    lazy_frames = [
        da.from_delayed(
            delayed(load_one)(t),
            shape=spatial_shape,
            dtype=np.uint16,
        )
        for t in range(paths.frame_count)
    ]
    return da.stack(lazy_frames, axis=0), spatial_shape


# =============================================================================
# Annotation truth / scoring labels
# =============================================================================


@dataclass(frozen=True)
class AnnotationTruth:
    forced_nonbirth: frozenset[Edge]
    manual_continue: frozenset[Edge]
    auto_repair: frozenset[Edge]
    unclassified_forced: frozenset[Edge]
    broken: frozenset[Edge]
    birth: frozenset[Edge]
    ignored_nodes: frozenset[Node]
    primary_continue: frozenset[Edge]
    primary_continue_source: str


def load_annotation_truth(paths: Paths) -> AnnotationTruth:
    payload = read_json(paths.track_state)

    forced = {
        parse_edge(raw)
        for raw in payload.get("forced_edges", [])
    }
    broken = {
        parse_edge(raw)
        for raw in payload.get("broken_edges", [])
    }

    birth: set[Edge] = set()
    for event in payload.get("birth_events", []):
        if not isinstance(event, dict):
            continue
        parent = parse_node(event["parent"])
        for raw_daughter in event.get("daughters", []):
            birth.add(_canonical_edge(parent, parse_node(raw_daughter)))

    ignored_nodes: set[Node] = set()
    for event in payload.get("ignored_events", []):
        if not isinstance(event, dict):
            continue
        for key in ("endpoint", "selected_node"):
            if key in event:
                try:
                    ignored_nodes.add(parse_node(event[key]))
                except Exception:
                    pass

    manual_history: set[Edge] = set()
    auto_history: set[Edge] = set()
    for operation in payload.get("history", []):
        if not isinstance(operation, dict):
            continue
        op_type = str(operation.get("type", ""))
        if op_type == "connect" and "edge" in operation:
            manual_history.add(parse_edge(operation["edge"]))
        elif op_type == "auto_repair":
            for state in operation.get("edge_states", []):
                if isinstance(state, dict) and "edge" in state:
                    auto_history.add(parse_edge(state["edge"]))

    forced_nonbirth = forced - birth

    # History contains active operations because undo() removes the undone
    # operation. Intersect with current forced state as an additional guard.
    manual_continue = manual_history & forced_nonbirth
    auto_repair = auto_history & forced_nonbirth
    unclassified = forced_nonbirth - manual_continue - auto_repair

    # Prefer explicitly manual Continue labels. Older annotation states may
    # contain forced edges without sufficiently detailed history; in that case
    # fall back to all active forced non-birth edges and say so explicitly.
    if manual_continue:
        primary = manual_continue
        source = "explicit_manual_continue"
    else:
        primary = forced_nonbirth
        source = "forced_nonbirth_fallback"

    return AnnotationTruth(
        forced_nonbirth=frozenset(forced_nonbirth),
        manual_continue=frozenset(manual_continue),
        auto_repair=frozenset(auto_repair),
        unclassified_forced=frozenset(unclassified),
        broken=frozenset(broken),
        birth=frozenset(birth),
        ignored_nodes=frozenset(ignored_nodes),
        primary_continue=frozenset(primary),
        primary_continue_source=source,
    )


# =============================================================================
# Oracle motion from annotation-owned corrected tracklets
# =============================================================================


@dataclass(frozen=True)
class ShiftResult:
    pairwise_float_zyx: np.ndarray   # (T-1,3), observed global displacement
    cumulative_float_zyx: np.ndarray # (T,3), cumulative observed displacement
    align_int_zyx: np.ndarray        # (T,3), translation applied to each frame
    placement_zyx: np.ndarray        # (T,3), nonnegative placement in padded canvas
    canvas_shape_zyx: tuple[int, int, int]
    rows: pd.DataFrame


def _robust_median_displacement(
    vectors_zyx: np.ndarray,
    *,
    spacing_zyx: Sequence[float],
    minimum_pairs: int,
    mad_scale: float,
    minimum_residual_gate_um: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    vectors = np.asarray(vectors_zyx, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Expected displacement array (N,3), got {vectors.shape}")
    if int(vectors.shape[0]) < int(minimum_pairs):
        raise RuntimeError(
            f"Only {vectors.shape[0]} curated continuation pairs are available; "
            f"need at least {minimum_pairs}."
        )

    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    center0 = np.median(vectors, axis=0)
    residual_um = np.linalg.norm((vectors - center0) * spacing[None, :], axis=1)

    residual_median = float(np.median(residual_um))
    mad = float(np.median(np.abs(residual_um - residual_median)))
    robust_sigma = 1.4826 * mad
    gate = max(
        float(minimum_residual_gate_um),
        residual_median + float(mad_scale) * robust_sigma,
    )
    keep = residual_um <= gate

    if int(np.count_nonzero(keep)) < int(minimum_pairs):
        # Keep the closest minimum_pairs rather than turning a frame pair into
        # a silent zero-shift fallback.
        order = np.argsort(residual_um)
        keep = np.zeros(len(vectors), dtype=bool)
        keep[order[: int(minimum_pairs)]] = True

    center = np.median(vectors[keep], axis=0)
    final_residual_um = np.linalg.norm(
        (vectors[keep] - center) * spacing[None, :],
        axis=1,
    )

    stats: dict[str, float | int] = {
        "pairs_total": int(len(vectors)),
        "pairs_inlier": int(np.count_nonzero(keep)),
        "gate_um": float(gate),
        "median_residual_um": (
            float(np.median(final_residual_um))
            if len(final_residual_um)
            else math.nan
        ),
        "p90_residual_um": (
            float(np.percentile(final_residual_um, 90))
            if len(final_residual_um)
            else math.nan
        ),
    }
    return center, keep, stats


def estimate_oracle_shifts(
    paths: Paths,
    truth: AnnotationTruth,
    spatial_shape_zyx: Sequence[int],
    *,
    spacing_zyx: Sequence[float],
    minimum_pairs: int,
    mad_scale: float,
    minimum_residual_gate_um: float,
) -> ShiftResult:
    tracks = pd.read_csv(paths.current_tracks)
    required = {"track_id", "frame", "cell_id", "z", "y", "x"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise ValueError(
            f"{paths.current_tracks} is missing columns: {missing}"
        )

    tracks = tracks.loc[:, ["track_id", "frame", "cell_id", "z", "y", "x"]].copy()
    tracks["track_id"] = tracks["track_id"].astype(np.int64)
    tracks["frame"] = tracks["frame"].astype(np.int64)
    tracks["cell_id"] = tracks["cell_id"].astype(np.int64)

    duplicate = tracks.duplicated(["track_id", "frame"], keep=False)
    if bool(duplicate.any()):
        example = tracks.loc[duplicate].head(10)
        raise RuntimeError(
            "current_tracks.csv contains more than one detection for the same "
            "track_id/frame, which is invalid for oracle translation estimation.\n"
            f"{example}"
        )

    vectors_by_frame: dict[int, list[np.ndarray]] = {
        t: [] for t in range(paths.frame_count - 1)
    }

    for _track_id, group in tracks.groupby("track_id", sort=False):
        ordered = group.sort_values("frame", kind="stable")
        rows = list(ordered.itertuples(index=False))
        for left, right in zip(rows[:-1], rows[1:]):
            t0 = int(left.frame)
            t1 = int(right.frame)
            if t1 != t0 + 1:
                continue

            left_node = (t0, int(left.cell_id))
            right_node = (t1, int(right.cell_id))

            # Do not use intentionally deferred/ignored endpoint evidence in the
            # oracle shift.
            if (
                left_node in truth.ignored_nodes
                or right_node in truth.ignored_nodes
            ):
                continue

            left_xyz = np.asarray(
                [left.z, left.y, left.x],
                dtype=np.float64,
            )
            right_xyz = np.asarray(
                [right.z, right.y, right.x],
                dtype=np.float64,
            )
            if not (
                np.all(np.isfinite(left_xyz))
                and np.all(np.isfinite(right_xyz))
            ):
                continue
            vectors_by_frame[t0].append(right_xyz - left_xyz)

    pairwise = np.zeros((paths.frame_count - 1, 3), dtype=np.float64)
    records: list[dict[str, Any]] = []

    for t in range(paths.frame_count - 1):
        vectors = np.asarray(vectors_by_frame[t], dtype=np.float64)
        if vectors.size == 0:
            vectors = np.empty((0, 3), dtype=np.float64)

        try:
            center, _keep, stats = _robust_median_displacement(
                vectors,
                spacing_zyx=spacing_zyx,
                minimum_pairs=minimum_pairs,
                mad_scale=mad_scale,
                minimum_residual_gate_um=minimum_residual_gate_um,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not estimate oracle global shift for t={t:03d}->{t+1:03d}: "
                f"{exc}"
            ) from exc

        pairwise[t] = center
        records.append(
            {
                "frame_from": int(t),
                "frame_to": int(t + 1),
                "pair_dz": float(center[0]),
                "pair_dy": float(center[1]),
                "pair_dx": float(center[2]),
                **stats,
            }
        )

    cumulative = np.zeros((paths.frame_count, 3), dtype=np.float64)
    for t in range(1, paths.frame_count):
        cumulative[t] = cumulative[t - 1] + pairwise[t - 1]

    # To cancel observed motion, translate by the negative cumulative motion.
    align_int = -np.rint(cumulative).astype(np.int64)

    minimum = align_int.min(axis=0)
    maximum = align_int.max(axis=0)
    offset = -minimum
    placement = align_int + offset[None, :]

    spatial_shape = np.asarray(spatial_shape_zyx, dtype=np.int64)
    canvas_shape = tuple(
        int(v)
        for v in (
            spatial_shape
            + (maximum - minimum)
        ).tolist()
    )

    for t in range(paths.frame_count - 1):
        records[t].update(
            {
                "cumulative_dz_at_to": float(cumulative[t + 1, 0]),
                "cumulative_dy_at_to": float(cumulative[t + 1, 1]),
                "cumulative_dx_at_to": float(cumulative[t + 1, 2]),
                "align_z_at_to": int(align_int[t + 1, 0]),
                "align_y_at_to": int(align_int[t + 1, 1]),
                "align_x_at_to": int(align_int[t + 1, 2]),
            }
        )

    return ShiftResult(
        pairwise_float_zyx=pairwise,
        cumulative_float_zyx=cumulative,
        align_int_zyx=align_int,
        placement_zyx=placement,
        canvas_shape_zyx=canvas_shape,
        rows=pd.DataFrame(records),
    )


# =============================================================================
# Shifted input materialization
# =============================================================================


@dataclass(frozen=True)
class ShiftedInputHandles:
    raw: np.ndarray
    labels: np.ndarray
    raw_path: Path
    labels_path: Path


def materialize_shifted_inputs(
    paths: Paths,
    shift: ShiftResult,
    *,
    max_temp_gib: float,
) -> ShiftedInputHandles:
    source = open_sample(paths.zarr)
    base = np.load(
        paths.base_instances,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_shape = (
        paths.frame_count,
        *tuple(int(v) for v in base.shape[1:]),
    )
    if tuple(int(v) for v in source.shape) != expected_shape:
        raise ValueError(
            f"Raw/instance shape mismatch: raw={source.shape}, labels={expected_shape}"
        )

    full_shape = (
        paths.frame_count,
        *shift.canvas_shape_zyx,
    )
    raw_bytes = int(np.prod(full_shape, dtype=np.int64)) * np.dtype(source.dtype).itemsize
    label_bytes = int(np.prod(full_shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
    total_gib = gibibytes(raw_bytes + label_bytes)

    print(
        f"[shift] padded canvas ZYX : {shift.canvas_shape_zyx}",
        flush=True,
    )
    print(
        f"[shift] temporary storage : {total_gib:.2f} GiB "
        f"(raw={gibibytes(raw_bytes):.2f}, labels={gibibytes(label_bytes):.2f})",
        flush=True,
    )

    if total_gib > float(max_temp_gib):
        raise RuntimeError(
            f"Shifted temporary inputs need ~{total_gib:.2f} GiB, exceeding "
            f"--max-temp-gib={float(max_temp_gib):.2f}. Increase the limit only "
            "after checking the padded canvas is reasonable."
        )

    paths.temp_dir.mkdir(parents=True, exist_ok=True)
    paths.shifted_raw.unlink(missing_ok=True)
    paths.shifted_labels.unlink(missing_ok=True)

    raw_mm = np.lib.format.open_memmap(
        paths.shifted_raw,
        mode="w+",
        dtype=np.dtype(source.dtype),
        shape=full_shape,
    )
    label_mm = np.lib.format.open_memmap(
        paths.shifted_labels,
        mode="w+",
        dtype=np.uint16,
        shape=full_shape,
    )

    spatial_shape = tuple(int(v) for v in base.shape[1:])

    for t in range(paths.frame_count):
        raw_mm[t].fill(0)
        label_mm[t].fill(0)

        start = shift.placement_zyx[t]
        destination = tuple(
            slice(
                int(start[d]),
                int(start[d]) + int(spatial_shape[d]),
            )
            for d in range(3)
        )

        raw_frame = np.asarray(source[t])
        label_frame = load_curated_frame(
            paths,
            t,
            base_handle=base,
        )
        raw_mm[(t, *destination)] = raw_frame
        label_mm[(t, *destination)] = label_frame

        if (
            t == 0
            or t == paths.frame_count - 1
            or (t + 1) % 10 == 0
        ):
            print(
                f"[shift] wrote {t + 1:3d}/{paths.frame_count} "
                f"placement={tuple(int(v) for v in start)}",
                flush=True,
            )

    raw_mm.flush()
    label_mm.flush()

    del source, base
    gc.collect()

    return ShiftedInputHandles(
        raw=raw_mm,
        labels=label_mm,
        raw_path=paths.shifted_raw,
        labels_path=paths.shifted_labels,
    )


# =============================================================================
# Trackastra
# =============================================================================


@dataclass(frozen=True)
class TrackastraResult:
    graph: Any
    predicted_edges: frozenset[Edge]
    seconds: float
    node_count: int
    edge_count: int


def graph_detection_edges(graph) -> set[Edge]:
    result: set[Edge] = set()
    for source_id, target_id in graph.edges:
        source = graph.nodes[source_id]
        target = graph.nodes[target_id]

        t0 = int(source["time"])
        t1 = int(target["time"])
        label0 = int(source["label"])
        label1 = int(target["label"])

        if t0 == t1:
            raise RuntimeError(
                "Trackastra returned a same-frame tracking edge: "
                f"(t={t0}, label={label0}) -> (t={t1}, label={label1})"
            )
        result.add(
            _canonical_edge(
                (t0, label0),
                (t1, label1),
            )
        )
    return result


def graph_edge_table(graph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_id, target_id, attributes in graph.edges(data=True):
        source = graph.nodes[source_id]
        target = graph.nodes[target_id]
        rows.append(
            {
                "source_frame": int(source["time"]),
                "source_label": int(source["label"]),
                "target_frame": int(target["time"]),
                "target_label": int(target["label"]),
                "weight": (
                    float(attributes["weight"])
                    if "weight" in attributes
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "source_frame",
            "source_label",
            "target_frame",
            "target_label",
            "weight",
        ],
    )


def graph_node_table(graph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node_id, attributes in graph.nodes(data=True):
        coords = np.asarray(attributes["coords"], dtype=np.float64)
        if coords.shape != (3,):
            raise RuntimeError(
                f"Expected 3-D Trackastra coordinates, got {coords.shape}"
            )
        rows.append(
            {
                "graph_node": int(node_id),
                "frame": int(attributes["time"]),
                "cell_id": int(attributes["label"]),
                "z": float(coords[0]),
                "y": float(coords[1]),
                "x": float(coords[2]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["graph_node", "frame", "cell_id", "z", "y", "x"],
    )


def build_baseline_inputs(paths: Paths):
    import dask.array as da

    source = open_sample(paths.zarr)
    source_chunks = getattr(source, "chunks", None)
    if source_chunks is None:
        source_chunks = (
            1,
            *tuple(int(v) for v in source.shape[1:]),
        )

    raw = da.from_array(
        source,
        chunks=source_chunks,
        asarray=False,
        fancy=False,
    )
    labels, spatial_shape = build_curated_label_dask(paths)

    if tuple(raw.shape) != tuple(labels.shape):
        raise ValueError(
            f"Baseline raw/curated-label shape mismatch: {raw.shape} vs {labels.shape}"
        )
    return source, raw, labels, spatial_shape


def build_shifted_dask(handles: ShiftedInputHandles):
    import dask.array as da

    chunks = (
        1,
        *tuple(int(v) for v in handles.raw.shape[1:]),
    )
    raw = da.from_array(
        handles.raw,
        chunks=chunks,
        asarray=False,
        fancy=False,
    )
    labels = da.from_array(
        handles.labels,
        chunks=chunks,
        asarray=False,
        fancy=False,
    )
    return raw, labels


def _trackastra_equivalent_greedy(
    candidate_graph,
    *,
    allow_divisions: bool,
    threshold: float = 0.5,
    edge_attr: str = "weight",
):
    """
    Exact decision rule used by Trackastra's track_greedy(), implemented with
    lightweight Python sets/counters instead of mutating NetworkX per candidate.

    Trackastra's rule is:
      1. sort edges globally by descending score;
      2. ignore all scores below 0.5;
      3. each target may have at most one incoming edge;
      4. each source may have at most two outgoing edges when divisions are
         enabled, otherwise one.

    We construct the NetworkX solution graph only after the decisions are made.
    This avoids the observed Trackastra/NetworkX greedy stall while preserving
    the same selected-edge semantics.
    """
    import networkx as nx

    started = time.perf_counter()

    edges = sorted(
        candidate_graph.edges(data=True),
        key=lambda edge: edge[2][edge_attr],
        reverse=True,
    )

    max_out = 2 if allow_divisions else 1
    targets_with_parent: set[Any] = set()
    source_out_count: dict[Any, int] = {}
    accepted: list[tuple[Any, Any, dict[str, Any]]] = []
    eligible = 0

    for node_in, node_out, attributes in edges:
        weight = float(attributes[edge_attr])
        if weight < float(threshold):
            break
        if not math.isfinite(weight) or weight > 1.0:
            raise RuntimeError(
                f"Invalid Trackastra candidate weight {weight!r} for "
                f"{node_in!r}->{node_out!r}"
            )

        eligible += 1

        if node_out in targets_with_parent:
            continue
        if source_out_count.get(node_in, 0) >= max_out:
            continue

        targets_with_parent.add(node_out)
        source_out_count[node_in] = source_out_count.get(node_in, 0) + 1
        accepted.append((node_in, node_out, dict(attributes)))

    solution = nx.DiGraph()
    for node_in, node_out, attributes in accepted:
        if node_in not in solution:
            solution.add_node(
                node_in,
                **dict(candidate_graph.nodes[node_in]),
            )
        if node_out not in solution:
            solution.add_node(
                node_out,
                **dict(candidate_graph.nodes[node_out]),
            )
        solution.add_edge(node_in, node_out, **attributes)

    seconds = time.perf_counter() - started
    return solution, {
        "candidate_edges_total": int(len(edges)),
        "candidate_edges_eligible": int(eligible),
        "accepted_edges": int(len(accepted)),
        "seconds": float(seconds),
        "allow_divisions": bool(allow_divisions),
        "threshold": float(threshold),
    }


def run_trackastra(
    model,
    raw,
    labels,
    *,
    mode: str,
    batch_size: int,
    output_dir: Path,
    metadata: dict[str, Any],
) -> TrackastraResult:
    """
    Run only the Trackastra stages needed by Investigation 43.

    Deliberately DO NOT call model.track():
      * model.track() hides prediction, candidate-graph construction and greedy
        solving behind one call;
      * it additionally applies the solution graph back onto the full mask
        movie, which Investigation 43 does not need;
      * the installed Trackastra greedy solver was observed to stall part-way
        through a small 26k-edge graph on this environment.

    Instead:
      1. model._predict()        -- unchanged Trackastra neural association;
      2. Trackastra build_graph -- unchanged candidate graph;
      3. equivalent local greedy decision rule;
      4. no tracked-mask materialization.

    The scientific quantity used by this investigation is the selected
    detection-edge set, so no information needed for scoring is discarded.
    """
    from trackastra.tracking.tracking import build_graph

    if mode not in {"greedy", "greedy_nodiv"}:
        raise ValueError(
            "Investigation 43's no-mask tracking path supports only "
            "'greedy' and 'greedy_nodiv'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()

    print(
        f"[trackastra] association batch size: {int(batch_size)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 1. Neural association prediction
    # ------------------------------------------------------------------
    predict_started = time.perf_counter()
    predictions = model._predict(
        raw,
        labels,
        batch_size=int(batch_size),
    )
    predict_seconds = time.perf_counter() - predict_started

    node_count = int(len(predictions["nodes"]))
    weight_count = int(len(predictions["weights"]))
    print(
        f"[trackastra] prediction complete: "
        f"nodes={node_count} predicted_weights={weight_count} "
        f"time={predict_seconds:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. Candidate graph -- same defaults as Trackastra._track_from_predictions
    # ------------------------------------------------------------------
    graph_started = time.perf_counter()
    candidate_graph = build_graph(
        nodes=predictions["nodes"],
        weights=predictions["weights"],
        use_distance=False,
        max_distance=256,
        max_neighbors=10,
        delta_t=1,
    )
    candidate_seconds = time.perf_counter() - graph_started

    print(
        f"[trackastra] candidate graph complete: "
        f"nodes={candidate_graph.number_of_nodes()} "
        f"edges={candidate_graph.number_of_edges()} "
        f"time={candidate_seconds:.2f}s",
        flush=True,
    )

    # predictions can contain substantial arrays; release them before solving.
    del predictions
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Greedy solver -- exact Trackastra selection semantics
    # ------------------------------------------------------------------
    graph, greedy_stats = _trackastra_equivalent_greedy(
        candidate_graph,
        allow_divisions=(mode == "greedy"),
        threshold=0.5,
        edge_attr="weight",
    )

    print(
        f"[trackastra] local greedy complete: "
        f"eligible={greedy_stats['candidate_edges_eligible']} "
        f"accepted={greedy_stats['accepted_edges']} "
        f"time={greedy_stats['seconds']:.3f}s",
        flush=True,
    )

    del candidate_graph
    gc.collect()

    seconds = time.perf_counter() - total_started

    predicted = graph_detection_edges(graph)
    edge_table = graph_edge_table(graph)
    node_table = graph_node_table(graph)

    with (output_dir / "track_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)

    atomic_csv(output_dir / "edges.csv", edge_table)
    atomic_csv(output_dir / "nodes.csv", node_table)
    atomic_json(
        output_dir / "trackastra_summary.json",
        {
            **metadata,
            "seconds": float(seconds),
            "prediction_seconds": float(predict_seconds),
            "candidate_graph_seconds": float(candidate_seconds),
            "greedy_seconds": float(greedy_stats["seconds"]),
            "prediction_nodes": int(node_count),
            "prediction_weights": int(weight_count),
            "candidate_edges_total": int(
                greedy_stats["candidate_edges_total"]
            ),
            "candidate_edges_eligible": int(
                greedy_stats["candidate_edges_eligible"]
            ),
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "predicted_detection_edges": int(len(predicted)),
            "solver": "trackastra_equivalent_local_greedy",
            "tracked_masks_persisted": False,
        },
    )

    cleanup_cuda()

    return TrackastraResult(
        graph=graph,
        predicted_edges=frozenset(predicted),
        seconds=float(seconds),
        node_count=int(graph.number_of_nodes()),
        edge_count=int(graph.number_of_edges()),
    )


def load_cached_trackastra(
    output_dir: Path,
    *,
    expected_signature: str,
) -> TrackastraResult | None:
    graph_path = output_dir / "track_graph.pkl"
    summary_path = output_dir / "trackastra_summary.json"
    if not (graph_path.is_file() and summary_path.is_file()):
        return None

    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    summary = read_json(summary_path)
    if str(summary.get("experiment_signature", "")) != str(expected_signature):
        return None
    predicted = graph_detection_edges(graph)
    return TrackastraResult(
        graph=graph,
        predicted_edges=frozenset(predicted),
        seconds=float(summary.get("seconds", math.nan)),
        node_count=int(graph.number_of_nodes()),
        edge_count=int(graph.number_of_edges()),
    )


# =============================================================================
# Metrics
# =============================================================================


def score_edge_set(
    predicted: set[Edge] | frozenset[Edge],
    positive: Iterable[Edge],
    *,
    name: str,
) -> dict[str, Any]:
    target = set(positive)
    recovered = target & set(predicted)
    missed = target - set(predicted)
    return {
        "name": name,
        "target_count": int(len(target)),
        "recovered_count": int(len(recovered)),
        "missed_count": int(len(missed)),
        "recall": safe_ratio(len(recovered), len(target)),
        "recovered_edges": [edge_json(edge) for edge in sorted(recovered)],
        "missed_edges": [edge_json(edge) for edge in sorted(missed)],
    }


def score_absent_edge_set(
    predicted: set[Edge] | frozenset[Edge],
    should_be_absent: Iterable[Edge],
    *,
    name: str,
) -> dict[str, Any]:
    target = set(should_be_absent)
    recreated = target & set(predicted)
    correctly_absent = target - set(predicted)
    return {
        "name": name,
        "target_count": int(len(target)),
        "correctly_absent_count": int(len(correctly_absent)),
        "recreated_count": int(len(recreated)),
        "correct_absence_rate": safe_ratio(len(correctly_absent), len(target)),
        "recreated_edges": [edge_json(edge) for edge in sorted(recreated)],
    }


def score_condition(
    result: TrackastraResult,
    truth: AnnotationTruth,
) -> dict[str, Any]:
    predicted = result.predicted_edges
    return {
        "graph_nodes": result.node_count,
        "graph_edges": result.edge_count,
        "seconds": result.seconds,
        "primary_continue_source": truth.primary_continue_source,
        "primary_continue": score_edge_set(
            predicted,
            truth.primary_continue,
            name="primary_continue",
        ),
        "manual_continue": score_edge_set(
            predicted,
            truth.manual_continue,
            name="manual_continue",
        ),
        "all_forced_nonbirth": score_edge_set(
            predicted,
            truth.forced_nonbirth,
            name="all_forced_nonbirth",
        ),
        "broken_edges": score_absent_edge_set(
            predicted,
            truth.broken,
            name="broken_edges",
        ),
        "birth_edges": score_edge_set(
            predicted,
            truth.birth,
            name="birth_edges",
        ),
    }


def center_lookup_from_current_tracks(paths: Paths) -> dict[Node, np.ndarray]:
    tracks = pd.read_csv(paths.current_tracks)
    result: dict[Node, np.ndarray] = {}
    for row in tracks.itertuples(index=False):
        node = (int(row.frame), int(row.cell_id))
        result[node] = np.asarray(
            [row.z, row.y, row.x],
            dtype=np.float64,
        )
    return result


def fallback_center(
    paths: Paths,
    node: Node,
    cache: dict[Node, np.ndarray],
) -> np.ndarray | None:
    if node in cache:
        return cache[node]

    t, cell_id = node
    frame = load_curated_frame(paths, t)
    coords = np.argwhere(frame == int(cell_id))
    if len(coords) == 0:
        return None
    center = coords.mean(axis=0, dtype=np.float64)
    cache[node] = center
    return center


def build_break_comparison(
    paths: Paths,
    truth: AnnotationTruth,
    baseline: TrackastraResult,
    shifted: TrackastraResult,
    shift: ShiftResult,
    *,
    spacing_zyx: Sequence[float],
) -> pd.DataFrame:
    centers = center_lookup_from_current_tracks(paths)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for edge in sorted(truth.primary_continue):
        left, right = edge
        c0 = fallback_center(paths, left, centers)
        c1 = fallback_center(paths, right, centers)

        original_delta = None
        aligned_delta = None
        original_distance = math.nan
        aligned_distance = math.nan

        if c0 is not None and c1 is not None:
            original_delta = c1 - c0
            aligned_delta = (
                original_delta
                + shift.align_int_zyx[right[0]]
                - shift.align_int_zyx[left[0]]
            )
            original_distance = float(
                np.linalg.norm(original_delta * spacing)
            )
            aligned_distance = float(
                np.linalg.norm(aligned_delta * spacing)
            )

        baseline_has = edge in baseline.predicted_edges
        shifted_has = edge in shifted.predicted_edges

        rows.append(
            {
                "frame_from": int(left[0]),
                "cell_from": int(left[1]),
                "frame_to": int(right[0]),
                "cell_to": int(right[1]),
                "baseline_connected": bool(baseline_has),
                "oracle_shift_connected": bool(shifted_has),
                "fixed_by_shift": bool((not baseline_has) and shifted_has),
                "regressed_by_shift": bool(baseline_has and (not shifted_has)),
                "original_dz": (
                    float(original_delta[0])
                    if original_delta is not None
                    else math.nan
                ),
                "original_dy": (
                    float(original_delta[1])
                    if original_delta is not None
                    else math.nan
                ),
                "original_dx": (
                    float(original_delta[2])
                    if original_delta is not None
                    else math.nan
                ),
                "original_distance_um": original_distance,
                "aligned_dz": (
                    float(aligned_delta[0])
                    if aligned_delta is not None
                    else math.nan
                ),
                "aligned_dy": (
                    float(aligned_delta[1])
                    if aligned_delta is not None
                    else math.nan
                ),
                "aligned_dx": (
                    float(aligned_delta[2])
                    if aligned_delta is not None
                    else math.nan
                ),
                "aligned_distance_um": aligned_distance,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Reporting
# =============================================================================


def annotation_audit(
    paths: Paths,
    truth: AnnotationTruth,
    shift: ShiftResult,
    *,
    manual_frames: Sequence[int],
    spatial_shape_zyx: Sequence[int],
    spacing_zyx: Sequence[float],
) -> dict[str, Any]:
    operations = read_json(paths.spatial_operations).get("operations", [])
    operation_counts: dict[str, int] = {}
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        key = str(operation.get("type", "unknown"))
        operation_counts[key] = operation_counts.get(key, 0) + 1

    return {
        "sample": paths.sample,
        "split": paths.split,
        "annotation_set": paths.annotation_set,
        "frame_count": paths.frame_count,
        "spatial_shape_zyx": [int(v) for v in spatial_shape_zyx],
        "spacing_zyx_um": [float(v) for v in spacing_zyx],
        "manual_instance_frames": [int(v) for v in manual_frames],
        "manual_instance_frame_count": int(len(manual_frames)),
        "spatial_operation_counts": dict(sorted(operation_counts.items())),
        "track_truth": {
            "forced_nonbirth": len(truth.forced_nonbirth),
            "explicit_manual_continue": len(truth.manual_continue),
            "auto_repair": len(truth.auto_repair),
            "unclassified_forced": len(truth.unclassified_forced),
            "broken": len(truth.broken),
            "birth_edges": len(truth.birth),
            "ignored_nodes": len(truth.ignored_nodes),
            "primary_continue_count": len(truth.primary_continue),
            "primary_continue_source": truth.primary_continue_source,
        },
        "oracle_shift": {
            "canvas_shape_zyx": [int(v) for v in shift.canvas_shape_zyx],
            "maximum_absolute_integer_alignment_zyx": [
                int(v)
                for v in np.max(
                    np.abs(shift.align_int_zyx),
                    axis=0,
                ).tolist()
            ],
        },
        "note": (
            "The oracle shift uses annotation-owned corrected one-to-one "
            "tracklet continuations. It is intentionally not an inference-time "
            "shift estimator."
        ),
    }


def print_audit(audit: dict[str, Any]) -> None:
    banner("INVESTIGATION 43 — BIOHUB TRACKASTRA GLOBAL MOTION COMPENSATION")
    print(f"sample              : {audit['split']}/{audit['sample']}")
    print(f"annotation set      : {audit['annotation_set']}")
    print(f"frames              : {audit['frame_count']}")
    print(f"shape ZYX           : {tuple(audit['spatial_shape_zyx'])}")
    print(f"spacing ZYX um      : {tuple(audit['spacing_zyx_um'])}")
    print(
        f"manual inst. frames : {audit['manual_instance_frame_count']} "
        f"{audit['manual_instance_frames']}"
    )
    print("")
    print("TRACK SCORING LABELS")
    print("-" * 104)
    truth = audit["track_truth"]
    print(f"manual Continue     : {truth['explicit_manual_continue']}")
    print(f"auto repair edges   : {truth['auto_repair']}")
    print(f"unclassified forced : {truth['unclassified_forced']}")
    print(f"all forced nonbirth : {truth['forced_nonbirth']}")
    print(f"manual Break        : {truth['broken']}")
    print(f"Birth edges         : {truth['birth_edges']}")
    print(f"ignored nodes       : {truth['ignored_nodes']}")
    print(
        f"PRIMARY Continue    : {truth['primary_continue_count']} "
        f"({truth['primary_continue_source']})"
    )
    print("")
    print("ORACLE SHIFT")
    print("-" * 104)
    print(
        f"padded canvas ZYX   : "
        f"{tuple(audit['oracle_shift']['canvas_shape_zyx'])}"
    )
    print(
        "max |alignment| ZYX : "
        f"{tuple(audit['oracle_shift']['maximum_absolute_integer_alignment_zyx'])}"
    )
    print("=" * 104, flush=True)


def print_final_comparison(comparison: dict[str, Any]) -> None:
    baseline = comparison["baseline"]
    shifted = comparison["oracle_shift"]

    b_continue = baseline["primary_continue"]
    s_continue = shifted["primary_continue"]
    b_break = baseline["broken_edges"]
    s_break = shifted["broken_edges"]
    b_birth = baseline["birth_edges"]
    s_birth = shifted["birth_edges"]

    banner("INVESTIGATION 43 — FINAL COMPARISON")

    print(
        f"primary Continue source : {comparison['primary_continue_source']}",
        flush=True,
    )
    print("")
    print(
        f"{'METHOD':<24}"
        f"{'CONTINUE':>18}"
        f"{'STILL BROKEN':>18}"
        f"{'WRONG EDGES':>18}"
        f"{'BIRTH':>18}"
    )
    print("-" * 96)

    def line(name: str, c: dict, br: dict, bi: dict) -> None:
        continue_text = (
            f"{c['recovered_count']}/{c['target_count']} "
            f"({format_pct(c['recall'])})"
        )
        wrong_text = (
            f"{br['recreated_count']}/{br['target_count']}"
        )
        birth_text = (
            f"{bi['recovered_count']}/{bi['target_count']} "
            f"({format_pct(bi['recall'])})"
        )
        print(
            f"{name:<24}"
            f"{continue_text:>18}"
            f"{c['missed_count']:>18}"
            f"{wrong_text:>18}"
            f"{birth_text:>18}"
        )

    line("baseline", b_continue, b_break, b_birth)
    line("oracle global shift", s_continue, s_break, s_birth)

    print("")
    print("KNOWN BREAK CHANGE")
    print("-" * 96)
    print(
        f"baseline -> shifted : "
        f"{b_continue['missed_count']} -> {s_continue['missed_count']}"
    )
    print(
        f"break reduction     : {comparison['known_break_reduction_count']}"
    )
    print(
        f"relative reduction  : "
        f"{format_pct(comparison['known_break_reduction_fraction'])}"
    )
    print(
        f"fixed by shift      : {comparison['fixed_by_shift_count']}"
    )
    print(
        f"regressed by shift  : {comparison['regressed_by_shift_count']}"
    )
    print("=" * 104, flush=True)


# =============================================================================
# Experiment
# =============================================================================


def run(args: argparse.Namespace) -> None:
    paths = make_paths(args)

    if args.force and paths.output.exists():
        shutil.rmtree(paths.output)

    paths.output.mkdir(parents=True, exist_ok=True)

    truth = load_annotation_truth(paths)
    manual_frames = manual_frame_ids(paths)

    base = np.load(
        paths.base_instances,
        mmap_mode="r",
        allow_pickle=False,
    )
    if base.ndim != 4:
        raise ValueError(f"Expected (T,Z,Y,X) final instances, got {base.shape}")
    spatial_shape = tuple(int(v) for v in base.shape[1:])
    del base

    spacing = tuple(float(v) for v in args.spacing_zyx_um)
    if len(spacing) != 3 or any(v <= 0 or not math.isfinite(v) for v in spacing):
        raise ValueError("--spacing-zyx-um must contain three positive finite values")

    shift = estimate_oracle_shifts(
        paths,
        truth,
        spatial_shape,
        spacing_zyx=spacing,
        minimum_pairs=int(args.min_shift_pairs),
        mad_scale=float(args.shift_mad_scale),
        minimum_residual_gate_um=float(args.minimum_residual_gate_um),
    )
    atomic_csv(paths.shifts_csv, shift.rows)

    shift_summary = {
        "pairwise_observed_displacement_zyx": shift.pairwise_float_zyx.tolist(),
        "cumulative_observed_displacement_zyx": shift.cumulative_float_zyx.tolist(),
        "integer_alignment_applied_zyx": shift.align_int_zyx.tolist(),
        "padded_placement_zyx": shift.placement_zyx.tolist(),
        "canvas_shape_zyx": list(shift.canvas_shape_zyx),
        "rounding": "numpy.rint(cumulative displacement), then negate",
        "wraparound": False,
    }
    atomic_json(paths.shift_summary_json, shift_summary)

    audit = annotation_audit(
        paths,
        truth,
        shift,
        manual_frames=manual_frames,
        spatial_shape_zyx=spatial_shape,
        spacing_zyx=spacing,
    )
    audit["provenance"] = provenance_signature(paths)
    atomic_json(paths.audit_json, audit)
    print_audit(audit)

    if args.audit_only:
        print(f"[audit-only] wrote: {paths.audit_json}")
        print(f"[audit-only] wrote: {paths.shifts_csv}")
        return

    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is required for Investigation 43. Activate the same "
            "repository environment used by dataset_curation inference."
        ) from exc

    experiment_base = {
        "provenance": provenance_signature(paths),
        "model": str(args.trackastra_model),
        "mode": str(args.trackastra_mode),
        "device": str(args.trackastra_device),
        "batch_size": int(args.trackastra_batch_size),
        "solver": "trackastra_equivalent_local_greedy_v1",
        "tracked_masks_persisted": False,
    }
    baseline_signature = digest_payload(
        {**experiment_base, "condition": "baseline"}
    )
    shifted_signature = digest_payload(
        {
            **experiment_base,
            "condition": "oracle_global_shift",
            "integer_alignment_applied_zyx": shift.align_int_zyx.tolist(),
            "canvas_shape_zyx": list(shift.canvas_shape_zyx),
        }
    )

    print("")
    print("[trackastra] loading model once for both conditions", flush=True)
    model = Trackastra.from_pretrained(
        str(args.trackastra_model),
        device=str(args.trackastra_device),
        batch_size=int(args.trackastra_batch_size),
    )

    # -------------------------------------------------------------------------
    # A. Baseline
    # -------------------------------------------------------------------------
    baseline = None
    if not args.force:
        baseline = load_cached_trackastra(
            paths.baseline_dir,
            expected_signature=baseline_signature,
        )
        if baseline is not None:
            print("[baseline] reusing cached Trackastra graph", flush=True)

    if baseline is None:
        banner("INVESTIGATION 43 — BASELINE TRACKASTRA ON CURATED INSTANCES")
        source_handle, raw, labels, _shape = build_baseline_inputs(paths)
        baseline = run_trackastra(
            model,
            raw,
            labels,
            mode=str(args.trackastra_mode),
            batch_size=int(args.trackastra_batch_size),
            output_dir=paths.baseline_dir,
            metadata={
                "condition": "baseline",
                "model": str(args.trackastra_model),
                "mode": str(args.trackastra_mode),
                "device": str(args.trackastra_device),
                "batch_size": int(args.trackastra_batch_size),
                "raw_input": str(paths.zarr),
                "instance_input": "lazy curated frames",
                "global_alignment": False,
                "experiment_signature": baseline_signature,
            },
        )
        del raw, labels, source_handle
        cleanup_cuda()

    baseline_metrics = score_condition(baseline, truth)
    atomic_json(paths.baseline_dir / "metrics.json", baseline_metrics)

    # -------------------------------------------------------------------------
    # B. Oracle shift
    # -------------------------------------------------------------------------
    shifted = None
    if not args.force:
        shifted = load_cached_trackastra(
            paths.shifted_dir,
            expected_signature=shifted_signature,
        )
        if shifted is not None:
            print("[oracle shift] reusing cached Trackastra graph", flush=True)

    if shifted is None:
        banner("INVESTIGATION 43 — ORACLE GLOBAL-SHIFT TRACKASTRA")

        handles = materialize_shifted_inputs(
            paths,
            shift,
            max_temp_gib=float(args.max_temp_gib),
        )
        raw_shifted, labels_shifted = build_shifted_dask(handles)

        shifted = run_trackastra(
            model,
            raw_shifted,
            labels_shifted,
            mode=str(args.trackastra_mode),
            batch_size=int(args.trackastra_batch_size),
            output_dir=paths.shifted_dir,
            metadata={
                "condition": "oracle_global_shift",
                "model": str(args.trackastra_model),
                "mode": str(args.trackastra_mode),
                "device": str(args.trackastra_device),
                "batch_size": int(args.trackastra_batch_size),
                "global_alignment": True,
                "shift_source": "annotation_owned_current_tracks",
                "integer_shift": True,
                "wraparound": False,
                "canvas_shape_zyx": list(shift.canvas_shape_zyx),
                "experiment_signature": shifted_signature,
            },
        )

        del raw_shifted, labels_shifted
        close_memmap(handles.raw)
        close_memmap(handles.labels)
        del handles
        cleanup_cuda()

        if not args.keep_shifted_inputs:
            # On Windows all memmap/Dask references must be gone first.
            gc.collect()
            paths.shifted_raw.unlink(missing_ok=True)
            paths.shifted_labels.unlink(missing_ok=True)
            try:
                paths.temp_dir.rmdir()
            except OSError:
                pass
            print("[shift] removed temporary shifted movies", flush=True)
        else:
            print(
                f"[shift] kept temporary movies in {paths.temp_dir}",
                flush=True,
            )

    shifted_metrics = score_condition(shifted, truth)
    atomic_json(paths.shifted_dir / "metrics.json", shifted_metrics)

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    break_frame = build_break_comparison(
        paths,
        truth,
        baseline,
        shifted,
        shift,
        spacing_zyx=spacing,
    )
    atomic_csv(paths.break_comparison_csv, break_frame)

    b_missed = int(
        baseline_metrics["primary_continue"]["missed_count"]
    )
    s_missed = int(
        shifted_metrics["primary_continue"]["missed_count"]
    )
    reduction = b_missed - s_missed
    reduction_fraction = (
        float(reduction) / float(b_missed)
        if b_missed > 0
        else None
    )

    comparison = {
        "sample": paths.sample,
        "split": paths.split,
        "annotation_set": paths.annotation_set,
        "model": str(args.trackastra_model),
        "mode": str(args.trackastra_mode),
        "primary_continue_source": truth.primary_continue_source,
        "baseline": baseline_metrics,
        "oracle_shift": shifted_metrics,
        "known_break_reduction_count": int(reduction),
        "known_break_reduction_fraction": reduction_fraction,
        "fixed_by_shift_count": (
            int(break_frame["fixed_by_shift"].sum())
            if not break_frame.empty
            else 0
        ),
        "regressed_by_shift_count": (
            int(break_frame["regressed_by_shift"].sum())
            if not break_frame.empty
            else 0
        ),
        "oracle_shift_is_ground_truth_assisted": True,
        "automatic_shift_estimation_tested": False,
        "interpretation_rule": (
            "A large reduction in missed primary Continue edges without an "
            "increase in recreated Break edges supports the hypothesis that "
            "global translation is a material Trackastra failure mode."
        ),
    }
    atomic_json(paths.comparison_json, comparison)
    print_final_comparison(comparison)

    print("")
    print(f"[output] {paths.output}", flush=True)


# =============================================================================
# CLI
# =============================================================================


def parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in str(text).split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "spacing must be comma-separated Z,Y,X"
        )
    if any(v <= 0 or not math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError(
            "spacing values must be positive finite numbers"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Investigation 43: compare Trackastra on curated BioHub instances "
            "before and after oracle global-translation compensation."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--annotation-set",
        default=DEFAULT_ANNOTATION_SET,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--spacing-zyx-um",
        type=parse_spacing,
        default=tuple(float(v) for v in DEFAULT_SPACING_ZYX_UM),
        help="Physical Z,Y,X spacing in micrometres.",
    )

    parser.add_argument(
        "--trackastra-model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trackastra-mode",
        default=DEFAULT_MODE,
    )
    parser.add_argument(
        "--trackastra-device",
        default=DEFAULT_DEVICE,
    )
    parser.add_argument(
        "--trackastra-batch-size",
        type=int,
        default=DEFAULT_TRACKASTRA_BATCH_SIZE,
        help=(
            "Trackastra association-prediction batch size. Default=1. "
            "Investigation 43 defaults to CPU because the current cu130 "
            "environment fails Trackastra transformer F.linear on CUDA."
        ),
    )

    parser.add_argument(
        "--min-shift-pairs",
        type=int,
        default=10,
        help=(
            "Minimum corrected one-to-one continuation pairs required for "
            "every adjacent frame pair."
        ),
    )
    parser.add_argument(
        "--shift-mad-scale",
        type=float,
        default=4.0,
        help="MAD multiplier for robust oracle displacement consensus.",
    )
    parser.add_argument(
        "--minimum-residual-gate-um",
        type=float,
        default=2.0,
        help="Minimum physical residual gate for oracle shift inliers.",
    )

    parser.add_argument(
        "--max-temp-gib",
        type=float,
        default=12.0,
        help=(
            "Safety limit for the combined padded shifted raw+label temporary "
            "movies."
        ),
    )
    parser.add_argument(
        "--keep-shifted-inputs",
        action="store_true",
        help="Keep the large padded shifted raw/label movies after Trackastra.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit annotations and calculate oracle shifts without Trackastra.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild this sample's Investigation-43 output.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if int(args.trackastra_batch_size) < 1:
        raise ValueError("--trackastra-batch-size must be >= 1")
    if int(args.min_shift_pairs) < 1:
        raise ValueError("--min-shift-pairs must be >= 1")
    if float(args.shift_mad_scale) <= 0:
        raise ValueError("--shift-mad-scale must be positive")
    if float(args.minimum_residual_gate_um) <= 0:
        raise ValueError("--minimum-residual-gate-um must be positive")
    if float(args.max_temp_gib) <= 0:
        raise ValueError("--max-temp-gib must be positive")

    run(args)


if __name__ == "__main__":
    main()
