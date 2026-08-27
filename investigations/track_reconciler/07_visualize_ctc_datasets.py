from __future__ import annotations

r"""
Investigation 07 — visualize Fluo-N3DH-CE or Fluo-N3DH-SIM+ raw 3-D CTC
movies together with available SEG/TRA ground truth in Napari.

Examples
--------
SIM+ sequence 01:
    python .\investigations\track_reconciler\07_visualize_ctc_datasets.py --dataset sim --sequence 01

CE sequence 02:
    python .\investigations\track_reconciler\07_visualize_ctc_datasets.py --dataset ce --sequence 02

Temporal subset:
    python .\investigations\track_reconciler\07_visualize_ctc_datasets.py --dataset sim --sequence 01 --start-frame 40 --end-frame 70

If CE ground truth is stored separately:
    python .\investigations\track_reconciler\07_visualize_ctc_datasets.py --dataset ce --sequence 01 --gt-root D:\path\to\01_GT

The TIFF movie is assembled lazily with Dask as [T,Z,Y,X], so the complete
sequence is not copied into RAM before Napari opens.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_NAME = "07_visualize_ctc_datasets"

DATASET_DIR = {
    "ce": "Fluo-N3DH-CE",
    "sim": "Fluo-N3DH-SIM+",
}
DATASET_NAME = {
    "ce": "Fluo-N3DH-CE",
    "sim": "Fluo-N3DH-SIM+",
}
ALIASES = {
    "ce": "ce",
    "fluo-n3dh-ce": "ce",
    "n3dh-ce": "ce",
    "sim": "sim",
    "sim+": "sim",
    "fluo-n3dh-sim+": "sim",
    "fluo-n3dh-sim": "sim",
    "n3dh-sim+": "sim",
}

# Z,Y,X micrometres. Override with --spacing when desired.
DEFAULT_SPACING_ZYX_UM = {
    "ce": (1.0, 0.09, 0.09),
    "sim": (0.2, 0.125, 0.125),
}

RAW_RE = re.compile(r"^t(\d+)\.tiff?$", re.IGNORECASE)
SEG_RE = re.compile(r"^man_seg(\d+)\.tiff?$", re.IGNORECASE)
TRA_RE = re.compile(r"^man_track(\d+)\.tiff?$", re.IGNORECASE)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
        if (candidate / "investigations").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not resolve repository root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def normalize_dataset(value: str) -> str:
    key = str(value).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown dataset {value!r}; use ce or sim")
    return ALIASES[key]


def parse_spacing(text: str) -> tuple[float, float, float]:
    values = tuple(float(v.strip()) for v in text.split(","))
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("--spacing must be positive Z,Y,X values")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class Paths:
    key: str
    root: Path
    sequence: str
    raw_dir: Path
    gt_root: Path
    seg_dir: Path
    tra_dir: Path
    track_txt: Path


@dataclass(frozen=True)
class Series:
    files: dict[int, Path]

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(sorted(self.files))


@dataclass(frozen=True)
class VolumeSpec:
    shape: tuple[int, int, int]
    dtype: np.dtype


def dataset_paths(dataset: str, sequence: str, root_override: str | None, gt_override: str | None) -> Paths:
    root = resolve(root_override) if root_override else (ROOT / "data" / "external" / DATASET_DIR[dataset]).resolve()
    gt_root = resolve(gt_override) if gt_override else root / f"{sequence}_GT"
    return Paths(
        key=dataset,
        root=root,
        sequence=sequence,
        raw_dir=root / sequence,
        gt_root=gt_root,
        seg_dir=gt_root / "SEG",
        tra_dir=gt_root / "TRA",
        track_txt=gt_root / "TRA" / "man_track.txt",
    )


def discover(directory: Path, pattern: re.Pattern[str], *, required: bool, label: str) -> Series | None:
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(f"{label} directory not found: {directory}")
        return None
    files: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            frame = int(match.group(1))
            if frame in files:
                raise RuntimeError(f"Duplicate {label} frame {frame}")
            files[frame] = path
    if not files:
        if required:
            raise FileNotFoundError(f"No {label} TIFF files in {directory}")
        return None
    return Series(files)


def read_volume(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("tifffile is required") from exc
    array = np.squeeze(np.asarray(tifffile.imread(path)))
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D TIFF at {path}, got {array.shape}")
    return array


def inspect(path: Path) -> VolumeSpec:
    array = read_volume(path)
    return VolumeSpec(tuple(int(v) for v in array.shape), np.dtype(array.dtype))


def choose_frames(raw: Series, start: int | None, end: int | None) -> tuple[int, ...]:
    all_frames = raw.frames
    lo = all_frames[0] if start is None else int(start)
    hi = all_frames[-1] if end is None else int(end)
    if lo > hi:
        raise ValueError("--start-frame cannot exceed --end-frame")
    frames = tuple(f for f in all_frames if lo <= f <= hi)
    if not frames:
        raise ValueError(f"No raw frames in [{lo}, {hi}]")
    missing = sorted(set(range(frames[0], frames[-1] + 1)) - set(frames))
    if missing:
        raise RuntimeError(f"Raw sequence has missing frame IDs: {missing[:20]}")
    return frames


def lazy_stack(frames: tuple[int, ...], files: dict[int, Path], spec: VolumeSpec):
    try:
        import dask.array as da
        from dask import delayed
    except ImportError as exc:
        raise RuntimeError("dask[array] is required for lazy visualization") from exc

    def load(path: Path | None) -> np.ndarray:
        if path is None:
            return np.zeros(spec.shape, dtype=spec.dtype)
        array = read_volume(path)
        if tuple(array.shape) != spec.shape:
            raise ValueError(f"Shape mismatch in {path}: {array.shape} != {spec.shape}")
        return array.astype(spec.dtype, copy=False)

    pieces = [
        da.from_delayed(delayed(load)(files.get(frame)), shape=spec.shape, dtype=spec.dtype)
        for frame in frames
    ]
    return da.stack(pieces, axis=0)


def aligned_label_stack(series: Series | None, frames: tuple[int, ...], raw_spec: VolumeSpec, label: str):
    if series is None:
        return None
    overlapping = [f for f in frames if f in series.files]
    if not overlapping:
        print(f"[warning] {label}: no GT frames overlap selected raw interval", flush=True)
        return None
    spec = inspect(series.files[overlapping[0]])
    if spec.shape != raw_spec.shape:
        raise ValueError(f"{label} shape {spec.shape} does not match raw {raw_spec.shape}")
    missing = [f for f in frames if f not in series.files]
    if missing:
        print(
            f"[warning] {label}: {len(missing)} selected frames have no GT; zero labels will be used. First: {missing[:10]}",
            flush=True,
        )
    return lazy_stack(frames, series.files, spec)


def lineage_summary(path: Path) -> None:
    if not path.is_file():
        return
    try:
        table = np.loadtxt(path, dtype=np.int64, ndmin=2)
        if table.shape[1] >= 4:
            print(
                f"[TRA] man_track.txt tracks={len(table):,} parented_tracks={(table[:,3] > 0).sum():,}",
                flush=True,
            )
    except Exception as exc:
        print(f"[warning] could not parse {path}: {exc}", flush=True)


def open_napari(paths: Paths, frames: tuple[int, ...], raw_stack, raw_spec: VolumeSpec, seg_stack, tra_stack, spacing):
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("napari is required") from exc

    scale = (1.0, *spacing)
    first = read_volume(paths.raw_dir / f"t{frames[0]:03d}.tif")
    low, high = np.percentile(first, [1.0, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(first.min()), float(first.max())
        if high <= low:
            high = low + 1.0

    viewer = napari.Viewer(ndisplay=3)
    viewer.title = f"{SCRIPT_NAME} | {DATASET_NAME[paths.key]} | {paths.sequence}"

    viewer.add_image(
        raw_stack,
        name=f"Raw — {DATASET_NAME[paths.key]} {paths.sequence}",
        scale=scale,
        rendering="mip",
        colormap="gray",
        contrast_limits=[float(low), float(high)],
    )

    if seg_stack is not None:
        seg = viewer.add_labels(
            seg_stack,
            name="GT SEG — instance segmentation",
            scale=scale,
            opacity=0.55,
        )
        seg.visible = True

    if tra_stack is not None:
        tra = viewer.add_labels(
            tra_stack,
            name="GT TRA — tracking identities",
            scale=scale,
            opacity=0.55,
        )
        tra.visible = seg_stack is None

    print("", flush=True)
    print("=" * 100, flush=True)
    print("INVESTIGATION 07 — CTC RAW + GROUND TRUTH", flush=True)
    print("=" * 100, flush=True)
    print(f"dataset        : {DATASET_NAME[paths.key]}", flush=True)
    print(f"sequence       : {paths.sequence}", flush=True)
    print(f"frames         : {frames[0]}..{frames[-1]} ({len(frames)})", flush=True)
    print(f"shape/frame    : {raw_spec.shape}", flush=True)
    print(f"dtype          : {raw_spec.dtype}", flush=True)
    print(f"spacing ZYX um : {spacing}", flush=True)
    print(f"GT SEG         : {'yes' if seg_stack is not None else 'no'}", flush=True)
    print(f"GT TRA         : {'yes' if tra_stack is not None else 'no'}", flush=True)
    print("loading        : lazy Dask-backed TIFF stack", flush=True)
    print("=" * 100, flush=True)
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Fluo-N3DH-CE or Fluo-N3DH-SIM+ raw 3-D data with available CTC GT."
    )
    parser.add_argument("--dataset", required=True, help="ce or sim")
    parser.add_argument("--sequence", choices=("01", "02"), default="01")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--gt-root", default=None, help="Override sequence GT directory, e.g. ...\\01_GT")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--spacing", default=None, help="Z,Y,X micrometres")
    parser.add_argument(
        "--require-gt",
        action="store_true",
        help="Fail if neither SEG nor TRA GT is found; otherwise raw-only viewing is allowed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = normalize_dataset(args.dataset)
    paths = dataset_paths(dataset, args.sequence, args.dataset_root, args.gt_root)

    raw = discover(paths.raw_dir, RAW_RE, required=True, label="raw")
    assert raw is not None
    frames = choose_frames(raw, args.start_frame, args.end_frame)
    raw_spec = inspect(raw.files[frames[0]])
    raw_stack = lazy_stack(frames, raw.files, raw_spec)

    seg_series = discover(paths.seg_dir, SEG_RE, required=False, label="SEG GT")
    tra_series = discover(paths.tra_dir, TRA_RE, required=False, label="TRA GT")

    if seg_series is None and tra_series is None:
        message = (
            f"No local GT found for {DATASET_NAME[dataset]} sequence {args.sequence}. "
            f"Expected {paths.gt_root}. Raw will still be shown. "
            "Use --gt-root if the GT is stored elsewhere."
        )
        if args.require_gt:
            raise FileNotFoundError(message)
        print(f"[warning] {message}", flush=True)

    seg_stack = aligned_label_stack(seg_series, frames, raw_spec, "SEG GT")
    tra_stack = aligned_label_stack(tra_series, frames, raw_spec, "TRA GT")
    lineage_summary(paths.track_txt)

    spacing = DEFAULT_SPACING_ZYX_UM[dataset] if args.spacing is None else parse_spacing(args.spacing)

    print(
        f"[raw] {paths.raw_dir} | frames={frames[0]}..{frames[-1]} | count={len(frames)} | shape={raw_spec.shape} | dtype={raw_spec.dtype}",
        flush=True,
    )
    print(
        f"[GT] {paths.gt_root} | SEG={'yes' if seg_series else 'no'} | TRA={'yes' if tra_series else 'no'}",
        flush=True,
    )

    open_napari(paths, frames, raw_stack, raw_spec, seg_stack, tra_stack, spacing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
