from __future__ import annotations

r"""
Investigation 02 — audit and visualize the ambiguous synthetic continuation
overfit dataset produced by 01_build_overfit_dataset.py.

This script is intentionally diagnostic.  It does not train anything.

It checks:
- schema and referential integrity,
- exactly one true continuation per source and target,
- presence of cross-identity hard negatives,
- candidate degree / density,
- whether geometry alone already makes every decision trivial,
- temporal-window continuity,
- provenance consistency with Investigation-36 arrays.

Optional Napari visualization shows one example directly on the original
BioHub movie:
- source and target synthetic tracklets,
- true continuation edges,
- wrong candidate edges,
- global-only and global+relative source predictions.

Typical usage
-------------
    python .\investigations\track_reconciler\02_inspect_overfit_dataset.py

Inspect the hardest example:
    python .\investigations\track_reconciler\02_inspect_overfit_dataset.py --viewer

Inspect a particular example:
    python .\investigations\track_reconciler\02_inspect_overfit_dataset.py ^
        --example 7 --viewer

Print all examples:
    python .\investigations\track_reconciler\02_inspect_overfit_dataset.py ^
        --show-all
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_NAME = "02_inspect_overfit_dataset"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.parent, *here.parents, Path.cwd().resolve())
    for candidate in candidates:
        if (
            (candidate / "learned" / "track_reconciler").is_dir()
            and (candidate / "investigations" / "stirnet").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "pyproject.toml").is_file()
        ):
            return candidate
    raise RuntimeError("Could not resolve the cell-tracking repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def default_dataset(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "investigations"
        / "01_build_overfit_dataset"
        / sample_id
    ).resolve()


@dataclass(frozen=True)
class Dataset:
    root: Path
    manifest: dict[str, Any]
    examples: pd.DataFrame
    tracklets: pd.DataFrame
    observations: pd.DataFrame
    edges: pd.DataFrame
    global_motion: pd.DataFrame

    @classmethod
    def load(cls, root: Path) -> "Dataset":
        required = (
            "manifest.json",
            "examples.csv",
            "tracklets.csv",
            "observations.csv",
            "edges.csv",
            "global_motion.csv",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Dataset is incomplete at {root}; missing {missing}"
            )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            root=root,
            manifest=manifest,
            examples=pd.read_csv(root / "examples.csv"),
            tracklets=pd.read_csv(root / "tracklets.csv"),
            observations=pd.read_csv(root / "observations.csv"),
            edges=pd.read_csv(root / "edges.csv"),
            global_motion=pd.read_csv(root / "global_motion.csv"),
        )


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_dataset(data: Dataset) -> list[str]:
    warnings: list[str] = []

    require_columns(
        data.examples,
        {
            "example_id",
            "cut_frame",
            "track_count",
            "candidate_edge_count",
            "positive_edge_count",
            "negative_edge_count",
            "median_wrong_minus_true_um",
        },
        "examples.csv",
    )
    require_columns(
        data.tracklets,
        {
            "example_id",
            "tracklet_index",
            "identity_index",
            "role",
            "original_track_id",
            "paired_tracklet_index",
            "start_frame",
            "end_frame",
            "observation_count",
        },
        "tracklets.csv",
    )
    require_columns(
        data.observations,
        {
            "example_id",
            "tracklet_index",
            "identity_index",
            "role",
            "sequence_index",
            "original_track_id",
            "frame",
            "cell_id",
            "z",
            "y",
            "x",
            "z_um",
            "y_um",
            "x_um",
        },
        "observations.csv",
    )
    require_columns(
        data.edges,
        {
            "example_id",
            "edge_index",
            "source_tracklet_index",
            "target_tracklet_index",
            "source_original_track_id",
            "target_original_track_id",
            "continuation_target",
            "direct_distance_um",
            "source_distance_rank",
            "target_distance_rank",
        },
        "edges.csv",
    )

    example_ids = set(data.examples["example_id"].astype(int))
    for frame, name in (
        (data.tracklets, "tracklets"),
        (data.observations, "observations"),
        (data.edges, "edges"),
    ):
        foreign = set(frame["example_id"].astype(int)).difference(example_ids)
        if foreign:
            raise ValueError(f"{name} references unknown example_ids: {sorted(foreign)}")

    if data.tracklets.duplicated(["example_id", "tracklet_index"]).any():
        raise ValueError("tracklets.csv has duplicate (example_id, tracklet_index)")
    if data.edges.duplicated(["example_id", "edge_index"]).any():
        raise ValueError("edges.csv has duplicate (example_id, edge_index)")
    if data.observations.duplicated(
        ["example_id", "tracklet_index", "sequence_index"]
    ).any():
        raise ValueError(
            "observations.csv has duplicate "
            "(example_id, tracklet_index, sequence_index)"
        )

    for example_id in sorted(example_ids):
        ex = data.examples[data.examples["example_id"] == example_id].iloc[0]
        tracklets = data.tracklets[data.tracklets["example_id"] == example_id]
        obs = data.observations[data.observations["example_id"] == example_id]
        edges = data.edges[data.edges["example_id"] == example_id]

        source = tracklets[tracklets["role"] == "source"]
        target = tracklets[tracklets["role"] == "target"]
        expected_tracks = int(ex["track_count"])

        if len(source) != expected_tracks or len(target) != expected_tracks:
            raise ValueError(
                f"example {example_id}: expected {expected_tracks} source and "
                f"target tracklets, got {len(source)} / {len(target)}"
            )

        known_tracklets = set(tracklets["tracklet_index"].astype(int))
        referenced = set(edges["source_tracklet_index"].astype(int)) | set(
            edges["target_tracklet_index"].astype(int)
        )
        if not referenced.issubset(known_tracklets):
            raise ValueError(
                f"example {example_id}: edges reference unknown tracklets "
                f"{sorted(referenced.difference(known_tracklets))}"
            )

        positive = edges[edges["continuation_target"].astype(int) == 1]
        negative = edges[edges["continuation_target"].astype(int) == 0]

        if len(positive) != expected_tracks:
            raise ValueError(
                f"example {example_id}: expected {expected_tracks} positives, "
                f"got {len(positive)}"
            )
        if negative.empty:
            raise ValueError(f"example {example_id}: has no wrong candidate edges")

        source_positive_counts = positive.groupby(
            "source_tracklet_index"
        ).size()
        target_positive_counts = positive.groupby(
            "target_tracklet_index"
        ).size()
        if not (source_positive_counts == 1).all() or len(
            source_positive_counts
        ) != expected_tracks:
            raise ValueError(
                f"example {example_id}: every source must have exactly one positive"
            )
        if not (target_positive_counts == 1).all() or len(
            target_positive_counts
        ) != expected_tracks:
            raise ValueError(
                f"example {example_id}: every target must have exactly one positive"
            )

        if not (
            positive["source_original_track_id"].astype(int)
            == positive["target_original_track_id"].astype(int)
        ).all():
            raise ValueError(
                f"example {example_id}: positive identity provenance is inconsistent"
            )
        if (
            negative["source_original_track_id"].astype(int)
            == negative["target_original_track_id"].astype(int)
        ).any():
            raise ValueError(
                f"example {example_id}: a negative edge preserves original identity"
            )

        for tracklet in tracklets.itertuples(index=False):
            rows = obs[
                obs["tracklet_index"].astype(int) == int(tracklet.tracklet_index)
            ].sort_values("sequence_index")
            if len(rows) != int(tracklet.observation_count):
                raise ValueError(
                    f"example {example_id}, tracklet {tracklet.tracklet_index}: "
                    f"observation_count mismatch"
                )
            frames = rows["frame"].to_numpy(dtype=np.int64)
            if len(frames) > 1 and not np.all(np.diff(frames) == 1):
                raise ValueError(
                    f"example {example_id}, tracklet {tracklet.tracklet_index}: "
                    "frames are not consecutive"
                )
            if int(frames[0]) != int(tracklet.start_frame) or int(
                frames[-1]
            ) != int(tracklet.end_frame):
                raise ValueError(
                    f"example {example_id}, tracklet {tracklet.tracklet_index}: "
                    "start/end frame mismatch"
                )

        source_degree = edges.groupby("source_tracklet_index").size()
        target_degree = edges.groupby("target_tracklet_index").size()
        if int(source_degree.min()) < 2:
            warnings.append(
                f"example {example_id}: at least one source has no wrong alternative"
            )
        if int(target_degree.min()) < 2:
            warnings.append(
                f"example {example_id}: at least one target has no wrong predecessor"
            )

    return warnings


def build_audit(data: Dataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ex in data.examples.sort_values("example_id").itertuples(index=False):
        example_id = int(ex.example_id)
        edges = data.edges[data.edges["example_id"] == example_id].copy()
        pos = edges[edges["continuation_target"].astype(int) == 1]
        neg = edges[edges["continuation_target"].astype(int) == 0]

        source_degrees = edges.groupby("source_tracklet_index").size()
        target_degrees = edges.groupby("target_tracklet_index").size()

        per_source_margin: list[float] = []
        wrong_closer_sources = 0
        for _, group in edges.groupby("source_tracklet_index"):
            p = group[group["continuation_target"].astype(int) == 1]
            n = group[group["continuation_target"].astype(int) == 0]
            if len(p) != 1 or n.empty:
                continue
            pdist = float(p["direct_distance_um"].iloc[0])
            ndist = float(n["direct_distance_um"].min())
            per_source_margin.append(ndist - pdist)
            wrong_closer_sources += int(ndist < pdist)

        rows.append(
            {
                "example_id": example_id,
                "cut_frame": int(ex.cut_frame),
                "tracks": int(ex.track_count),
                "edges": int(len(edges)),
                "positives": int(len(pos)),
                "negatives": int(len(neg)),
                "edge_density": float(
                    len(edges) / max(int(ex.track_count) ** 2, 1)
                ),
                "min_source_degree": int(source_degrees.min()),
                "max_source_degree": int(source_degrees.max()),
                "min_target_degree": int(target_degrees.min()),
                "max_target_degree": int(target_degrees.max()),
                "mean_positive_distance_um": float(
                    pos["direct_distance_um"].mean()
                ),
                "nearest_negative_distance_um": float(
                    neg["direct_distance_um"].min()
                ),
                "median_wrong_minus_true_um": float(
                    np.median(per_source_margin)
                ),
                "minimum_wrong_minus_true_um": float(
                    np.min(per_source_margin)
                ),
                "wrong_closer_sources": int(wrong_closer_sources),
                "positive_source_rank_gt1": int(
                    (pos["source_distance_rank"].astype(int) > 1).sum()
                ),
                "positive_target_rank_gt1": int(
                    (pos["target_distance_rank"].astype(int) > 1).sum()
                ),
            }
        )

    audit = pd.DataFrame(rows)
    # Hardest first: wrong closer, then smallest geometric margin.
    return audit.sort_values(
        ["wrong_closer_sources", "minimum_wrong_minus_true_um"],
        ascending=[False, True],
    ).reset_index(drop=True)


def print_report(data: Dataset, audit: pd.DataFrame, *, show_all: bool) -> None:
    stats = data.manifest.get("stats", {})
    parameters = data.manifest.get("parameters", {})

    print("=" * 118)
    print("TRACK RECONCILER — INVESTIGATION 02: OVERFIT DATASET AUDIT")
    print("=" * 118)
    print(f"dataset                 : {data.root}")
    print(f"sample                  : {data.manifest.get('sample_id')}")
    print(f"examples                : {len(data.examples)}")
    print(f"synthetic tracklets      : {len(data.tracklets)}")
    print(f"observations             : {len(data.observations)}")
    print(
        f"candidate edges          : {len(data.edges)} "
        f"({int((data.edges.continuation_target == 1).sum())} positive / "
        f"{int((data.edges.continuation_target == 0).sum())} negative)"
    )
    print(
        f"tracks/example           : "
        f"{parameters.get('tracks_per_example', '?')}"
    )
    print(
        f"window                   : "
        f"{parameters.get('pre_frames', '?')} left + "
        f"{parameters.get('post_frames', '?')} right"
    )
    print(
        f"candidate radius         : "
        f"{parameters.get('candidate_radius_um', '?')} um"
    )
    print("-" * 118)

    wrong_closer = int(audit["wrong_closer_sources"].sum())
    source_rank_hard = int(audit["positive_source_rank_gt1"].sum())
    target_rank_hard = int(audit["positive_target_rank_gt1"].sum())

    print(
        f"wrong candidate closer   : {wrong_closer} source decisions "
        f"across {int((audit.wrong_closer_sources > 0).sum())} examples"
    )
    print(
        f"positive source rank > 1 : {source_rank_hard}"
    )
    print(
        f"positive target rank > 1 : {target_rank_hard}"
    )
    print(
        "wrong-vs-true margin     : "
        f"median={audit['median_wrong_minus_true_um'].median():.3f} um, "
        f"minimum={audit['minimum_wrong_minus_true_um'].min():.3f} um"
    )
    print(
        "candidate density        : "
        f"median={audit['edge_density'].median():.3f}, "
        f"range=[{audit['edge_density'].min():.3f}, "
        f"{audit['edge_density'].max():.3f}]"
    )
    print("=" * 118)

    columns = [
        "example_id",
        "cut_frame",
        "edges",
        "negatives",
        "edge_density",
        "min_source_degree",
        "min_target_degree",
        "mean_positive_distance_um",
        "nearest_negative_distance_um",
        "minimum_wrong_minus_true_um",
        "wrong_closer_sources",
    ]
    shown = audit if show_all else audit.head(min(12, len(audit)))
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 180,
        "display.float_format", lambda value: f"{value:.3f}",
    ):
        print(shown[columns].to_string(index=False))

    print("")
    if wrong_closer == 0 and source_rank_hard == 0 and target_rank_hard == 0:
        print(
            "[interpretation] Every true edge is currently the nearest geometric "
            "candidate from both source and target sides."
        )
        print(
            "[interpretation] This is still valid for pipeline memorization, "
            "but a successful overfit does NOT yet prove that the edge reasoner "
            "beats nearest-neighbour geometry."
        )
    else:
        print(
            "[interpretation] The dataset contains cases where pure nearest "
            "geometry is insufficient; these are the most informative examples."
        )

    print(
        "[viewer] Hardest example by default: "
        f"{int(audit.iloc[0].example_id)}"
    )


def _source_path_from_manifest(data: Dataset, key: str) -> Path:
    raw = data.manifest.get("source", {}).get(key)
    if raw is None:
        raise KeyError(f"manifest source does not contain {key!r}")
    return resolve(raw)


def open_viewer(data: Dataset, *, example_id: int) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "Napari is required for --viewer. Activate the repository "
            "visualization environment."
        ) from exc

    examples = data.examples.set_index("example_id")
    if example_id not in examples.index:
        raise ValueError(
            f"Unknown example {example_id}; valid range is "
            f"{sorted(examples.index.astype(int).tolist())}"
        )

    spacing = tuple(
        float(v) for v in data.manifest["spacing_zyx_um"]
    )
    scale = (1.0, *spacing)

    raw_path = _source_path_from_manifest(data, "raw")
    labels_path = _source_path_from_manifest(data, "final_instances")
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)

    ex = examples.loc[example_id]
    obs = data.observations[
        data.observations["example_id"].astype(int) == int(example_id)
    ].copy()
    tracklets = data.tracklets[
        data.tracklets["example_id"].astype(int) == int(example_id)
    ].copy()
    edges = data.edges[
        data.edges["example_id"].astype(int) == int(example_id)
    ].copy()

    cut = int(ex.cut_frame)

    viewer = napari.Viewer(ndisplay=3)
    low, high = np.percentile(
        np.asarray(raw[max(0, cut - 1): min(raw.shape[0], cut + 3)]),
        [1.0, 99.8],
    )
    viewer.add_image(
        raw,
        name="Raw Volume",
        scale=scale,
        rendering="mip",
        colormap="gray",
        contrast_limits=(float(low), float(high)),
    )
    viewer.add_labels(
        labels,
        name="Spatial Final Instances",
        scale=scale,
        opacity=0.45,
        visible=True,
    )

    source_rows = obs[obs["role"] == "source"].copy()
    target_rows = obs[obs["role"] == "target"].copy()

    def tracks_array(frame: pd.DataFrame, offset: int) -> np.ndarray:
        rows = []
        for tracklet_index, group in frame.groupby("tracklet_index"):
            for row in group.sort_values("frame").itertuples(index=False):
                rows.append(
                    [
                        int(tracklet_index) + offset,
                        int(row.frame),
                        float(row.z),
                        float(row.y),
                        float(row.x),
                    ]
                )
        return np.asarray(rows, dtype=np.float64)

    source_array = tracks_array(source_rows, 0)
    target_array = tracks_array(target_rows, 1000)
    viewer.add_tracks(
        source_array,
        name="Synthetic source tracklets A",
        scale=scale,
        tail_length=20,
    )
    viewer.add_tracks(
        target_array,
        name="Synthetic target tracklets B",
        scale=scale,
        tail_length=20,
    )

    # Endpoint lookup in voxel coordinates.
    endpoint: dict[int, tuple[float, float, float, float]] = {}
    for tracklet_index, group in obs.groupby("tracklet_index"):
        role = str(group["role"].iloc[0])
        row = (
            group.sort_values("frame").iloc[-1]
            if role == "source"
            else group.sort_values("frame").iloc[0]
        )
        endpoint[int(tracklet_index)] = (
            float(row.frame),
            float(row.z),
            float(row.y),
            float(row.x),
        )

    positive_rows: list[list[float]] = []
    negative_rows: list[list[float]] = []
    positive_id = 0
    negative_id = 0
    for edge in edges.itertuples(index=False):
        src = endpoint[int(edge.source_tracklet_index)]
        dst = endpoint[int(edge.target_tracklet_index)]
        if int(edge.continuation_target) == 1:
            positive_rows.extend(
                [
                    [positive_id, *src],
                    [positive_id, *dst],
                ]
            )
            positive_id += 1
        else:
            negative_rows.extend(
                [
                    [negative_id, *src],
                    [negative_id, *dst],
                ]
            )
            negative_id += 1

    if positive_rows:
        viewer.add_tracks(
            np.asarray(positive_rows, dtype=np.float64),
            name="TRUE continuation edges",
            scale=scale,
            tail_length=2,
        )
    if negative_rows:
        layer = viewer.add_tracks(
            np.asarray(negative_rows, dtype=np.float64),
            name="WRONG candidate edges",
            scale=scale,
            tail_length=2,
        )
        layer.visible = True

    # Label source A_i / target B_i at their cut-side endpoints.
    point_rows = []
    point_labels = []
    for row in tracklets.sort_values("tracklet_index").itertuples(index=False):
        point_rows.append(endpoint[int(row.tracklet_index)])
        prefix = "A" if row.role == "source" else "B"
        point_labels.append(
            f"{prefix}{int(row.identity_index)} "
            f"(track {int(row.original_track_id)})"
        )
    points = viewer.add_points(
        np.asarray(point_rows, dtype=np.float64),
        name="Synthetic endpoints",
        scale=scale,
        size=5,
        properties={"label": np.asarray(point_labels, dtype=object)},
        text={
            "string": "{label}",
            "size": 8,
            "color": "white",
            "anchor": "center",
        },
    )

    # Expected positions from each source.  The values are physical; convert
    # to voxel coordinates because the layer scale supplies physical spacing.
    prediction_rows = []
    prediction_labels = []
    unique_source_edges = edges.sort_values("edge_index").drop_duplicates(
        "source_tracklet_index"
    )
    spacing_np = np.asarray(spacing, dtype=np.float64)
    for edge in unique_source_edges.itertuples(index=False):
        for label, prefix in (
            ("global", "expected_global"),
            ("global+relative", "expected_global_relative"),
        ):
            values = np.asarray(
                [
                    getattr(edge, f"{prefix}_z_um"),
                    getattr(edge, f"{prefix}_y_um"),
                    getattr(edge, f"{prefix}_x_um"),
                ],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                continue
            voxel = values / spacing_np
            prediction_rows.append(
                [float(cut + 1), float(voxel[0]), float(voxel[1]), float(voxel[2])]
            )
            prediction_labels.append(
                f"A{int(edge.source_identity_index)} {label}"
            )

    if prediction_rows:
        layer = viewer.add_points(
            np.asarray(prediction_rows, dtype=np.float64),
            name="Motion predictions at t+1",
            scale=scale,
            size=3,
            properties={"label": np.asarray(prediction_labels, dtype=object)},
            text={
                "string": "{label}",
                "size": 7,
                "color": "yellow",
                "anchor": "center",
            },
        )

    viewer.dims.set_current_step(0, cut)
    print(
        f"[viewer] example={example_id}, cut={cut}->{cut + 1}, "
        f"tracks={int(ex.track_count)}, edges={len(edges)}"
    )
    print(
        "[viewer] Toggle TRUE/WRONG edge layers and step between t and t+1."
    )
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and optionally visualize the first track-reconciler overfit set."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--example", type=int, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = (
        resolve(args.dataset)
        if args.dataset is not None
        else default_dataset(args.sample_id)
    )
    data = Dataset.load(dataset_root)
    warnings = validate_dataset(data)
    audit = build_audit(data)
    print_report(data, audit, show_all=bool(args.show_all))

    for warning in warnings:
        print(f"[warning] {warning}")

    if args.viewer:
        example_id = (
            int(args.example)
            if args.example is not None
            else int(audit.iloc[0].example_id)
        )
        open_viewer(data, example_id=example_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
