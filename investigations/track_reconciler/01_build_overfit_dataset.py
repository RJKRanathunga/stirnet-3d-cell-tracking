from __future__ import annotations

r"""
Investigation 01 — build a deliberately ambiguous track-reconciler overfit set
from the cached BioHub STIR-Net + Trackastra movie produced by Investigation 36.

Scientific objective
--------------------
The first learned-reconciliation overfit should *not* contain one isolated
broken track at a time.  That setup permits a trivial shortcut: every visible
break should be reconnected.

Instead this builder searches for several nearby Trackastra tracklets that:

1. are branch-free (not involved in a Trackastra parent/child relation),
2. have exactly one observation in every frame of a requested temporal window,
3. remain present for guard frames outside the synthetic window,
4. have stable spatial-instance matches, motion and volume,
5. lie close enough to create plausible cross-track alternatives.

For one selected cut between frames t and t+1, every selected original track is
artificially split simultaneously:

    original identity i:
        source tracklet A_i = [t-pre+1, ..., t]
        target tracklet B_i = [t+1, ..., t+post]

The candidate graph then contains the true A_i -> B_i edge plus nearby
cross-identity A_i -> B_j alternatives.  Labels therefore encode a local
one-to-one assignment problem rather than "always merge the only two pieces".

No STIR-Net or Trackastra inference is rerun.  The script reuses:

    runs/stirnet/evaluation/
      36_biohub_spatial_trackastra_visualization/<sample_id>/
        movies/raw.npy
        movies/preprocessed.npy
        movies/final_instances.npy
        cells_all.csv
        trackastra/tracks.csv
        trackastra/napari_graph.json

Output contract
---------------
    runs/track_reconciler/investigations/
      01_build_overfit_dataset/<sample_id>/
        manifest.json
        examples.csv
        tracklets.csv
        observations.csv
        edges.csv
        global_motion.csv

`original_track_id` and `cell_id` are supervision/provenance only.  They MUST
NOT be fed to the learned model as features.

Typical run
-----------
From repository root:

    python .\investigations\track_reconciler\01_build_overfit_dataset.py

A stricter/harder set:

    python .\investigations\track_reconciler\01_build_overfit_dataset.py ^
        --examples 32 ^
        --tracks-per-example 4 ^
        --pre-frames 4 ^
        --post-frames 4 ^
        --candidate-radius-um 12 ^
        --group-radius-um 20 ^
        --min-wrong-candidates-per-source 1 ^
        --overwrite

The default parameters intentionally target the tiny first overfit, not the
final training distribution.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_NAME = "01_build_overfit_dataset"
DEFAULT_SAMPLE_ID = "44b6_0113de3b"
DEFAULT_SPACING_ZYX_UM = (1.625, 0.40625, 0.40625)

DEFAULT_EXAMPLES = 32
DEFAULT_TRACKS_PER_EXAMPLE = 4
DEFAULT_PRE_FRAMES = 4
DEFAULT_POST_FRAMES = 4
DEFAULT_GUARD_FRAMES = 1

# Gap-1 direct candidate radius matches the learned reconciler's default
# CandidateConfig.radius_base_um.  Positives are always retained even if they
# marginally exceed this radius; quality filters normally keep them inside it.
DEFAULT_CANDIDATE_RADIUS_UM = 12.0
DEFAULT_GROUP_RADIUS_UM = 20.0
DEFAULT_MIN_WRONG_PER_SOURCE = 1
DEFAULT_MIN_WRONG_PER_TARGET = 1

DEFAULT_MAX_CELL_MATCH_UM = 2.5
DEFAULT_MAX_STEP_UM = 12.0
DEFAULT_MAX_VOLUME_RATIO = 2.0
DEFAULT_MIN_BOUNDARY_DISTANCE_UM = 5.0

DEFAULT_MAX_TRACK_REUSE = 3
DEFAULT_MAX_EXAMPLES_PER_CUT = 6
DEFAULT_SEED = 1337


# =============================================================================
# Repository / I/O
# =============================================================================


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


def default_source(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "stirnet"
        / "evaluation"
        / "36_biohub_spatial_trackastra_visualization"
        / sample_id
    ).resolve()


def default_output(sample_id: str) -> Path:
    return (
        ROOT
        / "runs"
        / "track_reconciler"
        / "investigations"
        / SCRIPT_NAME
        / sample_id
    ).resolve()


def parse_triplet(text: str, *, name: str) -> tuple[float, float, float]:
    values = tuple(float(token.strip()) for token in str(text).split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values")
    if not all(np.isfinite(values)) or not all(value > 0 for value in values):
        raise ValueError(f"{name} must contain three positive finite values")
    return values  # type: ignore[return-value]


def root_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
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


@dataclass(frozen=True)
class SourcePaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "movies" / "raw.npy"

    @property
    def preprocessed(self) -> Path:
        return self.root / "movies" / "preprocessed.npy"

    @property
    def final_instances(self) -> Path:
        return self.root / "movies" / "final_instances.npy"

    @property
    def cells(self) -> Path:
        return self.root / "cells_all.csv"

    @property
    def tracks(self) -> Path:
        return self.root / "trackastra" / "tracks.csv"

    @property
    def napari_graph(self) -> Path:
        return self.root / "trackastra" / "napari_graph.json"

    @property
    def spatial_summary(self) -> Path:
        return self.root / "spatial_summary.json"

    @property
    def trackastra_summary(self) -> Path:
        return self.root / "trackastra" / "summary.json"

    def validate(self) -> None:
        required = (
            self.raw,
            self.preprocessed,
            self.final_instances,
            self.cells,
            self.tracks,
            self.napari_graph,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            lines = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Investigation-36 cache is incomplete. Missing:\n" + lines
            )


# =============================================================================
# Source normalization
# =============================================================================


REQUIRED_TRACK_COLUMNS = {
    "track_id",
    "frame",
    "z",
    "y",
    "x",
    "cell_id",
}
REQUIRED_CELL_COLUMNS = {
    "frame",
    "cell_id",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "volume",
    "intensity_mean",
}


def load_lineage_track_ids(path: Path) -> set[int]:
    """Trackastra/Napari graph is child -> parent (or parent list)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    lineage_ids: set[int] = set()
    for child, parent in payload.items():
        lineage_ids.add(int(child))
        if isinstance(parent, (list, tuple)):
            lineage_ids.update(int(value) for value in parent)
        elif parent is not None:
            lineage_ids.add(int(parent))
    return lineage_ids


def load_observations(
    source: SourcePaths,
    *,
    spacing: tuple[float, float, float],
) -> tuple[pd.DataFrame, set[int], tuple[int, int, int, int]]:
    tracks = pd.read_csv(source.tracks)
    cells = pd.read_csv(source.cells)

    missing_tracks = REQUIRED_TRACK_COLUMNS.difference(tracks.columns)
    if missing_tracks:
        raise ValueError(
            f"{source.tracks} is missing columns: {sorted(missing_tracks)}"
        )
    missing_cells = REQUIRED_CELL_COLUMNS.difference(cells.columns)
    if missing_cells:
        raise ValueError(
            f"{source.cells} is missing columns: {sorted(missing_cells)}"
        )

    tracks = tracks.copy()
    tracks["track_id"] = pd.to_numeric(
        tracks["track_id"], errors="raise"
    ).astype(np.int64)
    tracks["frame"] = pd.to_numeric(
        tracks["frame"], errors="raise"
    ).astype(np.int64)
    tracks["cell_id"] = pd.to_numeric(
        tracks["cell_id"], errors="coerce"
    ).fillna(-1).astype(np.int64)

    duplicated = tracks.duplicated(["track_id", "frame"], keep=False)
    if duplicated.any():
        duplicate_rows = tracks.loc[
            duplicated, ["track_id", "frame"]
        ].drop_duplicates()
        raise ValueError(
            "Trackastra tracks contain duplicate observations for the same "
            f"(track_id, frame):\n{duplicate_rows.head(20)}"
        )

    cells = cells.copy()
    cells["frame"] = pd.to_numeric(
        cells["frame"], errors="raise"
    ).astype(np.int64)
    cells["cell_id"] = pd.to_numeric(
        cells["cell_id"], errors="raise"
    ).astype(np.int64)

    if cells.duplicated(["frame", "cell_id"]).any():
        raise ValueError("cells_all.csv must be unique by (frame, cell_id)")

    # Prefix cell-derived attributes that collide with Trackastra columns.
    cell_payload = cells.copy()
    obs = tracks.merge(
        cell_payload,
        on=["frame", "cell_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_cell"),
    )

    sz, sy, sx = spacing
    obs["z_um"] = pd.to_numeric(obs["z"], errors="coerce") * sz
    obs["y_um"] = pd.to_numeric(obs["y"], errors="coerce") * sy
    obs["x_um"] = pd.to_numeric(obs["x"], errors="coerce") * sx

    obs["centroid_z_um"] = (
        pd.to_numeric(obs["centroid_z"], errors="coerce") * sz
    )
    obs["centroid_y_um"] = (
        pd.to_numeric(obs["centroid_y"], errors="coerce") * sy
    )
    obs["centroid_x_um"] = (
        pd.to_numeric(obs["centroid_x"], errors="coerce") * sx
    )

    delta = obs[
        ["z_um", "y_um", "x_um"]
    ].to_numpy(dtype=np.float64) - obs[
        ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
    ].to_numpy(dtype=np.float64)
    obs["cell_match_error_um"] = np.linalg.norm(delta, axis=1)

    raw = np.load(source.raw, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 4:
        raise ValueError(f"Expected raw movie [T,Z,Y,X], got {raw.shape}")
    movie_shape = tuple(int(value) for value in raw.shape)
    del raw

    spatial_shape = np.asarray(movie_shape[-3:], dtype=np.float64)
    spacing_np = np.asarray(spacing, dtype=np.float64)
    maximum_xyz = (spatial_shape - 1.0) * spacing_np
    xyz = obs[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    lower = xyz
    upper = maximum_xyz[None, :] - xyz
    obs["distance_to_boundary_um"] = np.minimum(lower, upper).min(axis=1)

    lineage_ids = load_lineage_track_ids(source.napari_graph)
    obs["lineage_related"] = obs["track_id"].isin(lineage_ids)

    return obs.sort_values(["track_id", "frame"]).reset_index(drop=True), lineage_ids, movie_shape


# =============================================================================
# Global motion from persistent Trackastra tracks
# =============================================================================


def estimate_global_motion(
    observations: pd.DataFrame,
    *,
    lineage_ids: set[int],
) -> pd.DataFrame:
    """
    Robust frame-to-frame global translation estimate.

    This is intentionally computed from all branch-free Trackastra tracks, not
    only the tracks later selected for synthetic cuts.  It becomes useful
    source-side evidence for the reconciler's explicit global-motion expert.
    """
    branch_free = observations[
        ~observations["track_id"].isin(lineage_ids)
    ][["track_id", "frame", "z_um", "y_um", "x_um"]].copy()

    rows: list[dict[str, Any]] = []
    if branch_free.empty:
        return pd.DataFrame(
            columns=[
                "frame_from",
                "frame_to",
                "shift_z_um",
                "shift_y_um",
                "shift_x_um",
                "support_tracks",
                "residual_mad_um",
            ]
        )

    by_frame = {
        int(frame): frame_df.set_index("track_id")
        for frame, frame_df in branch_free.groupby("frame")
    }
    frames = sorted(by_frame)
    for frame in frames:
        next_frame = frame + 1
        if next_frame not in by_frame:
            continue
        left = by_frame[frame]
        right = by_frame[next_frame]
        common = left.index.intersection(right.index)
        if len(common) == 0:
            continue

        a = left.loc[common, ["z_um", "y_um", "x_um"]].to_numpy(
            dtype=np.float64
        )
        b = right.loc[common, ["z_um", "y_um", "x_um"]].to_numpy(
            dtype=np.float64
        )
        displacement = b - a
        shift = np.median(displacement, axis=0)
        residual_norm = np.linalg.norm(displacement - shift[None, :], axis=1)
        residual_mad = float(np.median(np.abs(
            residual_norm - np.median(residual_norm)
        )))
        rows.append(
            {
                "frame_from": int(frame),
                "frame_to": int(next_frame),
                "shift_z_um": float(shift[0]),
                "shift_y_um": float(shift[1]),
                "shift_x_um": float(shift[2]),
                "support_tracks": int(len(common)),
                "residual_mad_um": residual_mad,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Clean persistent windows
# =============================================================================


@dataclass(frozen=True)
class QualityConfig:
    pre_frames: int
    post_frames: int
    guard_frames: int
    max_cell_match_um: float
    max_step_um: float
    max_volume_ratio: float
    min_boundary_distance_um: float


@dataclass(frozen=True)
class CleanWindow:
    track_id: int
    cut_frame: int
    core_start: int
    core_end: int
    source_end_xyz_um: tuple[float, float, float]
    target_start_xyz_um: tuple[float, float, float]
    midpoint_xyz_um: tuple[float, float, float]
    positive_distance_um: float
    max_step_um: float
    max_volume_ratio: float
    max_match_error_um: float
    min_boundary_distance_um: float


def _consecutive_ratio_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 1.0
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        return float("inf")
    ratios = np.maximum(
        values[1:] / values[:-1],
        values[:-1] / values[1:],
    )
    return float(np.max(ratios))


def clean_window(
    track: pd.DataFrame,
    *,
    cut_frame: int,
    cfg: QualityConfig,
) -> CleanWindow | None:
    core_start = cut_frame - cfg.pre_frames + 1
    core_end = cut_frame + cfg.post_frames
    required_start = core_start - cfg.guard_frames
    required_end = core_end + cfg.guard_frames

    required_frames = np.arange(required_start, required_end + 1, dtype=np.int64)
    indexed = track.set_index("frame", drop=False)
    if not np.isin(required_frames, indexed.index.to_numpy()).all():
        return None

    window = indexed.loc[required_frames].copy()
    if len(window) != len(required_frames):
        return None

    # No division/birth/termination-adjacent Trackastra tracklets.
    if bool(window["lineage_related"].any()):
        return None

    numeric_required = (
        "z_um",
        "y_um",
        "x_um",
        "volume",
        "cell_match_error_um",
        "distance_to_boundary_um",
    )
    numeric = window[list(numeric_required)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        return None
    if (window["cell_id"].to_numpy(dtype=np.int64) < 0).any():
        return None

    match_max = float(window["cell_match_error_um"].max())
    if match_max > cfg.max_cell_match_um:
        return None

    boundary_min = float(window["distance_to_boundary_um"].min())
    if boundary_min < cfg.min_boundary_distance_um:
        return None

    xyz = window[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    max_step = float(steps.max()) if steps.size else 0.0
    if max_step > cfg.max_step_um:
        return None

    volume_ratio = _consecutive_ratio_max(
        window["volume"].to_numpy(dtype=np.float64)
    )
    if volume_ratio > cfg.max_volume_ratio:
        return None

    source = indexed.loc[
        cut_frame, ["z_um", "y_um", "x_um"]
    ].to_numpy(dtype=np.float64)
    target = indexed.loc[
        cut_frame + 1, ["z_um", "y_um", "x_um"]
    ].to_numpy(dtype=np.float64)

    return CleanWindow(
        track_id=int(track["track_id"].iloc[0]),
        cut_frame=int(cut_frame),
        core_start=int(core_start),
        core_end=int(core_end),
        source_end_xyz_um=tuple(float(v) for v in source),
        target_start_xyz_um=tuple(float(v) for v in target),
        midpoint_xyz_um=tuple(float(v) for v in 0.5 * (source + target)),
        positive_distance_um=float(np.linalg.norm(target - source)),
        max_step_um=max_step,
        max_volume_ratio=float(volume_ratio),
        max_match_error_um=match_max,
        min_boundary_distance_um=boundary_min,
    )


def enumerate_clean_windows(
    observations: pd.DataFrame,
    *,
    movie_frames: int,
    cfg: QualityConfig,
) -> dict[int, list[CleanWindow]]:
    by_track = {
        int(track_id): frame.sort_values("frame")
        for track_id, frame in observations.groupby("track_id", sort=False)
    }

    minimum_cut = cfg.pre_frames - 1 + cfg.guard_frames
    maximum_cut = (
        movie_frames - cfg.post_frames - cfg.guard_frames - 1
    )
    if maximum_cut < minimum_cut:
        raise ValueError(
            "Temporal window is larger than the available movie: "
            f"T={movie_frames}, pre={cfg.pre_frames}, post={cfg.post_frames}, "
            f"guard={cfg.guard_frames}"
        )

    windows: dict[int, list[CleanWindow]] = {}
    for cut_frame in range(minimum_cut, maximum_cut + 1):
        current: list[CleanWindow] = []
        for track in by_track.values():
            candidate = clean_window(track, cut_frame=cut_frame, cfg=cfg)
            if candidate is not None:
                current.append(candidate)
        if current:
            windows[int(cut_frame)] = current
    return windows


# =============================================================================
# Local ambiguous groups
# =============================================================================


@dataclass(frozen=True)
class GroupCandidate:
    cut_frame: int
    track_ids: tuple[int, ...]
    true_distances_um: tuple[float, ...]
    midpoint_diameter_um: float
    negative_min_distance_um: float
    median_wrong_minus_true_um: float
    wrong_closer_count: int
    candidate_edge_count: int


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def make_group_candidate(
    windows_by_id: dict[int, CleanWindow],
    track_ids: Iterable[int],
    *,
    candidate_radius_um: float,
    min_wrong_per_source: int,
    min_wrong_per_target: int,
) -> GroupCandidate | None:
    ids = tuple(sorted(int(value) for value in track_ids))
    if len(ids) < 2:
        return None

    windows = [windows_by_id[value] for value in ids]
    source = np.asarray(
        [window.source_end_xyz_um for window in windows],
        dtype=np.float64,
    )
    target = np.asarray(
        [window.target_start_xyz_um for window in windows],
        dtype=np.float64,
    )
    midpoint = np.asarray(
        [window.midpoint_xyz_um for window in windows],
        dtype=np.float64,
    )

    direct = _pairwise_distances(source, target)
    true = np.diag(direct)

    # First overfit should not depend on a long-gap rescue mechanism.
    if np.any(true > candidate_radius_um):
        return None

    candidate = direct <= candidate_radius_um
    np.fill_diagonal(candidate, True)

    wrong = candidate.copy()
    np.fill_diagonal(wrong, False)
    if np.any(wrong.sum(axis=1) < min_wrong_per_source):
        return None
    if np.any(wrong.sum(axis=0) < min_wrong_per_target):
        return None

    margins: list[float] = []
    wrong_closer = 0
    wrong_distances: list[float] = []
    for row in range(len(ids)):
        values = direct[row][wrong[row]]
        if values.size == 0:
            return None
        nearest_wrong = float(values.min())
        margins.append(nearest_wrong - float(true[row]))
        wrong_closer += int(nearest_wrong < float(true[row]))
        wrong_distances.extend(float(value) for value in values)

    midpoint_pair = _pairwise_distances(midpoint, midpoint)
    diameter = float(midpoint_pair.max()) if midpoint_pair.size else 0.0

    return GroupCandidate(
        cut_frame=int(windows[0].cut_frame),
        track_ids=ids,
        true_distances_um=tuple(float(value) for value in true),
        midpoint_diameter_um=diameter,
        negative_min_distance_um=float(min(wrong_distances)),
        median_wrong_minus_true_um=float(np.median(margins)),
        wrong_closer_count=int(wrong_closer),
        candidate_edge_count=int(candidate.sum()),
    )


def enumerate_group_candidates(
    clean: dict[int, list[CleanWindow]],
    *,
    tracks_per_example: int,
    group_radius_um: float,
    candidate_radius_um: float,
    min_wrong_per_source: int,
    min_wrong_per_target: int,
) -> list[GroupCandidate]:
    """
    Generate one nearest-neighbour group per anchor, then de-duplicate.

    `group_radius_um` limits the complete local scene.  The tighter
    `candidate_radius_um` determines actual A->B candidate edges.
    """
    groups: dict[tuple[int, tuple[int, ...]], GroupCandidate] = {}

    for cut_frame, windows in clean.items():
        if len(windows) < tracks_per_example:
            continue

        windows_by_id = {window.track_id: window for window in windows}
        ids = np.asarray(sorted(windows_by_id), dtype=np.int64)
        midpoint = np.asarray(
            [windows_by_id[int(track_id)].midpoint_xyz_um for track_id in ids],
            dtype=np.float64,
        )
        pairwise = _pairwise_distances(midpoint, midpoint)

        for anchor_index, _anchor_id in enumerate(ids):
            order = np.argsort(pairwise[anchor_index])
            nearby = [
                int(ids[index])
                for index in order
                if pairwise[anchor_index, index] <= group_radius_um
            ]
            if len(nearby) < tracks_per_example:
                continue

            chosen = tuple(sorted(nearby[:tracks_per_example]))

            # Keep the full group reasonably compact, not merely star-shaped.
            chosen_idx = [int(np.where(ids == value)[0][0]) for value in chosen]
            diameter = float(
                pairwise[np.ix_(chosen_idx, chosen_idx)].max()
            )
            if diameter > group_radius_um:
                continue

            candidate = make_group_candidate(
                windows_by_id,
                chosen,
                candidate_radius_um=candidate_radius_um,
                min_wrong_per_source=min_wrong_per_source,
                min_wrong_per_target=min_wrong_per_target,
            )
            if candidate is None:
                continue
            groups[(int(cut_frame), candidate.track_ids)] = candidate

    return list(groups.values())


def select_groups(
    candidates: list[GroupCandidate],
    *,
    examples: int,
    max_track_reuse: int,
    max_examples_per_cut: int,
    seed: int,
) -> list[GroupCandidate]:
    """
    Prefer genuinely ambiguous groups while limiting repeated identities.

    Ordering:
      1. examples where a wrong target is actually closer than the true target,
      2. smaller nearest-wrong minus true-distance margin,
      3. smaller local group diameter,
      4. deterministic random tie-break.
    """
    rng = np.random.default_rng(seed)
    decorated = [
        (
            -candidate.wrong_closer_count,
            candidate.median_wrong_minus_true_um,
            candidate.midpoint_diameter_um,
            float(rng.random()),
            candidate,
        )
        for candidate in candidates
    ]
    decorated.sort(key=lambda item: item[:4])

    track_use: Counter[int] = Counter()
    cut_use: Counter[int] = Counter()
    selected: list[GroupCandidate] = []

    for *_key, candidate in decorated:
        if cut_use[candidate.cut_frame] >= max_examples_per_cut:
            continue
        if any(
            track_use[track_id] >= max_track_reuse
            for track_id in candidate.track_ids
        ):
            continue

        selected.append(candidate)
        cut_use[candidate.cut_frame] += 1
        track_use.update(candidate.track_ids)

        if len(selected) >= examples:
            break

    return selected


# =============================================================================
# Materialize synthetic split examples
# =============================================================================


def _global_shift_lookup(
    global_motion: pd.DataFrame,
) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for row in global_motion.itertuples(index=False):
        lookup[int(row.frame_from)] = np.asarray(
            [row.shift_z_um, row.shift_y_um, row.shift_x_um],
            dtype=np.float64,
        )
    return lookup


def _source_relative_velocity(
    rows: pd.DataFrame,
    *,
    global_shift_by_frame: dict[int, np.ndarray],
) -> tuple[np.ndarray, int]:
    """
    Mean recent velocity after subtracting global drift.

    Returns a 3-vector in um/frame and the number of valid step samples.
    """
    rows = rows.sort_values("frame")
    xyz = rows[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    frames = rows["frame"].to_numpy(dtype=np.int64)
    residuals: list[np.ndarray] = []
    for index in range(len(rows) - 1):
        if frames[index + 1] != frames[index] + 1:
            continue
        shift = global_shift_by_frame.get(int(frames[index]))
        if shift is None:
            continue
        residuals.append((xyz[index + 1] - xyz[index]) - shift)
    if not residuals:
        return np.zeros(3, dtype=np.float64), 0
    # Recent steps matter most but a short mean is stable for the tiny overfit.
    recent = np.asarray(residuals[-3:], dtype=np.float64)
    return recent.mean(axis=0), int(len(recent))


def materialize(
    selected: list[GroupCandidate],
    observations: pd.DataFrame,
    clean: dict[int, list[CleanWindow]],
    global_motion: pd.DataFrame,
    *,
    pre_frames: int,
    post_frames: int,
    candidate_radius_um: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_track = {
        int(track_id): frame.sort_values("frame")
        for track_id, frame in observations.groupby("track_id", sort=False)
    }
    clean_lookup = {
        cut: {window.track_id: window for window in windows}
        for cut, windows in clean.items()
    }
    global_shift = _global_shift_lookup(global_motion)

    example_rows: list[dict[str, Any]] = []
    tracklet_rows: list[dict[str, Any]] = []
    observation_parts: list[pd.DataFrame] = []
    edge_rows: list[dict[str, Any]] = []

    for example_id, group in enumerate(selected):
        cut = int(group.cut_frame)
        ids = list(group.track_ids)
        k = len(ids)
        core_start = cut - pre_frames + 1
        core_end = cut + post_frames

        source_tracklet_index = {
            track_id: identity_index
            for identity_index, track_id in enumerate(ids)
        }
        target_tracklet_index = {
            track_id: k + identity_index
            for identity_index, track_id in enumerate(ids)
        }

        source_xyz: dict[int, np.ndarray] = {}
        target_xyz: dict[int, np.ndarray] = {}
        relative_velocity: dict[int, np.ndarray] = {}
        relative_velocity_samples: dict[int, int] = {}

        for identity_index, track_id in enumerate(ids):
            track = by_track[track_id]
            left = track[
                (track["frame"] >= core_start)
                & (track["frame"] <= cut)
            ].copy()
            right = track[
                (track["frame"] >= cut + 1)
                & (track["frame"] <= core_end)
            ].copy()

            if len(left) != pre_frames or len(right) != post_frames:
                raise RuntimeError(
                    "Selected clean window disappeared during materialization: "
                    f"example={example_id}, track={track_id}"
                )

            for role, part, tracklet_index, paired_index in (
                (
                    "source",
                    left,
                    source_tracklet_index[track_id],
                    target_tracklet_index[track_id],
                ),
                (
                    "target",
                    right,
                    target_tracklet_index[track_id],
                    source_tracklet_index[track_id],
                ),
            ):
                part = part.copy()
                part.insert(0, "example_id", int(example_id))
                part.insert(1, "tracklet_index", int(tracklet_index))
                part.insert(2, "identity_index", int(identity_index))
                part.insert(3, "role", role)
                part.insert(
                    4,
                    "sequence_index",
                    np.arange(len(part), dtype=np.int64),
                )
                part["original_track_id"] = int(track_id)
                part["paired_tracklet_index"] = int(paired_index)
                observation_parts.append(part)

                tracklet_rows.append(
                    {
                        "example_id": int(example_id),
                        "tracklet_index": int(tracklet_index),
                        "identity_index": int(identity_index),
                        "role": role,
                        "original_track_id": int(track_id),
                        "paired_tracklet_index": int(paired_index),
                        "start_frame": int(part["frame"].min()),
                        "end_frame": int(part["frame"].max()),
                        "observation_count": int(len(part)),
                        "division_prior_target": 0,
                        # Only the synthetic cut-side event is supervised in
                        # the first continuation overfit.
                        "appearance_target": (
                            0 if role == "target" else np.nan
                        ),
                        "termination_target": (
                            0 if role == "source" else np.nan
                        ),
                    }
                )

            source_xyz[track_id] = left.loc[
                left["frame"] == cut, ["z_um", "y_um", "x_um"]
            ].to_numpy(dtype=np.float64)[0]
            target_xyz[track_id] = right.loc[
                right["frame"] == cut + 1, ["z_um", "y_um", "x_um"]
            ].to_numpy(dtype=np.float64)[0]
            rv, samples = _source_relative_velocity(
                left, global_shift_by_frame=global_shift
            )
            relative_velocity[track_id] = rv
            relative_velocity_samples[track_id] = samples

        transition_shift = global_shift.get(cut)
        source_matrix = np.asarray(
            [source_xyz[track_id] for track_id in ids], dtype=np.float64
        )
        target_matrix = np.asarray(
            [target_xyz[track_id] for track_id in ids], dtype=np.float64
        )
        direct = _pairwise_distances(source_matrix, target_matrix)

        candidate_mask = direct <= candidate_radius_um
        np.fill_diagonal(candidate_mask, True)

        edge_id = 0
        source_ranks = np.argsort(np.argsort(direct, axis=1), axis=1) + 1
        target_ranks = np.argsort(np.argsort(direct, axis=0), axis=0) + 1

        for source_identity, source_id in enumerate(ids):
            positive_distance = float(direct[source_identity, source_identity])
            expected_global = (
                source_xyz[source_id] + transition_shift
                if transition_shift is not None
                else np.full(3, np.nan, dtype=np.float64)
            )
            expected_global_relative = (
                expected_global + relative_velocity[source_id]
                if transition_shift is not None
                and relative_velocity_samples[source_id] > 0
                else np.full(3, np.nan, dtype=np.float64)
            )

            for target_identity, target_id in enumerate(ids):
                if not bool(candidate_mask[source_identity, target_identity]):
                    continue

                distance = float(direct[source_identity, target_identity])
                target_position = target_xyz[target_id]
                is_positive = source_id == target_id

                edge_rows.append(
                    {
                        "example_id": int(example_id),
                        "edge_index": int(edge_id),
                        "source_tracklet_index": int(
                            source_tracklet_index[source_id]
                        ),
                        "target_tracklet_index": int(
                            target_tracklet_index[target_id]
                        ),
                        "source_identity_index": int(source_identity),
                        "target_identity_index": int(target_identity),
                        "source_original_track_id": int(source_id),
                        "target_original_track_id": int(target_id),
                        "gap_frames": 1,
                        "continuation_target": int(is_positive),
                        "direct_distance_um": distance,
                        "source_distance_rank": int(
                            source_ranks[source_identity, target_identity]
                        ),
                        "target_distance_rank": int(
                            target_ranks[source_identity, target_identity]
                        ),
                        "positive_direct_distance_um": positive_distance,
                        "wrong_closer_than_positive": int(
                            (not is_positive) and distance < positive_distance
                        ),
                        "global_prediction_valid": int(
                            transition_shift is not None
                        ),
                        "expected_global_z_um": float(expected_global[0]),
                        "expected_global_y_um": float(expected_global[1]),
                        "expected_global_x_um": float(expected_global[2]),
                        "global_error_um": (
                            float(np.linalg.norm(
                                target_position - expected_global
                            ))
                            if transition_shift is not None
                            else np.nan
                        ),
                        "global_relative_prediction_valid": int(
                            transition_shift is not None
                            and relative_velocity_samples[source_id] > 0
                        ),
                        "relative_velocity_samples": int(
                            relative_velocity_samples[source_id]
                        ),
                        "expected_global_relative_z_um": float(
                            expected_global_relative[0]
                        ),
                        "expected_global_relative_y_um": float(
                            expected_global_relative[1]
                        ),
                        "expected_global_relative_x_um": float(
                            expected_global_relative[2]
                        ),
                        "global_relative_error_um": (
                            float(np.linalg.norm(
                                target_position - expected_global_relative
                            ))
                            if transition_shift is not None
                            and relative_velocity_samples[source_id] > 0
                            else np.nan
                        ),
                    }
                )
                edge_id += 1

        example_windows = clean_lookup[cut]
        selected_windows = [example_windows[track_id] for track_id in ids]
        example_rows.append(
            {
                "example_id": int(example_id),
                "cut_frame": cut,
                "source_start_frame": int(core_start),
                "source_end_frame": cut,
                "target_start_frame": cut + 1,
                "target_end_frame": int(core_end),
                "track_count": k,
                "candidate_edge_count": int(candidate_mask.sum()),
                "positive_edge_count": k,
                "negative_edge_count": int(candidate_mask.sum() - k),
                "original_track_ids": json.dumps(ids),
                "midpoint_diameter_um": float(
                    group.midpoint_diameter_um
                ),
                "minimum_negative_distance_um": float(
                    group.negative_min_distance_um
                ),
                "median_wrong_minus_true_um": float(
                    group.median_wrong_minus_true_um
                ),
                "wrong_closer_source_count": int(
                    group.wrong_closer_count
                ),
                "maximum_clean_window_step_um": float(
                    max(window.max_step_um for window in selected_windows)
                ),
                "maximum_clean_window_volume_ratio": float(
                    max(window.max_volume_ratio for window in selected_windows)
                ),
                "maximum_cell_match_error_um": float(
                    max(window.max_match_error_um for window in selected_windows)
                ),
                "minimum_boundary_distance_um": float(
                    min(
                        window.min_boundary_distance_um
                        for window in selected_windows
                    )
                ),
            }
        )

    examples_df = pd.DataFrame(example_rows)
    tracklets_df = pd.DataFrame(tracklet_rows)
    observations_df = (
        pd.concat(observation_parts, ignore_index=True)
        if observation_parts
        else pd.DataFrame()
    )
    edges_df = pd.DataFrame(edge_rows)

    return examples_df, tracklets_df, observations_df, edges_df


# =============================================================================
# CLI / reporting
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tiny ambiguous continuation-overfit dataset by "
            "simultaneously splitting several nearby clean Trackastra tracks."
        )
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Investigation-36 output root. Default: "
            "runs/stirnet/evaluation/36_biohub_spatial_trackastra_visualization/"
            "<sample-id>"
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--spacing",
        default="1.625,0.40625,0.40625",
        help="Physical Z,Y,X voxel spacing in micrometres.",
    )

    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--tracks-per-example",
        type=int,
        default=DEFAULT_TRACKS_PER_EXAMPLE,
    )
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=DEFAULT_PRE_FRAMES,
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=DEFAULT_POST_FRAMES,
    )
    parser.add_argument(
        "--guard-frames",
        type=int,
        default=DEFAULT_GUARD_FRAMES,
        help=(
            "Require the original track to remain continuous this many frames "
            "outside each synthetic observation window."
        ),
    )

    parser.add_argument(
        "--candidate-radius-um",
        type=float,
        default=DEFAULT_CANDIDATE_RADIUS_UM,
        help=(
            "Direct gap-1 A->B radius for cross-track candidates. True edges "
            "are always retained."
        ),
    )
    parser.add_argument(
        "--group-radius-um",
        type=float,
        default=DEFAULT_GROUP_RADIUS_UM,
        help="Maximum physical diameter of the nearby multi-track scene.",
    )
    parser.add_argument(
        "--min-wrong-candidates-per-source",
        type=int,
        default=DEFAULT_MIN_WRONG_PER_SOURCE,
    )
    parser.add_argument(
        "--min-wrong-candidates-per-target",
        type=int,
        default=DEFAULT_MIN_WRONG_PER_TARGET,
    )

    parser.add_argument(
        "--max-cell-match-um",
        type=float,
        default=DEFAULT_MAX_CELL_MATCH_UM,
        help="Reject windows with a poorer Trackastra->spatial-cell match.",
    )
    parser.add_argument(
        "--max-step-um",
        type=float,
        default=DEFAULT_MAX_STEP_UM,
    )
    parser.add_argument(
        "--max-volume-ratio",
        type=float,
        default=DEFAULT_MAX_VOLUME_RATIO,
        help="Maximum adjacent-frame max(v1/v0, v0/v1).",
    )
    parser.add_argument(
        "--min-boundary-distance-um",
        type=float,
        default=DEFAULT_MIN_BOUNDARY_DISTANCE_UM,
    )

    parser.add_argument(
        "--max-track-reuse",
        type=int,
        default=DEFAULT_MAX_TRACK_REUSE,
    )
    parser.add_argument(
        "--max-examples-per-cut",
        type=int,
        default=DEFAULT_MAX_EXAMPLES_PER_CUT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "examples",
        "tracks_per_example",
        "pre_frames",
        "post_frames",
        "max_track_reuse",
        "max_examples_per_cut",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if int(args.tracks_per_example) < 2:
        raise ValueError("--tracks-per-example must be >= 2")
    if int(args.guard_frames) < 0:
        raise ValueError("--guard-frames must be >= 0")
    for name in (
        "candidate_radius_um",
        "group_radius_um",
        "max_cell_match_um",
        "max_step_um",
        "max_volume_ratio",
        "min_boundary_distance_um",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.max_volume_ratio < 1.0:
        raise ValueError("--max-volume-ratio must be >= 1")
    for name in (
        "min_wrong_candidates_per_source",
        "min_wrong_candidates_per_target",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    spacing = parse_triplet(args.spacing, name="spacing")
    source_root = (
        resolve(args.source)
        if args.source is not None
        else default_source(args.sample_id)
    )
    output = (
        resolve(args.output)
        if args.output is not None
        else default_output(args.sample_id)
    )
    source = SourcePaths(source_root)
    source.validate()

    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists and is non-empty: {output}\n"
            "Pass --overwrite to replace this investigation output."
        )
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 112, flush=True)
    print("TRACK RECONCILER — INVESTIGATION 01: AMBIGUOUS SYNTHETIC CUT DATASET", flush=True)
    print("=" * 112, flush=True)
    print(f"sample                 : {args.sample_id}", flush=True)
    print(f"source Investigation 36: {source.root}", flush=True)
    print(f"output                 : {output}", flush=True)
    print(f"spacing ZYX um         : {spacing}", flush=True)
    print(
        f"window                 : {args.pre_frames} left + "
        f"{args.post_frames} right, guard={args.guard_frames}",
        flush=True,
    )
    print(
        f"scene                  : {args.tracks_per_example} tracks/example, "
        f"group <= {args.group_radius_um:g} um",
        flush=True,
    )
    print(
        f"candidate gate         : <= {args.candidate_radius_um:g} um, "
        f"wrong/source >= {args.min_wrong_candidates_per_source}, "
        f"wrong/target >= {args.min_wrong_candidates_per_target}",
        flush=True,
    )
    print("=" * 112, flush=True)

    observations, lineage_ids, movie_shape = load_observations(
        source, spacing=spacing
    )
    print(
        f"[source] rows={len(observations):,} "
        f"tracks={observations['track_id'].nunique():,} "
        f"lineage-related-tracklets={len(lineage_ids):,}",
        flush=True,
    )

    global_motion = estimate_global_motion(
        observations, lineage_ids=lineage_ids
    )
    if global_motion.empty:
        print(
            "[warning] no frame-to-frame global motion estimates were available",
            flush=True,
        )
    else:
        print(
            f"[motion] robust global shifts for {len(global_motion)} transitions; "
            f"median support={global_motion['support_tracks'].median():.0f} tracks",
            flush=True,
        )

    quality = QualityConfig(
        pre_frames=int(args.pre_frames),
        post_frames=int(args.post_frames),
        guard_frames=int(args.guard_frames),
        max_cell_match_um=float(args.max_cell_match_um),
        max_step_um=float(args.max_step_um),
        max_volume_ratio=float(args.max_volume_ratio),
        min_boundary_distance_um=float(args.min_boundary_distance_um),
    )
    clean = enumerate_clean_windows(
        observations,
        movie_frames=movie_shape[0],
        cfg=quality,
    )
    clean_total = sum(len(values) for values in clean.values())
    print(
        f"[clean] valid track-windows={clean_total:,} "
        f"across {len(clean)} cut positions",
        flush=True,
    )

    group_candidates = enumerate_group_candidates(
        clean,
        tracks_per_example=int(args.tracks_per_example),
        group_radius_um=float(args.group_radius_um),
        candidate_radius_um=float(args.candidate_radius_um),
        min_wrong_per_source=int(args.min_wrong_candidates_per_source),
        min_wrong_per_target=int(args.min_wrong_candidates_per_target),
    )
    print(
        f"[groups] ambiguous local group candidates={len(group_candidates):,}",
        flush=True,
    )
    if not group_candidates:
        raise RuntimeError(
            "No ambiguous multi-track groups satisfy the current filters. "
            "The safest first relaxation is usually --group-radius-um 25 or "
            "--candidate-radius-um 16; inspect the reported source data before "
            "loosening quality filters."
        )

    selected = select_groups(
        group_candidates,
        examples=int(args.examples),
        max_track_reuse=int(args.max_track_reuse),
        max_examples_per_cut=int(args.max_examples_per_cut),
        seed=int(args.seed),
    )
    if not selected:
        raise RuntimeError("Selection constraints rejected every candidate group.")

    if len(selected) < int(args.examples):
        print(
            f"[warning] requested {args.examples} examples but only "
            f"{len(selected)} satisfy ambiguity + reuse constraints.",
            flush=True,
        )

    examples_df, tracklets_df, selected_observations, edges_df = materialize(
        selected,
        observations,
        clean,
        global_motion,
        pre_frames=int(args.pre_frames),
        post_frames=int(args.post_frames),
        candidate_radius_um=float(args.candidate_radius_um),
    )

    if edges_df.empty or int(edges_df["continuation_target"].sum()) == 0:
        raise RuntimeError("Materialized dataset contains no positive edges.")
    if not (edges_df["continuation_target"] == 0).any():
        raise RuntimeError("Materialized dataset contains no negative edges.")

    atomic_csv(output / "examples.csv", examples_df)
    atomic_csv(output / "tracklets.csv", tracklets_df)
    atomic_csv(output / "observations.csv", selected_observations)
    atomic_csv(output / "edges.csv", edges_df)
    atomic_csv(output / "global_motion.csv", global_motion)

    positive = edges_df[edges_df["continuation_target"] == 1]
    wrong_closer_positive_count = 0
    for (example_id, source_index), frame in edges_df.groupby(
        ["example_id", "source_tracklet_index"]
    ):
        pos = frame[frame["continuation_target"] == 1]
        neg = frame[frame["continuation_target"] == 0]
        if len(pos) != 1:
            raise RuntimeError(
                f"Expected exactly one positive for example={example_id}, "
                f"source={source_index}; found {len(pos)}"
            )
        if not neg.empty and float(neg["direct_distance_um"].min()) < float(
            pos["direct_distance_um"].iloc[0]
        ):
            wrong_closer_positive_count += 1

    source_refs = {
        "investigation_36_root": root_relative(source.root),
        "raw": root_relative(source.raw),
        "preprocessed": root_relative(source.preprocessed),
        "final_instances": root_relative(source.final_instances),
        "cells": root_relative(source.cells),
        "tracks": root_relative(source.tracks),
        "napari_graph": root_relative(source.napari_graph),
    }
    manifest = {
        "schema_version": 1,
        "investigation": SCRIPT_NAME,
        "purpose": (
            "First continuation overfit: simultaneously cut several nearby "
            "clean persistent Trackastra tracks and recover their one-to-one "
            "identity assignment."
        ),
        "sample_id": str(args.sample_id),
        "source": source_refs,
        "movie_shape_tzyx": list(movie_shape),
        "spacing_zyx_um": list(spacing),
        "parameters": {
            "examples_requested": int(args.examples),
            "tracks_per_example": int(args.tracks_per_example),
            "pre_frames": int(args.pre_frames),
            "post_frames": int(args.post_frames),
            "guard_frames": int(args.guard_frames),
            "candidate_radius_um": float(args.candidate_radius_um),
            "group_radius_um": float(args.group_radius_um),
            "min_wrong_candidates_per_source": int(
                args.min_wrong_candidates_per_source
            ),
            "min_wrong_candidates_per_target": int(
                args.min_wrong_candidates_per_target
            ),
            "max_cell_match_um": float(args.max_cell_match_um),
            "max_step_um": float(args.max_step_um),
            "max_volume_ratio": float(args.max_volume_ratio),
            "min_boundary_distance_um": float(
                args.min_boundary_distance_um
            ),
            "max_track_reuse": int(args.max_track_reuse),
            "max_examples_per_cut": int(args.max_examples_per_cut),
            "seed": int(args.seed),
        },
        "stats": {
            "source_track_rows": int(len(observations)),
            "source_tracklets": int(observations["track_id"].nunique()),
            "lineage_related_tracklets_excluded": int(len(lineage_ids)),
            "clean_track_windows": int(clean_total),
            "candidate_groups": int(len(group_candidates)),
            "selected_examples": int(len(examples_df)),
            "synthetic_tracklets": int(len(tracklets_df)),
            "selected_observation_rows": int(len(selected_observations)),
            "candidate_edges": int(len(edges_df)),
            "positive_edges": int(
                (edges_df["continuation_target"] == 1).sum()
            ),
            "negative_edges": int(
                (edges_df["continuation_target"] == 0).sum()
            ),
            "source_decisions_with_wrong_candidate_closer_than_positive": int(
                wrong_closer_positive_count
            ),
            "positive_direct_rank_gt_1": int(
                (positive["source_distance_rank"] > 1).sum()
            ),
        },
        "supervision": {
            "positive_continuation": (
                "source_original_track_id == target_original_track_id"
            ),
            "negative_continuation": (
                "nearby cross-identity source->target edge"
            ),
            "division": "excluded from this first overfit dataset",
            "identity_columns_are_model_inputs": False,
            "note": (
                "original_track_id, identity_index, paired_tracklet_index and "
                "cell_id exist only for provenance/supervision and must never "
                "be tensorized as learned features."
            ),
        },
        "files": {
            "examples": "examples.csv",
            "tracklets": "tracklets.csv",
            "observations": "observations.csv",
            "edges": "edges.csv",
            "global_motion": "global_motion.csv",
        },
    }
    atomic_json(output / "manifest.json", manifest)

    print("", flush=True)
    print("=" * 112, flush=True)
    print("DATASET BUILT", flush=True)
    print("=" * 112, flush=True)
    print(f"examples               : {len(examples_df):,}", flush=True)
    print(f"synthetic tracklets     : {len(tracklets_df):,}", flush=True)
    print(
        f"edges                   : {len(edges_df):,} "
        f"({manifest['stats']['positive_edges']} positive / "
        f"{manifest['stats']['negative_edges']} negative)",
        flush=True,
    )
    print(
        "wrong closer than truth : "
        f"{wrong_closer_positive_count} source decisions",
        flush=True,
    )
    print(
        "positive direct rank >1 : "
        f"{manifest['stats']['positive_direct_rank_gt_1']}",
        flush=True,
    )
    print(
        "difficulty margin       : "
        f"median={examples_df['median_wrong_minus_true_um'].median():.3f} um, "
        f"min={examples_df['median_wrong_minus_true_um'].min():.3f} um",
        flush=True,
    )
    print(f"saved                   : {output}", flush=True)
    print("=" * 112, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
