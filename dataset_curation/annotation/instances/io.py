from __future__ import annotations

"""Input validation, timepoint selection and optional suspect-score loading for instance curation."""

from pathlib import Path

import numpy as np

class AnnotationError(RuntimeError):
    pass

def parse_timepoint_selection(
    text: str,
    available: list[int],
) -> tuple[int, ...]:
    token = str(text).strip().lower()

    if not available:
        raise AnnotationError(
            "No completed Investigation-25 frames were found."
        )

    if token in {"all", "*"}:
        return tuple(available)

    selected: set[int] = set()

    for item in token.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            left, right = item.split("-", 1)
            first = int(left)
            last = int(right)

            if last < first:
                raise AnnotationError(
                    f"Invalid timepoint range: {item}"
                )

            selected.update(range(first, last + 1))
        else:
            selected.add(int(item))

    if not selected:
        raise AnnotationError("No timepoints selected.")

    missing = sorted(selected - set(available))
    if missing:
        raise AnnotationError(
            f"Requested timepoints are unavailable: {missing}. "
            f"Available frames: {available}"
        )

    return tuple(sorted(selected))

def _suspect_score_path(suspect_root: Path, dataset_t: int) -> Path:
    return suspect_root / f"t{int(dataset_t):03d}.npz"

def load_suspect_instance_frames(
    *,
    suspect_root: Path,
    timepoints: tuple[int, ...],
    instances: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize threshold-passing ORIGINAL instance IDs only."""
    if instances.ndim != 4:
        raise AnnotationError(
            f"Suspect display expects (T,Z,Y,X); got {instances.shape}."
        )
    if len(timepoints) != instances.shape[0]:
        raise AnnotationError("Suspect timepoint/instance-stack length mismatch.")

    output = np.zeros_like(instances)
    total_rows = 0
    total_displayed = 0

    for local_t, dataset_t in enumerate(timepoints):
        path = _suspect_score_path(suspect_root, dataset_t)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing merge-suspect score file:\n  {path}\n"
                "Run 03_ first or pass --build-suspects."
            )
        with np.load(path, allow_pickle=False) as payload:
            missing = [
                key for key in ("instance_id", "suspect_score")
                if key not in payload.files
            ]
            if missing:
                raise AnnotationError(f"{path} is missing arrays: {missing}")
            ids = np.asarray(payload["instance_id"], dtype=np.int64).reshape(-1)
            scores = np.asarray(payload["suspect_score"], dtype=np.float32).reshape(-1)

        if ids.shape != scores.shape:
            raise AnnotationError(f"ID/score mismatch in {path}")
        if ids.size and len(np.unique(ids)) != len(ids):
            raise AnnotationError(f"Duplicate instance IDs in {path}")
        if np.any(ids <= 0) or np.any(~np.isfinite(scores)):
            raise AnnotationError(f"Invalid suspect rows in {path}")

        frame = instances[local_t]
        max_label = int(frame.max(initial=0))
        lookup = np.zeros(max_label + 1, dtype=bool)
        passing_ids = ids[scores >= float(threshold)]
        passing_ids = passing_ids[passing_ids <= max_label]
        if passing_ids.size:
            lookup[passing_ids] = True
            frame_index = frame.astype(np.int64, copy=False)
            output[local_t] = np.where(lookup[frame_index], frame, 0).astype(
                instances.dtype, copy=False
            )

        total_rows += int(len(ids))
        total_displayed += int(len(passing_ids))
        print(
            f"[suspects] t={dataset_t}: {len(passing_ids)}/{len(ids)} "
            f"instances >= {float(threshold):.3f}"
        )

    return output, {
        "score_rows": int(total_rows),
        "displayed_instances": int(total_displayed),
    }

def validate_stacks(
    raw: np.ndarray,
    supervoxels: np.ndarray,
    instances: np.ndarray,
    foreground: np.ndarray,
) -> None:
    expected = raw.shape

    for name, array in (
        ("supervoxels", supervoxels),
        ("instances", instances),
        ("foreground", foreground),
    ):
        if array.shape != expected:
            raise AnnotationError(
                f"Shape mismatch: raw={expected}, {name}={array.shape}."
            )

    if np.any(supervoxels < 0):
        raise AnnotationError("Supervoxel IDs must be non-negative.")

    if np.any(instances < 0):
        raise AnnotationError("Instance IDs must be non-negative.")
