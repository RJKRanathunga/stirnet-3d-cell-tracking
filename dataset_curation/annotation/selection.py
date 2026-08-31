from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dataset_curation.catalog import BioHubCatalog, VolumeRecord
from dataset_curation.errors import ArtifactError, ManifestError
from dataset_curation.io.atomic import atomic_json, read_json


VALID_KINDS = ("instances", "tracks")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kind_root(
    record: VolumeRecord,
    *,
    kind: str,
    annotation_set: str,
) -> Path:
    if kind == "instances":
        return record.paths.instance_annotations(annotation_set)
    if kind == "tracks":
        return record.paths.track_annotations(annotation_set)
    raise ValueError(f"Unknown annotation kind: {kind!r}")


def _session_marker(
    record: VolumeRecord,
    *,
    kind: str,
    annotation_set: str,
) -> Path:
    return _kind_root(
        record,
        kind=kind,
        annotation_set=annotation_set,
    ) / "_session.json"


def annotation_started(
    record: VolumeRecord,
    *,
    kind: str,
    annotation_set: str,
) -> bool:
    root = _kind_root(
        record,
        kind=kind,
        annotation_set=annotation_set,
    )
    marker = _session_marker(
        record,
        kind=kind,
        annotation_set=annotation_set,
    )
    if marker.is_file():
        return True

    # Backward-compatible detection for sessions created before _session.json.
    if kind == "instances":
        return (
            (root / "supervoxel_split_corrections.json").is_file()
            or any(root.glob("manual_instances_t*.npy"))
        )
    if kind == "tracks":
        return (root / "track_annotations.json").is_file()
    return False


def annotation_mtime(
    record: VolumeRecord,
    *,
    kind: str,
    annotation_set: str,
) -> float:
    root = _kind_root(
        record,
        kind=kind,
        annotation_set=annotation_set,
    )
    if not root.exists():
        return 0.0

    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_file())

    latest = 0.0
    for path in candidates:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            pass
    return latest


def ensure_annotation_binding(
    record: VolumeRecord,
    *,
    run_id: str,
    annotation_set: str,
) -> Path:
    path = record.paths.annotation_manifest(annotation_set)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        payload = read_json(path)
        existing = str(payload.get("base_inference_run", ""))
        if existing and existing != str(run_id):
            raise ManifestError(
                f"Annotation set {annotation_set!r} for "
                f"{record.volume_id} is bound to run {existing!r}, "
                f"not {run_id!r}."
            )
        return path

    atomic_json(
        path,
        {
            "schema_version": 1,
            "kind": "annotation_set",
            "volume_id": record.volume_id,
            "split": record.split,
            "base_inference_run": str(run_id),
            "created_at": _now(),
        },
    )
    return path


def touch_annotation_session(
    record: VolumeRecord,
    *,
    kind: str,
    annotation_set: str,
    run_id: str,
) -> Path:
    root = _kind_root(
        record,
        kind=kind,
        annotation_set=annotation_set,
    )
    root.mkdir(parents=True, exist_ok=True)

    path = _session_marker(
        record,
        kind=kind,
        annotation_set=annotation_set,
    )

    old = read_json(path) if path.is_file() else {}
    atomic_json(
        path,
        {
            "schema_version": 1,
            "kind": kind,
            "volume_id": record.volume_id,
            "split": record.split,
            "annotation_set": annotation_set,
            "base_inference_run": run_id,
            "created_at": old.get("created_at", _now()),
            "last_opened_at": _now(),
        },
    )
    return path


def select_annotation_volume(
    catalog: BioHubCatalog,
    *,
    split: str,
    kind: str,
    run_id: str,
    annotation_set: str,
    volume_id: str | None = None,
    resume: bool = False,
    next_volume: bool = False,
) -> VolumeRecord:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}")

    records = catalog.discover(split)
    ready = [
        record
        for record in records
        if record.paths.inference_complete(
            run_id,
            frame_count=record.frame_count,
        )
    ]

    if volume_id is not None:
        record = catalog.get(volume_id, split=split)
        if not record.paths.inference_complete(
            run_id,
            frame_count=record.frame_count,
        ):
            raise ArtifactError(
                f"{record.volume_id} does not have a complete "
                f"inference run {run_id!r}."
            )
        return record

    if resume:
        started = [
            record
            for record in ready
            if annotation_started(
                record,
                kind=kind,
                annotation_set=annotation_set,
            )
        ]
        if not started:
            raise ArtifactError(
                f"No existing {kind} annotation session is available "
                f"to resume in split {split!r}."
            )
        return max(
            started,
            key=lambda record: annotation_mtime(
                record,
                kind=kind,
                annotation_set=annotation_set,
            ),
        )

    # --next is the default behavior when no explicit mode is supplied.
    fresh = [
        record
        for record in ready
        if not annotation_started(
            record,
            kind=kind,
            annotation_set=annotation_set,
        )
    ]
    if not fresh:
        raise ArtifactError(
            f"No unstarted {kind} annotation volume remains in "
            f"split {split!r} for run {run_id!r}."
        )
    return sorted(fresh, key=lambda record: record.volume_id)[0]
