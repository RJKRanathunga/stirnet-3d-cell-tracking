from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1

"""Unified annotation-session selection and canonical inference binding."""

from datetime import datetime, timezone
from pathlib import Path

from dataset_curation.catalog import BioHubCatalog, VolumeRecord
from dataset_curation.errors import ArtifactError, ManifestError
from dataset_curation.io.atomic import atomic_json, read_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _annotation_root(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> Path:
    return record.paths.annotation_set(annotation_set)


def _session_marker(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> Path:
    return _annotation_root(
        record,
        annotation_set=annotation_set,
    ) / "_session.json"


def annotation_started(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> bool:
    return _session_marker(
        record,
        annotation_set=annotation_set,
    ).is_file()


def annotation_mtime(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> float:
    root = _annotation_root(record, annotation_set=annotation_set)
    if not root.exists():
        return 0.0

    candidates = [
        root,
        *(path for path in root.rglob("*") if path.is_file()),
    ]
    latest = 0.0
    for path in candidates:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            pass
    return latest


def _required_inference_id(record: VolumeRecord) -> str:
    value = record.paths.inference_id()
    if value is None:
        raise ArtifactError(
            f"Volume {record.volume_id} does not have a valid canonical "
            "curation_manifest.json with inference_id."
        )
    return value


def ensure_annotation_binding(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> Path:
    path = record.paths.annotation_manifest(annotation_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    inference_id = _required_inference_id(record)

    if path.is_file():
        payload = read_json(path)
        existing = str(payload.get("base_inference_id", "")).strip()
        if not existing:
            raise ManifestError(
                f"Annotation set {annotation_set!r} for {record.volume_id} "
                "does not contain base_inference_id. The old run-name binding "
                "is intentionally unsupported."
            )
        if existing != inference_id:
            raise ManifestError(
                f"Annotation set {annotation_set!r} for {record.volume_id} "
                f"is bound to inference {existing!r}, not the current "
                f"canonical inference {inference_id!r}."
            )
        return path

    atomic_json(
        path,
        {
            "schema_version": 3,
            "kind": "unified_annotation_set",
            "volume_id": record.volume_id,
            "split": record.split,
            "base_inference_id": inference_id,
            "created_at": _now(),
        },
    )
    return path


def touch_annotation_session(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> Path:
    root = _annotation_root(record, annotation_set=annotation_set)
    root.mkdir(parents=True, exist_ok=True)
    path = _session_marker(record, annotation_set=annotation_set)
    inference_id = _required_inference_id(record)

    old = read_json(path) if path.is_file() else {}
    if old:
        existing = str(old.get("base_inference_id", "")).strip()
        if not existing:
            raise ManifestError(
                f"Annotation session for {record.volume_id} does not contain "
                "base_inference_id."
            )
        if existing != inference_id:
            raise ManifestError(
                f"Annotation session for {record.volume_id} is bound to "
                f"inference {existing!r}, not {inference_id!r}."
            )

    atomic_json(
        path,
        {
            "schema_version": 3,
            "kind": "unified_curation",
            "volume_id": record.volume_id,
            "split": record.split,
            "annotation_set": annotation_set,
            "base_inference_id": inference_id,
            "created_at": old.get("created_at", _now()),
            "last_opened_at": _now(),
        },
    )
    return path


def _skip_message(record: VolumeRecord) -> str:
    try:
        payload = record.paths.read_skip_record()
        reason = str(payload.get("reason_code", "recorded_skip"))
        frame = payload.get("trigger_frame")
        suffix = f" at t={int(frame):03d}" if frame is not None else ""
        return f"{reason}{suffix}"
    except Exception:
        return "recorded_skip"


def select_annotation_volume(
    catalog: BioHubCatalog,
    *,
    split: str,
    annotation_set: str,
    volume_id: str | None = None,
    resume: bool = False,
    next_volume: bool = False,
) -> VolumeRecord:
    records = catalog.discover(split)
    ready = [
        record
        for record in records
        if record.paths.inference_complete(frame_count=record.frame_count)
    ]

    if volume_id is not None:
        record = catalog.get(volume_id, split=split)
        if record.paths.inference_skipped():
            raise ArtifactError(
                f"{record.volume_id} was skipped from production inference: "
                f"{_skip_message(record)}."
            )
        if not record.paths.inference_complete(frame_count=record.frame_count):
            raise ArtifactError(
                f"{record.volume_id} does not have complete canonical inference."
            )
        return record

    if resume:
        started = [
            record
            for record in ready
            if annotation_started(record, annotation_set=annotation_set)
        ]
        if not started:
            raise ArtifactError(
                f"No unified annotation session is available to resume in "
                f"split {split!r}."
            )
        return max(
            started,
            key=lambda record: annotation_mtime(
                record,
                annotation_set=annotation_set,
            ),
        )

    fresh = [
        record
        for record in ready
        if not annotation_started(record, annotation_set=annotation_set)
    ]
    if not fresh:
        raise ArtifactError(
            f"No unstarted inference-ready unified annotation volume remains "
            f"in split {split!r}."
        )
    return sorted(fresh, key=lambda record: record.volume_id)[0]
