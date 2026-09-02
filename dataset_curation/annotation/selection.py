from __future__ import annotations

"""Unified annotation-session selection and inference-run binding."""

from datetime import datetime, timezone
from pathlib import Path

from dataset_curation.catalog import BioHubCatalog, VolumeRecord
from dataset_curation.errors import ArtifactError, ManifestError
from dataset_curation.io.atomic import atomic_json, read_json


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _annotation_root(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> Path:
    return record.paths.annotation_set(
        annotation_set
    )


def _session_marker(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> Path:
    return (
        _annotation_root(
            record,
            annotation_set=annotation_set,
        )
        / "_session.json"
    )


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
    root = _annotation_root(
        record,
        annotation_set=annotation_set,
    )
    if not root.exists():
        return 0.0

    candidates = [
        root,
        *(
            path
            for path in root.rglob("*")
            if path.is_file()
        ),
    ]
    latest = 0.0
    for path in candidates:
        try:
            latest = max(
                latest,
                path.stat().st_mtime,
            )
        except OSError:
            pass
    return latest


def ensure_annotation_binding(
    record: VolumeRecord,
    *,
    run_id: str,
    annotation_set: str,
) -> Path:
    path = record.paths.annotation_manifest(
        annotation_set
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.is_file():
        payload = read_json(path)
        existing = str(
            payload.get(
                "base_inference_run",
                "",
            )
        )
        if (
            existing
            and existing != str(run_id)
        ):
            raise ManifestError(
                f"Annotation set {annotation_set!r} for "
                f"{record.volume_id} is bound to inference run "
                f"{existing!r}, not {run_id!r}."
            )
        return path

    atomic_json(
        path,
        {
            "schema_version": 2,
            "kind": "unified_annotation_set",
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
    annotation_set: str,
    run_id: str,
) -> Path:
    root = _annotation_root(
        record,
        annotation_set=annotation_set,
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = _session_marker(
        record,
        annotation_set=annotation_set,
    )

    old = (
        read_json(path)
        if path.is_file()
        else {}
    )
    atomic_json(
        path,
        {
            "schema_version": 2,
            "kind": "unified_curation",
            "volume_id": record.volume_id,
            "split": record.split,
            "annotation_set": annotation_set,
            "base_inference_run": str(run_id),
            "created_at": old.get(
                "created_at",
                _now(),
            ),
            "last_opened_at": _now(),
        },
    )
    return path


def select_annotation_volume(
    catalog: BioHubCatalog,
    *,
    split: str,
    run_id: str,
    annotation_set: str,
    volume_id: str | None = None,
    resume: bool = False,
    next_volume: bool = False,
) -> VolumeRecord:
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
        record = catalog.get(
            volume_id,
            split=split,
        )
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
                annotation_set=annotation_set,
            )
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
        if not annotation_started(
            record,
            annotation_set=annotation_set,
        )
    ]
    if not fresh:
        raise ArtifactError(
            f"No unstarted unified annotation volume remains in split "
            f"{split!r} for run {run_id!r}."
        )
    return sorted(
        fresh,
        key=lambda record: record.volume_id,
    )[0]
