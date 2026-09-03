from __future__ import annotations

# DATASET_CURATION_LAZY_RUNTIME_IMPORTS_V1
# DATASET_CURATION_UNIFIED_ANNOTATION_V1
# DATASET_CURATION_CANONICAL_SKIP_V1

import argparse
import sys
from pathlib import Path

from dataset_curation.annotation.selection import (
    annotation_started,
    select_annotation_volume,
)
from dataset_curation.catalog import BioHubCatalog, VolumeRecord
from dataset_curation.config import BIOHUB_DATA_ROOT


def _extra(values) -> list[str]:
    values = list(values or [])
    return values[1:] if values and values[0] == "--" else values


def _catalog(args, *, ensure_outputs: bool = True) -> BioHubCatalog:
    catalog = BioHubCatalog(args.data_root)
    catalog.validate_root()
    if ensure_outputs:
        catalog.ensure_output_roots()
    return catalog


def _skip_detail(record: VolumeRecord) -> str:
    try:
        payload = record.paths.read_skip_record()
        reason = str(payload.get("reason_code", "recorded_skip"))
        raw_frame = payload.get("trigger_frame")
        if raw_frame is None:
            return reason
        return f"{reason}@t{int(raw_frame):03d}"
    except Exception:
        return "recorded_skip"


def _record_status(
    record: VolumeRecord,
    *,
    annotation_set: str,
) -> tuple[str, str, str]:
    paths = record.paths
    if paths.inference_complete(frame_count=record.frame_count):
        inference = "complete"
        detail = "-"
    elif paths.inference_skipped():
        inference = "skipped"
        detail = _skip_detail(record)
    elif paths.has_any_preprocessed_data():
        inference = "partial"
        detail = "-"
    else:
        inference = "missing"
        detail = "-"

    curation = (
        "started"
        if annotation_started(record, annotation_set=annotation_set)
        else "-"
    )
    return inference, detail, curation


def cmd_status(args) -> None:
    catalog = _catalog(args)
    records = (
        catalog.discover_all()
        if args.split == "all"
        else catalog.discover(args.split)
    )
    if args.limit is not None:
        records = records[: max(int(args.limit), 0)]

    headers = (
        "SPLIT",
        "VOLUME",
        "FRAMES",
        "SPARSE_GT",
        "INFERENCE",
        "DETAIL",
        "CURATION",
    )
    rows = []
    for record in records:
        inference, detail, curation = _record_status(
            record,
            annotation_set=args.annotation_set,
        )
        rows.append(
            (
                record.split,
                record.volume_id,
                str(record.frame_count or "?"),
                "yes" if record.has_ground_truth else "no",
                inference,
                detail,
                curation,
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]

    def line(values):
        return "  ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(values)
        )

    print(f"BioHub root: {catalog.data_root}")
    print(line(headers))
    print(line(tuple("-" * width for width in widths)))
    for row in rows:
        print(line(row))

    print()
    print(f"volumes: {len(rows)}")
    print(
        "SPARSE_GT only reports presence of sparse ground-truth CSVs. "
        "It is not interpreted as full-volume GT."
    )


def _inference_selection(
    args,
    catalog: BioHubCatalog,
) -> list[VolumeRecord]:
    records = catalog.discover(args.split)
    by_id = {record.volume_id: record for record in records}

    if args.id:
        selected = []
        missing = []
        for volume_id in args.id:
            record = by_id.get(volume_id)
            if record is None:
                missing.append(volume_id)
            else:
                selected.append(record)
        if missing:
            raise KeyError(f"Unknown {args.split} volume IDs: {missing}")
        return selected

    def eligible(record: VolumeRecord) -> bool:
        if record.paths.inference_skipped() and not args.retry_skipped:
            return False
        if args.force:
            return True
        return not record.paths.inference_complete(
            frame_count=record.frame_count
        )

    candidates = [record for record in records if eligible(record)]
    if args.all_volumes:
        return candidates

    count = 1 if args.count is None else int(args.count)
    if count < 1:
        raise ValueError("--count must be >= 1")
    return candidates[:count]


def cmd_infer(args) -> None:
    from dataset_curation.inference.backends.stirnet_trackastra import (
        StirNetTrackastraBackend,
    )

    catalog = _catalog(args)
    selected = _inference_selection(args, catalog)
    if not selected:
        print(f"No inference work is pending for split {args.split!r}.")
        return

    print("=" * 96)
    print("BIOHUB BATCH INFERENCE")
    print("=" * 96)
    print(f"data root     : {catalog.data_root}")
    print(f"split         : {args.split}")
    print(f"selected      : {len(selected)}")
    print(f"force         : {bool(args.force)}")
    print(f"retry skipped : {bool(args.retry_skipped)}")
    print("=" * 96)

    backend = StirNetTrackastraBackend()
    failures: list[tuple[str, str]] = []
    skipped_rows: list[tuple[str, str, int | None]] = []
    completed = 0
    reused = 0

    for index, record in enumerate(selected, start=1):
        print()
        print(f"[{index}/{len(selected)}] RUN {record.split}/{record.volume_id}")
        try:
            outcome = backend.run_volume(
                record,
                force=bool(args.force),
                retry_skipped=bool(args.retry_skipped),
                extra_args=_extra(args.extra),
            )
            if outcome.status == "complete":
                completed += 1
            elif outcome.status == "reused":
                reused += 1
            elif outcome.status == "skipped":
                skipped_rows.append(
                    (
                        record.volume_id,
                        outcome.reason_code or "recorded_skip",
                        outcome.trigger_frame,
                    )
                )
            else:
                raise RuntimeError(f"Unknown inference outcome: {outcome.status}")
        except Exception as exc:
            failures.append(
                (record.volume_id, f"{type(exc).__name__}: {exc}")
            )
            print(
                f"[FAILED] {record.volume_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise

    print()
    print("=" * 96)
    print("BATCH SUMMARY")
    print("=" * 96)
    print(f"completed : {completed}")
    print(f"reused    : {reused}")
    print(f"skipped   : {len(skipped_rows)}")
    print(f"failed    : {len(failures)}")
    if skipped_rows:
        print("skipped volumes:")
        for volume_id, reason, frame in skipped_rows:
            frame_text = f" t={frame:03d}" if frame is not None else ""
            print(f"  {volume_id}: {reason}{frame_text}")
    for volume_id, message in failures:
        print(f"  {volume_id}: {message}")
    print("=" * 96)

    if failures:
        raise SystemExit(1)


def cmd_annotate(args) -> None:
    from dataset_curation.annotation.curation_runner import run_annotation

    catalog = _catalog(args)
    record = select_annotation_volume(
        catalog,
        split=args.split,
        annotation_set=args.annotation_set,
        volume_id=args.id,
        resume=bool(args.resume),
        next_volume=bool(args.next),
    )

    print(f"[annotation] selected {record.split}/{record.volume_id}")
    run_annotation(
        record,
        annotation_set=args.annotation_set,
        boundary_margin_um=float(args.boundary_margin_um),
        resume=not bool(args.no_resume_data),
    )


# DATASET_CURATION_ANNOTATION_PROGRESS_V1
def cmd_progress(args) -> None:
    from dataset_curation.annotation.progress import (
        compute_annotation_progress,
        format_annotation_progress,
    )
    from dataset_curation.annotation.selection import annotation_mtime

    catalog = _catalog(args, ensure_outputs=False)

    if args.id:
        record = catalog.get(args.id, split=args.split)
    else:
        started = [
            candidate
            for candidate in catalog.discover(args.split)
            if annotation_started(
                candidate,
                annotation_set=args.annotation_set,
            )
        ]
        if not started:
            raise RuntimeError(
                f"No started annotation session exists in split {args.split!r} "
                f"for annotation set {args.annotation_set!r}."
            )
        record = max(
            started,
            key=lambda candidate: annotation_mtime(
                candidate,
                annotation_set=args.annotation_set,
            ),
        )

    if record.frame_count is None:
        raise RuntimeError(
            f"Could not determine frame count for {record.split}/{record.volume_id}."
        )

    annotation_root = record.paths.annotation_set(args.annotation_set)
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Annotation set does not exist: {annotation_root}")

    progress = compute_annotation_progress(
        annotation_root,
        frame_count=int(record.frame_count),
    )
    print(
        format_annotation_progress(
            progress,
            split=record.split,
            volume_id=record.volume_id,
            annotation_set=args.annotation_set,
        )
    )


def _find_source_record(
    catalog: BioHubCatalog,
    *,
    volume_id: str,
    split: str | None,
) -> VolumeRecord:
    if split is not None:
        return catalog.get(volume_id, split=split)

    matches = [
        record
        for record in catalog.discover_all()
        if record.volume_id == str(volume_id)
    ]
    if not matches:
        raise KeyError(f"Volume {volume_id!r} was not found in train or test.")
    if len(matches) > 1:
        raise KeyError(
            f"Volume {volume_id!r} exists in multiple splits. "
            "Pass --split train or --split test."
        )
    return matches[0]


def cmd_view_source(args) -> None:
    from dataset_curation.visualization.source_viewer import view_source_volume

    catalog = _catalog(args, ensure_outputs=False)
    record = _find_source_record(
        catalog,
        volume_id=args.id,
        split=args.split,
    )
    view_source_volume(record)


def _add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Advanced override. Default is the hard-coded external root "
            f"{BIOHUB_DATA_ROOT}."
        ),
    )


def _add_annotation_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--id", help="Open this exact volume ID.")
    group.add_argument(
        "--next",
        action="store_true",
        help=(
            "Open the first inference-ready volume whose unified curation "
            "session has never been started. This is the default."
        ),
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Open the most recently touched unified curation session.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_curation",
        description=(
            "BioHub inference, unified cell/track curation, and source viewing."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser(
        "status",
        help="List discovered volumes and curation state.",
    )
    _add_data_root(status)
    status.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="train",
    )
    status.add_argument("--annotation-set", default="main")
    status.add_argument("--limit", type=int, default=None)
    status.set_defaults(func=cmd_status)

    infer = sub.add_parser(
        "infer",
        help="Run inference on missing/selected volumes.",
    )
    _add_data_root(infer)
    infer.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )
    selection = infer.add_mutually_exclusive_group()
    selection.add_argument(
        "--id",
        action="append",
        help="Run a specific volume. Repeat --id to select multiple IDs.",
    )
    selection.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Run this many volumes that are neither complete nor already "
            "recorded as skipped."
        ),
    )
    selection.add_argument(
        "--all",
        dest="all_volumes",
        action="store_true",
        help="Run every eligible volume with missing inference.",
    )
    infer.add_argument(
        "--force",
        action="store_true",
        help="Recompute complete inference, but do not override skip records.",
    )
    infer.add_argument(
        "--retry-skipped",
        action="store_true",
        help=(
            "Explicitly re-evaluate volumes recorded as skipped. This is "
            "required even when --force is supplied."
        ),
    )
    infer.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch at the first real failure. Quality skips continue.",
    )
    infer.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=(
            "Additional production STIR-Net backend arguments after `--`, "
            "for example `-- --checkpoint <path>`."
        ),
    )
    infer.set_defaults(func=cmd_infer)

    annotate = sub.add_parser(
        "annotate",
        help="Open unified spatial + Trackastra curation for one volume.",
    )
    _add_data_root(annotate)
    annotate.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )
    _add_annotation_selector(annotate)
    annotate.add_argument("--annotation-set", default="main")
    annotate.add_argument(
        "--boundary-margin-um",
        type=float,
        default=4.0,
        help="Boundary margin used by notebook-09 broken/new track diagnostics.",
    )
    annotate.add_argument(
        "--no-resume-data",
        action="store_true",
        help="Ignore persisted unified spatial/track state for this volume.",
    )
    annotate.set_defaults(func=cmd_annotate)

    progress = sub.add_parser(
        "progress",
        help=(
            "Report saved spatial/track annotation activity. If --id is "
            "omitted, use the most recently touched started session."
        ),
    )
    _add_data_root(progress)
    progress.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )
    progress.add_argument(
        "--id",
        default=None,
        help=(
            "Volume ID. If omitted, use the most recently touched started "
            "annotation session in the selected split."
        ),
    )
    progress.add_argument("--annotation-set", default="main")
    progress.set_defaults(func=cmd_progress)

    source = sub.add_parser(
        "view-source",
        help="Open one raw source Zarr in Napari without requiring inference.",
    )
    _add_data_root(source)
    source.add_argument("--id", required=True, help="Volume ID to visualize.")
    source.add_argument(
        "--split",
        choices=("train", "test"),
        default=None,
        help="Optional split. If omitted, both source partitions are searched.",
    )
    source.set_defaults(func=cmd_view_source)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "annotate":
        if not args.id and not args.resume and not args.next:
            args.next = True
    args.func(args)
    return 0
