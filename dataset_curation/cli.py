from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dataset_curation._repo import repo_root
from dataset_curation.annotation.instances.curation_runner import (
    run_instance_annotation,
)
from dataset_curation.annotation.selection import (
    annotation_started,
    ensure_annotation_binding,
    select_annotation_volume,
    touch_annotation_session,
)
from dataset_curation.catalog import BioHubCatalog, VolumeRecord
from dataset_curation.config import BIOHUB_DATA_ROOT
from dataset_curation.errors import ArtifactError
from dataset_curation.inference.backends.investigation36 import (
    Investigation36Backend,
)


def _extra(values) -> list[str]:
    values = list(values or [])
    return values[1:] if values and values[0] == "--" else values


def _catalog(args) -> BioHubCatalog:
    catalog = BioHubCatalog(args.data_root)
    catalog.validate_root()
    catalog.ensure_output_roots()
    return catalog


def _record_status(
    record: VolumeRecord,
    *,
    run_id: str,
    annotation_set: str,
) -> tuple[str, str, str]:
    paths = record.paths
    complete = paths.inference_complete(
        run_id,
        frame_count=record.frame_count,
    )
    if complete:
        inference = "complete"
    elif paths.has_any_preprocessed_data(run_id):
        inference = "partial"
    else:
        inference = "missing"

    instances = (
        "started"
        if annotation_started(
            record,
            kind="instances",
            annotation_set=annotation_set,
        )
        else "-"
    )
    tracks = (
        "started"
        if annotation_started(
            record,
            kind="tracks",
            annotation_set=annotation_set,
        )
        else "-"
    )
    return inference, instances, tracks


def cmd_status(args) -> None:
    catalog = _catalog(args)

    if args.split == "all":
        records = catalog.discover_all()
    else:
        records = catalog.discover(args.split)

    if args.limit is not None:
        records = records[: max(int(args.limit), 0)]

    headers = (
        "SPLIT",
        "VOLUME",
        "FRAMES",
        "SPARSE_GT",
        "INFERENCE",
        "INST_ANN",
        "TRACK_ANN",
    )
    rows = []

    for record in records:
        inference, instances, tracks = _record_status(
            record,
            run_id=args.run_id,
            annotation_set=args.annotation_set,
        )
        rows.append(
            (
                record.split,
                record.volume_id,
                str(record.frame_count or "?"),
                "yes" if record.has_ground_truth else "no",
                inference,
                instances,
                tracks,
            )
        )

    widths = [
        max(
            len(headers[index]),
            *(len(row[index]) for row in rows),
        )
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
        "SPARSE_GT only reports presence of ground_truth_nodes.csv + "
        "ground_truth_edges.csv. It is not used as full-volume GT."
    )


def _inference_selection(args, catalog: BioHubCatalog) -> list[VolumeRecord]:
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
            raise KeyError(
                f"Unknown {args.split} volume IDs: {missing}"
            )
        return selected

    candidates = (
        records
        if args.force
        else [
            record
            for record in records
            if not record.paths.inference_complete(
                args.run_id,
                frame_count=record.frame_count,
            )
        ]
    )

    if args.all_volumes:
        return candidates

    count = 1 if args.count is None else int(args.count)
    if count < 1:
        raise ValueError("--count must be >= 1")
    return candidates[:count]


def cmd_infer(args) -> None:
    catalog = _catalog(args)
    selected = _inference_selection(args, catalog)

    if not selected:
        print(
            f"No inference work is pending for split {args.split!r}, "
            f"run {args.run_id!r}."
        )
        return

    print("=" * 96)
    print("BIOHUB BATCH INFERENCE")
    print("=" * 96)
    print(f"data root : {catalog.data_root}")
    print(f"split     : {args.split}")
    print(f"run id    : {args.run_id}")
    print(f"selected  : {len(selected)}")
    print(f"force     : {bool(args.force)}")
    print("=" * 96)

    backend = Investigation36Backend()
    failures: list[tuple[str, str]] = []
    completed = 0
    skipped = 0

    for index, record in enumerate(selected, start=1):
        already_complete = record.paths.inference_complete(
            args.run_id,
            frame_count=record.frame_count,
        )

        if already_complete and not args.force:
            print(
                f"[{index}/{len(selected)}] SKIP "
                f"{record.volume_id}: already complete"
            )
            skipped += 1
            continue

        print()
        print(
            f"[{index}/{len(selected)}] RUN "
            f"{record.split}/{record.volume_id}"
        )

        try:
            backend.run_volume(
                record,
                run_id=args.run_id,
                force=bool(args.force),
                extra_args=_extra(args.extra),
            )
            completed += 1
        except Exception as exc:
            failures.append(
                (
                    record.volume_id,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            print(
                f"[FAILED] {record.volume_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise

    print()
    print("=" * 96)
    print("BATCH SUMMARY")
    print("=" * 96)
    print(f"completed : {completed}")
    print(f"skipped   : {skipped}")
    print(f"failed    : {len(failures)}")
    for volume_id, message in failures:
        print(f"  {volume_id}: {message}")
    print("=" * 96)

    if failures:
        raise SystemExit(1)


def _choose_annotation_record(
    args,
    *,
    kind: str,
) -> VolumeRecord:
    catalog = _catalog(args)
    return select_annotation_volume(
        catalog,
        split=args.split,
        kind=kind,
        run_id=args.run_id,
        annotation_set=args.annotation_set,
        volume_id=args.id,
        resume=bool(args.resume),
        next_volume=bool(args.next),
    )


def cmd_annotate_instances(args) -> None:
    record = _choose_annotation_record(
        args,
        kind="instances",
    )

    print(
        f"[annotation] selected {record.split}/{record.volume_id} "
        f"for instance annotation"
    )

    run_instance_annotation(
        record,
        run_id=args.run_id,
        annotation_set=args.annotation_set,
        timepoint_selection=args.timepoints,
        suspect_threshold=float(args.suspect_threshold),
        resume=not bool(args.no_resume_data),
    )


def cmd_annotate_tracks(args) -> None:
    record = _choose_annotation_record(
        args,
        kind="tracks",
    )
    paths = record.paths

    ensure_annotation_binding(
        record,
        run_id=args.run_id,
        annotation_set=args.annotation_set,
    )
    touch_annotation_session(
        record,
        kind="tracks",
        annotation_set=args.annotation_set,
        run_id=args.run_id,
    )

    source = paths.inference_run(args.run_id)
    output = paths.track_annotations(args.annotation_set)
    output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "dataset_curation._compat.track_annotator",
        "--sample-id", record.volume_id,
        "--source-root", str(source),
        "--output-dir", str(output),
    ]
    if args.no_resume_data:
        command.append("--no-resume")
    command.extend(_extra(args.extra))

    print(
        f"[annotation] selected {record.split}/{record.volume_id} "
        f"for track annotation"
    )
    print("[dataset_curation] " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=repo_root(),
        check=True,
    )


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


def _add_annotation_selector(
    parser: argparse.ArgumentParser,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--id",
        help="Open this exact volume ID.",
    )
    group.add_argument(
        "--next",
        action="store_true",
        help=(
            "Open the first inference-ready volume whose annotation "
            "session has never been started. This is the default."
        ),
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Open the most recently touched existing annotation "
            "session for this annotation type."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_curation",
        description=(
            "External-drive BioHub inference and one-volume-at-a-time "
            "annotation workflow."
        ),
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

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
    status.add_argument("--run-id", default="current")
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
        help=(
            "Run a specific volume. Repeat --id to select multiple IDs."
        ),
    )
    selection.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Run this many volumes that do not already have a complete "
            "inference cache."
        ),
    )
    selection.add_argument(
        "--all",
        dest="all_volumes",
        action="store_true",
        help="Run every volume with missing inference.",
    )
    infer.add_argument("--run-id", default="current")
    infer.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the selected cache is complete.",
    )
    infer.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch at the first failed volume.",
    )
    infer.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=(
            "Additional Investigation-36 arguments after `--`, for example "
            "-- --checkpoint <path>."
        ),
    )
    infer.set_defaults(func=cmd_infer)

    instances = sub.add_parser(
        "annotate-instances",
        help="Open one volume in the merged-cell instance annotator.",
    )
    _add_data_root(instances)
    instances.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )
    _add_annotation_selector(instances)
    instances.add_argument("--run-id", default="current")
    instances.add_argument("--annotation-set", default="main")
    instances.add_argument(
        "--timepoints",
        default="all",
        help="all, 0-19, or comma/range selection.",
    )
    instances.add_argument(
        "--suspect-threshold",
        type=float,
        default=0.70,
    )
    instances.add_argument(
        "--no-resume-data",
        action="store_true",
        help="Ignore persisted corrections inside the selected volume.",
    )
    instances.set_defaults(func=cmd_annotate_instances)

    tracks = sub.add_parser(
        "annotate-tracks",
        help="Open one volume in the Trackastra association annotator.",
    )
    _add_data_root(tracks)
    tracks.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )
    _add_annotation_selector(tracks)
    tracks.add_argument("--run-id", default="current")
    tracks.add_argument("--annotation-set", default="main")
    tracks.add_argument(
        "--no-resume-data",
        action="store_true",
        help="Ignore persisted track corrections for the selected volume.",
    )
    tracks.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=(
            "Additional track-annotator arguments after `--`, e.g. "
            "-- --max-ray-distance-um 10."
        ),
    )
    tracks.set_defaults(func=cmd_annotate_tracks)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    # If no selector was supplied to an annotation command, `--next` is the
    # effective default.
    if args.command in {"annotate-instances", "annotate-tracks"}:
        if not args.id and not args.resume and not args.next:
            args.next = True

    args.func(args)
    return 0
