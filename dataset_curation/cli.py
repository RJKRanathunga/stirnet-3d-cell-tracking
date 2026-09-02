from __future__ import annotations

# DATASET_CURATION_LAZY_RUNTIME_IMPORTS_V1
# DATASET_CURATION_UNIFIED_ANNOTATION_V1

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
    return (
        values[1:]
        if values
        and values[0] == "--"
        else values
    )


def _catalog(
    args,
    *,
    ensure_outputs: bool = True,
) -> BioHubCatalog:
    catalog = BioHubCatalog(
        args.data_root
    )
    catalog.validate_root()
    if ensure_outputs:
        catalog.ensure_output_roots()
    return catalog


def _record_status(
    record: VolumeRecord,
    *,
    run_id: str,
    annotation_set: str,
) -> tuple[str, str]:
    paths = record.paths
    complete = paths.inference_complete(
        run_id,
        frame_count=record.frame_count,
    )
    if complete:
        inference = "complete"
    elif paths.has_any_preprocessed_data(
        run_id
    ):
        inference = "partial"
    else:
        inference = "missing"

    curation = (
        "started"
        if annotation_started(
            record,
            annotation_set=annotation_set,
        )
        else "-"
    )
    return (
        inference,
        curation,
    )


def cmd_status(args) -> None:
    catalog = _catalog(args)

    if args.split == "all":
        records = catalog.discover_all()
    else:
        records = catalog.discover(
            args.split
        )

    if args.limit is not None:
        records = records[
            : max(
                int(args.limit),
                0,
            )
        ]

    headers = (
        "SPLIT",
        "VOLUME",
        "FRAMES",
        "SPARSE_GT",
        "INFERENCE",
        "CURATION",
    )
    rows = []

    for record in records:
        inference, curation = (
            _record_status(
                record,
                run_id=args.run_id,
                annotation_set=args.annotation_set,
            )
        )
        rows.append(
            (
                record.split,
                record.volume_id,
                str(
                    record.frame_count
                    or "?"
                ),
                (
                    "yes"
                    if record.has_ground_truth
                    else "no"
                ),
                inference,
                curation,
            )
        )

    widths = [
        (
            max(
                len(headers[index]),
                *(
                    len(row[index])
                    for row in rows
                ),
            )
            if rows
            else len(
                headers[index]
            )
        )
        for index in range(
            len(headers)
        )
    ]

    def line(values):
        return "  ".join(
            str(value).ljust(
                widths[index]
            )
            for index, value in enumerate(
                values
            )
        )

    print(
        f"BioHub root: "
        f"{catalog.data_root}"
    )
    print(line(headers))
    print(
        line(
            tuple(
                "-" * width
                for width in widths
            )
        )
    )
    for row in rows:
        print(line(row))

    print()
    print(
        f"volumes: {len(rows)}"
    )
    print(
        "SPARSE_GT only reports presence of sparse ground-truth CSVs. "
        "It is not interpreted as full-volume GT."
    )


def _inference_selection(
    args,
    catalog: BioHubCatalog,
) -> list[VolumeRecord]:
    records = catalog.discover(
        args.split
    )
    by_id = {
        record.volume_id: record
        for record in records
    }

    if args.id:
        selected = []
        missing = []
        for volume_id in args.id:
            record = by_id.get(
                volume_id
            )
            if record is None:
                missing.append(
                    volume_id
                )
            else:
                selected.append(
                    record
                )
        if missing:
            raise KeyError(
                f"Unknown {args.split} volume IDs: "
                f"{missing}"
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

    count = (
        1
        if args.count is None
        else int(args.count)
    )
    if count < 1:
        raise ValueError(
            "--count must be >= 1"
        )
    return candidates[:count]


def cmd_infer(args) -> None:
    from dataset_curation.inference.backends.stirnet_trackastra import (
        StirNetTrackastraBackend,
    )

    catalog = _catalog(args)
    selected = (
        _inference_selection(
            args,
            catalog,
        )
    )

    if not selected:
        print(
            f"No inference work is pending for split "
            f"{args.split!r}, run {args.run_id!r}."
        )
        return

    print("=" * 96)
    print("BIOHUB BATCH INFERENCE")
    print("=" * 96)
    print(
        f"data root : "
        f"{catalog.data_root}"
    )
    print(
        f"split     : {args.split}"
    )
    print(
        f"run id    : {args.run_id}"
    )
    print(
        f"selected  : {len(selected)}"
    )
    print(
        f"force     : "
        f"{bool(args.force)}"
    )
    print("=" * 96)

    backend = (
        StirNetTrackastraBackend()
    )
    failures: list[
        tuple[str, str]
    ] = []
    completed = 0
    skipped = 0

    for index, record in enumerate(
        selected,
        start=1,
    ):
        already_complete = (
            record.paths.inference_complete(
                args.run_id,
                frame_count=record.frame_count,
            )
        )
        if (
            already_complete
            and not args.force
        ):
            print(
                f"[{index}/{len(selected)}] "
                f"SKIP {record.volume_id}: "
                "already complete"
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
                force=bool(
                    args.force
                ),
                extra_args=_extra(
                    args.extra
                ),
            )
            completed += 1
        except Exception as exc:
            failures.append(
                (
                    record.volume_id,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )
            print(
                f"[FAILED] {record.volume_id}: "
                f"{type(exc).__name__}: "
                f"{exc}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise

    print()
    print("=" * 96)
    print("BATCH SUMMARY")
    print("=" * 96)
    print(
        f"completed : {completed}"
    )
    print(
        f"skipped   : {skipped}"
    )
    print(
        f"failed    : {len(failures)}"
    )
    for volume_id, message in failures:
        print(
            f"  {volume_id}: {message}"
        )
    print("=" * 96)

    if failures:
        raise SystemExit(1)


def cmd_annotate(args) -> None:
    from dataset_curation.annotation.curation_runner import (
        run_annotation,
    )

    catalog = _catalog(args)
    record = select_annotation_volume(
        catalog,
        split=args.split,
        run_id=args.run_id,
        annotation_set=args.annotation_set,
        volume_id=args.id,
        resume=bool(args.resume),
        next_volume=bool(args.next),
    )

    print(
        f"[annotation] selected "
        f"{record.split}/{record.volume_id}"
    )

    run_annotation(
        record,
        run_id=args.run_id,
        annotation_set=args.annotation_set,
        boundary_margin_um=float(
            args.boundary_margin_um
        ),
        resume=not bool(
            args.no_resume_data
        ),
    )


def _find_source_record(
    catalog: BioHubCatalog,
    *,
    volume_id: str,
    split: str | None,
) -> VolumeRecord:
    if split is not None:
        return catalog.get(
            volume_id,
            split=split,
        )

    matches = [
        record
        for record in catalog.discover_all()
        if record.volume_id
        == str(volume_id)
    ]
    if not matches:
        raise KeyError(
            f"Volume {volume_id!r} was not found in train or test."
        )
    if len(matches) > 1:
        raise KeyError(
            f"Volume {volume_id!r} exists in multiple splits. "
            "Pass --split train or --split test."
        )
    return matches[0]


def cmd_view_source(args) -> None:
    from dataset_curation.visualization.source_viewer import (
        view_source_volume,
    )

    catalog = _catalog(
        args,
        ensure_outputs=False,
    )
    record = _find_source_record(
        catalog,
        volume_id=args.id,
        split=args.split,
    )
    view_source_volume(
        record
    )


def _add_data_root(
    parser: argparse.ArgumentParser,
) -> None:
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
    group = (
        parser.add_mutually_exclusive_group()
    )
    group.add_argument(
        "--id",
        help="Open this exact volume ID.",
    )
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
        help=(
            "Open the most recently touched unified curation session."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_curation",
        description=(
            "BioHub inference, unified cell/track curation, and source viewing."
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
        choices=(
            "train",
            "test",
            "all",
        ),
        default="train",
    )
    status.add_argument(
        "--run-id",
        default="current",
    )
    status.add_argument(
        "--annotation-set",
        default="main",
    )
    status.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    status.set_defaults(
        func=cmd_status
    )

    infer = sub.add_parser(
        "infer",
        help="Run inference on missing/selected volumes.",
    )
    _add_data_root(infer)
    infer.add_argument(
        "--split",
        choices=(
            "train",
            "test",
        ),
        default="train",
    )
    selection = (
        infer.add_mutually_exclusive_group()
    )
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
        help=(
            "Run every volume with missing inference."
        ),
    )
    infer.add_argument(
        "--run-id",
        default="current",
    )
    infer.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute even if the selected cache is complete."
        ),
    )
    infer.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop the batch at the first failed volume."
        ),
    )
    infer.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=(
            "Additional production STIR-Net backend arguments after `--`, "
            "for example `-- --checkpoint <path>`."
        ),
    )
    infer.set_defaults(
        func=cmd_infer
    )

    annotate = sub.add_parser(
        "annotate",
        help=(
            "Open unified spatial + Trackastra curation for one volume."
        ),
    )
    _add_data_root(annotate)
    annotate.add_argument(
        "--split",
        choices=(
            "train",
            "test",
        ),
        default="train",
    )
    _add_annotation_selector(
        annotate
    )
    annotate.add_argument(
        "--run-id",
        default="current",
    )
    annotate.add_argument(
        "--annotation-set",
        default="main",
    )
    annotate.add_argument(
        "--boundary-margin-um",
        type=float,
        default=4.0,
        help=(
            "Boundary margin used by notebook-09 broken/new track diagnostics."
        ),
    )
    annotate.add_argument(
        "--no-resume-data",
        action="store_true",
        help=(
            "Ignore persisted unified spatial/track state for this volume."
        ),
    )
    annotate.set_defaults(
        func=cmd_annotate
    )

    source = sub.add_parser(
        "view-source",
        help=(
            "Open one raw source Zarr in Napari without requiring inference."
        ),
    )
    _add_data_root(source)
    source.add_argument(
        "--id",
        required=True,
        help="Volume ID to visualize.",
    )
    source.add_argument(
        "--split",
        choices=(
            "train",
            "test",
        ),
        default=None,
        help=(
            "Optional split. If omitted, both source partitions are searched."
        ),
    )
    source.set_defaults(
        func=cmd_view_source
    )

    return parser


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    if args.command == "annotate":
        if (
            not args.id
            and not args.resume
            and not args.next
        ):
            args.next = True

    args.func(args)
    return 0
