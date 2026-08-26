from __future__ import annotations

"""Strict structural validator for Biohub Kaggle submission.csv files."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
INTEGER_COLUMNS = ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_count: int = 0
    node_count: int = 0
    edge_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def discover_test_zarrs(test_root: str | Path) -> dict[str, Path]:
    root = Path(test_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    result: dict[str, Path] = {}
    for pattern in ("*.zarr", "*/*.zarr"):
        for path in root.glob(pattern):
            if not path.is_dir():
                continue
            name = path.name[:-5] if path.name.endswith(".zarr") else path.stem
            if name in result and result[name].resolve() != path.resolve():
                raise RuntimeError(f"Duplicate test dataset name {name!r}")
            result[name] = path.resolve()
    if not result:
        raise FileNotFoundError(f"No .zarr datasets found under {root}")
    return result


def _zarr_shape(path: Path) -> tuple[int, int, int, int]:
    import zarr

    obj = zarr.open(str(path), mode="r")
    if hasattr(obj, "shape") and obj.shape is not None:
        shape = tuple(int(v) for v in obj.shape)
    elif "0" in obj:
        shape = tuple(int(v) for v in obj["0"].shape)
    else:
        arrays = list(obj.arrays())
        if not arrays:
            raise ValueError(f"No arrays found in Zarr group {path}")
        shape = tuple(int(v) for v in arrays[0][1].shape)
    if len(shape) != 4:
        raise ValueError(f"Expected T,Z,Y,X shape for {path}, got {shape}")
    return shape


def _coerce_integer_columns(frame: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    result = frame.copy()
    for column in INTEGER_COLUMNS:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if bool(numeric.isna().any()):
            report.errors.append(f"Column {column!r} contains non-numeric or missing values")
            continue
        values = numeric.to_numpy(dtype=float)
        if not np.all(np.equal(values, np.floor(values))):
            report.errors.append(f"Column {column!r} contains non-integer values")
            continue
        result[column] = numeric.astype(np.int64)
    return result


def validate_submission(
    csv_path: str | Path,
    *,
    test_root: str | Path | None = None,
    require_exact_dataset_set: bool = True,
    check_zarr_bounds: bool = True,
) -> ValidationReport:
    path = Path(csv_path).expanduser().resolve()
    report = ValidationReport()
    if not path.is_file():
        report.errors.append(f"Submission file does not exist: {path}")
        return report

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        report.errors.append(f"Could not read CSV: {type(exc).__name__}: {exc}")
        return report

    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        report.errors.append(
            "Columns/order must be exactly: " + ",".join(SUBMISSION_COLUMNS)
        )
        return report
    if frame.empty:
        report.errors.append("Submission is empty")
        return report

    frame = _coerce_integer_columns(frame, report)
    if report.errors:
        return report
    if frame["dataset"].isna().any() or (frame["dataset"].astype(str).str.len() == 0).any():
        report.errors.append("dataset contains missing/empty values")
    frame["dataset"] = frame["dataset"].astype(str)

    expected_ids = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["id"].to_numpy(dtype=np.int64), expected_ids):
        report.errors.append("id must be consecutive integers 0..N-1 in row order")

    valid_types = {"node", "edge"}
    observed_types = set(frame["row_type"].astype(str).unique())
    bad_types = sorted(observed_types - valid_types)
    if bad_types:
        report.errors.append(f"Unsupported row_type values: {bad_types}")

    node_rows = frame[frame["row_type"] == "node"].copy()
    edge_rows = frame[frame["row_type"] == "edge"].copy()
    report.dataset_count = int(frame["dataset"].nunique())
    report.node_count = int(len(node_rows))
    report.edge_count = int(len(edge_rows))
    if node_rows.empty:
        report.errors.append("Submission contains no node rows")

    if not node_rows.empty:
        if bool((node_rows["node_id"] < 0).any()):
            report.errors.append("Node rows require node_id >= 0")
        if bool((node_rows[["t", "z", "y", "x"]] < 0).any(axis=None)):
            report.errors.append("Node rows require t,z,y,x >= 0")
        if bool((node_rows[["source_id", "target_id"]] != -1).any(axis=None)):
            report.errors.append("Node rows require source_id=target_id=-1")
        duplicate_nodes = node_rows.duplicated(subset=["dataset", "node_id"], keep=False)
        if bool(duplicate_nodes.any()):
            report.errors.append("node_id must be unique within each dataset")

    if not edge_rows.empty:
        if bool((edge_rows[["node_id", "t", "z", "y", "x"]] != -1).any(axis=None)):
            report.errors.append("Edge rows require node_id,t,z,y,x=-1")
        if bool((edge_rows[["source_id", "target_id"]] < 0).any(axis=None)):
            report.errors.append("Edge rows require source_id,target_id >= 0")
        if bool((edge_rows["source_id"] == edge_rows["target_id"]).any()):
            report.errors.append("Self edges are not allowed")
        duplicate_edges = edge_rows.duplicated(
            subset=["dataset", "source_id", "target_id"], keep=False
        )
        if bool(duplicate_edges.any()):
            report.errors.append("Duplicate directed edges detected")

    test_zarrs: dict[str, Path] = {}
    if test_root is not None:
        try:
            test_zarrs = discover_test_zarrs(test_root)
        except Exception as exc:
            report.errors.append(f"Could not discover test Zarrs: {exc}")
            return report
        expected = set(test_zarrs)
        observed = set(frame["dataset"].unique())
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            report.errors.append(f"Missing test datasets: {missing}")
        if require_exact_dataset_set and extra:
            report.errors.append(f"Submission contains non-test datasets: {extra}")

    # Graph-level checks are done per dataset so node IDs can safely restart at 1.
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        nodes = dataset_frame[dataset_frame["row_type"] == "node"].copy()
        edges = dataset_frame[dataset_frame["row_type"] == "edge"].copy()
        if nodes.empty:
            report.errors.append(f"{dataset}: dataset has no node rows")
            continue
        node_time = dict(zip(nodes["node_id"].astype(int), nodes["t"].astype(int)))
        known = set(node_time)
        for row in edges.itertuples(index=False):
            source = int(row.source_id)
            target = int(row.target_id)
            if source not in known:
                report.errors.append(f"{dataset}: edge source_id {source} does not exist")
                continue
            if target not in known:
                report.errors.append(f"{dataset}: edge target_id {target} does not exist")
                continue
            if node_time[target] <= node_time[source]:
                report.errors.append(
                    f"{dataset}: edge {source}->{target} is not forward in time "
                    f"({node_time[source]}->{node_time[target]})"
                )
        if not edges.empty:
            indegree = edges.groupby("target_id").size()
            outdegree = edges.groupby("source_id").size()
            if bool((indegree > 1).any()):
                bad = indegree[indegree > 1].head(10).to_dict()
                report.errors.append(f"{dataset}: nodes with indegree >1: {bad}")
            if bool((outdegree > 2).any()):
                bad = outdegree[outdegree > 2].head(10).to_dict()
                report.errors.append(f"{dataset}: nodes with outdegree >2: {bad}")

        if check_zarr_bounds and dataset in test_zarrs:
            try:
                t_size, z_size, y_size, x_size = _zarr_shape(test_zarrs[dataset])
            except Exception as exc:
                report.errors.append(f"{dataset}: could not read Zarr shape: {exc}")
                continue
            bounds = {"t": t_size, "z": z_size, "y": y_size, "x": x_size}
            for column, size in bounds.items():
                bad = nodes[(nodes[column] < 0) | (nodes[column] >= size)]
                if not bad.empty:
                    report.errors.append(
                        f"{dataset}: {len(bad)} node(s) have {column} outside [0,{size - 1}]"
                    )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Biohub Kaggle submission.csv")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument(
        "--allow-extra-datasets",
        action="store_true",
        help="Do not require the CSV dataset set to exactly equal the discovered test set.",
    )
    parser.add_argument(
        "--skip-zarr-bounds",
        action="store_true",
        help="Skip t,z,y,x checks against test Zarr shapes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_submission(
        args.csv,
        test_root=args.test_root,
        require_exact_dataset_set=not args.allow_extra_datasets,
        check_zarr_bounds=not args.skip_zarr_bounds,
    )
    print(
        f"datasets={report.dataset_count} nodes={report.node_count} edges={report.edge_count}"
    )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
