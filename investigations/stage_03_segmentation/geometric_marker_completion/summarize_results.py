"""Combine generated component summaries without interpreting improvement."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.io.tables import load_csv, save_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    arguments = parser.parse_args()
    paths = sorted(arguments.root.glob("*/component_summary.csv"))
    tables = []
    for path in paths:
        table = load_csv(path)
        table.insert(0, "case", path.parent.name)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    save_csv(combined, arguments.root / "summary.csv")
    print(f"Wrote {len(combined)} component rows from {len(paths)} cases")


if __name__ == "__main__":
    main()
