"""Latest-state manifest for all sample/stage executions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .stage_registry import get_stage_spec


MANIFEST_COLUMNS = (
    "sample_id",
    "stage",
    "stage_name",
    "status",
    "started_at",
    "completed_at",
    "runtime_seconds",
    "output_directory",
    "error_type",
    "error_message",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    stage: int
    stage_name: str
    status: str
    started_at: str = ""
    completed_at: str = ""
    runtime_seconds: float | None = None
    output_directory: str = ""
    error_type: str = ""
    error_message: str = ""


class PipelineManifest:
    def __init__(self, output_root: str | Path) -> None:
        self.path = Path(output_root) / "pipeline_manifest.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        *,
        sample_id: str,
        stage: int,
        status: str,
        output_directory: str | Path,
        started_at: str = "",
        completed_at: str = "",
        runtime_seconds: float | None = None,
        error: BaseException | None = None,
    ) -> None:
        spec = get_stage_spec(stage)
        record = ManifestRecord(
            sample_id=str(sample_id),
            stage=int(stage),
            stage_name=spec.name,
            status=str(status),
            started_at=str(started_at),
            completed_at=str(completed_at),
            runtime_seconds=runtime_seconds,
            output_directory=str(Path(output_directory)),
            error_type=(type(error).__name__ if error is not None else ""),
            error_message=(str(error) if error is not None else ""),
        )

        if self.path.is_file():
            table = pd.read_csv(self.path)
            for column in MANIFEST_COLUMNS:
                if column not in table.columns:
                    table[column] = ""
            keep = ~(
                (table["sample_id"].astype(str) == str(sample_id))
                & (pd.to_numeric(table["stage"], errors="coerce") == int(stage))
            )
            table = table.loc[keep, list(MANIFEST_COLUMNS)]
        else:
            table = pd.DataFrame(columns=MANIFEST_COLUMNS)

        table = pd.concat(
            [table, pd.DataFrame([asdict(record)], columns=MANIFEST_COLUMNS)],
            ignore_index=True,
        )
        table["stage"] = pd.to_numeric(table["stage"], errors="coerce").astype("Int64")
        table = table.sort_values(
            ["sample_id", "stage"],
            kind="mergesort",
        ).reset_index(drop=True)

        temporary = self.path.with_suffix(".csv.tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(self.path)


__all__ = [
    "MANIFEST_COLUMNS",
    "ManifestRecord",
    "PipelineManifest",
    "utc_now",
]
