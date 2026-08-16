from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import time
from typing import Iterator

import torch


@dataclass(frozen=True)
class StageProfileRecord:
    name: str
    elapsed_seconds: float
    allocated_mb_before: float
    allocated_mb_after: float
    reserved_mb: float
    peak_allocated_mb: float


class StageProfiler:
    """Opt-in synchronized timing/VRAM profiler for coarse V2 stages."""

    def __init__(
        self,
        enabled: bool = False,
        device: torch.device | str | None = None,
    ):
        self.enabled = bool(enabled)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.records: list[StageProfileRecord] = []
        self.overall_peak_allocated_mb = 0.0

    def clear(self) -> None:
        self.records.clear()
        self.overall_peak_allocated_mb = 0.0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        if self.device.type == "cuda":
            before = torch.cuda.memory_allocated(self.device) / (1024**2)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            before = 0.0
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - started
            if self.device.type == "cuda":
                after = torch.cuda.memory_allocated(self.device) / (1024**2)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
                peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            else:
                after = reserved = peak = 0.0
            self.overall_peak_allocated_mb = max(
                self.overall_peak_allocated_mb, peak
            )
            self.records.append(
                StageProfileRecord(
                    name=name,
                    elapsed_seconds=elapsed,
                    allocated_mb_before=before,
                    allocated_mb_after=after,
                    reserved_mb=reserved,
                    peak_allocated_mb=peak,
                )
            )

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for record in self.records:
            row = result.setdefault(
                record.name,
                {
                    "elapsed_seconds": 0.0,
                    "allocated_mb_before": record.allocated_mb_before,
                    "allocated_mb_after": record.allocated_mb_after,
                    "reserved_mb": 0.0,
                    "peak_allocated_mb": 0.0,
                    "calls": 0.0,
                },
            )
            row["elapsed_seconds"] += record.elapsed_seconds
            row["allocated_mb_after"] = record.allocated_mb_after
            row["reserved_mb"] = max(row["reserved_mb"], record.reserved_mb)
            row["peak_allocated_mb"] = max(
                row["peak_allocated_mb"], record.peak_allocated_mb
            )
            row["calls"] += 1
        return result

    def to_dict(self) -> list[dict[str, float | str]]:
        return [asdict(record) for record in self.records]


__all__ = ["StageProfileRecord", "StageProfiler"]
