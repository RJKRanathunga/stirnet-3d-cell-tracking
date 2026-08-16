from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Iterator

import torch


_MIB = 1024**2


@dataclass(frozen=True)
class StageProfileRecord:
    timestamp_utc: str
    perf_time: float
    global_step: int
    refinement_stage_step: int
    phase: str
    stage: str
    status: str
    elapsed_seconds: float
    allocated_mb_before: float
    allocated_mb_after: float
    reserved_mb_before: float
    reserved_mb_after: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    device_free_mb_before: float
    device_free_mb_after: float
    device_total_mb: float
    metadata: dict[str, object] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

    @property
    def name(self) -> str:
        return self.stage

    @property
    def reserved_mb(self) -> float:
        return self.reserved_mb_after


@dataclass(frozen=True)
class _MemorySnapshot:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0
    total_mb: float = 0.0


class StageProfiler:
    """Streaming stage profiler whose partial JSONL survives a later failure."""

    def __init__(
        self,
        enabled: bool = False,
        device: torch.device | str | None = None,
        output_path: str | Path | None = None,
    ):
        self.enabled = bool(enabled)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_path = None if output_path is None else Path(output_path)
        if self.enabled and self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[StageProfileRecord] = []
        self.overall_peak_allocated_mb = 0.0
        self.global_step = 0
        self.refinement_stage_step = 0
        self.phase = ""
        self.context_metadata: dict[str, object] = {}
        self.last_profile_stage = ""
        self.last_profile_phase = ""
        self._active_stages: list[str] = []

    def clear(self) -> None:
        self.records.clear()
        self.overall_peak_allocated_mb = 0.0
        self.last_profile_stage = ""
        self.last_profile_phase = ""
        self._active_stages.clear()

    def set_context(
        self,
        *,
        global_step: int | None = None,
        refinement_stage_step: int | None = None,
        phase: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if global_step is not None:
            self.global_step = int(global_step)
        if refinement_stage_step is not None:
            self.refinement_stage_step = int(refinement_stage_step)
        if phase is not None:
            self.phase = phase
        if metadata is not None:
            self.context_metadata = dict(metadata)

    @contextmanager
    def phase_scope(self, phase: str) -> Iterator[None]:
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _snapshot(self, *, synchronize: bool) -> _MemorySnapshot:
        if self.device.type != "cuda":
            return _MemorySnapshot()
        if synchronize:
            self._sync()
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return _MemorySnapshot(
            allocated_mb=torch.cuda.memory_allocated(self.device) / _MIB,
            reserved_mb=torch.cuda.memory_reserved(self.device) / _MIB,
            free_mb=free_bytes / _MIB,
            total_mb=total_bytes / _MIB,
        )

    def _safe_snapshot(self) -> _MemorySnapshot:
        try:
            return self._snapshot(synchronize=False)
        except BaseException:
            return _MemorySnapshot()

    def _peaks(self) -> tuple[float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0
        return (
            torch.cuda.max_memory_allocated(self.device) / _MIB,
            torch.cuda.max_memory_reserved(self.device) / _MIB,
        )

    def _safe_peaks(self) -> tuple[float, float]:
        try:
            return self._peaks()
        except BaseException:
            return 0.0, 0.0

    def _qualified_stage(self, name: str, qualify: bool) -> str:
        if not qualify or not self.phase or name.startswith(f"{self.phase}_"):
            return name
        return f"{self.phase}_{name}"

    def _write_jsonl(self, payload: dict[str, object]) -> None:
        if self.output_path is None:
            return
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()

    def _emit_payload(self, payload: dict[str, object]) -> None:
        try:
            print(
                "[mem-profile] "
                + " ".join(
                    f"{key}={payload[key]}"
                    for key in (
                        "status",
                        "phase",
                        "stage",
                        "elapsed_seconds",
                        "allocated_mb_before",
                        "allocated_mb_after",
                        "reserved_mb_after",
                        "peak_allocated_mb",
                        "peak_reserved_mb",
                        "device_free_mb_after",
                        "device_total_mb",
                    )
                    if key in payload
                ),
                flush=True,
            )
        except BaseException:
            pass
        try:
            self._write_jsonl(payload)
        except BaseException as error:
            try:
                print(
                    f"[mem-profile] writer_warning={type(error).__name__}",
                    flush=True,
                )
            except BaseException:
                pass

    def _enter_payload(
        self,
        stage: str,
        before: _MemorySnapshot,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "perf_time": time.perf_counter(),
            "global_step": self.global_step,
            "refinement_stage_step": self.refinement_stage_step,
            "phase": self.phase,
            "stage": stage,
            "status": "enter",
            "elapsed_seconds": 0.0,
            "allocated_mb_before": before.allocated_mb,
            "allocated_mb_after": before.allocated_mb,
            "reserved_mb_before": before.reserved_mb,
            "reserved_mb_after": before.reserved_mb,
            "peak_allocated_mb": before.allocated_mb,
            "peak_reserved_mb": before.reserved_mb,
            "device_free_mb_before": before.free_mb,
            "device_free_mb_after": before.free_mb,
            "device_total_mb": before.total_mb,
            "metadata": metadata,
        }

    def _complete_record(
        self,
        *,
        stage: str,
        status: str,
        started: float,
        before: _MemorySnapshot,
        after: _MemorySnapshot,
        metadata: dict[str, object],
        error: BaseException | None,
    ) -> StageProfileRecord:
        peak_allocated, peak_reserved = self._safe_peaks()
        record = StageProfileRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            perf_time=time.perf_counter(),
            global_step=self.global_step,
            refinement_stage_step=self.refinement_stage_step,
            phase=self.phase,
            stage=stage,
            status=status,
            elapsed_seconds=time.perf_counter() - started,
            allocated_mb_before=before.allocated_mb,
            allocated_mb_after=after.allocated_mb,
            reserved_mb_before=before.reserved_mb,
            reserved_mb_after=after.reserved_mb,
            peak_allocated_mb=peak_allocated,
            peak_reserved_mb=peak_reserved,
            device_free_mb_before=before.free_mb,
            device_free_mb_after=after.free_mb,
            device_total_mb=after.total_mb or before.total_mb,
            metadata=metadata,
            error_type="" if error is None else type(error).__name__,
            error_message="" if error is None else str(error)[:500],
        )
        self.records.append(record)
        self.overall_peak_allocated_mb = max(
            self.overall_peak_allocated_mb, peak_allocated
        )
        self.last_profile_stage = stage
        self.last_profile_phase = self.phase
        self._emit_payload(asdict(record))
        return record

    @staticmethod
    def _is_checkpoint_early_stop(error: BaseException) -> bool:
        """Recognize non-reentrant checkpoint's private control-flow sentinel.

        PyTorch deliberately throws this inside a recomputed function once all
        tensors needed by backward have been rebuilt.  The checkpoint wrapper
        catches it, so a profiling scope nested in that function must re-raise
        it without reporting a false stage failure.
        """
        error_type = type(error)
        return (
            error_type.__name__ == "_StopRecomputationError"
            and error_type.__module__ == "torch.utils.checkpoint"
        )

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        metadata: dict[str, object] | None = None,
        qualify: bool = True,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        stage = self._qualified_stage(name, qualify)
        combined_metadata = {**self.context_metadata, **(metadata or {})}
        before = self._snapshot(synchronize=True)
        # Only an outer flat region resets peaks. Checkpoint recomputation may
        # create nested diagnostic regions inside the enclosing backward stage.
        if self.device.type == "cuda" and not self._active_stages:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._active_stages.append(stage)
        self.last_profile_stage = stage
        self.last_profile_phase = self.phase
        self._emit_payload(self._enter_payload(stage, before, combined_metadata))
        started = time.perf_counter()
        try:
            yield
            after = self._snapshot(synchronize=True)
        except BaseException as error:
            after = self._safe_snapshot()
            checkpoint_early_stop = self._is_checkpoint_early_stop(error)
            try:
                self._complete_record(
                    stage=stage,
                    status="ok" if checkpoint_early_stop else "failed",
                    started=started,
                    before=before,
                    after=after,
                    metadata=combined_metadata,
                    error=None if checkpoint_early_stop else error,
                )
            except BaseException:
                pass
            self._active_stages.pop()
            raise
        else:
            self._complete_record(
                stage=stage,
                status="ok",
                started=started,
                before=before,
                after=after,
                metadata=combined_metadata,
                error=None,
            )
            self._active_stages.pop()

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for record in self.records:
            row = result.setdefault(
                record.stage,
                {
                    "elapsed_seconds": 0.0,
                    "allocated_mb_before": record.allocated_mb_before,
                    "allocated_mb_after": record.allocated_mb_after,
                    "reserved_mb": 0.0,
                    "reserved_mb_before": record.reserved_mb_before,
                    "reserved_mb_after": record.reserved_mb_after,
                    "peak_allocated_mb": 0.0,
                    "peak_reserved_mb": 0.0,
                    "device_free_mb_after": record.device_free_mb_after,
                    "device_total_mb": record.device_total_mb,
                    "calls": 0.0,
                    "failed_calls": 0.0,
                },
            )
            row["elapsed_seconds"] += record.elapsed_seconds
            row["allocated_mb_after"] = record.allocated_mb_after
            row["reserved_mb"] = max(row["reserved_mb"], record.reserved_mb_after)
            row["reserved_mb_after"] = record.reserved_mb_after
            row["peak_allocated_mb"] = max(
                row["peak_allocated_mb"], record.peak_allocated_mb
            )
            row["peak_reserved_mb"] = max(
                row["peak_reserved_mb"], record.peak_reserved_mb
            )
            row["device_free_mb_after"] = record.device_free_mb_after
            row["device_total_mb"] = record.device_total_mb
            row["calls"] += 1
            row["failed_calls"] += float(record.status == "failed")
        return result

    def to_dict(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.records]


__all__ = ["StageProfileRecord", "StageProfiler"]
