from __future__ import annotations

from pathlib import Path

from learned.stirnet.inference.spatial_pipeline import (
    SpatialPreparationTimeout,
)


def test_timeout_exception_carries_frame_and_limit():
    error = SpatialPreparationTimeout(
        frame=7,
        timeout_seconds=300.0,
        elapsed_seconds=301.5,
    )

    assert error.frame == 7
    assert error.timeout_seconds == 300.0
    assert error.elapsed_seconds == 301.5
    assert "t=007" in str(error)
    assert "300.0s" in str(error)


def test_production_pipeline_uses_spawned_process_not_thread_executor():
    root = Path(__file__).resolve().parents[2]
    pipeline = (
        root
        / "learned"
        / "stirnet"
        / "inference"
        / "spatial_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" not in pipeline
    assert 'multiprocessing.get_context(\n            "spawn"' in pipeline
    assert "self._process.terminate()" in pipeline
    assert "SpatialPreparationTimeout" in pipeline
    assert "preparation_timeout_seconds: float = 300.0" in pipeline


def test_curation_backend_records_timeout_as_skip():
    root = Path(__file__).resolve().parents[2]
    backend = (
        root
        / "dataset_curation"
        / "inference"
        / "backends"
        / "stirnet_trackastra.py"
    ).read_text(encoding="utf-8")

    assert '"--frame-prep-timeout"' in backend
    assert "default=300.0" in backend
    assert 'reason_code = "preparation_timeout"' in backend
    assert "except SpatialPreparationTimeout as exc:" in backend
    assert "frame_preparation_timeout_seconds" in backend
