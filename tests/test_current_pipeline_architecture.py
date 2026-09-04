from __future__ import annotations

import ast
from pathlib import Path

from src.pipeline import (
    PipelineRequest,
    PipelineStages,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]


def _python_imports(path: Path) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_historical_numbered_src_stages_are_not_active_packages():
    historical = (
        "01_preprocessing",
        "02_masking",
        "03_segmentation",
        "04_detection",
        "05_feature_extraction",
        "07_cell_tracking",
        "08_track_stitching",
        "09_visualization",
        "10_cell_lineage",
        "11_track_reconciliation",
        "12_final_visualization",
    )
    for name in historical:
        assert not (ROOT / "src" / name).exists(), name


def test_semantic_current_pipeline_packages_exist():
    expected = (
        ROOT / "src" / "source_instances",
        ROOT / "src" / "tracking",
        ROOT / "src" / "pipeline",
        ROOT / "learned" / "stirnet",
        ROOT / "learned" / "track_reconciler",
        ROOT / "legacy" / "classical_pipeline",
    )
    for path in expected:
        assert path.is_dir(), path


def test_active_runtime_does_not_import_classical_legacy():
    active_roots = (
        ROOT / "src" / "source_instances",
        ROOT / "src" / "tracking",
        ROOT / "src" / "pipeline",
        ROOT / "learned" / "stirnet" / "inference",
        ROOT / "dataset_curation" / "inference",
    )
    violations: list[str] = []
    for base in active_roots:
        for path in base.rglob("*.py"):
            for module in _python_imports(path):
                if (
                    module == "legacy.classical_pipeline"
                    or module.startswith("legacy.classical_pipeline.")
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)} -> {module}"
                    )
    assert not violations, violations


def test_active_runtime_does_not_import_old_numbered_src_modules():
    forbidden = (
        "src.01_preprocessing",
        "src.02_masking",
        "src.03_segmentation",
        "src.04_detection",
        "src.05_feature_extraction",
        "src.07_cell_tracking",
        "src.08_track_stitching",
        "src.09_visualization",
        "src.10_cell_lineage",
        "src.11_track_reconciliation",
        "src.12_final_visualization",
    )
    violations: list[str] = []
    active_roots = (
        ROOT / "src" / "source_instances",
        ROOT / "src" / "tracking",
        ROOT / "src" / "pipeline",
        ROOT / "learned" / "stirnet" / "inference",
        ROOT / "dataset_curation" / "inference",
    )
    for base in active_roots:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    violations.append(
                        f"{path.relative_to(ROOT)} contains {token}"
                    )
    assert not violations, violations


def test_curation_uses_active_trackastra_runner():
    path = (
        ROOT
        / "dataset_curation"
        / "inference"
        / "trackastra.py"
    )
    modules = _python_imports(path)
    assert "src.tracking" in modules
    text = path.read_text(encoding="utf-8")
    assert "run_trackastra_core" in text


def test_current_pipeline_stage_order_is_explicit(tmp_path):
    calls: list[str] = []

    def source(request):
        calls.append("source_instances")
        return "source"

    def spatial(request, source_state):
        assert source_state == "source"
        calls.append("spatial_refinement")
        return "spatial"

    def tracking(request, spatial_state):
        assert spatial_state == "spatial"
        calls.append("primary_tracking")
        return "tracks"

    def stitching(request, spatial_state, tracking_state):
        assert spatial_state == "spatial"
        assert tracking_state == "tracks"
        calls.append("track_stitching")
        return "stitched"

    def export(request, spatial_state, stitching_state):
        assert spatial_state == "spatial"
        assert stitching_state == "stitched"
        calls.append("export")
        return "final"

    result = run_pipeline(
        PipelineRequest(
            source=tmp_path / "sample.zarr",
            output_directory=tmp_path / "out",
            sample_id="sample",
        ),
        PipelineStages(
            source_instances=source,
            spatial_refinement=spatial,
            primary_tracking=tracking,
            track_stitching=stitching,
            export=export,
        ),
    )

    assert calls == [
        "source_instances",
        "spatial_refinement",
        "primary_tracking",
        "track_stitching",
        "export",
    ]
    assert result.source_instances == "source"
    assert result.spatial_refinement == "spatial"
    assert result.primary_tracking == "tracks"
    assert result.track_stitching == "stitched"
    assert result.exported == "final"
