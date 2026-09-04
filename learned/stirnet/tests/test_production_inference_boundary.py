from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(
                    node.module
                )
    return modules


def test_production_inference_does_not_import_investigations():
    paths = (
        ROOT / "learned" / "stirnet" / "inference" / "runtime.py",
        ROOT / "learned" / "stirnet" / "inference" / "spatial_input.py",
        ROOT / "learned" / "stirnet" / "inference" / "spatial_pipeline.py",
        ROOT / "kaggle" / "run_submission.py",
        ROOT / "dataset_curation" / "inference" / "backends"
        / "stirnet_trackastra.py",
    )
    for path in paths:
        modules = _imported_modules(path)
        forbidden = sorted(
            module
            for module in modules
            if module == "investigations"
            or module.startswith("investigations.")
        )
        assert not forbidden, (
            f"{path.relative_to(ROOT)} imports temporary investigation code: "
            f"{forbidden}"
        )


def test_parallel_scheduler_has_single_production_owner():
    production = (
        ROOT
        / "learned"
        / "stirnet"
        / "inference"
        / "spatial_pipeline.py"
    ).read_text(encoding="utf-8")
    kaggle = (
        ROOT
        / "kaggle"
        / "run_submission.py"
    ).read_text(encoding="utf-8")

    # Production moved from the temporary thread-pool scheduler to one
    # persistent spawned CPU preparation process. The main process remains the
    # sole CUDA owner, and Kaggle consumes the shared production implementation.
    assert "multiprocessing.get_context" in production
    assert "STIRNET_PARALLEL_SPATIAL_PIPELINE_V3" in production
    assert "ThreadPoolExecutor" not in production
    assert "ThreadPoolExecutor" not in kaggle
    assert "investigations/stirnet/data" not in kaggle
