from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    result: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result.add(node.module)

    return result


def test_no_compat_or_workspace_architecture_remains():
    package = ROOT / "dataset_curation"

    assert not (package / "_compat").exists()
    assert not (package / "workspace").exists()
    assert not (package / "preprocessing").exists()
    assert not (package / "io" / "legacy.py").exists()
    assert not (
        package / "inference" / "backends" / "investigation36.py"
    ).exists()
    assert not (package / "annotation" / "points").exists()

    obsolete_entrypoints = (
        ROOT
        / "evaluation"
        / "segmentation"
        / "scripts"
        / "02_supervoxel_instance_annotator.py",
        ROOT
        / "evaluation"
        / "segmentation"
        / "scripts"
        / "03_biohub_merge_suspect_export.py",
        ROOT
        / "evaluation"
        / "segmentation"
        / "scripts"
        / "annotate_points.py",
        ROOT
        / "evaluation"
        / "track_annotation"
        / "01_track_annotator.py",
    )
    assert not any(path.exists() for path in obsolete_entrypoints)


def test_dataset_curation_has_no_legacy_runtime_imports():
    package = ROOT / "dataset_curation"

    forbidden_prefixes = (
        "dataset_curation._compat",
        "dataset_curation.workspace",
        "dataset_curation.preprocessing",
        "dataset_curation.io.legacy",
        "dataset_curation.inference.backends.investigation36",
        "investigations",
    )

    offenders: list[str] = []

    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for module in _imports(path):
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in forbidden_prefixes
            ):
                offenders.append(
                    f"{path.relative_to(ROOT)} -> {module}"
                )

    assert not offenders, "\n".join(offenders)
