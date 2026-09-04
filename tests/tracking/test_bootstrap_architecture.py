from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_production_tracking_has_no_investigation_or_annotation_dependency():
    violations: list[str] = []
    for path in (ROOT / "src" / "tracking").rglob("*.py"):
        for module in _imports(path):
            if (
                module == "investigations"
                or module.startswith("investigations.")
                or module == "dataset_curation.annotation"
                or module.startswith("dataset_curation.annotation.")
            ):
                violations.append(f"{path.relative_to(ROOT)} -> {module}")
    assert not violations, violations


def test_trackastra_public_api_is_package():
    package = ROOT / "src" / "tracking" / "trackastra"
    assert package.is_dir()
    assert (package / "__init__.py").is_file()
    assert not (ROOT / "src" / "tracking" / "trackastra.py").exists()
