from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def _top_level_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result

def test_cli_does_not_eagerly_load_runtimes():
    modules = _top_level_import_modules(ROOT / 'dataset_curation' / 'cli.py')
    assert 'dataset_curation.inference.backends.stirnet_trackastra' not in modules
    assert 'dataset_curation.annotation.instances.curation_runner' not in modules
    assert 'dataset_curation.annotation.tracks.curation_runner' not in modules

def test_annotation_runners_do_not_eagerly_import_viewers():
    instance_runner = ROOT / 'dataset_curation' / 'annotation' / 'instances' / 'curation_runner.py'
    track_runner = ROOT / 'dataset_curation' / 'annotation' / 'tracks' / 'curation_runner.py'
    instance_modules = _top_level_import_modules(instance_runner)
    track_modules = _top_level_import_modules(track_runner)
    assert 'napari' not in instance_modules
    assert 'dataset_curation.annotation.instances.viewer' not in instance_modules
    assert 'dataset_curation.annotation.tracks.viewer' not in track_modules

def test_importing_cli_does_not_load_torch_or_napari():
    code = (
        "import sys; import dataset_curation.cli; "
        "assert 'torch' not in sys.modules; "
        "assert 'napari' not in sys.modules"
    )
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
