from __future__ import annotations

# DATASET_CURATION_CANONICAL_SKIP_V1
# DATASET_CURATION_EMPTY_TRACKS_SAFE_V1

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
            result.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(
            node,
            ast.ImportFrom,
        ) and node.module:
            result.add(node.module)
    return result


def test_no_legacy_or_separate_annotation_ui_remains():
    package = ROOT / "dataset_curation"

    assert not (
        package / "_compat"
    ).exists()
    assert not (
        package / "workspace"
    ).exists()
    assert not (
        package / "preprocessing"
    ).exists()
    assert not (
        package
        / "io"
        / "legacy.py"
    ).exists()
    assert not (
        package
        / "inference"
        / "backends"
        / "investigation36.py"
    ).exists()

    obsolete_unified_ui = (
        package
        / "annotation"
        / "instances"
        / "viewer.py",
        package
        / "annotation"
        / "instances"
        / "curation_runner.py",
        package
        / "annotation"
        / "instances"
        / "io.py",
        package
        / "annotation"
        / "tracks"
        / "viewer.py",
        package
        / "annotation"
        / "tracks"
        / "curation_runner.py",
    )
    assert not any(
        path.exists()
        for path in obsolete_unified_ui
    )

    assert (
        package
        / "annotation"
        / "viewer.py"
    ).is_file()
    assert (
        package
        / "annotation"
        / "curation_runner.py"
    ).is_file()
    assert (
        package
        / "visualization"
        / "source_viewer.py"
    ).is_file()


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
                module == prefix
                or module.startswith(
                    prefix + "."
                )
                for prefix in forbidden_prefixes
            ):
                offenders.append(
                    f"{path.relative_to(ROOT)} -> {module}"
                )

    assert not offenders, "\n".join(
        offenders
    )


def test_unified_viewer_has_requested_layer_and_control_contract():
    viewer = (
        ROOT
        / "dataset_curation"
        / "annotation"
        / "viewer.py"
    ).read_text(
        encoding="utf-8"
    )

    required = (
        "Supervoxel IDs",
        "Cell instance centers",
        "Broken Tracks",
        "Broken Track Centers",
        "New Tracks",
        "New Track Centers",
        "Boundary Entry Tracks",
        "Boundary Exit Tracks",
        "Hidden tracks",
        "Hallucination",
        "Continue Track",
        "Break Track",
        "Complete Track",
    )
    for token in required:
        assert token in viewer

    assert "SV number leader lines" not in viewer
    assert "add_shapes" not in viewer


def test_cli_exposes_only_unified_annotation_entrypoint():
    cli = (
        ROOT
        / "dataset_curation"
        / "cli.py"
    ).read_text(
        encoding="utf-8"
    )
    assert '"annotate"' in cli
    assert '"view-source"' in cli
    assert "annotate-instances" not in cli
    assert "annotate-tracks" not in cli
def test_dataset_curation_has_one_canonical_inference_root():
    package = ROOT / "dataset_curation"
    offenders = []
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "run_id" in text or "base_inference_run" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "legacy inference-run concepts remain:\n" + "\n".join(offenders)

    cli = (package / "cli.py").read_text(encoding="utf-8")
    assert "--run-id" not in cli


def test_quality_gate_is_before_expensive_source_segmentation():
    text = (
        ROOT / "learned" / "stirnet" / "inference" / "spatial_input.py"
    ).read_text(encoding="utf-8")
    assert text.index("source_mask_validator(") < text.index(
        "source_labels = segment_instances("
    )

def test_unified_viewer_never_constructs_empty_napari_tracks():
    viewer = (
        ROOT
        / "dataset_curation"
        / "annotation"
        / "viewer.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "def _sync_tracks_layer(" in viewer
    assert "if array.shape[0] == 0:" in viewer
    assert "viewer.layers.remove(" in viewer

    # Every Tracks-layer creation/update must pass through the empty-safe
    # lifecycle helper. The only direct add_tracks call is inside that helper.
    assert viewer.count("viewer.add_tracks(") == 1

    # The two dynamic corrected graph layers must be synchronized rather than
    # assigned an empty (0, 5) array directly.
    assert 'name="Corrected Tracks - active"' in viewer
    assert 'name="Hidden tracks"' in viewer
    assert "group.track_layer = (" in viewer

