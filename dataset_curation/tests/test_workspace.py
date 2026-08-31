
from pathlib import Path
from dataset_curation.workspace.sample import CurationSample

def test_workspace_resume(tmp_path: Path):
    source = tmp_path / "movie.zarr"
    source.mkdir()
    sample = CurationSample.create(
        sample_id="sample_a",
        source_zarr=source,
        root=tmp_path / "curation",
    )
    sample.ensure_annotation_set("main", base_inference_run="run_a")
    reopened = CurationSample.open(
        sample_id="sample_a",
        root=tmp_path / "curation",
    )
    assert reopened.source_zarr == source.resolve()
    assert reopened.layout.instance_annotations("main").is_dir()
    assert reopened.layout.track_annotations("main").is_dir()
    assert reopened.layout.point_annotations("main").is_dir()
