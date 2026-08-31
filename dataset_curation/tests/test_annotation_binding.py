
from pathlib import Path
import pytest
from dataset_curation.errors import ManifestError
from dataset_curation.workspace.sample import CurationSample

def test_annotation_set_cannot_switch_base_inference(tmp_path: Path):
    source = tmp_path / "movie.zarr"
    source.mkdir()
    sample = CurationSample.create(
        sample_id="sample_a",
        source_zarr=source,
        root=tmp_path / "curation",
    )
    sample.ensure_annotation_set("main", base_inference_run="run_a")
    with pytest.raises(ManifestError):
        sample.ensure_annotation_set("main", base_inference_run="run_b")
