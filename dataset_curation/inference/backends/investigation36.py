
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from dataset_curation._repo import repo_root
from dataset_curation.errors import ArtifactError
from dataset_curation.io.atomic import atomic_json
from dataset_curation.workspace.sample import _now

class Investigation36Backend:
    name = "investigation36_current_stirnet_trackastra"

    def run(self, sample, *, run_id: str, extra_args=()):
        output = sample.ensure_inference_run(run_id)
        script = (
            repo_root()
            / "investigations"
            / "stirnet"
            / "36_biohub_spatial_trackastra_visualization.py"
        )
        if not script.is_file():
            raise FileNotFoundError(script)

        command = [
            sys.executable,
            str(script),
            "--sample-id", sample.layout.sample_id,
            "--sample-zarr", str(sample.source_zarr),
            "--output", str(output),
            "--no-viewer",
            *extra_args,
        ]
        subprocess.run(command, cwd=repo_root(), check=True)

        artifacts = {
            "raw": str(output / "movies" / "raw.npy"),
            "preprocessed": str(output / "movies" / "preprocessed.npy"),
            "binary_mask": str(output / "movies" / "binary_mask.npy"),
            "source_instances": str(output / "movies" / "source_instances.npy"),
            "final_instances": str(output / "movies" / "final_instances.npy"),
            "cells_csv": str(output / "cells_all.csv"),
            "spatial_summary": str(output / "spatial_summary.json"),
            "trackastra": {
                "track_graph": str(output / "trackastra" / "track_graph.pkl"),
                "tracked_masks": str(output / "trackastra" / "tracked_masks.npy"),
                "napari_tracks": str(output / "trackastra" / "napari_tracks.npy"),
                "napari_graph": str(output / "trackastra" / "napari_graph.json"),
                "tracks_csv": str(output / "trackastra" / "tracks.csv"),
                "summary": str(output / "trackastra" / "summary.json"),
            },
        }
        required = [
            Path(artifacts["raw"]),
            Path(artifacts["binary_mask"]),
            Path(artifacts["final_instances"]),
            Path(artifacts["cells_csv"]),
            Path(artifacts["trackastra"]["napari_graph"]),
            Path(artifacts["trackastra"]["tracks_csv"]),
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ArtifactError(
                "Investigation 36 finished without required artifacts:\n"
                + "\n".join(f"  {path}" for path in missing)
            )

        atomic_json(
            sample.layout.inference_manifest(run_id),
            {
                "schema_version": 1,
                "kind": "inference_run",
                "sample_id": sample.layout.sample_id,
                "run_id": run_id,
                "backend": self.name,
                "immutable_base_prediction": True,
                "source_zarr": str(sample.source_zarr),
                "command": command,
                "artifacts": artifacts,
                "spacing_zyx_um": list(sample.spacing_zyx_um),
                "created_at": _now(),
            },
        )
        return output
