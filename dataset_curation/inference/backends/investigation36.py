from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dataset_curation._repo import repo_root
from dataset_curation.catalog import VolumeRecord
from dataset_curation.errors import ArtifactError
from dataset_curation.io.atomic import atomic_json
from dataset_curation.workspace.sample import _now


class Investigation36Backend:
    """
    Adapter around the existing current STIR-Net + Trackastra implementation.

    `run()` is retained for the first-stage CurationSample API.
    `run_volume()` is the external-drive batch workflow.
    """

    name = "investigation36_current_stirnet_trackastra"

    @staticmethod
    def script_path() -> Path:
        return (
            repo_root()
            / "investigations"
            / "stirnet"
            / "36_biohub_spatial_trackastra_visualization.py"
        )

    def run(self, sample, *, run_id: str, extra_args=()):
        """Backwards-compatible first-stage workspace entry point."""
        output = sample.ensure_inference_run(run_id)
        script = self.script_path()
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
            "supervoxels": str(output / "movies" / "supervoxels.npy"),
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
            Path(artifacts["supervoxels"]),
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

    def run_volume(
        self,
        record: VolumeRecord,
        *,
        run_id: str = "current",
        force: bool = False,
        extra_args=(),
    ) -> Path:
        paths = record.paths
        paths.ensure_output_roots()

        if record.frame_count is None:
            raise ArtifactError(
                f"Could not determine frame count for {record.volume_id} "
                f"from Zarr metadata below {paths.zarr}."
            )

        if (
            not force
            and paths.inference_complete(
                run_id,
                frame_count=record.frame_count,
            )
        ):
            print(
                f"[skip] {record.split}/{record.volume_id}: "
                f"run {run_id!r} is already complete."
            )
            return paths.inference_run(run_id)

        output = paths.inference_run(run_id)
        output.mkdir(parents=True, exist_ok=True)

        script = self.script_path()
        if not script.is_file():
            raise FileNotFoundError(script)

        command = [
            sys.executable,
            str(script),
            "--sample-id", record.volume_id,
            "--sample-zarr", str(paths.zarr),
            "--frame-count", str(int(record.frame_count)),
            "--output", str(output),
            "--no-viewer",
        ]

        if force:
            command.extend(
                [
                    "--overwrite-spatial",
                    "--rebuild-trackastra",
                ]
            )

        command.extend(str(value) for value in extra_args)

        print(
            f"[inference] {record.split}/{record.volume_id} "
            f"frames={record.frame_count}"
        )
        print(f"[inference] source : {paths.zarr}")
        print(f"[inference] output : {output}")

        subprocess.run(
            command,
            cwd=repo_root(),
            check=True,
        )

        if not paths.inference_complete(
            run_id,
            frame_count=record.frame_count,
        ):
            raise ArtifactError(
                "Inference command returned successfully but the curation "
                f"contract is incomplete below {output}."
            )

        atomic_json(
            paths.inference_manifest(run_id),
            {
                "schema_version": 1,
                "kind": "biohub_curation_inference",
                "backend": self.name,
                "volume_id": record.volume_id,
                "split": record.split,
                "frame_count": int(record.frame_count),
                "run_id": str(run_id),
                "source_zarr": str(paths.zarr),
                "ground_truth_present": bool(record.has_ground_truth),
                "ground_truth_used_for_inference": False,
                "command": command,
                "artifacts": {
                    "raw": str(paths.raw(run_id)),
                    "preprocessed": str(paths.preprocessed(run_id)),
                    "binary_mask": str(paths.binary_mask(run_id)),
                    "source_instances": str(paths.source_instances(run_id)),
                    "supervoxels": str(paths.supervoxels(run_id)),
                    "final_instances": str(paths.final_instances(run_id)),
                    "cells_csv": str(paths.cells_csv(run_id)),
                    "trackastra": {
                        "track_graph": str(paths.track_graph(run_id)),
                        "tracked_masks": str(paths.tracked_masks(run_id)),
                        "napari_tracks": str(paths.napari_tracks(run_id)),
                        "napari_graph": str(paths.napari_graph(run_id)),
                        "tracks_csv": str(paths.tracks_csv(run_id)),
                    },
                },
                "created_at": _now(),
            },
        )

        return output
