from __future__ import annotations

# DATASET_CURATION_LAZY_RUNTIME_IMPORTS_V1
# DATASET_CURATION_COMPACT_CACHE_V1

from dataset_curation.annotation.selection import (
    ensure_annotation_binding,
    touch_annotation_session,
)
from dataset_curation.annotation.tracks.storage import (
    OutputPaths,
    SourcePaths,
)
from dataset_curation.catalog import VolumeRecord
from dataset_curation.config import DEFAULT_SPACING_ZYX_UM
from dataset_curation.errors import ArtifactError


DEFAULT_MAX_RAY_DISTANCE_UM = 8.0


def run_track_annotation(
    record: VolumeRecord,
    *,
    run_id: str = "current",
    annotation_set: str = "main",
    max_ray_distance_um: float = DEFAULT_MAX_RAY_DISTANCE_UM,
    resume: bool = True,
) -> None:
    """Open one inference-ready volume in the production track annotator."""
    from dataset_curation.annotation.tracks.viewer import open_viewer

    paths = record.paths

    if not paths.inference_complete(
        run_id,
        frame_count=record.frame_count,
    ):
        raise ArtifactError(
            f"Volume {record.volume_id} does not have a complete "
            f"inference run {run_id!r}."
        )

    source = SourcePaths(
        paths.inference_run(run_id)
    )
    output = OutputPaths(
        paths.track_annotations(
            annotation_set
        )
    )
    output.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = (
        source.final_instances,
        source.cells_csv,
        source.napari_graph,
        source.tracks_csv,
    )
    missing = [
        path
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise ArtifactError(
            "Track annotation artifacts are incomplete:\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )

    ensure_annotation_binding(
        record,
        run_id=run_id,
        annotation_set=annotation_set,
    )
    touch_annotation_session(
        record,
        kind="tracks",
        annotation_set=annotation_set,
        run_id=run_id,
    )

    print("=" * 88)
    print("DATASET CURATION — TRACK ANNOTATION")
    print("=" * 88)
    print(f"split          : {record.split}")
    print(f"volume         : {record.volume_id}")
    print(f"source Zarr    : {paths.zarr}")
    print(f"inference run  : {paths.inference_run(run_id)}")
    print(f"annotations    : {output.root}")
    print(f"resume         : {bool(resume)}")
    print("=" * 88)

    open_viewer(
        source=source,
        source_zarr=paths.zarr,
        output=output,
        sample_id=record.volume_id,
        spacing_zyx=DEFAULT_SPACING_ZYX_UM,
        max_ray_distance_um=float(
            max_ray_distance_um
        ),
        resume=bool(resume),
    )
