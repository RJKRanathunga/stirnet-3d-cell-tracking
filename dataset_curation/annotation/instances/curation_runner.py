from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_curation.catalog import VolumeRecord
from dataset_curation.errors import ArtifactError
from dataset_curation.annotation.selection import (
    ensure_annotation_binding,
    touch_annotation_session,
)


def _select_frames(
    array: np.ndarray,
    timepoints: tuple[int, ...],
) -> np.ndarray:
    if timepoints == tuple(range(int(array.shape[0]))):
        return np.asarray(array)
    return np.stack(
        [np.asarray(array[int(t)]) for t in timepoints],
        axis=0,
    )


def run_instance_annotation(
    record: VolumeRecord,
    *,
    run_id: str = "current",
    annotation_set: str = "main",
    timepoint_selection: str = "all",
    suspect_threshold: float = 0.70,
    resume: bool = True,
) -> None:
    """
    Open the exact current merged-cell annotator on a curation inference run.

    This reuses AnnotationSession + make_viewer from the preserved annotator,
    but loads the standardized external-drive inference artifacts directly.
    """
    from dataset_curation._compat import instance_annotator as impl

    paths = record.paths

    if not paths.inference_complete(
        run_id,
        frame_count=record.frame_count,
    ):
        raise ArtifactError(
            f"Volume {record.volume_id} does not have a complete "
            f"inference run {run_id!r}."
        )

    required = (
        paths.raw(run_id),
        paths.binary_mask(run_id),
        paths.supervoxels(run_id),
        paths.final_instances(run_id),
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ArtifactError(
            "Instance annotation artifacts are incomplete:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    raw_movie = np.load(
        paths.raw(run_id),
        mmap_mode="r",
        allow_pickle=False,
    )
    binary_movie = np.load(
        paths.binary_mask(run_id),
        mmap_mode="r",
        allow_pickle=False,
    )
    supervoxel_movie = np.load(
        paths.supervoxels(run_id),
        mmap_mode="r",
        allow_pickle=False,
    )
    instance_movie = np.load(
        paths.final_instances(run_id),
        mmap_mode="r",
        allow_pickle=False,
    )

    frame_count = int(raw_movie.shape[0])
    available = list(range(frame_count))
    timepoints = impl.parse_timepoint_selection(
        timepoint_selection,
        available,
    )

    raw = _select_frames(raw_movie, timepoints)
    stage6_binary_mask = _select_frames(
        binary_movie,
        timepoints,
    ).astype(np.uint8, copy=False)
    supervoxels = _select_frames(
        supervoxel_movie,
        timepoints,
    ).astype(np.int32, copy=False)
    instances = _select_frames(
        instance_movie,
        timepoints,
    ).astype(np.int32, copy=False)

    foreground = instances > 0

    impl.validate_stacks(
        raw,
        supervoxels,
        instances,
        foreground,
    )

    if stage6_binary_mask.shape != instances.shape:
        raise ArtifactError(
            "Binary-mask/inference shape mismatch: "
            f"{stage6_binary_mask.shape} vs {instances.shape}"
        )

    suspect_instances = None
    suspect_root = paths.suspect_scores(run_id)
    if suspect_root.is_dir():
        try:
            suspect_instances, stats = impl.load_suspect_instance_frames(
                suspect_root=suspect_root,
                timepoints=timepoints,
                instances=instances,
                threshold=float(suspect_threshold),
            )
            print(
                "[suspects] loaded "
                f"{stats['displayed_instances']} displayed instances"
            )
        except FileNotFoundError as exc:
            print(
                "[suspects] incomplete suspect cache; layer disabled."
            )
            print(exc)
            suspect_instances = None

    ensure_annotation_binding(
        record,
        run_id=run_id,
        annotation_set=annotation_set,
    )
    touch_annotation_session(
        record,
        kind="instances",
        annotation_set=annotation_set,
        run_id=run_id,
    )

    output_dir = paths.instance_annotations(annotation_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = impl.AnnotationSession(
        sample_id=record.volume_id,
        timepoints=timepoints,
        supervoxels=supervoxels,
        base_instances=instances,
        output_dir=output_dir,
        resume=bool(resume),
    )

    print("=" * 88)
    print("DATASET CURATION — INSTANCE ANNOTATION")
    print("=" * 88)
    print(f"split            : {record.split}")
    print(f"volume           : {record.volume_id}")
    print(f"source           : {paths.zarr}")
    print(f"inference run    : {paths.inference_run(run_id)}")
    print(f"timepoints       : {timepoints}")
    print(f"annotations      : {output_dir}")
    print(f"resume           : {bool(resume)}")
    print("=" * 88)

    impl.make_viewer(
        sample_id=record.volume_id,
        timepoints=timepoints,
        raw=raw,
        stage6_binary_mask=stage6_binary_mask,
        supervoxels=supervoxels,
        foreground=foreground,
        session=session,
        suspect_instances=suspect_instances,
    )

    impl.napari.run()
