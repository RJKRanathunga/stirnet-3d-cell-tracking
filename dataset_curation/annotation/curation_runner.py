from __future__ import annotations

# DATASET_CURATION_RAW_RAY_BIRTH_AUTOHIDE_V1

# DATASET_CURATION_CANONICAL_SKIP_V1

"""One-volume unified spatial + tracking curation runner."""

import json

import numpy as np
import pandas as pd

from dataset_curation.annotation.instances.centers import (
    frame_instance_centers,
)
from dataset_curation.annotation.instances.session import AnnotationSession
from dataset_curation.annotation.selection import (
    ensure_annotation_binding,
    touch_annotation_session,
)
from dataset_curation.annotation.source_data import (
    BinaryMaskFrameCache,
    open_source_movie,
)
from dataset_curation.annotation.tracks.diagnostics import (
    normalize_cells,
    normalize_tracks,
    prepare_endpoint_track_groups,
)
from dataset_curation.annotation.tracks.graph import (
    Node,
    build_trackastra_detection_edges,
)
from dataset_curation.annotation.tracks.session import TrackAnnotationSession
from dataset_curation.annotation.tracks.storage import OutputPaths
from dataset_curation.catalog import VolumeRecord
from dataset_curation.config import DEFAULT_SPACING_ZYX_UM
from dataset_curation.errors import ArtifactError


def _replace_frame_nodes_and_centers(
    valid_nodes: set[Node],
    centers: dict[Node, np.ndarray],
    *,
    frame: int,
    labels_zyx: np.ndarray,
) -> None:
    frame = int(frame)

    stale_nodes = {
        node
        for node in valid_nodes
        if node[0] == frame
    }
    valid_nodes.difference_update(
        stale_nodes
    )

    for node in [
        node
        for node in centers
        if node[0] == frame
    ]:
        del centers[node]

    corrected_centers = (
        frame_instance_centers(
            labels_zyx
        )
    )
    for instance_id, center in (
        corrected_centers.items()
    ):
        node = (
            frame,
            int(instance_id),
        )
        valid_nodes.add(node)
        centers[node] = np.asarray(
            center,
            dtype=np.float64,
        )


def run_annotation(
    record: VolumeRecord,
    *,
    annotation_set: str = "main",
    boundary_margin_um: float = 4.0,
    resume: bool = True,
) -> None:
    """
    Open the unified curation viewer.

    PyTorch is intentionally not imported by this command. Raw data remains
    lazy in source Zarr; persisted spatial movies remain uint16 memmaps.
    """
    import napari
    from dataset_curation.annotation.viewer import make_viewer

    paths = record.paths
    if not paths.inference_complete(
        frame_count=record.frame_count,
    ):
        raise ArtifactError(
            f"Volume {record.volume_id} does not have complete "
            "canonical inference."
        )

    required = (
        paths.supervoxels,
        paths.final_instances,
        paths.cells_csv,
        paths.tracks_csv,
        paths.napari_graph,
    )
    missing = [
        path
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise ArtifactError(
            "Unified annotation artifacts are incomplete:\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )

    raw = open_source_movie(
        paths.zarr
    )
    supervoxels = np.load(
        paths.supervoxels,
        mmap_mode="r",
        allow_pickle=False,
    )
    base_instances = np.load(
        paths.final_instances,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_shape = tuple(
        int(v)
        for v in raw.shape
    )
    if (
        tuple(
            int(v)
            for v in supervoxels.shape
        )
        != expected_shape
        or tuple(
            int(v)
            for v in base_instances.shape
        )
        != expected_shape
    ):
        raise ArtifactError(
            "Raw/supervoxel/final-instance movies do not align: "
            f"raw={expected_shape}, "
            f"SV={supervoxels.shape}, "
            f"instances={base_instances.shape}"
        )

    frame_count = int(
        expected_shape[0]
    )
    timepoints = tuple(
        range(frame_count)
    )

    unified_marker = (
        paths.annotation_set(annotation_set)
        / "_session.json"
    )
    resume_unified_state = (
        bool(resume)
        and unified_marker.is_file()
    )

    ensure_annotation_binding(
        record,
        annotation_set=annotation_set,
    )
    touch_annotation_session(
        record,
        annotation_set=annotation_set,
    )

    spatial_output = (
        paths.instance_annotations(
            annotation_set
        )
    )
    track_output = OutputPaths(
        paths.track_annotations(
            annotation_set
        )
    )
    spatial_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    track_output.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    spatial_session = AnnotationSession(
        sample_id=record.volume_id,
        timepoints=timepoints,
        supervoxels=supervoxels,
        base_instances=base_instances,
        output_dir=spatial_output,
        resume=resume_unified_state,
    )

    cells = normalize_cells(
        pd.read_csv(
            paths.cells_csv
        )
    )
    tracks = normalize_tracks(
        pd.read_csv(
            paths.tracks_csv
        )
    )
    lineage = json.loads(
        paths.napari_graph.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        lineage,
        dict,
    ):
        raise ArtifactError(
            f"Expected lineage dict in "
            f"{paths.napari_graph}."
        )

    valid_nodes: set[Node] = {
        (
            int(row.frame),
            int(row.cell_id),
        )
        for row in cells.itertuples(
            index=False
        )
    }
    centers: dict[
        Node,
        np.ndarray,
    ] = {
        (
            int(row.frame),
            int(row.cell_id),
        ): np.asarray(
            [
                row.centroid_z,
                row.centroid_y,
                row.centroid_x,
            ],
            dtype=np.float64,
        )
        for row in cells.itertuples(
            index=False
        )
    }

    # Resume-time spatial edits are authoritative over Trackastra's original
    # cell table. Only changed frames need to be rescanned.
    for local_t in (
        spatial_session.changed_local_indices()
    ):
        _replace_frame_nodes_and_centers(
            valid_nodes,
            centers,
            frame=int(local_t),
            labels_zyx=spatial_session.frame(
                local_t
            ),
        )

    base_edges = (
        build_trackastra_detection_edges(
            tracks,
            lineage,
        )
    )

    diagnostics = (
        prepare_endpoint_track_groups(
            tracks,
            cells,
            expected_shape[-3:],
            voxel_size_zyx=DEFAULT_SPACING_ZYX_UM,
            boundary_margin_um=float(
                boundary_margin_um
            ),
        )
    )

    boundary_entry_nodes = {
        (
            int(row.frame),
            int(row.cell_id),
        )
        for row in diagnostics.new_track_endpoints.itertuples(
            index=False
        )
        if bool(
            row.is_boundary_endpoint
        )
    }
    boundary_exit_nodes = {
        (
            int(row.frame),
            int(row.cell_id),
        )
        for row in diagnostics.ended_track_endpoints.itertuples(
            index=False
        )
        if bool(
            row.is_boundary_endpoint
        )
    }

    track_session = TrackAnnotationSession(
        sample_id=record.volume_id,
        source_root=paths.preprocessed_root,
        output=track_output,
        valid_nodes=valid_nodes,
        base_edges=base_edges,
        frame_count=frame_count,
        boundary_entry_nodes=boundary_entry_nodes,
        boundary_exit_nodes=boundary_exit_nodes,
        resume=resume_unified_state,
    )

    binary_cache = BinaryMaskFrameCache(
        paths.zarr,
        max_frames=3,
    )

    print("=" * 96)
    print("DATASET CURATION — UNIFIED ANNOTATION")
    print("=" * 96)
    print(f"split          : {record.split}")
    print(f"volume         : {record.volume_id}")
    print(f"source Zarr    : {paths.zarr}")
    print(f"inference run  : {paths.preprocessed_root}")
    print(f"annotations    : {paths.annotation_set(annotation_set)}")
    print(f"resume state   : {resume_unified_state}")
    print(
        "diagnostics    : notebook-09 broken/new/boundary endpoint groups"
    )
    print("=" * 96)

    make_viewer(
        sample_id=record.volume_id,
        raw=raw,
        binary_cache=binary_cache,
        supervoxels=supervoxels,
        spatial_session=spatial_session,
        track_session=track_session,
        track_centers=centers,
        original_tracks=tracks,
        diagnostics=diagnostics,
        spacing_zyx=DEFAULT_SPACING_ZYX_UM,
    )
    napari.run()
