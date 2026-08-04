from pathlib import Path

import numpy as np
import pandas as pd

from investigations.stage_10_cell_lineage.single_frame_division_phenotype.config import (
    PhenotypeInvestigationConfig,
)
from investigations.stage_10_cell_lineage.single_frame_division_phenotype.feature_extraction import (
    extract_frame_features,
)
from investigations.stage_10_cell_lineage.single_frame_division_phenotype.models import (
    PhenotypeCase,
)
from investigations.stage_10_cell_lineage.single_frame_division_phenotype.population import (
    attach_target_features,
    compare_targets_to_populations,
    summarize_feature_effects,
)
from investigations.stage_10_cell_lineage.single_frame_division_phenotype.repository_io import (
    FrameArtifacts,
    RepositoryData,
)
from investigations.stage_10_cell_lineage.single_frame_division_phenotype.scene_labels import (
    build_target_labels,
)


class FakeRepository:
    def track_metadata(self, frame, cell_id):
        return {
            "track_id": int(cell_id) + 100,
            "is_virtual_merge": False,
            "track_starts_here": False,
            "track_ends_here": False,
            "track_observation_count": 5,
        }

    def known_lineage_role(self, track_id):
        return ""

    def overlaps_segmentation_event(self, frame, cell_id, track_id):
        return False


def test_build_target_labels_keeps_primary_and_secondary_roles_separate():
    case = PhenotypeCase(
        case_id="001",
        scene_path=Path("001"),
        sample_id="sample",
        selected_cells={3: (10,), 4: (11,), 5: (20, 21), 6: (22, 23)},
        frame_numbers=(3, 4, 5, 6),
        event_frame=5,
        previous_parent_frame=4,
    )
    labels = build_target_labels([case])
    assert labels.loc[labels["frame"].eq(4), "phenotype_subtype"].tolist() == ["parent_final"]
    assert set(labels.loc[labels["frame"].eq(5), "phenotype_subtype"]) == {"daughter_birth"}
    assert labels["is_primary_target"].sum() == 3


def test_static_extractor_processes_every_cell_with_identical_schema():
    labels = np.zeros((9, 25, 25), dtype=np.int32)
    labels[2:6, 3:8, 3:8] = 1
    labels[2:5, 11:15, 4:8] = 2
    labels[3:7, 14:20, 14:20] = 3
    raw = np.full(labels.shape, 5.0, dtype=np.float32)
    raw[labels == 1] = 40.0
    raw[labels == 2] = 20.0
    raw[labels == 3] = 12.0
    raw[4, 5, 5] = 90.0
    preprocessed = raw / raw.max()
    cells = pd.DataFrame({"cell_id": [1, 2, 3]})
    artifacts = FrameArtifacts(
        sample_id="sample",
        frame=4,
        raw=raw,
        preprocessed=preprocessed,
        binary_mask=labels > 0,
        instance_labels=labels,
        cells=cells,
    )
    config = PhenotypeInvestigationConfig(create_plots=False, create_galleries=False)
    table = extract_frame_features(artifacts, FakeRepository(), config)
    assert table["cell_id"].tolist() == [1, 2, 3]
    assert {"volume_um3", "raw_mask_mean", "raw_internal_peak_count"}.issubset(table.columns)
    bright = table.set_index("cell_id").loc[1]
    ordinary = table.set_index("cell_id").loc[3]
    assert bright["raw_mask_mean"] > ordinary["raw_mask_mean"]


def test_population_comparison_uses_all_cells_and_keeps_size_analysis_separate():
    features = pd.DataFrame(
        {
            "sample_id": ["s"] * 5,
            "frame": [7] * 5,
            "cell_id": [1, 2, 3, 4, 5],
            "track_id": [101, 102, 103, 104, 105],
            "is_virtual_merge": [False] * 5,
            "track_starts_here": [False] * 5,
            "track_ends_here": [False] * 5,
            "track_observation_count": [5] * 5,
            "boundary_distance_um": [10.0] * 5,
            "is_boundary": [False] * 5,
            "known_lineage_role": [""] * 5,
            "is_known_lineage": [False] * 5,
            "overlaps_segmentation_event": [False] * 5,
            "centroid_z_um": [10, 11, 12, 13, 14],
            "centroid_y_um": [10, 10, 10, 10, 10],
            "centroid_x_um": [10, 10, 10, 10, 10],
            "volume_um3": [20, 40, 60, 80, 100],
            "volume_voxels": [20, 40, 60, 80, 100],
            "raw_mask_mean": [100, 20, 25, 30, 35],
        }
    )
    targets = pd.DataFrame(
        {
            "case_id": ["001"],
            "scene_path": ["001"],
            "sample_id": ["s"],
            "frame": [7],
            "cell_id": [1],
            "phenotype_role": ["daughter"],
            "phenotype_subtype": ["daughter_birth"],
            "daughter_ordinal": [0],
            "event_frame": [7],
            "relative_frame": [0],
            "is_primary_target": [True],
        }
    )
    target_features = attach_target_features(targets, features)
    config = PhenotypeInvestigationConfig(
        minimum_size_model_controls=3,
        create_plots=False,
        create_galleries=False,
    )
    contrasts, manifest = compare_targets_to_populations(target_features, features, config)
    row = contrasts.loc[
        contrasts["feature"].eq("raw_mask_mean")
        & contrasts["population"].eq("all_other_cells")
    ].iloc[0]
    assert row["control_count"] == 4
    assert row["percentile_rank"] == 1.0
    size_row = contrasts.loc[
        contrasts["feature"].eq("raw_mask_mean")
        & contrasts["population"].eq("clean_normal_cells")
    ].iloc[0]
    assert np.isfinite(size_row["size_adjusted_residual"])
    summary = summarize_feature_effects(contrasts)
    assert "raw_mask_mean" in set(summary["feature"])
    assert manifest.iloc[0]["all_other_count"] == 4


def test_empty_optional_segmentation_events_csv_is_treated_as_empty(tmp_path):
    class Paths:
        stage8_stitching = tmp_path / "stage8"
        stage10_lineage = tmp_path / "stage10"
        stage7_tracking = tmp_path / "stage7"

    Paths.stage8_stitching.mkdir(parents=True)
    (Paths.stage8_stitching / "segmentation_events.csv").write_bytes(b"")
    repository = RepositoryData(Paths())
    assert repository.segmentation_events().empty
    assert not repository.overlaps_segmentation_event(1, 2, 3)


def test_empty_optional_protected_tracks_csv_is_treated_as_empty(tmp_path):
    class Paths:
        stage8_stitching = tmp_path / "stage8"
        stage10_lineage = tmp_path / "stage10"
        stage7_tracking = tmp_path / "stage7"

    Paths.stage10_lineage.mkdir(parents=True)
    (Paths.stage10_lineage / "protected_tracks.csv").write_bytes(b"")
    repository = RepositoryData(Paths())
    assert repository.protected_tracks().empty
    assert repository.known_lineage_role(7) == ""
