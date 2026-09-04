from __future__ import annotations

# DATASET_CURATION_POINTS_TEXT_SYNC_V1
# DATASET_CURATION_POINTS_TEXT_RECREATE_V2_TEST_UPDATE

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "dataset_curation" / "annotation" / "viewer.py"


def test_viewer_has_safe_points_data_feature_text_sync():
    text = VIEWER.read_text(encoding="utf-8")

    assert "def _sync_points_data_and_features(" in text
    assert "layer.text = None" in text
    assert "layer.features = pd.DataFrame(" in text

    # These were the unsafe dynamic update sites that could transiently leave
    # N point coordinates paired with M feature-backed text values.
    assert "supervoxel_id_layer.data = sv_points" not in text
    assert "cell_centers_layer.data = center_points" not in text
    assert "all_centers_layer.data = all_points_array" not in text
    assert "group.point_layer.data = points_array" not in text


def test_supervoxel_and_cell_text_are_restored_after_sync():
    text = VIEWER.read_text(encoding="utf-8")

    assert '"string": "{sv_id}"' in text
    assert '"string": "{instance_id}"' in text
    assert '"string": "{cell_id}"' in text

    # V2 keeps the V1 in-place synchronizer for equal point counts, but
    # feature-backed dynamic text layers now use the stronger recreate-aware
    # wrapper so stale Napari `_indices_view` entries cannot outlive a point
    # count change.
    assert "def _sync_points_data_and_features(" in text
    assert "def _sync_feature_text_points_layer(" in text
    assert (
        "supervoxel_id_layer = _sync_feature_text_points_layer("
        in text
    )
    assert (
        "cell_centers_layer = _sync_feature_text_points_layer("
        in text
    )
