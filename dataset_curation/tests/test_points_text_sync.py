from __future__ import annotations

# DATASET_CURATION_POINTS_TEXT_SYNC_V1

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

    # There must be dynamic sync calls for both feature-backed point layers.
    assert "_sync_points_data_and_features(\n            supervoxel_id_layer," in text
    assert "_sync_points_data_and_features(\n            cell_centers_layer," in text
