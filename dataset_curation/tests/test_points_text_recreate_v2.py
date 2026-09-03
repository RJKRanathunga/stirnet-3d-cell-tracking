from __future__ import annotations

# DATASET_CURATION_POINTS_TEXT_RECREATE_V2

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "dataset_curation" / "annotation" / "viewer.py"


def test_dynamic_feature_text_layers_recreate_on_count_change():
    text = VIEWER.read_text(encoding="utf-8")

    assert "def _sync_feature_text_points_layer(" in text
    assert "if old_count != row_count:" in text
    assert "viewer.layers.remove(" in text
    assert "viewer.add_points(" in text

    # Current-frame dynamic text layers must use the recreate-aware helper.
    assert (
        "supervoxel_id_layer = _sync_feature_text_points_layer("
        in text
    )
    assert (
        "cell_centers_layer = _sync_feature_text_points_layer("
        in text
    )


def test_dynamic_text_layer_references_are_rebound_after_recreation():
    text = VIEWER.read_text(encoding="utf-8")

    # The refresh closure must be able to replace the actual layer objects.
    assert (
        "nonlocal supervoxel_id_layer, cell_centers_layer"
        in text
    )

    # Global all-centroid text can also change row count after graph edits.
    assert "nonlocal all_centers_layer" in text
    assert (
        "all_centers_layer = _sync_feature_text_points_layer("
        in text
    )


def test_v1_in_place_sync_is_retained_for_equal_point_counts():
    text = VIEWER.read_text(encoding="utf-8")

    assert "_sync_points_data_and_features(" in text
    assert "return layer" in text
