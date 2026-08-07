import pandas as pd

from learned.instance_segmentation.datasets.merge_real.fusion import fuse_candidates


def test_fusion_deduplicates_sources_and_promotes_source2_to_tier_a():
    source = pd.DataFrame([
        {
            "candidate_id": "s_t001_c00001", "sample_id": "s", "frame": 1, "cell_id": 1,
            "source": "source3_disappearing_track", "track_a": 1, "track_b": 2,
            "involved_track_ids": "1;2", "candidate_volume": 1000, "volume_a_reference": 500,
            "volume_b_reference": 500, "volume_sum_ratio": 1.0, "volume_sum_log_error": 0.0,
            "prediction_a_distance_um": 0.0, "prediction_b_distance_um": 0.0,
            "prediction_a_inside": True, "prediction_b_inside": True, "edt_peak_count": 0,
            "source_score": 3.0, "notes": "s3",
        },
        {
            "candidate_id": "s_t001_c00001", "sample_id": "s", "frame": 1, "cell_id": 1,
            "source": "source2_two_one_two", "track_a": 1, "track_b": 2,
            "involved_track_ids": "1;2", "candidate_volume": 1000, "volume_a_reference": 500,
            "volume_b_reference": 500, "volume_sum_ratio": 1.0, "volume_sum_log_error": 0.0,
            "prediction_a_distance_um": 0.0, "prediction_b_distance_um": 0.0,
            "prediction_a_inside": True, "prediction_b_inside": True, "edt_peak_count": 0,
            "source_score": 5.0, "notes": "s2",
        },
    ])
    fused = fuse_candidates(source)
    assert len(fused) == 1
    assert fused.iloc[0].tier == "A"
    assert fused.iloc[0].source_count == 2
    assert fused.iloc[0].has_source2_two_one_two
