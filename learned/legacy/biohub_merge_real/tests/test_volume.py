from learned.instance_segmentation.datasets.merge_real.config import MergeRealConfig
from learned.instance_segmentation.datasets.merge_real.evidence.volume import volume_sum_evidence


def test_volume_sum_prefers_conserved_pair():
    config = MergeRealConfig()
    evidence = volume_sum_evidence(980.0, 500.0, 520.0, config)
    assert evidence.broad_match
    assert evidence.strong_match
    assert abs(evidence.ratio - 980.0 / 1020.0) < 1e-12
    assert evidence.score > 0.8


def test_volume_sum_rejects_large_mismatch():
    config = MergeRealConfig()
    evidence = volume_sum_evidence(300.0, 500.0, 520.0, config)
    assert not evidence.broad_match
