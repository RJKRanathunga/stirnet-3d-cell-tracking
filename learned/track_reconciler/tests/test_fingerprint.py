import torch

from learned.track_reconciler.config import FingerprintConfig
from learned.track_reconciler.model.fingerprint import CellFingerprintEncoder


def test_fingerprint_encoder_handles_small_3d_batch():
    cfg = FingerprintConfig(in_channels=5, embedding_dim=96)
    model = CellFingerprintEncoder(cfg)
    crops = torch.randn(3, 5, 12, 20, 20)
    out = model(crops)
    assert out.shape == (3, 96)
    assert torch.isfinite(out).all()
