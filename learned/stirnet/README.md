# STIR-Net V1 implementation

This archive contains the planned V1 implementation for placement under `learned/stirnet/`.

```text
stirnet/
├── model/
├── data/
├── training/
└── inference/
```

The package is intentionally independent of Trackastra internals. Convert Trackastra pass-1 output to tensors with `stirnet.data.graph_builder` or your project-specific adapter/cache.

## Core dependencies

- Python 3.11+
- PyTorch
- NumPy
- SciPy

No `torch_geometric` dependency is required; the edge-aware GATv2-style layers are implemented in pure PyTorch.

## Minimal import

```python
from learned.stirnet import (
    RuntimeProfile,
    StirNet,
    StirNetConfig,
    apply_runtime_profile,
    describe_runtime_profile,
)

cfg = StirNetConfig()
apply_runtime_profile(cfg, RuntimeProfile.LOCAL_6GB)
model = StirNet(cfg)
print(describe_runtime_profile(cfg))
```

Select `local_6gb` or `cloud_24gb` explicitly before constructing the model and
criterion. A runtime profile changes activation recomputation, exact chunk
sizes, and the local-mask supervision cap; it does not change model parameter
shapes. Model/experiment settings such as reduced channel widths and
`decoder.max_spatial_tokens` remain orthogonal to the runtime profile. In
particular, `max_spatial_tokens` changes adaptive pooling and model-visible
information, while `local_masks.train_max_queries_per_batch` changes training
supervision density without changing checkpoint compatibility.

## Training

Prepare cached `.pt` samples following the data contract, put paths in a text file, then:

```bash
python -m stirnet.training.train --train-list train.txt --val-list val.txt --out runs/stirnet_v1
```

For mixed datasets, use spacing/shape buckets. The reference CLI defaults to `batch_size=1`, which is always safe for variable native patch shapes.

## Important integration contract

Coordinates passed into the model are relative to the patch center and use `(z, y, x)` order. Physical positions are in micrometres. Instance and GT center features also have cell-scale-normalized variants using `dref_um`.

The first implementation milestone should still be spatial-only pretraining. Temporal graph/co-reasoning modules are already implemented and can be enabled in the same model; empty temporal tensors are supported.
