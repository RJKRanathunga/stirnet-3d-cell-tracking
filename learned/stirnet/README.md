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
from stirnet import StirNet, StirNetConfig

cfg = StirNetConfig()
model = StirNet(cfg)
```

## Training

Prepare cached `.pt` samples following the data contract, put paths in a text file, then:

```bash
python -m stirnet.training.train --train-list train.txt --val-list val.txt --out runs/stirnet_v1
```

For mixed datasets, use spacing/shape buckets. The reference CLI defaults to `batch_size=1`, which is always safe for variable native patch shapes.

## Important integration contract

Coordinates passed into the model are relative to the patch center and use `(z, y, x)` order. Physical positions are in micrometres. Instance and GT center features also have cell-scale-normalized variants using `dref_um`.

The first implementation milestone should still be spatial-only pretraining. Temporal graph/co-reasoning modules are already implemented and can be enabled in the same model; empty temporal tensors are supported.
