# STIR-Net V1 implementation notes

This package implements the V1 architecture documented under `docs/stirnet/v1/`.

## Verified in this archive

- Package imports as both `stirnet` and `learned.stirnet`.
- Spatial-only and temporal-enabled forward passes run on synthetic 3D inputs.
- Query construction includes primary, split-companion, temporal, and discovery queries.
- Coarse mask decoding, Hungarian matching, native-resolution lazy mask rendering, and complete V1 criterion run.
- A full synthetic forward + loss + backward pass succeeds.
- Inference mask rendering and connected-component postprocessing run.
- Generic temporal graph construction and packed batch collation run.

Run the included test after placement under `learned/stirnet/`:

```bash
python -m learned.stirnet.smoke_test
```

## Project-specific integration still required

The neural model deliberately does not import Trackastra. The project integration layer must convert Trackastra pass-1 results into the generic detection/association records or directly into the cached graph tensors accepted by this package. `data/graph_builder.py` defines the generic adapter contract.

The existing project also needs to decide exactly where Stage-3/current segmentation outputs, marker heatmaps, and cached Trackastra results are stored. Those repository-path decisions are intentionally not hardcoded into the model package.

## Dependencies

See `requirements.txt`. No `torch_geometric` dependency is required.
