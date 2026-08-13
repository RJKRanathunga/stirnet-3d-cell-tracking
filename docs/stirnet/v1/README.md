# STIR-Net V1 Documentation

**STIR-Net** stands for **Spatiotemporal Instance Refinement Network**.

Version 1 is a learned refinement model whose only task is to correct the cell instances produced by the existing segmentation pipeline. It does **not** replace Trackastra and it does **not** predict final tracking. Trackastra is run once before STIR-Net to provide temporal clues and again after STIR-Net to track the corrected instances.

The model-specific implementation is expected to live under:

```text
learned/stirnet/
```

The architecture documented here is intentionally frozen at the information-flow level. Hyperparameters such as loss weights, query counts, attention radii, and channel widths may be tuned experimentally without changing the fundamental architecture.

## Core objective

For target frame $t$,

$$(I_t, S_t, \mathcal{T}_t, \mathbf{s}) \rightarrow S_t^*$$

where:

- $I_t$ is the native-resolution fluorescence volume.
- $S_t$ is the imperfect current instance segmentation.
- $\mathcal{T}_t$ is the provisional temporal evidence produced from Trackastra pass 1.
- $\mathbf{s} = (s_z, s_y, s_x)$ is the native voxel spacing.
- $S_t^*$ is the corrected set of cell instances.

Trackastra pass 2 is run on $S_t^*$ after refinement.

## Design principles

1. **Preserve native image samples.** V1 does not globally resample all datasets to a canonical isotropic voxel grid.
2. **Express geometry explicitly in physical coordinates.** Distances, positions, motion, EDT values, attention radii, and center losses use physical units or cell-scale-normalized physical units.
3. **Use convolution for dense local geometry.** The spatial branch remains a CNN because boundaries, local shape, intensity structure, and surface continuity are dense local phenomena.
4. **Use graph attention for sparse temporal evidence.** Every bounded-window detection is retained. Candidate relations permit learned reasoning over all within-sample pairs, while accepted Trackastra associations remain identifiable evidence rather than topology truth.
5. **Fuse before making instance decisions.** Fine detection-node and coarse tracklet memories are read before query creation and in every query-decoder layer; dense co-reasoning remains tracklet-level.
6. **Predict corrected cells, not error labels.** Under-segmentation, over-segmentation, missing cells, and false positives emerge from the difference between the input and output instance sets.
7. **Treat Trackastra as evidence, not authority.** Temporal hypotheses have learned salience and reliability and are intentionally corrupted during training.
8. **Keep V1 implementable.** No full-volume transformer, no deformable 3D attention, no tracking loss, and no explicit merge classifier are included in V1.

## Documentation map

| Document | Purpose |
|---|---|
| `01_system_overview.md` | End-to-end role of STIR-Net in the full pipeline |
| `02_coordinate_and_data_contract.md` | Native grids, physical coordinates, patch definition, dense inputs |
| `03_spatial_backbone.md` | Physical-aware ResUNet encoder/decoder |
| `04_temporal_graph.md` | Trackastra graph, node/edge schema, temporal hypotheses |
| `05_coreasoning.md` | Bidirectional cross-attention and graph/spatial updates |
| `06_queries_and_instance_decoder.md` | Query construction, masked decoder, mask rendering |
| `07_losses_and_matching.md` | Hungarian matching and all V1 loss terms |
| `08_training_data_and_curriculum.md` | Synthetic corruption, clean samples, augmentation, curriculum |
| `09_inference_and_postprocessing.md` | Tiling, filtering, native-mask rendering, final instance labels |
| `10_configuration.md` | Frozen V1 defaults and tunable hyperparameters |
| `11_code_architecture.md` | Proposed `learned/stirnet/` module layout and interfaces |
| `12_validation_and_ablations.md` | Metrics, ablations, acceptance criteria, V2 triggers |
| `13_hierarchical_temporal_memory.md` | Candidate graph, node/tracklet memory, query attention, cache/checkpoint/debug contracts |

## High-level flow

```text
Raw sequence
    |
    v
Existing segmentation pipeline
    |
    v
Initial instances
    |
    v
Trackastra pass 1
    |
    +---------------------------+
    |                           |
    v                           v
native image + masks       temporal graph clues
    |                           |
    v                           v
physical-aware CNN          GATv2 encoder
    |                           |
    +---------- co-reasoning ---+
                    |
                    v
             cell queries
                    |
                    v
          query mask decoder
                    |
                    v
          corrected instances
                    |
                    v
            Trackastra pass 2
                    |
                    v
              final tracking
```

## V1 output representation

The model predicts an unordered set:

$$\hat{\mathcal C}_t = \{(\hat p_i, \hat{\mathbf c}_i, \hat M_i)\}_{i=1}^{Q}$$

where each query predicts:

- $\hat p_i$: cell-existence probability.
- $\hat{\mathbf c}_i$: cell center in physical/cell-normalized coordinates.
- $\hat M_i$: native-resolution 3D mask.

There is no track identity in the output.

## Status

This documentation is the implementation specification for V1. The architecture should not be redesigned during initial coding unless a contradiction is found. Changes should first be recorded in the documentation and then propagated to code.
