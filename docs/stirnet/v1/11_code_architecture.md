# 11 — Code Architecture

## 1. Source root

Model-specific code lives under:

```text
learned/stirnet/
```

Recommended layout:

```text
learned/stirnet/
├── __init__.py
│
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   │
│   ├── blocks.py
│   ├── spacing.py
│   ├── coordinates.py
│   │
│   ├── spatial_encoder.py
│   ├── spatial_decoder.py
│   │
│   ├── graph_encoder.py
│   ├── temporal_hypotheses.py
│   │
│   ├── attention.py
│   ├── coreasoning.py
│   │
│   ├── query_builder.py
│   ├── query_decoder.py
│   ├── heads.py
│   │
│   ├── matcher.py
│   ├── losses.py
│   │
│   └── stir_net.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── patch_sampler.py
│   ├── graph_builder.py
│   ├── trackastra_cache.py
│   ├── corruptions.py
│   ├── augmentation.py
│   ├── targets.py
│   └── collate.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── checkpoint.py
│
├── inference/
│   ├── __init__.py
│   ├── tiling.py
│   ├── postprocess.py
│   └── refine.py
│
└── debugging/
    ├── inspect_batch.py
    ├── inspect_queries.py
    ├── inspect_attention.py
    └── visualize_refinement.py
```

## 2. `model/config.py`

Contains dataclasses only.

No learned model logic.

Must define all V1 hyperparameters in one place.

## 3. `model/types.py`

Define structured containers.

Recommended examples:

```python
@dataclass
class SpatialPyramid:
    features: list[Tensor]
    spacings_um: list[Tensor]
    padding_masks: list[Tensor]

@dataclass
class TemporalState:
    tokens: Tensor
    ref_um: Tensor
    ref_cellscale: Tensor
    salience: Tensor
    reliability: Tensor
    status: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch_index: Tensor

@dataclass
class QueryState:
    embeddings: Tensor
    references_cellscale: Tensor
    query_types: Tensor
    padding_mask: Tensor
    priors: object

@dataclass
class StirNetOutput:
    exist_logits: Tensor
    centers_cellscale: Tensor
    coarse_mask_logits: Tensor
    query_embeddings: Tensor
    aux_outputs: list
    dense_outputs: dict
    mask_features: Tensor
```

## 4. `model/spacing.py`

Responsibilities:

- acquisition feature construction;
- physical-aware stride schedule;
- feature-level spacing propagation;
- conversion between physical and native-grid coordinates;
- physical dilation radii to grid radii.

Must be independently unit tested.

## 5. `model/coordinates.py`

Responsibilities:

```text
voxel <-> physical
physical <-> patch-relative
physical <-> dref-normalized
feature-index <-> physical
```

Coordinate conversions should not be duplicated throughout the codebase.

## 6. `model/blocks.py`

Implement:

```text
PhysicalAwareResBlock
AxisFactorizedConv
SpacingGate
DownsampleBlock
UpsampleBlock
```

No graph or query logic.

## 7. `model/spatial_encoder.py`

Interface:

```python
pyramid = encoder(
    spatial_inputs,
    spacing_um,
    acquisition_embedding,
    spatial_padding_mask=None,
)
```

Returns `SpatialPyramid`.

## 8. `model/spatial_decoder.py`

Accepts:

- encoder pyramid;
- updated deepest features;
- co-reasoning hook/state.

Returns:

- final mask features;
- medium decoder features needed by query decoder;
- auxiliary dense feature.

## 9. `model/graph_encoder.py`

Implement:

```text
DetectionGraphEncoder
GATv2 residual block
HypothesisGraphBlock
```

It must not import Trackastra.

Inputs are plain tensors.

## 10. `model/temporal_hypotheses.py`

Implement:

```text
TrackletPooler
SalienceHead
ReliabilityHead
TemporalStateBuilder
```

Tracklet structure is supplied by preprocessing.

## 11. `model/attention.py`

Implement:

```text
PhysicalPositionBias
LocalPhysicalCrossAttention
GatedCrossAttention
```

Responsibilities:

- physical distance computation;
- radius masks;
- empty-neighbour handling;
- attention logging hooks.

Do not mix CNN block logic here.

## 12. `model/coreasoning.py`

`CoReasoningBlock`:

```python
spatial_out, temporal_out = block(
    spatial_feature,
    spatial_spacing_um,
    spatial_padding_mask,
    temporal_state,
    dref_um,
    acquisition_embedding,
)
```

Internal order must match the architecture documentation.

## 13. `model/query_builder.py`

Implement:

```text
InstanceQueryBuilder
SplitCompanionBuilder
TemporalQueryBuilder
DiscoveryQueryBank
QueryAssembler
```

It owns query-type embeddings and initial references.

## 14. `model/query_decoder.py`

Implement:

```text
QueryDecoderLayer
InstanceQueryDecoder
MaskAttentionSupportBuilder
```

The decoder must operate on progressively finer spatial features.

## 15. `model/heads.py`

Implement:

```text
ExistenceHead
CenterHead
MaskEmbeddingHead
DenseForegroundHead
DenseCenterHead
DenseBoundaryHead
```

## 16. `model/matcher.py`

Implement `HungarianMatcher3D`.

Inputs:

```text
existence logits
coarse masks
centers
GT masks
GT centers
valid flags
```

Output:

```text
matched_query_indices
matched_gt_indices
```

Keep matching code outside the model forward path where possible.

## 17. `model/losses.py`

Implement one criterion object:

```python
criterion = RefinementCriterion(config.losses)
loss_dict = criterion(outputs, targets)
```

Return a dict of named losses before weighted reduction.

## 18. `model/stir_net.py`

Top-level network.

Suggested forward contract:

```python
outputs = model(
    spatial_inputs,
    instance_labels,
    spacing_um,
    dref_um,
    acquisition_features,

    instance_features,
    instance_batch,

    graph_x,
    graph_edge_index,
    graph_edge_attr,
    graph_batch,
    tracklet_id,

    temporal_ref_um,
    temporal_status,

    hypothesis_edge_index,
    hypothesis_edge_attr,
    temporal_batch,

    spatial_padding_mask=None,
)
```

`stir_net.py` orchestrates modules but should contain little low-level math.

## 19. High-resolution mask renderer

Expose separately:

```python
mask_logits = model.render_masks(
    mask_features,
    query_embeddings,
    query_indices,
    query_priors,
)
```

This supports memory-efficient training and inference.

## 20. Data layer responsibilities

`data/` owns all Trackastra-specific conversion.

The neural model must not receive Trackastra Python classes.

`graph_builder.py` converts Trackastra output to plain tensors.

## 21. Test strategy

At minimum create tests for:

```text
coordinate round trips
spacing schedule
physical EDT scale
query count and type construction
empty temporal-state behavior
cross-attention empty-neighbour behavior
mask-prior shape
Hungarian matching
loss finite gradients
variable native spacing
full forward shape
```

## 22. Debugging hooks

Model forward should optionally return:

```text
temporal salience
temporal reliability
cross-attention maps
query attention support
coarse masks per layer
query types
query references
```

behind a debug flag.

Do not make debugging require modifying model source.
