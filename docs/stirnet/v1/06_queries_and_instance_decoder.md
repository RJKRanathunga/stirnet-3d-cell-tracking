# 06 — Queries and Instance Decoder

## 1. Output formulation

STIR-Net predicts an unordered set of biological cell hypotheses.

Each query predicts:

```text
existence probability
center
3D native-resolution mask
```

The model is not required to preserve input instance IDs.

Final tracking is rebuilt later.

## 2. Query sources

V1 uses four query types.

### 2.1 Primary current-instance queries

One per current input instance.

Meaning:

> The existing segmentation claims there is one cell here.

### 2.2 Split-companion queries

One additional query per current input instance.

Meaning:

> Could this current instance contain an additional biological cell?

This allows geometry alone to correct merges even when Trackastra provides no useful temporal hypothesis.

### 2.3 Temporal repair queries

One per temporal hypothesis.

Meaning:

> Temporal evidence suggests that a biological cell may exist near this location.

### 2.4 Discovery queries

Eight learned queries.

Meaning:

> Search for a cell missed by both the current segmentation and temporal priors.

Default:

```python
N_DISCOVERY_QUERIES = 8
MAX_QUERIES = 128
```

Total:

\[
Q=
2N_\text{instances}
+
N_\text{temporal}
+
8.
\]

Do not silently truncate.

## 3. Current-instance feature pooling

Use an E2-scale spatial feature map projected to 128 dimensions.

Downsample each current instance mask to that feature grid.

Compute:

\[
f_i^{mean}
=
\operatorname{MaskedMean}(F,M_i)
\]

and:

\[
f_i^{max}
=
\operatorname{MaskedMax}(F,M_i).
\]

Concatenate:

```text
128 + 128 = 256
```

project:

```text
256 -> 128
```

## 4. Current-instance geometry vector

Recommended V1 schema:

```text
log physical volume
bbox z/dref
bbox y/dref
bbox x/dref
PCA axis 1/dref
PCA axis 2/dref
PCA axis 3/dref
elongation
flatness
solidity
compactness
mean intensity
intensity std
marker count
```

This is a 14-D vector.

Projection:

```text
14 -> 64 -> 128
```

## 5. Instance query

\[
q_i^{primary}
=
LN(
q_i^{spatial}
+
q_i^{geom}
+
e_{primary}
).
\]

Reference position:

\[
r_i^{primary}
=
\text{current instance centroid in physical/cellscale coordinates}.
\]

## 6. Split companion

\[
q_i^{split}
=
LN(
q_i^{primary}
+
e_{split}
).
\]

It shares the same initial reference point and same current-mask support.

For correct single cells, it should predict `no-object`.

For a merge, it can become the additional cell.

## 7. Temporal query

\[
q_i^{temp}
=
LN(
T_i^{final}
+
e_{temporal}
).
\]

Reference position:

\[
r_i^{temp}
=
\text{Trackastra-derived target-frame estimate}.
\]

## 8. Discovery queries

Learn:

```python
discovery_queries: [8,128]
discovery_refs_unconstrained: [8,3]
```

Reference coordinates can be mapped to a normalized patch frame with sigmoid.

Discovery queries receive global coarse attention in decoder layer 1.

## 9. Query type embeddings

Learn embeddings:

```text
primary
split
temporal
discovery
```

Each is 128-D.

Query type is part of the query representation but is not a predicted class.

## 10. Initial mask priors

### 10.1 Primary instance

For current instance mask \(M_i\):

\[
L_i^{prior}(v)=
\begin{cases}
+1.5,& v\in M_i\\
-1.5,&v\notin M_i
\end{cases}
\]

### 10.2 Split companion

Use the same prior initially.

### 10.3 Temporal query

Use a weak physical Gaussian prior around the temporal reference:

\[
\sigma\approx0.75d_\text{ref}.
\]

If reference falls inside a current component, union that component with the temporal support for attention initialization.

### 10.4 Discovery query

No mask prior.

## 11. Query decoder

Defaults:

```python
QUERY_LAYERS = 3
QUERY_HEADS = 4
QUERY_FFN_DIM = 512
D_MODEL = 128
DROPOUT = 0.10
```

Each layer:

```text
pre-norm
 |
query self-attention
 |
residual
 |
pre-norm
 |
query -> spatial masked cross-attention
 |
residual
 |
pre-norm
 |
FFN 128 -> 512 -> 128
 |
residual
 |
existence head
center head
mask embedding head
```

## 12. Query self-attention

All queries in the same patch communicate.

This enables:

- primary and split queries to compete;
- temporal and current-instance queries to merge duplicate hypotheses;
- discovery queries to avoid duplicating seeded cells.

## 13. Decoder spatial scales

Use progressively finer feature scales.

Conceptual sequence:

```text
layer 1: deepest/coarse co-reasoned feature
layer 2: E2-scale co-reasoned decoder feature
layer 3: medium-resolution decoder feature
```

All are projected to 128 channels before attention.

Do not perform query attention over every full-resolution native voxel.

## 14. Decoder layer-1 attention support

### Primary / split

Allowed support:

```text
current instance
UNION
physical dilation by ~1 dref
```

### Temporal

Radius:

\[
R_i=(1.5+a_i)d_\text{ref}.
\]

Optionally union with nearest/current component around the reference.

### Discovery

Global coarse feature map.

## 15. Progressive masked attention

After layer 1, predict coarse mask \(P_i^{(1)}\).

Layer-2 support:

```text
P_i^(1) > 0.20
```

plus a physical dilation.

Layer 3 repeats with the layer-2 mask.

If a mask collapses to empty, fall back to a reference-radius support rather than disabling the query permanently.

## 16. Center update

Centers are predicted in cell-scale normalized physical coordinates.

At each layer:

\[
\Delta_i^{(l)}
=
MLP_{center}(q_i^{(l)}).
\]

Update:

\[
\tilde r_i^{(l)}
=
\tilde r_i^{(l-1)}
+
\Delta_i^{(l)}.
\]

A bounded update function may be used if instability is observed.

Only when accessing dense feature grids do we convert physical coordinates to native voxel coordinates.

## 17. Existence head

```text
128 -> 128 -> 1
```

Output:

```python
exist_logits: [B,Q]
```

There is only:

```text
cell
no-object
```

## 18. Mask embedding

Final query:

```text
128 -> 128 -> 32
```

produces:

\[
m_i\in\mathbb R^{32}.
\]

Native mask features:

\[
F_{mask}
\in
\mathbb R^{B\times32\times Z\times Y\times X}.
\]

Final mask logit:

\[
L_i(v)
=
m_i^TF_{mask}(v)+L_i^{prior}(v).
\]

## 19. Memory-aware rendering

Do not render full-resolution masks for all queries.

### Training

```text
all queries
 |
coarse masks
 |
Hungarian matching
 |
matched positive queries only
 |
native-resolution mask rendering
```

### Inference

```text
all queries
 |
coarse existence + mask
 |
candidate filtering
 |
surviving queries only
 |
native-resolution mask rendering
```

## 20. Decoder outputs

```python
{
    "exist_logits": [B,Q],
    "centers_cellscale": [B,Q,3],
    "coarse_mask_logits": [...],
    "query_embeddings": [B,Q,128],
    "query_types": [B,Q],
    "aux_outputs": [
        layer1_dict,
        layer2_dict,
    ],
}
```

Native high-resolution masks are generated lazily through a dedicated renderer.
