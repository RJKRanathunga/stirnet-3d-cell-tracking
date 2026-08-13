# 04 — Temporal Graph and Trackastra Priors

## 1. Role

The temporal branch converts Trackastra pass-1 output into a compact set of **segmentation priors**.

It does not attempt to correct tracking.

It does not predict final identities.

It answers questions such as:

- Is a cell expected near this target-frame location?
- Is an internal track start/end suspicious?
- Is a short gap likely to indicate a missed instance?
- Do two trajectories appear to pass through one current component?
- Is a temporal clue strong but unreliable?

## 2. Temporal window

V1 uses:

```python
TEMPORAL_RADIUS = 2
```

Target frame $t$ sees detections from:

$$[t-2,t-1,t,t+1,t+2].$$

## 3. Detection graph nodes

One node = one provisional detected object at one timepoint.

```python
graph_x: [N_nodes, 32]
```

### Node schema

| Dim | Feature |
|---|---|
| 0 | normalized time offset $\Delta t/2$ |
| 1–3 | relative XYZ position / $d_\text{ref}$ |
| 4 | log physical volume / reference physical volume |
| 5–7 | physical bbox dims / $d_\text{ref}$ |
| 8–10 | PCA axis lengths / $d_\text{ref}$ |
| 11 | log1p elongation (PCA axis 1 / axis 2) |
| 12 | log1p flatness (PCA axis 2 / axis 3) |
| 13 | solidity |
| 14 | compactness |
| 15 | normalized mean intensity |
| 16 | normalized intensity std |
| 17–19 | backward velocity / $d_\text{ref}$ |
| 20–22 | forward velocity / $d_\text{ref}$ |
| 23 | track length before |
| 24 | track length after |
| 25 | distance to physical volume boundary / $d_\text{ref}$ |
| 26 | distance to current patch boundary / $d_\text{ref}$ |
| 27 | target-frame indicator |
| 28 | interior-start indicator |
| 29 | interior-end indicator |
| 30 | division-related indicator |
| 31 | physical-boundary-related indicator |

All undefined quantities must have a corresponding validity convention or zero value produced by preprocessing.

For the two PCA ratios, floor the denominator at the finest physical voxel
extent before applying `log1p`. This represents unresolved thin axes at the
acquisition resolution and prevents degenerate components from producing
million-scale neural inputs. The same convention is used by current-instance
geometry features.

## 4. Detection graph edges

Use four relation types:

1. Trackastra forward temporal association.
2. Reverse temporal association.
3. Division relation.
4. Same-frame spatial-neighbour relation.

`edge_index`:

```python
[2, E]
```

`edge_attr`:

```python
[E, 14]
```

### Edge schema

```text
0      signed Δt
1-3    Δz, Δy, Δx normalized by dref
4      physical distance / dref
5      log physical volume ratio
6      normalized intensity difference
7      motion residual / dref
8      Trackastra association score
9      association score available
10-13  one-hot relation type
```

If association confidence is unavailable:

```text
edge_attr[:,8] = 0
edge_attr[:,9] = 0
```

Do not insert fake calibrated confidence.

## 5. Same-frame spatial edges

For every node, connect up to:

```python
K_SPATIAL_NEIGHBORS = 6
```

nearest same-frame detections within approximately:

$$2.5d_\text{ref}.$$

This lets nearby cell hypotheses interact even if Trackastra has no temporal edge between them.

## 6. Detection graph encoder

Input projection:

```text
32 -> 64 -> 128
```

Then two edge-aware GATv2 blocks.

Defaults:

```python
GRAPH_LAYERS = 2
GRAPH_HEADS = 4
GRAPH_FFN_DIM = 256
D_MODEL = 128
DROPOUT = 0.10
```

Each block:

```text
GATv2
 |
residual
 |
LayerNorm
 |
FFN 128 -> 256 -> 128
 |
residual
 |
LayerNorm
```

Output:

```python
node_embeddings: [N_nodes, 128]
```

## 7. Tracklet grouping

Convert provisional associations into maximal non-branching tracklets within the five-frame window.

A tracklet can contain:

```text
t-2 -> t-1 -> t
```

or:

```text
t-1 -> t+1
```

if the current frame is missing from the provisional track.

Division points split tracklets so that each temporal hypothesis is approximately one cell-continuation hypothesis.

## 8. Tracklet attention pooling

For each tracklet $m$:

$$a_n = \operatorname{softmax}_n \left[ w^T\tanh(W_hh_n + W_te_{\Delta t_n}) \right]$$

$$T_m=\sum_{n\in m}a_nh_n.$$

Output:

```python
temporal_tokens: [M, 128]
```

where M is the number of target-frame temporal hypotheses.

## 9. Target-frame reference position

Each hypothesis has:

```python
temporal_ref_um: [M,3]
```

Construction priority:

1. target-frame detection exists → its centroid;
2. valid past and future detections → interpolate;
3. only past exists → constant-velocity extrapolation;
4. only future exists → backward extrapolation.

Also store:

```python
temporal_ref_cellscale = temporal_ref_relative_um / dref_um
```

## 10. Temporal status vector

```python
temporal_status: [M,10]
```

Suggested fields:

```text
0 complete/stable
1 interior_start
2 interior_end
3 gap_candidate
4 division_related
5 physical_boundary_related
6 window_start_truncated
7 window_end_truncated
8 motion_inconsistent
9 association_uncertain
```

These are clues, not labels of true biological events.

## 11. Salience

Learn:

$$a_i = \sigma(MLP_a[T_i,status_i])$$

where $a_i\in[0,1]$.

Interpretation:

> How diagnostic is this temporal pattern of a segmentation problem?

Interior ends, starts, and gaps should receive a mild positive initialization prior.

Stable complete tracks should start with lower salience.

The value remains learnable.

## 12. Reliability

Learn separately:

$$r_i = \sigma(MLP_r[T_i,status_i])$$

Interpretation:

> How much should the model trust the predicted position/continuation represented by this clue?

Salience and reliability are deliberately separate.

Examples:

```text
stable complete track:
    low salience
    high reliability

clean gap surrounded by strong continuation:
    high salience
    high reliability

noisy single-frame disappearance:
    high salience
    low reliability
```

## 13. Hypothesis graph

After tracklet pooling, construct a second graph over temporal hypotheses.

Connect hypotheses if any condition holds:

- reference distance < $2.5d_\text{ref}$;
- provisional lineage relationship;
- potential gap relationship;
- same current target-frame component.

`hyp_edge_attr`:

```python
[Eh, 22]
```

Dimensions 0-7 preserve these legacy semantics as the prefix of the required
22-D schema in Section 16. Dimensions 8-21 add convergence, projected support,
component overlap, volume compatibility, and explicit validity.

This graph is used inside co-reasoning blocks after temporal hypotheses inspect image evidence.

## 14. Required outputs from temporal encoder

```python
TemporalState(
    tokens=[B or packed M, 128],
    ref_um=[...,3],
    ref_cellscale=[...,3],
    salience=[...,1],
    reliability=[...,1],
    status=[...,10],
    hyp_edge_index=[2,Eh],
    hyp_edge_attr=[Eh,22],
    batch_index=[M],
)
```

The spatial/query network must not depend on Trackastra-specific Python objects.

## 15. Historical instance fusion

The scalar node projection remains `32 -> 64 -> 128`. A compact history CNN
maps valid `4 x 12 x 12 x 12` grids to a second 128-D vector. A learned scalar
gate consumes the scalar embedding, history embedding, and validity, then
applies `scalar + valid * gate * history`. Its final weights are zero and bias is
negative at initialization. Invalid history is masked before the gate and is
exactly behavior-neutral. Fusion occurs before the existing two detection GAT
blocks and learned tracklet pooling.

For every tracklet, preprocessing selects at most the nearest valid past and
future observation. Hypothesis support stores only occupancy and signed distance:

```text
history_support             [M,2,2,12,12,12]
history_support_valid       [M,2]
history_support_dt          [M,2]
history_support_center_um   [M,2,3]
history_support_extent_um   [M,2]
```

## 16. Required 22-D hypothesis edge schema

```text
0-2   target-reference delta z,y,x / dref
3     target-reference distance / dref
4     same-current-component confidence
5     lineage-related
6     gap-related
7     combined provisional reliability
8     nearest common-past pair distance / dref
9     older common-past pair distance / dref
10    nearest minus older distance (negative means closing)
11    positive-when-closing speed / dref/frame
12-14 relative velocity z,y,x / dref/frame (j minus i)
15    projected A/B support overlap
16    A best-current-component overlap
17    B best-current-component overlap
18    combined previous physical volume / current-component volume
19    nearest-pair-distance valid
20    older-pair-distance valid
21    support/component-overlap valid
```

Projected positive occupancy samples, not reference centroids, provide the
primary current-component assignment and best/second overlap. Reference lookup
is only the no-valid-support fallback. Reverse edges negate 0-2 and 12-14, swap
16/17, and preserve symmetric dimensions. Missing quantities retain explicit
validity rather than fabricated calibration.
