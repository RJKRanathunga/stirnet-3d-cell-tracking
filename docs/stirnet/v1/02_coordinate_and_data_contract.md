# 02 — Coordinate and Data Contract

## 1. Native-resolution rule

STIR-Net V1 preserves the original dense image lattice.

For every sample, retain:

```text
image[t, z, y, x]
instance_labels[t, z, y, x]
spacing_um = (sz, sy, sx)
```

Do not globally resample all datasets to one isotropic spacing before the model.

The model must therefore never assume that one voxel has the same physical size across axes or datasets.

## 2. Coordinate conventions

Use array order:

```text
(z, y, x)
```

Every geometry-carrying API must make the coordinate system explicit.

### 2.1 Native voxel coordinates

$$p_\text{vox} = (z,y,x)$$

### 2.2 Physical coordinates

$$p_{\mu m} = (zs_z, ys_y, xs_x)$$

### 2.3 Cell-scale-normalized coordinates

For sequence-level robust reference diameter $d_\text{ref}$,

$$\tilde p = \frac{ p_{\mu m} - p_{\mu m,\text{patch center}} }{ d_\text{ref} }$$

Use cell-scale-normalized coordinates for cross-dataset relational features and physical coordinates when exact physical interpretation is needed.

## 3. Robust reference cell diameter

Estimate $d_\text{ref}$ from reliable current/ground-truth instances in the sequence.

Preferred definition:

$$d_\text{ref} = \operatorname{median} \left[ 2\left(\frac{3V_i}{4\pi}\right)^{1/3} \right]$$

where $V_i$ is physical volume in $\mu m^3$.

Use robust filtering to exclude:

- extremely small fragments;
- giant merged candidates;
- obvious boundary truncations;
- known artifacts.

Cache this value per sequence.

If no reliable estimate exists, use a dataset-level training statistic.

## 4. Training patch definition

Patch size is defined in physical/cell-scale space:

$$L_\text{context}=8d_\text{ref}$$

per axis.

Native voxel size:

$$N_z=\left\lceil\frac{L_\text{context}}{s_z}\right\rceil$$

and equivalently for Y/X.

The patch is centered on a selected training location and extracted on the native lattice.

### Valid output core

Only GT cells whose centers lie in the central:

$$L_\text{valid}=6d_\text{ref}$$

cube are treated as required outputs.

The outer context margin is available to the network but does not define owned output cells.

## 5. Dense input channels

`spatial_inputs`:

```python
shape = [B, 5, Z, Y, X]
dtype = float32
```

Channel schema:

```text
0 raw_intensity
1 foreground
2 physical_edt
3 instance_boundary
4 marker_heatmap
```

### 5.1 Raw intensity

Robust normalize each volume/frame using a consistent training-time policy.

The model input should be approximately bounded to `[0, 1]`, while augmentation can move values temporarily before clipping.

### 5.2 Foreground

```python
foreground = (instance_labels > 0).float()
```

### 5.3 Physical EDT

Compute inside each individual current instance.

Distance must be in physical units:

```python
scipy.ndimage.distance_transform_edt(
    binary_instance,
    sampling=(sz, sy, sx),
)
```

For a combined tensor, fill each instance's EDT values into its own voxels.

Background is zero.

Recommended further normalization:

$$EDT_\text{norm} = EDT_{\mu m}/d_\text{ref}$$

with clipping to a reasonable maximum.

### 5.4 Instance boundary

Compute from current instance labels after the labels are loaded on the native grid.

Prefer a one-voxel categorical boundary map for input. Physical-width boundary targets are used for auxiliary supervision.

### 5.5 Marker heatmap

Construct from current segmentation markers or marker positions.

Gaussian width must be defined physically.

For axis $a$,

$$\sigma_a^\text{vox} = \frac{\sigma_{\mu m}}{s_a}$$

so the Gaussian represents an approximately isotropic physical object despite an anisotropic voxel lattice.

## 6. Instance label tensor

```python
instance_labels:
[B, Z, Y, X]
dtype = int64
```

This tensor is **never concatenated into the CNN as a scalar channel**.

It is used for:

- instance pooling;
- query construction;
- current-mask priors;
- mapping temporal hypotheses to current components;
- postprocessing/debugging.

Instance ID numbers carry no ordinal meaning.

## 7. Acquisition metadata

Input vector:

```python
acquisition_features:
[B, 7]
```

Schema:

```text
0 log(sz)
1 log(sy)
2 log(sx)
3 log(sz / dref)
4 log(sy / dref)
5 log(sx / dref)
6 log(dref)
```

The vector is projected:

```text
7 -> 32 -> 64
```

and used by physical-aware CNN blocks.

## 8. Physical feature coordinates

Every CNN feature level must expose effective spacing:

$$s^{(l)}=(s_z^{(l)},s_y^{(l)},s_x^{(l)})$$

and coordinate generation utilities.

For feature index:

$$(j_z,j_y,j_x)$$

the physical coordinate relative to the patch center is:

$$p^{(l)}_{\mu m} = ( j_zs_z^{(l)}, j_ys_y^{(l)}, j_xs_x^{(l)} ) - p_{\mu m,\text{patch center}}.$$

Cross-attention must not use raw feature indices as spatial distances.

## 9. Batch strategy

Native spacings generate different patch shapes.

Use a bucketed sampler based on:

- dataset;
- native spacing regime;
- resulting patch shape.

A batch should contain shape-compatible samples.

Padding is allowed, but padded voxels must have an explicit `spatial_padding_mask`.

## 10. Required sample object

A training sample should provide at least:

```python
{
    "spatial_inputs": FloatTensor[5, Z, Y, X],
    "instance_labels": LongTensor[Z, Y, X],
    "spacing_um": FloatTensor[3],
    "dref_um": FloatTensor[],
    "acquisition_features": FloatTensor[7],

    "instance_features": FloatTensor[Ni, F_inst],

    "graph_x": FloatTensor[N, 32],
    "graph_edge_index": LongTensor[2, E],
    "graph_edge_attr": FloatTensor[E, 15],
    "accepted_association_edge_index": LongTensor[2, Ea],
    "accepted_association_edge_attr": FloatTensor[Ea, 3],

    "tracklet_id": LongTensor[N],
    "node_ids": LongTensor[N],
    "node_observed_ref_um": FloatTensor[N, 3],
    "node_time_offset": FloatTensor[N],

    "temporal_ref_um": FloatTensor[M, 3],
    "temporal_status": FloatTensor[M, 10],

    "hyp_edge_index": LongTensor[2, Eh],
    "hyp_edge_attr": FloatTensor[Eh, 22],

    "node_instance_grid": FloatTensor[N, 4, 12, 12, 12],
    "node_history_valid": BoolTensor[N],

    "history_support": FloatTensor[M, 2, 2, 12, 12, 12],
    "history_support_valid": BoolTensor[M, 2],
    "history_support_dt": FloatTensor[M, 2],
    "history_support_center_um": FloatTensor[M, 2, 3],
    "history_support_extent_um": FloatTensor[M, 2],

    "best_current_component_id": LongTensor[M],
    "best_component_overlap": FloatTensor[M],
    "second_best_component_overlap": FloatTensor[M],

    "gt_label_map": IntTensor[Z, Y, X],
    "gt_instance_ids": LongTensor[K],
    "gt_centers_um": FloatTensor[K, 3],
    "gt_valid": BoolTensor[K],
}
```

Variable-sized tensors are collated through index/batch vectors rather than forced into dense per-sample padding unless required by attention.

The preferred training representation keeps one integer GT label map. Coarse
per-instance masks are derived after downsampling, and native masks are rendered
only for matched queries in bounded chunks.

## 11. Historical descriptor coordinates

Historical grids are canonical physical cubes, not native-voxel crops. The full
grid extent is `history_extent_dref * dref_um` (default `2.5*dref_um`) on every
zyx axis, centered on the provisional detection and aligned with the global
acquisition z,y,x axes. Grid endpoints are at `-extent/2` and `+extent/2`;
sampling is physically computed from `spacing_um`.

The four node channels are occupancy in `[0,1]`, signed physical distance
divided by `dref_um` and clipped to `[-1,+1]` (positive inside), masked
robust-normalized intensity, and local robust-normalized intensity including
nearby context. Invalid rows may be absent or zero-filled, but
`node_history_valid=False` is authoritative and produces exactly zero history
contribution. Cached descriptors may be float16; coordinate, distance, overlap,
and attention-softmax arithmetic is float32.

Support side 0 is past and side 1 is future. `history_support_center_um` uses
the same target-patch-relative physical zyx convention as `temporal_ref_um`;
`history_support_dt` is target-relative frame offset. The support cube preserves
only occupancy and SDF. A translational projection maps the historical center
onto the immutable target-frame temporal reference. For target point `x`, the
historical sample point is `x - (temporal_ref_um - historical_center_um)`.
