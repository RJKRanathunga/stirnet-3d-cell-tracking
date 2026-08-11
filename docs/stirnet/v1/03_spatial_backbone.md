# 03 — Spatial Backbone

## 1. Role

The spatial branch must learn:

- raw intensity morphology;
- local 3D appearance;
- cell interior structure;
- boundaries and valleys;
- shape consistency;
- marker/EDT topology;
- high-resolution native-grid mask features.

It must preserve original dense measurements while remaining robust to different native voxel spacings.

## 2. Architecture family

Use a **Physical-Aware ResUNet**.

V1 has four levels:

```python
SPATIAL_CHANNELS = (16, 32, 64, 128)
SPATIAL_BLOCKS_PER_LEVEL = 2
```

Conceptually:

```text
E0: C=16
 |
down
 v
E1: C=32
 |
down
 v
E2: C=64
 |
down
 v
E3: C=128

decoder mirrors the exact stride schedule
```

## 3. PhysicalAwareResBlock

### Inputs

```python
x: [B, C, Z, Y, X]
acquisition_embedding: [B, 64]
```

### Internal axis branches

Use three factorized convolutions:

```python
conv_z = Conv3d(C, C, kernel_size=(3,1,1), padding=(1,0,0))
conv_y = Conv3d(C, C, kernel_size=(1,3,1), padding=(0,1,0))
conv_x = Conv3d(C, C, kernel_size=(1,1,3), padding=(0,0,1))
```

All branches operate on the same normalized/activated input.

### Spacing-conditioned gates

Project acquisition embedding:

```text
64 -> 3C
```

reshape to:

```python
[B, 3, C, 1, 1, 1]
```

and apply sigmoid.

Let gates be:

$$g_z,\;g_y,\;g_x$$

Then:

$$F = g_z\odot F_z + g_y\odot F_y + g_x\odot F_x.$$

This is followed by:

```python
Conv3d(C, C, kernel_size=1)
```

### Full residual block

Recommended structure:

```text
input
 |
GroupNorm
 |
SiLU
 |
axis branches + spacing gates
 |
1x1x1 fusion
 |
GroupNorm
 |
SiLU
 |
axis branches + spacing gates
 |
1x1x1 fusion
 |
+ projected residual if channels differ
```

Use GroupNorm with group count chosen to divide the channel count, e.g.:

```text
C=16 -> 4 groups
C=32 -> 8 groups
C=64 -> 8 groups
C=128 -> 8 groups
```

## 4. Why factorized axis processing?

A standard `3x3x3` kernel has different physical extents when voxel spacing is anisotropic.

Factorized axis branches let V1:

- separately encode through-plane and in-plane evidence;
- learn spacing-conditioned branch weights;
- preserve native resolution;
- avoid global resampling.

V1 does **not** claim this makes convolution perfectly physical-space invariant. It is a practical first step.

## 5. Physical-aware downsampling policy

Maintain current effective spacing:

```python
spacing = (sz, sy, sx)
```

Before each encoder downsample:

```python
max_s = max(spacing)
stride[a] = 2 if spacing[a] < max_s / ANISOTROPY_THRESHOLD else 1
```

with:

```python
ANISOTROPY_THRESHOLD = 1.5
```

If all axes would remain stride 1, use `(2,2,2)`.

At least one axis must downsample.

Update effective spacing:

```python
new_spacing[a] = spacing[a] * stride[a]
```

### Example: BioHub

```text
initial:
(1.625, 0.40625, 0.40625)

down 1:
stride (1,2,2)
effective spacing:
(1.625, 0.8125, 0.8125)

down 2:
stride (1,2,2)
effective spacing:
(1.625, 1.625, 1.625)

down 3:
stride (2,2,2)
effective spacing:
(3.25, 3.25, 3.25)
```

### Example: isotropic data

```text
initial:
(0.7, 0.7, 0.7)

down 1:
(2,2,2)
```

and so on.

## 6. Downsampling implementation

Recommended:

```python
Conv3d(
    in_channels=C_in,
    out_channels=C_out,
    kernel_size=stride,
    stride=stride,
)
```

or a carefully tested pooling + 1x1 projection implementation.

The exact operator is a tunable implementation choice, but the stride-selection policy is part of V1.

## 7. Encoder outputs

The encoder returns:

```python
SpatialPyramid(
    features=[E0, E1, E2, E3],
    spacings_um=[S0, S1, S2, S3],
    padding_masks=[...],
)
```

Shapes are sample-regime dependent.

Example conceptual tensors:

```text
E0 [B,  16, Z0, Y0, X0]
E1 [B,  32, Z1, Y1, X1]
E2 [B,  64, Z2, Y2, X2]
E3 [B, 128, Z3, Y3, X3]
```

## 8. Co-reasoning insertion points

Two co-reasoning blocks are used.

### CR-1

Runs at deepest spatial level E3.

Project:

```text
128 -> D_MODEL(128)
```

No channel change is required.

### CR-2

Runs after the decoder has upsampled back to the E2 physical scale.

The decoder feature is projected to 128 for co-reasoning and returned to its native decoder channel width if needed.

## 9. Decoder

The decoder mirrors the exact encoder stride schedule in reverse.

At each level:

```text
lower-resolution feature
 |
interpolate using inverse encoder stride
 |
1x1x1 projection
 |
concatenate corresponding encoder skip
 |
PhysicalAwareResBlock x2
```

Recommended interpolation:

```python
torch.nn.functional.interpolate(
    mode="trilinear",
    align_corners=False,
)
```

for feature maps.

This interpolation occurs only on learned internal features, not on the original input data or labels.

## 10. Final native mask features

Decoder output:

```python
F_mask: [B, 32, Z_native_patch, Y_native_patch, X_native_patch]
```

Projection:

```python
Conv3d(last_decoder_channels, 32, kernel_size=1)
```

This tensor is used for native-resolution query mask rendering.

## 11. Dense auxiliary heads

From the final decoder feature, predict:

```python
foreground_logits: [B,1,Z,Y,X]
center_logits:     [B,1,Z,Y,X]
boundary_logits:   [B,1,Z,Y,X]
```

Each may be implemented as:

```text
3x3x3/axis-aware block
1x1x1 output conv
```

These are auxiliary training outputs.

## 12. Spatial-only pretraining mode

The spatial backbone must support operation without temporal inputs.

When temporal clues are absent:

```text
co-reasoning blocks = identity on spatial stream
```

This is required for curriculum Phase A and ablation testing.

## 13. Training activation memory

The encoder residual blocks are checkpointed one level at a time, and each
decoder upsample/fusion/residual stage is checkpointed as one region when
training activation checkpointing is enabled. Recalculation uses non-reentrant
PyTorch checkpointing (`use_reentrant=False`). Native spatial tensors, samples,
and skip connections are not cropped or detached; the optimization changes only
which intermediate activations are retained for backward.

Checkpointing is inactive in evaluation mode and under `torch.no_grad()`.
