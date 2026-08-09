# Learned Instance Segmentation — Cubic External Dataset Pipeline

This package converts densely annotated 3-D microscopy datasets into the common
input representation used by `VectorInstanceCNN`.

## Canonical representation

Every selected single cell / neighboring pair / group becomes a fixed
`(64,64,64)` tensor with **cubic canonical voxels**.

```text
native image + dense GT labels
        ↓
select a single / pair / group
        ↓
tight union bbox in true physical coordinates
        ↓
ONE scalar normalization scale
        ↓
local image + labels + valid mask → cubic [64,64,64]
        ↓
synthetic Stage-2 connected component
        ↓
canonical EDT + effective-marker heatmap
        ↓
4-channel CNN input
        ↓
foreground + vectors + boundary + center targets
```

If the source bbox is `B=(Bz,By,Bx)` micrometres and the usable canonical voxel
span is `F`, the transform chooses

```text
scale_vox_per_um = min_i(component_occupancy * F_i / B_i)
```

subject to safety clamps.  The same scalar is applied to physical Z/Y/X, so
morphology is preserved.  The default occupancy is `0.78`.

## Why native spacing is still essential

The canonical voxels are cubic, but source voxels usually are not.  For example,
Biohub spacing `(1.625,0.40625,0.40625)` must first be used to recover physical
geometry.  Only then is the object resampled into the normalized cube.  This is
what prevents a physically near-spherical Biohub nucleus from remaining flat in
canonical voxel coordinates.

## Input contract

`TrainingSample.inputs` is `float32 [4,64,64,64]`:

1. normalized fluorescence,
2. synthetic Stage-2-like component mask,
3. normalized canonical EDT,
4. effective-marker Gaussian heatmap.

## Target contract

- `foreground`: `[1,64,64,64]`
- `vectors_normalized`: `[3,64,64,64]`
- `boundary`: `[1,64,64,64]`
- `center`: `[1,64,64,64]`
- `instance_labels`: `[64,64,64]`

Vectors are canonical axis fractions.  For 64^3, multiplying by `(63,63,63)`
recovers canonical voxel offsets.

## Boundary quality rule

Training instances that touch a native source-volume boundary are rejected by
default because they are truncated GT objects.  This does not apply to inference.

## Inference

`inference.build_inference_roi(...)` constructs the same cubic normalization
from a Stage-2 component and stores the invertible transform needed to map CNN
center predictions back to native Biohub coordinates.

## NIS3D

The NIS3D adapter parses the dedicated `Resolution:` field, converts XYZ metadata
to ZYX, and excludes derived `suggestive splitting` folders.  Dataset discovery
and native spacing remain independent of the canonical cube design.

## Debugging

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_sample `
  --dataset nis3d `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\NIS3D" `
  --volume-index 0 `
  --instance-ids 1 106 `
  --napari
```

The visualizer starts in 3-D and displays the cubic canonical lattice with unit
scale.

## Tests

```powershell
python -m pytest learned/instance_segmentation/datasets/tests learned/instance_segmentation/model/tests -q
```
