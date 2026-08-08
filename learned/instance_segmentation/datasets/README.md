# Learned Instance Segmentation — Object-Centric External Dataset Pipeline

This directory converts densely annotated 3-D microscopy volumes into the fixed
`VectorInstanceCNN` tensor shape without forcing every source dataset to share
Biohub's **absolute biological cell scale**.

## Architecture

```text
native image + dense GT labels
        ↓
select a single / neighboring pair / group
        ↓
tight union bbox in native physical coordinates
        ↓
ONE isotropic physical normalization scale
        ↓
local image + labels + valid mask → canonical [16,64,64]
        ↓
synthetic Stage-2 connected component
        ↓
canonical EDT + effective-marker heatmap
        ↓
4-channel CNN input
        ↓
foreground + canonical vectors + boundary + center supervision
```

At inference, the exact same transform is constructed from the observed Stage-2
connected-component bbox. Predicted canonical centers can therefore be mapped
back to original Biohub voxel coordinates exactly.

## Why the scale is per component

If a selected source component has physical bbox `B=(Bz,By,Bx)` and the usable
canonical span is `F`, the transform chooses

```text
scale = min_i(component_occupancy * F_i / B_i)
```

subject to safety clamps. The same scalar multiplies physical Z, Y, and X, so
source morphology is not anisotropically stretched. The output grid itself
remains `(16,64,64)` with canonical spacing `(1.625,0.40625,0.40625)` because
the existing CNN backbone is designed around that anisotropic tensor geometry.
Those values now define **canonical ROI distance units**, not the source cell's
original biological micrometre scale.

Default occupancy is `0.78`, leaving context around the selected group.

## Input contract

`TrainingSample.inputs` is `float32 [4,Z,Y,X]`:

1. robust-normalized fluorescence,
2. synthetic Stage-2-like connected component mask,
3. normalized canonical EDT,
4. effective-marker Gaussian heatmap evaluated on the canonical ROI.

The fluorescence and GT are resampled with the same transform. Only the mask is
synthetically bridged; raw fluorescence is never pasted or modified to fabricate
a merge.

## Target contract

- `foreground`: `[1,Z,Y,X]`
- `vectors_normalized`: `[3,Z,Y,X]`
- `boundary`: `[1,Z,Y,X]`
- `center`: `[1,Z,Y,X]`
- `instance_labels`: `[Z,Y,X]`

Vectors are **canonical axis-fraction displacements**. For a canonical shape
`(Z,Y,X)`, multiplying the 3 vector channels by `(Z-1,Y-1,X-1)` recovers the
predicted displacement in canonical voxels. No `vector_max_distance_um` exists
in the new model contract.

## Coordinate transform

`core/component_transform.py` owns the transform. `CanonicalTransform` exposes:

```python
canonical = transform.native_to_canonical(native_zyx)
native = transform.canonical_to_native(canonical_zyx)
```

The transform is stored directly on every `TrainingSample` and recorded in
sample metadata.

## Training / inference symmetry

Training:

```text
selected GT group bbox → canonical transform → sample
```

Inference:

```text
Stage-2 component bbox → canonical transform → CNN → inverse transform
```

Use `inference.build_inference_roi(...)` to construct the inference tensor with
the same normalization rule.

## Adapters

- `c_elegans.py`: Zenodo 5942575 nuclei volumes.
- `nis3d.py`: NIS3D dense 3-D nuclei benchmark.
- `blastospim.py`: extracted BlastoSPIM archives.

### NIS3D fixes included

The adapter no longer extracts the first three numbers from arbitrary prose in
`Info.txt`. It explicitly parses the `Resolution:` field. Therefore:

```text
Drosophila_1  -> (1.0, 1.0, 1.0) ZYX µm
Drosophila_2  -> (1.0, 1.0, 1.0)
MusMusculus_1 -> (1.0, 1.0, 1.0)
MusMusculus_2 -> (1.0, 1.0, 1.0)
Zebrafish_1   -> (2.5, 0.43, 0.43)
Zebrafish_2   -> (1.0, 1.0, 1.0)
```

Derived `suggestive splitting` directories are excluded when discovering the
six primary NIS3D volumes.

## Debugging

Inspect a volume:

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_volume `
  --dataset nis3d `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\NIS3D" `
  --list
```

Build a normalized pair:

```powershell
python -m learned.instance_segmentation.datasets.debugging.inspect_sample `
  --dataset nis3d `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\NIS3D" `
  --volume-index 0 --pair-index 0 --napari
```

The command prints the source bbox, isotropic normalization scale, canonical
bbox extent, canonical GT centers, and inverse-mapped native centers.

Scan buildability:

```powershell
python -m learned.instance_segmentation.datasets.debugging.scan_pairs `
  --dataset nis3d `
  --root "D:\Projects\Kaggle\cell-tracking\data\external\NIS3D" `
  --volume-index 0 --max-pairs 100
```

Large pairs are no longer rejected simply because their physical FOV exceeds
Biohub's fixed 26 µm crop. They are scaled. Rejections now represent genuine
quality problems such as insufficient canonical resolution, pathological aspect
ratio, source-boundary truncation, or marker generation failure.

## Model

The existing anisotropic residual 3-D U-Net is preserved. Its output heads remain:

- foreground,
- vectors,
- internal boundary,
- center heatmap.

Only vector semantics changed from fixed physical micrometre offsets to
canonical axis fractions.

## Tests

From the repository root after replacing `learned/instance_segmentation/`:

```powershell
python -m pytest learned/instance_segmentation/datasets/tests -q
```
