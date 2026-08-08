# Cubic object-centric learned instance segmentation

The learned correction model operates on a fixed **64 x 64 x 64 cubic canonical
volume**.  Native microscopy voxels are first interpreted with their true source
spacing in micrometres.  A single scalar scale is then applied to the complete
selected component/group so that its physical morphology and relative cell
geometry are preserved while the group occupies about 78% of the usable cube.

```text
native voxels + native spacing
            ↓
true physical 3-D geometry
            ↓
one scalar object normalization (vox/um)
            ↓
64 x 64 x 64 unit-cubic canonical lattice
            ↓
4-channel CNN input
```

The canonical lattice is not Biohub physical space.  Its voxels are symmetric
unit cubes.  Biohub anisotropy, NIS3D spacing, C. elegans spacing, and other
source acquisition geometries are handled only by the native-to-canonical
transform.

## Input channels

1. robust-normalized fluorescence,
2. Stage-2-like connected component mask,
3. normalized Euclidean distance transform in canonical voxels,
4. effective-marker heatmap in canonical voxels.

Raw fluorescence is never fabricated to create a merge.  Only the training mask
is synthetically connected when constructing merge examples.

## Supervision

The model predicts foreground, 3-D center vectors, internal boundary evidence,
and a center heatmap.  Dense vectors are canonical axis-fraction displacements.
For the default 64^3 tensor, multiplying each vector channel by 63 converts it to
canonical voxel displacement.

## Backbone

The backbone is a fully isotropic residual 3-D U-Net:

```text
4 x 64^3
  ↓
16 x 64^3
  ↓ 2x2x2
32 x 32^3
  ↓
64 x 16^3
  ↓
128 x 8^3
  ↓
192 x 4^3
```

The decoder mirrors those levels with trilinear upsampling and skip connections.
All residual spatial convolutions are isotropic 3x3x3 operations.

## Marker evidence

Production Stage-3 peak logic is reused, but with a dedicated normalized
canonical marker configuration.  Its distance parameters are canonical voxel
units rather than biological micrometres, and geometric completion is disabled
to avoid label-derived leakage.

## Training quality gates

Selected GT instances that already touch a native source-volume boundary are
rejected by default because their complete center/shape is unavailable.  This
is a training-data rule only; inference components near Biohub boundaries are
still processed.

## Training / inference symmetry

Training constructs the transform from the selected GT group bbox.  Inference
constructs the same transform from the observed Stage-2 connected-component
bbox.  Predicted canonical centers are inverse-mapped into native voxel
coordinates with the stored transform.
