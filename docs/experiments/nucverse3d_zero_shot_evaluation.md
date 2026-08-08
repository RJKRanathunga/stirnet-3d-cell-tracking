# NucVerse3D Zero-Shot Evaluation

## Purpose

NucVerse3D was evaluated as a possible pretrained alternative or supplement
to the project's 3D instance-segmentation pipeline.

The primary question was whether its pretrained model could solve the
remaining difficult touching/merged-nucleus cases without requiring a
project-specific learned model.

## Model

Repository:
Segovia-lab/NucVerse3D

Checkpoint:
`resunet_combined_scaled_1000/best_model.weights.h5`

Model:
Attention ResUNet 3D

Parameters:
40,458,005

Input:
`64 x 128 x 128 x 1`

Outputs:
- 2-channel foreground/background prediction
- 3-channel 3D vector field

TensorFlow:
2.16.2

Inference was performed under WSL using an NVIDIA RTX 4050 Laptop GPU.

## Biohub evaluation

The pretrained model was tested zero-shot on Biohub 3D volumes using native
Biohub data with voxel spacing:

- Z: 1.625 µm/voxel
- Y: 0.40625 µm/voxel
- X: 0.40625 µm/voxel

Samples inspected included:

- `44b6_0b24845f`
- `44b6_0c582fdc`
- `44b6_0113de3b`

## Observations

Performance varied across samples.

On `44b6_0b24845f`, the model substantially under-detected nuclei.

On `44b6_0c582fdc`, general nucleus detection was considerably better.

On `44b6_0113de3b`, the model produced very strong overall nucleus
segmentation. Almost all clearly visible nuclei were detected, with most
misses corresponding to very faint nuclei.

However, across the useful cases, the principal failure mode relevant to
this project remained unresolved:

**closely touching or merged nuclei were frequently predicted as a single
instance.**

This is also the principal remaining failure mode of the project's existing
computer-vision instance-segmentation pipeline.

## Conclusion

NucVerse3D is a strong general 3D nuclear instance-segmentation model and
demonstrated good zero-shot performance on some Biohub samples.

However, it does not provide a meaningful advantage for the specific problem
that remains unsolved in this project: separating difficult merged/touching
nuclei.

The existing computer-vision pipeline already provides strong general
segmentation and is substantially easier to inspect, modify, and adapt to
Biohub data.

Replacing or supplementing it with NucVerse3D would therefore add:

- a large TensorFlow model,
- a separate Python 3.10/WSL runtime,
- pretrained model storage,
- additional inference cost,
- additional integration complexity,

without resolving the target failure mode.

NucVerse3D was therefore not retained as part of the active codebase.

## Design implication

Learned development should remain focused on a specialized local model for
ambiguous components rather than another general-purpose nucleus segmenter.

The learned model should specifically address questions such as:

1. Does this apparent component contain one nucleus or multiple nuclei?
2. If multiple nuclei are present, where are their centers?
3. Which voxels belong to each nucleus?
4. Where should the separation boundary lie?

The existing computer-vision segmentation remains the front end, while the
learned model targets the difficult merge cases that the deterministic
pipeline cannot reliably resolve.