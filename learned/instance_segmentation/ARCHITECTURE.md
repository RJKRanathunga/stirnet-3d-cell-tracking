# Object-centric learned instance segmentation

The CNN does not receive a fixed biological field of view.  Each candidate
component/group is mapped to the fixed `(16, 64, 64)` tensor with one isotropic
scale in source physical space.  The limiting group-bounding-box axis occupies
`SampleBuildConfig.component_occupancy` of the usable canonical span (default
`0.78`), leaving context around the object.

Training and inference share the same `CanonicalTransform` contract.  The
transform stores source spacing, native center, normalization scale, canonical
sampling geometry, and supports exact forward/inverse coordinate conversion.

Dense center vectors are expressed in canonical axis fractions rather than
micrometres.  EDT, marker heatmaps, center Gaussians, synthetic merge bridges,
and boundary widths are likewise defined on the canonical ROI.  This makes the
learned task independent of absolute organism/cell size while preserving shape
and relative cell geometry.

Do not independently resize cells inside a pair/group.  One transform is always
applied to the complete local scene.
