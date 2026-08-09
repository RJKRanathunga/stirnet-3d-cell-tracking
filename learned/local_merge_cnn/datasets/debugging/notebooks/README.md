# Debugging notebooks

Use this directory for exploratory notebooks that inspect the cubic learned
instance-segmentation dataset pipeline. Useful checks include:

- native physical bbox versus canonical voxel bbox,
- normalized component occupancy inside `(64,64,64)`,
- source-boundary rejection,
- GT center versus effective-marker location,
- center-vector direction and magnitude,
- synthetic merge connectivity,
- inverse mapping from canonical centers to native coordinates.
