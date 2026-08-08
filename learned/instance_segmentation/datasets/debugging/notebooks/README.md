# Debugging notebooks

This directory is intentionally reserved for local exploratory notebooks.  The
production implementation lives in `datasets/core/`; notebooks should import
those functions rather than duplicate the transform or target logic.

For the object-centric architecture, useful diagnostics are:

- native group bounding box and source spacing,
- chosen isotropic `normalization_scale`,
- normalized component occupancy inside `(16, 64, 64)`,
- per-instance voxel/bbox survival after resampling,
- canonical target centers and their inverse-mapped native coordinates.
