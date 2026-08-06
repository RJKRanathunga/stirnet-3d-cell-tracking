# Stage 7 graph tracking package

This archive contains an integration-ready `graph_tracking/` package for
`src/07_cell_tracking/` in the `RJKRanathunga/cell-detection` repository.

It implements:

- sparse physical 3D cell graphs per frame;
- reliable temporal anchor extraction from the current Stage 7 assignment;
- vector-preserving Hough-style forward voting;
- local translation, similarity, and affine deformation prediction;
- neighbour-relative volume pattern scoring without copying node features;
- backward voting for boundary entries;
- outside-volume forward voting for boundary exits;
- graph-adjusted pair, miss, and birth costs;
- `disabled`, `shadow`, and `apply` modes;
- detailed candidate, vote, boundary, and decision diagnostics.

The package is integrated into the current Stage 7 production entry point.
Graph tracking remains disabled by default; shadow and apply modes are exposed
through `run_cell_tracking(..., graph_config=GraphTrackingConfig(...))`. See
`INTEGRATION.md` for the wiring contract used by `step02_association.py`,
`step03_pipeline.py`, `TrackingResult`, and the Stage 7 I/O layer.
