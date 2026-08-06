# Stage 7 integration contract

## Public API

The only entry point remains `src.api.run_cell_tracking` (also exported from
`src.07_cell_tracking`). `GraphTrackingConfig` is exported from both locations.

```python
GraphTrackingConfig(
    mode="shadow",                 # disabled | shadow | apply
    algorithm="windowed_4d",       # pairwise | windowed_4d
)
```

## Exact insertion points

For `pairwise`, refinement remains after the final per-transition global-motion
assignment and before candidate recording or track-state mutation.

For `windowed_4d`, `step03_pipeline.py` calls the current tracker once with graph
mode disabled. Every final provisional transition snapshot retains source and
target identities, pair/safety/distance/probability matrices, margins,
miss/birth costs, both predictors, decisions, and global motion. Only after the
provisional `TrackingResult` is complete does `run_four_d_graph_tracking(...)`
build and solve the multi-frame graph.

Shadow mode attaches the seven `graph4d_*` tables and leaves all production
tracks/decisions provisional. Apply mode replaces `tracks` and rebuilds
boundary events/predictions, missing predictions, tracking diagnostics,
association events/candidates, track states, metadata, summary, and StageTrace.
The provisional global-motion table remains the prior and is labelled as such.

## Artifacts

The pairwise `graph_*.csv` files are unchanged. The optional 4D files are:

- `graph4d_window_summary.csv`
- `graph4d_component_summary.csv`
- `graph4d_temporal_edges.csv`
- `graph4d_assignment_changes.csv`
- `graph4d_boundary_events.csv`
- `graph4d_solver_diagnostics.csv`
- `graph4d_track_id_map.csv`

`load_stage7_outputs()` uses optional reads for every graph file, so older Stage
7 directories remain loadable. Full dense provisional matrices remain in memory
and are not persisted by default.

When `four_d.save_detailed_debug_artifacts=True`, saving also writes
`graph4d_spatial_edges.npz`, `graph4d_pair_factors.npz`, and
`graph4d_relation_histories.npz`. These compact arrays are opt-in and are not
created by ordinary disabled, shadow, or apply runs.

## Downstream protections

Optimized `tracks` keep the Stage 7 columns used by Stage 8. There is exactly one
row per real observation and no duplicate `(track_id, frame)`. Stage 8 retains
merge reconstruction, Stage 10 receives ordinary one-to-one segments and keeps
its division protections, and Stage 11 continues to consume Stage 7 candidate
evidence and enforce merge/lineage endpoint protections.
