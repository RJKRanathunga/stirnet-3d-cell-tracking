# 4D validation and ablations

## Automated validation

Run:

```text
python -m pytest tests/graph_tracking -q
python -m pytest tests/test_cell_tracking_graph_integration.py -q
python -m pytest tests/test_cell_tracking_graph4d -q
python -m pytest tests/test_cell_tracking.py -q
python -m pytest tests/test_track_stitching.py tests/test_cell_lineage.py tests/test_track_reconciliation.py -q
python -m pytest -q
```

The internal validator independently checks complete observation coverage,
observation and `(track_id, frame)` uniqueness, forward acyclic edges, hard
safety validity, predecessor/successor degree, maximum gap, boundary faces,
deterministic IDs, remap integrity, downstream columns and stable sorting.

## Real-data workflow

Start with `mode="shadow", algorithm="windowed_4d"`. Review changed and rejected
edges, selected gaps/expanded candidates, boundary events, component fallbacks,
track-ID remaps and trajectories before enabling apply mode. Endpoint counts
alone are insufficient.

The current 20-frame, 4,421-observation sample exceeded a 300-second bounded
shadow benchmark because one connected ambiguity component required repeated
large fallback flow solves. Each individual solve has a finite configured time
limit and a failed component returns to its provisional assignment, but the
aggregate windowed run does not yet have a global deadline. Run real-data shadow
evaluation as a separately monitored batch job; do not enable apply mode until
runtime and assignment changes have both been reviewed.

After saving a run, compare it with a separately saved provisional tracks table:

```text
python -m investigations.stage_07_cell_tracking.graph4d_evaluation \
  data/processed/stage_7_cell_tracking \
  --provisional-tracks provisional_tracks.csv
```

The report includes candidate recall, transition-type changes, gaps, expansion,
boundary events, fragmentation, acceleration, neighbour residual, fallback rate
and phase runtimes.

## Cumulative ablations

Use `FourDGraphConfig().for_ablation(name)` with:

1. `unary`
2. `trajectory`
3. `spatial`
4. `persistent`
5. `expansion`
6. `boundary`
7. `complete`

Pass the result as `GraphTrackingConfig(..., four_d=ablation_config)`. Compare
the complete diagnostic report across the same samples and manually inspect all
new changed edges. Synthetic tests establish algorithmic behavior, not
biological improvement.
