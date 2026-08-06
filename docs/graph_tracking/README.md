# Stage 7 graph tracking

Stage 7 exposes two graph algorithms through the existing
`run_cell_tracking(..., graph_config=...)` entry point.

- `pairwise` is the original two-frame refinement. It runs inside a transition
  after global-motion assignment and before state mutation.
- `windowed_4d` runs the ordinary Stage 7 sequence as a provisional tracker,
  retains its evidence in memory, and optimizes ambiguous spatiotemporal
  components only after the entire sequence is available.

The production default remains disabled:

```python
run_cell_tracking(time_frames, sample_id=sample_id)
```

The recommended first real-data run is diagnostic-only:

```python
from src.api import GraphTrackingConfig, run_cell_tracking

result = run_cell_tracking(
    time_frames,
    sample_id=sample_id,
    graph_config=GraphTrackingConfig(
        mode="shadow",
        algorithm="windowed_4d",
    ),
)
```

Modes have strict meanings:

- `disabled` returns the provisional Stage 7 result and runs no graph solver.
- `shadow` runs the selected graph algorithm and emits diagnostics while
  returning provisional tracks and decisions.
- `apply` returns optimized tracks and consistently rebuilt Stage 7 artifacts.

The 4D solver is a one-predecessor/one-successor continuation model. Stage 8
still owns merge reconstruction, Stage 10 owns division/lineage, and Stage 11
owns residual post-lineage endpoint reconciliation.

See [FOUR_D_ARCHITECTURE.md](FOUR_D_ARCHITECTURE.md) for the model and
[FOUR_D_VALIDATION.md](FOUR_D_VALIDATION.md) for validation and ablations.
