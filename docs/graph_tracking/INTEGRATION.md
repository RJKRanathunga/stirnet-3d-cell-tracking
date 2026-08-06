# Integration into the current Stage 7 pipeline

## 1. Copy the package

Extract `graph_tracking/` into:

```text
src/07_cell_tracking/graph_tracking/
```

## 2. Preserve the current public API

In `src/07_cell_tracking/step03_pipeline.py`, add an optional configuration:

```python
from .graph_tracking import GraphTrackingConfig, refine_transition_with_graph


def run_cell_tracking(
    time_frames: list[pd.DataFrame],
    *,
    sample_id: str = "44b6_0113de3b",
    graph_config: GraphTrackingConfig | None = None,
    return_diagnostics: bool = False,
):
    graph_config = graph_config or GraphTrackingConfig(mode="disabled")
```

## 3. Insertion point

The current repository selects the final base assignment after its initial and
refined global-motion passes. Insert graph refinement immediately after that
selection and before `record_top_candidates(...)` or any track-state mutation:

```python
graph_result = refine_transition_with_graph(
    eligible_states=eligible_states,
    detections=detections,
    current_frame=current_frame,
    base_assignment=assignment,
    volume_shape_zyx=VOLUME_SHAPE_ZYX,
    voxel_size_zyx_um=VOXEL_SIZE_ZYX,
    config=graph_config,
)
assignment = graph_result.assignment
```

- `disabled`: current Stage 7 assignment is returned unchanged.
- `shadow`: current assignment is returned unchanged, while
  `graph_result.graph_assignment` and diagnostics show the alternative.
- `apply`: the graph-refined assignment is returned.

## 4. Reuse the repository's augmented solver

The package includes a compatible internal augmented Hungarian solver. After
refactoring the current `augmented_assignment` into a stable callable, it can be
passed through `assignment_solver`:

```python
def stage7_solver(pair_cost, miss_cost, birth_cost):
    return augmented_assignment(
        pair_cost_matrix=pair_cost,
        miss_costs=miss_cost,
        birth_costs=birth_cost,
    )


graph_result = refine_transition_with_graph(
    ...,
    assignment_solver=stage7_solver,
)
```

The callback must return:

```text
rows
cols
missed_state_indices
birth_detection_indices
objective_cost
```

## 5. Aggregate graph artifacts

Create frame-level record lists in `run_cell_tracking` and append:

```python
graph_transition_summary_records.append(graph_result.transition_summary)
graph_candidate_evidence_records.append(graph_result.candidate_evidence)
graph_anchor_vote_records.append(graph_result.anchor_votes)
graph_boundary_hypothesis_records.append(graph_result.boundary_hypotheses)
graph_refinement_event_records.append(graph_result.refinement_events)
```

At the end, concatenate with `pd.concat(..., ignore_index=True)`.

## 6. Extend `TrackingResult`

Add:

```text
graph_transition_summary
graph_candidate_evidence
graph_anchor_votes
graph_boundary_hypotheses
graph_refinement_events
```

## 7. Extend Stage 7 I/O

Update `src/io/stage_io.py` mappings with:

```text
graph_transition_summary.csv
graph_candidate_evidence.csv
graph_anchor_votes.csv
graph_boundary_hypotheses.csv
graph_refinement_events.csv
```

These files use stable columns defined in `graph_tracking/schemas.py`.

## 8. Boundary event integration

Do not immediately deactivate a track when graph exit evidence is present.
Map supported exit hypotheses to a pending event, retain the current boundary
reacquisition window, and confirm only after the existing pending period:

```text
graph_exit_predicted
graph_exit_pending
graph_exit_confirmed
graph_exit_rejected_reacquired
```

For supported entries, retain the current birth decision and annotate it as
`graph_entry_supported`.

An `inside_predecessor_vote` should increase the birth cost and allow the graph
assignment to reconnect the detection to an existing source track.

## 9. State additions

Optional first integration fields:

```python
"graph_last_confidence": 0.0,
"graph_anchor_support_count": 0,
"graph_boundary_hypothesis": "",
"graph_boundary_face": "",
"graph_boundary_confidence": 0.0,
"graph_boundary_since_frame": None,
```

Persistent neighbour-history state is intentionally deferred until the
adjacent-frame implementation is validated.

## 10. Rollout

1. Verify `disabled` output identity.
2. Run `shadow` on all curated failures.
3. Enable `apply` for interior ambiguous cases.
4. Enable coverage-aware boundary cases.
5. Add multi-frame graph optimization only after adjacent-frame validation.
