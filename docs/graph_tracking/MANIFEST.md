# Package contents

```text
src/07_cell_tracking/graph_tracking/
  anchors.py, assignment.py, boundary.py, config.py
  deformation.py, diagnostics.py, geometry.py, pipeline.py
  schemas.py, scoring.py, spatial_graph.py, types.py, validation.py, voting.py
  four_d/
    config.py, schemas.py, types.py, observations.py
    spatial_relations.py, temporal_candidates.py, relation_history.py
    boundary_events.py, factors.py, ambiguity.py, components.py
    milp_model.py, iterative_solver.py, windowing.py
    track_extraction.py, diagnostics.py, validation.py, pipeline.py

tests/graph_tracking/                       # original pairwise backend
tests/test_cell_tracking_graph4d/           # multi-frame backend

docs/graph_tracking/
  README.md, INTEGRATION.md, IMPLEMENTATION_STATUS.md
  FOUR_D_ARCHITECTURE.md, FOUR_D_VALIDATION.md, MANIFEST.md

investigations/stage_07_cell_tracking/graph4d_evaluation.py
```
