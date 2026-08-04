# Stage 12 final visualization

Copy the included paths into the repository root:

- `src/12_final_visualization/`
- `notebooks/12_final_visualization.ipynb`
- `tests/test_final_visualization.py` (recommended regression coverage)

The implementation does not modify Stage 9. It loads the stable Stage 11 artifacts through the existing `load_stage11_outputs` function and provides:

- final canonical tracks and cell-ID labels;
- valid boundary entry/exit, division, and merge endpoint groups;
- unexplained interior birth and termination failure groups;
- unresolved Stage 11 endings, forced repairs, modified tracks, short tracks, and temporal gaps;
- a prioritized per-track audit table and normalized event table;
- a Napari dock navigator that jumps directly to each failure or warning;
- reuse of the existing tracking-scene and cell-volume extractors from the notebook.

No changes to `src/api.py`, `src/io`, or `src/09_visualization` are required.
