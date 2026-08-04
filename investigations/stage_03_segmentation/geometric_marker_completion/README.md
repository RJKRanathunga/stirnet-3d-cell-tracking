# Geometric marker completion investigation

This directory validates the production-active Stage 3 geometric completion
path. The scripts call `src.03_segmentation.pipeline.segment_instances_detailed`
and the production dataframe helpers; they do not contain another geometry
implementation.

Run one saved 3-D binary component mask:

```powershell
.\.venv\Scripts\python.exe investigations\stage_03_segmentation\geometric_marker_completion\run_component_analysis.py path\to\mask.npy
```

Use `--force-geometric-analysis` to bypass only the cheap size eligibility
gate in a focused validation. This does not change the production default: all
eligible components run geometry normally.

Summarize any generated case directories:

```powershell
.\.venv\Scripts\python.exe investigations\stage_03_segmentation\geometric_marker_completion\summarize_results.py
```

Outputs include effective EDT peaks, caps, candidate and selected bodies,
cross-sections, supplemental/final markers, rejection reasons, and before/after
instance counts. Generated outputs are ignored by Git. No real-data improvement
is claimed until curated merge scenes have been evaluated.
