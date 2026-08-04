# Geometric marker completion investigation

This directory validates the production-active Stage 3 geometric completion
path. The scripts call `src.03_segmentation.pipeline.segment_instances_detailed`
and the production dataframe helpers; they do not contain another geometry
implementation.

Run one saved 3-D binary component mask:

```powershell
.\.venv\Scripts\python.exe investigations\stage_03_segmentation\geometric_marker_completion\run_component_analysis.py path\to\mask.npy
```

Use `--force-geometric-analysis` to run geometry regardless of the peak-based
candidate result in a focused validation. This diagnostic override does not
disable or change production candidate selection.

Benchmark one or more frames from a ZYX or TZYX NumPy mask:

```powershell
.\.venv\Scripts\python.exe investigations\stage_03_segmentation\geometric_marker_completion\benchmark_candidate_detection.py path\to\mask.npy --frames 0,1 --csv outputs\candidate_benchmark.csv
```

Summarize any generated case directories:

```powershell
.\.venv\Scripts\python.exe investigations\stage_03_segmentation\geometric_marker_completion\summarize_results.py
```

Outputs include effective EDT peaks, binary-LoG shape peaks, center proposals,
candidate routes, caps, candidate and selected bodies, cross-sections,
supplemental/final markers, rejection reasons, and before/after instance counts.
Generated outputs are ignored by Git. No real-data improvement is claimed until
curated merge scenes have been evaluated.
