# Stage 05 intensity-statistics investigation

This investigation compares intensity measurements from several image
representations while holding the spatial support fixed:

- **whole foreground:** saved Stage 2 `binary_mask`;
- **individual cells:** saved Stage 3 `instance_labels == cell_id`.

It never reruns segmentation for a comparison method and never modifies
production artifacts.

## Repository compatibility

The code is written against the current repository APIs and layout:

- `src.io.PipelinePaths.discover()`;
- `PipelinePaths.sample_zarr(sample_id)`;
- `PipelinePaths.processed_series(sample_id, name)`;
- `src.io.load_timepoint`;
- `src.api.preprocess_volume(..., return_diagnostics=True)`;
- saved Stage 6 series: `preprocessing`, `masking`, `segmentation`, `cells`;
- Stage 7 `tracks.csv` under `paths.stage7_tracking`.

The package was prepared after inspecting `RJKRanathunga/cell-detection`
`main` through commit `e8d9e0a79028b68093ee80ec057a4f39be8d3934`.

## Compared image representations

1. `raw`
2. `weak_gaussian_0p2um`
3. `weak_gaussian_0p4um`
4. `normalized`
5. `current_denoised`
6. `production_preprocessed`

Weak Gaussian filtering is applied directly to raw-scale `float32` data using
the physical voxel size `(1.625, 0.40625, 0.40625)` micrometers.

The canonical preprocessing pipeline is replayed for every frame. Its final
result is compared with the saved Stage 6 preprocessed array. A mismatch raises
an error by default, preventing accidental comparison against stale artifacts.

## Installation

Extract this ZIP into the repository root. It adds only:

```text
investigations/
tests/test_intensity_statistics.py
```

Generated outputs are written under `data/`, which the repository already
ignores.

## Run

From the activated repository environment:

```powershell
python -m investigations.stage_05_feature_extraction.intensity_statistics.run_analysis `
  --sample-id 44b6_0113de3b `
  --frames 0:20 `
  --weak-sigmas 0.2,0.4
```

Use manually checked complete tracks:

```powershell
python -m investigations.stage_05_feature_extraction.intensity_statistics.run_analysis `
  --sample-id 44b6_0113de3b `
  --track-ids 12,37,91
```

Specify another Stage 7 table when necessary:

```powershell
python -m investigations.stage_05_feature_extraction.intensity_statistics.run_analysis `
  --tracks-csv data/sample/processed/stage_7_cell_tracking/tracks.csv
```

## Output

A timestamped run directory is created at:

```text
data/investigations/stage_05_feature_extraction/intensity_statistics/
└── <sample_id>/
    └── <run_timestamp>/
        ├── run_metadata.json
        ├── tables/
        │   ├── cell_intensity_statistics.csv
        │   ├── foreground_statistics.csv
        │   ├── tracked_cell_intensity_statistics.csv
        │   ├── track_candidates.csv
        │   ├── track_temporal_statistics.csv
        │   ├── method_comparison.csv
        │   ├── between_within_statistics.csv
        │   ├── association_pairs.csv
        │   └── association_summary.csv
        └── figures/
```

## Interpretation

Do not select the method with the lowest variance alone. A useful method needs:

- low random within-track variation;
- preserved between-cell variation;
- strong raw rank agreement;
- low cost for correct consecutive-frame pairs;
- higher cost for plausible nearby incorrect pairs.

`between_within_statistics.csv` and the separation rows in
`association_summary.csv` are the most direct summaries of that trade-off.

Automatically selected complete tracks are candidates, not verified ground
truth. Review `track_candidates.csv`, then rerun with `--track-ids` for the
final analysis.
