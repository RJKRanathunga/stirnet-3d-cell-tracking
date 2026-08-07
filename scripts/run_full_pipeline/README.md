# Full-dataset pipeline runner

This runner discovers every sample under a configurable full Biohub dataset root and executes the current non-visual cell-tracking pipeline through Stage 11.

## Expected full-dataset layout

```text
<FULL_DATASET_ROOT>/
└── train/
    ├── 44b6_0b24845f/
    │   ├── 44b6_0b24845f.zarr/
    │   │   └── 0/
    │   ├── ground_truth/
    │   │   ├── ground_truth_nodes.csv
    │   │   └── ground_truth_edges.csv
    │   └── README.txt
    └── <another_sample>/
        └── <another_sample>.zarr/
            └── 0/
```

The `ground_truth` directory is recorded when present, but it is not required by the batch runner for the current Stage 1-11 execution path.

## What is executed

The batch order is:

```text
Stages 1-6  -> existing src.dataset_processing.process_dataset
Stage 7     -> current apply-mode windowed 4D graph tracking
Stage 8     -> track stitching / merge repair
Stage 9     -> skipped (visualization only)
Stage 10    -> cell lineage
Stage 11    -> track reconciliation
Stage 12    -> not run (visualization only)
```

The Stage 7 defaults intentionally match the current `07_cell_tracking.ipynb` research configuration:

```text
graph mode                 = apply
graph algorithm            = windowed_4d
window size                = 7
maximum gap frames         = 2
solver time limit          = 30 s
iterative fallback passes  = 5
```

## Default paths

`run.py` contains editable defaults:

```python
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "full"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "full_processed"
```

Both paths can also be overridden from the command line.

## Commands

Run from the repository root.

### Process every available full sample

```powershell
python scripts/run_full_pipeline/run.py
```

Equivalent with explicit paths:

```powershell
python scripts/run_full_pipeline/run.py `
    --dataset-root data/full `
    --output-root data/full_processed
```

### Process one sample first

```powershell
python scripts/run_full_pipeline/run.py `
    --sample 44b6_0b24845f
```

Repeat `--sample` to process a selected set:

```powershell
python scripts/run_full_pipeline/run.py `
    --sample 44b6_0b24845f `
    --sample 44b6_0113de3b
```

### Resume an interrupted or partially completed batch

```powershell
python scripts/run_full_pipeline/run.py --resume
```

Each successful stage gets a `_SUCCESS.json` marker. With `--resume`, valid completed stages are skipped. A stage with no valid success marker is treated as incomplete; its batch output directory is cleared and the stage is rerun.

### Rerun only Stage 7 through Stage 11

This expects the per-sample Stage 6 outputs to already exist under the configured output root.

```powershell
python scripts/run_full_pipeline/run.py `
    --start-stage 7 `
    --end-stage 11
```

Resume while doing the same:

```powershell
python scripts/run_full_pipeline/run.py `
    --start-stage 7 `
    --end-stage 11 `
    --resume
```

### Run only Stage 10 and Stage 11

```powershell
python scripts/run_full_pipeline/run.py `
    --start-stage 10 `
    --end-stage 11
```

The runner loads the required Stage 6, Stage 7, and Stage 8 artifacts from that sample's batch output directories.

### Stop on the first bad sample

By default, one failed sample is recorded and the runner continues with the next sample. To stop immediately:

```powershell
python scripts/run_full_pipeline/run.py --stop-on-error
```

### Override the Stage 7 graph experiment

For example, to run the provisional tracker without graph optimization:

```powershell
python scripts/run_full_pipeline/run.py `
    --graph-mode disabled `
    --graph-algorithm pairwise
```

The normal default is still the current apply-mode `windowed_4d` configuration.

## Output layout

Raw input is never written into. Batch results are isolated by sample:

```text
data/full_processed/
├── pipeline_manifest.csv
└── train/
    ├── 44b6_0b24845f/
    │   ├── source.json
    │   ├── pipeline_status.json
    │   ├── stage_6_processed_dataset/
    │   ├── stage_7_cell_tracking/
    │   ├── stage_8_track_stitching/
    │   ├── stage_10_cell_lineage/
    │   └── stage_11_track_reconciliation/
    └── 44b6_0113de3b/
        └── ...
```

This per-sample layout is intentional. The current notebook-oriented `PipelinePaths` uses shared Stage 7/8/10/11 directories for the selected sample, which would overwrite one sample with another during a full-dataset run. The batch runner instead calls the existing stage algorithms and savers with explicit per-sample output directories.

## Failure information

If a stage fails, the stage directory contains `_FAILED.json` with the exception and traceback. The latest state of each sample/stage is also written to:

```text
<OUTPUT_ROOT>/pipeline_manifest.csv
```

After fixing the issue, run again with `--resume`. The failed stage has no `_SUCCESS.json`, so it is rerun while earlier successful stages are preserved.
