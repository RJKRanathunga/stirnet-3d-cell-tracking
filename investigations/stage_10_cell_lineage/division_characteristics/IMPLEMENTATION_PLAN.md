# Implementation plan

## Repository integration

- Treat `data/tracking_scenes/divisions/` as manual event annotations.
- Read scene metadata through the existing `src.io.load_tracking_scene` contract.
- Reload full raw frames and saved Stage 6 artifacts through `PipelinePaths` and `load_timepoint`.
- Read Stage 8 tracks when available, with a Stage 7 fallback.
- Never rewrite production arrays or tracking tables.

## Event inference

1. Sort frames present in `scene.json -> selected_cells`.
2. Require one selected cell before the transition.
3. Detect the first selected frame containing two cells.
4. Define that frame as relative frame zero and the child-birth frame.
5. Require two selected cells in all later saved selections.
6. Accept non-consecutive extraction windows while recording the transition gap.

## Measurements

- physical morphology from the production instance mask;
- raw and preprocessed mask intensity distributions;
- eroded-core measurements;
- fixed physical-radius measurements;
- local background-shell correction;
- full-frame foreground-reference correction;
- daughter separation, combined-child conservation and asymmetry;
- parent-history baseline ratios using the available pre-event frames.

## Outputs

- validation and case manifest tables;
- wide per-observation feature table;
- long normalized trajectory table;
- combined-child and event summary tables;
- cross-case feature consistency table;
- per-case and event-aligned figures.
