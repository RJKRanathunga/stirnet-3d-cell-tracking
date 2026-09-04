"""Cached, non-GUI replay of canonical pipeline Stages 1 through 5."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
from scipy import ndimage

from src.api import create_binary_mask, detect_cells, extract_cell_features, preprocess_volume

from .component_debug import component_debug_result
from .models import PipelineReplayState, ReplayMode, ReplayResult
from .source import PipelineReplaySource

preprocessing_config_module = import_module("src.source_instances.preprocessing.config")
masking_config_module = import_module("src.source_instances.foreground.config")
segmentation_module = import_module("src.source_instances.segmentation.pipeline")
segmentation_config_module = import_module("src.source_instances.segmentation.config")

DEFAULT_PREPROCESSING_CONFIG = preprocessing_config_module.DEFAULT_PREPROCESSING_CONFIG
DEFAULT_MASKING_CONFIG = masking_config_module.DEFAULT_MASKING_CONFIG
DEFAULT_SEGMENTATION_CONFIG = segmentation_config_module.DEFAULT_SEGMENTATION_CONFIG


class PipelineReplayRunner:
    """Replay state machine; it never writes production artifacts or defaults."""

    def __init__(self, source: PipelineReplaySource) -> None:
        self.source = source
        self.state = PipelineReplayState(
            DEFAULT_PREPROCESSING_CONFIG,
            DEFAULT_MASKING_CONFIG,
            DEFAULT_SEGMENTATION_CONFIG,
        )
        self.baseline = None
        self._raw_work: np.ndarray | None = None
        self._processed_work: np.ndarray | None = None
        self._mask_work: np.ndarray | None = None
        self._labels_work: np.ndarray | None = None
        self._markers_work: np.ndarray | None = None
        self._cells_work = None
        self._features_work = None
        self._display_slices: tuple[slice, slice, slice] | None = None

    def restore_production_defaults(self) -> None:
        self.state.preprocessing_config = DEFAULT_PREPROCESSING_CONFIG
        self.state.masking_config = DEFAULT_MASKING_CONFIG
        self.state.segmentation_config = DEFAULT_SEGMENTATION_CONFIG
        self.invalidate_from(1)

    def reset_stage(self, stage: int) -> None:
        if stage == 1:
            self.state.preprocessing_config = DEFAULT_PREPROCESSING_CONFIG
        elif stage == 2:
            self.state.masking_config = DEFAULT_MASKING_CONFIG
        elif stage == 3:
            self.state.segmentation_config = DEFAULT_SEGMENTATION_CONFIG
        elif stage in (4, 5):
            self.invalidate_from(stage)
            return
        else:
            raise ValueError("stage must be between 1 and 5")
        self.invalidate_from(stage)

    def update_preprocessing_config(self, **changes) -> None:
        updated = replace(self.state.preprocessing_config, **changes)
        if updated != self.state.preprocessing_config:
            self.state.preprocessing_config = updated
            self.invalidate_from(1)

    def update_masking_config(self, **changes) -> None:
        updated = replace(self.state.masking_config, **changes)
        if updated != self.state.masking_config:
            self.state.masking_config = updated
            self.invalidate_from(2)

    def update_segmentation_config(self, **changes) -> None:
        updated = replace(self.state.segmentation_config, **changes)
        if updated != self.state.segmentation_config:
            self.state.segmentation_config = updated
            self.invalidate_from(3)

    def invalidate_from(self, stage: int) -> None:
        self.state.invalidate_from(stage)
        if stage <= 1:
            self._processed_work = None
        if stage <= 2:
            self._mask_work = None
        if stage <= 3:
            self._labels_work = self._markers_work = None
        if stage <= 4:
            self._cells_work = None
        if stage <= 5:
            self._features_work = None

    def reload_baseline(self, frame: int) -> None:
        self.baseline = self.source.load_frame(frame)
        self.state.frame = int(frame)
        self.invalidate_from(1)

    def _configure_work_volume(
        self,
        mode: ReplayMode,
        halo_zyx: tuple[int, int, int] | None,
    ) -> None:
        if self.baseline is None:
            raise RuntimeError("load a production frame before replay")
        shape = self.baseline.raw.shape
        if mode is ReplayMode.EXACT:
            origin = (0, 0, 0)
            stop = shape
        else:
            if halo_zyx is None:
                sigma = max(
                    self.state.preprocessing_config.denoise_sigma_um,
                    self.state.preprocessing_config.background_sigma_um,
                )
                halo_zyx = tuple(
                    int(np.ceil(3.0 * sigma / spacing))
                    for spacing in self.state.preprocessing_config.voxel_size_zyx_um
                )
            if len(halo_zyx) != 3 or any(value < 0 for value in halo_zyx):
                raise ValueError("halo_zyx must contain three non-negative integers")
            origin = tuple(max(0, value - halo) for value, halo in zip(self.source.crop_origin_zyx, halo_zyx))
            stop = tuple(min(limit, value + halo) for value, halo, limit in zip(self.source.crop_stop_zyx, halo_zyx, shape))
        work_crop = tuple(slice(start, end) for start, end in zip(origin, stop))
        previous_crop = self.state.work_crop_global
        display = tuple(
            slice(scene_start - work_start, scene_end - work_start)
            for scene_start, scene_end, work_start in zip(
                self.source.crop_origin_zyx, self.source.crop_stop_zyx, origin
            )
        )
        self._raw_work = self.baseline.raw[work_crop]
        self._display_slices = display  # type: ignore[assignment]
        self.state.work_origin_zyx = tuple(int(value) for value in origin)
        if self.state.mode != mode or previous_crop != work_crop:
            self.invalidate_from(1)
        self.state.work_crop_global = work_crop  # type: ignore[assignment]
        self.state.mode = mode

    def run(
        self,
        stage: int = 1,
        *,
        downstream: bool = True,
        mode: ReplayMode | str = ReplayMode.EXACT,
        halo_zyx: tuple[int, int, int] | None = None,
    ) -> ReplayResult:
        """Run a selected stage, optionally through Stage 5, honoring cache dirtiness."""

        if not 1 <= stage <= 5:
            raise ValueError("stage must be between 1 and 5")
        mode = ReplayMode(mode)
        self._configure_work_volume(mode, halo_zyx)
        target = 5 if downstream else stage
        first = min(self.state.dirty_from_stage, stage)
        for current in range(first, target + 1):
            getattr(self, f"_run_stage_{current}")()
            self.state.dirty_from_stage = current + 1
        return self.result()

    def _run_stage_1(self) -> None:
        self._processed_work, self.state.preprocessing_trace = preprocess_volume(
            self._raw_work,
            config=self.state.preprocessing_config,
            return_diagnostics=True,
        )

    def _run_stage_2(self) -> None:
        if self._processed_work is None:
            raise RuntimeError("Stage 2 requires Stage 1")
        self._mask_work, self.state.masking_trace = create_binary_mask(
            self._processed_work,
            config=self.state.masking_config,
            return_diagnostics=True,
        )

    def _run_stage_3(self) -> None:
        if self._mask_work is None or self._display_slices is None:
            raise RuntimeError("Stage 3 requires Stage 2")
        components, _ = ndimage.label(
            self._mask_work,
            structure=ndimage.generate_binary_structure(3, 1),
        )
        relevant = tuple(
            int(value)
            for value in np.unique(components[self._display_slices])
            if value > 0
        )
        if not relevant:
            raise RuntimeError("no binary connected component intersects the scene ROI")
        complete_relevant_mask = np.isin(components, relevant)
        detailed = segmentation_module.segment_instances_detailed(
            complete_relevant_mask,
            self.state.segmentation_config,
            retain_debug_artifacts=True,
        )
        self._labels_work = detailed.final_labels
        self._markers_work = detailed.markers
        self.state.trial_labels = self._labels_work
        self.state.trial_markers = self._markers_work
        self.state.connected_components = components
        self.state.relevant_component_ids = relevant
        padding = self.state.segmentation_config.component_padding_voxels
        diagnostics_by_component = {
            item.component_id: item for item in detailed.component_diagnostics
        }
        self.state.component_results = tuple(
            component_debug_result(
                artifact,
                diagnostics_by_component[artifact.component_id],
                padding,
            )
            for artifact in detailed.component_debug_artifacts
        )

    @staticmethod
    def _globalize_table(table, origin):
        table = table.copy()
        for axis, offset in zip("zyx", origin):
            for prefix in ("centroid_",):
                column = f"{prefix}{axis}"
                if column in table.columns:
                    table[column] = table[column] + offset
            for suffix in ("min", "max"):
                column = f"{axis}_{suffix}"
                if column in table.columns:
                    table[column] = table[column] + offset
        return table

    def _run_stage_4(self) -> None:
        if self._labels_work is None:
            raise RuntimeError("Stage 4 requires Stage 3")
        self._cells_work = detect_cells(self._labels_work)
        self.state.trial_cells = self._globalize_table(
            self._cells_work, self.state.work_origin_zyx
        )

    def _run_stage_5(self) -> None:
        if self._cells_work is None or self._labels_work is None or self._processed_work is None:
            raise RuntimeError("Stage 5 requires Stages 1, 3, and 4")
        self._features_work = extract_cell_features(
            self._cells_work, self._labels_work, self._processed_work
        )
        self.state.trial_features = self._globalize_table(
            self._features_work, self.state.work_origin_zyx
        )

    def result(self) -> ReplayResult:
        if self.baseline is None or self._raw_work is None or self._display_slices is None:
            raise RuntimeError("no replay frame is loaded")
        crop = self._display_slices
        display = lambda value: None if value is None else value[crop]
        warning = (
            "Local approximation: percentile normalization, Otsu thresholding, and "
            "components clipped by the computational halo may differ from production."
            if self.state.mode is ReplayMode.LOCAL else None
        )
        return ReplayResult(
            self.baseline.frame,
            self.state.mode,
            display(self._raw_work),
            display(self._processed_work),
            display(self._mask_work),
            display(self.state.connected_components),
            display(self._labels_work),
            display(self._markers_work),
            self.state.trial_cells,
            self.state.trial_features,
            self.state.component_results,
            self.state.work_origin_zyx,
            self.source.crop_origin_zyx,
            warning,
        )
