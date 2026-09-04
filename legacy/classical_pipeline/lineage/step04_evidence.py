"""Lazy segmentation and raw-fluorescence evidence for Stage 10."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .step01_config import CellLineageConfig
from .step02_observations import as_bool
from .step05_scoring import safe_ratio


@dataclass(frozen=True)
class CellRawFeatures:
    voxel_count: int
    raw_mask_sum: float
    raw_mask_mean: float
    raw_mask_median: float
    raw_core_mean: float
    raw_core_median: float
    raw_background_median: float
    raw_background_corrected_mean: float
    raw_background_corrected_sum: float
    core_fallback_to_full_mask: bool
    background_shell_available: bool
    frame_raw_foreground_median: float
    raw_mask_mean_frame_ratio: float
    raw_core_mean_frame_ratio: float
    raw_background_corrected_mean_frame_ratio: float


class RawEvidenceExtractor:
    """Cache frames and per-cell measurements only when a candidate needs them."""

    def __init__(
        self,
        raw_volume,
        segmentation_files: Sequence[str | Path],
        config: CellLineageConfig,
    ) -> None:
        self.raw_volume = raw_volume
        self.segmentation_files = tuple(Path(path) for path in segmentation_files)
        self.config = config
        self.raw_frame_cache: dict[int, np.ndarray] = {}
        self.segmentation_cache: dict[int, np.ndarray] = {}
        self.cell_raw_feature_cache: dict[tuple[int, int], CellRawFeatures] = {}
        self.frame_reference_cache: dict[int, float] = {}
        self.warnings: list[str] = []

    def load_segmentation(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if not 0 <= frame < len(self.segmentation_files):
            raise IndexError(f"Segmentation frame {frame} is outside the available sequence")
        if frame not in self.segmentation_cache:
            path = self.segmentation_files[frame]
            if not path.is_file():
                raise FileNotFoundError(f"Expected segmentation does not exist: {path}")
            labels = np.load(path, mmap_mode="r", allow_pickle=False)
            if labels.ndim != 3:
                raise ValueError(f"Segmentation frame {frame} must be 3-D, found {labels.shape}")
            self.segmentation_cache[frame] = labels
        return self.segmentation_cache[frame]

    def load_raw(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if frame not in self.raw_frame_cache:
            raw = np.asarray(self.raw_volume[frame])
            if raw.ndim != 3:
                raise ValueError(f"Raw frame {frame} must be 3-D, found {raw.shape}")
            labels = self.load_segmentation(frame)
            if raw.shape != labels.shape:
                raise ValueError(
                    f"Raw and segmentation shapes differ at frame {frame}: "
                    f"{raw.shape} vs {labels.shape}"
                )
            self.raw_frame_cache[frame] = raw
        return self.raw_frame_cache[frame]

    def cell_voxel_count(self, frame: int, cell_id: int) -> int:
        labels = self.load_segmentation(frame)
        count = int(np.count_nonzero(labels == int(cell_id)))
        if count == 0:
            raise ValueError(
                f"Instance mask is missing for frame={int(frame)}, cell_id={int(cell_id)}"
            )
        return count

    def _frame_reference(self, frame: int) -> float:
        if frame not in self.frame_reference_cache:
            labels = self.load_segmentation(frame)
            raw = self.load_raw(frame).astype(np.float64, copy=False)
            foreground = labels > 0
            value = (
                float(np.median(raw[foreground]))
                if np.any(foreground)
                else math.nan
            )
            self.frame_reference_cache[frame] = value
        return self.frame_reference_cache[frame]

    def cell_features(self, frame: int, cell_id: int) -> CellRawFeatures:
        key = (int(frame), int(cell_id))
        if key in self.cell_raw_feature_cache:
            return self.cell_raw_feature_cache[key]
        labels = self.load_segmentation(frame)
        raw = self.load_raw(frame).astype(np.float64, copy=False)
        target_full = labels == int(cell_id)
        coordinates = np.argwhere(target_full)
        if coordinates.size == 0:
            raise ValueError(
                f"Instance mask is missing for frame={int(frame)}, cell_id={int(cell_id)}"
            )
        spacing = np.asarray(self.config.voxel_size_zyx_um, dtype=float)
        margin = (
            np.ceil(self.config.background_shell_outer_um / spacing).astype(int) + 1
        )
        start = np.maximum(coordinates.min(axis=0) - margin, 0)
        stop = np.minimum(coordinates.max(axis=0) + 1 + margin, labels.shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        target = np.asarray(target_full[slices], dtype=bool)
        label_crop = np.asarray(labels[slices])
        raw_crop = np.asarray(raw[slices], dtype=np.float64)

        distance_inside = ndimage.distance_transform_edt(target, sampling=spacing)
        core = distance_inside >= self.config.core_erosion_um
        core_fallback = not np.any(core)
        if core_fallback:
            core = target.copy()
        distance_outside = ndimage.distance_transform_edt(~target, sampling=spacing)
        shell = (
            (distance_outside >= self.config.background_shell_inner_um)
            & (distance_outside <= self.config.background_shell_outer_um)
            & (label_crop == 0)
        )
        target_values = raw_crop[target].astype(np.float64, copy=False)
        core_values = raw_crop[core].astype(np.float64, copy=False)
        shell_available = bool(np.any(shell))
        background = float(np.median(raw_crop[shell])) if shell_available else math.nan
        corrected_mean = (
            float(np.mean(target_values - background))
            if shell_available else math.nan
        )
        corrected_sum = (
            float(np.sum(target_values - background))
            if shell_available else math.nan
        )
        frame_reference = self._frame_reference(frame)
        mask_mean = float(np.mean(target_values))
        core_mean = float(np.mean(core_values))
        features = CellRawFeatures(
            voxel_count=int(target_values.size),
            raw_mask_sum=float(np.sum(target_values)),
            raw_mask_mean=mask_mean,
            raw_mask_median=float(np.median(target_values)),
            raw_core_mean=core_mean,
            raw_core_median=float(np.median(core_values)),
            raw_background_median=background,
            raw_background_corrected_mean=corrected_mean,
            raw_background_corrected_sum=corrected_sum,
            core_fallback_to_full_mask=core_fallback,
            background_shell_available=shell_available,
            frame_raw_foreground_median=frame_reference,
            raw_mask_mean_frame_ratio=safe_ratio(mask_mean, frame_reference),
            raw_core_mean_frame_ratio=safe_ratio(core_mean, frame_reference),
            raw_background_corrected_mean_frame_ratio=safe_ratio(
                corrected_mean, frame_reference
            ),
        )
        self.cell_raw_feature_cache[key] = features
        return features

    def optional_features(
        self,
        frame: int,
        cell_id: int,
        *,
        candidate_id: str,
    ) -> CellRawFeatures | None:
        """Convert candidate-level optional raw failures into explicit warnings."""

        try:
            return self.cell_features(frame, cell_id)
        except (IndexError, OSError, TypeError, ValueError) as error:
            self.warnings.append(
                f"{candidate_id}: raw evidence unavailable for frame={frame}, "
                f"cell_id={cell_id}: {error}"
            )
            return None


def _median_feature(
    features: Sequence[CellRawFeatures | None],
    name: str,
) -> float:
    values = [
        float(getattr(feature, name))
        for feature in features
        if feature is not None and math.isfinite(float(getattr(feature, name)))
    ]
    return float(np.median(values)) if values else math.nan


def _paired_child_ratio(
    first: CellRawFeatures | None,
    second: CellRawFeatures | None,
    baselines: dict[str, float],
) -> float:
    if first is None or second is None:
        return math.nan
    preference = (
        "raw_background_corrected_mean",
        "raw_core_mean_frame_ratio",
        "raw_core_mean",
    )
    for name in preference:
        baseline = baselines.get(name, math.nan)
        values = (float(getattr(first, name)), float(getattr(second, name)))
        if all(math.isfinite(value) for value in values) and math.isfinite(baseline):
            ratio = safe_ratio(float(np.mean(values)), baseline)
            if math.isfinite(ratio):
                return ratio
    return math.nan


def extract_candidate_intensity_evidence(
    record: dict[str, object],
    observations: pd.DataFrame,
    extractor: RawEvidenceExtractor,
    config: CellLineageConfig,
) -> dict[str, object]:
    """Measure secondary parent timing and delayed paired-child brightening."""

    result = dict(record)
    candidate_id = str(result["candidate_id"])
    parent_id = int(result["parent_track_id"])
    parent_end = int(result["parent_end_frame"])
    parent = observations.loc[
        (observations["track_id"] == parent_id)
        & (~observations["is_virtual_merge"].map(as_bool))
    ].sort_values("frame", kind="mergesort")
    baseline_rows = parent.loc[parent["frame"] < parent_end].tail(
        config.parent_intensity_baseline_history
    )
    baseline_features = [
        extractor.optional_features(
            int(row.frame), int(row.cell_id), candidate_id=candidate_id
        )
        for row in baseline_rows.itertuples(index=False)
    ]
    final_row = parent.loc[parent["frame"] == parent_end].iloc[-1]
    final_feature = extractor.optional_features(
        parent_end, int(final_row["cell_id"]), candidate_id=candidate_id
    )
    names = (
        "raw_mask_sum", "raw_mask_mean", "raw_core_mean",
        "raw_background_corrected_mean", "raw_core_mean_frame_ratio",
    )
    baselines = {name: _median_feature(baseline_features, name) for name in names}

    def final_ratio(name: str) -> float:
        return (
            safe_ratio(getattr(final_feature, name), baselines[name])
            if final_feature is not None else math.nan
        )

    result.update({
        "parent_final_integrated_intensity_ratio": final_ratio("raw_mask_sum"),
        "parent_final_mask_mean_ratio": final_ratio("raw_mask_mean"),
        "parent_final_core_intensity_ratio": final_ratio("raw_core_mean"),
        "parent_final_background_corrected_ratio": final_ratio(
            "raw_background_corrected_mean"
        ),
        "parent_final_core_frame_ratio_change": final_ratio(
            "raw_core_mean_frame_ratio"
        ),
    })

    child_a = int(result["child_track_a"])
    child_b = int(result["child_track_b"])
    birth = int(result["child_birth_frame"])
    child_rows = observations.loc[
        observations["track_id"].isin([child_a, child_b])
        & observations["frame"].between(birth, birth + config.future_child_horizon)
    ]
    lookup = {
        (int(row.track_id), int(row.frame)): row
        for row in child_rows.itertuples(index=False)
    }

    ratios: list[tuple[int, float]] = []
    birth_ratio = math.nan
    for relative_frame in range(0, config.future_child_horizon + 1):
        frame = birth + relative_frame
        row_a = lookup.get((child_a, frame))
        row_b = lookup.get((child_b, frame))
        if row_a is None or row_b is None:
            continue
        feature_a = extractor.optional_features(
            frame, int(row_a.cell_id), candidate_id=candidate_id
        )
        feature_b = extractor.optional_features(
            frame, int(row_b.cell_id), candidate_id=candidate_id
        )
        ratio = _paired_child_ratio(feature_a, feature_b, baselines)
        if relative_frame == 0:
            birth_ratio = ratio
        elif math.isfinite(ratio):
            ratios.append((relative_frame, ratio))

    if ratios:
        relative = np.asarray([item[0] for item in ratios], dtype=float)
        values = np.asarray([item[1] for item in ratios], dtype=float)
        maximum_index = int(np.argmax(values))
        delayed_max = float(values[maximum_index])
        peak_frame = int(relative[maximum_index])
        slope = (
            float(np.polyfit(relative, values, 1)[0])
            if len(values) >= 2 else math.nan
        )
    else:
        delayed_max = math.nan
        peak_frame = math.nan
        slope = math.nan
    result.update({
        "child_birth_intensity_ratio": birth_ratio,
        "child_delayed_max_intensity_ratio": delayed_max,
        "child_delayed_intensity_slope": slope,
        "child_intensity_peak_relative_frame": peak_frame,
        "child_delayed_paired_frame_count": int(len(ratios)),
    })
    return result
