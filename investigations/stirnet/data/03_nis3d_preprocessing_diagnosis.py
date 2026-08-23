"""
03 - NIS3D preprocessing diagnosis.

Run from project root:
    python investigations/stirnet/data/03_nis3d_preprocessing_diagnosis.py

Default sample: Zebrafish_2

Useful options:
    --sample Zebrafish_1
    --full
    --no-viewer

The script uses the CURRENT repository preprocessing primitives and compares:
- current preprocessing
- no-background ablations
- denoise sigma sweep
- background sigma sweep
- normalization percentile sweep
- ordinary Otsu
- scaled Otsu
- a GT-informed oracle threshold (diagnostic only)

Outputs are written under:
    runs/stirnet/investigations/03_nis3d_preprocessing_diagnosis/<sample>/
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import tifffile


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NIS3D_ROOT = PROJECT_ROOT / "data" / "external" / "NIS3D" / "NIS3D"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "stirnet"
    / "investigations"
    / "03_nis3d_preprocessing_diagnosis"
)

SAMPLES = (
    "Drosophila_1",
    "Drosophila_2",
    "MusMusculus_1",
    "MusMusculus_2",
    "Zebrafish_1",
    "Zebrafish_2",
)

RAW_NAMES = ("data.tif", "Data.tif")
GT_NAMES = ("GroundTruth.tif", "groundtruth.tif", "gt.tif", "GT.tif")
CONF_NAMES = (
    "ConfidenceScore.tif",
    "confidencescore.tif",
    "scoreOfConfidence.tif",
    "ScoreOfConfidence.tif",
)
INFO_NAMES = ("Info.txt", "info.txt")


# -----------------------------------------------------------------------------
# CURRENT repository implementation
# -----------------------------------------------------------------------------

from src.api import preprocess_volume, create_binary_mask

_pre_cfg = import_module("src.01_preprocessing.config")
_norm = import_module("src.01_preprocessing.normalize")
_denoise = import_module("src.01_preprocessing.denoise")
_bg = import_module("src.01_preprocessing.background")
_thr = import_module("src.02_masking.threshold")

PreprocessingConfig = _pre_cfg.PreprocessingConfig
DEFAULT_PREPROCESSING_CONFIG = _pre_cfg.DEFAULT_PREPROCESSING_CONFIG
robust_normalize_with_percentiles = _norm.robust_normalize_with_percentiles
gaussian_denoise = _denoise.gaussian_denoise
background_correction = _bg.background_correction
compute_otsu_threshold = _thr.compute_otsu_threshold


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


def find_file(directory: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p

    lower = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    for name in candidates:
        p = lower.get(name.lower())
        if p is not None:
            return p

    raise FileNotFoundError(
        f"Could not find any of {candidates} in {directory}. "
        f"Available: {[p.name for p in directory.iterdir() if p.is_file()]}"
    )


def read_info(sample_dir: Path) -> str:
    try:
        return find_file(sample_dir, INFO_NAMES).read_text(
            encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        return ""


def parse_spacing(info: str) -> tuple[float, float, float] | None:
    """Parse NIS3D textual X x Y x Z spacing and return Z,Y,X."""
    text = (
        info.lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("×", "x")
    )
    n = r"([0-9]+(?:\.[0-9]+)?)"
    pattern = (
        n + r"\s*(?:um|micrometer(?:s)?)?\s*x\s*"
        + n + r"\s*(?:um|micrometer(?:s)?)?\s*x\s*"
        + n + r"\s*(?:um|micrometer(?:s)?)?"
    )
    m = re.search(pattern, text)
    if m is None:
        return None
    x, y, z = map(float, m.groups())
    return z, y, x


# -----------------------------------------------------------------------------
# Configuration / metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RepSpec:
    name: str
    family: str
    low: float
    high: float
    denoise_um: float
    bg_um: float | None


@dataclass
class Metrics:
    threshold: float
    precision: float
    recall: float
    dice: float
    iou: float
    foreground_fraction: float


@dataclass
class Row:
    name: str
    family: str
    low: float
    high: float
    denoise_um: float
    bg_um: float | None
    otsu_threshold: float
    otsu_precision: float
    otsu_recall: float
    otsu_dice: float
    otsu_iou: float
    otsu_fg_fraction: float
    scaled_multiplier: float
    scaled_threshold: float
    scaled_precision: float
    scaled_recall: float
    scaled_dice: float
    scaled_iou: float
    oracle_threshold: float
    oracle_precision: float
    oracle_recall: float
    oracle_dice: float
    oracle_iou: float
    oracle_gain: float


class Evaluator:
    def __init__(self, gt: np.ndarray, confidence: np.ndarray):
        self.gt = gt
        self.confidence = confidence
        self.valid = confidence != 1
        self.truth = gt > 0

        self.valid_flat = self.valid.ravel()
        self.truth_valid = self.truth.ravel()[self.valid_flat]
        self.gt_valid = gt.ravel()[self.valid_flat]

        self.pos = int(np.count_nonzero(self.truth_valid))
        self.neg = int(self.truth_valid.size - self.pos)
        self.max_label = int(np.max(gt))
        self.total_per_label = np.bincount(
            self.gt_valid, minlength=self.max_label + 1
        )

    @staticmethod
    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    def threshold_metrics(self, image: np.ndarray, threshold: float) -> Metrics:
        values = image.ravel()[self.valid_flat]
        pred = values > threshold
        truth = self.truth_valid

        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = self.pos - tp

        return Metrics(
            threshold=float(threshold),
            precision=self.div(tp, tp + fp),
            recall=self.div(tp, tp + fn),
            dice=self.div(2 * tp, 2 * tp + fp + fn),
            iou=self.div(tp, tp + fp + fn),
            foreground_fraction=float(np.mean(pred)),
        )

    def oracle_threshold(self, image: np.ndarray, bins: int) -> Metrics:
        """GT-informed histogram threshold search. Diagnostic only."""
        values = image.ravel()[self.valid_flat].astype(np.float32, copy=False)
        truth = self.truth_valid

        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            return self.threshold_metrics(image, lo)

        edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
        pos_hist, _ = np.histogram(values[truth], bins=edges)
        neg_hist, _ = np.histogram(values[~truth], bins=edges)

        tp = np.cumsum(pos_hist[::-1], dtype=np.int64)[::-1]
        fp = np.cumsum(neg_hist[::-1], dtype=np.int64)[::-1]
        fn = self.pos - tp
        denom = 2 * tp + fp + fn
        dice = np.divide(
            2 * tp,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )

        idx = int(np.argmax(dice))
        return self.threshold_metrics(image, float(edges[idx]))

    def cell_coverage(self, mask: np.ndarray) -> dict[str, float]:
        pred_valid = mask.ravel()[self.valid_flat].astype(np.uint8, copy=False)
        covered = np.bincount(
            self.gt_valid,
            weights=pred_valid,
            minlength=self.max_label + 1,
        )[1:]
        total = self.total_per_label[1:]
        exists = total > 0
        coverage = covered[exists] / total[exists]

        return {
            "n_cells": int(coverage.size),
            "median_coverage": float(np.median(coverage)),
            "cells_ge_95": float(np.mean(coverage >= 0.95)),
            "cells_ge_80": float(np.mean(coverage >= 0.80)),
            "cells_ge_50": float(np.mean(coverage >= 0.50)),
            "cells_zero": float(np.mean(coverage == 0.0)),
        }


# -----------------------------------------------------------------------------
# Sweep construction
# -----------------------------------------------------------------------------


def current_spec() -> RepSpec:
    c = DEFAULT_PREPROCESSING_CONFIG
    return RepSpec(
        "CURRENT",
        "current",
        float(c.low_percentile),
        float(c.high_percentile),
        float(c.denoise_sigma_um),
        float(c.background_sigma_um),
    )


def build_specs(full: bool) -> list[RepSpec]:
    c = DEFAULT_PREPROCESSING_CONFIG
    low = float(c.low_percentile)
    high = float(c.high_percentile)
    den = float(c.denoise_sigma_um)

    specs = [
        current_spec(),
        # Remove background and vary denoising.
        RepSpec("NORM_ONLY", "denoise_ablation", low, high, 0.0, None),
        RepSpec("NO_BG_DENOISE_0.4", "denoise_ablation", low, high, 0.4, None),
        RepSpec("NO_BG_DENOISE_0.8", "denoise_ablation", low, high, 0.8, None),
        RepSpec("NO_BG_DENOISE_1.2", "denoise_ablation", low, high, 1.2, None),
        # Keep current denoise, vary background scale.
        RepSpec("BG_2", "background_sweep", low, high, den, 2.0),
        RepSpec("BG_6", "background_sweep", low, high, den, 6.0),
        RepSpec("BG_8", "background_sweep", low, high, den, 8.0),
        RepSpec("BG_12", "background_sweep", low, high, den, 12.0),
        RepSpec("BG_16", "background_sweep", low, high, den, 16.0),
        # Vary clipping, without background, to isolate normalization itself.
        RepSpec("NORM_0_99.9", "normalization_sweep", 0.0, 99.9, den, None),
        RepSpec("NORM_0.1_99.9", "normalization_sweep", 0.1, 99.9, den, None),
        RepSpec("NORM_0.5_99.5", "normalization_sweep", 0.5, 99.5, den, None),
        RepSpec("NORM_1_99.9", "normalization_sweep", 1.0, 99.9, den, None),
    ]

    if full:
        specs += [
            RepSpec("NO_BG_DENOISE_1.6", "denoise_ablation", low, high, 1.6, None),
            RepSpec("BG_20", "background_sweep", low, high, den, 20.0),
            RepSpec("BG_24", "background_sweep", low, high, den, 24.0),
            RepSpec("NORM_0_99.99", "normalization_sweep", 0.0, 99.99, den, None),
            RepSpec("NORM_0.1_99.5", "normalization_sweep", 0.1, 99.5, den, None),
        ]

    # Deduplicate parameter-equivalent configurations.
    out = []
    seen = set()
    for s in specs:
        key = (s.low, s.high, s.denoise_um, s.bg_um)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def make_rep(raw: np.ndarray, spacing: tuple[float, float, float], spec: RepSpec):
    image, p_lo, p_hi = robust_normalize_with_percentiles(
        raw, spec.low, spec.high
    )

    if spec.denoise_um > 0:
        image = gaussian_denoise(
            image,
            voxel_size=spacing,
            sigma_um=spec.denoise_um,
        )

    if spec.bg_um is not None:
        image = background_correction(
            image,
            voxel_size=spacing,
            sigma_um=spec.bg_um,
        )

    return np.asarray(image, dtype=np.float32), float(p_lo), float(p_hi)


def evaluate_rep(
    evaluator: Evaluator,
    image: np.ndarray,
    spec: RepSpec,
    oracle_bins: int,
) -> Row:
    otsu = float(compute_otsu_threshold(image))
    om = evaluator.threshold_metrics(image, otsu)

    multipliers = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.10)
    best_x = None
    best_sm = None
    for x in multipliers:
        m = evaluator.threshold_metrics(image, otsu * x)
        if best_sm is None or m.dice > best_sm.dice:
            best_x = x
            best_sm = m

    assert best_x is not None and best_sm is not None
    oracle = evaluator.oracle_threshold(image, oracle_bins)

    return Row(
        name=spec.name,
        family=spec.family,
        low=spec.low,
        high=spec.high,
        denoise_um=spec.denoise_um,
        bg_um=spec.bg_um,
        otsu_threshold=otsu,
        otsu_precision=om.precision,
        otsu_recall=om.recall,
        otsu_dice=om.dice,
        otsu_iou=om.iou,
        otsu_fg_fraction=om.foreground_fraction,
        scaled_multiplier=float(best_x),
        scaled_threshold=best_sm.threshold,
        scaled_precision=best_sm.precision,
        scaled_recall=best_sm.recall,
        scaled_dice=best_sm.dice,
        scaled_iou=best_sm.iou,
        oracle_threshold=oracle.threshold,
        oracle_precision=oracle.precision,
        oracle_recall=oracle.recall,
        oracle_dice=oracle.dice,
        oracle_iou=oracle.iou,
        oracle_gain=oracle.dice - om.dice,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def row_by_name(rows: list[Row], name: str) -> Row:
    return next(r for r in rows if r.name == name)


def print_table(rows: list[Row]) -> None:
    print("\n" + "=" * 142)
    print("CONFIGURATION COMPARISON")
    print("=" * 142)
    print(
        f"{'name':<22} {'family':<20} "
        f"{'O-P':>7} {'O-R':>7} {'O-D':>7} "
        f"{'x':>5} {'S-R':>7} {'S-D':>7} "
        f"{'Q-R':>7} {'Q-D':>7} {'Qgain':>8}"
    )
    print("-" * 142)
    for r in rows:
        print(
            f"{r.name:<22} {r.family:<20} "
            f"{r.otsu_precision:7.4f} {r.otsu_recall:7.4f} {r.otsu_dice:7.4f} "
            f"{r.scaled_multiplier:5.2f} {r.scaled_recall:7.4f} {r.scaled_dice:7.4f} "
            f"{r.oracle_recall:7.4f} {r.oracle_dice:7.4f} {r.oracle_gain:+8.4f}"
        )


def print_cell_metrics(title: str, evaluator: Evaluator, mask: np.ndarray):
    m = evaluator.cell_coverage(mask)
    print(f"\n{title}")
    print("-" * 80)
    print(f"Evaluated nuclei       : {m['n_cells']:,}")
    print(f"Median voxel coverage  : {m['median_coverage']:.2%}")
    print(f"Cells >=95% retained   : {m['cells_ge_95']:.2%}")
    print(f"Cells >=80% retained   : {m['cells_ge_80']:.2%}")
    print(f"Cells >=50% retained   : {m['cells_ge_50']:.2%}")
    print(f"Completely missed cells: {m['cells_zero']:.2%}")
    return m


def diagnose(rows: list[Row]) -> list[str]:
    current = row_by_name(rows, "CURRENT")
    no_bg = [
        r for r in rows
        if r.bg_um is None
        and r.low == DEFAULT_PREPROCESSING_CONFIG.low_percentile
        and r.high == DEFAULT_PREPROCESSING_CONFIG.high_percentile
    ]
    bg_rows = [r for r in rows if r.family in {"current", "background_sweep"}]
    norm_rows = [r for r in rows if r.family == "normalization_sweep"]
    den_rows = [r for r in rows if r.family == "denoise_ablation"]

    best_no_bg = max(no_bg, key=lambda r: r.otsu_dice)
    best_bg = max(bg_rows, key=lambda r: r.otsu_dice)
    best_norm = max(norm_rows, key=lambda r: r.otsu_dice)
    best_den = max(den_rows, key=lambda r: r.otsu_dice)
    no_bg_ref = row_by_name(rows, "NO_BG_DENOISE_0.8")

    out = []

    threshold_headroom = current.oracle_dice - current.otsu_dice
    if threshold_headroom >= 0.15:
        out.append(
            f"Threshold choice is a MAJOR contributor: CURRENT has "
            f"{threshold_headroom:+.3f} oracle Dice headroom."
        )
    elif threshold_headroom >= 0.07:
        out.append(
            f"Threshold choice is a meaningful contributor: CURRENT has "
            f"{threshold_headroom:+.3f} oracle Dice headroom."
        )
    else:
        out.append(
            f"Threshold choice alone is not the main failure: CURRENT oracle "
            f"headroom is only {threshold_headroom:+.3f}."
        )

    bg_gain = best_no_bg.otsu_dice - current.otsu_dice
    if bg_gain >= 0.10:
        out.append(
            f"Background subtraction is strongly implicated: {best_no_bg.name} "
            f"improves ordinary-Otsu Dice by {bg_gain:+.3f}."
        )
    elif bg_gain >= 0.04:
        out.append(
            f"Background subtraction contributes materially: removing it gains "
            f"{bg_gain:+.3f} Dice."
        )
    else:
        out.append(
            f"Removing background subtraction alone changes Otsu Dice by only "
            f"{bg_gain:+.3f}."
        )

    if best_bg.name != "CURRENT":
        out.append(
            f"Best tested background scale is {best_bg.bg_um:g} um "
            f"({best_bg.name}), Otsu Dice={best_bg.otsu_dice:.4f}."
        )

    den_gain = best_den.otsu_dice - no_bg_ref.otsu_dice
    out.append(
        f"Best no-background denoise variant is {best_den.name}; "
        f"change vs 0.8 um is {den_gain:+.3f} Dice."
    )

    norm_gain = best_norm.otsu_dice - no_bg_ref.otsu_dice
    out.append(
        f"Best normalization variant is {best_norm.name}; "
        f"change vs current clipping (no background) is {norm_gain:+.3f} Dice."
    )

    best_o = max(rows, key=lambda r: r.otsu_dice)
    best_s = max(rows, key=lambda r: r.scaled_dice)
    best_q = max(rows, key=lambda r: r.oracle_dice)
    out.append(
        f"Best ordinary Otsu: {best_o.name}, Dice={best_o.otsu_dice:.4f}, "
        f"Recall={best_o.otsu_recall:.4f}."
    )
    out.append(
        f"Best scaled Otsu: {best_s.name} x{best_s.scaled_multiplier:.2f}, "
        f"Dice={best_s.scaled_dice:.4f}, Recall={best_s.scaled_recall:.4f}."
    )
    out.append(
        f"Best oracle representation: {best_q.name}, Dice={best_q.oracle_dice:.4f}, "
        f"Recall={best_q.oracle_recall:.4f}."
    )
    return out


def write_rows(rows: list[Row], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Row.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


# -----------------------------------------------------------------------------
# Finalist recomputation / Napari
# -----------------------------------------------------------------------------


def spec_by_name(specs: list[RepSpec], name: str) -> RepSpec:
    return next(s for s in specs if s.name == name)


def finalist(raw, spacing, specs, row: Row, kind: str):
    image, _, _ = make_rep(raw, spacing, spec_by_name(specs, row.name))
    if kind == "otsu":
        t = row.otsu_threshold
    elif kind == "scaled":
        t = row.scaled_threshold
    elif kind == "oracle":
        t = row.oracle_threshold
    else:
        raise ValueError(kind)
    return image, image > t, float(t)


def error_map(evaluator: Evaluator, mask: np.ndarray) -> np.ndarray:
    pred = mask.astype(bool, copy=False)
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[pred & evaluator.truth & evaluator.valid] = 1
    out[(~pred) & evaluator.truth & evaluator.valid] = 2
    out[pred & (~evaluator.truth) & evaluator.valid] = 3
    return out


def raw_limits(raw: np.ndarray):
    s = np.asarray(
        raw[
            ::max(1, raw.shape[0] // 32),
            ::max(1, raw.shape[1] // 256),
            ::max(1, raw.shape[2] // 256),
        ]
    )
    lo, hi = np.percentile(s, (0.5, 99.8))
    return float(lo), float(hi)


def open_viewer(
    raw,
    gt,
    confidence,
    spacing,
    evaluator,
    current_image,
    current_mask,
    best_o,
    best_o_image,
    best_o_mask,
    best_s,
    best_s_image,
    best_s_mask,
    best_q,
    best_q_image,
    best_q_mask,
    sample,
):
    import napari

    v = napari.Viewer(title=f"NIS3D preprocessing diagnosis — {sample}", ndisplay=3)

    v.add_image(raw, name="00 | RAW", scale=spacing, colormap="gray",
                contrast_limits=raw_limits(raw), visible=False)
    v.add_labels(gt, name="01 | GT INSTANCES", scale=spacing,
                 opacity=0.45, visible=False)
    v.add_image(confidence, name="02 | GT CONFIDENCE", scale=spacing,
                colormap="turbo", contrast_limits=(0, 4), opacity=0.6,
                visible=False)

    def add_group(prefix, title, image, mask, visible=False):
        v.add_image(image, name=f"{prefix}0 | {title} image", scale=spacing,
                    colormap="gray", contrast_limits=(0, 1), visible=visible)
        v.add_labels(mask.astype(np.uint8), name=f"{prefix}1 | {title} mask",
                     scale=spacing, opacity=0.40, visible=visible)
        v.add_labels(
            error_map(evaluator, mask),
            name=f"{prefix}2 | {title} error [1=TP 2=MISS 3=FALSE+]",
            scale=spacing,
            opacity=0.70,
            visible=False,
        )

    add_group("1", "CURRENT", current_image, current_mask, True)
    add_group("2", f"BEST OTSU {best_o.name}", best_o_image, best_o_mask)
    add_group(
        "3",
        f"BEST SCALED {best_s.name} x{best_s.scaled_multiplier:.2f}",
        best_s_image,
        best_s_mask,
    )
    add_group("4", f"BEST ORACLE {best_q.name}", best_q_image, best_q_mask)

    v.reset_view()
    napari.run()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", choices=SAMPLES, default="Zebrafish_2")
    p.add_argument("--full", action="store_true")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--oracle-bins", type=int, default=384)
    return p.parse_args()


def main():
    args = parse_args()
    if args.oracle_bins < 32:
        raise ValueError("--oracle-bins must be >= 32")

    sample_dir = NIS3D_ROOT / args.sample
    raw_path = find_file(sample_dir, RAW_NAMES)
    gt_path = find_file(sample_dir, GT_NAMES)
    conf_path = find_file(sample_dir, CONF_NAMES)

    spacing = parse_spacing(read_info(sample_dir))
    if spacing is None:
        raise RuntimeError("Could not parse NIS3D spacing from Info.txt")

    print("\n" + "=" * 100)
    print(f"03 - NIS3D PREPROCESSING DIAGNOSIS: {args.sample}")
    print("=" * 100)
    print(f"Spacing Z,Y,X : {spacing} um")
    print(f"Full sweep    : {args.full}")

    print("\nLoading TIFFs...")
    raw = tifffile.imread(raw_path)
    gt = tifffile.imread(gt_path)
    confidence = tifffile.imread(conf_path)

    if raw.shape != gt.shape or raw.shape != confidence.shape:
        raise RuntimeError(
            f"Shape mismatch: raw={raw.shape}, gt={gt.shape}, conf={confidence.shape}"
        )

    print(f"Shape          : {raw.shape}")
    print(f"GT max ID      : {int(np.max(gt)):,}")

    evaluator = Evaluator(gt, confidence)
    print(f"GT foreground  : {evaluator.pos / evaluator.truth_valid.size:.3%}")
    print(
        f"Evaluated cells: "
        f"{int(np.count_nonzero(evaluator.total_per_label[1:] > 0)):,}"
    )

    # Exact current API baseline on NIS3D's physical spacing.
    c = DEFAULT_PREPROCESSING_CONFIG
    cfg = PreprocessingConfig(
        low_percentile=c.low_percentile,
        high_percentile=c.high_percentile,
        denoise_sigma_um=c.denoise_sigma_um,
        background_sigma_um=c.background_sigma_um,
        voxel_size_zyx_um=spacing,
    )

    print("\n" + "=" * 100)
    print("CURRENT src.api BASELINE")
    print("=" * 100)
    api_current, _ = preprocess_volume(raw, config=cfg, return_diagnostics=True)
    api_mask, mask_trace = create_binary_mask(api_current, return_diagnostics=True)
    api_t = float(mask_trace.metrics["effective_threshold"])
    api_m = evaluator.threshold_metrics(api_current, api_t)
    print(f"Otsu      : {api_t:.6f}")
    print(f"Precision : {api_m.precision:.4f}")
    print(f"Recall    : {api_m.recall:.4f}")
    print(f"Dice      : {api_m.dice:.4f}")
    current_cell = print_cell_metrics("CURRENT cell coverage", evaluator, api_mask)

    specs = build_specs(args.full)
    rows = []

    print("\n" + "=" * 100)
    print(f"SWEEPING {len(specs)} REPRESENTATIONS")
    print("=" * 100)

    for i, spec in enumerate(specs, 1):
        print(
            f"\n[{i:02d}/{len(specs):02d}] {spec.name} | "
            f"norm={spec.low:g}-{spec.high:g}, "
            f"denoise={spec.denoise_um:g}um, "
            f"bg={'none' if spec.bg_um is None else f'{spec.bg_um:g}um'}"
        )

        image, p_lo, p_hi = make_rep(raw, spacing, spec)
        row = evaluate_rep(evaluator, image, spec, args.oracle_bins)
        rows.append(row)

        print(f"  raw clipping : {p_lo:.6g} .. {p_hi:.6g}")
        print(
            f"  Otsu         : P={row.otsu_precision:.4f} "
            f"R={row.otsu_recall:.4f} D={row.otsu_dice:.4f}"
        )
        print(
            f"  scaled Otsu  : x{row.scaled_multiplier:.2f} "
            f"R={row.scaled_recall:.4f} D={row.scaled_dice:.4f}"
        )
        print(
            f"  oracle       : R={row.oracle_recall:.4f} "
            f"D={row.oracle_dice:.4f} gain={row.oracle_gain:+.4f}"
        )

        del image
        gc.collect()

    print_table(rows)

    out_dir = OUTPUT_ROOT / args.sample
    out_csv = out_dir / "configuration_summary.csv"
    write_rows(rows, out_csv)
    print(f"\nSaved configuration summary:\n    {out_csv}")

    print("\n" + "=" * 100)
    print("AUTOMATED DIAGNOSIS")
    print("=" * 100)
    for line in diagnose(rows):
        print(f"- {line}")

    best_o = max(rows, key=lambda r: r.otsu_dice)
    best_s = max(rows, key=lambda r: r.scaled_dice)
    best_q = max(rows, key=lambda r: r.oracle_dice)

    print("\n" + "=" * 100)
    print("FINALISTS")
    print("=" * 100)
    print(
        f"BEST OTSU   : {best_o.name} | "
        f"D={best_o.otsu_dice:.4f} R={best_o.otsu_recall:.4f}"
    )
    print(
        f"BEST SCALED : {best_s.name} x{best_s.scaled_multiplier:.2f} | "
        f"D={best_s.scaled_dice:.4f} R={best_s.scaled_recall:.4f}"
    )
    print(
        f"BEST ORACLE : {best_q.name} | "
        f"D={best_q.oracle_dice:.4f} R={best_q.oracle_recall:.4f}"
    )

    best_o_image, best_o_mask, _ = finalist(raw, spacing, specs, best_o, "otsu")
    best_s_image, best_s_mask, _ = finalist(raw, spacing, specs, best_s, "scaled")
    best_q_image, best_q_mask, _ = finalist(raw, spacing, specs, best_q, "oracle")

    print("\n" + "=" * 100)
    print("FINALIST PER-CELL COVERAGE")
    print("=" * 100)
    best_o_cell = print_cell_metrics(
        f"BEST OTSU: {best_o.name}", evaluator, best_o_mask
    )
    best_s_cell = print_cell_metrics(
        f"BEST SCALED: {best_s.name} x{best_s.scaled_multiplier:.2f}",
        evaluator,
        best_s_mask,
    )
    best_q_cell = print_cell_metrics(
        f"BEST ORACLE: {best_q.name}", evaluator, best_q_mask
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    final_csv = out_dir / "finalists_cell_coverage.csv"
    with final_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "name",
            "n_cells",
            "median_coverage",
            "cells_ge_95",
            "cells_ge_80",
            "cells_ge_50",
            "cells_zero",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, m in [
            ("CURRENT", current_cell),
            (f"BEST_OTSU:{best_o.name}", best_o_cell),
            (f"BEST_SCALED:{best_s.name}:x{best_s.scaled_multiplier:.2f}", best_s_cell),
            (f"BEST_ORACLE:{best_q.name}", best_q_cell),
        ]:
            w.writerow({"name": name, **m})

    print(f"\nSaved finalist cell metrics:\n    {final_csv}")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print(
        "CURRENT poor + CURRENT oracle strong  -> thresholding is the main issue.\n"
        "CURRENT oracle poor + NO_BG strong    -> background subtraction destroys signal.\n"
        "larger BG sigma strong                -> 4 um background scale is too small.\n"
        "normalization variants similar        -> clipping is not the main issue.\n"
        "all oracle variants poor              -> global-intensity preprocessing itself is insufficient."
    )

    if args.no_viewer:
        return

    print("\nOpening Napari with CURRENT and finalists...")
    open_viewer(
        raw,
        gt,
        confidence,
        spacing,
        evaluator,
        api_current,
        api_mask,
        best_o,
        best_o_image,
        best_o_mask,
        best_s,
        best_s_image,
        best_s_mask,
        best_q,
        best_q_image,
        best_q_mask,
        args.sample,
    )


if __name__ == "__main__":
    main()
