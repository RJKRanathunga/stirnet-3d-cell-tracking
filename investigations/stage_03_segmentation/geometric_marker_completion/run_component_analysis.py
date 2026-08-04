"""Export one production Stage 3 geometric-completion analysis."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.tables import save_csv


pipeline_module = import_module("src.03_segmentation.pipeline")
completion_module = import_module("src.03_segmentation.marker_completion")
candidate_module = import_module("src.03_segmentation.candidate_detection")


def _peak_table(artifact) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component_id": artifact.component_id,
                "peak_id": peak.peak_id,
                "z": peak.position_zyx[0],
                "y": peak.position_zyx[1],
                "x": peak.position_zyx[2],
                "raw_depth_um": peak.raw_depth_um,
                "persistence_score": peak.persistence_score,
            }
            for peak in artifact.effective_peaks
        ]
    )


def run(mask_path: Path, output: Path, force: bool) -> None:
    mask = np.asarray(np.load(mask_path), dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("the input must be a non-empty 3-D NumPy mask")
    result = pipeline_module.segment_instances_detailed(
        mask,
        retain_debug_artifacts=True,
        force_geometric_analysis=force,
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "final_labels.npy", result.final_labels)
    np.save(output / "final_markers.npy", result.markers)

    summary_rows = []
    for artifact in result.component_debug_artifacts:
        completion = artifact.geometric_completion
        prefix = output / f"component_{artifact.component_id:04d}"
        save_csv(_peak_table(artifact), prefix / "effective_edt_peaks.csv")
        save_csv(
            candidate_module.shape_peaks_dataframe(
                artifact.candidate_result, artifact.component_id
            ),
            prefix / "shape_peaks.csv",
        )
        save_csv(
            candidate_module.center_proposals_dataframe(
                artifact.candidate_result, artifact.component_id
            ),
            prefix / "center_proposals.csv",
        )
        save_csv(
            candidate_module.candidate_summary_dataframe(
                artifact.candidate_result,
                artifact.component_id,
                len(artifact.raw_peaks),
                len(artifact.effective_peaks),
            ),
            prefix / "candidate_summary.csv",
        )
        caps = completion_module.surface_caps_dataframe(
            completion, artifact.component_id
        )
        bodies = completion_module.body_candidates_dataframe(
            completion, artifact.component_id
        )
        sections = completion_module.cross_sections_dataframe(
            completion, artifact.component_id
        )
        marker_completion = completion_module.marker_completion_dataframe(
            completion, artifact.component_id
        )
        save_csv(caps, prefix / "surface_caps.csv")
        save_csv(bodies, prefix / "candidate_bodies.csv")
        selected = (
            bodies[bodies["selected"].astype(bool)]
            if "selected" in bodies else bodies.copy()
        )
        save_csv(selected, prefix / "selected_bodies.csv")
        save_csv(sections, prefix / "cross_sections.csv")
        save_csv(marker_completion, prefix / "marker_completion.csv")
        save_csv(
            pd.DataFrame(
                [
                    {
                        "component_id": artifact.component_id,
                        "source": marker.source,
                        "source_reference_id": marker.source_reference_id,
                        "z": marker.position_zyx[0],
                        "y": marker.position_zyx[1],
                        "x": marker.position_zyx[2],
                        "confidence": marker.confidence,
                    }
                    for marker in artifact.final_markers
                ]
            ),
            prefix / "final_markers.csv",
        )

    for diagnostic in result.component_diagnostics:
        summary_rows.append(
            {
                **diagnostic.__dict__,
                "marker_positions_zyx": repr(diagnostic.marker_positions_zyx),
                "bbox_zyx": repr(diagnostic.bbox_zyx),
                "before_instance_count": diagnostic.effective_peak_count,
                "after_instance_count": diagnostic.instance_count,
            }
        )
    save_csv(pd.DataFrame(summary_rows), output / "component_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=Path, help="3-D binary .npy mask")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="case output directory (default: outputs/<mask stem>)",
    )
    parser.add_argument("--force-geometric-analysis", action="store_true")
    arguments = parser.parse_args()
    default_root = Path(__file__).resolve().parent / "outputs"
    output = arguments.output or default_root / arguments.mask.stem
    run(arguments.mask, output, arguments.force_geometric_analysis)


if __name__ == "__main__":
    main()
