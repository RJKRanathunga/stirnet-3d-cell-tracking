"""Optional persistence of Stage 12 diagnostic tables."""

from __future__ import annotations

import json
from pathlib import Path


def save_final_visualization_result(result, directory: str | Path) -> Path:
    """Save Stage 12 audit artifacts without changing Stage 11 outputs."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    result.track_summary.to_csv(root / "final_track_summary.csv", index=False)
    result.diagnostic_events.to_csv(root / "diagnostic_events.csv", index=False)
    result.failure_events.to_csv(root / "failure_events.csv", index=False)
    metadata = {
        "schema_version": 1,
        "stage_name": "12_final_visualization",
        "sequence_first_frame": int(result.sequence_first_frame),
        "sequence_last_frame": int(result.sequence_last_frame),
        "voxel_size_zyx_um": list(result.voxel_size_zyx),
        "summary": result.summary,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root
