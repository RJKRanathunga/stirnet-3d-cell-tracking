"""Top-level candidate mining orchestration."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, MergeRealConfig, MergeRealPaths
from .fusion import fuse_candidates
from .observations import ObservationIndex
from .repository_io import FullProcessedRepository, SampleArtifacts
from .sources import mine_source1, mine_source2, mine_source3, mine_source4


SOURCE_RUNNERS = (mine_source1, mine_source2, mine_source3, mine_source4)


def mine_sample(
    sample: SampleArtifacts,
    config: MergeRealConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ObservationIndex(sample, config)
    records = []
    for runner in SOURCE_RUNNERS:
        found = runner(sample, index, config)
        records.extend(found)
    source_candidates = pd.DataFrame([record.as_record() for record in records])
    if source_candidates.empty:
        source_candidates = pd.DataFrame(
            columns=[
                "candidate_id", "sample_id", "frame", "cell_id", "source",
                "track_a", "track_b", "involved_track_ids", "source_score",
            ]
        )
    fused = fuse_candidates(source_candidates)
    return source_candidates, fused


def mine_all_samples(
    paths: MergeRealPaths,
    config: MergeRealConfig = DEFAULT_CONFIG,
    *,
    sample_ids: tuple[str, ...] = (),
    print_progress: bool = True,
) -> pd.DataFrame:
    repository = FullProcessedRepository(paths)
    selected = repository.sample_ids(sample_ids)
    source_tables = []
    fused_tables = []

    for index, sample_id in enumerate(selected, start=1):
        if print_progress:
            print(f"[{index}/{len(selected)}] Mining {sample_id}...")
        sample = repository.sample(sample_id)
        source, fused = mine_sample(sample, config)
        source_tables.append(source)
        fused_tables.append(fused)
        if print_progress:
            print(f"  source records={len(source)} | fused candidates={len(fused)}")

    sources = pd.concat(source_tables, ignore_index=True) if source_tables else pd.DataFrame()
    candidates = pd.concat(fused_tables, ignore_index=True) if fused_tables else pd.DataFrame()
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["tier", "priority_score", "sample_id", "frame", "cell_id"],
            ascending=[True, False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

    paths.mining_dir.mkdir(parents=True, exist_ok=True)
    sources.to_csv(paths.mining_dir / "source_candidates.csv", index=False)
    candidates.to_csv(paths.mining_dir / "candidates.csv", index=False)

    summary = {
        "sample_count": len(selected),
        "source_candidate_count": int(len(sources)),
        "fused_candidate_count": int(len(candidates)),
        "tier_counts": candidates["tier"].value_counts().to_dict() if not candidates.empty else {},
        "source_counts": sources["source"].value_counts().to_dict() if not sources.empty else {},
    }
    _write_json(paths.mining_dir / "summary.json", summary)
    _write_json(
        paths.mining_dir / "run_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(paths.project_root),
            "full_processed_root": str(paths.full_processed_root),
            "output_root": str(paths.output_root),
            "sample_ids": list(selected),
            "git_commit": _git_commit(paths.project_root),
            "configuration": config.__dict__,
            "important_note": (
                "The current full-dataset runner calls process_dataset without a "
                "SegmentationConfig. At repository commit 6702b31 the Stage 3 default "
                "enable_geometric_completion is True. Verify the batch configuration "
                "if effective-peaks-only segmentation was intended."
            ),
        },
    )
    return candidates


def load_candidates(paths: MergeRealPaths) -> pd.DataFrame:
    path = paths.mining_dir / "candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run mining first; candidate manifest not found: {path}")
    return pd.read_csv(path)


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    temporary.replace(path)


__all__ = ["load_candidates", "mine_all_samples", "mine_sample"]
