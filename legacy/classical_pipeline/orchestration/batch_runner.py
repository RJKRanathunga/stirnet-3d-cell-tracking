"""Dataset-level orchestration across all discovered full samples."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FullPipelineConfig
from .dataset import discover_samples
from .manifest import PipelineManifest
from .sample_runner import SampleRunResult, SampleStageError, run_sample


@dataclass(frozen=True)
class FullPipelineSummary:
    discovered_sample_count: int
    completed_sample_count: int
    failed_sample_count: int
    results: tuple[SampleRunResult, ...]

    @property
    def succeeded(self) -> bool:
        return self.failed_sample_count == 0


def run_full_pipeline(config: FullPipelineConfig) -> FullPipelineSummary:
    samples = discover_samples(
        config.dataset_root,
        train_subdirectory=config.train_subdirectory,
        sample_ids=config.sample_ids,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest = PipelineManifest(config.output_root)

    print("Full-dataset cell-tracking pipeline")
    print(f"Dataset root: {config.dataset_root}")
    print(f"Output root:  {config.output_root}")
    print(f"Samples:      {len(samples)}")
    print(f"Stages:       {config.selected_stages}")
    print(
        "Stage 7 graph: "
        f"mode={config.graph_mode}, algorithm={config.graph_algorithm}, "
        f"window={config.graph_window_size}"
    )
    print(f"Resume:       {config.resume}")

    results: list[SampleRunResult] = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"\n=== Sample {index}/{len(samples)}: {sample.sample_id} ===",
            flush=True,
        )
        try:
            result = run_sample(sample, config, manifest)
        except SampleStageError as error:
            result = SampleRunResult(
                sample_id=sample.sample_id,
                status="failed",
                completed_stages=(),
                skipped_stages=(),
                failed_stage=error.stage,
                error=str(error.original_error),
            )
            print(f"FAILED: {error}", flush=True)
            results.append(result)
            if not config.continue_on_error:
                break
            continue
        except Exception as error:
            result = SampleRunResult(
                sample_id=sample.sample_id,
                status="failed",
                completed_stages=(),
                skipped_stages=(),
                failed_stage=None,
                error=f"{type(error).__name__}: {error}",
            )
            print(
                f"FAILED before/during stage setup: {type(error).__name__}: {error}",
                flush=True,
            )
            results.append(result)
            if not config.continue_on_error:
                break
            continue

        results.append(result)

    completed = sum(result.status == "completed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    summary = FullPipelineSummary(
        discovered_sample_count=len(samples),
        completed_sample_count=completed,
        failed_sample_count=failed,
        results=tuple(results),
    )

    print("\n=====================================")
    print("Full-dataset processing finished")
    print(f"Discovered samples: {summary.discovered_sample_count}")
    print(f"Completed samples:  {summary.completed_sample_count}")
    print(f"Failed samples:     {summary.failed_sample_count}")
    if failed:
        print("Failures:")
        for result in results:
            if result.status == "failed":
                stage = (
                    f"stage {result.failed_stage}"
                    if result.failed_stage is not None
                    else "setup"
                )
                print(f"  {result.sample_id}: {stage}: {result.error}")
    print(f"Manifest: {config.output_root / 'pipeline_manifest.csv'}")
    print("=====================================")
    return summary


__all__ = ["FullPipelineSummary", "run_full_pipeline"]
