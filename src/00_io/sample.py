from pathlib import Path

from src.io.paths import PipelinePaths


def get_sample_output_dir(sample_id: str) -> Path:
    return PipelinePaths.discover().processed_dataset(sample_id)


def get_stage_dir(sample_id: str, stage: str) -> Path:
    return get_sample_output_dir(sample_id) / stage
