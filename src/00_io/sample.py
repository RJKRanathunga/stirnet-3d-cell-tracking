from pathlib import Path

from .paths import PROCESSED_ROOT


def get_sample_output_dir(sample_id: str) -> Path:
    return PROCESSED_ROOT / sample_id


def get_stage_dir(sample_id: str, stage: str) -> Path:
    return get_sample_output_dir(sample_id) / stage