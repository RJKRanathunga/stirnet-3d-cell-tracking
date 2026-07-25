from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_ROOT = DATA_ROOT / "train"

PROCESSED_ROOT = DATA_ROOT / "processed"