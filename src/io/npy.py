from pathlib import Path

import numpy as np


def save_npy(array: np.ndarray, path: str | Path):
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, array)


def load_npy(path: str | Path):
    return np.load(path)