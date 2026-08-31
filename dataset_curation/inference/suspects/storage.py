
from __future__ import annotations
from pathlib import Path
import numpy as np

def load_frame_scores(root: str | Path, frame: int) -> dict[str, np.ndarray]:
    path = Path(root) / f"t{int(frame):03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}
