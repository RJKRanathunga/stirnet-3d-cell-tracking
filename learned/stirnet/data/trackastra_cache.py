from __future__ import annotations

from pathlib import Path
import os
import tempfile
import torch


def save_cache(path: str | Path, payload: dict) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name,suffix=".tmp",dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_cache(path: str | Path, map_location="cpu") -> dict:
    return torch.load(Path(path),map_location=map_location,weights_only=False)
