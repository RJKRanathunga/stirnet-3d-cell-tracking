from __future__ import annotations

from pathlib import Path
import os
import tempfile
import torch

from .historical_instances import HISTORY_CACHE_CONTRACT_VERSION


CACHE_CONTRACT_KEY = "stirnet_cache_contract_version"


def save_cache(path: str | Path, payload: dict) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(payload)
    payload[CACHE_CONTRACT_KEY]=HISTORY_CACHE_CONTRACT_VERSION
    fd,tmp=tempfile.mkstemp(prefix=path.name,suffix=".tmp",dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_cache(path: str | Path, map_location="cpu") -> dict:
    payload=torch.load(Path(path),map_location=map_location,weights_only=False)
    if not isinstance(payload,dict):
        raise TypeError("STIR-Net cache payload must be a dictionary")
    # Version 1/unversioned caches are accepted through explicit no-history defaults.
    payload.setdefault(CACHE_CONTRACT_KEY,1)
    return payload
