from __future__ import annotations

from pathlib import Path
import os
import tempfile
import torch

from .historical_instances import TEMPORAL_CACHE_CONTRACT_VERSION


CACHE_CONTRACT_KEY = "stirnet_cache_contract_version"


def save_cache(path: str | Path, payload: dict) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(payload)
    payload[CACHE_CONTRACT_KEY]=TEMPORAL_CACHE_CONTRACT_VERSION
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
    version=int(payload.get(CACHE_CONTRACT_KEY,1))
    if version != TEMPORAL_CACHE_CONTRACT_VERSION:
        raise ValueError(
            f"STIR-Net temporal cache contract v{version} cannot be loaded as v"
            f"{TEMPORAL_CACHE_CONTRACT_VERSION}. Rebuild the temporal cache so stale "
            "graph/status semantics are not silently reused."
        )
    return payload
