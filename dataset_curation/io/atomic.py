
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
