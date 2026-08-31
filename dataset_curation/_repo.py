
from __future__ import annotations
from pathlib import Path

def repo_root() -> Path:
    starts = [Path.cwd().resolve(), Path(__file__).resolve()]
    seen = set()
    for start in starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "learned").is_dir()
                and (candidate / "investigations").is_dir()
                and (candidate / "src").is_dir()
            ):
                return candidate
    raise RuntimeError("Could not resolve repository root.")
