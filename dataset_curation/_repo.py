from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Resolve the production repository root without requiring investigations/."""
    starts = (
        Path.cwd().resolve(),
        Path(__file__).resolve(),
    )
    seen: set[Path] = set()

    for start in starts:
        for candidate in (
            start,
            *start.parents,
        ):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)

            if (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "learned" / "stirnet").is_dir()
                and (candidate / "src").is_dir()
                and (candidate / "dataset_curation").is_dir()
            ):
                return candidate

    raise RuntimeError(
        "Could not resolve the cell-tracking repository root."
    )
