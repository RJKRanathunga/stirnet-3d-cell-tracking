
from __future__ import annotations

# DATASET_CURATION_REFACTOR_CURRENT_V1
import sys
from pathlib import Path

here = Path(__file__).resolve()
ROOT = None
for candidate in (here.parent, *here.parents, Path.cwd().resolve()):
    if (
        (candidate / "pyproject.toml").is_file()
        and (candidate / "learned").is_dir()
        and (candidate / "src").is_dir()
    ):
        ROOT = candidate
        break
if ROOT is None:
    raise RuntimeError("Could not resolve repository root.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_curation._compat.merge_suspect_exporter import main

if __name__ == "__main__":
    result = main()
if isinstance(result, int):
    raise SystemExit(result)
