
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

import runpy

if __name__ == "__main__":
    runpy.run_module("dataset_curation._compat.point_annotator", run_name="__main__")
