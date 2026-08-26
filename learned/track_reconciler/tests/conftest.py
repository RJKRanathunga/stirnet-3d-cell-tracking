"""Pytest bootstrap for running the package directly from the repository checkout.

The repository's pyproject currently installs only ``src`` and ``diagnostics``.
When pytest is launched through the Windows ``pytest.exe`` entry point, the
repository root is not guaranteed to be present on ``sys.path``. The runtime
package still works with ``python -m learned.track_reconciler...`` because
Python adds the working directory itself.

This keeps the documented command working:

    pytest learned/track_reconciler/tests -q
"""

from __future__ import annotations

from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_repository_root = str(_REPOSITORY_ROOT)
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)
