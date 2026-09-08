"""Make the repository importable regardless of the working directory.

``run_sandbox`` and the ``chebyshev_wavelet_py`` package live at the repository
root, which is only on ``sys.path`` automatically when pytest happens to be
invoked from there.  Without this the suite collects fine locally and fails in
CI with ``ModuleNotFoundError``, which is a confusing way to discover a path
problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
