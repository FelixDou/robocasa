"""Backward-compatible wrapper for the recovery failure dataset CLI.

Use ``robocasa/recovery/create_recovery_failure_dataset.py`` for new runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPL_PATH = Path(__file__).with_name("create_recovery_failure_dataset.py")
_SPEC = importlib.util.spec_from_file_location(
    "create_recovery_failure_dataset", _IMPL_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load recovery failure dataset CLI from {_IMPL_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)

main = _MODULE.main


if __name__ == "__main__":
    main()
