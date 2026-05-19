from __future__ import annotations

from pathlib import Path
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_ROOT = Path(__file__).resolve().parent
_INNER = _ROOT / "cleanmarl"

if _INNER.exists():
    inner = str(_INNER)
    if inner not in __path__:
        __path__.append(inner)
    # CleanMARL upstream files use imports such as `from env...`.
    if inner not in sys.path:
        sys.path.insert(0, inner)

__all__ = []
