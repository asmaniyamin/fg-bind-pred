# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Add the repository root to sys.path for direct script execution."""

from pathlib import Path
import sys


def _find_repo_root(start: Path):
    for candidate in (start, *start.parents):
        if (candidate / "chem").is_dir() and (candidate / "core").is_dir():
            return candidate
    return None


_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if _ROOT is not None:
    _ROOT_STR = str(_ROOT)
    if _ROOT_STR not in sys.path:
        sys.path.insert(0, _ROOT_STR)

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
