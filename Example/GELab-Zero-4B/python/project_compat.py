#!/usr/bin/env python3
"""Validate GELab's project-local Python implementation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = PROJECT_ROOT / "python"


def require_shared_sources() -> None:
    required = (
        LOCAL_PYTHON / "checkpoint.py",
        LOCAL_PYTHON / "contract.py",
        LOCAL_PYTHON / "prepare_media.py",
        LOCAL_PYTHON / "qwen_prefill.py",
        LOCAL_PYTHON / "npu_executor_config.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GELab project-local sources are missing: " + ", ".join(missing)
        )


def prepend_shared_paths() -> None:
    require_shared_sources()
    value = str(LOCAL_PYTHON)
    if value not in sys.path:
        sys.path.insert(0, value)
