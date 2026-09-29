"""Ultralytics YOLO training framework.

Importing this package imports ``trainer``, which registers the "yolo-seg"
setup with ``src.registry``.
"""
from __future__ import annotations

from pathlib import Path

from . import trainer  # noqa: F401  (import triggers @register)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Default config file per task, used by src/train.py when --config is omitted.
DEFAULT_CONFIGS = {
    "seg": _CONFIGS_DIR / "seg.yaml",
}
