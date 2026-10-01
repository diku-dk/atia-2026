"""EoMT (Kerssies et al., CVPR 2025) instance segmentation, built on upstream code copied from
tue-mps/eomt@7bd19dd (MIT, see ``LICENSE``): ``models/``, ``training/``, ``datasets/{lightning_data_module,
transforms}.py`` and ``cli.py`` (upstream ``main.py``), with imports made package-relative. Our own
glue: ``datasets/cropandweed_instance.py``, ``callbacks.py``, ``trainer.py``, ``evaluator.py`` and
``configs/``.

Importing this package imports ``trainer``, which registers the "eomt-seg" setup with ``src.registry``.
"""
from __future__ import annotations

from pathlib import Path

from . import trainer  # noqa: F401  (import triggers @register)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Default config file per task, used by src/train.py when --config is omitted.
DEFAULT_CONFIGS = {
    "seg": _CONFIGS_DIR / "seg.yaml",
}
