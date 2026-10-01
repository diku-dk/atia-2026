"""Framework-agnostic training registry.

Each training framework (Ultralytics YOLO today, more later) lives in its own
subpackage under ``src/`` and registers one trainer *class* per task via the
``register`` decorator below, keyed by ``"<framework>-<task>"`` (e.g.
``"yolo-seg"``). ``src/train.py`` is the single CLI that
looks a setup up in ``REGISTRY`` and calls it.

Trainer contract
-----------------
    cls(task: str, variant: str, config: dict, run: RunOptions).train() -> Any

The class builds the model from ``config`` (a framework-native dict, e.g.
parsed from an Ultralytics YAML), runs training in ``train()``, and returns
whatever result object the framework produces (for Ultralytics, its training
metrics). The class attributes ``BEST_CHECKPOINT`` and ``LAST_CHECKPOINT`` give
the best and last checkpoints' paths relative to the run dir (e.g.
``weights/best.pt``), used by ``src/evaluate.py`` and ``src/experiments.py``.

This module also centralises the dataset variants and data root so every
framework resolves paths the same way.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Dataset variants produced by scripts/convert_cropandweed.py.
VARIANTS = ("CropOrWeed2", "Fine24")

# Split seeds converted by scripts/split_seeds.sh, one dataset root each (data/seed<N>/).
SPLIT_SEEDS = (42, 0, 1)


def data_root(seed: int) -> Path:
    """Dataset root of one split seed, resolved relative to this file
    (src/registry.py -> project root -> data/seed<N>/), independent of the caller's cwd."""
    return Path(__file__).resolve().parent.parent / "data" / f"seed{seed}"


# Default dataset root: the seed-42 split.
DATA_ROOT = data_root(42)

REGISTRY: dict[str, Callable] = {}


@dataclass(frozen=True)
class RunOptions:
    """Framework-agnostic launcher settings every trainer must honour.

    Background execution (running detached, logging to a file) is a
    process-level concern handled by ``train.py`` itself, so it isn't part of
    this dataclass.
    """

    output_dir: Path = Path("output")
    wandb: bool = False
    # Comma-separated GPU indices ("0", "1", "0,1"); None keeps the config's.
    device: str | None = None
    # Dataset root of the split seed to train on (data/seed<N>/).
    data_root: Path = DATA_ROOT
    # Extra wandb.init kwargs (project, group, name, job_type, config, ...)
    # overriding the trainer's path-derived defaults.
    wandb_init: dict = field(default_factory=dict)


def register(framework: str, task: str) -> Callable[[Callable], Callable]:
    """Return a decorator that stores ``fn`` in ``REGISTRY`` under ``"<framework>-<task>"``.

    Raises ``ValueError`` if the key is already taken. Returns ``fn``
    unchanged, so the decorator can be stacked (one class registered for
    several tasks).
    """

    def decorator(fn: Callable) -> Callable:
        key = f"{framework}-{task}"
        if key in REGISTRY:
            raise ValueError(
                f"Setup '{key}' is already registered (by {REGISTRY[key]!r})"
            )
        REGISTRY[key] = fn
        return fn

    return decorator


def get(setup: str) -> Callable:
    """Look up a registered factory by setup name, e.g. "yolo-seg"."""
    try:
        return REGISTRY[setup]
    except KeyError:
        listed = ", ".join(available()) or "(none registered)"
        raise KeyError(f"Unknown setup '{setup}'. Available setups: {listed}") from None


def available() -> list[str]:
    """Return the sorted list of registered setup names."""
    return sorted(REGISTRY)
