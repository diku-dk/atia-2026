#!/usr/bin/env python3
"""Registry-based training CLI.

Picks a setup (``"<framework>-<task>"``, e.g. ``yolo-seg``),
loads that framework's config (an Ultralytics YAML),
applies any trailing ``key=value`` overrides, and runs it via ``src.registry``
for one or both dataset variants.

Usage
-----
    uv run -m src.train --list
    uv run -m src.train yolo-seg --variant CropOrWeed2 epochs=1 fraction=0.02
    uv run -m src.train yolo-seg --variant Fine24 --wandb
    uv run -m src.train yolo-seg --background --wandb --device 0
    uv run -m src.train yolo-seg --device both
    uv run -m src.train yolo-seg --variant CropOrWeed2 --split-seed 1
    uv run -m src.train yolo-seg --variant Fine24 --device 0 resume=output/yolo-seg/Fine24/train/weights/last.pt patience=4
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# `uv run src/train.py ...` runs this file as a script, which puts src/ (not
# the project root) on sys.path, so `import src...` would fail. Add the
# project root explicitly; harmless when already importable (e.g. via
# `uv run -m src.train`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.eomt as _eomt_framework  # noqa: E402  (populates the registry; also gives DEFAULT_CONFIGS)
import src.yolo as _yolo_framework  # noqa: E402
from src import registry  # noqa: E402

# Framework import list: importing each module registers its setups.
# Extend this (and _DEFAULT_CONFIGS below) as frameworks are added.
_DEFAULT_CONFIGS = {
    "yolo": _yolo_framework.DEFAULT_CONFIGS,
    "eomt": _eomt_framework.DEFAULT_CONFIGS,
}


def _load_config(path: Path) -> dict:
    """Load a framework config file, dispatching on its extension."""
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        if path.suffix == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported config extension '{path.suffix}' for {path}")


def _parse_overrides(pairs: list[str]) -> dict:
    """Parse trailing CLI ``key=value`` args, decoding each value with YAML.

    YAML gives us ints/floats/bools/None for free (``epochs=1``,
    ``plots=false``) while falling back to plain strings for anything else.
    """
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' isn't of the form key=value")
        key, _, value = pair.partition("=")
        overrides[key] = yaml.safe_load(value)
    return overrides


def _default_config_path(setup: str) -> Path:
    """Return the framework's default config file for this setup's task."""
    framework, _, task = setup.partition("-")
    return _DEFAULT_CONFIGS[framework][task]


def _run_background(
    argv: list[str], output_dir: Path, setup: str, variant: str
) -> None:
    """Re-exec this CLI without --background, detached, logging to a file.

    Runs in a new session (like ``nohup``) so the job survives the terminal
    closing. Prints the PID and log path, then returns immediately.
    """
    import subprocess

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{setup}_{variant}_{timestamp}.log"

    cmd = [sys.executable, "-m", "src.train"] + [a for a in argv if a != "--background"]
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(_PROJECT_ROOT),
        )
    print(f"Started in background: PID {proc.pid}, log at {log_path}")


def _parse_device(device: str | None) -> str | None:
    """Normalize ``--device``: 'both' expands to every visible GPU."""
    if device != "both":
        return device
    import torch

    return ",".join(str(i) for i in range(torch.cuda.device_count()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "setup",
        nargs="?",
        choices=registry.available(),
        help="e.g. yolo-seg",
    )
    parser.add_argument(
        "--variant",
        choices=(*registry.VARIANTS, "all"),
        default="all",
        help="Dataset variant to train on; 'all' runs each variant in sequence (default).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        choices=registry.SPLIT_SEEDS,
        default=42,
        help="Dataset split seed, i.e. the data/seed<N>/ root to train on (default: 42).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file; defaults to the setup's own default.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available setups and exit."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root for run outputs (default: output/).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (off by default).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="GPU(s) to train on: 0, 1, 0,1 or 'both' (all visible GPUs). "
        "Default: the config's device.",
    )
    parser.add_argument(
        "--background", action="store_true", help="Run detached; returns immediately."
    )
    # Trailing key=value overrides aren't declared as a positional: mixing a
    # REMAINDER positional with preceding optionals makes argparse swallow
    # those optionals (e.g. `--variant`) into the remainder instead of
    # parsing them. parse_known_args() instead parses every flag normally
    # and hands back whatever it didn't recognize (the key=value pairs).
    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    if args.list:
        for setup in registry.available():
            print(setup)
        return

    if args.setup is None:
        parser.error("the following arguments are required: setup (or pass --list)")

    if args.background:
        _run_background(sys.argv[1:], args.output_dir, args.setup, args.variant)
        return

    config_path = args.config or _default_config_path(args.setup)
    config = _load_config(config_path)
    config.update(_parse_overrides(args.overrides))

    run = registry.RunOptions(
        output_dir=args.output_dir,
        wandb=args.wandb,
        device=_parse_device(args.device),
        data_root=registry.data_root(args.split_seed),
    )
    _, _, task = args.setup.partition("-")
    trainer_cls = registry.get(args.setup)

    variants = registry.VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        print(f"=== {args.setup} / {variant} ===")
        trainer_cls(task, variant, dict(config), run).train()


if __name__ == "__main__":
    main()
