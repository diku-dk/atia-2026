#!/usr/bin/env python3
"""Final-evaluation CLI.

Evaluates trained runs on a dataset split with the framework's own high-level API:
``YOLO(best.pt).val()`` (``src/yolo/evaluator.py``) or Lightning's ``validate()`` of EoMT's
``best.ckpt`` (``src/eomt/evaluator.py``). The framework is picked from the run directory,
``output/<framework>-<task>/<variant>/<name>/``.

Each run's metrics are written as CSV to ``results/final_eval/<setup>/<variant>/<name>_<split>.csv``.

The predictions are also scored with hotcoco under the CropAndWeed paper's protocol
(Vegetation ignore regions, > 16^2 px only) and, for reference, plain COCO, into
``<name>_<split>_cropandweed.csv`` (one row per protocol, masks only; see ``src/cropandweed_eval.py``).

Usage
-----
    uv run -m src.evaluate output/yolo-seg/CropOrWeed2/train --device 0
    uv run -m src.evaluate --all --device 0
    uv run -m src.evaluate --setup yolo-seg --variant all --device 0
    uv run -m src.evaluate --all --background --device 0
    uv run -m src.evaluate output/yolo-seg/CropOrWeed2/train --split-seed 1 --device 0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# `uv run src/evaluate.py ...` puts src/ (not the project root) on sys.path, so `import src...`
# would fail; harmless when run as `uv run -m src.evaluate`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl  # noqa: E402

import src.eomt  # noqa: E402, F401  (registers the setups, whose trainers name the checkpoints)
import src.yolo  # noqa: E402, F401
from src import cropandweed_eval, registry  # noqa: E402
from src.eomt.evaluator import EoMTEvaluator  # noqa: E402
from src.yolo.evaluator import YOLOEvaluator  # noqa: E402

# Keyed by framework (the setup name's part before "-").
EVALUATORS: dict[str, type] = {"yolo": YOLOEvaluator, "eomt": EoMTEvaluator}

_DEFAULT_RESULTS_DIR = Path("results/final_eval")
# The frameworks' own val/eval scratch output (plots, predictions); not committed.
_DEFAULT_EVAL_LOG_DIR = Path("output/eval")


class FinalEvaluation:
    """Evaluate one ``output/<setup>/<variant>/<name>/`` run; the framework writes its metrics CSV."""

    def __init__(
        self,
        run_dir: Path,
        split: str = "test",
        device: str = "0",
        output_dir: Path = _DEFAULT_RESULTS_DIR,
        data_root: Path = registry.DATA_ROOT,
    ):
        self.run_dir = Path(run_dir)
        self.split = split
        self.device = device
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root)
        self.setup = self.run_dir.parent.parent.name
        self.variant = self.run_dir.parent.name
        self.name = self.run_dir.name
        framework = self.setup.partition("-")[0]
        self.log_dir = _DEFAULT_EVAL_LOG_DIR / self.setup / self.variant
        self.evaluator = EVALUATORS[framework](
            self.run_dir / registry.get(self.setup).BEST_CHECKPOINT,
            self.variant, self.data_root, self.log_dir, self.name,
        )

    def run(self) -> Path:
        print(f"[{self.setup}/{self.variant}/{self.name}] evaluating on {self.split}...")
        csv_path = self.output_dir / self.setup / self.variant / f"{self.name}_{self.split}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        predictions, _ = self.evaluator.evaluate(self.split, self.device, csv_path)
        print(f"Wrote {csv_path}")

        predictions_path = self.log_dir / f"{self.name}_{self.split}_predictions.json"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(json.dumps(predictions))
        rows = cropandweed_eval.evaluate(
            self.evaluator.dataset_dir,
            self.split,
            predictions,
            self.evaluator.iou_types,
        )
        cnw_csv_path = csv_path.with_name(f"{csv_path.stem}_cropandweed.csv")
        pl.DataFrame(rows).write_csv(cnw_csv_path)
        print(f"Wrote {cnw_csv_path}")
        return csv_path

    @staticmethod
    def discover(output_root: Path, setup: str = "*", variant: str = "all") -> list[Path]:
        """Every run dir with a best checkpoint under ``output_root/<setup>/<variant>/``."""
        variant_glob = "*" if variant == "all" else variant
        return sorted(
            p
            for p in Path(output_root).glob(f"{setup}/{variant_glob}/*")
            if p.parent.parent.name in registry.REGISTRY
            and (p / registry.get(p.parent.parent.name).BEST_CHECKPOINT).exists()
        )


def _run_background(argv: list[str], output_dir: Path) -> None:
    """Re-exec this CLI without --background, detached, logging to a file (as ``src/train.py`` does)."""
    import subprocess

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [sys.executable, "-m", "src.evaluate"] + [a for a in argv if a != "--background"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", type=Path, help="Run directories to evaluate.")
    parser.add_argument("--all", action="store_true", help="Evaluate every run under --output-root.")
    parser.add_argument("--setup", default=None, help="e.g. yolo-seg; evaluates that setup's runs.")
    parser.add_argument("--variant", choices=(*registry.VARIANTS, "all"), default="all", help="Used with --setup.")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--split-seed", type=int, choices=registry.SPLIT_SEEDS, default=42,
                        help="Dataset split seed, i.e. the data/seed<N>/ root to evaluate on (default: 42).")
    parser.add_argument("--device", default="0", help="GPU index (e.g. '0') or 'cpu'.")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=Path("output"), help="Root searched by --all/--setup.")
    parser.add_argument("--background", action="store_true", help="Run detached; returns immediately.")
    args = parser.parse_args()

    if args.background:
        _run_background(sys.argv[1:], Path("output"))
        return

    run_dirs = args.run_dirs
    if not run_dirs and (args.all or args.setup):
        run_dirs = FinalEvaluation.discover(args.output_root, args.setup or "*", args.variant)
    if not run_dirs:
        parser.error("No run directories to evaluate: pass run_dirs, --all, or --setup.")

    for run_dir in run_dirs:
        FinalEvaluation(
            run_dir, split=args.split, device=args.device, output_dir=args.output_dir,
            data_root=registry.data_root(args.split_seed),
        ).run()


if __name__ == "__main__":
    main()
