#!/usr/bin/env python3
"""Experiment runner: every experiment of ``src/experiments.yaml`` on every split seed x dataset variant.

Runs sequentially, experiment -> split seed -> variant. Each cell trains (the setup's registered
trainer, its default config plus the experiment's ``train`` overrides and any trailing ``key=value``
overrides), then evaluates on ``test`` with the framework's own evaluator at ``batch=1`` (so its
speed is single-image latency), then scores the predicted masks per class with
``src/cropandweed_eval.py``.

Each cell runs in its own process (this CLI with ``--in-process``), whose output goes to the
cell's ``run.log``; the runner itself prints one status line per cell (and a failed cell's last
log lines). Every process a finished cell left behind is killed, so a crashed DDP run can't leave
data-loader workers holding RAM for the next cell.

Layout, per cell ``<exp>/seed<N>/<variant>/``:
    output/experiments/<cell>/run.log   the cell's full output (appended to on every attempt)
    output/experiments/<cell>/train/    training run (weights/best.pt; ``.done`` once finished)
    output/experiments/<cell>/eval/     evaluation scratch output
    output/experiments/runner_<ts>.log  the runner's status lines (with --background)
    results/experiments/<cell>/         <framework>.csv (ultralytics.csv, eomt.csv), cropandweed.csv, summary.json
    results/experiments/summary.csv     one row per cell (CropAndWeed protocol, segmentation)
    results/experiments/summary_agg.csv mean/std over split seeds

A cell with a ``summary.json`` is skipped (``--force`` re-evaluates it, reusing finished
training); an unfinished training run is resumed from its ``last.pt``. So a crashed run is simply
relaunched with the same command. With ``--wandb``, every cell is one W&B run in
``--wandb-project``, grouped by experiment (``group``), with ``split_seed``/``variant`` in its
config: the training run (``job_type`` train), to which the evaluation is added (its summary metrics
and a one-row table under ``summary/<variant>``, which W&B merges across runs into one table per
variant).

Choosing what runs
------------------
There is no exclude flag: ``--experiments``, ``--seeds`` and ``--variants`` each take the values
to *include* (default: all), so exclude something by listing the rest. ``--list`` prints the
experiment names. Cells that already have a ``summary.json`` are skipped anyway, so finished work
never needs excluding; delete a cell's ``results/experiments/<cell>/`` (and its
``output/experiments/<cell>/`` to retrain it too) to have it run again. To drop an experiment for
good, remove it from ``src/experiments.yaml`` (or point ``--config`` at another file).

Usage
-----
    uv run -m src.experiments --list
    uv run -m src.experiments --background --wandb --device both
    uv run -m src.experiments --experiments ft-640 --background --wandb --device both  # all but ft-1280
    uv run -m src.experiments --seeds 0 1 --variants Fine24 --device both              # skip seed 42 and CropOrWeed2
    uv run -m src.experiments --experiments ft-640 --seeds 42 --variants Fine24 --device 0 epochs=1 fraction=0.02
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

# `uv run src/experiments.py ...` puts src/ (not the project root) on sys.path; harmless with -m.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import polars as pl  # noqa: E402

from src import cropandweed_eval, registry  # noqa: E402
from src.evaluate import EVALUATORS  # noqa: E402
from src.train import (
    _default_config_path,
    _load_config,
    _parse_device,
    _parse_overrides,
)  # noqa: E402

# A DDP rank's exception line, e.g. "[rank1]: ValueError: ..." (not its indented traceback frames).
_RANK_ERROR = re.compile(r"^\[rank\d+\]: [\w.]*(Error|Exception)\b")
# Environment variable tagging every process of a cell run by _run_cell_process.
_CELL_TOKEN = "EXPERIMENTS_CELL_TOKEN"
_DEFAULT_EXPERIMENTS = Path(__file__).resolve().parent / "experiments.yaml"
_DEFAULT_RESULTS_DIR = Path("results/experiments")
_SPLIT = "test"
# CropAndWeed-protocol COCO stats reported per cell, renamed for the summary tables.
_SUMMARY_STATS = {
    "AP": "mAP50-95",
    "AP50": "mAP50",
    "AP_small": "mAP50-95_small",
    "AP_medium": "mAP50-95_medium",
    "AP_large": "mAP50-95_large",
}


class Cell:
    """One experiment on one split seed and dataset variant."""

    def __init__(
        self,
        experiment: str,
        spec: dict,
        seed: int,
        variant: str,
        args: argparse.Namespace,
    ):
        self.experiment = experiment
        self.spec = spec
        self.seed = seed
        self.variant = variant
        self.args = args
        rel = Path(experiment) / f"seed{seed}" / variant
        self.output_dir = (args.output_dir / "experiments" / rel).resolve()
        self.results_dir = args.results_dir / rel
        self.data_root = registry.data_root(seed)
        self.name = f"{experiment}-seed{seed}-{variant}"
        self.wandb_config = {
            "experiment": experiment,
            "split_seed": seed,
            "variant": variant,
        }

    def done(self) -> bool:
        """Whether the cell has results and isn't to be redone (``--force``)."""
        return (self.results_dir / "summary.json").exists() and not self.args.force

    def run(self) -> None:
        if self.done():
            print(f"=== {self.name}: done, skipping")
            return
        print(f"=== {self.name}")
        self._evaluate(self._train(), self.results_dir / "summary.json")

    def _train(self) -> Path:
        """Train (or resume, or reuse a finished run); return its best checkpoint."""
        setup = self.spec["setup"]
        trainer_cls = registry.get(setup)
        train_dir = self.output_dir / "train"
        best, last, done = (
            train_dir / trainer_cls.BEST_CHECKPOINT,
            train_dir / trainer_cls.LAST_CHECKPOINT,
            train_dir / ".done",
        )
        if done.exists():
            print(f"Training finished earlier, reusing {best}")
            return best

        config = _load_config(_default_config_path(setup))
        config.update(self.spec["train"])
        config.update(self.args.overrides)
        config.update(project=str(self.output_dir), name="train", exist_ok=True)
        if last.exists():
            print(f"Resuming unfinished training from {last}")
            config["resume"] = str(last)
        run = registry.RunOptions(
            output_dir=self.args.output_dir,
            wandb=self.args.wandb,
            device=self.args.device,
            data_root=self.data_root,
            wandb_init={
                "project": self.args.wandb_project,
                "group": self.experiment,
                "name": self.name,
                "job_type": "train",
                "tags": [self.variant, f"seed{self.seed}"],
                "config": self.wandb_config,
            },
        )
        trainer_cls(setup.partition("-")[2], self.variant, config, run).train()
        done.touch()
        return best

    def _evaluate(self, checkpoint: Path, summary_path: Path) -> None:
        eval_args = {**self.spec.get("eval", {}), "batch": 1}
        framework = self.spec["setup"].partition("-")[0]
        evaluator = EVALUATORS[framework](
            checkpoint,
            self.variant,
            self.data_root,
            self.output_dir,
            "eval",
            **eval_args,
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Evaluation runs on a single GPU: the first of --device.
        device = self.args.device.split(",")[0] if self.args.device else None
        predictions, framework_summary = evaluator.evaluate(
            _SPLIT, device, self.results_dir / evaluator.csv_name
        )

        rows = cropandweed_eval.evaluate(
            evaluator.dataset_dir, _SPLIT, predictions, evaluator.iou_types
        )
        pl.from_dicts(rows, infer_schema_length=None).write_csv(
            self.results_dir / "cropandweed.csv"
        )
        cnw = next(
            r
            for r in rows
            if r["protocol"] == "cropandweed" and r["iou_type"] == "segm"
        )

        speed = framework_summary["speed_ms"]
        latency = {k: speed[k] for k in ("preprocess", "inference", "postprocess")}
        latency["total"] = sum(latency.values())
        summary = {
            **self.wandb_config,
            "checkpoint": str(checkpoint),
            "eval_args": eval_args,
            "framework_metrics": framework_summary["metrics"],
            "latency_ms": latency,
            # hotcoco reports -1 for a size bucket or class without ground truth.
            "cropandweed": {
                new: (cnw[old] if cnw[old] >= 0 else None)
                for old, new in _SUMMARY_STATS.items()
            },
            "per_class_ap": {
                k.removeprefix("AP/"): (v if v >= 0 else None)
                for k, v in cnw.items()
                if k.startswith("AP/")
            },
        }
        if self.args.wandb:
            self._log_wandb(summary, eval_args)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {summary_path}")

    def _log_wandb(self, summary: dict, eval_args: dict) -> None:
        """Add the evaluation to the cell's training run (one W&B run per cell); a new run if training wasn't logged."""
        import wandb

        train_dir = self.output_dir / "train"
        latest_run = train_dir / "wandb" / "latest-run"
        if latest_run.exists():
            # The id YOLOTrainer._wandb_init_callback gave the run: wandb/run-<timestamp>-<id>.
            run = wandb.init(
                project=self.args.wandb_project,
                id=latest_run.resolve().name.split("-", 2)[2],
                resume="allow",
                dir=str(train_dir),
            )
            run.config.update({"eval": eval_args}, allow_val_change=True)
        else:
            run = wandb.init(
                project=self.args.wandb_project,
                group=self.experiment,
                name=self.name,
                job_type="eval",
                tags=[self.variant, f"seed{self.seed}"],
                dir=str(self.output_dir),
                config={
                    **self.wandb_config,
                    "setup": self.spec["setup"],
                    "checkpoint": summary["checkpoint"],
                    **eval_args,
                },
            )
        run.summary.update(
            {
                **{
                    f"framework/{k}": v for k, v in summary["framework_metrics"].items()
                },
                **{f"latency/{k}_ms": v for k, v in summary["latency_ms"].items()},
                **{f"cnw/{k}": v for k, v in summary["cropandweed"].items()},
                **{f"cnw/AP/{c}": v for c, v in summary["per_class_ap"].items()},
            }
        )
        # Keyed by variant: W&B merges same-key tables of all runs into one panel, and the variants'
        # per-class columns differ.
        row = {
            "experiment": self.experiment,
            "split_seed": self.seed,
            **summary["cropandweed"],
            **{f"AP/{c}": v for c, v in summary["per_class_ap"].items()},
            "framework_mAP50-95(M)": summary["framework_metrics"].get(
                "metrics/mAP50-95(M)"
            ),
            "latency_total_ms": summary["latency_ms"]["total"],
        }
        run.log(
            {
                f"summary/{self.variant}": wandb.Table(
                    columns=list(row), data=[list(row.values())]
                )
            }
        )
        run.finish()


def write_summaries(results_dir: Path) -> None:
    """Rebuild ``summary.csv`` (one row per cell) and ``summary_agg.csv`` (mean/std over seeds)."""
    rows = []
    for path in sorted(results_dir.glob("*/seed*/*/summary.json")):
        s = json.loads(path.read_text())
        rows.append(
            {
                "experiment": s["experiment"],
                "split_seed": s["split_seed"],
                "variant": s["variant"],
                **s["cropandweed"],
                **{
                    f"framework_{k.removeprefix('metrics/')}": v
                    for k, v in s["framework_metrics"].items()
                    if k.startswith("metrics/mAP50-95")
                },
                **{f"latency_{k}_ms": v for k, v in s["latency_ms"].items()},
            }
        )
    if not rows:
        return
    df = pl.from_dicts(rows, infer_schema_length=None)
    df.write_csv(results_dir / "summary.csv")
    keys = ["experiment", "variant"]
    values = [c for c in df.columns if c not in (*keys, "split_seed")]
    agg = df.group_by(keys, maintain_order=True).agg(
        pl.col("split_seed").n_unique().alias("n_seeds"),
        *(pl.col(c).mean().alias(f"{c}_mean") for c in values),
        *(pl.col(c).std().alias(f"{c}_std") for c in values),
    )
    agg.write_csv(results_dir / "summary_agg.csv")
    print(f"Wrote {results_dir / 'summary.csv'} and {results_dir / 'summary_agg.csv'}")


def _kill_tagged(token: str) -> None:
    """SIGKILL every process whose environment has ``_CELL_TOKEN=token``."""
    needle = f"{_CELL_TOKEN}={token}".encode()
    for environ in Path("/proc").glob("[0-9]*/environ"):
        try:
            if needle in environ.read_bytes().split(b"\0"):
                os.kill(int(environ.parent.name), signal.SIGKILL)
        except OSError:  # exited meanwhile, or not ours
            pass


def _run_cell_process(
    cell: Cell, args: argparse.Namespace, raw_overrides: list[str]
) -> bool:
    """Run ``cell`` in its own process (this CLI with ``--in-process``), logging to the cell's ``run.log``.

    Every process the cell started is killed afterwards: a crashed DDP run can otherwise leave its
    ranks and data-loader workers alive, holding their RAM caches, and starve the next cell. They're
    found by an environment variable the child passes on, since torchrun starts each rank in a new
    session and a killed rank's workers are reparented, so neither the process group nor the process
    tree reaches them all.
    """
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.experiments",
        "--in-process",
        "--config",
        str(args.config.resolve()),
        "--experiments",
        cell.experiment,
        "--seeds",
        str(cell.seed),
        "--variants",
        cell.variant,
        "--wandb-project",
        args.wandb_project,
        "--output-dir",
        str(args.output_dir.resolve()),
        "--results-dir",
        str(args.results_dir.resolve()),
    ]
    cmd += ["--device", args.device] if args.device else []
    cmd += ["--wandb"] if args.wandb else []
    cmd += ["--force"] if args.force else []
    cmd += raw_overrides

    cell.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell.output_dir / "run.log"
    print(f"=== {cell.name}: started, log {log_path}")
    start = time.monotonic()
    with open(
        log_path, "a"
    ) as log_file:  # append: a relaunch keeps the earlier attempt's log
        log_file.write(f"\n##### {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
        log_file.flush()
        attempt_start = log_file.tell()
        token = f"{os.getpid()}-{time.time_ns()}"
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ, _CELL_TOKEN: token},
        )
        try:
            returncode = proc.wait()
        finally:
            _kill_tagged(token)
    minutes, seconds = divmod(int(time.monotonic() - start), 60)
    if returncode == 0:
        print(f"=== {cell.name}: done in {minutes}:{seconds:02d}")
        return True

    print(
        f"=== {cell.name}: FAILED (exit {returncode}) after {minutes}:{seconds:02d}; last lines of {log_path}:"
    )
    # tqdm redraws progress bars with \r; keep only each line's final state.
    lines = [
        line.rsplit("\r", 1)[-1]
        for line in log_path.read_bytes()[attempt_start:]
        .decode(errors="replace")
        .splitlines()
    ]
    tail = lines[-30:]
    print("\n".join(f"    {line}" for line in tail))
    # Under DDP the ranks' exceptions end up far above torchrun's and the parent's tracebacks.
    rank_errors = [line for line in lines if _RANK_ERROR.match(line)]
    if rank_errors:
        print("    -> errors in the DDP ranks:")
        print("\n".join(f"    {line}" for line in rank_errors))
    if any("SIGTERM" in line or "SIGKILL" in line for line in tail):
        print(
            "    -> killed by a signal, likely out of RAM (earlyoom); see `journalctl -t earlyoom`"
        )
    return False


def _run_background(argv: list[str], output_dir: Path) -> None:
    """Re-exec this CLI without --background, detached, logging to a file (as ``src/train.py`` does)."""
    log_dir = output_dir / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # -u: unbuffered, so the log shows progress as it happens.
    cmd = [sys.executable, "-u", "-m", "src.experiments"] + [
        a for a in argv if a != "--background"
    ]
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_EXPERIMENTS, help="Experiments YAML."
    )
    parser.add_argument(
        "--list", action="store_true", help="List the experiments and exit."
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None, help="Default: all, in file order."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=registry.SPLIT_SEEDS,
        default=list(registry.SPLIT_SEEDS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=registry.VARIANTS,
        default=list(registry.VARIANTS),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="GPU(s): 0, 1, 0,1 or 'both'. Default: the config's.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument("--wandb-project", default="atia-2026-experiments")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate cells that already have results.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Root for run outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--background", action="store_true", help="Run detached; returns immediately."
    )
    # Run the cells in this process instead of one child process per cell (used by those children).
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    # Trailing key=value training-config overrides, as in src/train.py.
    args, overrides = parser.parse_known_args()

    experiments = yaml.safe_load(args.config.read_text())
    if args.list:
        for name, spec in experiments.items():
            print(f"{name}: {spec}")
        return
    unknown = set(args.experiments or ()) - set(experiments)
    if unknown:
        parser.error(
            f"Unknown experiments: {sorted(unknown)}; available: {list(experiments)}"
        )

    if args.background:
        _run_background(sys.argv[1:], args.output_dir)
        return

    args.overrides = _parse_overrides(overrides)
    args.device = _parse_device(args.device)
    failed = []
    for name in args.experiments or experiments:
        for seed in args.seeds:
            for variant in args.variants:
                cell = Cell(name, experiments[name], seed, variant, args)
                if not args.in_process:
                    if cell.done():
                        print(f"=== {cell.name}: done, skipping")
                    elif not _run_cell_process(cell, args, overrides):
                        failed.append(cell.name)
                    write_summaries(args.results_dir)
                    continue
                try:
                    cell.run()
                except (
                    Exception
                ):  # keep going: one failed cell shouldn't stop a long run
                    traceback.print_exc()
                    failed.append(cell.name)
                    if (
                        args.wandb
                    ):  # a crashed training leaves its run open; the next cell would log into it
                        import wandb

                        wandb.finish(exit_code=1)
                write_summaries(args.results_dir)
    if failed:
        print(f"Failed cells: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
