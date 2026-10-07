#!/usr/bin/env python3
"""Experiment runner: every experiment of ``src/experiments.yaml`` on every split seed (CropOrWeed2).

Runs sequentially, experiment -> split seed. Each cell trains with the framework's own CLI
(``yolo segment train cfg=src/yolo/configs/seg.yaml ...``, or upstream EoMT's ``python -m src.eomt.cli
fit -c src/eomt/configs/seg.yaml ...``), with the experiment's ``train`` overrides and any trailing
``key=value`` overrides as that CLI's args. It then evaluates on ``test`` at native resolution, at the
experiment's ``eval`` batch (the largest that fits a GPU), with the framework's own evaluation (``YOLO(best.pt).val()``; upstream's
``validate`` with ``src/eomt/callbacks.py:CocoPredictionWriter``) and scores the predicted masks with
``src/cropandweed_eval.py``.

Each cell runs in its own process (this CLI with ``--in-process``), whose output goes to the
cell's ``run.log``; the runner itself prints one status line per cell (and a failed cell's last
log lines). Every process a finished cell left behind is killed, so a crashed DDP run can't leave
data-loader workers holding RAM for the next cell.

Layout, per cell ``<exp>/seed<N>/``:
    output/experiments/<cell>/run.log   the cell's full output (appended to on every attempt)
    output/experiments/<cell>/train/    training run (weights/best.pt or best.ckpt; ``.done`` once finished)
    output/experiments/<cell>/eval/     evaluation output (predictions.json, ...)
    output/experiments/runner_<ts>.log  the runner's status lines (with --background)
    results/experiments/<cell>/         cropandweed.csv, summary.json
    results/experiments/summary.csv     one row per cell (CropAndWeed protocol, segmentation)
    results/experiments/summary_agg.csv mean/std over split seeds

A cell with a ``summary.json`` is skipped (``--force`` re-evaluates it, reusing finished
training); an unfinished training run is resumed from its last checkpoint. So a crashed run is simply
relaunched with the same command. With ``--wandb``, the evaluation is added to the cell's
training run (its summary metrics and a one-row table under ``summary``, which W&B merges
across runs into one table). EoMT's training run is in ``--wandb-project``, grouped by
experiment, with ``split_seed`` in its config; YOLO's is Ultralytics' own (project named
after the run's output path). A cell whose training wasn't logged gets a new ``job_type`` eval run
in ``--wandb-project``.

Choosing what runs
------------------
There is no exclude flag: ``--experiments`` and ``--seeds`` each take the values
to *include* (default: all), so exclude something by listing the rest. ``--list`` prints the
experiment names. Cells that already have a ``summary.json`` are skipped anyway, so finished work
never needs excluding; delete a cell's ``results/experiments/<cell>/`` (and its
``output/experiments/<cell>/`` to retrain it too) to have it run again. To drop an experiment for
good, remove it from ``src/experiments.yaml`` (or point ``--config`` at another file).

Usage
-----
    uv run -m src.experiments --list
    uv run -m src.experiments --background --wandb --device both
    uv run -m src.experiments --experiments yolo-1024x592 --background --wandb --device both  # just one
    uv run -m src.experiments --seeds 0 1 --device both                                # skip seed 42
    uv run -m src.experiments --experiments yolo-1024x592 --seeds 42 --device 0 epochs=1 fraction=0.02
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

from src import cropandweed_eval  # noqa: E402
from src.datasets.splits import SPLIT_SEEDS, data_root  # noqa: E402

# A DDP rank's exception line, e.g. "[rank1]: ValueError: ..." (not its indented traceback frames).
_RANK_ERROR = re.compile(r"^\[rank\d+\]: [\w.]*(Error|Exception)\b")
# A frame of the Python stack NCCL's watchdog logs for a timed-out collective, e.g. "#12 on_validation_epoch_end from ...".
_NCCL_FRAME = re.compile(r"^#\d+ \w+ from /")
# Environment variable tagging every process of a cell run by _run_cell_process.
_CELL_TOKEN = "EXPERIMENTS_CELL_TOKEN"
_DEFAULT_EXPERIMENTS = Path(__file__).resolve().parent / "experiments.yaml"
_DEFAULT_RESULTS_DIR = Path("results/experiments")
_SPLIT = "test"
_CONFIGS = {
    "yolo-seg": _PROJECT_ROOT / "src" / "yolo" / "configs" / "seg.yaml",
    "eomt-seg": _PROJECT_ROOT / "src" / "eomt" / "configs" / "seg.yaml",
}
_CHECKPOINT_SUFFIX = {"yolo-seg": ".pt", "eomt-seg": ".ckpt"}
# Upstream's COCO-panoptic EoMT-S (DINOv3) weights, the counterpart of YOLO's COCO-pretrained yolo26m-seg.pt.
_EOMT_PRETRAINED = {"repo_id": "tue-mps/coco_panoptic_eomt_small_640_dinov3", "filename": "pytorch_model.bin"}
# CropAndWeed-protocol COCO stats reported per cell, renamed for the summary tables.
_SUMMARY_STATS = {
    "AP": "mAP50-95",
    "AP50": "mAP50",
    "AP_small": "mAP50-95_small",
    "AP_medium": "mAP50-95_medium",
    "AP_large": "mAP50-95_large",
}


class Cell:
    """One experiment on one split seed."""

    def __init__(
        self,
        experiment: str,
        spec: dict,
        seed: int,
        args: argparse.Namespace,
    ):
        self.experiment = experiment
        self.spec = spec
        self.seed = seed
        self.args = args
        rel = Path(experiment) / f"seed{seed}"
        self.output_dir = (args.output_dir / "experiments" / rel).resolve()
        self.results_dir = args.results_dir / rel
        self.dataset_dir = data_root(seed)
        self.name = f"{experiment}-seed{seed}"
        self.wandb_config = {"experiment": experiment, "split_seed": seed}

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
        """Train with the framework's own CLI (or resume, or reuse a finished run); return the best checkpoint."""
        setup = self.spec["setup"]
        train_dir = self.output_dir / "train"
        best, last = (train_dir / "weights" / f"{kind}{_CHECKPOINT_SUFFIX[setup]}" for kind in ("best", "last"))
        done = train_dir / ".done"
        if done.exists():
            print(f"Training finished earlier, reusing {best}")
            return best
        resume = last.exists()
        if resume:
            print(f"Resuming unfinished training from {last}")
        cmd = self._yolo_train(last if resume else None) if setup == "yolo-seg" else self._eomt_train(resume)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)
        done.touch()
        return best

    def _yolo_train(self, last: Path | None) -> list[str]:
        """``yolo segment train`` with the setup's ``cfg=`` file; resuming loads ``last`` with ``resume``
        (Ultralytics then takes the run's own args, except its resume-allowed ones such as ``device``)."""
        from ultralytics import settings

        settings.update({"wandb": self.args.wandb})  # Ultralytics' own W&B callback
        cmd = [
            str(Path(sys.executable).with_name("yolo")), "segment", "train",
            f"cfg={_CONFIGS['yolo-seg']}",
            f"data={self.dataset_dir / 'data.yaml'}",
            f"project={self.output_dir}",
            "name=train",
            "exist_ok=True",
            *(f"{k}={v}" for k, v in self.spec["train"].items()),
            *self.args.overrides,
        ]
        cmd += [f"device={self.args.device}"] if self.args.device else []
        return cmd + (["resume", f"model={last}"] if last else [])

    def _eomt_train(self, resume: bool) -> list[str]:
        """Upstream's ``fit`` (``python -m src.eomt.cli``) with the run's checkpoint callback and logger."""
        train_dir = self.output_dir / "train"
        checkpoint = {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": str(train_dir / "weights"),
                "filename": "best",
                "monitor": "metrics/val_ap_all",
                "mode": "max",
                "save_last": True,
            },
        }
        cmd = self._eomt_cmd("fit") + [
            f"--trainer.default_root_dir={train_dir}",
            f"--trainer.callbacks+={json.dumps(checkpoint)}",
        ]
        if self.args.device:
            cmd.append(f"--trainer.devices={json.dumps([int(d) for d in self.args.device.split(',')])}")
        if self.args.wandb:
            cmd.append(f"--trainer.logger={json.dumps(self._wandb_logger(train_dir, resume))}")
        if resume:
            # "last": the checkpoint callback's last.ckpt. A file path would make LightningCLI re-parse the
            # checkpoint's hyperparameters, which fails on upstream's (saved without _class_path).
            cmd += ["--ckpt_path=last", "--weights_only=false"]
        else:
            from huggingface_hub import hf_hub_download

            cmd.append(f"--model.init_args.ckpt_path={hf_hub_download(**_EOMT_PRETRAINED)}")
        cmd += [f"--{k}={json.dumps(v)}" for k, v in self.spec["train"].items()]
        return cmd + [f"--{o}" for o in self.args.overrides]

    def _eomt_cmd(self, subcommand: str) -> list[str]:
        return [
            sys.executable, "-m", "src.eomt.cli", subcommand,
            "-c", str(_CONFIGS["eomt-seg"]),
            f"--data.init_args.path={self.dataset_dir}",
            f"--data.init_args.num_classes={len(yaml.safe_load((self.dataset_dir / 'data.yaml').read_text())['names'])}",
        ]

    def _wandb_logger(self, train_dir: Path, resume: bool) -> dict:
        """``WandbLogger`` spec for the cell's W&B run; a resumed training continues its run."""
        init_args = {"project": self.args.wandb_project, "name": self.name, "save_dir": str(train_dir)}
        latest_run = train_dir / "wandb" / "latest-run"
        if resume and latest_run.exists():
            init_args["id"] = latest_run.resolve().name.split("-", 2)[2]  # wandb/run-<timestamp>-<id>
        return {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": init_args,
            "dict_kwargs": {
                "resume": "allow",
                "group": self.experiment,
                "job_type": "train",
                "tags": [f"seed{self.seed}"],
                "config": self.wandb_config,
            },
        }

    def _evaluate(self, checkpoint: Path, summary_path: Path) -> None:
        """Evaluate on ``test`` with the framework's own API, then score with the CropAndWeed protocol."""
        eval_args = self.spec.get("eval", {})
        eval_dir = self.output_dir / "eval"
        # Evaluation runs on a single GPU: the first of --device.
        device = self.args.device.split(",")[0] if self.args.device else None
        if self.spec["setup"] == "yolo-seg":
            predictions, framework_metrics, latency = self._yolo_eval(checkpoint, eval_dir, device)
        else:
            predictions, framework_metrics, latency = self._eomt_eval(checkpoint, eval_dir, device)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        rows = cropandweed_eval.evaluate(self.dataset_dir, _SPLIT, predictions, ("segm",))
        pl.from_dicts(rows, infer_schema_length=None).write_csv(
            self.results_dir / "cropandweed.csv"
        )
        (cnw,) = rows
        summary = {
            **self.wandb_config,
            "checkpoint": str(checkpoint),
            "eval_args": eval_args,
            "framework_metrics": framework_metrics,
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

    def _yolo_eval(self, checkpoint: Path, eval_dir: Path, device: str | None) -> tuple[list[dict], dict, dict]:
        """``YOLO(best.pt).val()``; ``save_json`` writes the predictions as COCO results."""
        from ultralytics import YOLO

        metrics = YOLO(str(checkpoint)).val(
            data=str(self.dataset_dir / "data.yaml"),
            split=_SPLIT,
            device=device,
            project=str(eval_dir.parent),
            name=eval_dir.name,
            exist_ok=True,
            save_json=True,
            **self.spec.get("eval", {}),
        )
        # For non-COCO datasets Ultralytics sets image_id to the file stem and category_id to cls + 1.
        # It writes no predictions.json at all when the model detected nothing.
        predictions_path = Path(metrics.save_dir) / "predictions.json"
        predictions = json.loads(predictions_path.read_text()) if predictions_path.exists() else []
        predictions = [
            {**{k: v for k, v in p.items() if k not in ("image_id", "file_name")},
             "stem": Path(p["file_name"]).stem, "category_id": p["category_id"] - 1}
            for p in predictions
        ]
        latency = {k: float(metrics.speed[k]) for k in ("preprocess", "inference", "postprocess")}
        latency["total"] = sum(latency.values())
        framework_metrics = {k: float(v) for k, v in metrics.results_dict.items() if k.endswith("(M)")}
        return predictions, framework_metrics, latency

    def _eomt_eval(self, checkpoint: Path, eval_dir: Path, device: str | None) -> tuple[list[dict], dict, dict]:
        """Upstream's ``validate`` (as its README evaluates a checkpoint) on ``test``, with
        ``CocoPredictionWriter`` (``src/eomt/callbacks.py``) writing the predictions, latency and metrics."""
        writer = {"class_path": "src.eomt.callbacks.CocoPredictionWriter", "init_args": {"output_dir": str(eval_dir)}}
        cmd = self._eomt_cmd("validate") + [
            f"--data.init_args.val_split={_SPLIT}",
            f"--model.init_args.ckpt_path={checkpoint}",
            # A fine-tuned checkpoint: absolute weights, its own class head.
            "--model.init_args.delta_weights=false",
            "--model.init_args.load_ckpt_class_head=true",
            # As upstream's README; fully annealed after training, so predictions are unchanged.
            "--model.init_args.network.init_args.masked_attn_enabled=false",
            # YOLO's max_det and src/cropandweed_eval.py's maxDets[2] (upstream's default: 100).
            f"--model.init_args.eval_top_k_instances={cropandweed_eval.MAX_DETS[-1]}",
            f"--trainer.devices={json.dumps([int(device)] if device else 1)}",
            "--trainer.logger=false",
            f"--trainer.callbacks+={json.dumps(writer)}",
            *(f"--{k}={json.dumps(v)}" for k, v in self.spec.get("eval", {}).items()),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)
        result = json.loads((eval_dir / "eval.json").read_text())
        predictions = json.loads((eval_dir / "predictions.json").read_text())
        return predictions, result["metrics"], result["latency_ms"]

    def _log_wandb(self, summary: dict, eval_args: dict) -> None:
        """Add the evaluation to the cell's training run (one W&B run per cell); a new run if training wasn't logged."""
        import wandb

        train_dir = self.output_dir / "train"
        latest_run = train_dir / "wandb" / "latest-run"
        if latest_run.exists():
            # The training's run (wandb/run-<timestamp>-<id>), in its project: Ultralytics names it after
            # `project` with "/" -> "-"; the EoMT logger uses --wandb-project.
            project = (
                str(self.output_dir).replace("/", "-")
                if self.spec["setup"] == "yolo-seg"
                else self.args.wandb_project
            )
            run = wandb.init(
                project=project,
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
                tags=[f"seed{self.seed}"],
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
        # W&B merges same-key tables of all runs into one panel.
        row = {
            "experiment": self.experiment,
            "split_seed": self.seed,
            **summary["cropandweed"],
            **{f"AP/{c}": v for c, v in summary["per_class_ap"].items()},
            "framework_mAP50-95(M)": _framework_map(summary["framework_metrics"]),
            "latency_total_ms": summary["latency_ms"]["total"],
        }
        run.log(
            {
                "summary": wandb.Table(
                    columns=list(row), data=[list(row.values())]
                )
            }
        )
        run.finish()


def _framework_map(metrics: dict) -> float | None:
    """The framework's own mask mAP50-95: Ultralytics' ``metrics/mAP50-95(M)``, EoMT's ``metrics/val_ap_all``."""
    return metrics.get("metrics/mAP50-95(M)", metrics.get("metrics/val_ap_all"))


def _parse_device(device: str | None) -> str | None:
    """Normalize ``--device``: 'both' expands to every visible GPU."""
    if device != "both":
        return device
    import torch

    return ",".join(str(i) for i in range(torch.cuda.device_count()))


def write_summaries(results_dir: Path) -> None:
    """Rebuild ``summary.csv`` (one row per cell) and ``summary_agg.csv`` (mean/std over seeds)."""
    rows = []
    for path in sorted(results_dir.glob("*/seed*/summary.json")):
        s = json.loads(path.read_text())
        rows.append(
            {
                "experiment": s["experiment"],
                "split_seed": s["split_seed"],
                **s["cropandweed"],
                "framework_mAP50-95(M)": _framework_map(s["framework_metrics"]),
                **{f"latency_{k}_ms": v for k, v in s["latency_ms"].items()},
            }
        )
    if not rows:
        return
    df = pl.from_dicts(rows, infer_schema_length=None)
    df.write_csv(results_dir / "summary.csv")
    keys = ["experiment"]
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
    timeouts = [line for line in lines if "Watchdog caught collective operation timeout" in line]
    if timeouts:
        # The tail is the C++ abort trace; NCCL also logs the Python stack of the stuck collective.
        print(f"    -> {timeouts[0][timeouts[0].find('[rank'):].strip()}")  # drop a progress bar's prefix
        print("    -> the stuck collective's Python stack (project frames):")
        print("\n".join(f"    {line}" for line in lines if _NCCL_FRAME.match(line) and "/src/" in line and "/.venv/" not in line))
    if any("SIGTERM" in line or "SIGKILL" in line for line in tail):
        print(
            "    -> killed by a signal, likely out of RAM (earlyoom); see `journalctl -t earlyoom`"
        )
    return False


def _run_background(argv: list[str], output_dir: Path) -> None:
    """Re-exec this CLI without --background, detached, logging to a file."""
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
        choices=SPLIT_SEEDS,
        default=list(SPLIT_SEEDS),
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
    # Trailing key=value overrides of the training config, passed to the framework's CLI: Ultralytics
    # args (epochs=1) or dotted LightningCLI keys (trainer.max_epochs=1).
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

    args.overrides = overrides
    args.device = _parse_device(args.device)
    failed = []
    for name in args.experiments or experiments:
        for seed in args.seeds:
            cell = Cell(name, experiments[name], seed, args)
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
