"""EoMT training entry point, registered with ``src.registry`` as ``eomt-seg``.

``EoMTTrainer`` fine-tunes EoMT (upstream code copied into ``src/eomt/``) for instance
segmentation on a variant's YOLO-seg dataset, through upstream's own ``LightningCLI``
(``src/eomt/cli.py``) built with ``run=False`` and its ``fit``. ``config`` is a LightningCLI config
(nested ``trainer``/``model``/``data``, as loaded from ``configs/seg.yaml``) plus a few wrapper-only
keys this class consumes:

- dotted keys (``trainer.max_epochs: 1``, ``data.init_args.img_size: [1280, 1280]``), e.g. from
  CLI ``key=value`` overrides or ``src/experiments.yaml``, are set into the nested config;
- ``project``/``name``/``exist_ok``: the run dir ``<project>/<name>`` (default
  ``output/eomt-seg/<variant>/train``, auto-incremented like Ultralytics unless ``exist_ok``);
- ``resume``: a ``last.ckpt`` to resume (the run dir is then its ``weights/``'s parent);
- ``pretrained``: ``{repo_id, filename}`` of upstream weights on the Hugging Face hub, downloaded
  and passed as ``model.init_args.ckpt_path`` (skipped when resuming).

The run dir holds ``config.yaml`` (the resolved LightningCLI config, read back by
``src/eomt/evaluator.py``), ``weights/best.ckpt`` (best ``metrics/val_ap_all``, upstream's
validation mask mAP) and ``weights/last.ckpt``, and ``metrics.csv`` (or ``wandb/`` with
``run.wandb``). ``data.init_args.batch_size`` is the total batch, split across the GPUs
(upstream's "devices x batch_size = 16").

Under DDP, Lightning re-launches the same command once per extra GPU, so every rank runs this
class; the run dir is handed to them through ``_RUN_DIR_ENV``, and after ``fit`` the extra ranks
exit so only rank 0 goes on (e.g. to the evaluation in ``src/experiments.py``). A process can only
run one multi-GPU fit, so multi-GPU ``src/train.py`` runs need one ``--variant`` per call.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.registry import RunOptions, register
from src.seg_dataset import read_names

# Set by rank 0 before `fit`; the DDP ranks Lightning launches inherit it.
_RUN_DIR_ENV = "EOMT_RUN_DIR"
# Keys consumed here, not LightningCLI's.
_WRAPPER_KEYS = ("project", "name", "exist_ok", "resume", "pretrained")


def apply_dotted(config: dict, overrides: dict) -> None:
    """Set each ``"a.b.c": value`` of ``overrides`` into the nested ``config``, in place."""
    for key, value in overrides.items():
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value


def _increment(path: Path) -> Path:
    """``path`` if free, else ``path2``, ``path3``, ... (Ultralytics' ``increment_path`` naming)."""
    n = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.name}{n}")
        n += 1
    return candidate


@register("eomt", "seg")
class EoMTTrainer:
    """Fine-tune EoMT for ``task`` on ``variant``, honouring ``run`` (see module docstring)."""

    BEST_CHECKPOINT = "weights/best.ckpt"
    LAST_CHECKPOINT = "weights/last.ckpt"
    # Whether this process already ran a fit (multi-GPU runs need a fresh process each).
    _fitted = False

    def __init__(self, task: str, variant: str, config: dict, run: RunOptions):
        self.task = task
        self.variant = variant
        self.config = config
        self.run = run

    def train(self) -> Any:
        from huggingface_hub import hf_hub_download

        from .cli import cli_main

        wrapper = {k: self.config[k] for k in _WRAPPER_KEYS if k in self.config}
        config = {k: v for k, v in self.config.items() if k not in _WRAPPER_KEYS and "." not in k}
        apply_dotted(config, {k: v for k, v in self.config.items() if "." in k})
        resume = wrapper.get("resume")
        run_dir = self._run_dir(wrapper)

        dataset_dir = self.run.data_root / self.variant / "yolo" / "segmentation"
        if not (dataset_dir / "data.yaml").exists():
            raise FileNotFoundError(f"No data.yaml for {self.variant} at {dataset_dir}")
        data_args = config["data"].setdefault("init_args", {})
        data_args["path"] = str(dataset_dir)
        data_args["num_classes"] = len(read_names(dataset_dir))
        if wrapper.get("pretrained") and not resume:
            config["model"]["init_args"]["ckpt_path"] = hf_hub_download(**wrapper["pretrained"])

        trainer_args = config.setdefault("trainer", {})
        if self.run.device is not None:
            trainer_args["devices"] = [int(d) for d in self.run.device.split(",")]
        trainer_args["default_root_dir"] = str(run_dir)
        trainer_args.setdefault("callbacks", []).append({
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": str(run_dir / Path(self.BEST_CHECKPOINT).parent),
                "filename": Path(self.BEST_CHECKPOINT).stem,
                "monitor": "metrics/val_ap_all",
                "mode": "max",
                "save_last": True,
            },
        })
        trainer_args["logger"] = self._logger(run_dir, resume)
        if not resume:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

        devices = trainer_args.get("devices", 1)
        n_devices = len(devices) if isinstance(devices, list) else int(devices)
        if n_devices > 1 and EoMTTrainer._fitted:
            raise RuntimeError(
                "A process can only run one multi-GPU EoMT fit; train one --variant per call."
            )
        if data_args.get("batch_size", 16) % n_devices:
            raise ValueError(f"batch_size {data_args['batch_size']} isn't divisible by {n_devices} GPUs")
        data_args["batch_size"] = data_args.get("batch_size", 16) // n_devices

        cli = cli_main(args=config, run=False)
        # The DDP ranks Lightning launches during `fit` inherit it; unset afterwards so a next variant
        # in this process gets its own run dir.
        os.environ[_RUN_DIR_ENV] = str(run_dir)
        try:
            # Upstream's checkpoints pickle the `network` hyperparameter, so they aren't weights-only.
            cli.fit(cli.model, datamodule=cli.datamodule, ckpt_path=resume, weights_only=False)
        finally:
            os.environ.pop(_RUN_DIR_ENV, None)
        EoMTTrainer._fitted = True
        if cli.trainer.global_rank != 0:
            sys.exit(0)
        return cli.trainer

    def _run_dir(self, wrapper: dict) -> Path:
        """The run dir: a DDP rank's inherited one, the resumed checkpoint's, or ``<project>/<name>``."""
        if _RUN_DIR_ENV in os.environ:
            run_dir = Path(os.environ[_RUN_DIR_ENV])
        elif wrapper.get("resume"):
            resume = Path(wrapper["resume"]).resolve()
            if not resume.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
            run_dir = resume.parent.parent
        else:
            project = Path(wrapper.get("project") or self.run.output_dir / f"eomt-{self.task}" / self.variant)
            run_dir = project.resolve() / wrapper.get("name", "train")
            if not wrapper.get("exist_ok"):
                run_dir = _increment(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _logger(self, run_dir: Path, resume: str | None) -> dict:
        """LightningCLI logger config: W&B with ``run.wandb`` (named as ``YOLOTrainer`` does), else a CSV log."""
        if not self.run.wandb:
            return {
                "class_path": "lightning.pytorch.loggers.CSVLogger",
                "init_args": {"save_dir": str(run_dir), "name": "", "version": ""},
            }
        try:  # output/eomt-seg/Fine24/train -> project eomt-seg-Fine24, run train
            project = "-".join(run_dir.parent.relative_to(self.run.output_dir.resolve()).parts)
        except ValueError:
            project = run_dir.parent.name
        latest_run = run_dir / "wandb" / "latest-run"
        resuming = resume is not None and latest_run.exists()
        return {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "project": project,
                "name": run_dir.name,
                # The id `src/experiments.py` reads back from wandb/run-<timestamp>-<id>.
                "id": latest_run.resolve().name.split("-", 2)[2]
                if resuming
                else f"{run_dir.name}_{datetime.now().astimezone():%Y%m%d_%H%M%S}",
                "resume": "allow",
                "save_dir": str(run_dir),
                **self.run.wandb_init,
            },
        }
