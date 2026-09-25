"""Ultralytics YOLO training entry points, registered with ``src.registry``.

One function, ``train_yolo``, handles both detection and instance
segmentation: the only difference is which ``data.yaml`` it points at and
which pretrained checkpoint the config names.
"""

import importlib.util
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO, settings
from ultralytics.utils.autobatch import check_train_batch_size

from src.registry import DATA_ROOT, RunOptions, register

# Maps a task name (as used in setup keys "yolo-<task>") to the dataset
# subdirectory produced by scripts/convert_cropandweed.py.
_TASK_TO_SUBDIR = {"detect": "detection", "seg": "segmentation"}


@register("yolo", "detect")
@register("yolo", "seg")
def train_yolo(task: str, variant: str, config: dict, run: RunOptions) -> Any:
    """Fine-tune a YOLO model for ``task`` on ``variant``, honouring ``run``.

    ``config`` is a flat dict of Ultralytics train-arg overrides (as loaded
    from ``configs/detect.yaml`` / ``configs/seg.yaml``, or overridden from
    the CLI). It is mutated in place: ``data``, ``project`` and ``model`` are
    resolved/consumed here before being handed to ``model.train(**config)``.
    """
    subdir = _TASK_TO_SUBDIR[task]
    data_yaml = DATA_ROOT / variant / "yolo" / subdir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No data.yaml for {variant}/{subdir} at {data_yaml}")
    config["data"] = str(data_yaml)

    # Ultralytics auto-increments `name` (train, train2, ...) below `project`.
    # Must be absolute: a *relative* `project` is resolved by Ultralytics
    # against the global SETTINGS["runs_dir"] (~/.config/Ultralytics/...),
    # not the current working directory, so a bare "output/..." can land
    # somewhere else entirely (e.g. a leftover runs_dir from another
    # project on the machine). Resolving here keeps output under run.output_dir.
    config.setdefault(
        "project", str((run.output_dir / f"yolo-{task}" / variant).resolve())
    )

    # Explicitly set (rather than only when enabling) so a stale global
    # setting from a previous run can't silently leak into this one. This
    # must happen before model.train(), since the wandb callback module
    # checks SETTINGS["wandb"] once, the first time it is imported in this
    # process (inside ultralytics.utils.callbacks.add_integration_callbacks,
    # itself called from BaseTrainer.__init__ -> model.train()).
    if run.wandb and importlib.util.find_spec("wandb") is None:
        raise RuntimeError(
            "run.wandb is True but the 'wandb' package isn't installed. "
            "Install it with `uv pip install wandb`."
        )
    settings.update({"wandb": run.wandb})

    if run.device is not None:
        config["device"] = run.device

    model = YOLO(config.pop("model"))
    if run.wandb:
        model.add_callback("on_pretrain_routine_start", _wandb_init(run.output_dir))

    devices = [d for d in str(config.get("device", "")).split(",") if d.strip()]
    batch = config.get("batch", 16)
    if len(devices) > 1 and (batch == -1 or 0 < batch < 1):
        config["batch"] =_ddp_auto_batch(model, config, devices, data_yaml.parent)
    return model.train(**config)


def _wandb_init(output_dir: Path):
    """Callback that starts the W&B run under a name mirroring the local path.

    Ultralytics' own callback names the W&B project after ``args.project``
    with "/" turned into "-", i.e. the whole absolute output path. It only
    calls ``wandb.init`` if no run is active, so starting one first (this
    callback is registered before the integration callbacks) wins. Callbacks
    are pickled into the DDP workers, so this also holds for multi-GPU runs.
    """
    output_dir = output_dir.resolve()

    def callback(trainer) -> None:
        import wandb

        if wandb.run:
            return
        save_dir = Path(trainer.save_dir)
        try:  # output/yolo-seg/Fine24/train -> project yolo-seg-Fine24, run train
            project = "-".join(save_dir.parent.relative_to(output_dir).parts)
        except ValueError:  # a config-supplied `project` outside output_dir
            project = save_dir.parent.name
        latest_run = save_dir / "wandb" / "latest-run"
        resuming = trainer.args.resume and latest_run.exists()
        wandb.init(
            project=project,
            name=save_dir.name,
            config=vars(trainer.args),
            id=latest_run.resolve().name.split("-", 2)[2]
            if resuming
            else f"{save_dir.name}_{datetime.now().astimezone():%Y%m%d_%H%M%S}",
            resume="allow" if resuming else None,
            dir=str(save_dir),
        )

    return callback


def _ddp_auto_batch(model: YOLO, config: dict, devices: list[str], data_dir: Path) -> int:
    """AutoBatch for multi-GPU runs, which Ultralytics refuses to do itself.

    Profiles the per-GPU batch on the first device the same way the
    single-GPU trainer does (``DetectionTrainer.auto_batch``), then scales it
    by the number of GPUs since DDP splits the batch evenly across them.
    """
    batch = config.get("batch", -1)
    stride = 32
    imgsz = config.get("imgsz", 640)
    max_imgsz = math.ceil(imgsz * (1 + config.get("multi_scale", 0.0)) / stride) * stride

    # Mirrors DetectionTrainer.auto_batch: most objects in one image, x4 for mosaic.
    label_files = list((data_dir / "labels" / "train").glob("*.txt"))
    max_num_obj = max(sum(1 for _ in f.open()) for f in label_files) * 4
    n_images = len(label_files)

    net = model.model.to(f"cuda:{devices[0]}")
    try:
        per_gpu = check_train_batch_size(
            net,
            imgsz=max_imgsz,
            amp=config.get("amp", True),
            batch=batch,
            max_num_obj=max_num_obj,
            dataset_size=n_images,
        )
    finally:
        model.model.cpu()
        torch.cuda.empty_cache()
    return min(per_gpu * len(devices), n_images)
