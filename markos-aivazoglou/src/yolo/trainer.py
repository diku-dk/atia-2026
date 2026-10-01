"""Ultralytics YOLO training entry point, registered with ``src.registry``.

One class, ``YOLOTrainer``, fine-tunes YOLO instance segmentation on a
variant's ``yolo/segmentation/data.yaml``, with the pretrained checkpoint
named in ``configs/seg.yaml``.

Native-resolution crop training (opt-in)
-----------------------------------------
Without ``crop_size`` (the default config), Ultralytics' stock trainer is used:
the whole frame letterboxed to ``imgsz``. With ``crop_size`` set (e.g. 640):
the dataset's native frames are 1920x1088; ``imgsz`` is set to 1920 (the
native long side, so ``BaseDataset.load_image`` loads them unscaled) while
``crop_size`` is the actual network input. Ultralytics'
documented ``augmentations=`` hook (and ``Albumentations`` more generally)
only runs *after* Mosaic/affine, by which point ``load_image`` has already
downscaled the frame to ``imgsz`` -- there's no built-in way to get random
640x640 crops of full-resolution frames feeding Mosaic. ``NativeCropDataset``
and ``_native_crop_trainer`` below hook ``get_image_and_label`` (which
Mosaic/MixUp/CutMix/CopyPaste all call per-tile) to crop every image to
``crop_size`` right after it's loaded natively, using Ultralytics' own
``Albumentations`` wrapper around stock
``albumentations.AtLeastOneBBoxRandomCrop`` (not plain ``RandomCrop``: the
wrapper silently returns the *uncropped* frame if a crop would drop every box
of an image that has boxes). Validation is unaffected -- it isn't built
through this hook and runs at the full ``imgsz`` (1920), rect-loaded, as
Ultralytics already does.
"""

import importlib.util
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import albumentations as A
import torch
from ultralytics import YOLO, settings
from ultralytics.data.augment import Albumentations
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_split_fraction
from ultralytics.utils import colorstr
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.patches import override_configs
from ultralytics.utils.torch_utils import unwrap_model

from src.registry import RunOptions, register

# Args Ultralytics' `check_resume` (engine/trainer.py) still lets through as
# overrides when resuming from a checkpoint; everything else in `config` is
# ignored in favour of the checkpoint's own saved args. Kept in sync with
# that function's allowlist.
_RESUME_ALLOWED_KEYS = (
    "imgsz",
    "batch",
    "device",
    "close_mosaic",
    "augmentations",
    "save_period",
    "workers",
    "cache",
    "patience",
    "time",
    "freeze",
    "val",
    "plots",
    "channels_last",
    "distill_model",
    "save_dir",
)


class NativeCropDataset(YOLODataset):
    """A ``YOLODataset`` whose images are random ``crop_size``x``crop_size`` crops of the native frame.

    ``imgsz`` (passed to the base class as usual) stays at the native long
    side so ``load_image`` doesn't downscale; the crop is applied in
    ``get_image_and_label``, before Mosaic/MixUp/CutMix/CopyPaste (which all
    fetch their extra tiles through this same method), so every tile any
    training transform sees is a native-resolution crop.
    """

    def __init__(self, *args, crop_size: int, **kwargs):
        self.crop_size = crop_size
        # Built before super().__init__(), which calls build_transforms().
        self.crop = Albumentations(
            p=1.0, transforms=[A.AtLeastOneBBoxRandomCrop(height=crop_size, width=crop_size)]
        )
        # Albumentations' internal yolo-format round trip can put a boundary coordinate
        # a few ULPs outside [0, 1] (e.g. -4.99e-07) for a box that starts exactly at the
        # frame edge -- common here (frame-edge-truncated plants, see data/README.md).
        # Its bbox validator then raises unless `clip` is set; that's Albumentations' own
        # documented `BboxParams` option for exactly this, but the Ultralytics wrapper
        # above doesn't expose it as a constructor arg, so it's set on the built pipeline.
        self.crop.transform.processors["bboxes"].params.clip = True
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Load the native-resolution image/label, then crop it to ``crop_size``."""
        # The wrapper seeds Albumentations' own RNG once, in the main process, and neither
        # DataLoader nor Ultralytics' seed_worker reseeds it, so every forked worker would
        # draw the same crop offsets. Reseed from the per-worker torch seed on first use
        # in each process (a no-op in the main process).
        if self.crop.transform.seed != torch.initial_seed():
            self.crop.transform.set_random_seed(torch.initial_seed())
        label = self.crop(super().get_image_and_label(index))
        label["resized_shape"] = label["img"].shape[:2]  # Mosaic reads this to place the tile
        return label

    def build_transforms(self, hyp=None):
        """Build the training pipeline sized for ``crop_size``, not the native ``imgsz``."""
        imgsz = self.imgsz
        self.imgsz = self.crop_size
        try:
            return super().build_transforms(hyp)
        finally:
            self.imgsz = imgsz


def _native_crop_trainer(base: type, crop_size: int) -> type:
    """Return a subclass of ``base`` (Ultralytics' SegmentationTrainer) that trains on native-frame crops.

    Defined *inside* this function rather than at module scope: multi-GPU
    Ultralytics runs (``utils/dist.py`` ``generate_ddp_file``) ``cloudpickle``
    ``type(trainer)`` to hand the trainer class to DDP worker processes. A
    class defined inside a function is pickled by value (closure included),
    so ``crop_size`` survives into the workers -- the same mechanism
    ``YOLOTrainer._wandb_init_callback``'s closure relies on.
    """

    class _NativeCropTrainer(base):
        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            """As ``DetectionTrainer.build_dataset``, but 'train' builds a ``NativeCropDataset``."""
            if mode != "train":
                return super().build_dataset(img_path, mode=mode, batch=batch)
            args = self.args
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            fraction = 1.0 if self.data.get("complete") else get_split_fraction(args.fraction, "train")
            return NativeCropDataset(
                img_path=img_path,
                imgsz=args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=args,
                rect=args.rect,
                cache=args.cache or None,
                single_cls=args.single_cls or False,
                stride=gs,
                pad=0.0,
                prefix=colorstr("train: "),
                task=args.task,
                classes=args.classes,
                data=self.data,
                fraction=fraction,
                crop_size=crop_size,
            )

        def auto_batch(self):
            """Profile AutoBatch at ``crop_size`` (the real network input), not the native ``imgsz``."""
            with override_configs(self.args, overrides={"imgsz": crop_size}) as self.args:
                return super().auto_batch()

    return _NativeCropTrainer


@register("yolo", "seg")
class YOLOTrainer:
    """Fine-tune a YOLO model for ``task`` on ``variant``, honouring ``run``.

    ``config`` is a flat dict of Ultralytics train-arg overrides (as loaded
    from ``configs/seg.yaml``, or overridden from
    the CLI). It is mutated in place by ``train()``: ``data``, ``project`` and
    ``model`` are resolved/consumed before being handed to
    ``model.train(**config)``. ``imgsz`` is the native frame's long side
    (1920); ``crop_size`` (e.g. 640, the real network input) isn't an Ultralytics
    arg, so it's popped from ``config`` and instead baked into a
    per-instance trainer subclass (see ``_native_crop_trainer``) passed as
    ``model.train(trainer=...)``. A ``crop_size`` of ``None`` (or no key) trains
    with Ultralytics' stock trainer instead, i.e. the whole frame letterboxed
    to ``imgsz``.

    If ``config`` has a ``resume`` key (a path to a ``last.pt``, set via the
    CLI override ``resume=path``), ``train()`` instead resumes that
    checkpoint: ``data``/``project`` come from the checkpoint itself, and only
    the Ultralytics resume-allowed args (``_RESUME_ALLOWED_KEYS``) are passed
    through from ``config``. ``crop_size`` still comes from ``config`` (the
    yaml + CLI overrides, same as a fresh run), since it isn't part of the
    checkpoint's own saved args.
    """

    BEST_CHECKPOINT = "weights/best.pt"
    LAST_CHECKPOINT = "weights/last.pt"

    def __init__(self, task: str, variant: str, config: dict, run: RunOptions):
        self.task = task
        self.variant = variant
        self.config = config
        self.run = run

    def train(self) -> Any:
        if "resume" in self.config:
            return self._resume()

        config = self.config
        data_yaml = self.run.data_root / self.variant / "yolo" / "segmentation" / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"No data.yaml for {self.variant} at {data_yaml}")
        config["data"] = str(data_yaml)

        # Ultralytics auto-increments `name` (train, train2, ...) below `project`.
        # Must be absolute: a *relative* `project` is resolved by Ultralytics
        # against the global SETTINGS["runs_dir"] (~/.config/Ultralytics/...),
        # not the current working directory, so a bare "output/..." can land
        # somewhere else entirely (e.g. a leftover runs_dir from another
        # project on the machine). Resolving here keeps output under run.output_dir.
        config.setdefault(
            "project",
            str((self.run.output_dir / f"yolo-{self.task}" / self.variant).resolve()),
        )

        # Explicitly set (rather than only when enabling) so a stale global
        # setting from a previous run can't silently leak into this one. This
        # must happen before model.train(), since the wandb callback module
        # checks SETTINGS["wandb"] once, the first time it is imported in this
        # process (inside ultralytics.utils.callbacks.add_integration_callbacks,
        # itself called from BaseTrainer.__init__ -> model.train()).
        self._check_wandb()
        settings.update({"wandb": self.run.wandb})

        if self.run.device is not None:
            config["device"] = self.run.device

        model = YOLO(config.pop("model"))
        if self.run.wandb:
            model.add_callback(
                "on_pretrain_routine_start", self._wandb_init_callback()
            )

        crop_size = config.pop("crop_size", None)
        devices = [d for d in str(config.get("device", "")).split(",") if d.strip()]
        batch = config.get("batch", 16)
        if len(devices) > 1 and (batch == -1 or 0 < batch < 1):
            config["batch"] = self._ddp_auto_batch(
                model, config, devices, data_yaml.parent, crop_size or config["imgsz"]
            )
        return model.train(**self._trainer_arg(model, crop_size), **config)

    def _resume(self) -> Any:
        """Resume training from ``config["resume"]`` (a ``last.pt`` checkpoint).

        Ultralytics' ``check_resume`` reads the checkpoint's own saved train
        args and only accepts overrides for the keys in
        ``_RESUME_ALLOWED_KEYS`` (e.g. ``data``/``project`` come from the
        checkpoint, not here). ``batch`` is dropped from the overrides when
        it's an AutoBatch sentinel (-1, or a fraction in (0, 1)): re-running
        AutoBatch would fight the checkpoint's already-resolved batch size,
        which we want to keep.
        """
        config = self.config
        resume_path = Path(config.pop("resume")).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        if self.run.device is not None:
            config["device"] = self.run.device

        self._check_wandb()
        settings.update({"wandb": self.run.wandb})

        model = YOLO(str(resume_path))
        if self.run.wandb:
            model.add_callback(
                "on_pretrain_routine_start", self._wandb_init_callback()
            )

        crop_size = config.pop("crop_size", None)
        batch = config.get("batch")
        is_autobatch = batch is not None and (batch == -1 or 0 < batch < 1)
        overrides = {
            key: config[key]
            for key in _RESUME_ALLOWED_KEYS
            if key in config and not (key == "batch" and is_autobatch)
        }
        return model.train(
            **self._trainer_arg(model, crop_size), resume=str(resume_path), **overrides
        )

    @staticmethod
    def _trainer_arg(model: YOLO, crop_size: int | None) -> dict:
        """``model.train`` kwargs selecting the native-crop trainer, or none (stock trainer) without ``crop_size``."""
        if crop_size is None:
            return {}
        return {"trainer": _native_crop_trainer(model.task_map[model.task]["trainer"], crop_size)}

    def _check_wandb(self) -> None:
        if self.run.wandb and importlib.util.find_spec("wandb") is None:
            raise RuntimeError(
                "run.wandb is True but the 'wandb' package isn't installed. "
                "Install it with `uv pip install wandb`."
            )

    def _wandb_init_callback(self):
        """Callback that starts the W&B run under a name mirroring the local path.

        Ultralytics' own callback names the W&B project after ``args.project``
        with "/" turned into "-", i.e. the whole absolute output path. It only
        calls ``wandb.init`` if no run is active, so starting one first (this
        callback is registered before the integration callbacks) wins.
        Callbacks are pickled into the DDP workers, so this also holds for
        multi-GPU runs. ``run.wandb_init`` overrides any of the ``wandb.init``
        kwargs (e.g. project/group/name for ``src/experiments.py``).
        """
        output_dir = self.run.output_dir.resolve()
        overrides = dict(self.run.wandb_init)

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
            kwargs = {
                "project": project,
                "name": save_dir.name,
                "id": latest_run.resolve().name.split("-", 2)[2]
                if resuming
                else f"{save_dir.name}_{datetime.now().astimezone():%Y%m%d_%H%M%S}",
                "resume": "allow" if resuming else None,
                "dir": str(save_dir),
                **overrides,
                "config": {**vars(trainer.args), **overrides.get("config", {})},
            }
            wandb.init(**kwargs)

        return callback

    def _ddp_auto_batch(
        self, model: YOLO, config: dict, devices: list[str], data_dir: Path, crop_size: int
    ) -> int:
        """AutoBatch for multi-GPU runs, which Ultralytics refuses to do itself.

        Profiles the per-GPU batch on the first device the same way the
        single-GPU trainer does (``DetectionTrainer.auto_batch``), then scales
        it by the number of GPUs since DDP splits the batch evenly across
        them. Profiles at ``crop_size`` (the real network input), not the
        native ``imgsz``, matching ``_native_crop_trainer.auto_batch``.
        """
        batch = config.get("batch", -1)
        stride = 32
        max_imgsz = math.ceil(crop_size * (1 + config.get("multi_scale", 0.0)) / stride) * stride

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
