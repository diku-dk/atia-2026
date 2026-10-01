"""EoMT evaluation: upstream's own validation (Lightning ``trainer.validate``) of a checkpoint on a dataset split.

The model, data module and trainer are rebuilt from the run's ``config.yaml`` through upstream's
``LightningCLI`` (``src/eomt/cli.py``), as in training, with the module swapped for ``EvalInstance``:
upstream's ``MaskClassificationInstance`` (with the GPU mask mAP of ``src/eomt/instance.py``) whose
``eval_step`` additionally times each stage and keeps the predictions as COCO results, which
``src/cropandweed_eval.py`` scores. Masked attention is
disabled (``masked_attn_enabled: False``, as in upstream's ``inference.ipynb``); after training its
annealing has already turned it off, so predictions are unchanged, and only the final block predicts.
Only mask (segmentation) metrics exist.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hotcoco import mask as coco_mask

from src.seg_dataset import read_names, read_split

from .cli import cli_main
from .instance import GpuMapInstance
from .metrics import GpuMaskMeanAveragePrecision
from .trainer import apply_dotted

# Scored detections per image: YOLO's max_det and src/cropandweed_eval.py's maxDets[2]
# (upstream's eval_top_k_instances, 100, stays for training-time validation).
_TOP_K = 300


class EvalInstance(GpuMapInstance):
    """``GpuMapInstance`` whose evaluation also times each stage and keeps its predictions.

    ``eval_step`` is upstream's, for the final block only, with CUDA-synchronised timers around
    preprocess (``resize_and_pad_imgs_instance_panoptic``), inference (the forward pass) and
    postprocess (mask upscaling, top-k, scoring and thresholding). The first batch runs one untimed
    warm-up forward. Its ``GpuMaskMeanAveragePrecision`` is built with ``class_metrics=True``; its
    results are kept in ``results``. Set ``stems`` (the loader's image stems, in order) before running.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stems: list[str] = []
        self.predictions: list[dict] = []
        self.images_seen = 0
        self.times_ms: list[tuple[float, float, float]] = []
        self.results: dict = {}

    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList(
            [GpuMaskMeanAveragePrecision(class_metrics=True) for _ in range(num_blocks)]
        )

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch

        t0 = self._now()
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        if batch_idx == 0:
            self(transformed_imgs)
            t0 = self._now()
            transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        t1 = self._now()
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)
        t2 = self._now()

        mask_logits = F.interpolate(mask_logits_per_layer[-1], self.img_size, mode="bilinear")
        mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)
        class_logits = class_logits_per_layer[-1]

        preds, targets_ = [], []
        for j in range(len(mask_logits)):
            scores = class_logits[j].softmax(dim=-1)[:, :-1]
            labels = (
                torch.arange(scores.shape[-1], device=self.device)
                .unsqueeze(0)
                .repeat(scores.shape[0], 1)
                .flatten(0, 1)
            )

            topk_scores, topk_indices = scores.flatten(0, 1).topk(
                self.eval_top_k_instances, sorted=False
            )
            labels = labels[topk_indices]

            topk_indices = topk_indices // scores.shape[-1]
            mask_logits[j] = mask_logits[j][topk_indices]

            masks = mask_logits[j] > 0
            mask_scores = (
                mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
            ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
            scores = topk_scores * mask_scores

            preds.append(dict(masks=masks, labels=labels, scores=scores))
            targets_.append(
                dict(
                    masks=targets[j]["masks"],
                    labels=targets[j]["labels"],
                    iscrowd=targets[j]["is_crowd"],
                )
            )
        t3 = self._now()
        n = len(imgs)
        self.times_ms.append(((t1 - t0) * 1e3 / n, (t2 - t1) * 1e3 / n, (t3 - t2) * 1e3 / n))

        self.update_metrics_instance(preds, targets_, len(self.metrics) - 1)
        for pred in preds:
            self.predictions += _to_coco_results(pred, self.stems[self.images_seen])
            self.images_seen += 1

    def on_validation_start(self):
        self.predictions, self.images_seen, self.times_ms = [], 0, []

    def on_validation_epoch_end(self):
        # compute() is cached, so upstream's logging below reuses it.
        self.results = {k: v.cpu() for k, v in self.metrics[-1].compute().items()}
        super().on_validation_epoch_end()


def _to_coco_results(pred: dict, stem: str) -> list[dict]:
    """One image's predictions as COCO results (RLE masks), with ``stem`` and 0-based ``category_id``; empty masks dropped."""
    keep = pred["masks"].flatten(1).any(1)
    # (N, W, H) C-contiguous, transposed on the GPU, is (H, W, N) Fortran-ordered: RLE's column-major
    # layout, so encoding needs no CPU copy.
    masks = pred["masks"][keep].transpose(1, 2).contiguous().cpu().numpy().transpose(2, 1, 0)
    if not masks.shape[-1]:
        return []
    rles = coco_mask.encode(masks)
    boxes = coco_mask.toBbox(rles)
    return [
        {
            "stem": stem,
            "category_id": int(label),
            "bbox": [round(float(v), 3) for v in box],
            "score": round(float(score), 5),
            "segmentation": {"size": rle["size"], "counts": rle["counts"].decode()},
        }
        for rle, box, label, score in zip(
            rles, boxes, pred["labels"][keep].tolist(), pred["scores"][keep].float().tolist()
        )
    ]


class EoMTEvaluator:
    """Evaluate one EoMT checkpoint (``<run>/weights/*.ckpt``) on ``variant`` of ``data_root``.

    ``val_args``: ``batch`` (the evaluation batch size; default: the run's), and any dotted
    LightningCLI keys (e.g. ``model.init_args.eval_top_k_instances``). The image size is the run's own
    (from its ``config.yaml``), like YOLO's ``imgsz`` from its checkpoint. Scratch output goes to
    ``log_dir/name/``.
    """

    csv_name = "eomt.csv"

    def __init__(self, checkpoint: Path, variant: str, data_root: Path, log_dir: Path, name: str, **val_args) -> None:
        self.checkpoint = Path(checkpoint)
        self.variant = variant
        self.dataset_dir = Path(data_root) / variant / "yolo" / "segmentation"
        self.log_dir = Path(log_dir)
        self.name = name
        self.val_args = val_args
        self.iou_types = ("segm",)

    def _config(self, device: str | None) -> dict:
        config = yaml.safe_load((self.checkpoint.parent.parent / "config.yaml").read_text())
        model_args = config["model"]["init_args"]
        config["model"]["class_path"] = f"{EvalInstance.__module__}.{EvalInstance.__qualname__}"
        model_args["network"]["init_args"]["masked_attn_enabled"] = False
        model_args["ckpt_path"] = None
        model_args["eval_top_k_instances"] = _TOP_K
        config["data"]["init_args"]["path"] = str(self.dataset_dir)
        trainer_args = config["trainer"]
        # Always the whole split, whatever the training run limited its validation to.
        trainer_args.pop("limit_val_batches", None)
        trainer_args["logger"] = False
        trainer_args["callbacks"] = [{"class_path": "lightning.pytorch.callbacks.TQDMProgressBar"}]
        trainer_args["default_root_dir"] = str((self.log_dir / self.name).resolve())
        if device == "cpu":
            trainer_args["accelerator"], trainer_args["devices"] = "cpu", 1
        else:
            trainer_args["devices"] = [int(device)] if device is not None else 1
        val_args = dict(self.val_args)
        if "batch" in val_args:
            config["data"]["init_args"]["batch_size"] = val_args.pop("batch")
        apply_dotted(config, val_args)
        return config

    def evaluate(self, split: str, device: str | None, csv_path: Path) -> tuple[list[dict], dict]:
        """Write EoMT's own per-class mask metrics to ``csv_path``; return the predictions as COCO results and a summary.

        Each result has ``stem`` instead of ``image_id`` and a 0-based ``category_id``. The summary is
        ``{"metrics": <the GPU metric's segm mAP50-95/mAP50/mAP75, named like Ultralytics' "metrics/mAP50-95(M)">,
        "speed_ms": <per-image preprocess/inference/postprocess time in ms>}``.
        """
        cli = cli_main(args=self._config(device), run=False)
        loader = cli.datamodule.eval_dataloader(split)
        module = cli.model
        module.stems = [img["stem"] for img in loader.dataset.images]
        # Upstream's checkpoints pickle the `network` hyperparameter, so they aren't weights-only.
        cli.trainer.validate(module, dataloaders=loader, ckpt_path=str(self.checkpoint), weights_only=False)

        results = module.results
        names = read_names(self.dataset_dir)
        images = read_split(self.dataset_dir, split)
        per_class = {int(c): (float(ap), float(ar)) for c, ap, ar in zip(
            results["classes"].reshape(-1), results["map_per_class"].reshape(-1), results["mar_100_per_class"].reshape(-1)
        )}
        pl.DataFrame([
            {
                "Class": name,
                "Images": sum(any(inst["cls"] == c for inst in img["instances"]) for img in images),
                "Instances": sum(inst["cls"] == c for img in images for inst in img["instances"]),
                # -1 for a class without ground truth (torchmetrics' convention).
                "mAP50-95(M)": round(per_class[c][0], 5) if c in per_class and per_class[c][0] >= 0 else None,
                "mAR100(M)": round(per_class[c][1], 5) if c in per_class and per_class[c][1] >= 0 else None,
            }
            for c, name in enumerate(names)
        ]).write_csv(csv_path)

        scratch = self.log_dir / self.name
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "predictions.json").write_text(json.dumps(module.predictions))
        times = np.array(module.times_ms).mean(axis=0)
        summary = {
            "metrics": {
                "metrics/mAP50(M)": float(results["map_50"]),
                "metrics/mAP75(M)": float(results["map_75"]),
                "metrics/mAP50-95(M)": float(results["map"]),
            },
            "speed_ms": dict(zip(("preprocess", "inference", "postprocess"), (float(t) for t in times))),
        }
        return module.predictions, summary
