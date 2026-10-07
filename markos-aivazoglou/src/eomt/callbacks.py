"""Lightning callbacks for the EoMT setup (our own code, referenced by class path from ``configs/seg.yaml``
or passed as ``--trainer.callbacks+=`` flags)."""

import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from hotcoco import mask as coco_mask
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torchvision.ops import masks_to_boxes


class AttnMaskAnnealing(Callback):
    """Set the module's masked-attention annealing steps from fractions of the run's total steps.

    Upstream configs give ``attn_mask_annealing_{start,end}_steps`` as absolute global steps of a
    COCO schedule (~7.4k steps per epoch), which on CropAndWeed's ~5.4k training images would never
    finish annealing. ``start``/``end`` hold one fraction per annealed block instead, converted with
    Lightning's own ``estimated_stepping_batches`` (so epochs, batch size, GPU count and
    ``limit_train_batches`` are all accounted for) before training starts. Masked attention is thus
    fully annealed (all ``attn_mask_prob_*`` 0) by the end of every run.
    """

    def __init__(self, start: list[float], end: list[float]):
        self.start = start
        self.end = end

    def on_fit_start(self, trainer, pl_module) -> None:
        total = trainer.estimated_stepping_batches
        pl_module.attn_mask_annealing_start_steps = [round(f * total) for f in self.start]
        pl_module.attn_mask_annealing_end_steps = [round(f * total) for f in self.end]
        print(
            f"Attention-mask annealing over {total} steps: start {pl_module.attn_mask_annealing_start_steps}, "
            f"end {pl_module.attn_mask_annealing_end_steps}"
        )


class CocoPredictionWriter(Callback):
    """Write a ``validate`` run's predictions as COCO results, plus its latency and logged metrics.

    Reads what the module's ``eval_step`` returns (the final block's predictions and its seconds per batch,
    see ``training/mask_classification_instance.py``). On ``validation_end`` it writes to ``output_dir``:
    ``predictions.json`` (COCO results with ``stem`` in place of ``image_id`` and 0-based ``category_id``,
    empty masks dropped, as ``src/cropandweed_eval.py`` scores them) and ``eval.json`` (``latency_ms``:
    preprocess + inference + postprocess per image, the first batch skipped as warm-up; ``metrics``: the
    metrics the module logged). For a single-device ``validate`` (the loader's order gives the stems).
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def on_validation_start(self, trainer, pl_module) -> None:
        self.predictions, self.images_seen, self.seconds, self.timed_images = [], 0, 0.0, 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        imgs = trainer.val_dataloaders.dataset.imgs
        for pred in outputs["preds"]:
            self.predictions += _to_coco_results(pred, Path(imgs[self.images_seen]).stem)
            self.images_seen += 1
        if batch_idx > 0:
            self.seconds += outputs["seconds"]
            self.timed_images += len(outputs["preds"])

    def on_validation_end(self, trainer, pl_module) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predictions.json").write_text(json.dumps(self.predictions))
        summary = {
            "latency_ms": {"total": self.seconds * 1e3 / max(self.timed_images, 1)},
            "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
        }
        (self.output_dir / "eval.json").write_text(json.dumps(summary, indent=2))


class PredictionPlotter(Callback):
    """After every validation, log the predictions next to the ground truth of the ``num_images`` val frames
    with the most labelled plants to the W&B run (``val/predictions``; rank 0, not in the sanity check, only
    with a ``WandbLogger``).

    Full frames, also when validation runs on tiles (``TiledDataset``), predicted with the module's own
    ``predict_instances`` (tiled with ``eval_tile_overlap``). Left: the predictions scoring at least
    ``score_thresh`` (mask, box, class and score); right: the ground truth (mask, box, class), with the
    Vegetation crowd regions as grey dashed boxes.
    """

    def __init__(self, num_images: int = 2, score_thresh: float = 0.5):
        self.num_images = num_images
        self.score_thresh = score_thresh

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero or not isinstance(trainer.logger, WandbLogger):
            return
        dataset = trainer.val_dataloaders.dataset
        dataset = getattr(dataset, "dataset", dataset)  # the frames behind a TiledDataset
        counts = [sum(not c for c in dataset.is_crowd_by_id.get(img, {}).values()) for img in dataset.imgs]
        indices = sorted(range(len(counts)), key=lambda i: -counts[i])[: self.num_images]
        names = yaml.safe_load((Path(trainer.datamodule.path) / "data.yaml").read_text())["names"]

        images = []
        for i in indices:
            img, target = dataset[i]
            with torch.no_grad(), trainer.precision_plugin.forward_context():
                (pred,) = pl_module.predict_instances([img.to(pl_module.device)])
            keep = pred["scores"] >= self.score_thresh
            images.append(_plot_side_by_side(
                img, {k: v[keep].cpu() for k, v in pred.items()}, target, names, Path(dataset.imgs[i]).stem
            ))
        trainer.logger.log_image("val/predictions", images, caption=[Path(dataset.imgs[i]).stem for i in indices])


def _plot_side_by_side(img: torch.Tensor, pred: dict, target: dict, names: dict, stem: str) -> Image.Image:
    """``img`` with ``pred`` (left) and ``target`` (right): masks, boxes, class names (and scores)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=100)
    crowd = target["is_crowd"]
    gt = {"masks": target["masks"][~crowd], "labels": target["labels"][~crowd]}
    for ax, title, instances in ((axes[0], "prediction", pred), (axes[1], "ground truth", gt)):
        ax.imshow(img.permute(1, 2, 0).numpy())
        overlay = np.zeros((*img.shape[-2:], 4))
        masks = instances["masks"].bool()
        nonempty = masks.flatten(1).any(1)
        masks, labels = masks[nonempty], instances["labels"][nonempty].tolist()
        scores = instances["scores"][nonempty].float().tolist() if "scores" in instances else [None] * len(labels)
        for mask, label, score, (x0, y0, x1, y1) in zip(masks, labels, scores, masks_to_boxes(masks).tolist()):
            color = colors(label)
            overlay[mask.numpy()] = (*color[:3], 0.45)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1))
            text = names[label] if score is None else f"{names[label]} {score:.2f}"
            ax.text(x0, y0 - 2, text, color="white", fontsize=6, bbox=dict(facecolor=color, pad=0.5, linewidth=0))
        ax.imshow(overlay)
        if "scores" not in instances:
            for x0, y0, x1, y1 in {tuple(b) for b in masks_to_boxes(target["masks"][crowd]).tolist()}:
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="grey", linestyle="--"))
        ax.set_title(f"{stem}: {title}, {len(labels)} instances")
        ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _rle_counts(masks: torch.Tensor) -> list[np.ndarray]:
    """COCO (uncompressed) RLE counts of ``(N, H, W)`` bool masks, computed on the masks' device.

    Copying the full-resolution masks to the CPU for ``coco_mask.encode`` (300 x 1088x1920 per image,
    ~630 MB) plus the CPU encoding took ~1.2 s per image; here only the run boundaries leave the GPU.
    COCO RLE runs over each mask in column-major order and starts with a run of zeros (0 if the first
    pixel is set).
    """
    n = masks.shape[0]
    flat = masks.transpose(1, 2).reshape(n, -1)
    rows, starts = (flat[:, 1:] != flat[:, :-1]).nonzero(as_tuple=True)  # (mask, start - 1) of each run after the first
    rows, starts, first = rows.cpu().numpy(), starts.cpu().numpy() + 1, flat[:, 0].cpu().numpy()
    counts = []
    for i, s in enumerate(np.split(starts, np.searchsorted(rows, np.arange(1, n)))):
        c = np.diff(np.concatenate(([0], s, [flat.shape[1]])))
        counts.append(np.concatenate(([0], c)) if first[i] else c)
    return counts


def _to_coco_results(pred: dict, stem: str) -> list[dict]:
    """One image's predictions as COCO results (RLE masks), with ``stem`` and 0-based ``category_id``; empty masks dropped."""
    keep = pred["masks"].flatten(1).any(1)
    masks = pred["masks"][keep]
    if not len(masks):
        return []
    height, width = masks.shape[-2:]
    # hotcoco compresses the uncompressed RLE: byte-identical to coco_mask.encode of the masks.
    rles = coco_mask.frPyObjects(
        [{"size": [height, width], "counts": c.tolist()} for c in _rle_counts(masks)], height, width
    )
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
