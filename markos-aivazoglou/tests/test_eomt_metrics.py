"""``GpuMaskMeanAveragePrecision`` (``src/eomt/metrics.py``) against stock pycocotools ``COCOeval`` on RLEs and
torchmetrics' segm ``MeanAveragePrecision`` (which it replaces), on random toy masks (run on the CPU).

torchmetrics' segm metric skips images without ground truth, so their false positives never count; COCO
(and ``GpuMaskMeanAveragePrecision``) keeps them. The torchmetrics comparison therefore uses images that
all have ground truth."""

import unittest

import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection import MeanAveragePrecision

from src.eomt.metrics import GpuMaskMeanAveragePrecision

_H, _W, _CLASSES = 48, 64, 3


def _boxes_to_masks(boxes: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.arange(_H)[None, :, None], torch.arange(_W)[None, None, :]
    x0, y0, x1, y1 = (boxes[:, i, None, None] for i in range(4))
    return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)


def _random_boxes(gen: torch.Generator, n: int, min_size: int = 2, max_size: int = 30) -> torch.Tensor:
    size = torch.randint(min_size, max_size, (n, 2), generator=gen)
    x0 = (torch.rand(n, generator=gen) * (_W - size[:, 0])).long()
    y0 = (torch.rand(n, generator=gen) * (_H - size[:, 1])).long()
    return torch.stack([x0, y0, x0 + size[:, 0], y0 + size[:, 1]], 1)


def _toy_batch(seed: int = 0, empty_gt: bool = True) -> tuple[list[dict], list[dict]]:
    """Images of GT boxes, detected as jittered copies (some relabelled) plus random false positives;
    one image without detections and, with ``empty_gt``, one without ground truth."""
    gen = torch.Generator().manual_seed(seed)
    preds, targets = [], []
    for i in range(8):
        n_gt = 0 if i == 0 and empty_gt else int(torch.randint(1, 12, (1,), generator=gen))
        gt_boxes = _random_boxes(gen, n_gt)
        gt_labels = torch.randint(0, _CLASSES, (n_gt,), generator=gen)
        jitter = torch.randint(-3, 4, (n_gt, 4), generator=gen)
        det_boxes = torch.cat([(gt_boxes + jitter).clamp_min(0), _random_boxes(gen, 5)])
        det_labels = torch.cat([gt_labels, torch.randint(0, _CLASSES, (5,), generator=gen)])
        flip = torch.rand(len(det_labels), generator=gen) < 0.2
        det_labels[flip] = (det_labels[flip] + 1) % _CLASSES
        if i == 1:
            det_boxes, det_labels = det_boxes[:0], det_labels[:0]
        masks = _boxes_to_masks(det_boxes)
        keep = masks.flatten(1).any(1)
        preds.append({
            "masks": masks[keep],
            "labels": det_labels[keep],
            "scores": torch.rand(int(keep.sum()), generator=gen),
        })
        targets.append({
            "masks": _boxes_to_masks(gt_boxes),
            "labels": gt_labels,
            "iscrowd": torch.zeros(n_gt, dtype=torch.bool),
        })
    return preds, targets


def _rles(masks: torch.Tensor) -> list[dict]:
    return coco_mask.encode(np.asfortranarray(masks.permute(1, 2, 0).numpy().astype(np.uint8))) if len(masks) else []


def _pycocotools(preds: list[dict], targets: list[dict], class_metrics: bool) -> dict:
    """Stock COCOeval over every image, at the metric's thresholds: the 12 stats and, per class, AP and
    AR@100 (torchmetrics' way)."""
    images, gts, dts = [], [], []
    for img_id, (pred, target) in enumerate(zip(preds, targets), 1):
        images.append({"id": img_id, "height": _H, "width": _W})
        for rle, label in zip(_rles(target["masks"]), target["labels"].tolist()):
            gts.append({"id": len(gts) + 1, "image_id": img_id, "category_id": label, "segmentation": rle,
                        "area": float(coco_mask.area(rle)), "iscrowd": 0})
        for rle, label, score in zip(_rles(pred["masks"]), pred["labels"].tolist(), pred["scores"].tolist()):
            dts.append({"image_id": img_id, "category_id": label, "segmentation": rle, "score": score})
    classes = sorted({a["category_id"] for a in gts + dts})
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": gts, "categories": [{"id": c} for c in classes]}
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dts)

    def stats(cat_ids):
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.params.catIds = cat_ids
        coco_eval.params.iouThrs = np.array(GpuMaskMeanAveragePrecision().iou_thresholds)
        coco_eval.params.recThrs = np.array(GpuMaskMeanAveragePrecision().rec_thresholds)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    names = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large",
             "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large"]
    result = dict(zip(names, (torch.tensor(v) for v in stats(classes))))
    if class_metrics:
        per_class = [stats([c]) for c in classes]
        result["map_per_class"] = torch.tensor([s[0] for s in per_class])
        result["mar_100_per_class"] = torch.tensor([s[8] for s in per_class])
    return result


class TestGpuMaskMeanAveragePrecision(unittest.TestCase):
    def _assert_matches(self, expected: dict, actual: dict, **context):
        self.assertGreater(float(expected["map"]), 0)
        for key, value in expected.items():
            with self.subTest(key=key, **context):
                self.assertIn(key, actual)
                torch.testing.assert_close(
                    actual[key].reshape(-1).double(), value.reshape(-1).double(), rtol=0, atol=1e-6
                )

    def test_matches_pycocotools(self):
        for seed in range(3):
            preds, targets = _toy_batch(seed)
            ours = GpuMaskMeanAveragePrecision(class_metrics=True)
            ours.update(preds, targets)
            self._assert_matches(_pycocotools(preds, targets, class_metrics=True), ours.compute(), seed=seed)

    def test_matches_torchmetrics(self):
        for seed in range(3):
            preds, targets = _toy_batch(seed, empty_gt=False)
            ours = GpuMaskMeanAveragePrecision(class_metrics=True)
            reference = MeanAveragePrecision(iou_type="segm", class_metrics=True)
            # Two updates, as over two validation batches.
            for metric in (ours, reference):
                metric.update(preds[:4], targets[:4])
                metric.update(preds[4:], targets[4:])
            self._assert_matches(reference.compute(), ours.compute(), seed=seed)

    def test_rejects_crowds(self):
        preds, targets = _toy_batch()
        targets[2]["iscrowd"][0] = True
        with self.assertRaises(ValueError):
            GpuMaskMeanAveragePrecision().update(preds, targets)


if __name__ == "__main__":
    unittest.main()
