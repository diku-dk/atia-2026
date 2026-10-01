"""COCO mask mAP with the mask IoU computed on the GPU, a drop-in for upstream's torchmetrics metric.

From ``~/pmt`` (``image/training/metrics.py``, ``GpuMaskMeanAveragePrecision``), with ``class_metrics``
added for ``src/eomt/evaluator.py``'s per-class CSV, torchmetrics' thresholds, and the intersections as
a matmul instead of a loop over the ground truths (~50x faster at 1920x1088).
"""

import contextlib
import io
from collections import defaultdict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics import Metric


class GpuMaskMeanAveragePrecision(Metric):
    """COCO mask mAP whose IoU is computed on the GPU.

    torchmetrics' ``MeanAveragePrecision(iou_type="segm")`` keeps every mask as an RLE, encoded and
    matched by pycocotools on the CPU whatever the metric's device. At 1920x1088 that dominates
    validation. Mask IoU is only a pairwise intersection count, so this computes it on the GPU at
    ``update()``, as one matmul of the 0/1 masks (exact integer counts, so identical to pycocotools'),
    and keeps just the small per-image IoU matrices.

    Matching, accumulation and summarisation still run through stock pycocotools ``COCOeval``, with
    torchmetrics' IoU and recall thresholds (float32 ``linspace``, so an IoU exactly on a threshold
    matches as there); only the source of the IoUs changes, so the results (keys and values) are
    torchmetrics'. With ``class_metrics``, also ``classes``, ``map_per_class`` and ``mar_<max>_per_class``.
    One deliberate difference: torchmetrics' segm metric skips images without ground truth, so their
    false positives never count; like COCO (and ``src/cropandweed_eval.py``), this keeps them.

    Crowd regions are not supported (COCO scores them with intersection-over-detection): ``update()``
    raises rather than report a wrong number.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, max_detection_thresholds=None, class_metrics: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_detection_thresholds = max_detection_thresholds or [1, 10, 100]
        self.class_metrics = class_metrics
        self.iou_thresholds = torch.linspace(0.5, 0.95, 10).tolist()
        self.rec_thresholds = torch.linspace(0.0, 1.00, 101).tolist()

        # Every state is a list of 1-D tensors, one entry per image, so that the DDP gather can pad
        # and concatenate them. The IoU matrix is stored flat with its shape alongside for the same reason.
        for name in ("ious", "iou_shapes", "det_labels", "det_scores", "det_areas", "gt_labels", "gt_areas"):
            self.add_state(name, default=[], dist_reduce_fx=None)

    @staticmethod
    def mask_iou(det: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Pairwise IoU between flattened boolean masks, (n_det, n_gt), float64."""
        n_det, n_gt = det.shape[0], gt.shape[0]
        if n_det == 0 or n_gt == 0:
            return torch.zeros((n_det, n_gt), dtype=torch.float64, device=det.device)

        if det.shape[1] > 2**24:
            raise ValueError("masks over 2^24 pixels: float32 intersection counts would no longer be exact")
        det_area = det.sum(1, dtype=torch.int64)[:, None]
        gt_area = gt.sum(1, dtype=torch.int64)[None, :]
        # Every product is 0 or 1 (exact even in TF32/bf16) and float32 sums integers exactly up to 2^24,
        # so this is the exact count. Outside autocast, which would compute it in float16.
        with torch.autocast(det.device.type, enabled=False):
            inter = (det.float() @ gt.float().T).long()
        union = (det_area + gt_area - inter).clamp_min(1)
        return inter.double() / union.double()

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        for pred, target in zip(preds, targets):
            iscrowd = target.get("iscrowd")
            if iscrowd is not None and bool(iscrowd.any()):
                raise ValueError("GpuMaskMeanAveragePrecision does not support crowd ground truth")

            det_masks = pred["masks"].flatten(1)
            gt_masks = target["masks"].flatten(1).to(det_masks.device)
            iou = self.mask_iou(det_masks, gt_masks)

            self.ious.append(iou.flatten())
            self.iou_shapes.append(torch.tensor(iou.shape, device=self.device, dtype=torch.long))
            self.det_labels.append(pred["labels"].to(self.device))
            self.det_scores.append(pred["scores"].to(self.device))
            self.det_areas.append(det_masks.sum(1, dtype=torch.float64))
            self.gt_labels.append(target["labels"].to(self.device))
            self.gt_areas.append(gt_masks.sum(1, dtype=torch.float64))

    def compute(self) -> dict:
        images, gt_anns, dt_anns, iou_by_img = [], [], [], {}
        ann_id = 1
        for i, shape in enumerate(self.iou_shapes):
            img_id = i + 1
            n_det, n_gt = int(shape[0]), int(shape[1])
            images.append({"id": img_id})
            iou_by_img[img_id] = self.ious[i].reshape(n_det, n_gt).cpu().numpy()

            gt_labels, gt_areas = self.gt_labels[i].tolist(), self.gt_areas[i].tolist()
            for col in range(n_gt):
                gt_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": int(gt_labels[col]),
                    "area": float(gt_areas[col]), "iscrowd": 0,
                    "_col": col,  # which column of this image's IoU matrix this annotation is
                })
                ann_id += 1

            det_labels, det_scores = self.det_labels[i].tolist(), self.det_scores[i].tolist()
            det_areas = self.det_areas[i].tolist()
            for row in range(n_det):
                dt_anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": int(det_labels[row]),
                    "area": float(det_areas[row]), "score": float(det_scores[row]), "iscrowd": 0,
                    "_row": row,
                })
                ann_id += 1

        class_ids = sorted({a["category_id"] for a in gt_anns + dt_anns})
        if not images or not class_ids:
            return self._to_dict([-1.0] * 12, class_ids, [], [])

        coco_gt, coco_dt = COCO(), COCO()
        for coco, anns in ((coco_gt, gt_anns), (coco_dt, dt_anns)):
            coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in class_ids]}
            with contextlib.redirect_stdout(io.StringIO()):
                coco.createIndex()

        coco_eval = _PrecomputedIoUCOCOeval(coco_gt, coco_dt, iou_by_img)
        coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
        coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
        coco_eval.params.maxDets = self.max_detection_thresholds
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        map_per_class, mar_per_class = [], []
        if self.class_metrics:
            # COCOeval.summarize's stats[0] and stats[8] (area "all", largest maxDets), one class at a time.
            precision, recall = coco_eval.eval["precision"], coco_eval.eval["recall"]
            for k in range(len(class_ids)):
                p, r = precision[:, :, k, 0, -1], recall[:, k, 0, -1]
                map_per_class.append(float(p[p > -1].mean()) if (p > -1).any() else -1.0)
                mar_per_class.append(float(r[r > -1].mean()) if (r > -1).any() else -1.0)
        return self._to_dict(list(coco_eval.stats), class_ids, map_per_class, mar_per_class)

    def _to_dict(self, stats: list, class_ids: list, map_per_class: list, mar_per_class: list) -> dict:
        """torchmetrics' ``MeanAveragePrecision`` keys and dtypes."""
        mdt = self.max_detection_thresholds
        names = [
            "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
            f"mar_{mdt[0]}", f"mar_{mdt[1]}", f"mar_{mdt[2]}", "mar_small", "mar_medium", "mar_large",
        ]
        result = {name: torch.tensor(stats[i], dtype=torch.float32) for i, name in enumerate(names)}
        result["map_per_class"] = torch.tensor(map_per_class if self.class_metrics else -1.0, dtype=torch.float32)
        result[f"mar_{mdt[2]}_per_class"] = torch.tensor(
            mar_per_class if self.class_metrics else -1.0, dtype=torch.float32
        )
        result["classes"] = torch.tensor(class_ids, dtype=torch.int32)
        return result


class _PrecomputedIoUCOCOeval(COCOeval):
    """``COCOeval`` that reads IoUs computed elsewhere; matching, accumulation and summary are stock."""

    def __init__(self, coco_gt, coco_dt, iou_by_img):
        super().__init__(coco_gt, coco_dt, iouType="segm")
        self.iou_by_img = iou_by_img
        self.params.imgIds = sorted(coco_gt.getImgIds())
        self.params.catIds = sorted(coco_gt.getCatIds())

    def _prepare(self):
        # COCOeval._prepare minus the segmentation -> RLE conversion: there are no masks here.
        p = self.params
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        for gt in gts:
            gt["ignore"] = gt.get("ignore", 0)
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)
        self.eval = {}

    def computeIoU(self, imgId, catId):
        # COCOeval.computeIoU's selection of detections, then the precomputed matrix instead of maskUtils.iou.
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]
        if len(gt) == 0 or len(dt) == 0:
            return []
        return self.iou_by_img[imgId][np.ix_([d["_row"] for d in dt], [g["_col"] for g in gt])]
