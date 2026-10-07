"""Mask mAP under the CropAndWeed protocol (``src/cropandweed_eval.py``), with the mask IoU on the GPU.

As Ultralytics' validator (``ultralytics/utils/metrics.py:mask_iou``), the IoU of every prediction with every
ground-truth mask of an image is one matmul of the flattened masks on the GPU; only the small ``[P, G]`` IoU
matrix goes to the CPU. Matching and accumulation then follow pycocotools' ``COCOeval.evaluateImg`` and
``accumulate`` (which hotcoco reproduces), with ``cropandweed_eval``'s parameters:

- per class, detections in score order each take the unmatched ground truth with the highest IoU at or above
  the threshold, preferring non-ignored ground truth; a crowd (Vegetation) region is matched by intersection
  over the detection's area, can be matched any number of times, and a detection matching it is ignored;
- ground truth is ignored if crowd or its **bbox** area (``w * h``, as ``cropandweed_eval`` sets ``area``) is
  outside the area range, and an unmatched detection if its bbox area is (hotcoco's ``loadRes`` sets ``area``
  to the result's ``bbox`` w * h, and ``CocoPredictionWriter`` writes the mask's box);
- ``AREA_RANGES`` (all from 16^2, small, medium, large); 101-point interpolated AP averaged over IoU
  0.50:0.05:0.95 and over the classes with ground truth; every prediction of an image is scored (the
  module's top-k is at most ``MAX_DETS``).

The ``torchmetrics.Metric`` base class only gathers the per-detection states across DDP ranks.
"""

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from ..cropandweed_eval import AREA_RANGES

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _bbox_areas(masks: torch.Tensor) -> torch.Tensor:
    """``(N, H, W)`` bool masks -> their bounding boxes' ``w * h`` in pixels (COCO's ``toBbox``; 0 if empty)."""
    extents = []
    for occupied in (masks.any(2), masks.any(1)):  # rows, columns
        first = occupied.int().argmax(1)
        last = occupied.shape[1] - 1 - occupied.flip(1).int().argmax(1)
        extents.append((last - first + 1) * occupied.any(1))
    return extents[0] * extents[1]


class MaskAP(Metric):
    """COCO-style mask mAP of ``update(preds, targets)`` batches, as torchmetrics' ``MeanAveragePrecision``
    takes them (preds: ``masks``, ``scores``, ``labels``; targets: ``masks``, ``labels``, ``iscrowd``), returning
    its keys (``map``, ``map_50``, ``map_75``, ``map_small``, ``map_medium``, ``map_large``; -1 without
    ground truth).

    ``area_scale`` multiplies the area ranges, for masks downsampled by ``sqrt(area_scale)``.
    """

    full_state_update = False

    def __init__(self, num_classes: int, area_scale: float = 1.0, chunk_size: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.area_ranges = np.array(AREA_RANGES, dtype=float) * area_scale
        self.chunk_size = chunk_size
        for name in ("scores", "labels", "matched", "ignored"):
            self.add_state(name, default=[], dist_reduce_fx="cat")
        self.add_state(
            "num_gt",
            default=torch.zeros(num_classes, len(AREA_RANGES), dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        for pred, target in zip(preds, targets):
            self._update_image(pred, target)

    def _update_image(self, pred: dict, target: dict) -> None:
        nonempty = pred["masks"].flatten(1).any(1)  # empty masks aren't predictions (CocoPredictionWriter)
        dt_masks = pred["masks"][nonempty].flatten(1)
        gt_masks = target["masks"].flatten(1)
        dt_area = dt_masks.sum(1)
        inter = torch.zeros(len(dt_masks), len(gt_masks), device=dt_masks.device)
        if len(dt_masks) and len(gt_masks):
            gt_bf16 = gt_masks.to(torch.bfloat16).T  # 0/1 exact; float32 accumulation exact to 2^24 px
            inter = torch.cat([
                torch.mm(chunk.to(torch.bfloat16), gt_bf16, out_dtype=torch.float32)
                for chunk in dt_masks.split(self.chunk_size)
            ])
        crowd = target["iscrowd"].bool()
        union = dt_area[:, None] + gt_masks.sum(1)[None] - inter
        iou = torch.where(crowd[None], inter / dt_area[:, None], inter / union)

        iou, crowd = iou.cpu().numpy(), crowd.cpu().numpy()
        dt_area = _bbox_areas(pred["masks"][nonempty]).cpu().numpy()
        gt_area = _bbox_areas(target["masks"]).cpu().numpy()
        dt_scores = pred["scores"][nonempty].float().cpu().numpy()
        dt_labels = pred["labels"][nonempty].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        lo, hi = self.area_ranges[:, :1], self.area_ranges[:, 1:]  # [A, 1]

        scores, labels, matched, ignored = [], [], [], []
        num_gt = np.zeros(tuple(self.num_gt.shape), dtype=np.int64)
        for c in np.union1d(dt_labels, gt_labels):
            d = np.flatnonzero(dt_labels == c)
            d = d[np.argsort(-dt_scores[d], kind="mergesort")]
            g = np.flatnonzero(gt_labels == c)
            g_crowd = crowd[g]
            g_ignore = g_crowd | (gt_area[g] < lo) | (gt_area[g] > hi)  # [A, G]
            num_gt[c] += (~g_ignore).sum(1)
            d_matched, d_ignored = self._match(iou[np.ix_(d, g)], g_crowd, g_ignore)  # [D, T, A]
            outside = (dt_area[d, None] < lo.T) | (dt_area[d, None] > hi.T)  # [D, A]
            d_ignored |= ~d_matched & outside[:, None]
            scores.append(dt_scores[d])
            labels.append(np.full(len(d), c))
            matched.append(d_matched)
            ignored.append(d_ignored)

        num_t, num_a = len(IOU_THRESHOLDS), len(self.area_ranges)
        device = self.num_gt.device
        self.scores.append(torch.from_numpy(np.concatenate(scores, dtype=np.float32) if scores else np.zeros(0, np.float32)).to(device))
        self.labels.append(torch.from_numpy(np.concatenate(labels).astype(np.int64) if labels else np.zeros(0, np.int64)).to(device))
        for state, parts in ((self.matched, matched), (self.ignored, ignored)):
            stacked = np.concatenate(parts) if parts else np.zeros((0, num_t, num_a), bool)
            state.append(torch.from_numpy(stacked.astype(np.uint8)).to(device))
        self.num_gt += torch.from_numpy(num_gt).to(device)

    @staticmethod
    def _match(iou: np.ndarray, crowd: np.ndarray, gt_ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``evaluateImg``'s greedy matching of one class's score-sorted detections (``iou`` rows) to its ground
        truth, for every IoU threshold and area range at once: ``[D, T, A]`` matched, and matched to ignored."""
        num_d, num_g = iou.shape
        num_t, num_a = len(IOU_THRESHOLDS), len(gt_ignore)
        matched = np.zeros((num_d, num_t, num_a), bool)
        ignored = np.zeros((num_d, num_t, num_a), bool)
        if not num_g:
            return matched, ignored
        taken = np.zeros((num_t, num_a, num_g), bool)
        gt_ignore = np.broadcast_to(gt_ignore, (num_t, num_a, num_g))
        thresholds = np.minimum(IOU_THRESHOLDS, 1 - 1e-10)[:, None, None]
        for k in range(num_d):
            ok = (iou[k] >= thresholds) & (~taken | crowd)
            regular = ok & ~gt_ignore
            candidates = np.where(regular.any(-1, keepdims=True), regular, ok)
            # The best IoU; on ties the later ground truth, as evaluateImg's `if iou < best: continue`.
            best = num_g - 1 - np.where(candidates, iou[k], -1.0)[..., ::-1].argmax(-1)
            hit = candidates.any(-1)
            taken |= hit[..., None] & (np.arange(num_g) == best[..., None])
            matched[k] = hit
            ignored[k] = hit & np.take_along_axis(gt_ignore, best[..., None], -1)[..., 0]
        return matched, ignored

    def compute(self) -> dict[str, torch.Tensor]:
        scores = dim_zero_cat(self.scores).cpu().numpy() if len(self.scores) else np.zeros(0)
        labels = dim_zero_cat(self.labels).cpu().numpy() if len(self.labels) else np.zeros(0)
        shape = (0, len(IOU_THRESHOLDS), len(self.area_ranges))
        matched = dim_zero_cat(self.matched).cpu().numpy().astype(bool) if len(self.matched) else np.zeros(shape, bool)
        ignored = dim_zero_cat(self.ignored).cpu().numpy().astype(bool) if len(self.ignored) else np.zeros(shape, bool)
        num_gt = self.num_gt.cpu().numpy()

        ap = np.full((self.num_classes, *shape[1:]), -1.0)  # [C, T, A]
        for c in range(self.num_classes):
            sel = labels == c
            order = np.argsort(-scores[sel], kind="mergesort")
            m, ig = matched[sel][order], ignored[sel][order]
            tp = np.cumsum(m & ~ig, 0, dtype=float)
            fp = np.cumsum(~m & ~ig, 0, dtype=float)
            for a in range(shape[2]):
                if not num_gt[c, a]:
                    continue
                if not len(tp):
                    ap[c, :, a] = 0.0
                    continue
                for t in range(shape[1]):
                    recall = tp[:, t, a] / num_gt[c, a]
                    precision = tp[:, t, a] / (tp[:, t, a] + fp[:, t, a] + np.spacing(1))
                    precision = np.maximum.accumulate(precision[::-1])[::-1]
                    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
                    q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
                    ap[c, t, a] = q.mean()

        def mean(values: np.ndarray) -> torch.Tensor:  # over the classes with ground truth
            valid = values[values > -1]
            return torch.tensor(valid.mean() if len(valid) else -1.0, dtype=torch.float32)

        return {
            "map": mean(ap[:, :, 0].mean(1)),
            "map_50": mean(ap[:, 0, 0]),
            "map_75": mean(ap[:, 5, 0]),
            "map_small": mean(ap[:, :, 1].mean(1)),
            "map_medium": mean(ap[:, :, 2].mean(1)),
            "map_large": mean(ap[:, :, 3].mean(1)),
        }
