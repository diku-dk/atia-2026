"""Upstream's instance-segmentation module with the mask mAP computed on the GPU."""

import torch.nn as nn

from .metrics import GpuMaskMeanAveragePrecision
from .training.mask_classification_instance import MaskClassificationInstance


class GpuMapInstance(MaskClassificationInstance):
    """``MaskClassificationInstance`` whose metrics are ``GpuMaskMeanAveragePrecision`` (``src/eomt/metrics.py``)
    instead of torchmetrics' CPU/RLE-based ``MeanAveragePrecision(iou_type="segm")``: same numbers, far faster."""

    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList([GpuMaskMeanAveragePrecision() for _ in range(num_blocks)])
