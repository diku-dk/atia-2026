# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
#
# Copied from tue-mps/eomt@7bd19dd (training/mask_classification_instance.py). Changes: imports made
# package-relative; the mAP (src/eomt/mask_ap.py:MaskAP) is computed on the final layer only (upstream: every
# masked-attention block too), optionally on masks downscaled by `metric_mask_scale` (its area ranges scaled
# to match), its postprocessing moved into `predict_instances` (also used by
# src/eomt/callbacks.py:PredictionPlotter); `eval_step` returns the predictions and the CUDA-synchronised
# seconds from before preprocessing to after postprocessing (excluding the metric update), for
# src/eomt/callbacks.py:CocoPredictionWriter; with `eval_tile_overlap` set (our addition, upstream has no
# tiled instance inference), it runs on native-resolution img_size tiles of each image instead of the
# image resized to fit img_size, and `merge_tile_preds` pastes the tiles' predictions back into the image.
# ---------------------------------------------------------------


import math
import time
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from .mask_classification_loss import MaskClassificationLoss
from .lightning_module import LightningModule


class MaskClassificationInstance(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        eval_tile_overlap: Optional[int] = None,
        metric_mask_scale: float = 1.0,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes: List[int] = []
        self.eval_top_k_instances = eval_top_k_instances
        self.eval_tile_overlap = eval_tile_overlap
        self.metric_mask_scale = metric_mask_scale

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_instance(1, area_scale=metric_mask_scale**2)  # the final layer only

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        preds = self.predict_instances(imgs)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start

        preds_ = [
            dict(pred, masks=self.scale_metric_masks(pred["masks"])) for pred in preds
        ]
        targets_ = [
            dict(
                masks=self.scale_metric_masks(target["masks"]),
                labels=target["labels"],
                iscrowd=target["is_crowd"],
            )
            for target in targets
        ]
        self.update_metrics_instance(preds_, targets_, 0)

        return {"preds": preds, "seconds": seconds}

    @torch.compiler.disable
    def scale_metric_masks(self, masks):
        """``(N, H, W)`` bool masks downscaled by ``metric_mask_scale``: pixels at least half covered."""
        if self.metric_mask_scale == 1.0:
            return masks
        size = [round(s * self.metric_mask_scale) for s in masks.shape[-2:]]
        if not len(masks):
            return masks.new_zeros((0, *size))
        return F.interpolate(masks[None].float(), size, mode="area")[0] >= 0.5

    def predict_instances(self, imgs):
        """The final layer's top-k instances (masks at each image's size, labels, scores) per image."""
        img_sizes = [img.shape[-2:] for img in imgs]
        if self.eval_tile_overlap is not None:
            imgs, origins = self.tile_imgs(imgs)
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)
        class_logits = class_logits_per_layer[-1]

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], self.img_size, mode="bilinear"
        )
        mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, [img.shape[-2:] for img in imgs]
        )

        preds = []
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
            mask_scores = (mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)).sum(
                1
            ) / (masks.flatten(1).sum(1) + 1e-6)
            scores = topk_scores * mask_scores

            preds.append(
                dict(
                    masks=masks,
                    labels=labels,
                    scores=scores,
                )
            )

        if self.eval_tile_overlap is not None:
            preds = self.merge_tile_preds(preds, origins, img_sizes)
        return preds

    def tile_starts(self, length: int, tile: int) -> list[int]:
        """Evenly spread starts of ``tile``-long windows covering ``length``, neighbours overlapping by at
        least ``eval_tile_overlap``."""
        if length <= tile:
            return [0]
        n = math.ceil(
            (length - self.eval_tile_overlap) / (tile - self.eval_tile_overlap)
        )
        return [round(k * (length - tile) / (n - 1)) for k in range(n)]

    @staticmethod
    def owned_ranges(starts: list[int], tile: int) -> list[tuple[float, float]]:
        """Each window's share of the axis: neighbours split their overlap at its midpoint."""
        bounds = (
            [-math.inf]
            + [(a + tile + b) / 2 for a, b in zip(starts, starts[1:])]
            + [math.inf]
        )
        return list(zip(bounds, bounds[1:]))

    def tile_imgs(self, imgs):
        """Native-resolution ``img_size`` tiles of each image (a side that fits is one tile), with each
        tile's (image index, y, x, owned y range, owned x range)."""
        tiles, origins = [], []
        for i, img in enumerate(imgs):
            (h, w), (th, tw) = img.shape[-2:], self.img_size
            th, tw = min(th, h), min(tw, w)
            ys, xs = self.tile_starts(h, th), self.tile_starts(w, tw)
            for y, y_range in zip(ys, self.owned_ranges(ys, th)):
                for x, x_range in zip(xs, self.owned_ranges(xs, tw)):
                    tiles.append(img[:, y : y + th, x : x + tw])
                    origins.append((i, y, x, y_range, x_range))
        return tiles, origins

    @torch.compiler.disable
    def merge_tile_preds(self, tile_preds, origins, img_sizes):
        """Paste each tile's predictions into its image, keeping an instance only from the tile that owns
        its box centre (so one up to the overlap wide comes from a tile holding it whole, once), then the
        image's top ``eval_top_k_instances``. Empty masks are dropped."""
        parts = [[] for _ in img_sizes]
        for pred, (i, y, x, (y_lo, y_hi), (x_lo, x_hi)) in zip(tile_preds, origins):
            pred = {k: v[pred["masks"].flatten(1).any(1)] for k, v in pred.items()}
            boxes = masks_to_boxes(pred["masks"])
            cx = x + (boxes[:, 0] + boxes[:, 2]) / 2
            cy = y + (boxes[:, 1] + boxes[:, 3]) / 2
            own = (cx >= x_lo) & (cx < x_hi) & (cy >= y_lo) & (cy < y_hi)
            pred = {k: v[own] for k, v in pred.items()}
            (h, w), (th, tw) = img_sizes[i], pred["masks"].shape[-2:]
            pred["masks"] = F.pad(pred["masks"], (x, w - x - tw, y, h - y - th))
            parts[i].append(pred)

        merged = []
        for image_parts in parts:
            pred = {k: torch.cat([p[k] for p in image_parts]) for k in image_parts[0]}
            top = (
                pred["scores"]
                .topk(min(self.eval_top_k_instances, len(pred["scores"])))
                .indices
            )
            merged.append({k: v[top] for k, v in pred.items()})
        return merged

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
