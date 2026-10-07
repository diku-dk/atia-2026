# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
#
# Copied from tue-mps/eomt@7bd19dd (datasets/transforms.py). Changes: imports made package-relative;
# `scale_range=None` skips the scale jitter (crops at native scale); `min_visible_fraction` also drops
# instances with less than that fraction of their mask area left after the crop (upstream drops only
# those with nothing left, and retries an image left with no instance).
# ---------------------------------------------------------------

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor
from torch import nn, Tensor
from typing import Any, Optional, Union


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: Optional[tuple[float, float]],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        min_visible_fraction: float = 0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0
        self.min_visible_fraction = min_visible_fraction

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = (
            T.ScaleJitter(target_size=img_size, scale_range=scale_range)
            if scale_range is not None
            else None
        )
        self.random_crop = T.RandomCrop(img_size)

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        if self.scale_jitter is not None:
            img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        area = target["masks"].flatten(1).sum(1)
        img, target = self.random_crop(img, target)

        visible = target["masks"].flatten(1).sum(1)
        valid = (visible > 0) & (visible >= self.min_visible_fraction * area)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target
