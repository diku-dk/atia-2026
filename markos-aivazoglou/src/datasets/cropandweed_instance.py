"""CropAndWeed instance segmentation for EoMT, following upstream's ``datasets/coco_instance.py``.

``path`` is a split seed's dataset dir (``data/seed<N>``): ``dataset.py`` reads ``images/<split>/`` and
the COCO ground truth ``annotations/<split>.json``. ``target_parser`` is ``COCOInstance``'s without its
COCO class mapping (our ``category_id`` is already 0-based). ``val_split`` is the split ``validate``
runs on (``test`` for the final evaluation). Images without instances are dropped from ``train`` only
(``check_empty_targets``): val/test keep every image, so the evaluation sees the whole split.
The Vegetation boxes of val/test are ``iscrowd`` annotations: they reach the targets as ``is_crowd``,
so the mAP ignores detections matching them (upstream's train ``Transforms`` drop crowd targets).

``scale_range=None`` trains on crops at native scale and ``min_visible_fraction`` drops instances mostly cut off by
the crop (both ``transforms.py`` edits). ``val_batch_size`` (default: ``batch_size``) is the per-GPU batch of
the val loader. ``val_tile_size`` validates on ``TiledDataset``: a fixed grid of tiles of
every frame, each scored as its own image, with instances mostly cut off by the tile (less than
``min_visible_fraction`` of their mask visible) as ``is_crowd``. The runner's test evaluation doesn't set it.
"""

import math
from typing import Optional, Union
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch

from .lightning_data_module import LightningDataModule
from .transforms import Transforms
from .dataset import Dataset


class CropAndWeedInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_classes: int,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        color_jitter_enabled=False,
        scale_range: Optional[tuple[float, float]] = (0.1, 2.0),
        check_empty_targets=True,
        val_split: str = "val",
        min_visible_fraction: float = 0.0,
        val_tile_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.val_split = val_split
        self.min_visible_fraction = min_visible_fraction
        self.val_tile_size = val_tile_size
        self.val_batch_size = val_batch_size

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            min_visible_fraction=min_visible_fraction,
        )

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        **kwargs
    ):
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():
            segmentation = polygons_by_id[label_id]
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles) if isinstance(rles, list) else rles

            masks.append(tv_tensors.Mask(coco_mask.decode(rle), dtype=torch.bool))
            labels.append(cls_id)
            is_crowd.append(is_crowd_by_id[label_id])

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        self.train_dataset = Dataset(
            self.path,
            "train",
            self.target_parser,
            self.check_empty_targets,
            transforms=self.transforms,
        )
        self.val_dataset = Dataset(
            self.path, self.val_split, self.target_parser, check_empty_targets=False
        )
        if self.val_tile_size is not None:
            self.val_dataset = TiledDataset(
                self.val_dataset, self.val_tile_size, self.min_visible_fraction
            )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **{**self.dataloader_kwargs, "batch_size": self.val_batch_size or self.dataloader_kwargs["batch_size"]},
        )


class TiledDataset(torch.utils.data.Dataset):
    """A fixed grid of ``tile`` x ``tile`` crops of every frame of ``dataset`` (``ceil(side / tile)`` per side,
    spread evenly; all CropAndWeed frames are 1920x1088, so 2 x 2), each its own sample. A tile's
    instances are cropped: one with nothing visible is dropped, one with less than ``min_visible_fraction``
    of its mask visible becomes ``is_crowd``."""

    def __init__(self, dataset: Dataset, tile: int, min_visible_fraction: float):
        self.dataset = dataset
        self.tile = tile
        self.min_visible_fraction = min_visible_fraction
        height, width = dataset[0][0].shape[-2:]
        self.origins = [(y, x) for y in self._starts(height) for x in self._starts(width)]

    def _starts(self, length: int) -> list[int]:
        n = math.ceil(length / self.tile)
        return [round(k * (length - self.tile) / (n - 1)) for k in range(n)] if n > 1 else [0]

    def __len__(self):
        return len(self.dataset) * len(self.origins)

    def __getitem__(self, index: int):
        img, target = self.dataset[index // len(self.origins)]
        y, x = self.origins[index % len(self.origins)]
        area = target["masks"].flatten(1).sum(1)
        masks = target["masks"][:, y : y + self.tile, x : x + self.tile]
        visible = masks.flatten(1).sum(1)
        keep = visible > 0
        is_crowd = target["is_crowd"] | (visible < self.min_visible_fraction * area)
        return img[:, y : y + self.tile, x : x + self.tile], {
            "masks": masks[keep],
            "labels": target["labels"][keep],
            "is_crowd": is_crowd[keep],
        }
