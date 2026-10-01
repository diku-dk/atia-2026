"""CropAndWeed instance segmentation for EoMT: our counterpart of upstream ``datasets/coco_instance.py``.

Upstream's base ``Dataset`` (``datasets/dataset.py``, not copied) only reads zipped COCO/ADE20K
releases, so ``CropAndWeedDataset`` reads a split of our YOLO-seg dataset instead
(``data/seed<N>/<V>/yolo/segmentation``, via ``src/seg_dataset.py``) and returns the same
``(image, {"masks", "labels", "is_crowd"})`` samples. Each instance's polygon is rasterised the way
upstream's ``COCOInstance.target_parser`` does (``frPyObjects`` -> ``merge`` -> ``decode``, here with
hotcoco's pycocotools-compatible ``mask`` module). The Vegetation ignore regions are not used
(``is_crowd`` is all False): as for YOLO, they only matter in evaluation (``src/cropandweed_eval.py``).

Unlike upstream, which drops images without instances from every split, ``check_empty_targets``
only applies to ``train``: val/test keep every image, so the evaluation sees the whole split.
"""

from pathlib import Path
from typing import Union

import torch
from hotcoco import mask as coco_mask
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from src.seg_dataset import read_split

from .lightning_data_module import LightningDataModule
from .transforms import Transforms


class CropAndWeedDataset(torch.utils.data.Dataset):
    """One split of a YOLO-seg dataset dir, as EoMT samples; ``images`` keeps ``read_split``'s order (sorted by stem)."""

    def __init__(self, dataset_dir: Path, split: str, transforms=None, skip_empty: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transforms = transforms
        self.images = [img for img in read_split(self.dataset_dir, split) if img["instances"] or not skip_empty]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        height, width = image["height"], image["width"]
        with Image.open(self.dataset_dir / "images" / self.split / f"{image['stem']}.jpg") as img:
            img = tv_tensors.Image(img.convert("RGB"))

        masks = torch.zeros((len(image["instances"]), height, width), dtype=torch.bool)
        for i, inst in enumerate(image["instances"]):
            rles = coco_mask.frPyObjects([inst["polygon"].flatten().tolist()], height, width)
            masks[i] = torch.from_numpy(coco_mask.decode(coco_mask.merge(rles))).bool()
        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": torch.tensor([inst["cls"] for inst in image["instances"]], dtype=torch.long),
            "is_crowd": torch.zeros(len(image["instances"]), dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CropAndWeedInstance(LightningDataModule):
    """``path`` is a variant's YOLO-seg dir; ``num_classes`` is filled in from its ``data.yaml`` by ``src/eomt/trainer.py``."""

    def __init__(
        self,
        path,
        num_classes: int,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
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

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        if stage in ("fit", None):
            self.train_dataset = CropAndWeedDataset(
                self.path, "train", self.transforms, skip_empty=self.check_empty_targets
            )
        self.val_dataset = CropAndWeedDataset(self.path, "val")

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
            **self.dataloader_kwargs,
        )

    def eval_dataloader(self, split: str) -> DataLoader:
        """Unshuffled loader over every image of ``split`` (used by ``src/eomt/evaluator.py``)."""
        return DataLoader(
            CropAndWeedDataset(self.path, split),
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
