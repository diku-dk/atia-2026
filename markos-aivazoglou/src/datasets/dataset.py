# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Copied from tue-mps/eomt@7bd19dd (datasets/dataset.py). Changes: reads images from
# <dataset_dir>/images/<split>/ and the COCO annotations json from <dataset_dir>/annotations/<split>.json
# (plain files instead of zip archives), keeping upstream's annotation loop; the zip, mask-image
# (semantic/panoptic) and target_instance branches are removed; images without annotations are
# skipped only with check_empty_targets (upstream always skips them), and get (0, H, W) masks otherwise.
# ---------------------------------------------------------------


import json
from pathlib import Path
from typing import Callable, Optional
import torch
from PIL import Image
from torchvision import tv_tensors


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
    ):
        self.img_folder = Path(dataset_dir) / "images" / split
        self.target_parser = target_parser
        self.transforms = transforms

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}

        with open(Path(dataset_dir) / "annotations" / f"{split}.json") as file:
            annotation_data = json.load(file)

        image_id_to_file_name = {
            image["id"]: image["file_name"] for image in annotation_data["images"]
        }

        for annotation in annotation_data["annotations"]:
            img_filename = image_id_to_file_name[annotation["image_id"]]

            if img_filename not in self.labels_by_id:
                self.labels_by_id[img_filename] = {}

            if img_filename not in self.polygons_by_id:
                self.polygons_by_id[img_filename] = {}

            if img_filename not in self.is_crowd_by_id:
                self.is_crowd_by_id[img_filename] = {}

            self.labels_by_id[img_filename][annotation["id"]] = annotation[
                "category_id"
            ]
            self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                "segmentation"
            ]
            self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                annotation["iscrowd"]
            )

        self.imgs = []

        for img_filename in sorted(image_id_to_file_name.values()):
            if check_empty_targets and not self.labels_by_id.get(img_filename):
                continue

            self.imgs.append(img_filename)

    def __getitem__(self, index: int):
        with Image.open(self.img_folder / self.imgs[index]) as img:
            img = tv_tensors.Image(img.convert("RGB"))

        masks, labels, is_crowd = self.target_parser(
            polygons_by_id=self.polygons_by_id.get(self.imgs[index], {}),
            labels_by_id=self.labels_by_id.get(self.imgs[index], {}),
            is_crowd_by_id=self.is_crowd_by_id.get(self.imgs[index], {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        target = {
            "masks": tv_tensors.Mask(
                torch.stack(masks)
                if masks
                else torch.zeros((0, *img.shape[-2:]), dtype=torch.bool)
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
