
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset

def load_nuinsseg_sample(image_path, instance_mask_path):
    """
    Load one NuInsSeg RGB image and its instance-label mask.

    The instance-label mask contains:
        0 = background
        positive integer = nucleus instance ID
    """

    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    instance_mask = np.array(
        Image.open(instance_mask_path)
    )

    if image.shape[:2] != instance_mask.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match "
            f"instance mask shape {instance_mask.shape}"
        )

    if not np.issubdtype(instance_mask.dtype, np.integer):
        raise TypeError(
            "Instance mask must contain integer instance IDs."
        )

    return image, instance_mask


def extract_nucleus(instance_mask, nucleus_id):
    """
    Extract one nucleus as a binary mask and calculate
    its tight bounding box in original image coordinates.
    """

    nucleus_mask = instance_mask == nucleus_id

    if not nucleus_mask.any():
        raise ValueError(
            f"Nucleus ID {nucleus_id} does not exist."
        )

    ys, xs = np.where(nucleus_mask)

    x_min = int(xs.min())
    y_min = int(ys.min())

    # Exclusive upper bounds, suitable for box coordinates.
    x_max = int(xs.max()) + 1
    y_max = int(ys.max()) + 1

    bbox = np.array(
        [x_min, y_min, x_max, y_max],
        dtype=np.float32
    )

    return nucleus_mask, bbox


def get_nucleus_ids(instance_mask):
    """
    Return all nucleus IDs, excluding background.
    """

    ids = np.unique(instance_mask)

    return ids[ids != 0]



class NuInsSegNucleusDataset(Dataset):
    """
    Loads individual nuclei from selected tissue folds.

    Each item contains:
    - RGB image
    - Binary ground-truth nucleus mask
    - Bounding-box prompt
    - Tissue, sample ID, and nucleus ID
    """

    def __init__(self, data_root, folds_csv, selected_folds):
        self.data_root = Path(data_root)

        folds = pd.read_csv(folds_csv)

        selected_tissues = folds.loc[
            folds["fold"].isin(selected_folds), "tissue"
        ].tolist()

        self.samples = []

        for tissue in selected_tissues:
            tissue_dir = self.data_root / tissue
            image_dir = tissue_dir / "tissue images"
            mask_dir = tissue_dir / "label masks modify"

            for image_path in sorted(image_dir.glob("*.png")):
                mask_path = mask_dir / f"{image_path.stem}.tif"

                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)

                _, instance_mask = load_nuinsseg_sample(
                    image_path, mask_path
                )

                for nucleus_id in get_nucleus_ids(instance_mask):
                    self.samples.append((
                        image_path,
                        mask_path,
                        tissue,
                        int(nucleus_id)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, tissue, nucleus_id = (
            self.samples[index]
        )

        image, instance_mask = load_nuinsseg_sample(
            image_path, mask_path
        )

        nucleus_mask, bbox = extract_nucleus(
            instance_mask, nucleus_id
        )

        return {
            "image": image,
            "mask": nucleus_mask,
            "bbox": bbox,
            "tissue": tissue,
            "sample_id": image_path.stem,
            "nucleus_id": nucleus_id,
        }
