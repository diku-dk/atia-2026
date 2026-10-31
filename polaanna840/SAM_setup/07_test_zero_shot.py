# selects the first available image automatically, purely to check that
# MobileSAM can produce a segmentation mask from a bounding-box prompt.
from pathlib import Path

import numpy as np
import torch
from mobile_sam import SamPredictor

from models.base import load_mobile_sam
from experiments.nucleus_dataset_loader import (
    load_nuinsseg_sample,
    extract_nucleus,
    get_nucleus_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"

# Select the first available image-mask pair deterministically.
pairs = []

for tissue_dir in sorted(DATA_ROOT.iterdir()):
    if not tissue_dir.is_dir():
        continue

    image_dir = tissue_dir / "tissue images"
    mask_dir = tissue_dir / "label masks modify"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        continue

    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / f"{image_path.stem}.tif"

        if mask_path.is_file():
            pairs.append((image_path, mask_path))

assert len(pairs) == 665, f"Expected 665 pairs, found {len(pairs)}"

image_path, mask_path = pairs[0]

image, instance_mask = load_nuinsseg_sample(
    image_path, mask_path
)

nucleus_id = get_nucleus_ids(instance_mask)[0]
ground_truth, bbox = extract_nucleus(instance_mask, nucleus_id)

print("Sample:", image_path.stem)
print("Nucleus ID:", nucleus_id)
print("Image shape:", image.shape)
print("Bounding box:", bbox)

# Zero-shot inference only.
model = load_mobile_sam()
model.eval()

predictor = SamPredictor(model)
predictor.set_image(image)

with torch.no_grad():
    masks, scores, logits = predictor.predict(
        box=bbox,
        multimask_output=False
    )

prediction = masks[0]

assert prediction.shape == ground_truth.shape, (
    f"Prediction shape {prediction.shape} "
    f"does not match GT shape {ground_truth.shape}"
)

intersection = np.logical_and(prediction, ground_truth).sum()
denominator = prediction.sum() + ground_truth.sum()

dice = (
    2 * intersection / denominator
    if denominator > 0 else 1.0
)

print("\nPrediction shape:", prediction.shape)
print("Ground-truth shape:", ground_truth.shape)
print("Predicted pixels:", int(prediction.sum()))
print("Ground-truth pixels:", int(ground_truth.sum()))
print("Dice:", float(dice))
print("\nZERO-SHOT FORWARD TEST PASSED")
