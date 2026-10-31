from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from mobile_sam import sam_model_registry, SamPredictor



# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"

SAM_CHECKPOINT = (
    PROJECT_ROOT
    / "SAM_code_github"
    / "MobileSAM"
    / "weights"
    / "mobile_sam.pt"
)


# one NuInsSeg sample just to check
TISSUE = "human bladder"
SAMPLE_NAME = "human_bladder_01"

image_path = (
    DATA_ROOT
    / TISSUE
    / "tissue images"
    / f"{SAMPLE_NAME}.png"
)

label_path = (
    DATA_ROOT
    / TISSUE
    / "label masks modify"
    / f"{SAMPLE_NAME}.tif"
)

print("=" * 70)
print("ZERO-SHOT MobileSAM: SINGLE NUCLEUS TEST")
print("=" * 70)

print("\nImage:", image_path)
print("Image exists:", image_path.exists())

print("\nLabel:", label_path)
print("Label exists:", label_path.exists())

print("\nCheckpoint:", SAM_CHECKPOINT)
print("Checkpoint exists:", SAM_CHECKPOINT.exists())



# loading H&E image and corrected instance mask
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# OpenCV loads images as BGR, but MobileSAM expects RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

if labels is None:
    raise FileNotFoundError(f"Could not load label mask: {label_path}")

print("\nImage shape:", image.shape)
print("Label shape:", labels.shape)


# Find the nuclei in the instance mask
    # each nucleus has its own integer ID
    # background has ID 0
instance_ids = np.unique(labels)
instance_ids = instance_ids[instance_ids != 0]

print("Number of nuclei in image:", len(instance_ids))

if len(instance_ids) == 0:
    raise ValueError("No nucleus instances were found in the label mask.")


# select one ground-truth nucleus N_i

# For this first test, just choose an instance around the middle of the list of nucleus IDs.
nucleus_id = instance_ids[len(instance_ids) // 2]

# Binary mask for this particular nucleus
gt_mask = labels == nucleus_id

print("\nSelected nucleus ID:", nucleus_id)
print("Ground-truth nucleus area:", gt_mask.sum(), "pixels")



# Generate bounding-box prompt B_i from N_i
y_coords, x_coords = np.where(gt_mask)

x_min = x_coords.min()
x_max = x_coords.max()
y_min = y_coords.min()
y_max = y_coords.max()

# SAM expects a box in the form:
# [x_min, y_min, x_max, y_max]
box = np.array(
    [x_min, y_min, x_max, y_max],
    dtype=np.float32
)

print("Bounding box B_i:", box)


# Load pretrained MobileSAM
device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nLoading MobileSAM...")
print("Device:", device)

mobile_sam = sam_model_registry["vit_t"](
    checkpoint=str(SAM_CHECKPOINT)
)

mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)

print("MobileSAM loaded successfully.")


# Give the H&E image to MobileSAM
print("\nEncoding image...")
predictor.set_image(image)
print("Image encoded.")



# Predict the selected nucleus using B_i
print("Predicting nucleus...")

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box,
    multimask_output=False
)

pred_mask = masks[0]

print("Prediction complete.")
print("Predicted mask area:", pred_mask.sum(), "pixels")
print("MobileSAM predicted IoU score:", round(float(scores[0]), 4))


# Evaluate prediction against N_i
intersection = np.logical_and(gt_mask, pred_mask).sum()
union = np.logical_or(gt_mask, pred_mask).sum()

if union > 0:
    iou = intersection / union
else:
    iou = 0.0

denominator = gt_mask.sum() + pred_mask.sum()

if denominator > 0:
    dice = (2 * intersection) / denominator
else:
    dice = 0.0

print("\n" + "-" * 70)
print("RESULT")
print("-" * 70)

print("Ground-truth area:", gt_mask.sum())
print("Predicted area:", pred_mask.sum())
print("Intersection:", intersection)
print(f"IoU:  {iou:.4f}")
print(f"Dice: {dice:.4f}")


# Visualise result
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# panel 1: original H&E image
axes[0].imshow(image)
axes[0].set_title("H&E image")
axes[0].axis("off")

# panel 2: selected GT nucleus + bounding-box prompt
axes[1].imshow(image)

axes[1].imshow(
    np.ma.masked_where(~gt_mask, gt_mask),
    alpha=0.5
)

rectangle = plt.Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    fill=False,
    linewidth=2
)

axes[1].add_patch(rectangle)
axes[1].set_title(
    f"Ground truth $N_i$ + prompt $B_i$\n"
    f"Nucleus ID = {nucleus_id}"
)
axes[1].axis("off")


# panel 3: zero-shot MobileSAM prediction
axes[2].imshow(image)

axes[2].imshow(
    np.ma.masked_where(~pred_mask, pred_mask),
    alpha=0.5
)

axes[2].set_title(
    f"Zero-shot MobileSAM prediction\n"
    f"Dice = {dice:.3f}, IoU = {iou:.3f}"
)
axes[2].axis("off")


plt.tight_layout()
plt.show()