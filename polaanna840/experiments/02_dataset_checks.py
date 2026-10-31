
# What exactly is in NuInsSeg and how is it distributed across tissues

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)
rows = []


# Go through every tissue folder
for tissue_dir in sorted(DATA_ROOT.iterdir()):

    if not tissue_dir.is_dir():
        continue

    tissue_name = tissue_dir.name

    image_dir = tissue_dir / "tissue images"
    label_dir = tissue_dir / "label masks"
    vague_dir = tissue_dir / "vague areas" / "mask binary"

    # Skip anything that does not have the expected structure
    if not image_dir.exists():
        continue

    for image_path in sorted(image_dir.glob("*.png")):

        sample_id = image_path.stem

        label_path = label_dir / f"{sample_id}.tif"
        vague_path = vague_dir / f"{sample_id}.png"

        # Load instance mask
        if label_path.exists():
            instance_mask = np.array(Image.open(label_path))

            instance_ids = np.unique(instance_mask)
            instance_ids = instance_ids[instance_ids != 0]

            n_nuclei = len(instance_ids)
        else:
            n_nuclei = np.nan

        # Load vague-area mask
        if vague_path.exists():
            vague_mask = np.array(Image.open(vague_path))

            vague_binary = vague_mask > 0

            vague_pixels = vague_binary.sum()
            vague_fraction = vague_pixels / vague_binary.size
            has_vague_area = vague_pixels > 0
        else:
            vague_pixels = np.nan
            vague_fraction = np.nan
            has_vague_area = False

        # Store one row for this image
        rows.append({
            "tissue": tissue_name,
            "sample_id": sample_id,
            "n_nuclei": n_nuclei,
            "has_vague_area": has_vague_area,
            "vague_pixels": vague_pixels,
            "vague_fraction": vague_fraction,
        })



# Create dataframe
df = pd.DataFrame(rows)

print(df.head())
print()
print("Number of images:", len(df))
print("Number of tissues:", df["tissue"].nunique())
print("Total nuclei:", df["n_nuclei"].sum())
print("Images with vague areas:", df["has_vague_area"].sum())


# Save result
output_path = RESULTS_ROOT / "dataset_inventory.csv"
df.to_csv(output_path, index=False)

print()
print("Saved to:", output_path)


'''
Number of images: 665
Number of tissues: 31
Total nuclei: 35138
Images with vague areas: 664
'''