"""
creating the corrected NuInsSeg dataset inventory using
"label masks modify" as the instance ground truth.

The earlier mask comparison showed that this version gives
30,698 nuclei, matching the published dataset total.

Output:
results/dataset_inventory_corrected.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
RESULTS_ROOT = PROJECT_ROOT / "results"

RESULTS_ROOT.mkdir(exist_ok=True)



# Build inventory
rows = []

for tissue_dir in sorted(DATA_ROOT.iterdir()):

    if not tissue_dir.is_dir():
        continue

    tissue_name = tissue_dir.name

    image_dir = tissue_dir / "tissue images"
    label_dir = tissue_dir / "label masks modify"
    vague_dir = tissue_dir / "vague areas" / "mask binary"

    if not image_dir.exists():
        print(f"Skipping {tissue_name}: no tissue images folder")
        continue

    for image_path in sorted(image_dir.glob("*.png")):

        sample_id = image_path.stem

        label_path = label_dir / f"{sample_id}.tif"
        vague_path = vague_dir / f"{sample_id}.png"


        # Count nuclei
        if label_path.exists():
            instance_mask = np.array(Image.open(label_path))
            instance_ids = np.unique(instance_mask)
            instance_ids = instance_ids[instance_ids != 0]
            n_nuclei = len(instance_ids)

        else:
            print(f"Missing label mask: {label_path}")
            n_nuclei = np.nan


        # Measure vague areas
        if vague_path.exists():
            vague_mask = np.array(Image.open(vague_path))
            vague_binary = vague_mask > 0

            vague_pixels = int(vague_binary.sum())
            vague_fraction = vague_pixels / vague_binary.size
            has_vague_area = vague_pixels > 0

        else:
            print(f"Missing vague mask: {vague_path}")

            vague_pixels = np.nan
            vague_fraction = np.nan
            has_vague_area = False


        rows.append({
            "tissue": tissue_name,
            "sample_id": sample_id,
            "n_nuclei": n_nuclei,
            "has_vague_area": has_vague_area,
            "vague_pixels": vague_pixels,
            "vague_fraction": vague_fraction,
        })



# Results
df = pd.DataFrame(rows)

print("\nFirst five rows:")
print(df.head())

print("\nCorrected dataset summary")
print("Number of images:", len(df))
print("Number of tissues:", df["tissue"].nunique())
print("Total nuclei:", int(df["n_nuclei"].sum()))
print("Images with vague areas:", int(df["has_vague_area"].sum()))

print("\nVague-area summary")
print("Mean vague fraction:", df["vague_fraction"].mean())
print("Median vague fraction:", df["vague_fraction"].median())
print("Maximum vague fraction:", df["vague_fraction"].max())



# Save
output_path = RESULTS_ROOT / "dataset_inventory_corrected.csv"
df.to_csv(output_path, index=False)

print("\nSaved corrected inventory to:")
print(output_path)














from pathlib import Path
import numpy as np
from PIL import Image

DATASET = Path(
    r"C:\Users\H P\Desktop\NuInsSeg-MobileSAM\data\NuInsSeg"
)

tissue = DATASET / "human bladder"
sample = "human_bladder_01"

files = {
    "Tissue image":
        tissue / "tissue images" / f"{sample}.png",

    "Label mask":
        tissue / "label masks" / f"{sample}.tif",

    "Label mask modify":
        tissue / "label masks modify" / f"{sample}.tif",

    "Binary mask":
        tissue / "mask binary" / f"{sample}.png",

    "Binary mask without border":
        tissue / "mask binary without border" / f"{sample}.png",

    "Distance map":
        tissue / "distance maps" / f"{sample}.png",

    "Weighted map":
        tissue / "weighted_maps" / f"{sample}.png",

    "Vague mask":
        tissue / "vague areas" / f"{sample}.png",
}


for name, path in files.items():

    print("\n" + "=" * 70)
    print(name)
    print("File:", path.name)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    arr = np.array(Image.open(path))

    print("Shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min:", arr.min())
    print("max:", arr.max())

    unique = np.unique(arr)

    print("Number of unique values:", len(unique))

    if len(unique) <= 50:
        print("Unique values:", unique)
    else:
        print("First 50 unique values:", unique[:50])