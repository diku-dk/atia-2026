"""
NuInsSeg has two types of instance-labelled masks:

1. "label masks"
2. "label masks modify"

According to their repository, the ordinary label masks
assign a NEW numerical value to regions where nuclei overlap.
So counting unique non-zero values in those masks counts
overlap regions as additional nuclei which is incorrect (this was visible in 02_dataset_checks)

The "label masks modify" version assigns overlapping pixels to one of the
actual nucleus instances and is the version recommended by the dataset
authors for evaluation.

---------------------
the dataset checks (experiment 02) counted 35,138 unique non-zero labels using
"label masks", but the NuInsSeg paper reports 30,698 nuclei.
This script checks whether using "label masks modify" resolves this problem.

---------------------
so here for every image in every tissue the code:

- loads the ordinary label mask
- loads the modified label mask
- counts unique non-zero labels in both
- stores the counts
- compares the dataset-wide totals

expected result
---------------
The modified-label total should be much closer to the published total
of 30,698 nuclei ---> this is exactly what it says after running the code

If confirmed, "label masks modify" will be used as our instance ground truth
for evaluation throughout the project.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


# paths again
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)


# Helper function
def count_instances(mask_path):
    """
    Count unique non-background labels in an instance mask.

    Background is assumed to have value 0.
    """
    mask = np.array(Image.open(mask_path))
    labels = np.unique(mask)
    labels = labels[labels != 0]

    return len(labels)



# Compare both kinds of label masks
rows = []

for tissue_dir in sorted(DATA_ROOT.iterdir()):

    if not tissue_dir.is_dir():
        continue

    tissue_name = tissue_dir.name

    original_dir = tissue_dir / "label masks"
    modified_dir = tissue_dir / "label masks modify"

    if not original_dir.exists() or not modified_dir.exists():
        continue

    for original_path in sorted(original_dir.glob("*.tif")):

        sample_id = original_path.stem
        modified_path = modified_dir / f"{sample_id}.tif"

        if not modified_path.exists():
            print("Missing modified mask:", modified_path)
            continue

        n_original = count_instances(original_path)
        n_modified = count_instances(modified_path)

        rows.append({
            "tissue": tissue_name,
            "sample_id": sample_id,
            "n_original_labels": n_original,
            "n_modified_labels": n_modified,
            "difference": n_original - n_modified,
        })


# Create results table
df = pd.DataFrame(rows)

print(df.head())


print("\n DATASET TOTALS")
print("Images checked:", len(df))
print("Original label count:", df["n_original_labels"].sum())
print("Modified label count:", df["n_modified_labels"].sum())
print(
    "Extra labels in original masks:",
    df["difference"].sum()
)

print(
    "Images where counts differ:",
    (df["difference"] != 0).sum()
)



# Save detailed comparison
output_path = RESULTS_ROOT / "label_mask_comparison.csv"
df.to_csv(output_path, index=False)

print("\nSaved comparison to:")
print(output_path)