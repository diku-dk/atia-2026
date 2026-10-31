# Which nuclei are actually associated with vague regions?


"""
Outputs

1. results/nucleus_vague_overlap.csv   ---> one row per nucleus
2. Printed dataset-level summary:
   - total nuclei
   - number of nuclei touching vague regions
   - percentage of nuclei touching vague regions
   - mean/median vague overlap among all nuclei
   - mean/median vague overlap among vague-associated nuclei only
   - nuclei with very small / moderate / large vague overlap
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


# paths again, not sure how to do this more efficiently
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)


# Storage
# one row for every nucleus instance in the whole dataset
rows = []

# Go through every tissue folder
for tissue_dir in sorted(DATA_ROOT.iterdir()):

    if not tissue_dir.is_dir():
        continue

    tissue_name = tissue_dir.name
    instance_dir = tissue_dir / "label masks modify"
    vague_dir = tissue_dir / "vague areas" / "mask binary"

    if not instance_dir.exists():
        print(f"Skipping {tissue_name}: no modified label-mask folder")
        continue

    if not vague_dir.exists():
        print(f"Skipping {tissue_name}: no vague-area mask folder")
        continue

    # Go through every instance mask
    for instance_path in sorted(instance_dir.glob("*.tif")):

        sample_id = instance_path.stem
        vague_path = vague_dir / f"{sample_id}.png"

        if not vague_path.exists():
            print(f"Missing vague mask: {vague_path}")
            continue

        # Load masks
        instance_mask = np.array(Image.open(instance_path))
        vague_mask = np.array(Image.open(vague_path))
        vague_binary = vague_mask > 0 # boolean form: fals = not vague, true = vague

        # Find all nucleus IDs in this image
        # background has value 0, so remove
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids != 0]

        # Analyze each nucleus separately
        for instance_id in instance_ids:
            nucleus_binary = instance_mask == instance_id #boolean mask for one nucleus only

            # nucleus area
            nucleus_area = int(nucleus_binary.sum())
            # again: overlap between this nucleus and vague region
            # a pixel belongs to the overlap only if it is:
            # part of this nucleus   AND   part of the vague-area mask
            overlap_binary = nucleus_binary & vague_binary
            vague_overlap_pixels = int(overlap_binary.sum())

            # Fraction of this nucleus marked as vague
            if nucleus_area > 0:
                fraction_of_nucleus_vague = (
                    vague_overlap_pixels / nucleus_area
                )
            else:
                fraction_of_nucleus_vague = np.nan

            # Simple binary indicator
            overlaps_vague = vague_overlap_pixels > 0


            # nucleus-level result
            rows.append({
                "tissue": tissue_name,
                "sample_id": sample_id,
                "instance_id": int(instance_id),
                "nucleus_area": nucleus_area,
                "vague_overlap_pixels": vague_overlap_pixels,
                "fraction_of_nucleus_vague": fraction_of_nucleus_vague,
                "overlaps_vague": overlaps_vague,
            })


# to df
df = pd.DataFrame(rows)


# sanity checks
print("\n Nucleus vague overlap analysis")
print("Total nuclei analyzed:", len(df))
print("Number of tissues:", df["tissue"].nunique())


# how many nuclei touch vague regions?
n_vague = int(df["overlaps_vague"].sum())
n_clear = int((~df["overlaps_vague"]).sum())

percentage_vague = 100 * n_vague / len(df)

print("\nNuclei overlapping vague areas:", n_vague)
print("Nuclei with no vague overlap:", n_clear)
print("Percentage of nuclei overlapping vague areas:", round(percentage_vague, 2),"%")


# Overlap distributions
print("\n Overlap fraction summary")
print("Mean overlap fraction across ALL nuclei:", df["fraction_of_nucleus_vague"].mean())
print("Median overlap fraction across ALL nuclei:", df["fraction_of_nucleus_vague"].median())

# Only nuclei that have at least some vague overlap
vague_nuclei = df[df["overlaps_vague"]]

if len(vague_nuclei) > 0:

    print("\nMean overlap fraction among vague-associated nuclei:",
          vague_nuclei["fraction_of_nucleus_vague"].mean())

    print("Median overlap fraction among vague-associated nuclei:",
        vague_nuclei["fraction_of_nucleus_vague"].median())

    print("Maximum overlap fraction:",
        vague_nuclei["fraction_of_nucleus_vague"].max())



# Rough overlap categories
# they are only descriptive EDA categories to help understand the distribution, could be helpful
df["overlap_category"] = pd.cut(
    df["fraction_of_nucleus_vague"],
    bins=[-0.001, 0, 0.10, 0.50, 1.0],
    labels=[
        "no overlap",
        "0-10%",
        "10-50%",
        "50-100%",
    ],
    include_lowest=True,
)


print("\n Roughly - overlap categories")
print(df["overlap_category"].value_counts().sort_index())


# Tissue-level summary
# will help see whether certain tissues contain many more vague-associated nuclei than others

tissue_summary = (
    df.groupby("tissue")
    .agg(
        n_nuclei=("instance_id", "count"),
        n_vague_nuclei=("overlaps_vague", "sum"),
        mean_vague_fraction=("fraction_of_nucleus_vague", "mean"),
        median_vague_fraction=("fraction_of_nucleus_vague", "median"),
    )
    .reset_index()
)
tissue_summary["fraction_vague_nuclei"] = (
    tissue_summary["n_vague_nuclei"]
    / tissue_summary["n_nuclei"]
)

print("\n First five tissue summaries")
print(tissue_summary.head())



# Save nucleus-level table
output_path = RESULTS_ROOT / "nucleus_vague_overlap.csv"
df.to_csv(output_path, index=False)
print("\nSaved nucleus-level results to:")
print(output_path)

# Save tissue-level summary
tissue_output_path = RESULTS_ROOT / "nucleus_vague_overlap_by_tissue.csv"
tissue_summary.to_csv(tissue_output_path, index=False)
print("\nSaved tissue-level summary to:")
print(tissue_output_path)




'''
#Validation so i know the script is working on complete corrected set 
    30,698 nuclei ✓
    31 tissues ✓

#Result: 4980 / 30698 = 16.22% of nuclei overlap a vague region


#Comparison
Clear / non-vague-associated nuclei
25,718 (83.78%)
             vs
Vague-associated nuclei
4,980 (16.22%)


Among nuclei that touch a vague region:
        mean nucleus overlap = 81.4%
        median = 100%
        maximum = 100%
        
        
Of the 4,980 nuclei that overlap vague areas,
        4078/4980 approx= 81.9%

have more than half of their nucleus inside the vague mask.
And the median being exactly 1.0 means that at least half of the vague-associated nuclei are completely contained within a vague area.

Basically from this:
0 overlap                    25,718
tiny overlap (<10%)             723
intermediate (10–50%)           179
majority overlap (>50%)       4,078


'''














# ==================================================
# DISTRIBUTION OF NUCLEUS-VAGUE OVERLAP
# ==================================================
#
# This variable is central to the planned experiment.
#
# NucEval uses essentially this same quantity to decide
# whether an instance should be excluded from evaluation:
#
#     vague overlap =
#         nucleus pixels inside vague area
#         --------------------------------
#         total nucleus pixels
#
# In our project, we do NOT currently threshold or
# exclude nuclei based on this value.
#
# Instead, the continuous overlap fraction will later
# be related to adaptation gain:
#
#     PEFT score - zero-shot score
#
# Therefore, before deciding whether any categories
# or thresholds are useful, inspect the actual
# distribution carefully.
# --------------------------------------------------

overlap = df["fraction_of_nucleus_vague"]

print("\n--------------------------------")
print("EXACT OVERLAP DISTRIBUTION")
print("--------------------------------")

print("\nAll nuclei:")
print(overlap.describe(
    percentiles=[
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]
))


# --------------------------------------------------
# Same summary only among nuclei that overlap
# a vague region at least once.
# --------------------------------------------------

positive_overlap = df.loc[
    df["fraction_of_nucleus_vague"] > 0,
    "fraction_of_nucleus_vague"
]

print("\nVague-associated nuclei only:")
print(positive_overlap.describe(
    percentiles=[
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]
))


# --------------------------------------------------
# Exact special cases
# --------------------------------------------------

n_zero = (overlap == 0).sum()
n_full = (overlap == 1).sum()
n_partial = ((overlap > 0) & (overlap < 1)).sum()

print("\nExact overlap cases:")
print(f"No overlap (r = 0):       {n_zero:,}")
print(f"Partial overlap (0 < r < 1): {n_partial:,}")
print(f"Full overlap (r = 1):      {n_full:,}")


# --------------------------------------------------
# NucEval-style threshold counts
# --------------------------------------------------
#
# IMPORTANT:
# These are NOT exclusions in our experiment.
#
# This only tells us how many of our nuclei would be
# removed if we applied different vague-overlap
# thresholds like NucEval does.
# --------------------------------------------------

thresholds = [0.01, 0.10, 0.25, 0.50, 0.75, 0.99]

print("\nNuclei exceeding possible overlap thresholds:")

for threshold in thresholds:

    n_above = (overlap > threshold).sum()
    percentage = 100 * n_above / len(df)

    print(
        f"r > {threshold:0.2f}: "
        f"{n_above:,} nuclei "
        f"({percentage:.2f}%)"
    )




import matplotlib.pyplot as plt


# ============================================================
# EDA FIGURE: NUCLEUS-LEVEL AMBIGUITY
# ============================================================

# ------------------------------------------------------------
# Panel A data: overall overlap categories
# ------------------------------------------------------------

n_zero = (df["fraction_of_nucleus_vague"] == 0).sum()
n_partial = (
    (df["fraction_of_nucleus_vague"] > 0)
    & (df["fraction_of_nucleus_vague"] < 1)
).sum()
n_full = (df["fraction_of_nucleus_vague"] == 1).sum()

counts = [n_zero, n_partial, n_full]
total = len(df)

percentages = [100 * count / total for count in counts]

categories = [
    r"No overlap" + "\n" + r"($r_i = 0$)",
    r"Partial overlap" + "\n" + r"($0 < r_i < 1$)",
    r"Complete overlap" + "\n" + r"($r_i = 1$)"
]


# ------------------------------------------------------------
# Panel B data: ambiguity prevalence by tissue
# ------------------------------------------------------------

tissue_plot = tissue_summary.copy()

tissue_plot["percentage_vague_nuclei"] = (
    tissue_plot["fraction_vague_nuclei"] * 100
)

# Sort so differences are easy to see
tissue_plot = tissue_plot.sort_values(
    "percentage_vague_nuclei"
)


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(13, 8),
    gridspec_kw={"width_ratios": [1, 1.5]}
)


# ------------------------------------------------------------
# Panel A
# ------------------------------------------------------------

bars = ax1.bar(categories, counts, width=0.65)

for bar, count, percentage in zip(bars, counts, percentages):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 400,
        f"{count:,}\n({percentage:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax1.set_ylabel("Number of nuclei")
ax1.set_title("A. Overall vague-region overlap")

ax1.set_ylim(0, max(counts) * 1.15)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1.yaxis.grid(True, alpha=0.2)
ax1.set_axisbelow(True)


# ------------------------------------------------------------
# Panel B
# ------------------------------------------------------------

ax2.barh(
    tissue_plot["tissue"],
    tissue_plot["percentage_vague_nuclei"]
)

ax2.set_xlabel("Vague-associated nuclei (%)")
ax2.set_title("B. Vague-region overlap by tissue")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.xaxis.grid(True, alpha=0.2)
ax2.set_axisbelow(True)


# ------------------------------------------------------------
# Finish and save
# ------------------------------------------------------------

plt.tight_layout()

output_path = RESULTS_ROOT / "ambiguity_eda.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved EDA figure to:\n{output_path}")

plt.show()