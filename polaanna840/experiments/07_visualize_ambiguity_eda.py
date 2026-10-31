"""

Visualize the main findings from the NuInsSeg ambiguity EDA performed
in experiments 04, 05, and 06.

input files:
results/nucleus_vague_overlap.csv
results/vague_region_analysis.csv

output files:
Figures are saved in: results/figures/

The script creates:

01_clear_vs_vague_nuclei.png
02_vague_nucleus_overlap_distribution.png
03_nuclei_per_vague_region.png
04_vague_nuclei_by_tissue.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = RESULTS_ROOT / "figures"

FIGURE_ROOT.mkdir(exist_ok=True)



nucleus_path = RESULTS_ROOT / "nucleus_vague_overlap.csv"
region_path = RESULTS_ROOT / "vague_region_analysis.csv"

nuclei = pd.read_csv(nucleus_path)
regions = pd.read_csv(region_path)



# FIGURE 1
# Clear vs vague-associated nuclei


n_clear = (~nuclei["overlaps_vague"]).sum()
n_vague = nuclei["overlaps_vague"].sum()

counts = [n_clear, n_vague]
labels = ["Clear", "Vague-associated"]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(labels, counts)

ax.set_ylabel("Number of nuclei")
ax.set_title("Clear vs vague-associated nuclei")

# exact counts and percentages above bars
total = len(nuclei)

for bar, count in zip(bars, counts):

    percentage = 100 * count / total

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output = FIGURE_ROOT / "01_clear_vs_vague_nuclei.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()


# FIGURE 2
# Distribution of vague overlap among vague-associated nuclei

# Experiment 05 showed a median of 1.0 --> at
# least half of vague-associated nuclei are completely
# contained in vague regions.

vague_nuclei = nuclei[nuclei["overlaps_vague"]].copy()

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    vague_nuclei["fraction_of_nucleus_vague"],
    bins=np.linspace(0, 1, 21),
    edgecolor="black"
)

ax.set_xlabel("Fraction of nucleus overlapping vague area")
ax.set_ylabel("Number of nuclei")
ax.set_title(
    "Vague-area overlap among vague-associated nuclei"
)

# Mark the median
median_overlap = (
    vague_nuclei["fraction_of_nucleus_vague"].median()
)

ax.axvline(
    median_overlap,
    linestyle="--",
    linewidth=2,
    label=f"Median = {median_overlap:.2f}"
)

ax.legend()

plt.tight_layout()

output = (
    FIGURE_ROOT
    / "02_vague_nucleus_overlap_distribution.png"
)

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()





# FIGURE 4
# Fraction of vague-associated nuclei by tissue


tissue_summary = (
    nuclei.groupby("tissue")
    .agg(
        total_nuclei=("instance_id", "count"),
        vague_nuclei=("overlaps_vague", "sum")
    )
    .reset_index()
)

tissue_summary["vague_percentage"] = (
    100
    * tissue_summary["vague_nuclei"]
    / tissue_summary["total_nuclei"]
)

# Sort tissues from least to most vague-associated
tissue_summary = tissue_summary.sort_values(
    "vague_percentage",
    ascending=True
)


fig, ax = plt.subplots(figsize=(10, 10))

ax.barh(
    tissue_summary["tissue"],
    tissue_summary["vague_percentage"]
)

ax.set_xlabel("Vague-associated nuclei (%)")
ax.set_ylabel("Tissue")
ax.set_title(
    "Prevalence of vague-associated nuclei by tissue"
)

plt.tight_layout()

output = FIGURE_ROOT / "04_vague_nuclei_by_tissue.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()



# printing everything to make sure it worked and was created

print("\n--------------------------------")
print("VISUALIZATIONS COMPLETE")
print("--------------------------------")

print("Figures saved to:")
print(FIGURE_ROOT)

print("\nCreated:")
print("01_clear_vs_vague_nuclei.png")
print("02_vague_nucleus_overlap_distribution.png")
print("03_nuclei_per_vague_region.png")
print("04_vague_nuclei_by_tissue.png")









# adding:
# FIGURE 5
# Structure of nucleus-vague overlap

overlap = nuclei["fraction_of_nucleus_vague"]

overlap_structure = pd.Series({
    "No overlap\n(r = 0)": (overlap == 0).sum(),
    "Partial overlap\n(0 < r < 1)": (
        (overlap > 0) & (overlap < 1)
    ).sum(),
    "Complete overlap\n(r = 1)": (overlap == 1).sum(),
})

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    overlap_structure.index,
    overlap_structure.values
)

ax.set_ylabel("Number of nuclei")
ax.set_title("Structure of nucleus–vague-region overlap")

for bar, count in zip(
    bars,
    overlap_structure.values
):

    percentage = 100 * count / len(nuclei)

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output = (
    FIGURE_ROOT
    / "07_nucleus_vague_overlap_structure.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()