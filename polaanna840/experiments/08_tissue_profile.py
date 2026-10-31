
"""

Looking into the tissues before making cross validarion

For every tissue, this script calculates:

- species (human or mouse)
- number of images
- total number of nuclei
- number of vague-associated nuclei
- number of clear nuclei
- percentage of nuclei that are vague-associated
- mean fraction of image area marked as vague
- median fraction of image area marked as vague

input files:
results/dataset_inventory_corrected.csv
results/nucleus_vague_overlap.csv

output is here:
results/tissue_profile.csv

results/figures/
    05_nuclei_by_tissue.png
    06_tissue_size_vs_ambiguity.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = RESULTS_ROOT / "figures"

FIGURE_ROOT.mkdir(exist_ok=True)



inventory_path = (
    RESULTS_ROOT / "dataset_inventory_corrected.csv"
)

nucleus_path = (
    RESULTS_ROOT / "nucleus_vague_overlap.csv"
)

inventory = pd.read_csv(inventory_path)
nuclei = pd.read_csv(nucleus_path)



# PART 1
# IMAGE-LEVEL INFORMATION BY TISSUE

# From the corrected inventory we calculate:
# - number of images
# - mean vague fraction
# - median vague fraction

image_summary = (
    inventory
    .groupby("tissue")
    .agg(
        n_images=("sample_id", "count"),

        mean_image_vague_fraction=(
            "vague_fraction",
            "mean"
        ),

        median_image_vague_fraction=(
            "vague_fraction",
            "median"
        ),
    )
    .reset_index()
)



# PART 2
# NUCLEUS-LEVEL INFORMATION BY TISSUE
nucleus_summary = (
    nuclei
    .groupby("tissue")
    .agg(
        n_nuclei=("instance_id", "count"),

        n_vague_nuclei=(
            "overlaps_vague",
            "sum"
        ),
    )
    .reset_index()
)

# Number of clear nuclei
nucleus_summary["n_clear_nuclei"] = (
    nucleus_summary["n_nuclei"]
    - nucleus_summary["n_vague_nuclei"]
)

# Percentage of nuclei associated with vague areas
nucleus_summary["vague_nuclei_percentage"] = (
    100
    * nucleus_summary["n_vague_nuclei"]
    / nucleus_summary["n_nuclei"]
)


# PART 3
# COMBINE IMAGE AND NUCLEUS INFORMATION
tissue_profile = pd.merge(
    image_summary,
    nucleus_summary,
    on="tissue",
    how="inner"
)



# Determine species - human or mouse

def get_species(tissue_name):

    tissue_name = tissue_name.lower()
    if tissue_name.startswith("human"):
        return "human"
    elif tissue_name.startswith("mouse"):
        return "mouse"
    else:
        return "unknown" # just in case

tissue_profile["species"] = (
    tissue_profile["tissue"]
    .apply(get_species)
)


# Reorder columns so the table is easier to read
tissue_profile = tissue_profile[
    [
        "tissue",
        "species",
        "n_images",
        "n_nuclei",
        "n_clear_nuclei",
        "n_vague_nuclei",
        "vague_nuclei_percentage",
        "mean_image_vague_fraction",
        "median_image_vague_fraction",
    ]
]



# Sort by number of nuclei
tissue_profile = tissue_profile.sort_values(
    "n_nuclei",
    ascending=False
)



# PART 4 - summaries

print("\n--------------------------------")
print("TISSUE PROFILE")
print("--------------------------------")
print(tissue_profile.to_string(index=False))



# Sanity checks
print("\n--------------------------------")
print("SANITY CHECKS")
print("--------------------------------")
print(
    "Number of tissues:",
    len(tissue_profile)
)
print(
    "Total images:",
    tissue_profile["n_images"].sum()
)
print(
    "Total nuclei:",
    tissue_profile["n_nuclei"].sum()
)
print(
    "Total vague-associated nuclei:",
    tissue_profile["n_vague_nuclei"].sum()
)
print(
    "Total clear nuclei:",
    tissue_profile["n_clear_nuclei"].sum()
)

print("\nSpecies counts:")

print(
    tissue_profile["species"]
    .value_counts()
)



# FIGURE 1
# Number of nuclei per tissue
plot_data = tissue_profile.sort_values(
    "n_nuclei",
    ascending=True
)
fig, ax = plt.subplots(
    figsize=(10, 10)
)
ax.barh(
    plot_data["tissue"],
    plot_data["n_nuclei"]
)
ax.set_xlabel(
    "Number of nuclei"
)
ax.set_ylabel(
    "Tissue"
)
ax.set_title(
    "Number of annotated nuclei by tissue"
)

plt.tight_layout()

output_path = (
    FIGURE_ROOT
    / "05_nuclei_by_tissue.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# FIGURE 2
# Tissue size vs ambiguity prevalence
fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.scatter(
    tissue_profile["n_nuclei"],
    tissue_profile["vague_nuclei_percentage"],
    s=60
)



# Label every tissue
for _, row in tissue_profile.iterrows():

    ax.annotate(
        row["tissue"],
        (
            row["n_nuclei"],
            row["vague_nuclei_percentage"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


ax.set_xlabel(
    "Number of nuclei"
)

ax.set_ylabel(
    "Vague-associated nuclei (%)"
)

ax.set_title(
    "Tissue size vs ambiguity prevalence"
)

plt.tight_layout()

output_path = (
    FIGURE_ROOT
    / "06_tissue_size_vs_ambiguity.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# SAVE TISSUE PROFILE
output_path = (
    RESULTS_ROOT / "tissue_profile.csv"
)

tissue_profile.to_csv(
    output_path,
    index=False
)


print("\n--------------------------------")
print("TISSUE PROFILE SAVED")
print("--------------------------------")

print(output_path)

print("\nFigures saved to:")
print(FIGURE_ROOT)



