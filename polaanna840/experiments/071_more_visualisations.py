"""
071_more_visualisations.py

For nucleus i:

    r_i = |N_i ∩ V| / |N_i|

where N_i is the binary mask of nucleus i and V is the binary vague mask
provided by NuInsSeg.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image



PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "visualisations" / "vague_overlap"

INVENTORY_PATH = RESULTS_ROOT / "dataset_inventory_corrected.csv"
NUCLEUS_OVERLAP_PATH = RESULTS_ROOT / "nucleus_vague_overlap.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# loading

def load_image(tissue, sample_id):
    """Load the original H&E image."""
    path = DATA_ROOT / tissue / "tissue images" / f"{sample_id}.png"

    if not path.exists():
        raise FileNotFoundError(f"H&E image not found:\n{path}")

    return np.array(Image.open(path).convert("RGB"))


def load_instance_mask(tissue, sample_id):
    """
    Load the modified instance mask.

    We use 'label masks modify' because these are the conventional
    non-overlapping instance masks used in the corrected dataset analysis.
    """
    path = DATA_ROOT / tissue / "label masks modify" / f"{sample_id}.tif"

    if not path.exists():
        raise FileNotFoundError(f"Instance mask not found:\n{path}")

    return np.array(Image.open(path))


def load_vague_mask(tissue, sample_id):
    """
    Load the binary NuInsSeg vague-area mask.

    Important:
    the vague masks are inside:

        vague areas / mask binary /

    rather than directly inside 'vague areas'.
    """
    path = (
        DATA_ROOT
        / tissue
        / "vague areas"
        / "mask binary"
        / f"{sample_id}.png"
    )

    if not path.exists():
        raise FileNotFoundError(f"Vague mask not found:\n{path}")

    mask = np.array(Image.open(path))

    # NuInsSeg stores these as 0/255 images.
    return mask > 0


def load_sample(tissue, sample_id):
    """Load the H&E image, modified instance mask, and binary vague mask."""
    image = load_image(tissue, sample_id)
    instance_mask = load_instance_mask(tissue, sample_id)
    vague_mask = load_vague_mask(tissue, sample_id)

    return image, instance_mask, vague_mask


# helper functions

def save_figure(fig, filename):
    """Save a figure and close it."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path.name}")


def make_vague_overlay(image, vague_mask):
    """
    Make a simple overlay where vague pixels are highlighted in red.

    The underlying H&E image remains visible so the vague annotations
    can be interpreted in their histological context.
    """
    overlay = image.astype(float).copy()

    highlight = np.zeros_like(overlay)
    highlight[..., 0] = 255

    alpha = 0.45

    overlay[vague_mask] = (
        (1 - alpha) * overlay[vague_mask]
        + alpha * highlight[vague_mask]
    )

    return overlay.astype(np.uint8)


def get_crop_bounds(binary_mask, padding=30):
    """
    Return a crop around a binary object with some surrounding context.
    """
    rows, cols = np.where(binary_mask)

    if len(rows) == 0:
        raise ValueError("Cannot crop an empty mask.")

    y_min = max(0, rows.min() - padding)
    y_max = min(binary_mask.shape[0], rows.max() + padding + 1)

    x_min = max(0, cols.min() - padding)
    x_max = min(binary_mask.shape[1], cols.max() + padding + 1)

    return y_min, y_max, x_min, x_max


def get_column(df, possible_names):
    """
    Return the first matching column name.

    This keeps the script compatible with the CSVs already produced
    by the earlier dataset-analysis scripts.
    """
    for name in possible_names:
        if name in df.columns:
            return name

    raise KeyError(
        f"Could not find any of these columns:\n{possible_names}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )



# 1. IMAGE-LEVEL VAGUE-AREA EXTREMES
def plot_image_level_extremes(inventory):
    """
    Compare:
    - the image with the smallest vague fraction;
    - the image with the largest vague fraction.

    Because the corrected inventory found vague pixels in 664/665 images,
    the minimum should correspond to the single image with no vague area.
    """

    vague_fraction_col = get_column(
        inventory,
        ["vague_fraction", "image_vague_fraction"]
    )

    tissue_col = get_column(inventory, ["tissue"])
    sample_col = get_column(
        inventory,
        ["sample_id", "sample", "image_id"]
    )

    minimum = inventory.loc[inventory[vague_fraction_col].idxmin()]
    maximum = inventory.loc[inventory[vague_fraction_col].idxmax()]

    print("\n" + "=" * 80)
    print("IMAGE-LEVEL VAGUE-AREA EXTREMES")
    print("=" * 80)

    print(
        f"Lowest vague fraction:\n"
        f"  {minimum[tissue_col]} / {minimum[sample_col]}\n"
        f"  vague fraction = {minimum[vague_fraction_col]:.4f}"
    )

    print(
        f"\nHighest vague fraction:\n"
        f"  {maximum[tissue_col]} / {maximum[sample_col]}\n"
        f"  vague fraction = {maximum[vague_fraction_col]:.4f}"
    )

    examples = [
        ("No vague pixels", minimum),
        ("Largest vague-area fraction", maximum),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    for row, (label, record) in enumerate(examples):

        tissue = record[tissue_col]
        sample_id = record[sample_col]
        vague_fraction = record[vague_fraction_col]

        image, _, vague_mask = load_sample(tissue, sample_id)
        overlay = make_vague_overlay(image, vague_mask)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"{label}\n{tissue} — {sample_id}"
        )

        axes[row, 1].imshow(vague_mask, cmap="gray")
        axes[row, 1].set_title(
            f"Vague mask\n"
            f"{100 * vague_fraction:.2f}% of image pixels"
        )

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Vague mask overlay")

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(
        "NuInsSeg image-level vague annotations",
        fontsize=15
    )

    fig.tight_layout()

    save_figure(
        fig,
        "01_image_level_vague_extremes.png"
    )



# 2. OVERALL NUCLEUS-LEVEL r_i DISTRIBUTION
def plot_overlap_distribution(nuclei):
    """
    Show the natural decomposition of r_i:

        r_i = 0
        0 < r_i < 1
        r_i = 1

    This is more informative than a normal histogram because most nuclei
    lie exactly at one of the two endpoints.
    """

    r_col = get_column(
        nuclei,
        [
            "fraction_of_nucleus_vague",
            "vague_overlap_fraction",
            "overlap_fraction",
            "r_i"
        ]
    )

    r = nuclei[r_col].to_numpy()

    no_overlap = np.isclose(r, 0)
    complete_overlap = np.isclose(r, 1)
    partial_overlap = (~no_overlap) & (~complete_overlap)

    counts = np.array([
        no_overlap.sum(),
        partial_overlap.sum(),
        complete_overlap.sum()
    ])

    percentages = counts / len(r) * 100

    labels = [
        r"$r_i = 0$",
        r"$0 < r_i < 1$",
        r"$r_i = 1$"
    ]

    print("\n" + "=" * 80)
    print("NUCLEUS-LEVEL OVERLAP")
    print("=" * 80)

    print(f"Total nuclei:     {len(r):,}")
    print(
        f"r = 0:            {counts[0]:,} "
        f"({percentages[0]:.2f}%)"
    )
    print(
        f"0 < r < 1:        {counts[1]:,} "
        f"({percentages[1]:.2f}%)"
    )
    print(
        f"r = 1:            {counts[2]:,} "
        f"({percentages[2]:.2f}%)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, percentages)

    ax.set_ylabel("Percentage of nuclei (%)")
    ax.set_title("Distribution of nucleus–vague overlap")

    ax.set_ylim(0, max(percentages) * 1.15)

    for bar, count, percentage in zip(
        bars,
        counts,
        percentages
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    save_figure(
        fig,
        "02_nucleus_overlap_categories.png"
    )



# 3. DISTRIBUTION AMONG VAGUE-ASSOCIATED NUCLEI
def plot_vague_associated_distribution(nuclei):
    """
    Examine only nuclei with r_i > 0.

    This makes the strong concentration near r_i = 1 visible.
    """

    r_col = get_column(
        nuclei,
        [
            "fraction_of_nucleus_vague",
            "vague_overlap_fraction",
            "overlap_fraction",
            "r_i"
        ]
    )

    r = nuclei[r_col].to_numpy()
    vague_r = r[r > 0]

    mean_r = vague_r.mean()
    median_r = np.median(vague_r)

    fully_contained = np.isclose(vague_r, 1).sum()
    fully_contained_pct = fully_contained / len(vague_r) * 100

    print("\n" + "=" * 80)
    print("VAGUE-ASSOCIATED NUCLEI")
    print("=" * 80)

    print(f"Nuclei with r > 0:       {len(vague_r):,}")
    print(f"Mean r:                  {mean_r:.4f}")
    print(f"Median r:                {median_r:.4f}")
    print(
        f"Completely contained:    {fully_contained:,} "
        f"({fully_contained_pct:.2f}%)"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        vague_r,
        bins=np.linspace(0, 1, 41),
        edgecolor="black"
    )

    ax.axvline(
        mean_r,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_r:.3f}"
    )

    ax.axvline(
        median_r,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_r:.3f}"
    )

    ax.set_xlabel(r"Nucleus–vague overlap $r_i$")
    ax.set_ylabel("Number of nuclei")
    ax.set_title(
        "Nucleus–vague overlap among vague-associated nuclei"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "03_vague_associated_overlap_distribution.png"
    )



# 4. PARTIAL-OVERLAP CASES ONLY
def plot_partial_overlap_distribution(nuclei):
    """
    Look only at nuclei with 0 < r_i < 1.

    There are relatively few of these compared with the large masses at
    r_i = 0 and r_i = 1, so plotting them separately shows the continuous
    part of the overlap distribution.
    """

    r_col = get_column(
        nuclei,
        [
            "fraction_of_nucleus_vague",
            "vague_overlap_fraction",
            "overlap_fraction",
            "r_i"
        ]
    )

    r = nuclei[r_col].to_numpy()

    partial = r[
        (r > 0)
        & (~np.isclose(r, 1))
    ]

    print("\n" + "=" * 80)
    print("PARTIAL-OVERLAP NUCLEI")
    print("=" * 80)

    print(f"Number of partial cases: {len(partial):,}")

    print(
        f"Median r:                "
        f"{np.median(partial):.4f}"
    )

    print(
        f"Mean r:                  "
        f"{partial.mean():.4f}"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        partial,
        bins=np.linspace(0, 1, 41),
        edgecolor="black"
    )

    ax.set_xlabel(r"Nucleus–vague overlap $r_i$")
    ax.set_ylabel("Number of nuclei")
    ax.set_title(
        "Distribution of partial nucleus–vague overlap"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "04_partial_overlap_distribution.png"
    )



# 5. FIND REPRESENTATIVE NUCLEI
def find_representative_nuclei(nuclei):
    """
    Select real nuclei whose r_i values are closest to:

        0.00
        0.25
        0.50
        0.75
        1.00

    These are examples for visual interpretation only. They do not define
    analysis bins.
    """

    r_col = get_column(
        nuclei,
        [
            "fraction_of_nucleus_vague",
            "vague_overlap_fraction",
            "overlap_fraction",
            "r_i"
        ]
    )

    targets = [0, 0.25, 0.50, 0.75, 1]

    selected = []

    for target in targets:

        distances = np.abs(
            nuclei[r_col].to_numpy() - target
        )

        index = np.argmin(distances)
        row = nuclei.iloc[index].copy()

        selected.append(
            (target, row)
        )

    return selected, r_col



# 6. VISUALISE REPRESENTATIVE NUCLEI
def plot_representative_nuclei(nuclei):
    """
    Show nuclei spanning the range of nucleus-vague overlap.

    Each row contains:
    1. H&E crop
    2. target nucleus mask
    3. vague mask
    4. combined overlay

    The nucleus outline is drawn on the H&E image and vague pixels are
    highlighted so the meaning of r_i can be inspected visually.
    """

    selected, r_col = find_representative_nuclei(nuclei)

    tissue_col = get_column(nuclei, ["tissue"])
    sample_col = get_column(
        nuclei,
        ["sample_id", "sample", "image_id"]
    )

    instance_col = get_column(
        nuclei,
        ["instance_id", "nucleus_id", "label_id"]
    )

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(13, 15)
    )

    print("\n" + "=" * 80)
    print("REPRESENTATIVE NUCLEI")
    print("=" * 80)

    for row_index, (target, record) in enumerate(selected):

        tissue = record[tissue_col]
        sample_id = record[sample_col]
        instance_id = int(record[instance_col])
        r_value = float(record[r_col])

        image, instance_mask, vague_mask = load_sample(
            tissue,
            sample_id
        )

        nucleus_mask = instance_mask == instance_id

        if nucleus_mask.sum() == 0:
            raise ValueError(
                f"Instance {instance_id} was not found in "
                f"{tissue} / {sample_id}"
            )

        y0, y1, x0, x1 = get_crop_bounds(
            nucleus_mask,
            padding=35
        )

        image_crop = image[y0:y1, x0:x1]
        nucleus_crop = nucleus_mask[y0:y1, x0:x1]
        vague_crop = vague_mask[y0:y1, x0:x1]

        # Overlay:
        # vague area is highlighted on the H&E image.
        overlay = make_vague_overlay(
            image_crop,
            vague_crop
        )

        print(
            f"Target r ≈ {target:.2f}: "
            f"{tissue} / {sample_id} / "
            f"instance {instance_id} / "
            f"actual r = {r_value:.4f}"
        )

        # H&E
        axes[row_index, 0].imshow(image_crop)

        # Draw target nucleus boundary.
        axes[row_index, 0].contour(
            nucleus_crop,
            levels=[0.5],
            linewidths=1.5
        )

        axes[row_index, 0].set_title(
            f"H&E + target nucleus\n"
            f"$r_i$ = {r_value:.3f}"
        )

        # Nucleus mask
        axes[row_index, 1].imshow(
            nucleus_crop,
            cmap="gray"
        )
        axes[row_index, 1].set_title("Target nucleus $N_i$")

        # Vague mask
        axes[row_index, 2].imshow(
            vague_crop,
            cmap="gray"
        )
        axes[row_index, 2].set_title("Vague mask $V$")

        # Combined
        axes[row_index, 3].imshow(overlay)

        axes[row_index, 3].contour(
            nucleus_crop,
            levels=[0.5],
            linewidths=1.5
        )

        axes[row_index, 3].set_title(
            f"Overlap\n"
            f"{100 * r_value:.1f}% of nucleus vague"
        )

        for ax in axes[row_index]:
            ax.axis("off")

    fig.suptitle(
        "Examples of nucleus–vague overlap",
        fontsize=16
    )

    fig.tight_layout()

    save_figure(
        fig,
        "05_representative_nucleus_overlap.png"
    )





def main():

    print("=" * 80)
    print("VISUALISING NuInsSeg VAGUE OVERLAP")
    print("=" * 80)


    # Load the results
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Corrected inventory not found:\n{INVENTORY_PATH}"
        )

    if not NUCLEUS_OVERLAP_PATH.exists():
        raise FileNotFoundError(
            f"Nucleus-overlap table not found:\n{NUCLEUS_OVERLAP_PATH}"
        )

    inventory = pd.read_csv(INVENTORY_PATH)
    nuclei = pd.read_csv(NUCLEUS_OVERLAP_PATH)

    print(f"\nLoaded {len(inventory):,} image records.")
    print(f"Loaded {len(nuclei):,} nucleus records.")

    print(f"\nFigures will be saved to:\n{OUTPUT_DIR}")


    # Image-level examples
    plot_image_level_extremes(inventory)


    # Nucleus-level overlap distributions
    plot_overlap_distribution(nuclei)
    plot_vague_associated_distribution(nuclei)
    plot_partial_overlap_distribution(nuclei)


    # Actual histological examples showing what different r_i values mean.
    plot_representative_nuclei(nuclei)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()