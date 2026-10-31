from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# Paths

ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATH = ROOT / "results" / "splits_final_v1" / "tissue_folds.json"
PROFILE_PATH = ROOT / "results" / "tissue_profile.csv"

for path in (SPLIT_PATH, PROFILE_PATH):
    if not path.is_file():
        raise FileNotFoundError(path)


# Load the frozen split

with SPLIT_PATH.open("r", encoding="utf-8") as f:
    split = json.load(f)

df = pd.read_csv(PROFILE_PATH)

folds = split["folds"]
profile = df.set_index("tissue")

expected_keys = [f"fold_{i}" for i in range(1, 6)]

if set(folds) != set(expected_keys):
    raise ValueError("Unexpected fold keys.")

assigned = [
    tissue
    for key in expected_keys
    for tissue in folds[key]
]

if len(assigned) != 31 or len(set(assigned)) != 31:
    raise ValueError("Expected 31 unique tissues.")

if not profile.index.is_unique:
    raise ValueError("Duplicate tissues found in tissue profile.")

if set(assigned) != set(profile.index):
    raise ValueError("Split and tissue profile do not match.")

canonical = json.dumps(folds, sort_keys=True)
actual_hash = hashlib.sha256(
    canonical.encode("utf-8")
).hexdigest()

expected_hash = split.get("assignment_sha256")

if expected_hash and actual_hash != expected_hash:
    raise ValueError("Frozen assignment hash mismatch.")


# Calculate fold statistics

summary_rows = []

for i in range(1, 6):
    subset = profile.loc[folds[f"fold_{i}"]]

    n_images = int(subset["n_images"].sum())
    n_nuclei = int(subset["n_nuclei"].sum())
    n_vague = int(subset["n_vague_nuclei"].sum())

    summary_rows.append({
        "fold": i,
        "n_tissues": len(folds[f"fold_{i}"]),
        "n_images": n_images,
        "n_nuclei": n_nuclei,
        "vague_percentage": 100 * n_vague / n_nuclei,
    })

summary = pd.DataFrame(summary_rows)

if summary["n_images"].sum() != 665:
    raise ValueError("Unexpected total image count.")

if summary["n_nuclei"].sum() != 30698:
    raise ValueError("Unexpected total nucleus count.")

print("Frozen split verified.")
print(summary.to_string(index=False))


# Plot styling

BLUE = "#347DA7"
GREEN = "#A9D18E"
YELLOW = "#FFD966"
ORANGE = "#F4B183"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "axes.titlesize": 25,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})


# Figure 1: Fold composition and cross-validation

fig1, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(21, 9),
    gridspec_kw={"width_ratios": [1.18, 1]},
)

fig1.subplots_adjust(
    left=0.075,
    right=0.975,
    top=0.87,
    bottom=0.22,
    wspace=0.28,
)


# Left panel: Fold composition

x = np.arange(1, 6)

nuclei = summary["n_nuclei"].to_numpy()
images = summary["n_images"].to_numpy()
vague = summary["vague_percentage"].to_numpy()

bars = ax1.bar(
    x,
    nuclei,
    color=BLUE,
    width=0.58,
)

for i, bar in enumerate(bars):
    ax1.annotate(
        f"{images[i]} images\n"
        f"{nuclei[i]:,} nuclei\n"
        f"{vague[i]:.1f}% vague",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()
        ),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=17,
        linespacing=1.15,
    )

ax1.set_title(
    "Composition of tissue folds",
    fontsize=25,
    pad=22,
)

ax1.set_xlabel(
    "Fold",
    fontsize=24,
    labelpad=12,
)

ax1.set_ylabel(
    "Number of nuclei",
    fontsize=24,
    labelpad=12,
)

ax1.set_xticks(x)
ax1.set_xlim(0.4, 5.6)
ax1.set_ylim(0, max(nuclei) * 1.55)

ax1.tick_params(
    axis="both",
    labelsize=20,
)

ax1.spines[["top", "right"]].set_visible(False)


# Right panel: Cross-validation matrix

matrix = np.zeros((5, 5), dtype=int)

for iteration in range(5):
    matrix[iteration, iteration] = 2
    matrix[(iteration + 1) % 5, iteration] = 1

cmap = ListedColormap([GREEN, YELLOW, ORANGE])

ax2.imshow(
    matrix,
    cmap=cmap,
    vmin=0,
    vmax=2,
    aspect="equal",
)

labels = {
    0: "Train",
    1: "Val",
    2: "Test",
}

for r in range(5):
    for c in range(5):
        ax2.text(
            c,
            r,
            labels[matrix[r, c]],
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

ax2.set_title(
    "Five-fold tissue-level cross-validation",
    fontsize=25,
    pad=22,
)

ax2.set_xlabel(
    "Iteration",
    fontsize=24,
    labelpad=12,
)

ax2.set_ylabel(
    "Tissue fold",
    fontsize=24,
    labelpad=12,
)

ax2.set_xticks(range(5))
ax2.set_xticklabels(range(1, 6))

ax2.set_yticks(range(5))
ax2.set_yticklabels(range(1, 6))

ax2.tick_params(
    axis="both",
    labelsize=20,
)

ax2.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, 5, 1), minor=True)

ax2.grid(
    which="minor",
    color="white",
    linewidth=2.5,
)

ax2.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

ax2.legend(
    handles=[
        Patch(facecolor=GREEN, label="PEFT training"),
        Patch(facecolor=YELLOW, label="Validation"),
        Patch(facecolor=ORANGE, label="Held-out evaluation"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=3,
    frameon=False,
    fontsize=16,
    columnspacing=1.1,
    handlelength=1.3,
)


# Figure 2: Tissue composition by fold

fig2, axes = plt.subplots(
    5, 1,
    figsize=(15, 24),
    sharex=True,
)

max_images = int(df["n_images"].max())

for i, ax in enumerate(axes, start=1):

    tissue_list = sorted(
        folds[f"fold_{i}"],
        key=lambda t: int(profile.loc[t, "n_images"]),
        reverse=True,
    )

    counts = [
        int(profile.loc[t, "n_images"])
        for t in tissue_list
    ]

    positions = np.arange(len(tissue_list))

    bars = ax.barh(
        positions,
        counts,
        height=0.68,
        color=BLUE,
    )

    ax.set_yticks(positions)

    ax.set_yticklabels(
        [t.replace("_", " ") for t in tissue_list],
        fontsize=17,
    )

    ax.invert_yaxis()
    ax.set_xlim(0, max_images * 1.22)

    fold_total = int(
        summary.loc[
            summary["fold"] == i,
            "n_images"
        ].iloc[0]
    )

    ax.set_title(
        f"Fold {i} — {fold_total} images",
        fontsize=23,
        pad=12,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            count + 0.6,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=18,
        )

    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    ax.spines[["top", "right"]].set_visible(False)

    ax.tick_params(
        axis="x",
        labelsize=20,
    )

axes[-1].set_xlabel(
    "Number of images",
    fontsize=24,
    labelpad=12,
)

fig2.suptitle(
    "Tissue composition by fold",
    fontsize=26,
    y=0.99,
)

fig2.subplots_adjust(
    left=0.40,
    right=0.96,
    top=0.94,
    bottom=0.055,
    hspace=0.70,
)


# Display only

plt.show()