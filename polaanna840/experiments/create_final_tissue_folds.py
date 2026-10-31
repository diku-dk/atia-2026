
from pathlib import Path
import json
import random
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# paths

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "NuInsSeg"
PROFILE_PATH = ROOT / "results" / "tissue_profile.csv"

# New directory, completely separate from previous splits.
OUTPUT_DIR = ROOT / "results" / "splits_final_v1"

SEED = 42
N_FOLDS = 5
CAPACITIES = [7, 6, 6, 6, 6]

EXPECTED_TISSUES = 31
EXPECTED_IMAGES = 665
EXPECTED_NUCLEI = 30698
EXPECTED_VAGUE_NUCLEI = 4980

REQUIRED_COLUMNS = [
    "tissue",
    "species",
    "n_images",
    "n_nuclei",
    "n_vague_nuclei",
]


# data

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"\nSTOP: Output directory already exists:\n{OUTPUT_DIR}\n"
        "No files have been changed. Choose a new version name "
        "only after inspecting the existing outputs."
    )

if not DATA_DIR.is_dir():
    raise FileNotFoundError(DATA_DIR)

if not PROFILE_PATH.is_file():
    raise FileNotFoundError(PROFILE_PATH)

df = pd.read_csv(PROFILE_PATH)

missing = set(REQUIRED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df[REQUIRED_COLUMNS].copy()

if df["tissue"].isna().any():
    raise ValueError("Missing tissue names")

if df["tissue"].duplicated().any():
    raise ValueError("Duplicate tissue names in tissue_profile.csv")

if len(df) != EXPECTED_TISSUES:
    raise ValueError(f"Expected 31 tissues, found {len(df)}")

for col in ["n_images", "n_nuclei", "n_vague_nuclei"]:
    df[col] = pd.to_numeric(df[col], errors="raise")

    if df[col].isna().any():
        raise ValueError(f"Missing values in {col}")

    if (df[col] < 0).any():
        raise ValueError(f"Negative values in {col}")

    if not np.all(df[col] == df[col].astype(int)):
        raise ValueError(f"Non-integer values in {col}")

    df[col] = df[col].astype(int)

if (df["n_vague_nuclei"] > df["n_nuclei"]).any():
    raise ValueError("Vague-associated count exceeds nucleus count")

if df["n_images"].sum() != EXPECTED_IMAGES:
    raise ValueError("Image total does not match 665")

if df["n_nuclei"].sum() != EXPECTED_NUCLEI:
    raise ValueError("Nucleus total does not match 30,698")

if df["n_vague_nuclei"].sum() != EXPECTED_VAGUE_NUCLEI:
    raise ValueError("Vague-associated total does not match 4,980")

# Check the actual tissue directories and image files.
actual_tissues = {
    p.name for p in DATA_DIR.iterdir() if p.is_dir()
}
profile_tissues = set(df["tissue"])

if actual_tissues != profile_tissues:
    raise ValueError(
        f"Tissue mismatch.\n"
        f"Only in dataset: {sorted(actual_tissues - profile_tissues)}\n"
        f"Only in profile: {sorted(profile_tissues - actual_tissues)}"
    )

for row in df.itertuples(index=False):
    image_dir = DATA_DIR / row.tissue / "tissue images"

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    actual_count = sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    if actual_count != row.n_images:
        raise ValueError(
            f"{row.tissue}: profile says {row.n_images} images, "
            f"but directory contains {actual_count}"
        )

df["species"] = df["species"].str.strip().str.lower()

for row in df.itertuples(index=False):
    expected_species = row.tissue.split(" ", 1)[0]
    if row.species != expected_species:
        raise ValueError(
            f"Species mismatch for {row.tissue}: {row.species}"
        )

if not df["species"].isin(["human", "mouse"]).all():
    raise ValueError("Unexpected species value")

print("Dataset validation passed.")
print(f"Tissues: {len(df)}")
print(f"Images: {df['n_images'].sum()}")
print(f"Nuclei: {df['n_nuclei'].sum()}")
print(f"Vague-associated nuclei: {df['n_vague_nuclei'].sum()}")


# making the folds
#
# Predefined method:
#
# - Shuffle tissues with fixed seed for tie-breaking.
# - Sort by image count, largest first.
# - Assign each tissue to the eligible fold currently
#   containing the fewest images.
# - Respect capacities [7, 6, 6, 6, 6].
#


rng = random.Random(SEED)

records = df.to_dict("records")
rng.shuffle(records)

# Python sorting is stable, preserving shuffled order for ties.
records.sort(key=lambda r: r["n_images"], reverse=True)

folds = {f"fold_{i}": [] for i in range(1, 6)}
fold_images = {f"fold_{i}": 0 for i in range(1, 6)}

for record in records:
    eligible = [
        f"fold_{i}"
        for i in range(1, 6)
        if len(folds[f"fold_{i}"]) < CAPACITIES[i - 1]
    ]

    chosen = min(
        eligible,
        key=lambda f: (fold_images[f], int(f.split("_")[1]))
    )

    folds[chosen].append(record["tissue"])
    fold_images[chosen] += record["n_images"]

for fold in folds:
    folds[fold].sort()

assigned = [t for members in folds.values() for t in members]

if len(assigned) != 31 or len(set(assigned)) != 31:
    raise AssertionError("Invalid tissue assignment")

if set(assigned) != profile_tissues:
    raise AssertionError("Missing or unexpected tissues")

if [len(folds[f"fold_{i}"]) for i in range(1, 6)] != CAPACITIES:
    raise AssertionError("Incorrect fold sizes")

# Build a lookup for tissue statistics.
profile = df.set_index("tissue")

assignment_rows = []
summary_rows = []

for i in range(1, 6):
    fold_name = f"fold_{i}"
    tissues = folds[fold_name]
    subset = profile.loc[tissues]

    n_images = int(subset["n_images"].sum())
    n_nuclei = int(subset["n_nuclei"].sum())
    n_vague = int(subset["n_vague_nuclei"].sum())

    summary_rows.append({
        "fold": i,
        "n_tissues": len(tissues),
        "n_images": n_images,
        "n_nuclei": n_nuclei,
        "n_vague_nuclei": n_vague,
        "vague_percentage": 100 * n_vague / n_nuclei,
        "n_human_tissues": int((subset["species"] == "human").sum()),
        "n_mouse_tissues": int((subset["species"] == "mouse").sum()),
    })

    for tissue in tissues:
        row = profile.loc[tissue]
        assignment_rows.append({
            "fold": i,
            "tissue": tissue,
            "species": row["species"],
            "n_images": int(row["n_images"]),
            "n_nuclei": int(row["n_nuclei"]),
            "n_vague_nuclei": int(row["n_vague_nuclei"]),
        })

summary = pd.DataFrame(summary_rows)
assignments = pd.DataFrame(assignment_rows)

assert summary["n_images"].sum() == EXPECTED_IMAGES
assert summary["n_nuclei"].sum() == EXPECTED_NUCLEI
assert summary["n_vague_nuclei"].sum() == EXPECTED_VAGUE_NUCLEI

# Hash identifies this exact assignment.
canonical = json.dumps(folds, sort_keys=True)
split_hash = hashlib.sha256(canonical.encode()).hexdigest()

split_json = {
    "description": (
        "Tissue-exclusive five-fold split. Deterministic "
        "largest-first image-count balancing with seeded "
        "tie-breaking. No ambiguity or model-performance "
        "optimization."
    ),
    "method": "largest_first_greedy_image_balancing",
    "random_seed": SEED,
    "n_folds": N_FOLDS,
    "fold_capacities": CAPACITIES,
    "assignment_sha256": split_hash,
    "folds": folds,
}

print("\nFINAL FOLD SUMMARY")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\nTISSUE ASSIGNMENTS")
for i in range(1, 6):
    print(f"\nFOLD {i}")
    for tissue in folds[f"fold_{i}"]:
        print(f"  {tissue}: {int(profile.loc[tissue, 'n_images'])} images")

print(f"\nAssignment SHA256: {split_hash}")

# figure 1

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

fig1, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(17, 6.5),
    gridspec_kw={"width_ratios": [1.15, 1.25]},
    constrained_layout=True,
)

x = np.arange(1, 6)
nuclei = summary["n_nuclei"].to_numpy()
images = summary["n_images"].to_numpy()
vague = summary["vague_percentage"].to_numpy()

bars = ax1.bar(x, nuclei, color="#347DA7", width=0.68)

for idx, bar in enumerate(bars):
    ax1.annotate(
        f"{images[idx]} images\n"
        f"{nuclei[idx]:,} nuclei\n"
        f"{vague[idx]:.1f}% vague",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax1.set_title("Composition of tissue folds", fontweight="bold")
ax1.set_xlabel("Fold")
ax1.set_ylabel("Number of nuclei")
ax1.set_xticks(x)
ax1.set_ylim(0, max(nuclei) * 1.32)
ax1.spines[["top", "right"]].set_visible(False)

# Rows = tissue folds, columns = CV iterations.
# 0 = training, 1 = validation, 2 = held-out.
#
# Same rotation previously defined:
# Iteration 1: val 2, test 1
# Iteration 2: val 3, test 2
# Iteration 3: val 4, test 3
# Iteration 4: val 5, test 4
# Iteration 5: val 1, test 5

matrix = np.zeros((5, 5), dtype=int)

for iteration in range(5):
    matrix[iteration, iteration] = 2
    matrix[(iteration + 1) % 5, iteration] = 1

cmap = ListedColormap(["#A9D18E", "#FFD966", "#F4B183"])

ax2.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="equal")

names = {0: "PEFT", 1: "VALIDATION", 2: "HELD-OUT"}

for r in range(5):
    for c in range(5):
        ax2.text(
            c, r, names[matrix[r, c]],
            ha="center", va="center",
            fontsize=9, fontweight="bold",
        )

ax2.set_title(
    "Five-fold tissue-level cross-validation",
    fontweight="bold"
)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Tissue fold")
ax2.set_xticks(range(5), range(1, 6))
ax2.set_yticks(range(5), range(1, 6))

ax2.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax2.grid(which="minor", color="white", linewidth=2)
ax2.tick_params(which="minor", bottom=False, left=False)

ax2.legend(
    handles=[
        Patch(facecolor="#A9D18E", label="PEFT training"),
        Patch(facecolor="#FFD966", label="Validation"),
        Patch(facecolor="#F4B183", label="Held-out evaluation"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False,
    fontsize=9,
)

# figure 2

fig2, axes = plt.subplots(
    1, 5,
    figsize=(23, 8),
    sharex=True,
)

max_images = int(df["n_images"].max())

for i, ax in enumerate(axes, start=1):
    fold_name = f"fold_{i}"

    tissue_list = sorted(
        folds[fold_name],
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
        height=0.7,
        color="#347DA7",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(tissue_list, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max_images * 1.25)
    ax.set_xlabel("Number of images", fontsize=10)

    fold_total = int(summary.loc[
        summary["fold"] == i, "n_images"
    ].iloc[0])

    ax.set_title(
        f"Fold {i}\n{fold_total} images",
        fontweight="bold",
        fontsize=13,
        pad=15,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            count + 0.4,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig2.suptitle(
    "Tissue composition by fold",
    fontsize=17,
    fontweight="bold",
    y=0.98,
)
fig2.subplots_adjust(
    left=0.13,
    right=0.98,
    top=0.85,
    bottom=0.12,
    wspace=1.2,
)

#  SAVE EVERYTHING WITHOUT OVERWRITING

# mkdir(exist_ok=False) refuses to reuse an existing directory.
# File modes "x" and "xb" also refuse to overwrite files.

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

with (OUTPUT_DIR / "tissue_folds.json").open(
    "x", encoding="utf-8"
) as f:
    json.dump(split_json, f, indent=4)

with (OUTPUT_DIR / "tissue_folds.csv").open(
    "x", newline="", encoding="utf-8"
) as f:
    assignments.to_csv(f, index=False)

with (OUTPUT_DIR / "fold_summary.csv").open(
    "x", newline="", encoding="utf-8"
) as f:
    summary.to_csv(f, index=False)

with (OUTPUT_DIR / "split_sha256.txt").open(
    "x", encoding="utf-8"
) as f:
    f.write(split_hash + "\n")

with (OUTPUT_DIR / "fold_composition_cv.png").open("xb") as f:
    fig1.savefig(f, format="png", dpi=300, bbox_inches="tight")

with (OUTPUT_DIR / "tissue_distribution.png").open("xb") as f:
    fig2.savefig(f, format="png", dpi=300, bbox_inches="tight")

print("\nALL OUTPUTS SAVED:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" ", p.relative_to(ROOT))

plt.show()
