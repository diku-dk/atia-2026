import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import os


# loading one sample
# this is how i tried to do it at first but python was trying to run it from a directory that doesnt exist
# because of how folders are set up (/experiments/data...) which is wrong
# print("Working directory:", os.getcwd())   # with this i checked whats happenign

#image = np.array(Image.open("data/NuInsSeg/human bladder/tissue images/human_bladder_12.png"))
#instance_mask = np.array(Image.open("data/NuInsSeg/human bladder/label masks modify/human_bladder_12.tif"))
#vague_mask = np.array(Image.open("data/NuInsSeg/human bladder/vague areas/mask binary/human_bladder_12.png"))


# better setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "NuInsSeg"

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)



#----------------
# VISUALIZATION 1
# just loading one specific example
tissue_dir = PROJECT_ROOT / "data" / "NuInsSeg" / "human bladder"
image_path = ( tissue_dir / "tissue images" / "human_bladder_12.png" )
instance_path = ( tissue_dir / "label masks" / "human_bladder_12.tif" )
vague_path = ( tissue_dir / "vague areas" / "mask binary" / "human_bladder_12.png" )


# loading in
image = np.array(Image.open(image_path))
instance_mask = np.array(Image.open(instance_path)) # a mask where different objects/nuclei have different IDs
vague_mask = np.array(Image.open(vague_path)) # a mask showing which pixels belong to ambiguous region


# starting with one specific sample
# all unique nucleus IDs
# but remove 0 because 0 represents the background
instance_ids = np.unique(instance_mask)
instance_ids = instance_ids[instance_ids != 0]

print("Image")
print("  shape:", image.shape)
print("  dtype:", image.dtype)

print("\nInstance mask")
print("  shape:", instance_mask.shape)
print("  dtype:", instance_mask.dtype)
print("  instance IDs:", instance_ids)
print("  number of nuclei:", len(instance_ids))

print("\nVague mask")
print("  shape:", vague_mask.shape)
print("  dtype:", vague_mask.dtype)
print("  unique values:", np.unique(vague_mask))



# Visualizing


# what does the H&E image look like?
# what are the instance labels?
# where are the vague regions?

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. original H&E
# which is actual input img -> purple/dark shapes are nuclei, SAM will see this and try to segment later
axes[0, 0].imshow(image)
axes[0, 0].set_title("H&E image")

# 2. instance mask -> ground truth answer for the nuclei provided by NuInsSeg
# the mask is 512×512 array of numbers, where 0 is background, 1 is nucleus #1 etc... (they are given different colors)
# its useful cause here we have 19 nuclei, its not a binary segmentation mask (0 or 1)
axes[0, 1].imshow(instance_mask, cmap="nipy_spectral")
axes[0, 1].set_title(f"Instance mask ({len(instance_ids)} nuclei)")

# 3. vague-area mask
# black is normal, non-vague area
# white is area marked as ambigious
# its completely separate from the other mask cayse this is where the image/annotation is considered vague
axes[0, 2].imshow(vague_mask, cmap="gray")
axes[0, 2].set_title("Vague-area mask")


# boolean vague mask
vague = vague_mask > 0

# 4. H&E + vague overlay
# so the actual tissue image + locations that are marked as vague
# shows what is happening in the H&E image that caused the annotators to call this area vague
axes[1, 0].imshow(image)
axes[1, 0].imshow(
    vague,
    cmap="Reds",
    alpha=np.where(vague, 0.5, 0)
)
axes[1, 0].set_title("H&E + vague areas")


# 5. nuclei + vague overlay
# black is background, white is some nucleus, red is vague region drawn over it
axes[1, 1].imshow(instance_mask > 0, cmap="gray")
axes[1, 1].imshow(
    vague,
    cmap="Reds",
    alpha=np.where(vague, 0.6, 0)
)
axes[1, 1].set_title("Nuclei + vague areas")


# 6. only nuclei pixels that overlap vague areas
# here its similat to the other one but a pixel has to satisfy both:
# is this pixel nucleus?    and     is this pixel vague?
overlap = (instance_mask > 0) & vague

axes[1, 2].imshow(instance_mask > 0, cmap="gray")
axes[1, 2].imshow(
    overlap,
    cmap="Reds",
    alpha=np.where(overlap, 0.8, 0)
)
axes[1, 2].set_title("Nuclei pixels inside vague areas")


for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()



# ------------------------------------

# VISUALIZATION 2
# examples from several different tissues
# better for the paper

N_TISSUES = 3 # 3 examples
RANDOM_SEED = 42 # fixed so that i get the same examples every time

OUTPUT_PATH = FIGURE_DIR / "07_dataset_examples.png"


# Helper function
def get_files(folder):
    """Return image files from a folder."""

    extensions = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    }

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in extensions
        ]
    )


# Find all tissue folders
all_tissues = sorted(
    [
        folder.name
        for folder in DATA_ROOT.iterdir()
        if folder.is_dir()
    ]
)

print(
    f"Found {len(all_tissues)} tissue folders."
)


# Select random tissues reproducibly
random.seed(RANDOM_SEED)
selected_tissues = random.sample(all_tissues, N_TISSUES)


print("\nSelected tissues:")
for tissue in selected_tissues:
    print(f"  - {tissue}")


# Create figure
fig, axes = plt.subplots(N_TISSUES,4, figsize=(11, 8))

# Reduce space between images
plt.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.92,
    bottom=0.04,
    wspace=0.08,
    hspace=0.28
)


# loop through tissues
for row, tissue in enumerate(selected_tissues):

    tissue_folder = DATA_ROOT / tissue

    image_folder = tissue_folder / "tissue images"
    label_folder = tissue_folder / "label masks modify"
    vague_folder = tissue_folder / "vague areas" / "mask binary"


    # select one image
    image_files = get_files(image_folder)
    # in case smth is wrong
    if not image_files:
        raise FileNotFoundError(f"No images found in:\n{image_folder}")

    image_path = random.choice(image_files)
    stem = image_path.stem


    # Find corresponding masks
    label_path = next(
        file
        for file in get_files(label_folder)
        if file.stem == stem
    )


    vague_path = next(
        file
        for file in get_files(vague_folder)
        if file.stem == stem
    )


    # load
    image = np.array(Image.open(image_path))
    label = np.array(Image.open(label_path))
    vague = np.array(Image.open(vague_path))

    vague_binary = vague > 0

    # Column 1: H&E
    axes[row, 0].imshow(image)
    axes[row, 0].axis("off")

    display_name = (
        tissue
        .replace("_", " ")
        .title()
    )

    axes[row, 0].text(
        -0.12,
        0.5,
        display_name,
        transform=axes[row, 0].transAxes,
        fontsize=11,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center"
    )

    # Column 2: instance mask
    axes[row, 1].imshow(
        label,
        cmap="nipy_spectral"
    )
    axes[row, 1].axis("off")

    # Column 3: vague mask
    axes[row, 2].imshow(
        vague_binary,
        cmap="gray"
    )
    axes[row, 2].axis("off")

    # Column 4: vague overlay
    axes[row, 3].imshow(image)

    axes[row, 3].imshow(
        np.ma.masked_where(
            ~vague_binary,
            vague_binary
        ),
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    axes[row, 3].axis("off")

# Column titles
axes[0, 0].set_title("H&E", fontsize=12)
axes[0, 1].set_title("Instance ground truth", fontsize=12)
axes[0, 2].set_title("Vague mask", fontsize=12)
axes[0, 3].set_title("Vague regions on H&E", fontsize=12)


# Save
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nSaved figure to:\n{OUTPUT_PATH}")
print(f"\nRandom seed: {RANDOM_SEED}")