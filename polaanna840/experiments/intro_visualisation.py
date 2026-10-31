from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb


DATASET = Path(
    r"C:\Users\H P\Desktop\NuInsSeg-MobileSAM\data\NuInsSeg"
)


examples = [
    {
        "name": "Human spleen",
        "tissue": "human spleen",
        "file": "human_spleen_01"
    },
    {
        "name": "Mouse femur",
        "tissue": "mouse femur",
        "file": "mouse_femur_01"
    }
]



def load(path):
    return np.array(Image.open(path))


def colour_instance_mask(label_mask):

    label_mask = np.squeeze(np.asarray(label_mask))

    ids = np.unique(label_mask)
    ids = ids[ids != 0]

    rgb = np.zeros(
        (*label_mask.shape, 3),
        dtype=float
    )

    if len(ids) == 0:
        return rgb

    hues = np.linspace(
        0,
        1,
        len(ids),
        endpoint=False
    )

    rng = np.random.default_rng(42)
    rng.shuffle(hues)

    for instance_id, hue in zip(ids, hues):

        colour = hsv_to_rgb([
            hue,
            0.85,
            1.0
        ])

        rgb[label_mask == instance_id] = colour

    return rgb


fig, axes = plt.subplots(
    nrows=2,
    ncols=6,
    figsize=(20, 7)
)


titles = [
    "H&E image",
    "Instance labels",
    "Binary mask",
    "Border-removed\nbinary mask",
    "Distance map",
    "Weighted binary\nmask"
]


for col, title in enumerate(titles):

    axes[0, col].set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=15
    )


for row, example in enumerate(examples):

    tissue = DATASET / example["tissue"]
    name = example["file"]


    image_path = (
        tissue
        / "tissue images"
        / f"{name}.png"
    )

    label_path = (
        tissue
        / "label masks modify"
        / f"{name}.tif"
    )

    binary_path = (
        tissue
        / "mask binary"
        / f"{name}.png"
    )

    border_path = (
        tissue
        / "mask binary without border"
        / f"{name}.png"
    )

    distance_path = (
        tissue
        / "distance maps"
        / f"{name}.png"
    )

    weighted_path = (
        tissue
        / "weighted_maps"
        / f"{name}.png"
    )

    image = load(image_path)
    labels = load(label_path)
    binary = load(binary_path)
    border_removed = load(border_path)
    distance = load(distance_path)
    weighted = load(weighted_path)

#colors
    labels_rgb = colour_instance_mask(labels)



    axes[row, 0].imshow(image)
    axes[row, 1].imshow(labels_rgb)
    axes[row, 2].imshow(
        binary,
        cmap="gray"
    )

    axes[row, 3].imshow(
        border_removed,
        cmap="gray"
    )

    axes[row, 4].imshow(
        distance,
        cmap="gray"
    )

    axes[row, 5].imshow(
        weighted,
        cmap="gray"
    )



    axes[row, 0].set_ylabel(
        example["name"],
        fontsize=18,
        fontweight="bold",
        rotation=90,
        labelpad=28
    )



    for col in range(6):

        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

        for spine in axes[row, col].spines.values():
            spine.set_visible(False)


plt.subplots_adjust(
    left=0.10,
    right=0.995,
    top=0.86,
    bottom=0.04,
    wspace=0.035,
    hspace=0.07
)


plt.show()







