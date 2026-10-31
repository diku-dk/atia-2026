
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(r"C:\Users\H P\Desktop\NuInsSeg-MobileSAM")
DATA = ROOT / "data" / "NuInsSeg"
ARCHIVES = ROOT / "results" / "qualitative" / "render_v3" / "predicted_masks"

CASES = [
    {
        "tissue": "human tongue",
        "image_id": "human_tongue_31.png",
        "fold": 4,
        "nucleus": 48,
        "archive": "36_fold4_nucleus48.npz",
    },
    {
        "tissue": "mouse kidney",
        "image_id": "mouse_kidney_38.png",
        "fold": 5,
        "nucleus": 18,
        "archive": "38_fold5_nucleus18.npz",
    },
    {
        "tissue": "human pancreas",
        "image_id": "human_pancreas_05.png",
        "fold": 3,
        "nucleus": 11,
        "archive": "41_fold3_nucleus11.npz",
    },
]

SEED = 42
CROP_SIZE = 220
COLUMN_FONT = 25
ROW_FONT = 25
SCORE_FONT = 21


def dice_score(prediction, ground_truth):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    denominator = prediction.sum() + ground_truth.sum()

    if denominator == 0:
        return 1.0

    intersection = np.logical_and(prediction, ground_truth).sum()
    return float(2 * intersection / denominator)


def load_image(tissue, image_id):
    path = DATA / tissue / "tissue images" / image_id

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def get_crop(box, image_shape):
    height, width = image_shape[:2]

    x0, y0, x1, y1 = map(float, box)

    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2

    size = min(CROP_SIZE, width, height)

    left = round(center_x - size / 2)
    top = round(center_y - size / 2)

    left = max(0, min(left, width - size))
    top = max(0, min(top, height - size))

    return left, top, left + size, top + size


def crop_array(array, crop):
    left, top, right, bottom = crop
    return array[top:bottom, left:right]


def overlay(ax, mask, color, alpha):
    mask = mask.astype(bool)

    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(float) * alpha

    ax.imshow(rgba, interpolation="nearest")


def load_case(case):
    he = load_image(case["tissue"], case["image_id"])
    archive_path = ARCHIVES / case["archive"]

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with np.load(archive_path, allow_pickle=False) as data:
        gt = data["ground_truth"].astype(bool).copy()
        vague = data["vague_region"].astype(bool).copy()
        box = data["bounding_box"].copy()

        zs = data["zero_shot"].astype(bool).copy()
        lora = data[f"lora_{SEED}"].astype(bool).copy()
        adapter = data[f"adapter_{SEED}"].astype(bool).copy()

    if not all(
        array.shape == he.shape[:2]
        for array in [gt, vague, zs, lora, adapter]
    ):
        raise ValueError(f"Dimension mismatch: {case['archive']}")

    dice = [
        dice_score(zs, gt),
        dice_score(lora, gt),
        dice_score(adapter, gt),
    ]

    crop = get_crop(box, he.shape)

    return {
        "he": crop_array(he, crop),
        "gt": crop_array(gt, crop),
        "vague": crop_array(vague, crop),
        "predictions": [
            crop_array(zs, crop),
            crop_array(lora, crop),
            crop_array(adapter, crop),
        ],
        "dice": dice,
    }


def main():
    loaded = [load_case(case) for case in CASES]

    fig = plt.figure(figsize=(25, 13), facecolor="white")

    grid = fig.add_gridspec(
        nrows=3,
        ncols=5,
        left=0.09,
        right=0.985,
        bottom=0.065,
        top=0.87,
        wspace=0.035,
        hspace=0.32,
    )

    column_titles = [
        "H&E image",
        "GT + vague",
        "Zero-shot",
        "LoRA",
        "Adapter",
    ]

    for row, (case, data) in enumerate(zip(CASES, loaded)):
        he = data["he"]
        gt = data["gt"]
        vague = data["vague"]

        first_image_ax = None

        for col in range(5):
            ax = fig.add_subplot(grid[row, col])

            if col == 0:
                first_image_ax = ax

            ax.imshow(
                he,
                interpolation="nearest",
                aspect="equal",
            )

            if col == 1:
                overlay(
                    ax,
                    vague,
                    color=(0.85, 0.0, 0.75),
                    alpha=0.35,
                )

                overlay(
                    ax,
                    gt,
                    color=(0.0, 0.9, 0.2),
                    alpha=0.65,
                )

            elif col >= 2:
                prediction = data["predictions"][col - 2]

                overlay(
                    ax,
                    gt,
                    color=(0.0, 0.9, 0.2),
                    alpha=0.40,
                )

                overlay(
                    ax,
                    prediction,
                    color=(1.0, 0.55, 0.0),
                    alpha=0.55,
                )

                ax.text(
                    0.5,
                    -0.06,
                    f"Dice = {data['dice'][col - 2]:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=SCORE_FONT,
                    fontweight="bold",
                    clip_on=False,
                )

            ax.set_xlim(-0.5, he.shape[1] - 0.5)
            ax.set_ylim(he.shape[0] - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.axis("off")

            if row == 0:
                ax.set_title(
                    column_titles[col],
                    fontsize=COLUMN_FONT,
                    fontweight="bold",
                    pad=18,
                )

        # Center the vertical tissue name using the actual image position.
        # No extra label axes, so the text cannot shift between rows.
        pos = first_image_ax.get_position()

        fig.text(
            pos.x0 - 0.018,
            (pos.y0 + pos.y1) / 2,
            case["tissue"].title(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=ROW_FONT,
            fontweight="bold",
        )

        print(
            f"{case['tissue'].title()} | "
            f"Fold {case['fold']} | "
            f"Nucleus {case['nucleus']} | "
            f"Zero-shot={data['dice'][0]:.3f} | "
            f"LoRA={data['dice'][1]:.3f} | "
            f"Adapter={data['dice'][2]:.3f}"
        )

    fig.suptitle(
        "Qualitative comparison (LoRA and Adapter: seed 42)",
        fontsize=27,
        fontweight="bold",
        y=0.985,
    )

    plt.show()


if __name__ == "__main__":
    main()
