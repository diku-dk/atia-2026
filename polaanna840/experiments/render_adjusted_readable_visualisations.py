
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image




ROOT = Path(r"C:\Users\H P\Desktop\NuInsSeg-MobileSAM")

DATA = ROOT / "data" / "NuInsSeg"

SELECTION = (
    ROOT / "results" / "qualitative"
    / "selection_v1" / "selected_cases.csv"
)

ARCHIVES = (
    ROOT / "results" / "qualitative"
    / "render_v3" / "predicted_masks"
)


# CASES TO DISPLAY
CASES = [
    ("human epiglottis", 1, 21),
    ("human tongue", 4, 48),
    ("mouse kidney", 5, 18),
    ("mouse heart", 3, 22),
    ("human tongue", 4, 28),
    ("human pancreas", 3, 11),
    ("human pancreas", 3, 39),
    ("human salivory gland", 2, 23),
]

SEED = 42

# Set to False to display the complete 512x512 images
CROP_AROUND_NUCLEUS = True

# Square crop, in original image pixels
CROP_SIZE = 220

FONT_SIZE = 27
HEADER_SIZE = 28

# Display each figure separately
SHOW_FIGURES = True


# BASIC FUNCTIONS
def dice_score(prediction, ground_truth):
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(ground_truth).astype(bool)

    denominator = int(pred.sum()) + int(gt.sum())

    if denominator == 0:
        return 1.0

    intersection = int(np.logical_and(pred, gt).sum())

    return 2.0 * intersection / denominator


def normalize_text(value):
    return str(value).strip().lower().replace("_", " ")


def load_he_image(tissue, image_id):
    """
    Load the actual H&E image, not a similarly named
    annotation or distance map.
    """
    filename = Path(str(image_id)).name

    path = DATA / tissue / "tissue images" / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"H&E image not found:\n{path}"
        )

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))

    return rgb, path


def get_crop(box, image_shape):
    """
    One common square crop for all five panels.
    Centered on the saved bounding-box prompt.
    """
    height, width = image_shape[:2]

    if not CROP_AROUND_NUCLEUS:
        return (0, 0, width, height)

    x0, y0, x1, y1 = map(float, box)

    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0

    size = min(CROP_SIZE, height, width)

    left = round(center_x - size / 2)
    top = round(center_y - size / 2)

    left = max(0, min(left, width - size))
    top = max(0, min(top, height - size))

    return (left, top, left + size, top + size)


def crop_array(array, bounds):
    left, top, right, bottom = bounds
    return array[top:bottom, left:right]


def add_overlay(ax, mask, color, alpha):
    """
    Display a transparent mask overlay without
    modifying the underlying H&E image.
    """
    mask = np.asarray(mask).astype(bool)

    rgba = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.float32
    )

    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask.astype(float) * alpha

    ax.imshow(rgba, interpolation="nearest")


# MATCH CASES TO SELECTION RECORDS
def find_case_record(df, tissue, fold, nucleus_id):
    """
    Identify a nucleus using tissue, fold and nucleus ID.

    If more than one source image matches, do not guess.
    """
    tissue_values = df["tissue"].map(normalize_text)

    matches = df[
        (tissue_values == normalize_text(tissue))
        & (pd.to_numeric(df["iteration"]) == fold)
        & (pd.to_numeric(df["nucleus_id"]) == nucleus_id)
    ]

    if len(matches) != 1:
        print(
            f"\nSKIPPED: {tissue}, fold {fold}, "
            f"nucleus {nucleus_id}"
        )
        print(
            f"Matching selection records: {len(matches)}"
        )

        if len(matches):
            print(matches.to_string(index=False))

        return None

    return matches.iloc[0]


# MATCH SAVED PREDICTIONS
def find_archive(record):
    """
    Find the correct archive.

    When multiple archives have the same fold and nucleus
    ID, compare the stored ambiguity ratio against the
    selection record, then require a unique match.

    No archive is changed.
    """
    fold = int(record["iteration"])
    nucleus_id = int(record["nucleus_id"])
    expected_r = float(record["ambiguity_ratio"])

    candidates = sorted(
        ARCHIVES.glob(
            f"*_fold{fold}_nucleus{nucleus_id}.npz"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No prediction archive for fold {fold}, "
            f"nucleus {nucleus_id}"
        )

    valid = []

    for path in candidates:
        with np.load(path, allow_pickle=False) as data:
            gt = data["ground_truth"].astype(bool)
            vague = data["vague_region"].astype(bool)

            if not gt.any():
                continue

            calculated_r = (
                np.logical_and(gt, vague).sum() / gt.sum()
            )

            if abs(calculated_r - expected_r) < 1e-6:
                valid.append(path)

    if len(valid) != 1:
        raise RuntimeError(
            f"Cannot uniquely match archive for "
            f"fold {fold}, nucleus {nucleus_id}.\n"
            f"Candidates: {[p.name for p in candidates]}\n"
            f"Matching ambiguity ratio: "
            f"{[p.name for p in valid]}"
        )

    return valid[0]



# PLOT ONE CASE

def plot_case(record, archive_path):
    tissue = normalize_text(record["tissue"])
    image_id = str(record["image_id"])
    fold = int(record["iteration"])
    nucleus_id = int(record["nucleus_id"])
    ambiguity = float(record["ambiguity_ratio"])

    he, image_path = load_he_image(tissue, image_id)

    with np.load(archive_path, allow_pickle=False) as data:
        gt = data["ground_truth"].astype(bool).copy()
        vague = data["vague_region"].astype(bool).copy()
        box = data["bounding_box"].copy()

        zs = data["zero_shot"].astype(bool).copy()
        lora = data[f"lora_{SEED}"].astype(bool).copy()
        adapter = data[f"adapter_{SEED}"].astype(bool).copy()

    # Validate dimensions before plotting.
    shapes = [
        he.shape[:2],
        gt.shape,
        vague.shape,
        zs.shape,
        lora.shape,
        adapter.shape,
    ]

    if len(set(shapes)) != 1:
        raise ValueError(
            f"Image/mask dimensions do not match: {shapes}"
        )

    # Dice is calculated on the full 512x512 masks.
    dice_zs = dice_score(zs, gt)
    dice_lora = dice_score(lora, gt)
    dice_adapter = dice_score(adapter, gt)

    # Same crop for every panel.
    crop = get_crop(box, he.shape)

    he_c = crop_array(he, crop)
    gt_c = crop_array(gt, crop)
    vague_c = crop_array(vague, crop)

    zs_c = crop_array(zs, crop)
    lora_c = crop_array(lora, crop)
    adapter_c = crop_array(adapter, crop)

    panels = [
        ("H&E image", None),
        ("GT + vague", None),
        (f"Zero-shot\nDice = {dice_zs:.3f}", zs_c),
        (
            f"LoRA (seed {SEED})\nDice = {dice_lora:.3f}",
            lora_c
        ),
        (
            f"Adapter (seed {SEED})\nDice = {dice_adapter:.3f}",
            adapter_c
        ),
    ]

    fig, axes = plt.subplots(
        1, 5,
        figsize=(28, 6.8),
        layout="constrained"
    )

    for index, (ax, (title, prediction)) in enumerate(
        zip(axes, panels)
    ):
        ax.imshow(
            he_c,
            interpolation="nearest",
            aspect="equal"
        )

        if index == 1:
            # Vague-region annotation: magenta
            add_overlay(
                ax,
                vague_c,
                color=(0.85, 0.0, 0.75),
                alpha=0.35
            )

            # Ground truth: green
            add_overlay(
                ax,
                gt_c,
                color=(0.0, 0.9, 0.2),
                alpha=0.65
            )

        elif prediction is not None:
            # Ground truth: green
            add_overlay(
                ax,
                gt_c,
                color=(0.0, 0.9, 0.2),
                alpha=0.40
            )

            # Prediction: orange
            add_overlay(
                ax,
                prediction,
                color=(1.0, 0.55, 0.0),
                alpha=0.55
            )

        ax.set_title(
            title,
            fontsize=FONT_SIZE,
            fontweight="bold",
            pad=16
        )

        ax.set_xlim(-0.5, he_c.shape[1] - 0.5)
        ax.set_ylim(he_c.shape[0] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"{tissue.title()} | Fold {fold} | "
        f"Nucleus {nucleus_id} | r = {ambiguity:.3f}",
        fontsize=HEADER_SIZE,
        fontweight="bold"
    )

    print("\n" + "=" * 70)
    print(f"{tissue} | fold {fold} | nucleus {nucleus_id}")
    print("Image:", image_path)
    print("Archive:", archive_path.name)
    print(f"Ambiguity ratio: {ambiguity:.6f}")
    print(f"Zero-shot Dice: {dice_zs:.6f}")
    print(f"LoRA seed {SEED} Dice: {dice_lora:.6f}")
    print(f"Adapter seed {SEED} Dice: {dice_adapter:.6f}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)







def main():
    if not SELECTION.is_file():
        raise FileNotFoundError(SELECTION)

    if not ARCHIVES.is_dir():
        raise FileNotFoundError(ARCHIVES)

    df = pd.read_csv(SELECTION)

    print("\nAll salivary gland records:")
    print(
        df[
            df.astype(str).apply(
                lambda col: col.str.contains(
                    "salivary", case=False, na=False
                )
            ).any(axis=1)
        ].to_string(index=False)
    )

    print("\nAll fold 2, nucleus 23 records:")
    print(
        df[
            (pd.to_numeric(df["iteration"], errors="coerce") == 2)
            & (pd.to_numeric(df["nucleus_id"], errors="coerce") == 23)
            ].to_string(index=False)
    )


    print(f"Loaded {len(df)} selected nuclei")
    print(f"Available prediction archives: "
          f"{len(list(ARCHIVES.glob('*.npz')))}")

    for tissue, fold, nucleus_id in CASES:
        record = find_case_record(
            df, tissue, fold, nucleus_id
        )

        if record is None:
            continue

        try:
            archive_path = find_archive(record)
            plot_case(record, archive_path)

        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            KeyError
        ) as error:
            print(
                f"\nSKIPPED {tissue}, fold {fold}, "
                f"nucleus {nucleus_id}:\n{error}"
            )


if __name__ == "__main__":
    main()
