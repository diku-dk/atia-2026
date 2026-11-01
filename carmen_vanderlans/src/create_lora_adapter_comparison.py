import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.transform import resize

from mobile_sam import sam_model_registry as mobile_registry, SamPredictor
from src.dataset import load_cheng_sample
from src.preprocessing import preprocess_mri
from src.prompts import mask_to_box
from src.metrics import dice_score
from src.lora import apply_lora
from src.adapter import apply_adapter


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError(
        "CUDA GPU required. Run this script inside an srun GPU session."
    )

# (a) = Adapter better than LoRA
# (b) = LoRA better than Adapter
EXAMPLES = [
    ("a", "brainTumorDataPublic_2299-3064/2763.mat"),
    ("b", "brainTumorDataPublic_1533-2298/2050.mat"),
]

COLUMNS = [
    "Original MRI",
    "Ground truth",
    "MobileSAM",
    "MedSAM",
    "LoRA",
    "Adapter",
]

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helper paths
# ---------------------------------------------------------------------

def first_existing(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these paths exist:\n" + "\n".join(str(p) for p in paths)
    )


RECORDED_SCORE_PATHS = {
    "MobileSAM": [
        "results/baselines/mobilesam_test.csv",
    ],
    "MedSAM": [
        "results/baselines/medsam_test.csv",
        "experiment_medsam/medsam_test.csv",
    ],
    "LoRA": [
        "results/qualitative/lora_25_seed_1.csv",
    ],
    "Adapter": [
        "results/qualitative/adapter_25_seed_1.csv",
    ],
}

LORA_CHECKPOINT = first_existing([
    "experiment_v2/training/lora_budget_25_seed_1/best.pt",
    "results/training/lora_budget_25_seed_1/best.pt",
])

ADAPTER_CHECKPOINT = first_existing([
    "experiment_v2/training/adapter_budget_25_seed_1/best.pt",
    "results/training/adapter_budget_25_seed_1/best.pt",
])


# ---------------------------------------------------------------------
# Load samples and recorded scores
# ---------------------------------------------------------------------

samples = []
for panel, filename in EXAMPLES:
    sample = load_cheng_sample(Path("data/cheng") / filename)
    samples.append((panel, filename, sample))

recorded_scores = {
    key: pd.read_csv(first_existing(paths)).set_index("file")
    for key, paths in RECORDED_SCORE_PATHS.items()
}

predictions = {}


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------

def predict_mobile(method):
    model = mobile_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )

    if method == "LoRA":
        model = apply_lora(model, rank=4)
    elif method == "Adapter":
        model = apply_adapter(model)

    if method == "LoRA":
        checkpoint = torch.load(
            LORA_CHECKPOINT, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
    elif method == "Adapter":
        checkpoint = torch.load(
            ADAPTER_CHECKPOINT, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])

    model = model.to(DEVICE).eval()
    predictor = SamPredictor(model)

    with torch.inference_mode():
        for _, filename, sample in samples:
            image = preprocess_mri(sample["image"])
            box = mask_to_box(sample["mask"])
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )
            predictions[(method, filename)] = masks[0].astype(bool).copy()

    del predictor, model
    gc.collect()
    torch.cuda.empty_cache()


def predict_medsam():
    sys.path.insert(0, str(Path("MedSAM").resolve()))
    from segment_anything import sam_model_registry as med_registry

    model = med_registry["vit_b"](
        checkpoint="MedSAM/weights/medsam_vit_b.pth"
    ).to(DEVICE).eval()

    with torch.inference_mode():
        for _, filename, sample in samples:
            img = sample["image"]
            h, w = img.shape

            rgb = np.repeat(img[:, :, None], 3, axis=-1)
            img_1024 = resize(
                rgb,
                (1024, 1024),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)

            img_1024 = (
                (img_1024 - img_1024.min()) /
                max(float(img_1024.max() - img_1024.min()), 1e-8)
            )

            tensor = torch.as_tensor(
                img_1024.copy(),
                dtype=torch.float32,
                device=DEVICE
            ).permute(2, 0, 1).unsqueeze(0)

            box = np.asarray(
                mask_to_box(sample["mask"]),
                dtype=np.float32
            )
            box_1024 = box / np.array([w, h, w, h], dtype=np.float32) * 1024
            box_tensor = torch.as_tensor(
                box_1024[None, None, :],
                dtype=torch.float32,
                device=DEVICE
            )

            embedding = model.image_encoder(tensor)
            sparse, dense = model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None
            )
            logits, _ = model.mask_decoder(
                image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False
            )

            probabilities = F.interpolate(
                torch.sigmoid(logits),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )
            prediction = probabilities.squeeze().cpu().numpy() > 0.5
            predictions[("MedSAM", filename)] = prediction.copy()

    del model
    gc.collect()
    torch.cuda.empty_cache()


def draw_boundary(ax, mask, color, width=1.6):
    if np.any(mask):
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=width
        )


# ---------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------

print("Generating MobileSAM predictions...", flush=True)
predict_mobile("MobileSAM")

print("Generating LoRA predictions...", flush=True)
predict_mobile("LoRA")

print("Generating Adapter predictions...", flush=True)
predict_mobile("Adapter")

print("Generating MedSAM predictions...", flush=True)
predict_medsam()


# ---------------------------------------------------------------------
# Plot figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(samples),
    ncols=len(COLUMNS),
    figsize=(13.8, 4.6)
)

if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

scores = []

for i, (panel_label, filename, sample) in enumerate(samples):
    image = sample["image"]
    truth = sample["mask"]

    for j, column in enumerate(COLUMNS):
        ax = axes[i, j]
        ax.imshow(image, cmap="gray")
        ax.axis("off")

        if i == 0:
            ax.set_title(column, fontsize=10, fontweight="bold", pad=6)

        if column == "Ground truth":
            draw_boundary(ax, truth, "lime", 1.8)

        elif column not in ("Original MRI", "Ground truth"):
            prediction = predictions[(column, filename)]

            if prediction.shape != truth.shape:
                raise ValueError(
                    f"Shape mismatch for {column}, {filename}: "
                    f"{prediction.shape} vs {truth.shape}"
                )

            draw_boundary(ax, truth, "lime", 1.4)
            draw_boundary(ax, prediction, "magenta", 1.4)

            dice = dice_score(prediction, truth)
            expected = float(recorded_scores[column].loc[filename, "dice"])
            diff = abs(dice - expected)

            print(
                f"{Path(filename).stem:>5} | {column:9s} | "
                f"Dice {dice:.5f} | recorded {expected:.5f} | "
                f"diff {diff:.6f}",
                flush=True
            )

            if diff > 1e-3:
                raise ValueError(
                    f"Dice does not match recorded evaluation for "
                    f"{column}, {filename}"
                )

            ax.text(
                0.5,
                -0.10,
                f"Dice = {dice:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5
            )

            scores.append({
                "panel": panel_label,
                "file": filename,
                "method": column,
                "dice": dice
            })

    # (a) / (b) labels on the left, like the old figure
    axes[i, 0].text(
        -0.23,
        0.5,
        f"({panel_label})",
        transform=axes[i, 0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

# Small legend-like note at the bottom
fig.text(
    0.5,
    0.045,
    "Green: ground-truth boundary   |   Magenta: predicted boundary",
    ha="center",
    va="center",
    fontsize=8
)

plt.subplots_adjust(
    left=0.06,
    right=0.995,
    top=0.86,
    bottom=0.16,
    wspace=0.04,
    hspace=0.18
)

png_path = OUTPUT_DIR / "lora_adapter_comparison.png"
pdf_path = OUTPUT_DIR / "lora_adapter_comparison.pdf"
csv_path = OUTPUT_DIR / "lora_adapter_comparison_scores.csv"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

pd.DataFrame(scores).to_csv(csv_path, index=False)

print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")
print(f"Saved: {csv_path}")