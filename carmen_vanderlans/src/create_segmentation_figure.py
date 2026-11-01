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

EXAMPLES = [
    ("Meningioma: good", "brainTumorDataPublic_1-766/234.mat"),
    ("Glioma: improvement", "brainTumorDataPublic_2299-3064/2595.mat"),
    ("Glioma: difficult", "brainTumorDataPublic_767-1532/829.mat"),
    ("Meningioma: MedSAM failure", "brainTumorDataPublic_1-766/639.mat"),
    ("Pituitary: difficult", "brainTumorDataPublic_1533-2298/1741.mat"),
]

COLUMNS = [
    "MRI", "Ground truth", "MobileSAM",
    "MedSAM", "LoRA", "Adapter"
]

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

samples = []
for label, filename in EXAMPLES:
    sample = load_cheng_sample(Path("data/cheng") / filename)
    samples.append((label, filename, sample))

predictions = {}
recorded_scores = {
    "MobileSAM": pd.read_csv("results/baselines/mobilesam_test.csv").set_index("file"),
    "MedSAM": pd.read_csv("results/baselines/medsam_test.csv").set_index("file"),
    "LoRA": pd.read_csv(
        "results/qualitative/lora_25_seed_1.csv"
    ).set_index("file"),
    "Adapter": pd.read_csv(
        "results/qualitative/adapter_25_seed_1.csv"
    ).set_index("file"),
}


def predict_mobile(method):
    model = mobile_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )

    if method == "LoRA":
        model = apply_lora(model, rank=4)
    elif method == "Adapter":
        model = apply_adapter(model)

    if method in ("LoRA", "Adapter"):
        name = method.lower()
        checkpoint_path = Path(
            f"experiment_v2/training/{name}_budget_25_seed_1/best.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])

    model = model.cuda().eval()
    predictor = SamPredictor(model)

    with torch.inference_mode():
        for _, filename, sample in samples:
            image = preprocess_mri(sample["image"])
            box = mask_to_box(sample["mask"])
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                box=box, multimask_output=False
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
    ).cuda().eval()

    with torch.inference_mode():
        for _, filename, sample in samples:
            img = sample["image"]
            H, W = img.shape

            rgb = np.repeat(img[:, :, None], 3, axis=-1)

            img_1024 = resize(
                rgb, (1024, 1024),
                order=3,
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.float32)

            img_1024 = (
                (img_1024 - img_1024.min()) /
                max(float(img_1024.max() - img_1024.min()), 1e-8)
            )

            tensor = torch.as_tensor(
                img_1024.copy(),
                dtype=torch.float32,
                device="cuda"
            ).permute(2, 0, 1).unsqueeze(0)

            box = np.asarray(
                mask_to_box(sample["mask"]), dtype=np.float32
            )
            box_1024 = box / np.array(
                [W, H, W, H], dtype=np.float32
            ) * 1024

            box_tensor = torch.as_tensor(
                box_1024[None, None, :],
                dtype=torch.float32,
                device="cuda"
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
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

            prediction = (
                probabilities.squeeze().cpu().numpy() > 0.5
            )
            predictions[("MedSAM", filename)] = prediction.copy()

    del model
    gc.collect()
    torch.cuda.empty_cache()


def draw_boundary(ax, mask, color, width=1.5):
    if np.any(mask):
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=width
        )


print("Generating MobileSAM predictions...", flush=True)
predict_mobile("MobileSAM")

print("Generating LoRA predictions...", flush=True)
predict_mobile("LoRA")

print("Generating Adapter predictions...", flush=True)
predict_mobile("Adapter")

print("Generating MedSAM predictions...", flush=True)
predict_medsam()

fig, axes = plt.subplots(
    len(samples), len(COLUMNS),
    figsize=(17, 15),
    constrained_layout=True
)

scores = []

for i, (description, filename, sample) in enumerate(samples):
    image = sample["image"]
    truth = sample["mask"]

    for j, column in enumerate(COLUMNS):
        ax = axes[i, j]
        ax.imshow(image, cmap="gray")
        ax.axis("off")

        if j == 0:
            ax.set_ylabel(description, fontsize=10)

        if column == "Ground truth":
            draw_boundary(ax, truth, "lime", 1.8)

        elif column not in ("MRI", "Ground truth"):
            prediction = predictions[(column, filename)]

            if prediction.shape != truth.shape:
                raise ValueError(
                    f"Shape mismatch: {column}, {filename}"
                )

            draw_boundary(ax, truth, "lime", 1.3)
            draw_boundary(ax, prediction, "magenta", 1.3)

            dice = dice_score(prediction, truth)
            expected = float(recorded_scores[column].loc[filename, "dice"])
            difference = abs(dice - expected)

            print(
                f"{Path(filename).stem:>5} | {column:9s} | "
                f"Dice {dice:.5f} | recorded {expected:.5f} | "
                f"diff {difference:.6f}",
                flush=True
            )

            if difference > 1e-3:
                raise ValueError(
                    f"Dice does not match original evaluation: "
                    f"{column}, {filename}"
                )

            ax.text(
                0.5, -0.045,
                f"Dice = {dice:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9
            )

            scores.append({
                "file": filename,
                "example": description,
                "method": column,
                "dice": dice
            })

        if i == 0:
            ax.set_title(column, fontsize=11, fontweight="bold")

    axes[i, 0].text(
        0.02, 0.04,
        description,
        transform=axes[i, 0].transAxes,
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none")
    )

fig.suptitle(
    "Qualitative comparison of brain tumour segmentation",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.5, 0.01,
    "Green: ground-truth boundary     "
    "Magenta: predicted boundary     "
    "Fine-tuned models: 25% budget, seed 1",
    ha="center",
    fontsize=10
)

png = OUTPUT_DIR / "segmentation_comparison.png"
pdf = OUTPUT_DIR / "segmentation_comparison.pdf"
csv = OUTPUT_DIR / "segmentation_comparison_scores.csv"

fig.savefig(png, dpi=250, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

pd.DataFrame(scores).to_csv(csv, index=False)

print(f"\nSaved: {png}")
print(f"Saved: {pdf}")
print(f"Saved: {csv}")
