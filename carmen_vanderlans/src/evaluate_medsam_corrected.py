"""Zero-shot MedSAM evaluation on the Cheng test set."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.transform import resize

sys.path.insert(0, str(Path("MedSAM").resolve()))
from segment_anything import sam_model_registry

from src.dataset import load_cheng_sample
from src.prompts import mask_to_box
from src.metrics import dice_score


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cheng_test_files.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="results/medsam_test.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; run through Slurm.")

    model = sam_model_registry["vit_b"](
        checkpoint="MedSAM/weights/medsam_vit_b.pth"
    ).cuda().eval()

    files = pd.read_csv(args.csv, dtype={"PID": str})
    if args.limit > 0:
        files = files.head(args.limit)

    results = []

    for _, row in files.iterrows():
        sample = load_cheng_sample(Path("data/cheng") / row["file"])

        if sample["patient_id"] != row["PID"]:
            raise ValueError(f"Patient mismatch: {row['file']}")

        img = sample["image"]
        mask = sample["mask"]
        H, W = img.shape

        rgb = np.repeat(img[:, :, None], 3, axis=-1)

        img_1024 = resize(
            rgb,
            (1024, 1024),
            order=3,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)
        img_1024 = (
            (img_1024 - img_1024.min()) /
            max(float(img_1024.max() - img_1024.min()), 1e-8)
        )

        tensor = torch.as_tensor(
            img_1024.copy(), dtype=torch.float32,
            device="cuda"
        ).permute(2, 0, 1).unsqueeze(0)

        box = np.asarray(mask_to_box(mask), dtype=np.float32)
        box_1024 = box / np.array(
            [W, H, W, H], dtype=np.float32
        ) * 1024

        box_tensor = torch.as_tensor(
            box_1024[None, None, :],
            dtype=torch.float32,
            device="cuda"
        )

        image_embedding = model.image_encoder(tensor)

        sparse, dense = model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None
        )

        logits, _ = model.mask_decoder(
            image_embeddings=image_embedding,
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

        dice = dice_score(prediction, mask)

        results.append({
            "file": row["file"],
            "PID": row["PID"],
            "label": row["label"],
            "dice": dice
        })

        if len(results) % 50 == 0:
            print(f"Evaluated {len(results)}/{len(files)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)

    df = pd.DataFrame(results)
    print(f"Mean slice Dice: {df['dice'].mean():.5f}")
    print(
        "Mean patient Dice: "
        f"{df.groupby('PID')['dice'].mean().mean():.5f}"
    )
    print(f"Patients: {df['PID'].nunique()}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
