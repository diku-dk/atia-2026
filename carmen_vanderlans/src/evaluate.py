import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mobile_sam import sam_model_registry, SamPredictor
from src.dataset import load_cheng_sample
from src.preprocessing import preprocess_mri
from src.prompts import mask_to_box
from src.metrics import dice_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cheng_validation_files.csv")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default="results/mobilesam_validation.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Submit this using Slurm.")

    device = "cuda"

    model = sam_model_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )
    model.to(device)
    model.eval()

    predictor = SamPredictor(model)
    files = pd.read_csv(args.csv, dtype={"PID": str})

    if args.limit < 0:
        raise ValueError("--limit must be zero or positive")

    if args.limit > 0:
        files = files.head(args.limit)

    results = []

    for _, row in files.iterrows():
        sample = load_cheng_sample(Path("data/cheng") / row["file"])

        if sample["patient_id"] != row["PID"]:
            raise ValueError(f"Patient ID mismatch: {row['file']}")

        rgb = preprocess_mri(sample["image"])
        box = mask_to_box(sample["mask"])

        with torch.inference_mode():
            predictor.set_image(rgb)
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )

        prediction = masks[0]
        dice = dice_score(prediction, sample["mask"])

        # Save predicted mask for later visualisation
        mask_path = (
            Path("results/predictions/mobilesam_test" if "test" in args.csv else "results/predictions/mobilesam_zero_shot_corrected")
            / Path(row["file"]).with_suffix(".npy")
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, prediction.astype(bool))

        results.append({
            "file": row["file"],
            "PID": sample["patient_id"],
            "label": sample["label"],
            "height": sample["image"].shape[0],
            "width": sample["image"].shape[1],
            "tumour_pixels": int(sample["mask"].sum()),
            "box_area": float(
                (box[2] - box[0] + 1) *
                (box[3] - box[1] + 1)
            ),
            "dice": dice
        })

        print(
            f"{len(results)}/{len(files)} "
            f"{row['file']} Dice={dice:.4f}",
            flush=True
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output, index=False)

    print(f"\nSaved: {output}")
    results_df = pd.DataFrame(results)

    slice_mean = results_df["dice"].mean()
    patient_means = results_df.groupby("PID")["dice"].mean()

    print(f"Mean Dice (per slice): {slice_mean:.4f}")
    print(f"Mean Dice (per patient): {patient_means.mean():.4f}")
    print(f"Median Dice (per slice): {results_df['dice'].median():.4f}")
    print(f"Patients evaluated: {len(patient_means)}")


if __name__ == "__main__":
    main()
