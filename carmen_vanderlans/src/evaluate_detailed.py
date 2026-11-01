import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.train_v2 import (
    ChengDataset,
    load_cheng_sample,
    preprocess_mri,
    mask_to_box,
    dice_score,
    sam_model_registry,
    apply_lora,
    apply_adapter,
)
from mobile_sam import SamPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["lora", "adapter"])
    parser.add_argument("--budget", required=True,
                        choices=["10", "25", "50", "100", "10_fewpatients"])
    parser.add_argument("--seed", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--checkpoint-root", default="experiment_v2/training")
    parser.add_argument("--output-root", default="experiment_extended/detailed_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    checkpoint_path = (
        Path(args.checkpoint_root)
        / f"{args.method}_budget_{args.budget}_seed_{args.seed}/best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = sam_model_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )

    model = (
        apply_lora(model, rank=4)
        if args.method == "lora"
        else apply_adapter(model)
    ).cuda()

    checkpoint = torch.load(
        checkpoint_path, map_location="cuda", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictor = SamPredictor(model)
    dataset = ChengDataset("data/cheng_test_files.csv")

    output = (
        Path(args.output_root)
        / f"{args.method}_{args.budget}_seed_{args.seed}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with torch.no_grad():
        for row in dataset.rows:
            sample = load_cheng_sample(Path("data/cheng") / row["file"])

            if sample["patient_id"] != row["PID"]:
                raise ValueError(f"Patient mismatch: {row['file']}")

            image = preprocess_mri(sample["image"])
            box = mask_to_box(sample["mask"])

            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                box=box, multimask_output=False
            )

            dice = dice_score(masks[0], sample["mask"])

            results.append({
                "method": args.method,
                "budget": args.budget,
                "seed": args.seed,
                "file": row["file"],
                "PID": row["PID"],
                "label": sample["label"],
                "dice": float(dice),
            })

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    slice_mean = np.mean([r["dice"] for r in results])
    patient_scores = {}
    for r in results:
        patient_scores.setdefault(r["PID"], []).append(r["dice"])
    patient_mean = np.mean([
        np.mean(scores) for scores in patient_scores.values()
    ])

    print(f"Saved {len(results)} image scores to {output}")
    print(f"Slice Dice: {slice_mean:.6f}")
    print(f"Patient Dice: {patient_mean:.6f}")


if __name__ == "__main__":
    main()
