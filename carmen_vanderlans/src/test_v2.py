"""Evaluate trained MobileSAM models on the untouched Cheng test set."""

import argparse
import csv
from pathlib import Path

import torch
from mobile_sam import sam_model_registry

from src.adapter import apply_adapter
from src.lora import apply_lora
from src.train_full import ChengDataset, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "adapter"], required=True)
    parser.add_argument("--budget", choices=["10", "25", "50", "100", "10_fewpatients"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3, 4, 5], required=True)
    args = parser.parse_args()

    checkpoint_path = Path(
        f"experiment_v2/training/{args.method}_budget_{args.budget}_seed_{args.seed}/best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = sam_model_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )

    if args.method == "lora":
        model = apply_lora(model, rank=4)
    else:
        model = apply_adapter(model)

    model = model.to("cuda")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    test_data = ChengDataset("data/cheng_test_files.csv")

    with torch.no_grad():
        slice_dice, patient_dice = validate(model, test_data)

    output_path = Path(f"experiment_v2/test_results/test_{args.method}_{args.budget}_seed_{args.seed}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "method", "budget", "seed", "best_epoch",
                "slice_dice", "patient_dice"
            ])
        writer.writerow([
            args.method, args.budget, args.seed,
            checkpoint["epoch"], slice_dice, patient_dice
        ])

    print(
        f"TEST RESULT: {args.method} | {args.budget} | "
        f"seed {args.seed} | slice Dice {slice_dice:.5f} | "
        f"patient Dice {patient_dice:.5f}",
        flush=True
    )


if __name__ == "__main__":
    main()
