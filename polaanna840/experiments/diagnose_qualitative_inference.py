
"""
Read-only diagnostic for qualitative inference.

Uses the ORIGINAL evaluation implementation.
Does not modify checkpoints, splits, scores, or selections.

Target:
    Fold 1
    human epiglottis
    human_epiglottis_10.png
    nucleus 11
    LoRA seed 42
"""

import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from experiments import evaluate_heldout as ev


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT / "results/evaluation/manifest.json"
)

SCORES_PATH = (
    ROOT / "results/evaluation/nucleus_scores.csv"
)

TARGET = {
    "iteration": 1,
    "tissue": "human epiglottis",
    "image_id": "human_epiglottis_10.png",
    "nucleus_id": 11,
    "method": "lora",
    "seed": 42,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_manifest():
    require(
        MANIFEST_PATH.is_file(),
        f"Missing manifest: {MANIFEST_PATH}",
    )

    with MANIFEST_PATH.open(
        "r", encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    require(
        manifest.get("status") == "completed",
        "Evaluation manifest is not completed.",
    )

    require(
        "split_sha256" in manifest,
        "Missing split hash in manifest.",
    )

    require(
        "vague_mask_template" in manifest,
        "Missing vague-mask template.",
    )

    return manifest


def load_saved_result():
    require(
        SCORES_PATH.is_file(),
        f"Missing scores: {SCORES_PATH}",
    )

    matches = []

    with SCORES_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if (
                int(row["iteration"])
                == TARGET["iteration"]
                and row["tissue"]
                == TARGET["tissue"]
                and row["image_id"]
                == TARGET["image_id"]
                and int(row["nucleus_id"])
                == TARGET["nucleus_id"]
                and row["method"]
                == TARGET["method"]
                and int(row["seed"])
                == TARGET["seed"]
            ):
                matches.append(row)

    require(
        len(matches) == 1,
        f"Expected one saved result, found {len(matches)}",
    )

    return matches[0]


def find_original_record(folds, template):
    records_by_fold = ev.collect_test_records(
        folds, template
    )

    records = records_by_fold[
        TARGET["iteration"]
    ]

    matches = [
        record
        for record in records
        if (
            record[0] == TARGET["tissue"]
            and record[1].name == TARGET["image_id"]
        )
    ]

    require(
        len(matches) == 1,
        f"Expected one image record, found {len(matches)}",
    )

    return matches[0]


def extract_target(rows):
    matches = [
        row
        for row in rows
        if (
            int(row["nucleus_id"])
            == TARGET["nucleus_id"]
        )
    ]

    require(
        len(matches) == 1,
        f"Expected one nucleus, found {len(matches)}",
    )

    return matches[0]


def print_metrics(label, result):
    print(f"\n{label}")
    print("-" * 55)

    for key in (
        "dice",
        "iou",
        "gt_area",
        "pred_area",
        "intersection",
        "ambiguity_ratio",
    ):
        print(f"{key:20s} {result[key]}")


def compare(saved, actual, label):
    print(f"\nComparison: {label}")
    print("-" * 55)

    for key in (
        "dice",
        "iou",
        "gt_area",
        "pred_area",
        "intersection",
        "ambiguity_ratio",
    ):
        a = float(saved[key])
        b = float(actual[key])

        print(
            f"{key:20s} "
            f"saved={a:.12f} "
            f"new={b:.12f} "
            f"diff={b-a:+.12f}"
        )


def main():
    print("=" * 65)
    print("QUALITATIVE INFERENCE DIAGNOSTIC")
    print("=" * 65)

    require(
        torch.cuda.is_available(),
        "CUDA unavailable. Run on a GPU compute node.",
    )

    device = torch.device("cuda")

    print("\nEnvironment")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    print("\nReading original evaluation manifest...")
    manifest = load_manifest()

    print("Checking original frozen split...")
    folds, actual_split_sha = ev.load_frozen_folds(
        manifest["split_sha256"]
    )

    print("Split SHA256:", actual_split_sha)

    print("\nFinding original image...")
    record = find_original_record(
        folds,
        manifest["vague_mask_template"],
    )

    print("Tissue:", record[0])
    print("Image:", record[1])
    print("Instance mask:", record[2])
    print("Vague mask:", record[3])

    print("\nReading saved evaluation...")
    saved = load_saved_result()

    print_metrics("SAVED EVALUATION", saved)

    print("\nVerifying checkpoint identities...")
    identities = ev.preflight_checkpoints()

    identity = identities[
        (
            TARGET["iteration"],
            TARGET["method"],
            TARGET["seed"],
        )
    ]

    print("Checkpoint:", identity["path"])
    print("Checkpoint SHA256:", identity["sha256"])

    require(
        saved["checkpoint_sha256"].lower()
        == identity["sha256"].lower(),
        "Saved score references a different checkpoint.",
    )

    print("\nLoading original model...")
    model, checkpoint_sha = ev.construct_model(
        TARGET["method"],
        TARGET["iteration"],
        TARGET["seed"],
        device,
        identities,
    )

    require(
        checkpoint_sha.lower()
        == identity["sha256"].lower(),
        "Loaded checkpoint hash mismatch.",
    )

    print("\nRunning original evaluation: PASS 1")

    rows1 = ev.evaluate_one_image(
        model=model,
        record=record,
        iteration=TARGET["iteration"],
        method=TARGET["method"],
        seed=TARGET["seed"],
        checkpoint_sha=checkpoint_sha,
        device=device,
    )

    result1 = extract_target(rows1)

    print_metrics("REGENERATED PASS 1", result1)

    print("\nRunning original evaluation: PASS 2")

    rows2 = ev.evaluate_one_image(
        model=model,
        record=record,
        iteration=TARGET["iteration"],
        method=TARGET["method"],
        seed=TARGET["seed"],
        checkpoint_sha=checkpoint_sha,
        device=device,
    )

    result2 = extract_target(rows2)

    print_metrics("REGENERATED PASS 2", result2)

    compare(saved, result1, "Saved vs pass 1")
    compare(saved, result2, "Saved vs pass 2")
    compare(result1, result2, "Pass 1 vs pass 2")

    dice_saved = float(saved["dice"])
    dice1 = float(result1["dice"])
    dice2 = float(result2["dice"])

    print("\n" + "=" * 65)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 65)

    if (
        dice1 == dice_saved
        and dice2 == dice_saved
    ):
        print(
            "Both original evaluation passes reproduce "
            "the saved Dice exactly."
        )
        print(
            "Investigate the renderer implementation."
        )

    elif dice1 == dice2:
        print(
            "Original evaluation is repeatable "
            "within this GPU execution."
        )
        print(
            "But it does not reproduce "
            "the historical saved Dice."
        )
        print(
            "Investigate cross-run differences "
            "and historical execution settings."
        )

    else:
        print(
            "Original evaluation produces different "
            "Dice scores across repeated passes."
        )
        print(
            "Investigate inference nondeterminism."
        )

    print("\nREAD-ONLY DIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()
