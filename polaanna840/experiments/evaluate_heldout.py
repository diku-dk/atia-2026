
import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from mobile_sam import SamPredictor

from models.base import load_mobile_sam
from models.lora import build_lora
from models.adapter import build_adapter
from experiments.nucleus_dataset_loader import (
    load_nuinsseg_sample,
    get_nucleus_ids,
    extract_nucleus,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "NuInsSeg"
SPLIT_FILE = ROOT / "results" / "splits_final_v1" / "tissue_folds.json"
CHECKPOINT_ROOT = ROOT / "results/training/final_full_nuclei"

SEEDS = (42, 123, 456)
METHODS = ("zero_shot", "lora", "adapter")
PROMPT_BATCH_SIZE = 8
EXPECTED_TISSUES = 31
EXPECTED_IMAGES = 665
EXPECTED_NUCLEI = 30698

FIELDS = [
    "iteration", "test_fold", "tissue", "image_id",
    "nucleus_id", "method", "seed", "checkpoint_sha256",
    "dice", "iou", "precision", "recall",
    "gt_area", "pred_area", "intersection",
    "ambiguity_ratio", "ambiguity_group",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message):
    raise RuntimeError(message)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_frozen_folds(expected_sha):
    if not SPLIT_FILE.is_file():
        fail(f"Frozen split missing: {SPLIT_FILE}")

    actual_sha = sha256(SPLIT_FILE)

    if actual_sha.lower() != expected_sha.lower():
        fail(
            "FROZEN SPLIT HASH MISMATCH\n"
            f"Expected: {expected_sha}\n"
            f"Actual:   {actual_sha}\n"
            f"Path:     {SPLIT_FILE}"
        )

    data = read_json(SPLIT_FILE)
    folds = data["folds"]

    expected_keys = {f"fold_{i}" for i in range(1, 6)}
    if set(folds) != expected_keys:
        fail("Frozen split does not contain exactly five expected folds.")

    all_tissues = []
    for i in range(1, 6):
        tissues = folds[f"fold_{i}"]
        if not isinstance(tissues, list) or not tissues:
            fail(f"Invalid or empty fold_{i}.")
        if not all(isinstance(t, str) for t in tissues):
            fail(f"Invalid tissue names in fold_{i}.")
        all_tissues.extend(tissues)

    if len(all_tissues) != EXPECTED_TISSUES:
        fail(f"Expected 31 tissue assignments; found {len(all_tissues)}.")

    if len(set(all_tissues)) != EXPECTED_TISSUES:
        fail("Tissue overlap detected between folds.")

    for tissue in all_tissues:
        if not (DATA_ROOT / tissue).is_dir():
            fail(f"Missing tissue directory: {tissue}")

    return folds, actual_sha


def ambiguity_group(r):
    if r == 0:
        return "none"
    if r == 1:
        return "full"
    if 0 < r < 1:
        return "partial"
    fail(f"Invalid ambiguity ratio: {r}")


def vague_mask_path(template, tissue, image_path):
    # Explicit template; never infer a vague-mask directory.
    allowed = {
        "data_root": str(DATA_ROOT),
        "tissue": tissue,
        "stem": image_path.stem,
        "filename": image_path.name,
    }
    try:
        path = Path(template.format(**allowed))
    except (KeyError, ValueError) as exc:
        fail(f"Invalid vague-mask template: {exc}")

    if not path.is_absolute():
        fail("Vague-mask template must resolve to an absolute path.")

    return path


def load_vague_mask(path, expected_shape):
    if not path.is_file():
        fail(f"Missing vague-region mask: {path}")

    with Image.open(path) as image:
        vague = np.asarray(image)

    if vague.ndim != 2:
        fail(
            f"Vague mask must be single-channel: {path}, "
            f"shape={vague.shape}"
        )

    if vague.shape != expected_shape:
        fail(
            f"Vague-mask shape mismatch: {path}, "
            f"{vague.shape} != {expected_shape}"
        )

    values = set(np.unique(vague).tolist())

    # Only established binary encodings are accepted.
    # Other label conventions must be explicitly handled,
    # not silently interpreted as nonzero = vague.
    if not values.issubset({0, 1, 255}):
        fail(
            f"Unexpected vague-mask values in {path}: "
            f"{sorted(values)[:20]}"
        )

    if 1 in values and 255 in values:
        fail(f"Mixed vague-mask encodings in {path}")

    return vague != 0


def collect_test_records(folds, vague_template):
    records_by_fold = {}
    total_images = 0
    seen_images = set()

    for iteration in range(1, 6):
        records = []

        for tissue in sorted(folds[f"fold_{iteration}"]):
            image_dir = DATA_ROOT / tissue / "tissue images"
            mask_dir = DATA_ROOT / tissue / "label masks modify"

            if not image_dir.is_dir() or not mask_dir.is_dir():
                fail(f"Missing image or instance-mask directory: {tissue}")

            images = sorted(image_dir.glob("*.png"))
            if not images:
                fail(f"No images found for tissue: {tissue}")

            for image_path in images:
                mask_path = mask_dir / f"{image_path.stem}.tif"
                vague_path = vague_mask_path(
                    vague_template, tissue, image_path
                )

                if not mask_path.is_file():
                    fail(f"Missing instance mask: {mask_path}")
                if not vague_path.is_file():
                    fail(f"Missing vague mask: {vague_path}")

                key = (tissue, image_path.name)
                if key in seen_images:
                    fail(f"Duplicate image across folds: {key}")
                seen_images.add(key)

                records.append(
                    (tissue, image_path, mask_path, vague_path)
                )

        records_by_fold[iteration] = records
        total_images += len(records)

    if total_images != EXPECTED_IMAGES:
        fail(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {total_images}."
        )

    return records_by_fold


def checkpoint_path(iteration, method, seed):
    return (
        CHECKPOINT_ROOT
        / f"iteration_{iteration}"
        / method
        / f"seed_{seed}"
        / "best_model.pt"
    )


def validate_checkpoint_metadata(config, iteration, method, seed):
    expected = {
        "method": method,
        "iteration": iteration,
        "seed": seed,
        "lora_rank": 4,
        "lora_alpha": 4,
        "adapter_dim": 16,
        "prompt_batch_size": 8,
        "max_train_nuclei": None,
    }

    for key, value in expected.items():
        if key not in config or config[key] != value:
            fail(
                f"Checkpoint configuration mismatch: {key}. "
                f"Expected {value!r}, found {config.get(key)!r}"
            )


def preflight_checkpoints():
    identities = {}

    for iteration in range(1, 6):
        for method in ("lora", "adapter"):
            for seed in SEEDS:
                path = checkpoint_path(iteration, method, seed)

                if not path.is_file() or path.stat().st_size == 0:
                    fail(f"Missing/empty checkpoint: {path}")

                config_path = path.parent / "config.json"
                completed_path = path.parent / "completed.json"

                if not config_path.is_file():
                    fail(f"Missing configuration: {config_path}")
                if not completed_path.is_file():
                    fail(f"Missing completion record: {completed_path}")

                config = read_json(config_path)
                completed = read_json(completed_path)

                validate_checkpoint_metadata(
                    config, iteration, method, seed
                )

                for key, expected in (
                    ("method", method),
                    ("iteration", iteration),
                    ("seed", seed),
                    ("status", "completed"),
                ):
                    if completed.get(key) != expected:
                        fail(f"Completion metadata mismatch: {completed_path}")

                identities[(iteration, method, seed)] = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "config": config,
                    "completed": completed,
                }

    if len(identities) != 30:
        fail("Expected exactly 30 PEFT checkpoint identities.")

    return identities


def construct_model(method, iteration, seed, device, identities):
    if method == "zero_shot":
        model = load_mobile_sam(device=device)
        checkpoint_sha = ""
    elif method == "lora":
        model, _ = build_lora(
            rank=4, alpha=4, device=device
        )
    elif method == "adapter":
        model, _ = build_adapter(
            bottleneck_dim=16, device=device
        )
    else:
        fail(f"Unknown method: {method}")

    model.to(device)

    if method != "zero_shot":
        identity = identities[(iteration, method, seed)]
        path = Path(identity["path"])

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            fail(f"Invalid checkpoint structure: {path}")

        if "config" not in checkpoint:
            fail(f"Checkpoint lacks embedded config: {path}")

        validate_checkpoint_metadata(
            checkpoint["config"], iteration, method, seed
        )

        if checkpoint["config"] != identity["config"]:
            fail(f"Embedded and external configs differ: {path}")

        if "model_state_dict" not in checkpoint:
            fail(f"Missing model state: {path}")

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        checkpoint_sha = identity["sha256"]
        del checkpoint

    model.eval()
    return model, checkpoint_sha


def segmentation_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.shape != gt.shape:
        fail(f"Mask shape mismatch: {pred.shape} != {gt.shape}")

    gt_area = int(np.count_nonzero(gt))
    pred_area = int(np.count_nonzero(pred))
    intersection = int(np.count_nonzero(pred & gt))

    if gt_area == 0:
        fail("Encountered an empty ground-truth nucleus.")

    dice = 2 * intersection / (gt_area + pred_area)
    union = gt_area + pred_area - intersection
    iou = intersection / union

    # Since GT is nonempty, empty predictions receive
    # zero precision, recall, Dice, and IoU.
    precision = intersection / pred_area if pred_area else 0.0
    recall = intersection / gt_area

    metrics = (dice, iou, precision, recall)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in metrics):
        fail("Invalid segmentation metric.")

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "intersection": intersection,
    }


@torch.inference_mode()
def evaluate_one_image(
    model, record, iteration, method, seed,
    checkpoint_sha, device
):
    tissue, image_path, mask_path, vague_path = record

    image, instance_mask = load_nuinsseg_sample(
        image_path, mask_path
    )

    if image.ndim != 3 or image.shape[2] != 3:
        fail(f"Expected RGB image: {image_path}")

    if instance_mask.shape != image.shape[:2]:
        fail(f"Instance-mask shape mismatch: {mask_path}")

    vague = load_vague_mask(vague_path, instance_mask.shape)

    nucleus_ids = get_nucleus_ids(instance_mask).tolist()
    if not nucleus_ids:
        fail(f"No annotated nuclei: {image_path}")

    if len(set(nucleus_ids)) != len(nucleus_ids):
        fail(f"Duplicate nucleus IDs: {image_path}")

    boxes = []
    targets = []
    ambiguities = []

    for nucleus_id in nucleus_ids:
        gt, bbox = extract_nucleus(instance_mask, nucleus_id)
        gt = np.asarray(gt, dtype=bool)

        if gt.shape != instance_mask.shape:
            fail(f"Invalid extracted nucleus shape: {image_path}")

        gt_area = int(np.count_nonzero(gt))
        if gt_area == 0:
            fail(f"Empty nucleus {nucleus_id}: {image_path}")

        overlap = int(np.count_nonzero(gt & vague))
        r = overlap / gt_area

        boxes.append(bbox)
        targets.append(gt)
        ambiguities.append((r, ambiguity_group(r)))

    boxes = np.stack(boxes)

    predictor = SamPredictor(model)

    transformed_image = predictor.transform.apply_image(image)
    image_tensor = torch.as_tensor(
        transformed_image,
        device=device,
    ).permute(2, 0, 1).contiguous().unsqueeze(0)

    embedding = model.image_encoder(
        model.preprocess(image_tensor)
    )

    input_size = transformed_image.shape[:2]
    original_size = image.shape[:2]
    threshold = float(model.mask_threshold)

    rows = []

    for start in range(0, len(nucleus_ids), PROMPT_BATCH_SIZE):
        end = min(start + PROMPT_BATCH_SIZE, len(nucleus_ids))

        transformed_boxes = predictor.transform.apply_boxes(
            boxes[start:end], original_size
        )

        box_tensor = torch.as_tensor(
            transformed_boxes,
            dtype=torch.float32,
            device=device,
        )

        sparse, dense = model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        logits = model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )

        expected_shape = (
            end - start, 1, *original_size
        )
        if tuple(logits.shape) != expected_shape:
            fail(
                f"Unexpected decoder output: "
                f"{tuple(logits.shape)} != {expected_shape}"
            )

        predictions = (
            logits[:, 0] > threshold
        ).cpu().numpy()

        for j, prediction in enumerate(predictions):
            idx = start + j
            nucleus_id = int(nucleus_ids[idx])
            r, group = ambiguities[idx]

            metrics = segmentation_metrics(
                prediction, targets[idx]
            )

            rows.append({
                "iteration": iteration,
                "test_fold": iteration,
                "tissue": tissue,
                "image_id": image_path.name,
                "nucleus_id": nucleus_id,
                "method": method,
                "seed": "" if seed is None else seed,
                "checkpoint_sha256": checkpoint_sha,
                **metrics,
                "ambiguity_ratio": r,
                "ambiguity_group": group,
            })

    if len(rows) != len(nucleus_ids):
        fail(f"Incomplete nucleus evaluation: {image_path}")

    return rows


def write_json_new(path, data):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-split-sha",
        required=True,
        help="SHA-256 of the frozen split used during training.",
    )
    parser.add_argument(
        "--vague-template",
        required=True,
        help=(
            "Absolute path template using {data_root}, "
            "{tissue}, {stem}, or {filename}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="A NEW directory that must not already exist.",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()

    if output.exists():
        fail(f"REFUSING TO OVERWRITE EXISTING PATH: {output}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        fail(
            "Full evaluation requires a CUDA allocation. "
            "Run through Slurm on a GPU compute node."
        )

    print("Checking frozen split...", flush=True)
    folds, split_sha = load_frozen_folds(
        args.expected_split_sha
    )

    print("Checking dataset paths...", flush=True)
    records_by_fold = collect_test_records(
        folds, args.vague_template
    )

    print("Checking 30 checkpoint identities...", flush=True)
    identities = preflight_checkpoints()

    # Create a completely new output directory.
    # No pre-existing directory is accepted.
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_path": str(SPLIT_FILE),
        "split_sha256": split_sha,
        "frozen_folds": folds,
        "vague_mask_template": args.vague_template,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "mask_rule": "postprocessed_logits > model.mask_threshold",
        "metric_definition": "hard binary per-nucleus Dice",
        "checkpoint_identities": {
            f"{i}_{m}_{s}": info
            for (i, m, s), info in identities.items()
        },
    }

    write_json_new(output / "manifest.json", manifest)

    scores_path = output / "nucleus_scores.csv"
    counts = {}
    expected_total_rows = EXPECTED_NUCLEI * 7
    written = 0

    # CSV is opened in exclusive-create mode.
    with open(
        scores_path, "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for iteration in range(1, 6):
            records = records_by_fold[iteration]

            conditions = [("zero_shot", None)]
            conditions += [
                (method, seed)
                for method in ("lora", "adapter")
                for seed in SEEDS
            ]

            for method, seed in conditions:
                print(
                    f"\nFold {iteration}/5 | {method} | "
                    f"seed={seed} | {len(records)} images",
                    flush=True,
                )

                model, checkpoint_sha = construct_model(
                    method, iteration, seed, device, identities
                )

                count = 0
                dice_sum = 0.0

                for index, record in enumerate(records, start=1):
                    rows = evaluate_one_image(
                        model=model,
                        record=record,
                        iteration=iteration,
                        method=method,
                        seed=seed,
                        checkpoint_sha=checkpoint_sha,
                        device=device,
                    )

                    writer.writerows(rows)

                    count += len(rows)
                    written += len(rows)
                    dice_sum += sum(r["dice"] for r in rows)

                    if index % 25 == 0 or index == len(records):
                        print(
                            f"  {index}/{len(records)} images | "
                            f"{count} nuclei",
                            flush=True,
                        )
                        f.flush()

                counts[f"{iteration}_{method}_{seed}"] = count

                print(
                    f"Completed: {count} nuclei | "
                    f"mean nucleus Dice={dice_sum / count:.6f}",
                    flush=True,
                )

                del model
                torch.cuda.empty_cache()

    # Every fold must contain the same nuclei across methods.
    for iteration in range(1, 6):
        zs = counts[f"{iteration}_zero_shot_None"]

        for method in ("lora", "adapter"):
            for seed in SEEDS:
                other = counts[f"{iteration}_{method}_{seed}"]
                if other != zs:
                    fail(
                        f"Nucleus count mismatch: fold {iteration}, "
                        f"{method}, seed {seed}: {other} != {zs}"
                    )

    if written != expected_total_rows:
        fail(
            f"Expected {expected_total_rows} score rows, "
            f"wrote {written}."
        )

    # Verify that nucleus identifiers and ambiguity values
    # match exactly across all seven model conditions.
    seen = {}
    per_nucleus_counts = {}

    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["iteration"],
                row["tissue"],
                row["image_id"],
                row["nucleus_id"],
            )
            condition = (row["method"], row["seed"])

            if key not in seen:
                seen[key] = {
                    "ambiguity_ratio": row["ambiguity_ratio"],
                    "gt_area": row["gt_area"],
                    "conditions": set(),
                }

            reference = seen[key]

            if (
                reference["ambiguity_ratio"] != row["ambiguity_ratio"]
                or reference["gt_area"] != row["gt_area"]
            ):
                fail(f"Inconsistent GT/ambiguity for nucleus: {key}")

            if condition in reference["conditions"]:
                fail(f"Duplicate model result for nucleus: {key}")

            reference["conditions"].add(condition)
            per_nucleus_counts[key] = len(reference["conditions"])

    if len(seen) != EXPECTED_NUCLEI:
        fail(
            f"Expected {EXPECTED_NUCLEI} distinct nuclei, "
            f"found {len(seen)}."
        )

    if any(n != 7 for n in per_nucleus_counts.values()):
        fail("Not every nucleus has seven model evaluations.")

    completion = {
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "score_file": scores_path.name,
        "score_sha256": sha256(scores_path),
        "total_score_rows": written,
        "distinct_nuclei": len(seen),
        "model_evaluations": 35,
        "nuclei_by_condition": counts,
    }

    write_json_new(output / "completed.json", completion)

    print("\nFULL EVALUATION COMPLETED", flush=True)
    print(f"Scores: {scores_path}")
    print(f"Rows: {written}")
    print(f"Distinct nuclei: {len(seen)}")
    print(f"CSV SHA-256: {completion['score_sha256']}")


if __name__ == "__main__":
    main()
