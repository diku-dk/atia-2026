
"""
Qualitative rendering of 46 frozen NuInsSeg cases.

Source of truth:
    - Original completed held-out evaluation
    - Frozen 46-case selection
    - Original model checkpoints
    - Original evaluation inference procedure

Produces:
    - 46 nine-panel figures
    - 46 compressed prediction archives (7 binary masks each)
    - Historical vs regenerated metric comparison (322 rows)
    - Case summary
    - Pairwise prediction Dice comparisons
    - Run manifest, artifact hashes, and completion record

The original training, evaluation, and selection are READ-ONLY.

Run from project root on a Slurm GPU node:

    python -u -m experiments.render_qualitative_cases_v3
"""

import csv
import gc
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import torch

from mobile_sam import SamPredictor

from experiments import evaluate_heldout as ev
from experiments.nucleus_dataset_loader import (
    load_nuinsseg_sample,
    extract_nucleus,
    get_nucleus_ids,
)


# configurarion

ROOT = ev.ROOT

SELECTED = (
    ROOT
    / "results/qualitative/selection_v1/selected_cases.csv"
)

EVAL_DIR = (
    ROOT
    / "results/evaluation/heldout_v1"
)

EVAL_SCORES = EVAL_DIR / "nucleus_scores.csv"
EVAL_MANIFEST = EVAL_DIR / "manifest.json"
EVAL_COMPLETED = EVAL_DIR / "completed.json"

OUTPUT = (
    ROOT
    / "results/qualitative/render_v3"
)

SPLIT_SHA = (
    "1bc26ee83c8e345cb84bcf33ae3e3fe474a869942005724fa0115b1c6ca4104f"
)

CONDITIONS = (
    [("zero_shot", None)]
    + [
        (method, seed)
        for method in ("lora", "adapter")
        for seed in ev.SEEDS
    ]
)

N_CASES = 46
N_CONDITIONS = 7
N_COMPARISONS = N_CASES * N_CONDITIONS

DICE_TOL = 1e-6


# helpers

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path):
    require(path.is_file(), f"Missing file: {path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def write_csv_exclusive(path, rows):
    require(bool(rows), f"No rows for {path}")

    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json_exclusive(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def case_key(row):
    return (
        int(row["iteration"]),
        row["tissue"],
        row["image_id"],
        int(row["nucleus_id"]),
    )


def condition_name(method, seed):
    if method == "zero_shot":
        return "zero_shot"

    return f"{method}_{seed}"


def mask_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    require(
        pred.shape == gt.shape,
        "Prediction and GT shape mismatch",
    )

    pred_area = int(np.count_nonzero(pred))
    gt_area = int(np.count_nonzero(gt))
    intersection = int(np.count_nonzero(pred & gt))

    denominator = pred_area + gt_area
    union = pred_area + gt_area - intersection

    dice = (
        2 * intersection / denominator
        if denominator
        else 1.0
    )

    iou = (
        intersection / union
        if union
        else 1.0
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "gt_area": gt_area,
        "pred_area": pred_area,
        "intersection": intersection,
    }



def verify_evaluation():
    print("Verifying original held-out evaluation...", flush=True)

    for path in (
        SELECTED,
        EVAL_SCORES,
        EVAL_MANIFEST,
        EVAL_COMPLETED,
    ):
        require(path.is_file(), f"Missing: {path}")

    manifest = json.loads(
        EVAL_MANIFEST.read_text(encoding="utf-8")
    )

    completion = json.loads(
        EVAL_COMPLETED.read_text(encoding="utf-8")
    )

    require(
        completion.get("status") == "completed",
        "Held-out evaluation is not marked completed",
    )

    require(
        completion.get("score_sha256", "").lower()
        == digest(EVAL_SCORES).lower(),
        "Original evaluation CSV hash mismatch",
    )

    require(
        int(completion.get("total_score_rows", -1))
        == 214886,
        "Unexpected evaluation row count",
    )

    require(
        int(completion.get("distinct_nuclei", -1))
        == 30698,
        "Unexpected evaluation nucleus count",
    )

    require(
        manifest.get("split_sha256", "").lower()
        == SPLIT_SHA,
        "Evaluation split hash mismatch",
    )

    vague_template = manifest.get(
        "vague_mask_template"
    )

    require(
        isinstance(vague_template, str)
        and bool(vague_template),
        "Missing vague mask template",
    )

    folds, actual_split_sha = ev.load_frozen_folds(
        SPLIT_SHA
    )

    require(
        actual_split_sha.lower() == SPLIT_SHA,
        "Frozen split mismatch",
    )

    print("Original evaluation verified.", flush=True)

    return manifest, completion, vague_template, folds



# VERIFY FROZEN SELECTION AND CHECKPOINTS
def load_selection_and_scores(folds):
    selected = read_csv(SELECTED)

    require(
        len(selected) == N_CASES,
        f"Expected 46 cases, got {len(selected)}",
    )

    keys = [case_key(row) for row in selected]

    require(
        len(set(keys)) == N_CASES,
        "Duplicate selected nuclei",
    )

    for row in selected:
        key = case_key(row)

        require(
            row["tissue"] in folds[f"fold_{key[0]}"],
            f"Fold mismatch: {key}",
        )

    selected_keys = set(keys)
    expected = {}

    for row in read_csv(EVAL_SCORES):
        key = case_key(row)

        if key not in selected_keys:
            continue

        method = row["method"]

        seed = (
            None
            if method == "zero_shot"
            else int(row["seed"])
        )

        require(
            (method, seed) in CONDITIONS,
            f"Unexpected condition: {method}, {seed}",
        )

        name = condition_name(method, seed)
        eval_key = (key, name)

        require(
            eval_key not in expected,
            f"Duplicate evaluation: {eval_key}",
        )

        expected[eval_key] = row

    require(
        len(expected) == N_COMPARISONS,
        f"Expected 322 scores, got {len(expected)}",
    )

    for row in selected:
        key = case_key(row)

        for method, seed in CONDITIONS:
            name = condition_name(method, seed)

            column = (
                "zero_shot_dice"
                if seed is None
                else name
            )

            selected_dice = float(row[column])
            saved_dice = float(expected[(key, name)]["dice"])

            require(
                abs(selected_dice - saved_dice)
                <= DICE_TOL,
                f"Selection/evaluation mismatch: {key}, {name}",
            )

    print("Verifying all 30 checkpoints...", flush=True)

    identities = ev.preflight_checkpoints()

    for (key, name), row in expected.items():
        if name == "zero_shot":
            continue

        method, seed_str = name.rsplit("_", 1)

        checkpoint_sha = identities[
            (key[0], method, int(seed_str))
        ]["sha256"]

        require(
            row["checkpoint_sha256"].lower()
            == checkpoint_sha.lower(),
            f"Checkpoint mismatch: {key}, {name}",
        )

    print(
        "Selection and checkpoint identities verified.",
        flush=True,
    )

    return selected, keys, expected, identities


# LOAD ORIGINAL IMAGES AND ANNOTATIONS
def prepare_cases(selected, vague_template):
    grouped = defaultdict(list)

    for row in selected:
        key = case_key(row)
        grouped[key[:3]].append(row)

    cases = {}

    for image_key, rows in grouped.items():
        iteration, tissue, image_id = image_key

        image_path = (
            ev.DATA_ROOT
            / tissue
            / "tissue images"
            / image_id
        )

        mask_path = (
            ev.DATA_ROOT
            / tissue
            / "label masks modify"
            / f"{image_path.stem}.tif"
        )

        require(
            image_path.is_file() and mask_path.is_file(),
            f"Missing image or instance mask: {image_key}",
        )

        image, instances = load_nuinsseg_sample(
            image_path,
            mask_path,
        )

        vague_path = ev.vague_mask_path(
            vague_template,
            tissue,
            image_path,
        )

        vague = ev.load_vague_mask(
            vague_path,
            instances.shape,
        )

        # Same full nucleus ordering as the evaluator.
        nucleus_ids = get_nucleus_ids(instances).tolist()

        require(
            len(nucleus_ids) == len(set(nucleus_ids)),
            f"Duplicate nucleus IDs: {image_key}",
        )

        full_boxes = [
            extract_nucleus(instances, nid)[1]
            for nid in nucleus_ids
        ]

        id_to_index = {
            int(nid): i
            for i, nid in enumerate(nucleus_ids)
        }

        for row in rows:
            key = case_key(row)

            require(
                key[3] in id_to_index,
                f"Missing nucleus: {key}",
            )

            gt, bbox = extract_nucleus(
                instances,
                key[3],
            )

            gt = np.asarray(gt, dtype=bool)
            area = int(np.count_nonzero(gt))

            require(
                area > 0
                and area == int(float(row["gt_area"])),
                f"GT area mismatch: {key}",
            )

            ambiguity = (
                np.count_nonzero(gt & vague) / area
            )

            require(
                abs(
                    ambiguity
                    - float(row["ambiguity_ratio"])
                ) <= DICE_TOL,
                f"Ambiguity mismatch: {key}",
            )

            cases[key] = {
                "row": row,
                "image": image,
                "gt": gt,
                "vague": vague,
                "box": bbox,
                "full_boxes": full_boxes,
                "full_index": id_to_index[key[3]],
                "image_path": image_path,
                "mask_path": mask_path,
                "vague_path": vague_path,
                "predictions": {},
                "metrics": {},
            }

    require(
        len(cases) == N_CASES,
        "Not all selected nuclei were loaded",
    )

    return cases


# INFERENCE: ORIGINAL EVALUATION PROCEDURE

@torch.inference_mode()
def predict_selected(
    model,
    image,
    boxes,
    selected_indices,
    device,
):
    """
    Match evaluate_heldout.evaluate_one_image.

    All nuclei are prompted in the original order and
    original prompt batches. Only selected predictions
    are retained.
    """

    predictor = SamPredictor(model)

    transformed_image = (
        predictor.transform.apply_image(image)
    )

    image_tensor = (
        torch.as_tensor(
            transformed_image,
            device=device,
        )
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
    )

    embedding = model.image_encoder(
        model.preprocess(image_tensor)
    )

    input_size = transformed_image.shape[:2]
    original_size = image.shape[:2]
    threshold = float(model.mask_threshold)

    selected_indices = set(selected_indices)
    predictions = {}

    for start in range(
        0,
        len(boxes),
        ev.PROMPT_BATCH_SIZE,
    ):
        end = min(
            start + ev.PROMPT_BATCH_SIZE,
            len(boxes),
        )

        batch = np.stack(boxes[start:end])

        transformed_boxes = (
            predictor.transform.apply_boxes(
                batch,
                original_size,
            )
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

        require(
            tuple(logits.shape)
            == (end - start, 1, *original_size),
            "Unexpected mask decoder output shape",
        )

        binary_masks = (
            (logits[:, 0] > threshold)
            .cpu()
            .numpy()
        )

        for offset, mask in enumerate(binary_masks):
            index = start + offset

            if index in selected_indices:
                predictions[index] = mask.copy()

    require(
        set(predictions) == selected_indices,
        "Missing selected predictions",
    )

    return predictions


# figure helpers

def crop_bounds(gt, pad=45):
    ys, xs = np.nonzero(gt)

    require(len(xs) > 0, "Empty GT mask")

    h, w = gt.shape

    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)

    side = min(max(x1 - x0, y1 - y0), h, w)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, cy - side // 2))

    return (
        slice(top, top + side),
        slice(left, left + side),
    )


def overlay(image, mask, color, alpha=0.45):
    out = image.astype(np.float32).copy()
    mask = np.asarray(mask, dtype=bool)

    out[mask] = (
        (1 - alpha) * out[mask]
        + alpha * np.asarray(color, dtype=np.float32)
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_case(case, output_path):
    row = case["row"]

    image = case["image"]
    gt = case["gt"]
    vague = case["vague"]
    bbox = case["box"]

    sl_y, sl_x = crop_bounds(gt)

    panels = [
        ("H&E image", None),
        ("GT (green), vague (magenta)", "gt"),
        ("Zero-shot", "zero_shot"),
    ]

    panels += [
        (f"LoRA seed {seed}", f"lora_{seed}")
        for seed in ev.SEEDS
    ]

    panels += [
        (f"Adapter seed {seed}", f"adapter_{seed}")
        for seed in ev.SEEDS
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 15),
        constrained_layout=True,
    )

    for ax, (title, condition) in zip(
        axes.flat,
        panels,
    ):
        rgb = image.copy()

        if condition == "gt":
            rgb = overlay(
                rgb, vague, (220, 0, 210), 0.30
            )
            rgb = overlay(
                rgb, gt, (0, 255, 50), 0.58
            )

        elif condition is not None:
            pred = case["predictions"][condition]

            rgb = overlay(
                rgb, pred, (255, 170, 0), 0.46
            )
            rgb = overlay(
                rgb, gt, (0, 255, 40), 0.24
            )

            regenerated_dice = (
                case["metrics"][condition]["dice"]
            )

            title += (
                f" | Regenerated Dice "
                f"{regenerated_dice:.3f}"
            )

        ax.imshow(rgb[sl_y, sl_x])

        if condition not in (None, "gt"):
            bx = np.asarray(bbox, dtype=float)

            ax.add_patch(
                Rectangle(
                    (
                        bx[0] - sl_x.start,
                        bx[1] - sl_y.start,
                    ),
                    bx[2] - bx[0],
                    bx[3] - bx[1],
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.2,
                    linestyle="--",
                )
            )

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"Fold {row['iteration']} | "
        f"{row['tissue']} | "
        f"{row['image_id']} | "
        f"nucleus {row['nucleus_id']}\n"
        f"Ambiguity r="
        f"{float(row['ambiguity_ratio']):.3f} | "
        f"{row.get('selection_category', '')}",
        fontsize=14,
    )

    fig.savefig(
        output_path,
        dpi=160,
        facecolor="white",
    )

    plt.close(fig)


# rendering

def main():
    # Never reuse or overwrite an existing run.
    require(
        not OUTPUT.exists(),
        f"REFUSING TO OVERWRITE: {OUTPUT}",
    )

    require(
        torch.cuda.is_available(),
        "CUDA GPU required. Run through Slurm.",
    )

    manifest, completion, vague_template, folds = (
        verify_evaluation()
    )

    selected, keys, expected, identities = (
        load_selection_and_scores(folds)
    )

    cases = prepare_cases(
        selected,
        vague_template,
    )

    # Source hashes are recorded before inference.
    source_paths = {
        "renderer": Path(__file__),
        "evaluator": Path(ev.__file__),
        "selection": SELECTED,
        "evaluation_scores": EVAL_SCORES,
        "evaluation_manifest": EVAL_MANIFEST,
        "evaluation_completed": EVAL_COMPLETED,
    }

    source_hashes = {
        name: digest(path)
        for name, path in source_paths.items()
    }

    input_hashes = {}

    for key, case in cases.items():
        image_key = key[:3]

        if image_key not in input_hashes:
            input_hashes[image_key] = {
                "image_sha256": digest(case["image_path"]),
                "instance_mask_sha256": digest(
                    case["mask_path"]
                ),
                "vague_mask_sha256": digest(
                    case["vague_path"]
                ),
            }

    # Exclusive output creation, after all preflight checks.
    OUTPUT.mkdir(parents=True, exist_ok=False)

    figdir = OUTPUT / "figures"
    maskdir = OUTPUT / "predicted_masks"

    figdir.mkdir(exist_ok=False)
    maskdir.mkdir(exist_ok=False)

    metadata = {
        "status": "running",
        "started_utc": timestamp(),
        "renderer_version": "v3",
        "source_hashes": source_hashes,
        "input_hashes": [
            {
                "iteration": key[0],
                "tissue": key[1],
                "image_id": key[2],
                **value,
            }
            for key, value in input_hashes.items()
        ],
        "split_sha256": SPLIT_SHA,
        "selected_nuclei": N_CASES,
        "conditions": [
            condition_name(*condition)
            for condition in CONDITIONS
        ],
        "prompt_batch_size": ev.PROMPT_BATCH_SIZE,
        "inference": (
            "Original evaluate_heldout image, box, "
            "prompt encoder, decoder, postprocessing, "
            "threshold, and full nucleus batch ordering"
        ),
        "historical_scores_modified": False,
        "figure_score_source": "regenerated_predictions",
        "historical_masks_available": False,
        "checkpoint_identities": [
            {
                "iteration": iteration,
                "method": method,
                "seed": seed,
                "sha256": info["sha256"],
            }
            for (iteration, method, seed), info
            in sorted(identities.items())
        ],
        "environment": {
            "python_torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cudnn": torch.backends.cudnn.version(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
    }

    write_json_exclusive(
        OUTPUT / "render_manifest.json",
        metadata,
    )

    device = torch.device("cuda")

    names = [
        condition_name(*condition)
        for condition in CONDITIONS
    ]

    comparison_rows = []
    case_rows = []
    pairwise_rows = []

    try:
        # regenerate predictions
        for iteration in range(1, 6):
            fold_keys = [
                key for key in keys
                if key[0] == iteration
            ]

            if not fold_keys:
                continue

            image_groups = defaultdict(list)

            for key in fold_keys:
                image_groups[key[1:3]].append(key)

            for method, seed in CONDITIONS:
                name = condition_name(method, seed)

                print(
                    f"Fold {iteration} | {name} | "
                    f"{len(fold_keys)} nuclei",
                    flush=True,
                )

                model, checkpoint_sha = (
                    ev.construct_model(
                        method,
                        iteration,
                        seed,
                        device,
                        identities,
                    )
                )

                try:
                    for image_keys in image_groups.values():
                        first = cases[image_keys[0]]

                        selected_indices = [
                            cases[key]["full_index"]
                            for key in image_keys
                        ]

                        predictions = predict_selected(
                            model,
                            first["image"],
                            first["full_boxes"],
                            selected_indices,
                            device,
                        )

                        require(
                            len(predictions) == len(image_keys),
                            "Prediction count mismatch",
                        )

                        for key in image_keys:
                            case = cases[key]

                            pred = np.asarray(
                                predictions[case["full_index"]],
                                dtype=bool,
                            )

                            metrics = mask_metrics(
                                pred,
                                case["gt"],
                            )

                            saved = expected[(key, name)]

                            saved_dice = float(saved["dice"])
                            saved_iou = float(saved["iou"])
                            saved_gt_area = int(
                                float(saved["gt_area"])
                            )
                            saved_pred_area = int(
                                float(saved["pred_area"])
                            )
                            saved_intersection = int(
                                float(saved["intersection"])
                            )

                            require(
                                metrics["gt_area"]
                                == saved_gt_area,
                                f"GT area mismatch: {key}, {name}",
                            )

                            dice_delta = (
                                metrics["dice"] - saved_dice
                            )

                            comparison_rows.append({
                                "iteration": key[0],
                                "tissue": key[1],
                                "image_id": key[2],
                                "nucleus_id": key[3],
                                "condition": name,
                                "saved_dice": saved_dice,
                                "regenerated_dice": metrics["dice"],
                                "dice_difference": dice_delta,
                                "absolute_dice_difference": abs(
                                    dice_delta
                                ),
                                "dice_matches_tolerance": (
                                    abs(dice_delta) <= DICE_TOL
                                ),
                                "saved_iou": saved_iou,
                                "regenerated_iou": metrics["iou"],
                                "saved_gt_area": saved_gt_area,
                                "regenerated_gt_area": (
                                    metrics["gt_area"]
                                ),
                                "saved_pred_area": saved_pred_area,
                                "regenerated_pred_area": (
                                    metrics["pred_area"]
                                ),
                                "pred_area_difference": (
                                    metrics["pred_area"]
                                    - saved_pred_area
                                ),
                                "saved_intersection": (
                                    saved_intersection
                                ),
                                "regenerated_intersection": (
                                    metrics["intersection"]
                                ),
                                "intersection_difference": (
                                    metrics["intersection"]
                                    - saved_intersection
                                ),
                            })

                            case["predictions"][name] = pred.copy()
                            case["metrics"][name] = metrics

                            if abs(dice_delta) > DICE_TOL:
                                print(
                                    "HISTORICAL DIFFERENCE: "
                                    f"{key}, {name}, "
                                    f"saved={saved_dice:.9f}, "
                                    f"new={metrics['dice']:.9f}, "
                                    f"delta={dice_delta:+.9f}",
                                    flush=True,
                                )

                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

        require(
            len(comparison_rows) == N_COMPARISONS,
            "Expected 322 regenerated comparisons",
        )

        print(
            "All 322 predictions generated.",
            flush=True,
        )

        # save actual binary masks and figures
        for index, key in enumerate(keys, 1):
            case = cases[key]

            require(
                set(case["predictions"]) == set(names),
                f"Missing predictions: {key}",
            )

            basename = (
                f"{index:02d}_fold{key[0]}_"
                f"nucleus{key[3]}"
            )

            mask_filename = basename + ".npz"
            figure_filename = basename + ".png"

            # Compressed archive contains the exact regenerated
            # binary masks for this nucleus, plus the original GT.
            # No further GPU inference is needed to read them.
            archive = {
                "ground_truth": case["gt"].astype(np.uint8),
                "vague_region": case["vague"].astype(np.uint8),
                "bounding_box": np.asarray(
                    case["box"],
                    dtype=np.float32,
                ),
            }

            for name in names:
                archive[name] = (
                    case["predictions"][name].astype(np.uint8)
                )

            mask_path = maskdir / mask_filename

            with mask_path.open("xb") as f:
                np.savez_compressed(f, **archive)

            # Read back and verify saved predictions.
            with np.load(
                mask_path,
                allow_pickle=False,
            ) as saved_masks:
                for name in names:
                    require(
                        np.array_equal(
                            saved_masks[name].astype(bool),
                            case["predictions"][name],
                        ),
                        f"Saved mask verification failed: "
                        f"{key}, {name}",
                    )

            figure_path = figdir / figure_filename

            require(
                not figure_path.exists(),
                f"REFUSING TO OVERWRITE: {figure_path}",
            )

            draw_case(case, figure_path)

            case_rows.append({
                "iteration": key[0],
                "tissue": key[1],
                "image_id": key[2],
                "nucleus_id": key[3],
                "ambiguity_ratio": (
                    case["row"]["ambiguity_ratio"]
                ),
                "selection_set": case["row"].get(
                    "selection_set", ""
                ),
                "selection_category": case["row"].get(
                    "selection_category", ""
                ),
                "figure": figure_filename,
                "prediction_archive": mask_filename,
                "prediction_archive_sha256": digest(mask_path),
                **{
                    f"regenerated_dice_{name}": (
                        case["metrics"][name]["dice"]
                    )
                    for name in names
                },
                **{
                    f"saved_dice_{name}": float(
                        expected[(key, name)]["dice"]
                    )
                    for name in names
                },
            })

            for a, b in combinations(names, 2):
                pairwise_rows.append({
                    "iteration": key[0],
                    "tissue": key[1],
                    "image_id": key[2],
                    "nucleus_id": key[3],
                    "condition_a": a,
                    "condition_b": b,
                    "prediction_dice": mask_metrics(
                        case["predictions"][a],
                        case["predictions"][b],
                    )["dice"],
                })

            print(
                f"Figure {index}/{N_CASES}: "
                f"{figure_filename}",
                flush=True,
            )

        # ----------------------------------------------------
        # C. WRITE COMPARISON REPORTS
        # ----------------------------------------------------

        write_csv_exclusive(
            OUTPUT / "historical_vs_regenerated.csv",
            comparison_rows,
        )

        write_csv_exclusive(
            OUTPUT / "case_summary.csv",
            case_rows,
        )

        write_csv_exclusive(
            OUTPUT / "pairwise_prediction_dice.csv",
            pairwise_rows,
        )

        mismatches = [
            row
            for row in comparison_rows
            if not row["dice_matches_tolerance"]
        ]

        max_difference = max(
            row["absolute_dice_difference"]
            for row in comparison_rows
        )

        mean_difference = float(np.mean([
            row["absolute_dice_difference"]
            for row in comparison_rows
        ]))

        # save output

        artifact_rows = []

        for path in sorted(OUTPUT.rglob("*")):
            if not path.is_file():
                continue

            # Completion record is created afterwards.
            artifact_rows.append({
                "relative_path": str(
                    path.relative_to(OUTPUT)
                ),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            })

        write_csv_exclusive(
            OUTPUT / "artifact_hashes.csv",
            artifact_rows,
        )

        # completition record

        completed = {
            **metadata,
            "status": "completed",
            "completed_utc": timestamp(),
            "figures": len(case_rows),
            "prediction_archives": len(case_rows),
            "regenerated_predictions": len(comparison_rows),
            "dice_matches": (
                len(comparison_rows) - len(mismatches)
            ),
            "dice_mismatches": len(mismatches),
            "max_absolute_dice_difference": (
                max_difference
            ),
            "mean_absolute_dice_difference": (
                mean_difference
            ),
            "historical_pixel_exact_reproduction": (
                "not established"
            ),
            "historical_scores_modified": False,
            "mask_source": (
                "Regenerated inference using frozen checkpoints"
            ),
        }

        write_json_exclusive(
            OUTPUT / "completed.json",
            completed,
        )

        print("=" * 60, flush=True)
        print("QUALITATIVE RENDERING COMPLETED", flush=True)
        print(f"Figures: {len(case_rows)}", flush=True)
        print(
            f"Saved prediction archives: {len(case_rows)}",
            flush=True,
        )
        print(
            f"Regenerated predictions: {len(comparison_rows)}",
            flush=True,
        )
        print(
            f"Historical Dice mismatches: {len(mismatches)}",
            flush=True,
        )
        print(
            f"Maximum absolute Dice difference: "
            f"{max_difference:.9f}",
            flush=True,
        )
        print(f"Output: {OUTPUT}", flush=True)

    except Exception:
        print(
            "RENDERING FAILED. Partial output preserved "
            f"at {OUTPUT}. Do not overwrite it.",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
