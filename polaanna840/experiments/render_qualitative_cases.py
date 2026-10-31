"""Render 46 frozen qualitative cases with the exact held-out evaluation inference.

Run on a Slurm GPU node from project root:
    python -u -m experiments.render_qualitative_cases

Inputs are read-only. Output directory must not exist.
"""

import csv
import gc
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations

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


ROOT = ev.ROOT

SELECTED = (
    ROOT / "results/qualitative/selection_v1/selected_cases.csv"
)

EVAL_SCORES = (
    ROOT / "results/evaluation/nucleus_scores.csv"
)

EVAL_MANIFEST = (
    ROOT / "results/evaluation/manifest.json"
)

OUTPUT = (
    ROOT / "results/qualitative/render_v2"
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

TOL = 1e-6


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    require(path.is_file(), f"Missing: {path}")

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def case_key(row):
    return (
        int(row["iteration"]),
        row["tissue"],
        row["image_id"],
        int(row["nucleus_id"]),
    )


def cond_name(method, seed):
    if method == "zero_shot":
        return "zero_shot"

    return f"{method}_{seed}"


def expected_eval_dice(row):
    return float(row["dice"])


def mask_dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    require(
        a.shape == b.shape,
        "Mask shape mismatch",
    )

    denominator = int(a.sum()) + int(b.sum())

    if denominator == 0:
        return 1.0

    intersection = np.count_nonzero(a & b)

    return float(
        2 * intersection / denominator
    )


@torch.inference_mode()
def predict_selected(model, image, boxes, selected_indices, device):
    """
    Reproduce evaluate_heldout.evaluate_one_image inference
    using all nuclei in the original evaluation order.

    Return predictions only for the selected nuclei.
    """

    predictor = SamPredictor(model)

    transformed_image = predictor.transform.apply_image(image)

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

        transformed_boxes = predictor.transform.apply_boxes(
            batch,
            original_size,
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
            end - start,
            1,
            *original_size,
        )

        require(
            tuple(logits.shape) == expected_shape,
            "Unexpected decoder output",
        )

        binary_masks = (
            (logits[:, 0] > threshold)
            .cpu()
            .numpy()
        )

        for offset, mask in enumerate(binary_masks):
            idx = start + offset

            if idx in selected_indices:
                predictions[idx] = mask.copy()

    require(
        set(predictions) == selected_indices,
        "Missing selected predictions",
    )

    return predictions


def crop_bounds(gt, pad=45):
    ys, xs = np.nonzero(gt)

    require(
        len(xs) > 0,
        "Empty ground-truth mask",
    )

    h, w = gt.shape

    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)

    side = min(
        max(x1 - x0, y1 - y0),
        h,
        w,
    )

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    left = max(
        0,
        min(w - side, cx - side // 2),
    )

    top = max(
        0,
        min(h - side, cy - side // 2),
    )

    return (
        slice(top, top + side),
        slice(left, left + side),
    )


def overlay(image, mask, color, alpha=0.45):
    out = image.astype(np.float32).copy()

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    out[mask] = (
        (1 - alpha) * out[mask]
        + alpha * np.asarray(
            color,
            dtype=np.float32,
        )
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)


def draw_case(case, output_path):
    row = case["row"]

    image = case["image"]
    gt = case["gt"]
    vague = case["vague"]
    bbox = case["box"]

    sl_y, sl_x = crop_bounds(gt)

    x0 = sl_x.start
    y0 = sl_y.start

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
                rgb,
                vague,
                (220, 0, 210),
                0.30,
            )

            rgb = overlay(
                rgb,
                gt,
                (0, 255, 50),
                0.58,
            )

        elif condition is not None:
            pred = case["predictions"][condition]

            rgb = overlay(
                rgb,
                pred,
                (255, 170, 0),
                0.46,
            )

            rgb = overlay(
                rgb,
                gt,
                (0, 255, 40),
                0.24,
            )

            title += (
                f" | Dice "
                f"{case['dice'][condition]:.3f}"
            )

        ax.imshow(
            rgb[sl_y, sl_x]
        )

        if (
            condition is not None
            and condition != "gt"
        ):
            bx = np.asarray(
                bbox,
                dtype=float,
            )

            ax.add_patch(
                Rectangle(
                    (
                        bx[0] - x0,
                        bx[1] - y0,
                    ),
                    bx[2] - bx[0],
                    bx[3] - bx[1],
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.2,
                    linestyle="--",
                )
            )

        ax.set_title(
            title,
            fontsize=11,
        )

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


def main():
    # Never overwrite an existing rendering run.
    require(
        not OUTPUT.exists(),
        f"REFUSING TO OVERWRITE: {OUTPUT}",
    )

    require(
        torch.cuda.is_available(),
        "CUDA GPU required: run in a Slurm GPU allocation",
    )

    require(
        SELECTED.is_file()
        and EVAL_SCORES.is_file()
        and EVAL_MANIFEST.is_file(),
        "Missing selection CSV, evaluation CSV, or manifest",
    )

    print(
        "Reading selection, scores and evaluation manifest...",
        flush=True,
    )

    selected = read_csv(SELECTED)

    require(
        len(selected) == 46,
        f"Expected 46 selected cases, got {len(selected)}",
    )

    keys = [
        case_key(row)
        for row in selected
    ]

    require(
        len(set(keys)) == 46,
        "Duplicate selected nucleus",
    )

    manifest = json.loads(
        EVAL_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        manifest.get(
            "split_sha256",
            "",
        ).lower() == SPLIT_SHA,
        "Evaluation split hash mismatch",
    )

    vague_template = manifest.get(
        "vague_mask_template"
    )

    require(
        isinstance(vague_template, str)
        and bool(vague_template),
        "Evaluation manifest lacks vague_mask_template",
    )

    folds, actual_sha = ev.load_frozen_folds(
        SPLIT_SHA
    )

    for row in selected:
        fold_name = (
            f"fold_{int(row['iteration'])}"
        )

        require(
            row["tissue"] in folds[fold_name],
            f"Selection fold mismatch: {case_key(row)}",
        )

    # Read the original evaluation scores.
    expected = {}

    selected_keys = set(keys)

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

        name = cond_name(
            method,
            seed,
        )

        require(
            (method, seed) in CONDITIONS,
            f"Unexpected method/seed: {name}",
        )

        eval_key = (
            key,
            name,
        )

        require(
            eval_key not in expected,
            f"Duplicate evaluation row: {eval_key}",
        )

        expected[eval_key] = row

    require(
        len(expected) == 46 * 7,
        (
            "Expected 322 evaluation rows, "
            f"got {len(expected)}"
        ),
    )

    # Verify that the frozen selection agrees
    # with the original evaluation scores.
    for row in selected:
        key = case_key(row)

        for method, seed in CONDITIONS:
            name = cond_name(
                method,
                seed,
            )

            column = (
                "zero_shot_dice"
                if seed is None
                else name
            )

            selected_dice = float(
                row[column]
            )

            saved_dice = expected_eval_dice(
                expected[(key, name)]
            )

            require(
                abs(selected_dice - saved_dice)
                <= TOL,
                (
                    "Selection and evaluation Dice disagree: "
                    f"{key}, {name}"
                ),
            )

    print(
        "Verifying checkpoint identities "
        "(30 SHA-256 hashes)...",
        flush=True,
    )

    identities = ev.preflight_checkpoints()

    for (key, name), row in expected.items():
        if name == "zero_shot":
            continue

        method, seed_str = name.rsplit(
            "_",
            1,
        )

        checkpoint_sha = identities[
            (
                key[0],
                method,
                int(seed_str),
            )
        ]["sha256"]

        require(
            row["checkpoint_sha256"].lower()
            == checkpoint_sha.lower(),
            (
                "Checkpoint SHA mismatch: "
                f"{key}, {name}"
            ),
        )

    # Check every selected nucleus before
    # creating the output directory.
    print(
        "Checking 46 selected nuclei "
        "and ambiguity annotations...",
        flush=True,
    )

    cases = {}

    grouped_images = defaultdict(list)

    for row in selected:
        key = case_key(row)

        image_key = (
            key[0],
            key[1],
            key[2],
        )

        grouped_images[image_key].append(
            row
        )

    for (
        iteration,
        tissue,
        image_id,
    ), rows in grouped_images.items():

        image_path = (
            ev.DATA_ROOT
            / tissue
            / "tissue images"
            / image_id
        )

        require(
            image_path.is_file(),
            f"Missing H&E image: {image_path}",
        )

        mask_path = (
            ev.DATA_ROOT
            / tissue
            / "label masks modify"
            / f"{image_path.stem}.tif"
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

        # Preserve the exact ordering of all nuclei
        # used by the original held-out evaluation.
        nucleus_ids = get_nucleus_ids(instances).tolist()

        require(
            bool(nucleus_ids),
            f"No annotated nuclei: {image_path}",
        )

        require(
            len(nucleus_ids) == len(set(nucleus_ids)),
            f"Duplicate nucleus IDs: {image_path}",
        )

        full_boxes = [
            extract_nucleus(instances, nid)[1]
            for nid in nucleus_ids
        ]

        id_to_index = {
            int(nid): idx
            for idx, nid in enumerate(nucleus_ids)
        }

        for row in rows:
            key = case_key(row)

            require(
                key[3] in id_to_index,
                f"Selected nucleus missing: {key}",
            )

            gt, bbox = extract_nucleus(
                instances,
                key[3],
            )

            gt = np.asarray(
                gt,
                dtype=bool,
            )

            area = int(
                gt.sum()
            )

            require(
                area > 0
                and area == int(float(row["gt_area"])),
                f"GT area mismatch: {key}",
            )

            ambiguity = (
                np.count_nonzero(gt & vague)
                / area
            )

            require(
                abs(
                    ambiguity
                    - float(row["ambiguity_ratio"])
                ) <= TOL,
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
                "predictions": {},
                "dice": {},
            }

    # Exclusive output directory creation.
    OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    figdir = OUTPUT / "figures"

    figdir.mkdir(
        exist_ok=False
    )

    metadata = {
        "status": "running",
        "started_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "selection_sha256": digest(SELECTED),
        "evaluation_scores_sha256": digest(EVAL_SCORES),
        "evaluation_manifest_sha256": digest(EVAL_MANIFEST),
        "split_sha256": actual_sha,
        "selected_nuclei": len(cases),
        "conditions": [
            cond_name(*condition)
            for condition in CONDITIONS
        ],
        "dice_tolerance": TOL,
        "inference": (
            "evaluate_heldout exact "
            "image/box/decoder/threshold path; "
            "all nuclei in original evaluation order "
            "and original prompt batches"
        ),
    }

    with (
        OUTPUT / "render_manifest.json"
    ).open(
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    device = torch.device("cuda")

    try:
        # Perform inference for each held-out fold
        # and each of the seven model conditions.
        for iteration in range(1, 6):

            fold_keys = [
                key
                for key in keys
                if key[0] == iteration
            ]

            if not fold_keys:
                continue

            image_groups = defaultdict(list)

            for key in fold_keys:
                image_key = (
                    key[1],
                    key[2],
                )

                image_groups[image_key].append(
                    key
                )

            for method, seed in CONDITIONS:
                name = cond_name(
                    method,
                    seed,
                )

                print(
                    f"Fold {iteration} | "
                    f"{name} | "
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

                        image = cases[
                            image_keys[0]
                        ]["image"]

                        full_boxes = cases[
                            image_keys[0]
                        ]["full_boxes"]

                        selected_indices = [
                            cases[key]["full_index"]
                            for key in image_keys
                        ]

                        predictions = predict_selected(
                            model,
                            image,
                            full_boxes,
                            selected_indices,
                            device,
                        )

                        require(
                            len(predictions) == len(image_keys),
                            "Prediction count mismatch",
                        )

                        for key in image_keys:
                            pred = predictions[
                                cases[key]["full_index"]
                            ]

                            pred = np.asarray(
                                pred,
                                dtype=bool,
                            )

                            dice = mask_dice(
                                pred,
                                cases[key]["gt"],
                            )

                            saved_dice = (
                                expected_eval_dice(
                                    expected[
                                        (key, name)
                                    ]
                                )
                            )

                            require(
                                abs(dice - saved_dice)
                                <= TOL,
                                (
                                    f"DICE MISMATCH: "
                                    f"{key}, {name}: "
                                    f"recomputed={dice:.9f}, "
                                    f"saved={saved_dice:.9f}"
                                ),
                            )

                            cases[key][
                                "predictions"
                            ][name] = pred

                            cases[key][
                                "dice"
                            ][name] = dice

                finally:
                    del model

                    gc.collect()

                    torch.cuda.empty_cache()

        print(
            "All 322 GT Dice values match. "
            "Rendering 46 figures...",
            flush=True,
        )

        summary_rows = []
        pairwise_rows = []

        names = [
            cond_name(*condition)
            for condition in CONDITIONS
        ]

        for index, key in enumerate(
            keys,
            1,
        ):
            case = cases[key]

            require(
                set(case["predictions"])
                == set(names),
                f"Missing predictions: {key}",
            )

            filename = (
                f"{index:02d}_"
                f"fold{key[0]}_"
                f"nucleus{key[3]}.png"
            )

            draw_case(
                case,
                figdir / filename,
            )

            summary_rows.append(
                {
                    "iteration": key[0],
                    "tissue": key[1],
                    "image_id": key[2],
                    "nucleus_id": key[3],
                    "figure": filename,
                    "ambiguity_ratio": (
                        case["row"]["ambiguity_ratio"]
                    ),
                    "selection_set": (
                        case["row"].get(
                            "selection_set",
                            "",
                        )
                    ),
                    "selection_category": (
                        case["row"].get(
                            "selection_category",
                            "",
                        )
                    ),
                    **{
                        f"dice_{name}": (
                            case["dice"][name]
                        )
                        for name in names
                    },
                }
            )

            for a, b in combinations(
                names,
                2,
            ):
                pairwise_rows.append(
                    {
                        "iteration": key[0],
                        "tissue": key[1],
                        "image_id": key[2],
                        "nucleus_id": key[3],
                        "condition_a": a,
                        "condition_b": b,
                        "prediction_dice": mask_dice(
                            case["predictions"][a],
                            case["predictions"][b],
                        ),
                    }
                )

            print(
                f"Figure {index}/46: {filename}",
                flush=True,
            )

        # Save case-level summary and pairwise
        # prediction comparisons.
        outputs = [
            (
                "case_summary.csv",
                summary_rows,
            ),
            (
                "pairwise_prediction_dice.csv",
                pairwise_rows,
            ),
        ]

        for filename, rows in outputs:
            path = OUTPUT / filename

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

        metadata["status"] = "completed"

        metadata["completed_utc"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        metadata["figures"] = len(
            summary_rows
        )

        metadata["validated_gt_dice"] = (
            46 * 7
        )

        metadata["pairwise_rows"] = len(
            pairwise_rows
        )

        # Separate completion record:
        # preserve the original run manifest.
        with (
            OUTPUT / "completed.json"
        ).open(
            "x",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        print(
            f"DONE: {len(summary_rows)} figures, "
            f"{len(pairwise_rows)} pairwise comparisons "
            f"-> {OUTPUT}",
            flush=True,
        )

    except Exception:
        print(
            "FAILED: partial output preserved "
            f"for diagnosis at {OUTPUT}",
            flush=True,
        )

        raise


if __name__ == "__main__":
    main()