
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from mobile_sam import SamPredictor

from models.lora import build_lora
from models.adapter import build_adapter
from models.losses import dice_bce_loss
from experiments.nucleus_dataset_loader import (
    load_nuinsseg_sample,
    extract_nucleus,
    get_nucleus_ids,
)


# CONFIGURATION

METHODS = ["lora", "adapter"]
ITERATIONS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 456]

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

LORA_RANK = 4
LORA_ALPHA = 4
ADAPTER_DIM = 16

# Use every annotated nucleus in each image
MAX_TRAIN_NUCLEI = None

# Number of nucleus prompts decoded together
PROMPT_BATCH_SIZE = 8




ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "NuInsSeg"
SPLIT_FILE = ROOT / "results" / "splits_final_v1" / "tissue_folds.json"
OUTPUT_ROOT = (
    ROOT / "results" / "training" / "final_full_nuclei"
)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# load CV splits
def get_image_records(tissues):
    records = []

    for tissue in sorted(tissues):
        tissue_dir = DATA_ROOT / tissue

        image_dir = tissue_dir / "tissue images"
        mask_dir = tissue_dir / "label masks modify"

        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / f"{image_path.stem}.tif"

            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            records.append(
                (image_path, mask_path, tissue)
            )

    return records



def get_iteration_records(iteration_number):
    if iteration_number not in range(1, 6):
        raise ValueError(f"Invalid iteration: {iteration_number}")

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    folds = split_data["folds"]

    expected_keys = {f"fold_{i}" for i in range(1, 6)}
    if set(folds) != expected_keys:
        raise ValueError("Unexpected fold keys in final split.")

    all_tissues = [
        tissue
        for fold_tissues in folds.values()
        for tissue in fold_tissues
    ]

    if len(all_tissues) != 31 or len(set(all_tissues)) != 31:
        raise ValueError("Invalid final split: tissues missing or duplicated.")

    # Fixed CV rotation:
    # Iteration 1: test 1, val 2, train 3/4/5
    # Iteration 2: test 2, val 3, train 1/4/5
    # ...
    # Iteration 5: test 5, val 1, train 2/3/4

    test_fold = iteration_number
    val_fold = iteration_number % 5 + 1

    train_folds = [
        i for i in range(1, 6)
        if i not in (test_fold, val_fold)
    ]

    train_tissues = set().union(
        *(folds[f"fold_{i}"] for i in train_folds)
    )

    val_tissues = set(folds[f"fold_{val_fold}"])
    test_tissues = set(folds[f"fold_{test_fold}"])

    if not train_tissues.isdisjoint(val_tissues):
        raise RuntimeError("Train/validation tissue leakage.")

    if not train_tissues.isdisjoint(test_tissues):
        raise RuntimeError("Train/test tissue leakage.")

    if not val_tissues.isdisjoint(test_tissues):
        raise RuntimeError("Validation/test tissue leakage.")

    print(
        f"Iteration {iteration_number}: "
        f"train={train_folds}, "
        f"val={val_fold}, "
        f"test={test_fold}",
        flush=True,
    )

    train_records = get_image_records(train_tissues)
    val_records = get_image_records(val_tissues)

    # Held-out evaluation is deliberately not loaded for training.
    return train_records, val_records



# building the model
def build_model(method):

    if method == "lora":
        result = build_lora(
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            device=str(DEVICE)
        )

    elif method == "adapter":
        result = build_adapter(
            bottleneck_dim=ADAPTER_DIM,
            device=str(DEVICE)
        )

    else:
        raise ValueError(
            "Method must be 'lora' or 'adapter'."
        )

    # Both existing builders return (model, adapted_layers).
    model = result[0] if isinstance(result, tuple) else result

    model.to(DEVICE)

    # Keep frozen backbone statistics unchanged.
    # This does not disable gradients.
    model.eval()

    return model


# encoding images
def encode_image(model, predictor, image):

    transformed_image = (
        predictor.transform.apply_image(image)
    )

    image_tensor = torch.as_tensor(
        transformed_image,
        device=DEVICE
    )

    image_tensor = (
        image_tensor
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
    )

    image_embedding = model.image_encoder(
        model.preprocess(image_tensor)
    )

    input_size = transformed_image.shape[:2]

    return image_embedding, input_size


# prediciting masks from box prompt
def predict_logits(
    model,
    predictor,
    image_embedding,
    input_size,
    original_size,
    boxes
):

    transformed_boxes = (
        predictor.transform.apply_boxes(
            boxes,
            original_size
        )
    )

    box_tensor = torch.as_tensor(
        transformed_boxes,
        dtype=torch.float32,
        device=DEVICE
    )

    sparse_embeddings, dense_embeddings = (
        model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None
        )
    )

    # Uses one image embedding with multiple box prompts.
    # Assumes the MobileSAM decoder repeats the embedding
    # internally for the prompt batch.
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    logits = model.postprocess_masks(
        low_res_masks,
        input_size=input_size,
        original_size=original_size
    )

    return logits


# for one img
def process_image(
    model,
    predictor,
    record,
    training,
    rng
):

    image_path, mask_path, tissue = record

    image, instance_mask = load_nuinsseg_sample(
        image_path,
        mask_path
    )

    nucleus_ids = get_nucleus_ids(instance_mask).tolist()

    if not nucleus_ids:
        return None

    # Full training uses all annotated nuclei.
    if (
        training
        and MAX_TRAIN_NUCLEI is not None
        and len(nucleus_ids) > MAX_TRAIN_NUCLEI
    ):
        nucleus_ids = rng.sample(
            nucleus_ids,
            MAX_TRAIN_NUCLEI
        )

    boxes = []
    masks = []

    for nucleus_id in nucleus_ids:
        nucleus_mask, bbox = extract_nucleus(
            instance_mask,
            nucleus_id
        )

        boxes.append(bbox)
        masks.append(nucleus_mask)

    boxes = np.stack(boxes)
    masks = np.stack(masks)

    # Encode the image only once.
    image_embedding, input_size = encode_image(
        model,
        predictor,
        image
    )

    n_nuclei = len(nucleus_ids)

    if training:
        # Accumulate decoder gradients separately, then
        # propagate them through the image encoder once.
        prompt_embedding = (
            image_embedding.detach().requires_grad_(True)
        )
    else:
        prompt_embedding = image_embedding

    total_loss_value = 0.0

    for start in range(0, n_nuclei, PROMPT_BATCH_SIZE):

        end = min(
            start + PROMPT_BATCH_SIZE,
            n_nuclei
        )

        batch_boxes = boxes[start:end]
        batch_masks = masks[start:end]

        logits = predict_logits(
            model=model,
            predictor=predictor,
            image_embedding=prompt_embedding,
            input_size=input_size,
            original_size=image.shape[:2],
            boxes=batch_boxes
        )

        targets = torch.as_tensor(
            batch_masks,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        loss, _, _ = dice_bce_loss(
            logits,
            targets
        )

        # Each nucleus contributes equally to the image loss.
        # This also handles a smaller final prompt batch.
        batch_weight = (end - start) / n_nuclei

        weighted_loss = loss * batch_weight

        if not torch.isfinite(weighted_loss).item():
            raise RuntimeError(
                f"Non-finite loss for image {image_path.name}"
            )

        total_loss_value += weighted_loss.detach().item()

        if training:
            weighted_loss.backward()

        del logits, targets, loss, weighted_loss

    if training:

        if prompt_embedding.grad is None:
            raise RuntimeError(
                "No gradient reached the image embedding."
            )

        # Backpropagate accumulated gradients into TinyViT.
        image_embedding.backward(prompt_embedding.grad)

    return total_loss_value, n_nuclei


# one epoch
def run_epoch(
    model,
    records,
    optimizer,
    epoch,
    training,
    seed
):

    rng = random.Random(seed + epoch)

    records = list(records)

    if training:
        rng.shuffle(records)

    model.eval()

    predictor = SamPredictor(model)

    total_loss = 0.0
    total_nuclei = 0
    total_images = 0

    phase = "TRAIN" if training else "VALIDATION"

    for index, record in enumerate(records, start=1):

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            result = process_image(
                model=model,
                predictor=predictor,
                record=record,
                training=training,
                rng=rng
            )

        if result is None:
            continue

        loss_value, n_nuclei = result

        if not np.isfinite(loss_value):
            raise RuntimeError(
                f"Non-finite {phase} loss."
            )

        if training:
            optimizer.step()

        total_loss += loss_value
        total_nuclei += n_nuclei
        total_images += 1

        if index % 25 == 0 or index == len(records):
            print(
                f"{phase}: "
                f"{index}/{len(records)} images",
                flush=True
            )

    if total_images == 0:
        raise RuntimeError(
            f"No images processed during {phase}."
        )

    # Mean of image-level mean losses.
    average_loss = total_loss / total_images

    return average_loss, total_nuclei


# one experiment
def train_one_experiment(iteration, method, seed):

    if method not in ("lora", "adapter"):
        raise ValueError("Invalid method.")
    if iteration not in range(1, 6):
        raise ValueError("Iteration must be 1-5.")
    if MAX_EPOCHS < 1:
        raise ValueError("MAX_EPOCHS must be positive.")
    if EARLY_STOPPING_PATIENCE < 1:
        raise ValueError(
            "EARLY_STOPPING_PATIENCE must be positive."
        )
    if PROMPT_BATCH_SIZE < 1:
        raise ValueError(
            "PROMPT_BATCH_SIZE must be positive."
        )

    set_seed(seed)

    output_dir = (
        OUTPUT_ROOT
        / f"iteration_{iteration}"
        / method
        / f"seed_{seed}"
    )

    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.csv"
    config_path = output_dir / "config.json"
    completion_path = output_dir / "completed.json"

    # Protect existing results from being overwritten.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Existing results found: {output_dir}. "
            "Review this run before restarting it."
        )

    train_records, val_records = (
        get_iteration_records(iteration)
    )

    model = build_model(method)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable parameters found."
        )

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    config = {
        "method": method,
        "iteration": iteration,
        "seed": seed,
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "adapter_dim": ADAPTER_DIM,
        "max_train_nuclei": MAX_TRAIN_NUCLEI,
        "prompt_batch_size": PROMPT_BATCH_SIZE,
        "device": str(DEVICE),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "trainable_parameters": sum(
            p.numel() for p in trainable_parameters
        )
    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 60)
    print("MOBILESAM PEFT TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Method:", method)
    print("CV iteration:", iteration)
    print("Seed:", seed)

    print("Training images:", len(train_records))
    print("Validation images:", len(val_records))

    print(
        "Trainable parameters:",
        config["trainable_parameters"]
    )

    print("Learning rate:", LEARNING_RATE)
    print("Maximum epochs:", MAX_EPOCHS)
    print(
        "Early stopping patience:",
        EARLY_STOPPING_PATIENCE
    )
    print("Max training nuclei:", MAX_TRAIN_NUCLEI)
    print("Prompt batch size:", PROMPT_BATCH_SIZE)

    print("=" * 60, flush=True)

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        print(
            f"\nEPOCH {epoch}/{MAX_EPOCHS}",
            flush=True
        )

        # TRAIN ----------------

        train_loss, train_nuclei = run_epoch(
            model=model,
            records=train_records,
            optimizer=optimizer,
            epoch=epoch,
            training=True,
            seed=seed
        )

        # VALIDATE ----------------

        val_loss, val_nuclei = run_epoch(
            model=model,
            records=val_records,
            optimizer=None,
            epoch=epoch,
            training=False,
            seed=seed
        )

        if (
            not np.isfinite(train_loss)
            or not np.isfinite(val_loss)
        ):
            raise RuntimeError(
                "Non-finite epoch loss."
            )

        print("\nEpoch results:")
        print(
            f"Training loss:   {train_loss:.6f} "
            f"({train_nuclei} nuclei)"
        )
        print(
            f"Validation loss: {val_loss:.6f} "
            f"({val_nuclei} nuclei)"
        )

        # MODEL SELECTION ----------------

        improved = val_loss < best_val_loss

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "method": method,
                "iteration": iteration,
                "seed": seed,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": config,
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            del checkpoint

            print(
                "New best model saved.",
                flush=True
            )

        else:

            epochs_without_improvement += 1

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
                "/",
                EARLY_STOPPING_PATIENCE,
                flush=True
            )

        # TRAINING HISTORY ----------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_nuclei": train_nuclei,
            "val_nuclei": val_nuclei,
            "best_val_loss": best_val_loss,
            "improved": improved
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        # EARLY STOPPING ----------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEARLY STOPPING",
                flush=True
            )
            break

    # COMPLETION ----------------

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", checkpoint_path)
    print("Training history:", history_path)

    # This marker is written only after successful training.
    completion = {
        "status": "completed",
        "method": method,
        "iteration": iteration,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

    with open(
        completion_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(completion, f, indent=4)

    # Release the model and optimizer between runs.
    del optimizer
    del trainable_parameters
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# RUN ALL 30 EXPERIMENTS

def main():

    runs = [
        (iteration, method, seed)
        for iteration in ITERATIONS
        for method in METHODS
        for seed in SEEDS
    ]

    total_runs = len(runs)

    print("\n" + "=" * 60)
    print("FULL MOBILESAM PEFT EXPERIMENT")
    print("=" * 60)

    print("Total experiments:", total_runs)
    print("Iterations:", ITERATIONS)
    print("Methods:", METHODS)
    print("Seeds:", SEEDS)
    print("Device:", DEVICE)

    print("=" * 60, flush=True)

    for run_number, (iteration, method, seed) in enumerate(
        runs,
        start=1
    ):

        output_dir = (
            OUTPUT_ROOT
            / f"iteration_{iteration}"
            / method
            / f"seed_{seed}"
        )

        completion_path = output_dir / "completed.json"
        checkpoint_path = output_dir / "best_model.pt"
        history_path = output_dir / "history.csv"

        print(
            f"\nRUN {run_number}/{total_runs}: "
            f"Iteration={iteration}, "
            f"Method={method}, "
            f"Seed={seed}",
            flush=True
        )

        # Skip runs that have a verified completion marker.
        if completion_path.exists():

            with open(
                completion_path,
                encoding="utf-8"
            ) as f:
                completion = json.load(f)

            valid_completion = (
                completion.get("status") == "completed"
                and completion.get("method") == method
                and completion.get("iteration") == iteration
                and completion.get("seed") == seed
                and checkpoint_path.is_file()
                and history_path.is_file()
            )

            if not valid_completion:
                raise RuntimeError(
                    f"Inconsistent completion files: {output_dir}"
                )

            print(
                "Already completed. Skipping.",
                flush=True
            )
            continue

        # Stop if files exist but training did not finish.
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete experiment found: {output_dir}\n"
                "Review its results before restarting."
            )

        # Train this combination.
        train_one_experiment(
            iteration=iteration,
            method=method,
            seed=seed
        )

    print("\n" + "=" * 60)
    print("ALL 30 EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
