
"""MobileSAM LoRA training on patient-separated Cheng budgets."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.transforms import ResizeLongestSide

from src.dataset import load_cheng_sample
from src.preprocessing import preprocess_mri
from src.prompts import mask_to_box
from src.metrics import dice_score
from src.lora import apply_lora
from src.adapter import apply_adapter


class ChengDataset(Dataset):
    def __init__(self, csv_path):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError(f"Empty dataset: {csv_path}")

        self.transform = ResizeLongestSide(1024)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = load_cheng_sample(Path("data/cheng") / row["file"])

        if sample["patient_id"] != row["PID"]:
            raise ValueError(f"Patient mismatch: {row['file']}")

        image = preprocess_mri(sample["image"])
        box = mask_to_box(sample["mask"])

        resized = self.transform.apply_image(image)
        resized_box = self.transform.apply_boxes(
            box[None, :], image.shape[:2]
        )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(resized)
            ).permute(2, 0, 1).float(),
            "box": torch.from_numpy(resized_box.copy()).float(),
            "mask": torch.from_numpy(
                sample["mask"].astype(np.float32).copy()
            )[None],
            "original_size": sample["mask"].shape,
            "input_size": resized.shape[:2],
            "file": row["file"],
            "PID": row["PID"],
        }


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum()
    denominator = probabilities.sum() + target.sum()

    dice = 1 - (2 * intersection + 1) / (denominator + 1)
    return bce + dice


def train_epoch(model, loader, optimizer, device, max_batches=0):
    model.eval()
    total_loss = 0.0
    total_images = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        optimizer.zero_grad(set_to_none=True)

        images = torch.stack([
            model.preprocess(item["image"].to(device))
            for item in batch
        ])

        boxes = torch.cat([
            item["box"].to(device) for item in batch
        ], dim=0)

        image_embeddings = model.image_encoder(images)

        with torch.no_grad():
            sparse, dense = model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
            image_pe = model.prompt_encoder.get_dense_pe()

        losses = []

        for i, item in enumerate(batch):
            low_res_logits, _ = model.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse[i:i + 1],
                dense_prompt_embeddings=dense[i:i + 1],
                multimask_output=False,
            )

            logits = model.postprocess_masks(
                low_res_logits,
                input_size=item["input_size"],
                original_size=item["original_size"],
            )

            target = item["mask"].unsqueeze(0).to(device)
            losses.append(segmentation_loss(logits, target))

        loss = torch.stack(losses).mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        total_images += len(batch)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(loader)}, "
                f"loss={loss.item():.5f}",
                flush=True,
            )

    if total_images == 0:
        raise RuntimeError("No training images processed")

    return total_loss / total_images, total_images


@torch.no_grad()
def validate(model, dataset, limit=0):
    model.eval()
    predictor = SamPredictor(model)

    results = []
    rows = dataset.rows[:limit] if limit else dataset.rows

    for row in rows:
        sample = load_cheng_sample(Path("data/cheng") / row["file"])

        if sample["patient_id"] != row["PID"]:
            raise ValueError(f"Validation patient mismatch: {row['file']}")

        image = preprocess_mri(sample["image"])
        box = mask_to_box(sample["mask"])

        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )

        dice = dice_score(masks[0], sample["mask"])
        results.append((row["PID"], dice))

    mean_slice_dice = float(np.mean([d for _, d in results]))

    patient_scores = {}
    for pid, score in results:
        patient_scores.setdefault(pid, []).append(score)

    mean_patient_dice = float(np.mean([
        np.mean(scores) for scores in patient_scores.values()
    ]))

    return mean_slice_dice, mean_patient_dice


def save_checkpoint(path, model, optimizer, scheduler,
                    epoch, best_dice, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "config": vars(args),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", required=True,
                        choices=["10", "25", "50", "100",
                                 "10_fewpatients"])
    parser.add_argument("--seed", type=int, required=True,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument("--method", choices=["lora", "adapter"], default="lora")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--output-root", type=str,
                        default="experiment_extended/training")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required: use Slurm")

    if args.epochs < 1 or args.batch_size < 1 or args.val_every < 1:
        raise ValueError("Invalid training configuration")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    train_csv = (
        Path(args.train_csv) if args.train_csv
        else Path(f"data/experiment_v2/budgets/seed_{args.seed}/budget_{args.budget}.csv")
    )
    train_data = ChengDataset(train_csv)
    val_data = ChengDataset("data/cheng_validation_files.csv")

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda samples: samples,
    )

    model = sam_model_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )
    model = (apply_lora(model, rank=4) if args.method == "lora"
             else apply_adapter(model)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    pilot = bool(args.max_train_batches or args.val_limit)
    run_name = (
        f"{args.method}_budget_{args.budget}_seed_{args.seed}"
        + ("_pilot" if pilot else "")
    )
    lr_label = f"{args.lr:.0e}".replace("+", "")
    output_dir = Path(args.output_root) / lr_label / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "history.csv"

    start_epoch = 1
    best_dice = -1.0
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(
            last_path, map_location=device, weights_only=False
        )

        saved_config = checkpoint["config"]
        for key in ("budget", "seed", "batch_size", "lr",
                    "val_every", "max_train_batches", "val_limit"):
            if getattr(args, key.replace("_", "-"), None) is not None:
                pass
            if vars(args)[key] != saved_config[key]:
                raise ValueError(f"Resume setting changed: {key}")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        if log_path.exists():
            with log_path.open(newline="") as f:
                history = list(csv.DictReader(f))
            best_epoch = max(
                (int(r["epoch"]) for r in history
                 if float(r["val_dice"]) >= best_dice),
                default=0,
            )
            epochs_without_improvement = (
                checkpoint["epoch"] - best_epoch
            )

    print(f"Run: {run_name}", flush=True)
    print(f"Training images: {len(train_data)}", flush=True)
    print(f"Validation images available: {len(val_data)}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if not log_path.exists() or not args.resume:
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_dice",
                "val_patient_dice", "elapsed_seconds"
            ])

    run_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()

        loss, n_images = train_epoch(
            model, loader, optimizer, device,
            max_batches=args.max_train_batches,
        )

        scheduler.step()

        val_dice = None
        val_patient_dice = None

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_dice, val_patient_dice = validate(
                model, val_data, limit=args.val_limit
            )

            if val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0
                save_checkpoint(
                    best_path, model, optimizer, scheduler,
                    epoch, best_dice, args,
                )
            else:
                epochs_without_improvement += args.val_every

        elapsed = time.perf_counter() - start

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch, loss,
                "" if val_dice is None else val_dice,
                "" if val_patient_dice is None else val_patient_dice,
                elapsed,
            ])

        save_checkpoint(
            last_path, model, optimizer, scheduler,
            epoch, best_dice, args,
        )

        print(
            f"Epoch {epoch}: train_loss={loss:.5f}, "
            f"val_dice={val_dice}, "
            f"images={n_images}, time={elapsed:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            print("Early stopping", flush=True)
            break

    torch.cuda.synchronize()

    print(f"Best validation Dice: {best_dice:.5f}")
    print(f"Total elapsed: {time.perf_counter() - run_start:.1f}s")
    print(
        f"Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )
    print(f"Checkpoints saved to: {output_dir}")
    print("TRAINING FINISHED", flush=True)


if __name__ == "__main__":
    main()

























