import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mobile_sam import sam_model_registry
from src.train_v2 import ChengDataset, segmentation_loss
from src.lora import apply_lora
from src.adapter import apply_adapter


def run(method, batch):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = sam_model_registry["vit_t"](
        checkpoint="MobileSAM/weights/mobile_sam.pt"
    )
    model = (
        apply_lora(model, rank=4)
        if method == "lora"
        else apply_adapter(model)
    ).cuda()

    model.eval()  # Match the original training implementation.

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.1,
    )

    images = torch.stack([
        model.preprocess(item["image"].cuda())
        for item in batch
    ])
    boxes = torch.cat([
        item["box"].cuda() for item in batch
    ], dim=0)
    targets = [item["mask"].unsqueeze(0).cuda() for item in batch]

    def step():
        optimizer.zero_grad(set_to_none=True)

        embeddings = model.image_encoder(images)

        with torch.no_grad():
            sparse, dense = model.prompt_encoder(
                points=None, boxes=boxes, masks=None
            )
            image_pe = model.prompt_encoder.get_dense_pe()

        losses = []
        for i, item in enumerate(batch):
            logits, _ = model.mask_decoder(
                image_embeddings=embeddings[i:i+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse[i:i+1],
                dense_prompt_embeddings=dense[i:i+1],
                multimask_output=False,
            )
            logits = model.postprocess_masks(
                logits,
                input_size=item["input_size"],
                original_size=item["original_size"],
            )
            losses.append(segmentation_loss(logits, targets[i]))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

    # Warm-up
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()

        step()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    result = {
        "method": method,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": len(batch),
        "measured_steps": len(times),
        "mean_step_seconds": np.mean(times),
        "median_step_seconds": np.median(times),
        "sd_step_seconds": np.std(times, ddof=1),
        "peak_allocated_gib":
            torch.cuda.max_memory_allocated() / 1024**3,
        "trainable_parameters":
            sum(p.numel() for p in model.parameters()
                if p.requires_grad),
    }

    del model, optimizer, images, boxes, targets
    torch.cuda.empty_cache()
    return result


def main():
    dataset = ChengDataset(
        "data/experiment_v2/budgets/seed_1/budget_10.csv"
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda samples: samples,
    )
    batch = next(iter(loader))

    results = []
    for method in ["lora", "adapter"]:
        result = run(method, batch)
        results.append(result)
        print(result, flush=True)

    output = Path("experiment_extended/peft_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
