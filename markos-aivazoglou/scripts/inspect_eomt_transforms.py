#!/usr/bin/env python3
"""Visualise EoMT's train transforms (``src/datasets/transforms.py``) step by step on a random train image.

Builds the data module from the EoMT config (``--config``, default ``src/eomt/configs/seg.yaml``: its
``data.init_args`` set img_size, scale_range, color jitter) and replays ``Transforms.forward`` one step at
a time with the module's own ``Transforms`` instance, so every step is exactly what training applies.
Writes, under ``--out`` (default ``output/eomt_transforms/``, git-ignored):

    <image stem>_seed<seed>_steps.png   one panel per step: image + class-coloured instance masks,
                                        pixel axes, resolution and instance count in the title
    <image stem>_seed<seed>_final.png   the transformed image as the model sees it

Usage::

    uv run -m scripts.inspect_eomt_transforms [--data data/seed42] [--seed N] [--color-jitter]
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Patch
from PIL import Image
from torchvision.transforms.v2 import functional as F

from src.datasets.cropandweed_instance import CropAndWeedInstance
from src.datasets.dataset import Dataset

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "Crop", 1: "Weed"}
CLASS_COLORS = {0: (0.11, 0.69, 0.48), 1: (0.92, 0.41, 0.20)}
MASK_ALPHA = 0.45


def transform_steps(t, img, target):
    """``Transforms.forward`` unrolled: the (name, image, target) after each step. Retries like upstream
    when the crop leaves no instance."""
    target = t._filter(target, ~target["is_crowd"])
    steps = [("input", img, target)]

    jittered = t.color_jitter(img)
    steps.append(("color_jitter" + ("" if t.color_jitter_enabled else " (disabled)"), jittered, target))

    flipped, target = t.random_horizontal_flip(jittered, target)
    did_flip = not torch.equal(flipped, jittered) and torch.equal(flipped, F.horizontal_flip(jittered))
    steps.append((f"random_horizontal_flip ({'flipped' if did_flip else 'not flipped'})", flipped, target))

    scaled, target = t.scale_jitter(flipped, target)
    ratio = scaled.shape[-2] / flipped.shape[-2]
    steps.append((f"scale_jitter (x{ratio:.3f})", scaled, target))

    padded, target = t.pad(scaled, dict(target))  # pad writes into the dict it gets
    steps.append(("pad", padded, target))

    cropped, target = t.random_crop(padded, target)
    valid = target["masks"].flatten(1).any(1)
    if not valid.any():
        print("crop left no instance: retrying (as Transforms.forward does)")
        return transform_steps(t, img, steps[0][2])
    steps.append((f"random_crop (+ drop empty masks: -{int((~valid).sum())})", cropped, t._filter(target, valid)))

    return steps


def overlay(img, target):
    """HWC float image with each instance mask blended in its class colour."""
    out = img.permute(1, 2, 0).numpy().astype(np.float32) / 255.0
    for mask, label in zip(target["masks"].numpy(), target["labels"].tolist()):
        out[mask] = (1 - MASK_ALPHA) * out[mask] + MASK_ALPHA * np.array(CLASS_COLORS[label])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/seed42")
    parser.add_argument("--config", type=Path, default=ROOT / "src/eomt/configs/seg.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "output/eomt_transforms")
    parser.add_argument("--seed", type=int, default=None, help="picks the image and the transforms' randomness")
    parser.add_argument("--color-jitter", action="store_true", help="enable color jitter (off in the config)")
    args = parser.parse_args()

    seed = random.SystemRandom().randrange(2**31) if args.seed is None else args.seed
    torch.manual_seed(seed)

    init_args = yaml.safe_load(args.config.read_text())["data"]["init_args"]
    if args.color_jitter:
        init_args["color_jitter_enabled"] = True
    datamodule = CropAndWeedInstance(path=args.data, num_classes=len(CLASS_NAMES), **init_args)
    dataset = Dataset(args.data, "train", datamodule.target_parser, datamodule.check_empty_targets)

    index = random.Random(seed).randrange(len(dataset))
    stem = Path(dataset.imgs[index]).stem
    img, target = dataset[index]
    print(f"seed {seed}: train image {dataset.imgs[index]} ({index}/{len(dataset)})")

    steps = transform_steps(datamodule.transforms, img, target)

    ncols = 3
    nrows = -(-len(steps) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows), constrained_layout=True)
    for ax, (name, step_img, step_target) in zip(axes.flat, steps):
        h, w = step_img.shape[-2:]
        ax.imshow(overlay(step_img, step_target))
        ax.set_title(f"{name}\n{h} x {w} px (H x W), {len(step_target['labels'])} instances", fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    for ax in axes.flat[len(steps):]:
        ax.axis("off")
    fig.legend(handles=[Patch(color=CLASS_COLORS[c], alpha=MASK_ALPHA, label=n) for c, n in CLASS_NAMES.items()],
               loc="outside lower center", ncols=len(CLASS_NAMES))
    fig.suptitle(f"EoMT train transforms: {dataset.imgs[index]} (seed {seed})")

    args.out.mkdir(parents=True, exist_ok=True)
    steps_path = args.out / f"{stem}_seed{seed}_steps.png"
    final_path = args.out / f"{stem}_seed{seed}_final.png"
    fig.savefig(steps_path, dpi=120)
    Image.fromarray(steps[-1][1].permute(1, 2, 0).numpy()).save(final_path)
    print(f"wrote {steps_path}\nwrote {final_path}")


if __name__ == "__main__":
    main()
