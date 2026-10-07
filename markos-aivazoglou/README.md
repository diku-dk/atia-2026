# My ATIA project

What really counts for detecting small objects in high res images

## Training and evaluation

Both models are trained and evaluated with their framework's own entry points; there is no training or evaluation code of ours around them.

- `yolo-seg`: Ultralytics YOLO26m-seg instance segmentation, chosen to be comparable in size to EoMT-small. Vanilla Ultralytics, with the `cfg=` file `src/yolo/configs/seg.yaml`.
- `eomt-seg`: EoMT-S instance segmentation (DINOv3 ViT-S/16), initialised from upstream's COCO-panoptic EoMT-S weights with a new class head. Upstream's LightningCLI (`src/eomt/cli.py`, upstream `main.py`) with `src/eomt/configs/seg.yaml`, based on upstream's DINOv3 EoMT-S and instance-segmentation configs.

```bash
# YOLO (W&B: `uv run yolo settings wandb=True`)
D=data/seed42
uv run yolo segment train cfg=src/yolo/configs/seg.yaml data=$D/data.yaml project=$PWD/output/yolo-seg name=train device=0
uv run yolo segment train resume model=output/yolo-seg/train/weights/last.pt
uv run yolo segment val model=output/yolo-seg/train/weights/best.pt data=$D/data.yaml split=test save_json=True

# EoMT
uv run -m src.eomt.cli fit -c src/eomt/configs/seg.yaml --data.init_args.path=$D --data.init_args.num_classes=2 \
  --model.init_args.ckpt_path=$(uv run hf download tue-mps/coco_panoptic_eomt_small_640_dinov3 pytorch_model.bin) --trainer.devices=[0]
```

The experiment runner below builds these same commands per cell (checkpoint callback, W&B, resume, test evaluation); docs/yolo.md and docs/eomt.md have the details.

### Reused code: EoMT

The EoMT model, training module, loss, LR schedule, augmentations, base dataset and data module and Lightning CLI are copied from [tue-mps/eomt](https://github.com/tue-mps/eomt) at commit `7bd19dd` (MIT, `src/eomt/LICENSE`). Only the files needed are copied, into `src/eomt/{models,training}/`, `src/eomt/cli.py` (from upstream `main.py`) and `src/datasets/`, with imports made package-relative. A few copies have small changes, listed in their headers: `cli.py` (no W&B code upload), `src/datasets/dataset.py` (reads an image folder and COCO json instead of zips), `training/lightning_module.py` (our GPU mask mAP instead of torchmetrics') and `training/mask_classification_instance.py` (`eval_step` returns its predictions and timing). Our own code is:

- the CropAndWeed data module, `src/datasets/cropandweed_instance.py` (upstream's `coco_instance.py` for our labels);
- two Lightning callbacks in `src/eomt/callbacks.py`: the attention-mask annealing schedule as fractions of the run, and `CocoPredictionWriter`, which writes `validate`'s predictions as COCO results with the latency;
- `src/eomt/mask_ap.py`, the val and test-eval mask mAP under the CropAndWeed protocol with the mask IoU on the GPU (as Ultralytics' validator);
- the config.

## Experiments

`src/experiments.py` runs the experiments listed in `src/experiments.yaml` sequentially, on every split seed (42, 0, 1) of the CropOrWeed2 dataset. The experiments are `yolo-1024x592`/`yolo-1280x736` (YOLO) and `eomt-1024x592`/`eomt-1280x736` (EoMT), with the whole frame downscaled to 1024 or 1280 wide, plus `eomt-native-1024` (EoMT on native-resolution 1024 crops without scale jitter, validated on a 2x2 grid of native tiles, tested on full frames from merged native tiles). All use the same 50-epoch budget. Each cell (experiment/seed) trains, and is then evaluated on the test split. Every framework's predictions are scored by the same hotcoco CropAndWeed-protocol evaluation. It reports mask (segmentation) mAP50-95 per class, mAP50-95 per object size (small/medium/large under the CropAndWeed protocol) and per-image time at the eval batch (single-image latency is measured separately); no box or class-agnostic metrics. The results go to `results/experiments/`: per-cell files, plus `summary.csv` and `summary_agg.csv` (mean/std over seeds). With `--wandb`, the cells' rows are merged into one W&B `summary` table.

```bash
uv run -m src.experiments --list
uv run -m src.experiments --background --wandb --device both   # relaunch the same command to continue after a crash
# status lines: output/experiments/runner_<timestamp>.log; each cell's full log: output/experiments/<exp>/seed<N>/run.log
```

See docs/experiments.md for details.
